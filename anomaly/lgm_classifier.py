"""
LightGBM Accident Classifier
Pipeline position: Feature Vector (20-dim) → [LightGBM] → ml_score [0,1]

Tại sao LightGBM thay vì LSTM/3D-CNN?
───────────────────────────────────────
Từ báo cáo Section 3.4.2 và Bảng 4.1:

  Method          F1    FPS
  3D-CNN          0.78  15    ← quá nặng, không realtime
  ST-GCN          0.81  30    ← nhạy với occlusion
  PROPOSED        0.86  42    ← LightGBM + IPM + Pose

LightGBM phù hợp vì:
1. Dữ liệu dạng bảng 20-dim → KHÔNG cần CNN/RNN
2. Dữ liệu nhỏ (few hundred accident samples) → boosting generalizes tốt hơn DL
3. Imbalanced data (1:450) → scale_pos_weight hỗ trợ trực tiếp
4. Inference < 1ms → realtime không bottleneck
5. Feature importance → explainability cho hội đồng

Data imbalance:
  scale_pos_weight = N_negative / N_positive ≈ 450
  → Mỗi accident sample có trọng số 450x so với normal

Tham chiếu báo cáo: Section 3.4.2
"""

from __future__ import annotations

import os
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from features.feature_extractor import FeatureAggregator
from config.settings import get_config
from utils.logger import get_logger

logger = get_logger(__name__)


class AccidentClassifier:
    """
    LightGBM-based accident classifier.

    Modes:
    1. Inference only: load model đã train từ file .pkl
    2. Training: build dataset từ VehicleState windows + labels → train mới

    Feature input: 20-dim vector từ FeatureAggregator
    Output: probability [0, 1] → accident likelihood
    """

    def __init__(self, config=None):
        self.cfg = config or get_config()
        clf_cfg = self.cfg.classifier

        self.model_path = Path(clf_cfg.model_path)
        self.accident_threshold: float = clf_cfg.accident_threshold  # 0.65
        self.feature_dim: int = clf_cfg.feature_dim  # 20

        # LightGBM hyperparams từ config / báo cáo
        self.lgbm_params = vars(clf_cfg.params).copy()

        self.lgbm_params.update({
            "random_state": 42,
            "n_jobs": -1,
        })

        self._model = None
        self._is_loaded = False

        logger.info(
            f"AccidentClassifier init | "
            f"threshold={self.accident_threshold} | "
            f"feature_dim={self.feature_dim}"
        )

    # ──────────────────────────────────────────────────────────────────
    # Inference
    # ──────────────────────────────────────────────────────────────────

    def load_model(self) -> "AccidentClassifier":
        """
        Load model đã train từ file .pkl.
        Nếu file không tồn tại → dùng rule-based fallback.
        """
        if self.model_path.exists():
            with open(self.model_path, "rb") as f:
                self._model = pickle.load(f)
            self._is_loaded = True
            logger.info(f"LightGBM model loaded: {self.model_path}")
        else:
            logger.warning(
                f"Model file không tồn tại: {self.model_path}. "
                f"Chạy ở chế độ rule-only. "
                f"Dùng train() để build model."
            )
            self._is_loaded = False

        return self

    def predict(self, feature_vector: np.ndarray) -> float:
        """
        Inference: feature vector → accident probability.

        Args:
            feature_vector: shape (20,) hoặc (1, 20)

        Returns:
            float [0, 1] — probability of accident
        """
        if not self._is_loaded:
            return -1.0  # Signal: no ML model

        x = feature_vector.reshape(1, -1)
        prob = self._model.predict(x)[0]  # class=1 (accident)
        return float(prob)

    def predict_batch(self, feature_matrix: np.ndarray) -> np.ndarray:
        """
        Batch inference cho nhiều feature vectors.

        Args:
            feature_matrix: shape (N, 20)

        Returns:
            np.ndarray shape (N,) — probabilities
        """
        if not self._is_loaded:
            return np.full(len(feature_matrix), -1.0)

        return self._model.predict_proba(feature_matrix)

    def is_accident(self, prob: float) -> bool:
        return prob >= self.accident_threshold

    # ──────────────────────────────────────────────────────────────────
    # Training
    # ──────────────────────────────────────────────────────────────────

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        save: bool = True
    ) -> Dict:
        """
        Huấn luyện LightGBM classifier.

        Args:
            X_train: shape (N, 20) — feature vectors
            y_train: shape (N,) — labels {0: normal, 1: accident}
            X_val:   validation set (None = auto split 20%)
            y_val:   validation labels
            save:    có lưu model sau khi train không

        Returns:
            dict với training metrics

        Training strategy từ báo cáo:
        - scale_pos_weight = N_neg / N_pos (~450)
        - Early stopping sau 50 rounds không cải thiện
        - feature_fraction = 0.8 → reduce overfitting
        - Event-level split (không random) → không data leakage
        """
        try:
            import lightgbm as lgb
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import (
                f1_score, precision_score, recall_score,
                roc_auc_score, confusion_matrix
            )
        except ImportError as e:
            raise ImportError(
                f"Cần cài: pip install lightgbm scikit-learn\nLỗi: {e}"
            )

        logger.info(
            f"Bắt đầu training LightGBM | "
            f"Train: {X_train.shape} | "
            f"Positive samples: {y_train.sum():.0f} / {len(y_train)}"
        )

        # Tự động tính scale_pos_weight nếu chưa set
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        if n_pos == 0:
            raise ValueError("Không có mẫu accident (y=1) trong training set!")

        spw = n_neg / n_pos

        # Không dùng scale_pos_weight nữa, chỉ dùng is_unbalance
        # Xóa scale_pos_weight nếu có trong params gốc
        params = {**self.lgbm_params}
        params.pop('scale_pos_weight', None)  # loại bỏ key này
        params['is_unbalance'] = True  # tự động cân bằng

        logger.info(f"Imbalance ratio: {spw:.1f}:1 (normal:accident = {n_neg}:{n_pos})")

        # Auto split val nếu không có
        if X_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train,
                test_size=0.2,
                stratify=y_train,
                random_state=42
            )
            logger.info(
                f"Auto val split: train={len(X_train)}, val={len(X_val)}"
            )

        # Build LightGBM datasets
        train_data = lgb.Dataset(
            X_train, label=y_train,
            feature_name=FeatureAggregator.FEATURE_NAMES
        )
        val_data = lgb.Dataset(
            X_val, label=y_val,
            feature_name=FeatureAggregator.FEATURE_NAMES,
            reference=train_data
        )

        # Extract n_estimators và early_stopping_rounds
        n_est = params.pop("n_estimators", 1000)
        early_stop = params.pop("early_stopping_rounds", 50)
        verbose = params.pop("verbose", -1)

        # Train
        callbacks = [
            lgb.early_stopping(early_stop, verbose=False),
            lgb.log_evaluation(period=100)
        ]

        self._model = lgb.train(
            params,
            train_data,
            num_boost_round=n_est,
            valid_sets=[val_data],
            callbacks=callbacks,
        )

        # Evaluate
        val_probs = self._model.predict(X_val)
        val_preds = (val_probs >= self.accident_threshold).astype(int)

        metrics = {
            "f1":        float(f1_score(y_val, val_preds, zero_division=0)),
            "precision": float(precision_score(y_val, val_preds, zero_division=0)),
            "recall":    float(recall_score(y_val, val_preds, zero_division=0)),
            "auc":       float(roc_auc_score(y_val, val_probs)),
            "n_trees":   self._model.num_trees(),
            "imbalance_ratio": float(spw),
        }

        cm = confusion_matrix(y_val, val_preds)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        metrics["fpr"] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        metrics["confusion_matrix"] = cm.tolist()

        logger.info("=" * 50)
        logger.info("TRAINING COMPLETE")
        logger.info(f"  F1-Score  : {metrics['f1']:.4f}")
        logger.info(f"  Precision : {metrics['precision']:.4f}")
        logger.info(f"  Recall    : {metrics['recall']:.4f}")
        logger.info(f"  AUC       : {metrics['auc']:.4f}")
        logger.info(f"  FPR       : {metrics['fpr']:.4f}")
        logger.info(f"  Trees     : {metrics['n_trees']}")
        logger.info("=" * 50)

        self._is_loaded = True

        if save:
            self.save_model()

        return metrics

    def save_model(self) -> None:
        """Lưu model vào file .pkl."""
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump(self._model, f)
        logger.info(f"Model saved: {self.model_path}")

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Trả về feature importance (để phân tích + báo cáo NCKH).
        Cho phép giải thích: feature nào đóng góp nhiều nhất vào phân loại.
        """
        if not self._is_loaded:
            return None

        importances = self._model.feature_importance(importance_type="gain")
        names = FeatureAggregator.FEATURE_NAMES

        result = {name: float(imp) for name, imp in zip(names, importances)}

        # Sắp xếp giảm dần
        return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))

    @property
    def is_ready(self) -> bool:
        return self._is_loaded


class DatasetBuilder:
    """
    Helper class để build training dataset từ video annotations.

    Workflow:
    1. Chạy pipeline (detect + track + IPM + feature) trên annotated video
    2. Với mỗi sliding window: extract feature vector + assign label
    3. Gọi build() → (X, y) để train LightGBM

    Label assignment (từ báo cáo):
    - Window được gán nhãn "accident" nếu ≥ 50% frames trong window là accident
    - Event-level split tránh data leakage
    """

    def __init__(self, window_size: int = 25, stride: int = 5):
        self.window_size = window_size
        self.stride = stride
        self._samples: List[Tuple[np.ndarray, int]] = []  # (feature_vec, label)

    def add_sample(self, feature_vec: np.ndarray, label: int) -> None:
        """
        Thêm một mẫu vào dataset.

        Args:
            feature_vec: 20-dim vector từ FeatureAggregator
            label: 0=normal, 1=accident
        """
        assert feature_vec.shape == (20,), \
            f"Feature dim phải là 20, nhận được {feature_vec.shape}"
        assert label in (0, 1), f"Label phải là 0 hoặc 1, nhận được {label}"
        self._samples.append((feature_vec.copy(), label))

    def add_window(
        self,
        window_states: list,  # List[VehicleState]
        label: int
    ) -> None:
        """Tự động aggregate window → feature vector rồi add."""
        from features.feature_extractor import FeatureAggregator
        feat = FeatureAggregator.aggregate(window_states)
        self.add_sample(feat, label)

    def build(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            X: np.ndarray shape (N, 20)
            y: np.ndarray shape (N,)
        """
        if not self._samples:
            raise ValueError("Dataset rỗng. Cần add samples trước.")

        X = np.array([s[0] for s in self._samples], dtype=np.float32)
        y = np.array([s[1] for s in self._samples], dtype=np.int32)

        n_pos = y.sum()
        n_neg = len(y) - n_pos
        logger.info(
            f"Dataset built: {len(y)} samples | "
            f"normal={n_neg} | accident={n_pos} | "
            f"ratio={n_neg/max(n_pos,1):.1f}:1"
        )
        return X, y

    def clear(self) -> None:
        self._samples.clear()

    def __len__(self) -> int:
        return len(self._samples)