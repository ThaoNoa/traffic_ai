"""
Rule-Based Risk Engine
Pipeline position: Feature Vector → [Rule Engine] → Risk Score → LightGBM Fusion

Tại sao cần Rule-Based Layer?
─────────────────────────────
Hybrid AI = ML + Physics + Rules:

  Rule engine không thay thế LightGBM — nó là PREFILTER:
  1. Phát hiện vi phạm vật lý rõ ràng (phanh cực gấp, tốc độ 0 đột ngột)
  2. Giảm False Positive của LightGBM (ML có thể bị lừa bởi edge cases)
  3. Tăng tốc inference (không cần chạy LightGBM nếu rule đã loại)
  4. Explainability: có thể giải thích tại sao cảnh báo (quan trọng cho demo)

Risk State Machine:
  NORMAL → WARNING → DANGER → ACCIDENT
      ↑______________|

Fusion formula:
  final_score = λ_rule × rule_score + λ_ml × ml_score
  (λ_rule=0.3, λ_ml=0.7 — ML có trọng số cao hơn)

Tham chiếu báo cáo: Section 3.4.1 — "Rule-based Risk Analysis"
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import IntEnum

from features.feature_extractor import VehicleState, FeatureAggregator
from config.settings import get_config
from utils.logger import get_logger

logger = get_logger(__name__)


class RiskLevel(IntEnum):
    NORMAL  = 0
    WARNING = 1
    DANGER  = 2
    ACCIDENT = 3


@dataclass
class RuleViolation:
    """Ghi lại một vi phạm rule cụ thể."""
    rule_name: str
    severity: float        # 0.0 → 1.0
    description: str
    feature_value: float   # Giá trị trigger rule


@dataclass
class RiskResult:
    """Kết quả đánh giá risk cho một phương tiện tại một thời điểm."""
    track_id: int
    frame_id: int
    rule_score: float          # 0.0 → 1.0 (từ rule engine)
    ml_score: float            # 0.0 → 1.0 (từ LightGBM, -1 nếu chưa có)
    final_score: float         # 0.0 → 1.0 (fusion)
    risk_level: RiskLevel
    violations: List[RuleViolation] = field(default_factory=list)
    is_accident: bool = False

    def explain(self) -> str:
        """Tạo chuỗi giải thích cho UI/log."""
        lines = [
            f"Track {self.track_id} | Risk: {self.risk_level.name} "
            f"(rule={self.rule_score:.2f}, ml={self.ml_score:.2f}, "
            f"final={self.final_score:.2f})"
        ]
        for v in self.violations:
            lines.append(
                f"  ⚠ [{v.rule_name}] {v.description} "
                f"(val={v.feature_value:.2f}, sev={v.severity:.2f})"
            )
        return "\n".join(lines)


class PhysicsRuleEngine:
    """
    Rule-Based Risk Scoring dựa trên kiến thức vật lý về tai nạn.

    Mỗi rule:
    - Kiểm tra một điều kiện vật lý cụ thể
    - Trả về severity score 0→1
    - Severity được weighted sum thành rule_score

    Rules được thiết kế từ phân tích case study tai nạn xe máy:
    1. SUDDEN_BRAKE    : Gia tốc < -5 m/s² đột ngột → phanh khẩn cấp
    2. HIGH_LEAN       : Góc nghiêng > 30° → nguy cơ ngã
    3. LEAN_RAPID_CHANGE: Δθ > 20°/s → mất thăng bằng
    4. COLLISION_RISK  : d_min < 2m khi v > 3 m/s → nguy cơ va chạm
    5. SUDDEN_STOP     : Speed → 0 trong 3 frame → dừng đột ngột
    6. HIGH_SPEED_CURVE: v > 15 m/s + heading change lớn → ôm cua tốc cao
    """

    def __init__(self, config=None):
        self.cfg = config or get_config()

        # Thresholds (có thể override qua config)
        self.BRAKE_THRESH      = -4.0   # m/s² — phanh gấp
        self.HARD_BRAKE_THRESH = -7.0   # m/s² — phanh rất gấp
        self.LEAN_THRESH       = 25.0   # độ — nghiêng nguy hiểm
        self.HARD_LEAN_THRESH  = 40.0   # độ — nghiêng rất nguy hiểm
        self.LEAN_DELTA_THRESH = 15.0   # °/s — thay đổi góc nghiêng nhanh
        self.COLLISION_DIST    = 3.0    # mét — khoảng cách nguy hiểm
        self.COLLISION_SPEED   = 3.0    # m/s — tốc độ tối thiểu để collision matter
        self.STOP_SPEED        = 0.5    # m/s — coi là "dừng"

        # Fusion weights
        self.lambda_rule = 0.3
        self.lambda_ml   = 0.7

        # Decision thresholds
        self.WARNING_THRESH  = 0.3
        self.DANGER_THRESH   = 0.55
        self.ACCIDENT_THRESH = 0.65  # từ config

        logger.info("PhysicsRuleEngine initialized.")

    def evaluate_state(
        self,
        window: List[VehicleState],
        track_id: int,
        frame_id: int
    ) -> RiskResult:
        """
        Đánh giá risk từ window của VehicleState.

        Args:
            window: 25 VehicleState gần nhất
            track_id: ID phương tiện
            frame_id: frame hiện tại

        Returns:
            RiskResult với rule_score và violations
        """
        violations = []

        if not window:
            return RiskResult(
                track_id=track_id, frame_id=frame_id,
                rule_score=0.0, ml_score=-1.0, final_score=0.0,
                risk_level=RiskLevel.NORMAL
            )

        current = window[-1]  # state mới nhất

        # ── Rule 1: Sudden Brake ────────────────────────────────────
        v = self._check_sudden_brake(window, violations)

        # ── Rule 2: High Lean Angle ─────────────────────────────────
        self._check_lean_angle(current, violations)

        # ── Rule 3: Rapid Lean Change ───────────────────────────────
        self._check_lean_rate(window, violations)

        # ── Rule 4: Collision Risk ──────────────────────────────────
        self._check_collision_proximity(current, violations)

        # ── Rule 5: Sudden Stop ─────────────────────────────────────
        self._check_sudden_stop(window, violations)

        # ── Tổng hợp rule_score ─────────────────────────────────────
        if violations:
            # Weighted max (không chỉ sum — tránh nhiều rule nhỏ trigger cùng lúc)
            severities = [v.severity for v in violations]
            rule_score = float(np.clip(
                0.6 * max(severities) + 0.4 * np.mean(severities),
                0.0, 1.0
            ))
        else:
            rule_score = 0.0

        # ML score chưa có → fusion chỉ dùng rule
        result = RiskResult(
            track_id=track_id,
            frame_id=frame_id,
            rule_score=rule_score,
            ml_score=-1.0,
            final_score=rule_score,   # sẽ update sau khi có ML score
            risk_level=self._score_to_level(rule_score),
            violations=violations,
        )

        return result

    def fuse_with_ml(
        self,
        result: RiskResult,
        ml_score: float
    ) -> RiskResult:
        """
        Fusion rule score + ML score → final score.

        Formula từ báo cáo:
          final = λ_rule × rule_score + λ_ml × ml_score

        Với điều kiện đặc biệt:
        - Nếu rule_score > 0.8: không cần ML, cảnh báo ngay (safety critical)
        - Nếu ml_score < 0.2 và rule_score < 0.3: NORMAL chắc chắn
        """
        result.ml_score = ml_score

        # Safety override: rule phát hiện nguy hiểm cực cao
        if result.rule_score >= 0.85:
            final = max(ml_score, result.rule_score)
        # Clear normal: cả rule và ML đều thấp
        elif result.rule_score < 0.2 and ml_score < 0.25:
            final = max(result.rule_score, ml_score)
        else:
            final = (
                self.lambda_rule * result.rule_score
                + self.lambda_ml * ml_score
            )

        result.final_score = float(np.clip(final, 0.0, 1.0))
        result.risk_level = self._score_to_level(result.final_score)
        result.is_accident = result.final_score >= self.ACCIDENT_THRESH

        return result

    # ──────────────────────────────────────────────────────────────────
    # Private rule checkers
    # ──────────────────────────────────────────────────────────────────

    def _check_sudden_brake(
        self,
        window: List[VehicleState],
        violations: List[RuleViolation]
    ) -> None:
        """Phát hiện phanh gấp: gia tốc âm lớn trong cửa sổ."""
        accels = [s.acceleration for s in window[-10:]]  # 10 frame gần nhất
        min_accel = min(accels)

        if min_accel < self.BRAKE_THRESH:
            # Normalize: -4 m/s² → 0.3 severity, -10 m/s² → 1.0
            sev = np.clip(
                (abs(min_accel) - abs(self.BRAKE_THRESH)) /
                (abs(self.HARD_BRAKE_THRESH) - abs(self.BRAKE_THRESH)),
                0.0, 1.0
            )
            violations.append(RuleViolation(
                rule_name="SUDDEN_BRAKE",
                severity=float(0.3 + 0.7 * sev),
                description=f"Phanh gấp: a={min_accel:.1f} m/s²",
                feature_value=min_accel
            ))

    def _check_lean_angle(
        self,
        current: VehicleState,
        violations: List[RuleViolation]
    ) -> None:
        """Phát hiện góc nghiêng nguy hiểm."""
        lean = abs(current.body_lean_angle)

        if lean > self.LEAN_THRESH:
            sev = np.clip(
                (lean - self.LEAN_THRESH) /
                (self.HARD_LEAN_THRESH - self.LEAN_THRESH),
                0.0, 1.0
            )
            violations.append(RuleViolation(
                rule_name="HIGH_LEAN_ANGLE",
                severity=float(0.3 + 0.7 * sev),
                description=f"Góc nghiêng nguy hiểm: θ={lean:.1f}°",
                feature_value=lean
            ))

    def _check_lean_rate(
        self,
        window: List[VehicleState],
        violations: List[RuleViolation]
    ) -> None:
        """Phát hiện thay đổi góc nghiêng đột ngột (mất thăng bằng)."""
        deltas = [abs(s.body_lean_delta) for s in window[-5:]]
        max_delta = max(deltas) if deltas else 0.0

        if max_delta > self.LEAN_DELTA_THRESH:
            sev = np.clip(
                (max_delta - self.LEAN_DELTA_THRESH) / 30.0,
                0.0, 1.0
            )
            violations.append(RuleViolation(
                rule_name="LEAN_RAPID_CHANGE",
                severity=float(0.25 + 0.75 * sev),
                description=f"Mất thăng bằng: Δθ={max_delta:.1f}°/s",
                feature_value=max_delta
            ))

    def _check_collision_proximity(
        self,
        current: VehicleState,
        violations: List[RuleViolation]
    ) -> None:
        """Phát hiện nguy cơ va chạm: gần xe khác + tốc độ cao."""
        d = current.min_dist_to_others
        v = current.speed

        if d < self.COLLISION_DIST and v > self.COLLISION_SPEED:
            # Severity tỉ lệ nghịch với distance, thuận với speed
            dist_sev  = np.clip(1.0 - d / self.COLLISION_DIST, 0.0, 1.0)
            speed_sev = np.clip(v / 15.0, 0.0, 1.0)
            sev = 0.5 * dist_sev + 0.5 * speed_sev

            violations.append(RuleViolation(
                rule_name="COLLISION_RISK",
                severity=float(0.4 + 0.6 * sev),
                description=f"Nguy cơ va chạm: d={d:.1f}m, v={v:.1f}m/s",
                feature_value=d
            ))

    def _check_sudden_stop(
        self,
        window: List[VehicleState],
        violations: List[RuleViolation]
    ) -> None:
        """
        Phát hiện dừng đột ngột:
        Speed > 5 m/s → Speed < 0.5 m/s trong 5 frame (< 0.2 giây)
        """
        if len(window) < 5:
            return

        recent = window[-5:]
        speeds = [s.speed for s in recent]

        was_moving = speeds[0] > 5.0
        now_stopped = speeds[-1] < self.STOP_SPEED

        if was_moving and now_stopped:
            speed_drop = speeds[0] - speeds[-1]
            sev = np.clip(speed_drop / 15.0, 0.0, 1.0)
            violations.append(RuleViolation(
                rule_name="SUDDEN_STOP",
                severity=float(0.5 + 0.5 * sev),
                description=f"Dừng đột ngột: {speeds[0]:.1f}→{speeds[-1]:.1f} m/s",
                feature_value=speed_drop
            ))

    def _score_to_level(self, score: float) -> RiskLevel:
        if score >= self.ACCIDENT_THRESH:
            return RiskLevel.ACCIDENT
        elif score >= self.DANGER_THRESH:
            return RiskLevel.DANGER
        elif score >= self.WARNING_THRESH:
            return RiskLevel.WARNING
        else:
            return RiskLevel.NORMAL 