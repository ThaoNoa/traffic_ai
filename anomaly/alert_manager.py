"""
Alert Manager
Pipeline position: RiskResult → [Alert Manager] → User Notification + Clip Save

Nhiệm vụ:
1. Cooldown control: không spam alert (5 giây giữa 2 alert liên tiếp)
2. Clip saving: lưu video 10 giây xung quanh sự cố
3. State machine: track alert state per vehicle
4. Logging: ghi log có timestamp để post-analysis

Tham chiếu báo cáo: Section 3.2.2 (Alert system), config alert.*
"""

from __future__ import annotations

import time
import cv2
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from anomaly.rule_engine import RiskResult, RiskLevel
from config.settings import get_config
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AlertEvent:
    """Một sự kiện cảnh báo đã được xác nhận."""
    track_id: int
    timestamp: float        # unix time
    frame_id: int
    final_score: float
    risk_level: RiskLevel
    rule_violations: List[str]
    clip_path: Optional[str] = None


class AlertManager:
    """
    Quản lý vòng đời cảnh báo:
    1. Nhận RiskResult từ anomaly pipeline
    2. Apply cooldown (tránh spam)
    3. Trigger alert nếu hợp lệ
    4. Buffer frames để lưu clip

    Frame buffer:
    - Luôn giữ N frame gần nhất trong bộ nhớ
    - Khi có accident: lưu frame_buffer + N frame tiếp theo
    - Output: .mp4 clip xung quanh sự cố
    """

    def __init__(self, config=None):
        self.cfg = config or get_config()
        alert_cfg = self.cfg.alert

        self.cooldown_sec: float = alert_cfg.cooldown_seconds    # 5.0
        self.clip_sec: float = alert_cfg.save_clip_seconds       # 10
        self.output_dir = Path(alert_cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # track_id → thời điểm alert cuối cùng
        self._last_alert_time: Dict[int, float] = {}

        # Frame buffer (luôn giữ clip_sec/2 frame gần nhất)
        fps = self.cfg.video.fps
        self._pre_buffer_size = int(fps * self.clip_sec / 2)
        self._frame_buffer: deque = deque(maxlen=self._pre_buffer_size)

        # Post-alert buffer (lưu frame sau khi trigger)
        self._post_buffer: List[np.ndarray] = []
        self._post_frames_needed: int = int(fps * self.clip_sec / 2)
        self._recording: bool = False
        self._current_alert: Optional[AlertEvent] = None

        # Lịch sử tất cả alert
        self._alert_history: List[AlertEvent] = []

        # Video properties (cần biết để lưu clip)
        self._frame_w: int = 1920
        self._frame_h: int = 1080
        self._fps: float = fps

        logger.info(
            f"AlertManager init | "
            f"cooldown={self.cooldown_sec}s | "
            f"clip={self.clip_sec}s | "
            f"output={self.output_dir}"
        )

    def set_video_properties(self, w: int, h: int, fps: float) -> None:
        """Gọi sau khi mở video source."""
        self._frame_w = w
        self._frame_h = h
        self._fps = fps
        self._pre_buffer_size = int(fps * self.clip_sec / 2)
        self._frame_buffer = deque(maxlen=self._pre_buffer_size)
        self._post_frames_needed = int(fps * self.clip_sec / 2)

    def push_frame(self, frame: np.ndarray) -> None:
        """
        Thêm frame vào circular buffer.
        Gọi mỗi frame, bất kể có alert hay không.
        """
        self._frame_buffer.append(frame.copy())

        # Nếu đang record post-alert
        if self._recording:
            self._post_buffer.append(frame.copy())
            if len(self._post_buffer) >= self._post_frames_needed:
                self._save_clip()
                self._recording = False
                self._post_buffer.clear()

    def process(
        self,
        risk_results: Dict[int, RiskResult],
        frame_id: int
    ) -> List[AlertEvent]:
        """
        Xử lý risk results → phát sinh alert events.

        Args:
            risk_results: dict track_id → RiskResult
            frame_id: frame ID hiện tại

        Returns:
            List[AlertEvent] — những alert được trigger trong frame này
        """
        now = time.time()
        triggered = []

        for track_id, result in risk_results.items():
            if not result.is_accident:
                continue

            # Cooldown check
            last = self._last_alert_time.get(track_id, 0.0)
            if now - last < self.cooldown_sec:
                logger.debug(
                    f"Alert suppressed (cooldown) | Track {track_id} | "
                    f"Remaining: {self.cooldown_sec - (now - last):.1f}s"
                )
                continue

            # Trigger alert
            self._last_alert_time[track_id] = now

            violations_str = [v.rule_name for v in result.violations]
            event = AlertEvent(
                track_id=track_id,
                timestamp=now,
                frame_id=frame_id,
                final_score=result.final_score,
                risk_level=result.risk_level,
                rule_violations=violations_str,
            )

            self._alert_history.append(event)
            triggered.append(event)

            # Bắt đầu record post-alert
            if not self._recording:
                self._recording = True
                self._current_alert = event
                self._post_buffer.clear()

            logger.warning(
                f"🚨 ACCIDENT DETECTED | "
                f"Track {track_id} | "
                f"Score={result.final_score:.3f} | "
                f"Frame={frame_id} | "
                f"Violations: {violations_str}"
            )

        return triggered

    def _save_clip(self) -> None:
        """Lưu clip = pre_buffer + post_buffer."""
        if not self._current_alert:
            return

        ts = int(self._current_alert.timestamp)
        fname = f"accident_track{self._current_alert.track_id}_{ts}.mp4"
        fpath = self.output_dir / fname

        frames = list(self._frame_buffer) + self._post_buffer
        if not frames:
            return

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(fpath), fourcc, self._fps,
            (self._frame_w, self._frame_h)
        )
        for f in frames:
            # Resize nếu cần
            if f.shape[1] != self._frame_w or f.shape[0] != self._frame_h:
                f = cv2.resize(f, (self._frame_w, self._frame_h))
            writer.write(f)
        writer.release()

        self._current_alert.clip_path = str(fpath)
        logger.info(f"Clip saved: {fpath} ({len(frames)} frames)")

    @property
    def alert_count(self) -> int:
        return len(self._alert_history)

    @property
    def alert_history(self) -> List[AlertEvent]:
        return list(self._alert_history)

    def is_in_cooldown(self, track_id: int) -> bool:
        now = time.time()
        last = self._last_alert_time.get(track_id, 0.0)
        return (now - last) < self.cooldown_sec