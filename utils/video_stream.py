"""
Video Stream handler - Đọc video từ file, webcam hoặc RTSP.
Tách riêng để main pipeline không bị coupled với nguồn input.
"""

import cv2
import threading
import queue
import time
from pathlib import Path
from typing import Optional, Tuple, Any
from utils.logger import get_logger

logger = get_logger(__name__)


class VideoStream:
    """
    Thread-safe video stream reader.
    
    Tại sao cần thread riêng?
    → Đọc frame từ disk/network là I/O bound.
    → Nếu đọc trong main thread, GPU ngồi chờ I/O → FPS giảm.
    → Thread riêng đọc trước, buffer frames vào queue → GPU luôn có frame.
    """
    
    def __init__(
        self, 
        source: str | int,
        buffer_size: int = 8,
        target_fps: Optional[float] = None
    ):
        """
        Args:
            source: Path file, 0 (webcam), hoặc "rtsp://..."
            buffer_size: Số frame buffer trong queue
            target_fps: Giới hạn FPS đọc (None = max speed)
        """
        self.source = source
        self.buffer_size = buffer_size
        self.target_fps = target_fps
        
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_queue: queue.Queue = queue.Queue(maxsize=buffer_size)
        self._thread: Optional[threading.Thread] = None
        self._stopped = False
        self._lock = threading.Lock()
        
        # Metadata
        self.width: int = 0
        self.height: int = 0
        self.fps: float = 0.0
        self.total_frames: int = 0
        self.frame_count: int = 0
        
    def start(self) -> "VideoStream":
        """Mở stream và bắt đầu đọc."""
        self._cap = cv2.VideoCapture(self.source)
        
        if not self._cap.isOpened():
            raise RuntimeError(f"Không thể mở video source: {self.source}")
        
        # Lấy metadata
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self._cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(
            f"Stream opened: {self.source} | "
            f"{self.width}×{self.height} @ {self.fps:.1f}FPS | "
            f"Total frames: {self.total_frames}"
        )
        
        # Start reader thread
        self._stopped = False
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()
        
        return self
    
    def _reader_loop(self):
        """Vòng lặp đọc frame chạy trong thread riêng."""
        frame_interval = 1.0 / self.target_fps if self.target_fps else 0
        
        while not self._stopped:
            t_start = time.time()
            
            ret, frame = self._cap.read()
            
            if not ret:
                logger.info("Video stream kết thúc.")
                self._stopped = True
                break
            
            self.frame_count += 1
            
            # Đẩy vào queue, nếu queue đầy → bỏ frame cũ nhất (realtime priority)
            if self._frame_queue.full():
                try:
                    self._frame_queue.get_nowait()  # drop oldest frame
                except queue.Empty:
                    pass
            
            try:
                self._frame_queue.put_nowait((self.frame_count, frame))
            except queue.Full:
                pass
            
            # FPS limiter
            elapsed = time.time() - t_start
            if frame_interval > elapsed:
                time.sleep(frame_interval - elapsed)
    
    def read(self, timeout: float = 1.0) -> Tuple[bool, Optional[int], Optional[any]]:
        """
        Lấy frame tiếp theo từ buffer.
        
        Returns:
            (success, frame_id, frame)
        """
        try:
            frame_id, frame = self._frame_queue.get(timeout=timeout)
            return True, frame_id, frame
        except queue.Empty:
            if self._stopped:
                return False, None, None
            return False, None, None
    
    def is_running(self) -> bool:
        return not self._stopped or not self._frame_queue.empty()
    
    def stop(self):
        """Dừng stream và giải phóng resource."""
        self._stopped = True
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
        logger.info("VideoStream stopped.")
    
    @property
    def progress(self) -> float:
        """Phần trăm tiến độ video (0-100)."""
        if self.total_frames <= 0:
            return 0.0
        return (self.frame_count / self.total_frames) * 100
    
    def __enter__(self):
        return self.start()
    
    def __exit__(self, *args):
        self.stop()
