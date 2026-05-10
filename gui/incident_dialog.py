# gui/incident_dialog.py
"""
Cửa sổ phân tích sự cố - Hiển thị khi phát hiện accident
Đây là "át chủ bài" cho phần demo trước hội đồng
"""

import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QFrame, QTextEdit, QGroupBox,
                             QScrollArea, QWidget, QGridLayout, QMessageBox,
                             QFileDialog)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPalette


class IncidentDialog(QDialog):
    """
    Cửa sổ pop-up hiển thị chi tiết sự cố giao thông.
    
    Hiển thị:
    - Ảnh chụp màn hình tại thời điểm sự cố
    - Thông tin chi tiết: loại sự cố, độ tin cậy, luật vi phạm
    - Biểu đồ động học (nếu có)
    - Nút xuất báo cáo
    """
    
    def __init__(self, alert_data, frame=None, parent=None):
        """
        Args:
            alert_data: dict chứa thông tin sự cố
                {
                    'track_id': int,
                    'timestamp': float,
                    'frame_id': int,
                    'final_score': float,
                    'risk_level': str,
                    'violations': list of str,
                    'clip_path': str (optional),
                    'location': str (optional)
                }
            frame: numpy array - ảnh tại thời điểm sự cố
            parent: QWidget cha
        """
        super().__init__(parent)
        
        self.alert_data = alert_data
        self.frame = frame
        
        # Cấu hình cửa sổ
        self.setWindowTitle("🚨 PHÂN TÍCH SỰ CỐ GIAO THÔNG")
        self.setMinimumSize(900, 700)
        self.setModal(True)  # Chặn tương tác với cửa sổ chính
        
        # Style cho dialog
        self.setStyleSheet("""
            QDialog {
                background-color: #1a1a2e;
            }
            QLabel {
                color: #ffffff;
            }
            QGroupBox {
                color: #e94560;
                border: 2px solid #e94560;
                border-radius: 8px;
                margin-top: 12px;
                padding: 12px;
                font-size: 14px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 8px;
                background-color: #1a1a2e;
            }
            QTextEdit {
                background-color: #16213e;
                color: #ffffff;
                border: 1px solid #e94560;
                border-radius: 5px;
                font-size: 13px;
            }
        """)
        
        self.init_ui()
        
        # Tự động đóng sau 30 giây (có thể tắt)
        self.auto_close_timer = QTimer()
        self.auto_close_timer.timeout.connect(self.close)
        # self.auto_close_timer.start(30000)  # Bỏ comment nếu muốn tự đóng

    def init_ui(self):
        """Khởi tạo giao diện dialog."""
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # ===== HEADER =====
        header = self.create_header()
        main_layout.addWidget(header)
        
        # ===== NỘI DUNG CHÍNH (2 cột) =====
        content_layout = QHBoxLayout()
        content_layout.setSpacing(15)
        
        # Cột trái: Ảnh sự cố
        left_panel = self.create_image_panel()
        content_layout.addWidget(left_panel, 60)
        
        # Cột phải: Thông tin chi tiết
        right_panel = self.create_info_panel()
        content_layout.addWidget(right_panel, 40)
        
        main_layout.addLayout(content_layout)
        
        # ===== FOOTER: Nút điều khiển =====
        footer = self.create_footer()
        main_layout.addWidget(footer)

    def create_header(self):
        """Tạo phần header với tiêu đề và trạng thái."""
        header_frame = QFrame()
        header_frame.setStyleSheet("""
            QFrame {
                background-color: #e74c3c;
                border-radius: 10px;
                padding: 12px;
            }
        """)
        header_layout = QHBoxLayout(header_frame)
        
        # Icon và tiêu đề
        title_label = QLabel("🚨 SỰ CỐ GIAO THÔNG ĐƯỢC PHÁT HIỆN 🚨")
        title_label.setStyleSheet("""
            color: white;
            font-size: 22px;
            font-weight: bold;
        """)
        header_layout.addWidget(title_label)
        
        return header_frame

    def create_image_panel(self):
        """Tạo panel hiển thị ảnh/video sự cố."""
        panel = QGroupBox("📸 HÌNH ẢNH SỰ CỐ")
        layout = QVBoxLayout(panel)
        
        # Label hiển thị ảnh
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 350)
        self.image_label.setStyleSheet("""
            border: 2px solid #e94560;
            border-radius: 8px;
            background-color: #0f0f23;
        """)
        
        # Hiển thị frame nếu có
        if self.frame is not None:
            self.display_frame(self.frame)
        elif self.alert_data.get('clip_path'):
            # Thử đọc frame đầu tiên từ clip
            self.load_clip_preview(self.alert_data['clip_path'])
        else:
            self.image_label.setText("Không có ảnh sự cố")
            self.image_label.setStyleSheet("""
                border: 2px solid #e94560;
                border-radius: 8px;
                background-color: #0f0f23;
                color: #666;
                font-size: 16px;
            """)
        
        layout.addWidget(self.image_label)
        
        # Thông tin vị trí và thời gian
        info_text = ""
        if 'location' in self.alert_data:
            info_text += f"📍 Vị trí: {self.alert_data['location']}\n"
        if 'timestamp' in self.alert_data:
            time_str = datetime.fromtimestamp(self.alert_data['timestamp']).strftime('%H:%M:%S - %d/%m/%Y')
            info_text += f"🕐 Thời gian: {time_str}\n"
        if 'frame_id' in self.alert_data:
            info_text += f"🎬 Frame: {self.alert_data['frame_id']}"
        
        if info_text:
            info_label = QLabel(info_text)
            info_label.setStyleSheet("color: #2c3e50; font-size: 12px; padding: 5px;")
            layout.addWidget(info_label)
        
        return panel

    def create_info_panel(self):
        """Tạo panel thông tin chi tiết."""
        panel = QGroupBox("📊 THÔNG TIN CHI TIẾT")
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)
        
        # --- Thông tin cơ bản ---
        basic_group = QGroupBox("Thông tin cơ bản")
        basic_layout = QGridLayout(basic_group)
        
        track_id = self.alert_data.get('track_id', 'N/A')
        score = self.alert_data.get('final_score', 0.0)
        risk_level = self.alert_data.get('risk_level', 'UNKNOWN')
        
        basic_layout.addWidget(self._make_label("ID Đối tượng:"), 0, 0)
        basic_layout.addWidget(self._make_value(f"Track #{track_id}", "#3498db"), 0, 1)
        
        basic_layout.addWidget(self._make_label("Loại phương tiện:"), 1, 0)
        basic_layout.addWidget(self._make_value("Xe máy 🏍", "#2ecc71"), 1, 1)
        
        basic_layout.addWidget(self._make_label("Mức độ tin cậy:"), 2, 0)
        score_color = "#2ecc71" if score > 0.8 else "#f39c12" if score > 0.6 else "#e74c3c"
        basic_layout.addWidget(self._make_value(f"{score:.3f} ({score*100:.1f}%)", score_color), 2, 1)
        
        basic_layout.addWidget(self._make_label("Mức độ rủi ro:"), 3, 0)
        risk_colors = {
            'ACCIDENT': '#e74c3c',
            'DANGER': '#e67e22',
            'WARNING': '#f1c40f',
            'NORMAL': '#2ecc71'
        }
        basic_layout.addWidget(self._make_value(risk_level, risk_colors.get(risk_level, '#ffffff')), 3, 1)
        
        layout.addWidget(basic_group)
        
        # --- Luật vi phạm ---
        violations_group = QGroupBox("⚠ Luật vi phạm")
        violations_layout = QVBoxLayout(violations_group)
        
        violations = self.alert_data.get('violations', [])
        if violations:
            for violation in violations:
                violation_label = QLabel(f"• {violation}")
                violation_label.setStyleSheet("color: #e74c3c; font-size: 13px; padding: 3px;")
                violation_label.setWordWrap(True)
                violations_layout.addWidget(violation_label)
        else:
            no_violation = QLabel("Không có luật vi phạm cụ thể")
            no_violation.setStyleSheet("color: #5a6c7d; font-style: italic;")
            violations_layout.addWidget(no_violation)
        
        layout.addWidget(violations_group)
        
        # --- Phân tích ---
        analysis_group = QGroupBox("🔍 Phân tích")
        analysis_layout = QVBoxLayout(analysis_group)
        
        analysis_text = self._generate_analysis()
        analysis_label = QLabel(analysis_text)
        analysis_label.setWordWrap(True)
        analysis_label.setStyleSheet("""
            color: #2c3e50;
            background-color: #fafbfc;
            font-size: 13px;
            line-height: 1.6;
            padding: 10px;
            border-radius: 4px;
        """)
        analysis_layout.addWidget(analysis_label)
        
        layout.addWidget(analysis_group)
        
        return panel

    def create_footer(self):
        """Tạo phần footer với các nút điều khiển."""
        footer_layout = QHBoxLayout()
        footer_layout.setSpacing(10)
        
        # Nút xuất báo cáo
        btn_export = QPushButton("📄 Xuất báo cáo")
        btn_export.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                font-size: 14px;
                padding: 12px 20px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        btn_export.clicked.connect(self.export_report)
        footer_layout.addWidget(btn_export)
        
        # Nút lưu clip
        if self.alert_data.get('clip_path'):
            btn_save_clip = QPushButton("💾 Lưu clip")
            btn_save_clip.setStyleSheet("""
                QPushButton {
                    background-color: #27ae60;
                    color: white;
                    font-size: 14px;
                    padding: 12px 20px;
                    border-radius: 5px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #219a52;
                }
            """)
            btn_save_clip.clicked.connect(self.save_clip)
            footer_layout.addWidget(btn_save_clip)
        
        footer_layout.addStretch()
        
        # Nút đóng
        btn_close = QPushButton("✕ Đóng")
        btn_close.setStyleSheet("""
            QPushButton {
                background-color: #7f8c8d;
                color: white;
                font-size: 14px;
                padding: 12px 30px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #6c7a7a;
            }
        """)
        btn_close.clicked.connect(self.close)
        footer_layout.addWidget(btn_close)
        
        # Wrap trong frame
        footer_frame = QFrame()
        footer_frame.setLayout(footer_layout)
        
        return footer_frame

    def display_frame(self, frame):
        """Hiển thị frame lên image_label."""
        if frame is None:
            return
        
        # Resize frame để vừa với panel
        h, w = frame.shape[:2]
        max_w = 500
        if w > max_w:
            scale = max_w / w
            new_w = int(w * scale)
            new_h = int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h))
        
        # Vẽ thêm thông tin lên frame
        frame_with_info = self._draw_incident_info(frame)
        
        # Chuyển đổi sang QImage
        rgb_image = cv2.cvtColor(frame_with_info, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        self.image_label.setPixmap(QPixmap.fromImage(qt_image))

    def _draw_incident_info(self, frame):
        """Vẽ thông tin sự cố lên frame."""
        result = frame.copy()
        h, w = result.shape[:2]
        
        # Vẽ khung đỏ xung quanh
        cv2.rectangle(result, (5, 5), (w-5, h-5), (0, 0, 255), 3)
        
        # Vẽ text sự cố
        score = self.alert_data.get('final_score', 0.0)
        track_id = self.alert_data.get('track_id', '?')
        
        # Header
        header_text = f"ACCIDENT DETECTED - Track #{track_id}"
        cv2.putText(result, header_text, (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Score
        score_text = f"Score: {score:.3f}"
        cv2.putText(result, score_text, (20, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Violations
        violations = self.alert_data.get('violations', [])
        y_offset = 110
        for violation in violations[:3]:  # Tối đa 3 violation
            cv2.putText(result, f"• {violation}", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
            y_offset += 30
        
        # Timestamp
        if 'timestamp' in self.alert_data:
            time_str = datetime.fromtimestamp(self.alert_data['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
            cv2.putText(result, time_str, (20, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return result

    def load_clip_preview(self, clip_path):
        """Load frame đầu tiên từ clip để hiển thị preview."""
        try:
            cap = cv2.VideoCapture(clip_path)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                self.display_frame(frame)
            else:
                self.image_label.setText("Không thể đọc clip")
        except Exception as e:
            self.image_label.setText(f"Lỗi: {str(e)}")

    def _generate_analysis(self):
        """Tạo text phân tích tự động."""
        score = self.alert_data.get('final_score', 0.0)
        violations = self.alert_data.get('violations', [])
        
        analysis = "📋 PHÂN TÍCH SỰ CỐ:\n\n"
        
        if score >= 0.9:
            analysis += "🔴 Mức độ NGUY HIỂM - Xác suất tai nạn rất cao\n"
        elif score >= 0.7:
            analysis += "🟠 Mức độ CẢNH BÁO - Có dấu hiệu bất thường\n"
        else:
            analysis += "🟡 Mức độ CHÚ Ý - Cần theo dõi thêm\n"
        
        if violations:
            analysis += "\nCác yếu tố kích hoạt:\n"
            for v in violations:
                if 'BRAKE' in v:
                    analysis += "• Phanh đột ngột → Nguy cơ va chạm\n"
                if 'COLLISION' in v:
                    analysis += "• Khoảng cách quá gần → Nguy cơ va chạm\n"
                if 'LEAN' in v:
                    analysis += "• Góc nghiêng lớn → Nguy cơ ngã xe\n"
                if 'STOP' in v:
                    analysis += "• Dừng đột ngột → Cản trở giao thông\n"
        
        analysis += "\n✅ Hệ thống đã ghi nhận và lưu trữ sự kiện này."
        
        return analysis

    def _make_label(self, text):
        """Tạo label thông thường."""
        label = QLabel(text)
        label.setStyleSheet("color: #5a6c7d; font-size: 13px; font-weight: bold;")
        return label

    def _make_value(self, text, color="#ffffff"):
        """Tạo label giá trị với màu."""
        label = QLabel(text)
        label.setStyleSheet(f"color: {color}; font-size: 14px; font-weight: bold;")
        return label

    def export_report(self):
        """Xuất báo cáo sự cố."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        default_name = f"incident_report_{timestamp}.txt"
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Xuất báo cáo", default_name, 
            "Text Files (*.txt);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write("=" * 60 + "\n")
                    f.write("BÁO CÁO SỰ CỐ GIAO THÔNG\n")
                    f.write("=" * 60 + "\n\n")
                    f.write(f"Thời gian: {datetime.fromtimestamp(self.alert_data.get('timestamp', 0)).strftime('%d/%m/%Y %H:%M:%S')}\n")
                    f.write(f"ID Đối tượng: Track #{self.alert_data.get('track_id', 'N/A')}\n")
                    f.write(f"Mức độ tin cậy: {self.alert_data.get('final_score', 0):.3f}\n")
                    f.write(f"Mức độ rủi ro: {self.alert_data.get('risk_level', 'N/A')}\n")
                    f.write(f"Frame: {self.alert_data.get('frame_id', 'N/A')}\n\n")
                    f.write("Luật vi phạm:\n")
                    for v in self.alert_data.get('violations', []):
                        f.write(f"  • {v}\n")
                    f.write(f"\nVị trí: {self.alert_data.get('location', 'Đường Lĩnh Nam - Hà Nội')}\n")
                    f.write(f"Clip: {self.alert_data.get('clip_path', 'Không có')}\n")
                    f.write("\n" + "=" * 60 + "\n")
                    f.write("Hệ thống Giám sát Giao thông Lĩnh Nam - NCKH 2026\n")
                
                QMessageBox.information(self, "Thành công", f"Đã xuất báo cáo:\n{file_path}")
            except Exception as e:
                QMessageBox.warning(self, "Lỗi", f"Không thể xuất báo cáo:\n{str(e)}")

    def save_clip(self):
        """Lưu clip sự cố."""
        clip_path = self.alert_data.get('clip_path')
        if not clip_path:
            QMessageBox.warning(self, "Lỗi", "Không có clip để lưu")
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        default_name = f"incident_clip_{timestamp}.mp4"
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Lưu clip sự cố", default_name,
            "Video Files (*.mp4);;All Files (*)"
        )
        
        if file_path:
            try:
                import shutil
                shutil.copy2(clip_path, file_path)
                QMessageBox.information(self, "Thành công", f"Đã lưu clip:\n{file_path}")
            except Exception as e:
                QMessageBox.warning(self, "Lỗi", f"Không thể lưu clip:\n{str(e)}")


# ===== HÀM TIỆN ÍCH =====

def show_incident_dialog(alert_data, frame=None, parent=None):
    """
    Hàm tiện ích để hiển thị dialog từ bất kỳ đâu.
    
    Usage:
        from gui.incident_dialog import show_incident_dialog
        
        alert_data = {
            'track_id': 42,
            'timestamp': time.time(),
            'frame_id': 1234,
            'final_score': 0.956,
            'risk_level': 'ACCIDENT',
            'violations': ['SUDDEN_BRAKE', 'COLLISION_RISK'],
            'location': 'Đường Lĩnh Nam - Hà Nội'
        }
        show_incident_dialog(alert_data, frame)
    """
    dialog = IncidentDialog(alert_data, frame, parent)
    dialog.exec_()


def show_demo_incident(parent=None):
    """
    Hiển thị dialog với dữ liệu mẫu (dùng cho demo).
    """
    import time
    
    # Tạo frame mẫu (màu đen với text)
    demo_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(demo_frame, "INCIDENT FRAME", (150, 240),
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    cv2.rectangle(demo_frame, (50, 50), (590, 430), (0, 0, 255), 2)
    
    alert_data = {
        'track_id': 42,
        'timestamp': time.time(),
        'frame_id': 1234,
        'final_score': 0.956,
        'risk_level': 'ACCIDENT',
        'violations': [
            'SUDDEN_BRAKE: Phanh gấp a=-6.8 m/s²',
            'COLLISION_RISK: Khoảng cách 1.3m khi v=8.5 m/s',
            'HIGH_LEAN_ANGLE: Góc nghiêng 34°'
        ],
        'location': 'Ngã tư Lĩnh Nam - Hà Nội',
        'clip_path': None
    }
    
    dialog = IncidentDialog(alert_data, demo_frame, parent)
    dialog.exec_()


# ===== TEST =====
if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    
    # Test với dữ liệu mẫu
    show_demo_incident()
    
    sys.exit(app.exec_())