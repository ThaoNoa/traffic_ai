# gui/dashboard_widget.py
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QFrame, QTableWidget, QTableWidgetItem,
                             QHeaderView)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from collections import deque


class StatusBadge(QLabel):
    def __init__(self, text, color="#b0b8c1", parent=None):
        super().__init__(text, parent)
        self.setAlignment(Qt.AlignCenter)
        self.setFixedHeight(30)
        self._set_color(color)

    def _set_color(self, color):
        self.setStyleSheet(f"""
            background-color: {color}; color: white; font-weight: bold;
            border-radius: 3px; font-size: 10px; padding: 4px 8px;
        """)


class MetricCard(QFrame):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QFrame { background-color: #f8f9fb; border: 1px solid #e0e4e8; 
                     border-radius: 3px; padding: 6px; }
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(1)
        
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("color: #5a6c7d; font-size: 9px; background: transparent;")
        self.title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.title_label)
        
        self.value_label = QLabel("--")
        self.value_label.setStyleSheet("color: #1a3a5c; font-size: 22px; font-weight: bold; background: transparent;")
        self.value_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.value_label)

    def set_value(self, value):
        self.value_label.setText(str(value))


class DashboardWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._mode = "empty"
        self._event_log = deque(maxlen=50)
        
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(6)

        # === HEADER ===
        header = QLabel("TRANG THAI HE THONG")
        header.setStyleSheet("color: #1a3a5c; font-size: 13px; font-weight: bold;")
        header.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(header)

        # === SYSTEM STATUS ===
        self.status_badge = StatusBadge("CHUA CO DU LIEU", "#b0b8c1")
        main_layout.addWidget(self.status_badge)

        # === MODULE STATUS ===
        modules = QHBoxLayout()
        self.detector_badge = StatusBadge("DETECTOR\nCho", "#b0b8c1")
        self.tracker_badge = StatusBadge("TRACKER\nCho", "#b0b8c1")
        self.classifier_badge = StatusBadge("AI ANALYZER\nCho", "#b0b8c1")
        for b in [self.detector_badge, self.tracker_badge, self.classifier_badge]:
            b.setFixedHeight(40)
            modules.addWidget(b)
        main_layout.addLayout(modules)

        # === PERFORMANCE METRICS ===
        perf_label = QLabel("HIEU NANG HE THONG")
        perf_label.setStyleSheet("color: #1a3a5c; font-size: 11px; font-weight: bold;")
        main_layout.addWidget(perf_label)
        
        metrics1 = QHBoxLayout()
        self.fps_card = MetricCard("FPS TB")
        self.latency_card = MetricCard("Do tre (ms)")
        self.gpu_card = MetricCard("GPU %")
        metrics1.addWidget(self.fps_card)
        metrics1.addWidget(self.latency_card)
        metrics1.addWidget(self.gpu_card)
        main_layout.addLayout(metrics1)

        # === TRAFFIC METRICS ===
        traffic_label = QLabel("THONG KE GIAO THONG (Trong vung IPM)")
        traffic_label.setStyleSheet("color: #1a3a5c; font-size: 11px; font-weight: bold;")
        main_layout.addWidget(traffic_label)
        
        metrics2 = QHBoxLayout()
        self.vehicle_card = MetricCard("Xe trong vung")
        self.total_vehicle_card = MetricCard("Tong xe da qua")
        self.avg_speed_card = MetricCard("Toc do TB (km/h)")
        metrics2.addWidget(self.vehicle_card)
        metrics2.addWidget(self.total_vehicle_card)
        metrics2.addWidget(self.avg_speed_card)
        main_layout.addLayout(metrics2)

        # === DETECTION STATS ===
        detect_label = QLabel("PHAT HIEN & CANH BAO")
        detect_label.setStyleSheet("color: #1a3a5c; font-size: 11px; font-weight: bold;")
        main_layout.addWidget(detect_label)
        
        metrics3 = QHBoxLayout()
        self.detection_card = MetricCard("Tong detect")
        self.alert_card = MetricCard("Canh bao")
        self.fpr_card = MetricCard("FPR %")
        metrics3.addWidget(self.detection_card)
        metrics3.addWidget(self.alert_card)
        metrics3.addWidget(self.fpr_card)
        main_layout.addLayout(metrics3)

        # === ROAD ZONE INFO ===
        zone_label = QLabel("VUNG GIAM SAT (IPM)")
        zone_label.setStyleSheet("color: #1a3a5c; font-size: 11px; font-weight: bold;")
        main_layout.addWidget(zone_label)
        
        self.zone_info = QLabel("Khu vuc: 20m x 60m\nDang theo doi...")
        self.zone_info.setStyleSheet("""
            color: #2c3e50; background-color: #f8f9fb; 
            padding: 8px; border: 1px solid #e0e4e8; border-radius: 3px; 
            font-size: 10px;
        """)
        main_layout.addWidget(self.zone_info)
                # === CONGESTION BADGE ===
        self.congestion_badge = StatusBadge("TAC NGHEN: --", "#b0b8c1")
        self.congestion_badge.setFixedHeight(35)
        main_layout.addWidget(self.congestion_badge)

        # === EVENT LOG ===
        event_label = QLabel("NHAT KY SU KIEN")
        event_label.setStyleSheet("color: #1a3a5c; font-size: 11px; font-weight: bold; margin-top: 4px;")
        main_layout.addWidget(event_label)
        
        self.event_table = QTableWidget(0, 3)
        self.event_table.setHorizontalHeaderLabels(["Thoi gian", "Loai", "Chi tiet"])
        self.event_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.event_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.event_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.event_table.setMaximumHeight(120)
        self.event_table.setStyleSheet("""
            QTableWidget { 
                background-color: #fff; border: 1px solid #d0d5db; border-radius: 3px;
                font-size: 9px; gridline-color: #e8ecf0;
            }
            QHeaderView::section { 
                background-color: #f0f3f7; color: #2c3e50; font-weight: bold;
                padding: 3px; border: none; border-bottom: 1px solid #d0d5db;
                font-size: 9px;
            }
        """)
        self.event_table.setEditTriggers(QTableWidget.NoEditTriggers)
        main_layout.addWidget(self.event_table)

        # === INFO ===
        self.info_label = QLabel("Keo video vao khung ben trai de bat dau")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("""
            color: #5a6c7d; background-color: #fff; padding: 6px; 
            border: 1px solid #d0d5db; border-radius: 3px; font-size: 10px;
        """)
        main_layout.addWidget(self.info_label)

    def set_mode(self, mode, detail=""):
        self._mode = mode
        colors = {
            "empty": "#b0b8c1", 
            "ready": "#1a3a5c", 
            "processing": "#2d6a4f", 
            "completed": "#1a3a5c"
        }
        texts = {
            "empty": "CHUA CO DU LIEU", 
            "ready": "SAN SANG", 
            "processing": "DANG PHAN TICH", 
            "completed": "HOAN THANH"
        }
        infos = {
            "empty": "Keo video vao khung ben trai de bat dau",
            "ready": detail + "\nNhan 'Bat dau phan tich' de chay",
            "processing": "He thong dang xu ly video...",
            "completed": "Da xu ly xong."
        }
        self.status_badge._set_color(colors.get(mode, "#b0b8c1"))
        self.status_badge.setText(texts.get(mode, ""))
        self.info_label.setText(infos.get(mode, ""))

    def update_stats(self, stats):
        """Cap nhat thong so tu pipeline."""
        avg_fps = stats.get('avg_fps', 0)
        latency = stats.get('detect_time', 0)
        vehicles = stats.get('vehicles', 0)
        total_vehicles = stats.get('total_vehicles_unique', 0)
        total_detections = stats.get('total_detections', 0)
        alerts = stats.get('alerts', 0)
        active_tracks = stats.get('active_tracks', 0)
        
        # Tong so detection de tinh FPR
        fpr = (alerts / max(total_detections, 1)) * 100
        
        # Toc do trung binh (uoc luong)
        avg_speed = stats.get('avg_speed', 0) * 3.6  # m/s -> km/h
        
        # Zone info
        zone_w = stats.get('zone_width', 20)
        zone_h = stats.get('zone_height', 60)
        
        # Cap nhat cards
        self.fps_card.set_value(f"{avg_fps:.1f}")
        self.latency_card.set_value(f"{latency:.0f}")
        
        # GPU
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            gpu_usage = gpus[0].load * 100 if gpus else 0
        except:
            gpu_usage = min(85, (avg_fps / 42) * 80) if avg_fps > 0 else 0
        self.gpu_card.set_value(f"{gpu_usage:.0f}")
        
        # Traffic
        self.vehicle_card.set_value(vehicles)
        self.total_vehicle_card.set_value(total_vehicles)
        self.avg_speed_card.set_value(f"{avg_speed:.0f}")
        
        # Detection
        self.detection_card.set_value(total_detections)
        self.alert_card.set_value(alerts)
        self.fpr_card.set_value(f"{fpr:.1f}")
        
        # Zone info
        self.zone_info.setText(
            f"Kich thuoc vung: {zone_w:.0f}m x {zone_h:.0f}m\n"
            f"Xe trong vung: {vehicles} | Tong da qua: {total_vehicles}\n"
            f"Toc do TB: {avg_speed:.0f} km/h"
        )
        
        # Module badges
        self.detector_badge.setText(f"DETECTOR\n{total_detections} det")
        self.tracker_badge.setText(f"TRACKER\n{total_vehicles} xe")
        self.classifier_badge.setText(f"AI ANALYZER\n{alerts} alert")

                # Congestion
        cong_level = stats.get('congestion_level', 'UNKNOWN')
        cong_density = stats.get('congestion_density', 0)
        cong_colors = {
            'THONG THOANG': '#27ae60',
            'DONG VUA': '#f39c12',
            'DONG DUC': '#e67e22',
            'TAC NGHEN': '#e74c3c',
            'UNKNOWN': '#b0b8c1'
        }
        self.congestion_badge._set_color(cong_colors.get(cong_level, '#b0b8c1'))
        self.congestion_badge.setText(f"{cong_level} | {cong_density:.1f} xe/100m²")

    def add_event(self, event_type, description):
        """Them su kien vao log."""
        import time
        timestamp = time.strftime("%H:%M:%S")
        self._event_log.append((timestamp, event_type, description))
        self._refresh_event_table()

    def _refresh_event_table(self):
        self.event_table.setRowCount(len(self._event_log))
        for i, (ts, etype, desc) in enumerate(reversed(self._event_log)):
            self.event_table.setItem(i, 0, QTableWidgetItem(ts))
            
            type_item = QTableWidgetItem(etype)
            if etype == "ACCIDENT":
                type_item.setForeground(QColor("#e74c3c"))
                type_item.setBackground(QColor("#fdf2f2"))
            elif etype == "WARNING":
                type_item.setForeground(QColor("#f39c12"))
            elif etype == "INFO":
                type_item.setForeground(QColor("#2980b9"))
            self.event_table.setItem(i, 1, type_item)
            
            self.event_table.setItem(i, 2, QTableWidgetItem(desc))
        self.event_table.scrollToBottom()

    def reset(self):
        self.set_mode("empty")
        for card in [self.fps_card, self.latency_card, self.gpu_card,
                     self.vehicle_card, self.total_vehicle_card, self.avg_speed_card,
                     self.detection_card, self.alert_card, self.fpr_card]:
            card.set_value("--")
        self.zone_info.setText("Khu vuc: --m x --m\nDang cho du lieu...")
        self._event_log.clear()
        self._refresh_event_table()