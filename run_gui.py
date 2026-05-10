# run_gui.py
import os
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)  # Đã có sẵn

# THÊM DÒNG NÀY:
import cv2  
# ĐỂ cv2 import TRƯỚC PyQt5 — fix xung đột plugin

import sys
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtWidgets import QApplication
from gui.main_window import MainWindow

STYLE = """
QMainWindow { 
    background-color: #f5f6fa; 
}
QLabel { 
    color: #2c3e50; 
    font-family: 'Times New Roman', 'Segoe UI', serif;
}
QGroupBox {
    color: #1a3a5c;
    border: 1px solid #c0c7cf;
    border-radius: 4px;
    margin-top: 12px;
    padding: 12px;
    font-size: 13px;
    font-weight: bold;
    background-color: #ffffff;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 6px;
    background-color: #ffffff;
}
QPushButton {
    background-color: #1a3a5c;
    color: white;
    border: none;
    padding: 8px 18px;
    border-radius: 3px;
    font-size: 12px;
    font-family: 'Times New Roman', 'Segoe UI', serif;
}
QPushButton:hover {
    background-color: #2c5282;
}
QPushButton:pressed {
    background-color: #0f2440;
}
QPushButton:disabled {
    background-color: #b0b8c1;
    color: #e0e0e0;
}
QComboBox {
    background-color: #ffffff;
    color: #2c3e50;
    border: 1px solid #c0c7cf;
    border-radius: 3px;
    padding: 6px 10px;
    font-size: 12px;
    font-family: 'Times New Roman', 'Segoe UI', serif;
}
QComboBox:hover {
    border-color: #1a3a5c;
}
QComboBox::drop-down {
    border: none;
}
QComboBox QAbstractItemView {
    background-color: #ffffff;
    color: #2c3e50;
    selection-background-color: #1a3a5c;
    selection-color: white;
}
QCheckBox {
    color: #2c3e50;
    font-size: 12px;
    font-family: 'Times New Roman', 'Segoe UI', serif;
}
QSlider::groove:horizontal {
    height: 5px;
    background: #d0d5db;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #1a3a5c;
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}
QProgressBar {
    background-color: #e8ecf0;
    border-radius: 3px;
    border: none;
    text-align: center;
    font-size: 10px;
    color: #2c3e50;
}
QProgressBar::chunk {
    background-color: #1a3a5c;
    border-radius: 3px;
}
QSplitter::handle {
    background-color: #c0c7cf;
    width: 1px;
}
QScrollArea {
    border: none;
    background-color: transparent;
}
QStatusBar {
    background-color: #e8ecf0;
    color: #5a6c7d;
    border-top: 1px solid #c0c7cf;
    font-size: 11px;
}
QToolBar {
    background-color: #ffffff;
    border-bottom: 1px solid #c0c7cf;
    padding: 4px;
    spacing: 8px;
}
QToolButton {
    color: #2c3e50;
    font-size: 12px;
    padding: 5px 12px;
    border-radius: 3px;
    font-family: 'Times New Roman', 'Segoe UI', serif;
}
QToolButton:hover {
    background-color: #e8ecf0;
}
QTextEdit {
    background-color: #ffffff;
    color: #2c3e50;
    border: 1px solid #c0c7cf;
    border-radius: 3px;
    font-size: 12px;
    font-family: 'Times New Roman', 'Segoe UI', serif;
}
QLineEdit {
    background-color: #ffffff;
    color: #2c3e50;
    border: 1px solid #c0c7cf;
    border-radius: 3px;
    padding: 6px 10px;
    font-size: 12px;
}
"""

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())