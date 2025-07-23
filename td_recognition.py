# -*- coding: utf-8 -*-

# RoboCup 3D 目标检测系统 - 美化版界面
# 现代化设计，支持深色主题和渐变效果

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap, QFont, QPalette, QColor, QLinearGradient, QBrush
from PyQt5.QtCore import Qt


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1400, 800)  # 增大窗口尺寸
        MainWindow.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #2C3E50, stop:1 #34495E);
            }
        """)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.ImgLabel = QtWidgets.QLabel(self.centralwidget)
        self.ImgLabel.setGeometry(QtCore.QRect(30, 50, 720, 540))  # 增大显示区域
        self.ImgLabel.setStyleSheet("""
            QLabel {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #ECF0F1, stop:1 #BDC3C7);
                border: 3px solid #3498DB;
                border-radius: 15px;
                padding: 5px;
            }
        """)
        self.ImgLabel.setText("📹 摄像头画面")
        self.ImgLabel.setAlignment(Qt.AlignCenter)
        self.ImgLabel.setFont(QFont("Microsoft YaHei", 16, QFont.Bold))
        self.ImgLabel.setObjectName("ImgLabel")

        self.TurningImg = QtWidgets.QLabel(self.centralwidget)
        self.TurningImg.setGeometry(QtCore.QRect(1200, 600, 120, 120))
        self.TurningImg.setStyleSheet("""
            QLabel {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #E74C3C, stop:1 #C0392B);
                border: 3px solid #A93226;
                border-radius: 60px;
                color: white;
                font: bold 14px "Microsoft YaHei";
            }
        """)
        self.TurningImg.setText("🔄\n转向中")
        self.TurningImg.setAlignment(Qt.AlignCenter)
        self.TurningImg.setScaledContents(True)
        self.TurningImg.setObjectName("TurningImg")
        self.TurningImg.setVisible(False)



        self.ResultLabel = QtWidgets.QLabel(self.centralwidget)
        self.ResultLabel.setGeometry(QtCore.QRect(780, 50, 580, 450))
        self.ResultLabel.setFont(QFont("Microsoft YaHei", 11))
        self.ResultLabel.setStyleSheet("""
            QLabel {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #FFFFFF, stop:1 #F8F9FA);
                border: 2px solid #27AE60;
                border-radius: 12px;
                padding: 15px;
                color: #2C3E50;
            }
        """)
        self.ResultLabel.setText("🎯 检测结果将在这里显示...")
        self.ResultLabel.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.ResultLabel.setWordWrap(True)
        self.ResultLabel.setObjectName("ResultLabel")



        self.StartButton = QtWidgets.QPushButton(self.centralwidget)
        self.StartButton.setGeometry(QtCore.QRect(300, 620, 200, 60))
        self.StartButton.setFont(QFont("Microsoft YaHei", 14, QFont.Bold))
        self.StartButton.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #3498DB, stop:1 #2980B9);
                border: none;
                border-radius: 30px;
                color: white;
                padding: 10px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #5DADE2, stop:1 #3498DB);
                transform: scale(1.05);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #2980B9, stop:1 #1F618D);
            }
        """)
        self.StartButton.setObjectName("StartButton")
        self.StartButton.setVisible(False)

        # 新增标题标签
        self.titleLabel = QtWidgets.QLabel(self.centralwidget)
        self.titleLabel.setGeometry(QtCore.QRect(30, 10, 720, 40))
        self.titleLabel.setText("🤖 RoboCup 3D 目标检测系统")
        self.titleLabel.setFont(QFont("Microsoft YaHei", 18, QFont.Bold))
        self.titleLabel.setStyleSheet("""
            QLabel {
                color: white;
                background: transparent;
            }
        """)
        self.titleLabel.setAlignment(Qt.AlignCenter)
        self.titleLabel.setObjectName("titleLabel")





        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(780, 520, 580, 90))
        self.label.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        self.label.setStyleSheet("""
            QLabel {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #F39C12, stop:1 #E67E22);
                border: 2px solid #D68910;
                border-radius: 12px;
                padding: 10px;
                color: white;
            }
        """)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")



        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1400, 25))
        self.menubar.setStyleSheet("""
            QMenuBar {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #34495E, stop:1 #2C3E50);
                color: white;
                font: bold 11px "Microsoft YaHei";
            }
            QMenuBar::item:selected {
                background: #3498DB;
            }
        """)
        self.menubar.setObjectName("menubar")
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setObjectName("menu")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setStyleSheet("""
            QStatusBar {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2C3E50, stop:1 #34495E);
                color: white;
                border-top: 1px solid #3498DB;
            }
        """)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar.addAction(self.menu.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "🤖 RoboCup 3D 目标检测系统"))
        self.StartButton.setText(_translate("MainWindow", "🚀 开始检测"))
        self.label.setText(_translate("MainWindow", "💤 系统空闲 - 等待开始检测"))
        self.menu.setTitle(_translate("MainWindow", "关于"))
