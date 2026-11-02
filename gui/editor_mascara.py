"""
Módulo: editor_mascara.py
========================
Diálogo interactivo para retocar manualmente la máscara de la lesión (Pincel/Borrador).
"""

import cv2
import numpy as np
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt


class EditorMascaraDialog(QDialog):
    def __init__(self, rgb, mascara_inicial, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Editor Interactivo de Máscara de Lesión")
        self.resize(600, 650)

        self.rgb = rgb.copy()
        
        # Validación si la máscara viene nula o booleana
        if mascara_inicial is None:
            self.mascara = np.zeros(self.rgb.shape[:2], dtype=np.uint8)
        else:
            self.mascara = (mascara_inicial.astype(np.uint8)) * 255 if mascara_inicial.dtype == bool else mascara_inicial.copy()

        self.modo_dibujo = True  # True = Agregar Lesión, False = Borrar
        self.radio_pincel = 12

        layout = QVBoxLayout(self)
        
        self.lbl_canvas = QLabel()
        self.lbl_canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_canvas.mouseMoveEvent = self.dibujar
        self.lbl_canvas.mousePressEvent = self.iniciar_dibujo
        layout.addWidget(self.lbl_canvas)

        btn_layout = QHBoxLayout()
        
        btn_pintar = QPushButton("🖌️ Agregar Lesión")
        btn_pintar.clicked.connect(lambda: self.set_modo(True))
        
        btn_borrar = QPushButton("🧹 Borrar Fondo")
        btn_borrar.clicked.connect(lambda: self.set_modo(False))
        
        btn_ok = QPushButton("✅ Guardar Corrección")
        btn_ok.clicked.connect(self.accept)

        btn_layout.addWidget(btn_pintar)
        btn_layout.addWidget(btn_borrar)
        btn_layout.addWidget(btn_ok)
        layout.addLayout(btn_layout)

        self.actualizar_vista()

    def set_modo(self, agregar):
        self.modo_dibujo = agregar

    def iniciar_dibujo(self, event):
        self.aplicar_pincel(event.pos())

    def dibujar(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton:
            self.aplicar_pincel(event.pos())

    def aplicar_pincel(self, pos):
        pm = self.lbl_canvas.pixmap()
        if pm is None or pm.width() == 0 or pm.height() == 0:
            return

        # El pixmap va centrado dentro de la etiqueta: restar el margen antes de escalar
        off_x = (self.lbl_canvas.width() - pm.width()) / 2
        off_y = (self.lbl_canvas.height() - pm.height()) / 2

        h, w = self.mascara.shape[:2]
        x = int((pos.x() - off_x) * w / pm.width())
        y = int((pos.y() - off_y) * h / pm.height())
        if not (0 <= x < w and 0 <= y < h):
            return

        val = 255 if self.modo_dibujo else 0
        cv2.circle(self.mascara, (x, y), self.radio_pincel, val, -1)
        self.actualizar_vista()

    def actualizar_vista(self):
        overlay = self.rgb.copy()
        mask_bool = self.mascara > 127
        overlay[mask_bool] = (0.55 * overlay[mask_bool] + 0.45 * np.array([255, 0, 255])).astype(np.uint8)

        h, w, _ = overlay.shape
        q_img = QImage(overlay.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self.lbl_canvas.setPixmap(QPixmap.fromImage(q_img).scaled(512, 512, Qt.AspectRatioMode.KeepAspectRatio))

    def obtener_mascara_final(self):
        return self.mascara > 127