import os
import sys
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QListWidget, QListWidgetItem,
    QMessageBox, QStackedWidget, QScrollArea, QGridLayout, QFrame,
    QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
import pyqtgraph as pg
from gui.editor_mascara import EditorMascaraDialog

# Importaciones de las etapas disponibles
from etapas.e0_normalizacion_roi import (
    cargar_imagen_rgb,
    normalizar_espacialmente,
    guardar_imagen_roi,
    registrar_roi_csv
)
from etapas.e1_mascara_lab import procesar_etapa1, guardar_visualizacion_e1
from etapas.e2_mejoramiento import procesar_etapa2, guardar_visualizacion_e2
from etapas.e3_segmentacion_superpixeles import procesar_etapa3, guardar_visualizacion_e3
from etapas.e4_morfologia import procesar_etapa4, guardar_visualizacion_e4, exportar_e4, pintar_zonas, _contorno
from etapas import e6_clasificacion

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sistema Biomédico - Detección de Tinea Corporis")
        self.resize(1350, 850)

        self.current_path = None
        self.current_rgb = None
        self.rois_rotadas_guardadas = {}  # { ruta_imagen: matriz_512x512 }
        self.imagenes_seleccionadas = []
        self.mascaras_automaticas_e3 = {}
        self.mascaras_manuales = {}
        self.resultados_etapa5 = {}  # { ruta_imagen: resultado de e6_clasificacion.clasificar }
        self.clasificador = None     # se lee de resultados/Etapa_5/clasificador.json al llegar a la Etapa 5

        # CONTENEDOR MULTIPANTALLA (QStackedWidget)
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        # Diccionario para almacenar los layouts de las grillas por etapa
        self.grids = {}

        # -------------------------------------------------------------------
        # CONSTRUCCIÓN DE LA SECUENCIA DE PANTALLAS (0 a 6)
        # -------------------------------------------------------------------
        self.stack.addWidget(self.crear_pantalla_0_recorte())                                                        # Índice 0
        self.stack.addWidget(self.crear_pantalla_etapa(1, "Etapa 1: Espacio de Color CIELAB", 0, 2, "Ir a Mejoramiento ->")) # Índice 1
        self.stack.addWidget(self.crear_pantalla_etapa(2, "Etapa 2: Mejoramiento y Resta Periférica", 1, 3, "Ir a Segmentación ->")) # Índice 2
        self.stack.addWidget(self.crear_pantalla_etapa(3, "Etapa 3: Segmentación (SLIC + K-Means)", 2, 4, "Ir a Morfología ->"))   # Índice 3
        self.stack.addWidget(self.crear_pantalla_etapa(4, "Etapa 4: Limpieza Morfológica", 3, 5, "Ir a Texturas ->"))             # Índice 4
        self.stack.addWidget(self.crear_pantalla_etapa(5, "Etapa 5: Descriptores de Textura (GLCM + LBP)", 4, 6, "Ir a Diagnóstico Final ->"))# Índice 5
        self.stack.addWidget(self.crear_pantalla_6_diagnostico())                                                   # Índice 6

    # =========================================================================
    # PANTALLA 0: Recorte y Ajuste de ROIs
    # =========================================================================
    def crear_pantalla_0_recorte(self):
        widget = QWidget()
        layout_principal = QHBoxLayout(widget)

        # 1. Panel de archivos
        panel_lista = QVBoxLayout()
        btn_cargar = QPushButton("1. Cargar Carpeta de Datos")
        btn_cargar.clicked.connect(self.cargar_carpeta)

        layout_sel = QHBoxLayout()
        btn_marcar = QPushButton("Marcar Todas")
        btn_marcar.clicked.connect(lambda: self.marcar_todas(True))
        btn_desmarcar = QPushButton("Desmarcar Todas")
        btn_desmarcar.clicked.connect(lambda: self.marcar_todas(False))
        layout_sel.addWidget(btn_marcar)
        layout_sel.addWidget(btn_desmarcar)

        self.list_widget = QListWidget()
        self.list_widget.itemClicked.connect(self.imagen_seleccionada)

        btn_iniciar = QPushButton("Iniciar Pipeline (Ir a E1) ->")
        btn_iniciar.setStyleSheet("font-weight: bold; background-color: #2e7d32; color: white; padding: 12px; font-size: 14px;")
        btn_iniciar.clicked.connect(self.ejecutar_e1_y_mostrar)

        panel_lista.addWidget(btn_cargar)
        panel_lista.addLayout(layout_sel)
        panel_lista.addWidget(self.list_widget)
        panel_lista.addWidget(btn_iniciar)

        # 2. Visor interactivo Pyqtgraph
        panel_visor = QVBoxLayout()
        panel_visor.addWidget(QLabel("<b>2. Ajustar ROI Rotada (512x512 px):</b>"))

        self.graph_widget = pg.GraphicsLayoutWidget()
        self.view_box = self.graph_widget.addViewBox()
        self.view_box.setAspectLocked(True)

        self.img_item = pg.ImageItem()
        self.view_box.addItem(self.img_item)

        self.roi = pg.RectROI([50, 50], [200, 200], rotatable=True, pen=pg.mkPen('r', width=2))
        self.roi.getHandles()
        self.roi.addRotateHandle([0.5, 1.25], [0.5, 0.5])
        self.roi.addScaleHandle([1, 1], [0, 0])
        self.roi.addScaleHandle([0, 0], [1, 1])
        self.roi.addScaleHandle([1, 0], [0, 1])
        self.roi.addScaleHandle([0, 1], [1, 0])
        self.view_box.addItem(self.roi)

        self.roi.sigRegionChanged.connect(self.generar_vista_previa_rgb)

        btn_guardar_roi = QPushButton("Guardar ROI de esta Imagen")
        btn_guardar_roi.setStyleSheet("background-color: #1976D2; color: white; font-weight: bold; padding: 8px;")
        btn_guardar_roi.clicked.connect(self.guardar_roi_confirmada)

        panel_visor.addWidget(self.graph_widget)
        panel_visor.addWidget(btn_guardar_roi)

        # 3. Previsualización ROI RGB
        panel_preview = QVBoxLayout()
        panel_preview.addWidget(QLabel("<b>Previsualización ROI RGB:</b>"))
        self.lbl_preview = QLabel("Esperando selección...")
        self.lbl_preview.setFixedSize(256, 256)
        self.lbl_preview.setStyleSheet("border: 2px dashed #9e9e9e; background: #f5f5f5;")
        self.lbl_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        panel_preview.addWidget(self.lbl_preview, alignment=Qt.AlignmentFlag.AlignCenter)
        panel_preview.addStretch()

        layout_principal.addLayout(panel_lista, 2)
        layout_principal.addLayout(panel_visor, 4)
        layout_principal.addLayout(panel_preview, 2)

        return widget

    # =========================================================================
    # PLANTILLA GENERADORA PARA PANTALLAS DE GRILLA (ETAPAS 1 a 5)
    # =========================================================================
    def crear_pantalla_etapa(self, num_etapa, titulo, index_atras, index_siguiente, texto_siguiente):
        widget = QWidget()
        layout_principal = QVBoxLayout(widget)

        header_layout = QHBoxLayout()
        btn_atras = QPushButton("<- Volver")
        btn_atras.setStyleSheet("background-color: #757575; color: white; font-weight: bold; padding: 8px;")
        btn_atras.clicked.connect(lambda: self.stack.setCurrentIndex(index_atras))

        lbl_titulo = QLabel(f"<h2>{titulo}</h2>")
        header_layout.addWidget(btn_atras)
        header_layout.addWidget(lbl_titulo)
        header_layout.addStretch()

        btn_siguiente = QPushButton(texto_siguiente)
        btn_siguiente.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold; padding: 8px;")
        
        # Conectar el botón a la función correspondiente según la etapa
        if num_etapa == 1:
            btn_siguiente.clicked.connect(self.ejecutar_e2_y_mostrar)
        elif num_etapa == 2:
            btn_siguiente.clicked.connect(self.ejecutar_e3_y_mostrar)
        elif num_etapa == 3:
            btn_siguiente.clicked.connect(self.ejecutar_e4_y_mostrar)
        elif num_etapa == 4:
            btn_siguiente.clicked.connect(self.ejecutar_e5_y_mostrar)
        elif num_etapa == 5:
            btn_siguiente.clicked.connect(self.ejecutar_diagnostico_y_mostrar)

        header_layout.addWidget(btn_siguiente)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        grid_layout = QGridLayout(scroll_content)
        scroll_area.setWidget(scroll_content)

        self.grids[num_etapa] = grid_layout

        layout_principal.addLayout(header_layout)
        layout_principal.addWidget(scroll_area)

        return widget

    # =========================================================================
    # PANTALLA 6: Diagnóstico Final
    # =========================================================================
    def crear_pantalla_6_diagnostico(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        header = QHBoxLayout()
        btn_atras = QPushButton("<- Volver a Texturas")
        btn_atras.setStyleSheet("background-color: #757575; color: white; font-weight: bold; padding: 8px;")
        btn_atras.clicked.connect(lambda: self.stack.setCurrentIndex(5))
        
        lbl_titulo = QLabel("<h2>Diagnóstico Final y Evaluación del Lote</h2>")
        header.addWidget(btn_atras)
        header.addWidget(lbl_titulo)
        header.addStretch()

        # Las columnas de descriptores se arman al mostrar el diagnóstico (dependen de clasificador.json)
        self.tabla_diagnostico = QTableWidget()
        self.tabla_diagnostico.setColumnCount(2)
        self.tabla_diagnostico.setHorizontalHeaderLabels(["Imagen", "Diagnóstico (regla de votos)"])
        self.tabla_diagnostico.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        layout.addLayout(header)
        layout.addWidget(self.tabla_diagnostico)

        return widget

    # =========================================================================
    # GESTIÓN DE ROIS Y CARGADO DE ARCHIVOS
    # =========================================================================
    def cargar_carpeta(self):
        folder = QFileDialog.getExistingDirectory(self, "Seleccionar Carpeta")
        if folder:
            self.list_widget.clear()
            self.rois_rotadas_guardadas = {}
            self.mascaras_automaticas_e3 = {}
            self.mascaras_manuales = {}
            self.resultados_etapa5 = {}

            for file in sorted(os.listdir(folder)):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    full_path = os.path.join(folder, file)
                    item = QListWidgetItem(file)
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    item.setCheckState(Qt.CheckState.Checked)
                    item.setData(Qt.ItemDataRole.UserRole, full_path)
                    self.list_widget.addItem(item)

    def marcar_todas(self, estado):
        check = Qt.CheckState.Checked if estado else Qt.CheckState.Unchecked
        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setCheckState(check)

    def imagen_seleccionada(self, item):
        path = item.data(Qt.ItemDataRole.UserRole)
        self.current_path = path
        self.current_rgb = cargar_imagen_rgb(path)

        img_display = np.transpose(self.current_rgb, (1, 0, 2))
        self.img_item.setImage(img_display)

        h, w = self.current_rgb.shape[:2]
        self.roi.setPos([w // 4, h // 4])
        self.roi.setSize([w // 2, h // 2])
        self.roi.setAngle(0)
        self.generar_vista_previa_rgb()

    def obtener_roi_normalizada_actual(self):
        if self.current_path is None or self.current_rgb is None:
            return None

        img_display = np.transpose(self.current_rgb, (1, 0, 2))
        arr_corte = self.roi.getArrayRegion(img_display, self.img_item)

        if arr_corte is None or arr_corte.size == 0:
            return None

        roi_recortada = np.transpose(arr_corte, (1, 0, 2))
        if roi_recortada.shape[2] != 3 and roi_recortada.shape[0] == 3:
            roi_recortada = np.moveaxis(roi_recortada, 0, -1)

        return normalizar_espacialmente(roi_recortada)

    def generar_vista_previa_rgb(self):
        roi_normalizada = self.obtener_roi_normalizada_actual()
        if roi_normalizada is None:
            return

        h, w, ch = roi_normalizada.shape
        q_img = QImage(roi_normalizada.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
        self.lbl_preview.setPixmap(QPixmap.fromImage(q_img).scaled(256, 256, Qt.AspectRatioMode.KeepAspectRatio))

    def guardar_roi_confirmada(self):
        roi_normalizada = self.obtener_roi_normalizada_actual()
        if roi_normalizada is None:
            return

        self.rois_rotadas_guardadas[self.current_path] = roi_normalizada
        # La ROI cambió: las máscaras calculadas sobre la anterior ya no valen
        self.mascaras_automaticas_e3.pop(self.current_path, None)
        self.mascaras_manuales.pop(self.current_path, None)
        self.resultados_etapa5.pop(self.current_path, None)
        guardar_imagen_roi(roi_normalizada, self.current_path)

        pos = self.roi.pos()
        size = self.roi.size()
        angulo = self.roi.angle()
        centro = (pos.x() + size.x() / 2, pos.y() + size.y() / 2)
        tamano = (size.x(), size.y())
        registrar_roi_csv(self.current_path, centro, tamano, angulo)

        item_actual = self.list_widget.currentItem()
        if item_actual:
            item_actual.setText(f"✓ {os.path.basename(self.current_path)}")
            item_actual.setForeground(Qt.GlobalColor.darkGreen)

    def obtener_seleccionadas(self):
        seleccionadas = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                seleccionadas.append(item.data(Qt.ItemDataRole.UserRole))
        return seleccionadas

    # =========================================================================
    # CONTROLADORES DE TRANSICIÓN ENTRE PANTALLAS
    # =========================================================================
    def obtener_img_norm(self, path):
        """ROI guardada si existe; si no, la imagen completa normalizada (sin recalcular de más)."""
        if path in self.rois_rotadas_guardadas:
            return self.rois_rotadas_guardadas[path]
        return normalizar_espacialmente(cargar_imagen_rgb(path))

    def limpiar_grilla(self, grid):
        for i in reversed(range(grid.count())):
            w = grid.itemAt(i).widget()
            if w:
                w.setParent(None)

    def ejecutar_e1_y_mostrar(self):
        self.imagenes_seleccionadas = self.obtener_seleccionadas()
        if not self.imagenes_seleccionadas:
            QMessageBox.warning(self, "Atención", "No hay imágenes seleccionadas en la lista.")
            return

        grid = self.grids[1]
        self.limpiar_grilla(grid)

        columnas = 2
        for idx, path in enumerate(self.imagenes_seleccionadas):
            img_norm = self.obtener_img_norm(path)
            res_e1 = procesar_etapa1(img_norm)
            guardar_visualizacion_e1(res_e1, path)

            card = self.crear_tarjeta_e1(os.path.basename(path), res_e1)
            grid.addWidget(card, idx // columnas, idx % columnas)

        self.stack.setCurrentIndex(1)

    def ejecutar_e2_y_mostrar(self):
        grid = self.grids[2]
        self.limpiar_grilla(grid)

        columnas = 2
        for idx, path in enumerate(self.imagenes_seleccionadas):
            img_norm = self.obtener_img_norm(path)
            res_e1 = procesar_etapa1(img_norm)
            res_e2 = procesar_etapa2(res_e1)
            guardar_visualizacion_e2(res_e1, res_e2, path)

            card = self.crear_tarjeta_e2(os.path.basename(path), res_e1, res_e2)
            grid.addWidget(card, idx // columnas, idx % columnas)

        self.stack.setCurrentIndex(2)

    def ejecutar_e3_y_mostrar(self):
        grid = self.grids[3]
        self.limpiar_grilla(grid)

        columnas = 2
        for idx, path in enumerate(self.imagenes_seleccionadas):
            img_norm = self.obtener_img_norm(path)
            
            res_e1 = procesar_etapa1(img_norm)
            res_e2 = procesar_etapa2(res_e1)
            res_e3 = procesar_etapa3(res_e1, res_e2)
            guardar_visualizacion_e3(res_e1, res_e3, path)

            # Guardar la propuesta automática de E3 para que el editor la tome como base
            if path not in self.mascaras_automaticas_e3:
                self.mascaras_automaticas_e3[path] = res_e3.get("mascara_lesion")

            card = self.crear_tarjeta_e3(os.path.basename(path), res_e1, res_e3, path)
            grid.addWidget(card, idx // columnas, idx % columnas)

        self.stack.setCurrentIndex(3)

    def ejecutar_e4_y_mostrar(self):
        grid = self.grids[4]
        self.limpiar_grilla(grid)

        columnas = 2
        for idx, path in enumerate(self.imagenes_seleccionadas):
            img_norm = self.obtener_img_norm(path)
            
            res_e1 = procesar_etapa1(img_norm)
            res_e2 = procesar_etapa2(res_e1)
            res_e3 = procesar_etapa3(res_e1, res_e2)
            
            # Tomar la máscara refinada en E3 si el usuario la editó; si no, la automática de E3
            mascara_final = self.mascaras_manuales.get(path, res_e3["mascara_lesion"])
            
            res_e4 = procesar_etapa4(mascara_final, res_e2)
            guardar_visualizacion_e4(res_e1, res_e4, path)
            exportar_e4(res_e1, res_e2, res_e4, path)

            card = self.crear_tarjeta_e4(os.path.basename(path), res_e1, res_e4)
            grid.addWidget(card, idx // columnas, idx % columnas)

        self.stack.setCurrentIndex(4)

    def cargar_clasificador(self):
        """Lee resultados/Etapa_5/clasificador.json (lo genera el notebook de descriptores)."""
        if self.clasificador is None:
            try:
                self.clasificador = e6_clasificacion.cargar_clasificador()
            except FileNotFoundError:
                QMessageBox.critical(
                    self, "Falta el clasificador",
                    f"No se encontró {e6_clasificacion.RUTA_CLASIFICADOR}.\n"
                    "Corré el notebook de elección de descriptores para generarlo."
                )
                return False
        return True

    def ejecutar_e5_y_mostrar(self):
        """Etapa 5: descriptores de textura (GLCM + LBP) y regla de votos (etapas/e6_clasificacion.py)."""
        if not self.cargar_clasificador():
            return

        grid = self.grids[5]
        self.limpiar_grilla(grid)

        columnas = 2
        for idx, path in enumerate(self.imagenes_seleccionadas):
            img_norm = self.obtener_img_norm(path)

            res_e1 = procesar_etapa1(img_norm)
            res_e2 = procesar_etapa2(res_e1)
            res_e3 = procesar_etapa3(res_e1, res_e2)

            mascara_final = self.mascaras_manuales.get(path, res_e3["mascara_lesion"])
            res_e4 = procesar_etapa4(mascara_final, res_e2)

            # Descriptores sobre L* sin filtrar (el mismo que guarda exportar_e4) + votos
            resultado = e6_clasificacion.clasificar(res_e1["L"], res_e4, self.clasificador)
            self.resultados_etapa5[path] = resultado

            card = self.crear_tarjeta_e5(os.path.basename(path), res_e1, res_e4, resultado)
            grid.addWidget(card, idx // columnas, idx % columnas)

        self.stack.setCurrentIndex(5)

    def ejecutar_diagnostico_y_mostrar(self):
        """Etapa 6: veredicto de la regla de votos para cada imagen + valor de cada descriptor usado."""
        reglas = self.clasificador["descriptores"] if self.clasificador else []
        encabezados = ["Imagen", "Diagnóstico (regla de votos)"] + [
            f"{r['descriptor']}\n(tiña si {'>' if r['sentido'] == 'mayor' else '<'} {r['umbral']:.4g})" for r in reglas
        ]
        self.tabla_diagnostico.setRowCount(0)
        self.tabla_diagnostico.setColumnCount(len(encabezados))
        self.tabla_diagnostico.setHorizontalHeaderLabels(encabezados)

        for row, path in enumerate(self.imagenes_seleccionadas):
            self.tabla_diagnostico.insertRow(row)
            self.tabla_diagnostico.setItem(row, 0, QTableWidgetItem(os.path.basename(path)))

            resultado = self.resultados_etapa5.get(path)
            if resultado is None:
                self.tabla_diagnostico.setItem(row, 1, QTableWidgetItem("Sin datos (falta pasar por Etapa 5)"))
                continue

            es_tina = resultado["veredicto"] == "tiña"
            diag_texto = (f"{'Tiña Corporis' if es_tina else 'No Tiña / Otra'} "
                          f"({resultado['votos_tina']}/{resultado['validos']} votos)")
            item = QTableWidgetItem(diag_texto)
            item.setForeground(Qt.GlobalColor.darkRed if es_tina else Qt.GlobalColor.darkBlue)
            self.tabla_diagnostico.setItem(row, 1, item)

            for col, d in enumerate(resultado["detalle"], start=2):
                if d["voto_tina"] is None:
                    texto = "N/D"
                else:
                    texto = f"{d['valor']:.4g}  ({'tiña' if d['voto_tina'] else 'no tiña'})"
                self.tabla_diagnostico.setItem(row, col, QTableWidgetItem(texto))

        self.stack.setCurrentIndex(6)

    # =========================================================================
    # TARJETAS DE MUESTRA PARA GRILLAS
    # =========================================================================
    def crear_tarjeta_e1(self, nombre, res_e1):
        frame = QFrame()
        frame.setStyleSheet("background: #ffffff; border-radius: 8px; border: 1px solid #dcdcdc; padding: 8px;")
        layout = QVBoxLayout(frame)

        # Encabezado con el nombre de la imagen
        lbl_titulo = QLabel(f"<b>{nombre}</b>")
        lbl_titulo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(lbl_titulo)
        
        h_layout = QHBoxLayout()

        # Datos de las 3 imágenes a mostrar
        canales = [
            (res_e1["rgb"], "Original (RGB)", None),
            (res_e1["L"], "L* (Luminosidad)", "gray"),
            (res_e1["a"], "a* (Eritema)", "magma")
        ]

        for img, texto_etiqueta, cmap in canales:
            col_box = QVBoxLayout()
            lbl_img = QLabel()
            h, w = img.shape[:2]
            
            # Preparar formato de color según el canal
            if cmap == "gray":
                img_norm = np.clip((img / 100.0) * 255, 0, 255).astype(np.uint8)
                display = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2RGB)
            elif cmap == "magma":
                vmin, vmax = np.percentile(img, 2), np.percentile(img, 98)
                a_norm = np.clip((img - vmin) / (vmax - vmin + 1e-5), 0, 1)
                display = cv2.cvtColor(cv2.applyColorMap((a_norm * 255).astype(np.uint8), cv2.COLORMAP_MAGMA), cv2.COLOR_BGR2RGB)
            else:
                display = img

            # Renderizar la imagen
            q_img = QImage(display.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
            lbl_img.setPixmap(QPixmap.fromImage(q_img).scaled(150, 150, Qt.AspectRatioMode.KeepAspectRatio))
            lbl_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
            # Crear etiqueta de texto visible debajo de la imagen
            lbl_txt = QLabel(f"<b><small>{texto_etiqueta}</small></b>")
            lbl_txt.setStyleSheet("color: #424242;")
            lbl_txt.setAlignment(Qt.AlignmentFlag.AlignCenter)

            col_box.addWidget(lbl_img)
            col_box.addWidget(lbl_txt)
            h_layout.addLayout(col_box)

        layout.addLayout(h_layout)
        return frame

    def crear_tarjeta_e2(self, nombre, res_e1, res_e2):
            frame = QFrame()
            frame.setStyleSheet("background: #ffffff; border-radius: 8px; border: 1px solid #dcdcdc; padding: 8px;")
            layout = QVBoxLayout(frame)
            
            lbl_titulo = QLabel(f"<b>{nombre}</b>")
            lbl_titulo.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(lbl_titulo)

            h_layout = QHBoxLayout()

            canales_e2 = [
                (res_e1["rgb"], "Original (RGB)", False),
                (res_e2["a_suave"], "a* Mediana", False),
                (res_e2["a_eq"], "a* Ecualizado", True),
                (res_e2["a_diff"], "a* Resta periferia", False)
            ]

            for img, texto_etiqueta, is_eq in canales_e2:
                col_box = QVBoxLayout()
                lbl_img = QLabel()
                h, w = res_e1["rgb"].shape[:2]

                if texto_etiqueta == "Original (RGB)":
                    display = img
                elif is_eq:
                    display = cv2.cvtColor(cv2.applyColorMap((img * 255).astype(np.uint8), cv2.COLORMAP_MAGMA), cv2.COLOR_BGR2RGB)
                else:
                    vmin, vmax = np.percentile(img, 2), np.percentile(img, 98)
                    norm = np.clip((img - vmin) / (vmax - vmin + 1e-5), 0, 1)
                    display = cv2.cvtColor(cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_MAGMA), cv2.COLOR_BGR2RGB)

                q_img = QImage(display.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
                lbl_img.setPixmap(QPixmap.fromImage(q_img).scaled(130, 130, Qt.AspectRatioMode.KeepAspectRatio))
                lbl_img.setAlignment(Qt.AlignmentFlag.AlignCenter)

                lbl_txt = QLabel(f"<b><small>{texto_etiqueta}</small></b>")
                lbl_txt.setStyleSheet("color: #424242;")
                lbl_txt.setAlignment(Qt.AlignmentFlag.AlignCenter)

                col_box.addWidget(lbl_img)
                col_box.addWidget(lbl_txt)
                h_layout.addLayout(col_box)

            layout.addLayout(h_layout)
            return frame

    def crear_tarjeta_e3(self, nombre, res_e1, res_e3, path_imagen):
        frame = QFrame()
        frame.setStyleSheet("background: #ffffff; border-radius: 8px; border: 1px solid #dcdcdc; padding: 8px;")
        layout = QVBoxLayout(frame)
        
        layout.addWidget(QLabel(f"<b>{nombre}</b>", alignment=Qt.AlignmentFlag.AlignCenter))
        h_layout = QHBoxLayout()

        # Si el usuario ya retocó la máscara para esta foto, mostramos esa; si no, la automática de E3
        img_mascara = self.mascaras_manuales.get(path_imagen, res_e3["mascara_lesion"])

        canales_e3 = [
            (res_e1["rgb"], "Original (RGB)", "rgb"),
            (img_mascara, "Lesión (GrabCut)", "gray")
        ]

        for img, texto_etiqueta, modo in canales_e3:
            col_box = QVBoxLayout()
            lbl_img = QLabel()
            h, w = res_e1["rgb"].shape[:2]

            if modo == "rgb":
                display = img
            elif modo == "gray":
                mask_u8 = (img.astype(np.uint8)) * 255
                display = cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2RGB)

            q_img = QImage(display.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
            lbl_img.setPixmap(QPixmap.fromImage(q_img).scaled(130, 130, Qt.AspectRatioMode.KeepAspectRatio))
            lbl_img.setAlignment(Qt.AlignmentFlag.AlignCenter)

            lbl_txt = QLabel(f"<b><small>{texto_etiqueta}</small></b>")
            lbl_txt.setStyleSheet("color: #424242;")
            lbl_txt.setAlignment(Qt.AlignmentFlag.AlignCenter)

            col_box.addWidget(lbl_img)
            col_box.addWidget(lbl_txt)
            h_layout.addLayout(col_box)

        layout.addLayout(h_layout)

        # Botón para retocar la máscara propuesta antes de pasar a la Etapa 4
        btn_editar = QPushButton("✏️ Retocar Segmentación")
        btn_editar.setStyleSheet("background-color: #f5f5f5; border: 1px solid #cccccc; border-radius: 4px; padding: 4px; font-weight: bold;")
        btn_editar.clicked.connect(lambda _, p=path_imagen: self.abrir_editor_mascara(p))
        layout.addWidget(btn_editar)

        return frame

    def abrir_editor_mascara(self, path_imagen):
        img_norm = self.obtener_img_norm(path_imagen)
        res_e1 = procesar_etapa1(img_norm)

        # 1. Si el usuario YA retocó esta máscara antes, usamos esa versión corregida
        if path_imagen in self.mascaras_manuales:
            mascara_a_editar = self.mascaras_manuales[path_imagen]
        else:
            # 2. Si no, la automática cacheada (se calcula solo si falta)
            if path_imagen not in self.mascaras_automaticas_e3:
                res_e2 = procesar_etapa2(res_e1)
                res_e3 = procesar_etapa3(res_e1, res_e2)
                self.mascaras_automaticas_e3[path_imagen] = res_e3["mascara_lesion"]
            mascara_a_editar = self.mascaras_automaticas_e3[path_imagen]

        # 3. Abrimos el editor con la máscara exacta
        dlg = EditorMascaraDialog(res_e1["rgb"], mascara_a_editar, self)
        if dlg.exec():
            # Guardamos la corrección final realizada por el usuario
            self.mascaras_manuales[path_imagen] = dlg.obtener_mascara_final()
            # Refrescamos la vista de la Pantalla 3
            self.ejecutar_e3_y_mostrar()

    def crear_tarjeta_e4(self, nombre, res_e1, res_e4):
        frame = QFrame()
        frame.setStyleSheet("background: #ffffff; border-radius: 8px; border: 1px solid #dcdcdc; padding: 8px;")
        layout = QVBoxLayout(frame)
        
        lbl_titulo = QLabel(f"<b>{nombre}</b>", alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(lbl_titulo)
        h_layout = QHBoxLayout()

        rgb = res_e1["rgb"]

        # Vistas de la Etapa 4
        img_contorno = _contorno(rgb.copy(), res_e4["lesion"])
        img_zonas = pintar_zonas(rgb, res_e4)

        canales_e4 = [
            (rgb, "Original (RGB)"),
            (img_contorno, "Lesión Delimitada"),
            (img_zonas, "Zonas (Perilesional/Borde/Centro)"),
        ]

        for display, texto_etiqueta in canales_e4:
            col_box = QVBoxLayout()
            lbl_img = QLabel()
            h, w = rgb.shape[:2]

            q_img = QImage(display.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
            lbl_img.setPixmap(QPixmap.fromImage(q_img).scaled(130, 130, Qt.AspectRatioMode.KeepAspectRatio))
            lbl_img.setAlignment(Qt.AlignmentFlag.AlignCenter)

            lbl_txt = QLabel(f"<b><small>{texto_etiqueta}</small></b>")
            lbl_txt.setStyleSheet("color: #424242;")
            lbl_txt.setAlignment(Qt.AlignmentFlag.AlignCenter)

            col_box.addWidget(lbl_img)
            col_box.addWidget(lbl_txt)
            h_layout.addLayout(col_box)

        layout.addLayout(h_layout)
        return frame

    def crear_tarjeta_e5(self, nombre, res_e1, res_e4, resultado):
        """Tarjeta de la Etapa 5: zonas usadas + descriptores elegidos con su umbral y su voto."""
        frame = QFrame()
        frame.setStyleSheet("background: #ffffff; border-radius: 8px; border: 1px solid #dcdcdc; padding: 8px;")
        layout = QVBoxLayout(frame)

        lbl_titulo = QLabel(f"<b>{nombre}</b>", alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(lbl_titulo)
        h_layout = QHBoxLayout()

        rgb = res_e1["rgb"]
        img_zonas = pintar_zonas(rgb, res_e4)

        canales_e5 = [
            (rgb, "Original (RGB)"),
            (img_zonas, "Zonas (Perilesional/Borde/Centro)"),
        ]

        for display, texto_etiqueta in canales_e5:
            col_box = QVBoxLayout()
            lbl_img = QLabel()
            h, w = rgb.shape[:2]

            q_img = QImage(display.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
            lbl_img.setPixmap(QPixmap.fromImage(q_img).scaled(150, 150, Qt.AspectRatioMode.KeepAspectRatio))
            lbl_img.setAlignment(Qt.AlignmentFlag.AlignCenter)

            lbl_txt = QLabel(f"<b><small>{texto_etiqueta}</small></b>")
            lbl_txt.setStyleSheet("color: #424242;")
            lbl_txt.setAlignment(Qt.AlignmentFlag.AlignCenter)

            col_box.addWidget(lbl_img)
            col_box.addWidget(lbl_txt)
            h_layout.addLayout(col_box)

        layout.addLayout(h_layout)

        lineas = []
        for d in resultado["detalle"]:
            signo = "&gt;" if d["sentido"] == "mayor" else "&lt;"   # escapado: el texto es HTML
            if d["voto_tina"] is None:
                voto = "N/D"
            else:
                voto = "<span style='color:#c0392b'>tiña</span>" if d["voto_tina"] else "no tiña"
            valor = "N/D" if d["voto_tina"] is None else f"{d['valor']:.4g}"
            lineas.append(f"{d['descriptor']}: <b>{valor}</b> (tiña si {signo} {d['umbral']:.4g}) → {voto}")

        lbl_datos = QLabel(
            "<small>" + "<br>".join(lineas) +
            f"<br>Votos tiña: <b>{resultado['votos_tina']}/{resultado['validos']}</b> → "
            f"<b>{resultado['veredicto']}</b></small>"
        )
        lbl_datos.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_datos.setStyleSheet("color: #212121; padding-top: 4px;")
        layout.addWidget(lbl_datos)

        return frame


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())