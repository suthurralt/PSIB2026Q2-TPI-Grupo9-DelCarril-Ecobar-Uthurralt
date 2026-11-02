"""Etapa 1 — Enmascarado del fondo y conversión RGB -> CIELAB.

Problema: varias imágenes de SkinDisNet tienen el fondo recortado y reemplazado
por blanco. Esos píxeles no son piel y, si se incluyen, distorsionan la
umbralización.

Criterio para el fondo:
  1. Candidatos: píxeles muy claros (V alto) y casi sin color (S baja) en HSV.
  2. Solo se considera fondo una región candidata que TOCA EL BORDE de la imagen
     y tiene un área mínima. Esto evita borrar la descamación blanquecina o los
     reflejos que están dentro de la lesión, que también son claros y poco
     saturados pero quedan rodeados de piel.
  3. Se agranda el fondo unos píxeles (dilatación) para descartar la franja de
     transición piel/blanco, donde el JPEG mezcla ambos colores.

Conversión a Lab: L* = luminosidad (0-100), a* = eje verde(-)/rojo(+),
b* = eje azul(-)/amarillo(+). El eritema se mide en a*, separado del brillo.
"""
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Parámetros fijos (no se aprenden de los datos; la máscara se calcula imagen por imagen)
V_MIN_FONDO = 235          # brillo mínimo (0-255) para ser candidato a fondo
S_MAX_FONDO = 25           # saturación máxima (0-255) para ser candidato a fondo
AREA_MIN_FONDO = 0.005     # fracción mínima de la imagen para aceptar una región como fondo
MARGEN_PX = 5              # píxeles que se descartan en el borde piel/fondo


def rgb_a_lab(rgb):
    """Convierte RGB uint8 a CIELAB en float32.

    Se convierte en punto flotante (RGB/255) para que OpenCV devuelva valores en
    unidades reales: L* en [0, 100], a* y b* aprox. en [-128, 127], con 0 = neutro.
    (Con uint8, OpenCV reescala L a 0-255 y suma 128 a a* y b*.)
    """
    return cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)


def procesar_etapa1(rgb):
    """
    Procesa la ROI (512x512) en memoria RAM.
    
    Devuelve un diccionario con:
      - 'rgb': imagen original
      - 'mascara': matriz booleana (True en toda la ROI)
      - 'L': canal L* (luminosidad/textura)
      - 'a': canal a* (eritema)
      - 'b': canal b* (amarillo/azul)
    """
    lab = rgb_a_lab(rgb)
    alto, ancho = rgb.shape[:2]

    # Al no haber fondo blanco artificial, toda la ROI recortada se considera válida
    mascara_valida = np.ones((alto, ancho), dtype=bool)

    return {
        "rgb": rgb,
        "mascara": mascara_valida,
        "L": lab[..., 0],
        "a": lab[..., 1],
        "b": lab[..., 2]
    }


def guardar_visualizacion_e1(res_e1, nombre_archivo, carpeta_salida="resultados/Etapa_1"):
    """
    Genera y guarda en disco una imagen comparativa (Original | L* Textura | a* Eritema)
    en la carpeta unificada 'resultados/Etapa_1/'.
    """
    carpeta = Path(carpeta_salida)
    carpeta.mkdir(parents=True, exist_ok=True)
    
    # Rango adaptativo de a* por percentiles
    valores_a = res_e1["a"]
    vmin_a = np.percentile(valores_a, 2)
    vmax_a = np.percentile(valores_a, 98)

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    
    # 1. ROI RGB Original
    ax[0].imshow(res_e1["rgb"])
    ax[0].set_title("ROI Original (512x512)")
    ax[0].axis("off")

    # 2. Canal L* (Luminosidad / Textura)
    im_l = ax[1].imshow(res_e1["L"], cmap="gray", vmin=0, vmax=100)
    ax[1].set_title("Canal L* (Luminosidad)")
    ax[1].axis("off")
    fig.colorbar(im_l, ax=ax[1], fraction=0.046, pad=0.04)

    # 3. Canal a* (Eritema / Inflamación)
    im_a = ax[2].imshow(res_e1["a"], cmap="magma", vmin=vmin_a, vmax=vmax_a)
    ax[2].set_title("Canal a* (Eritema)")
    ax[2].axis("off")
    fig.colorbar(im_a, ax=ax[2], fraction=0.046, pad=0.04)

    fig.tight_layout()
    
    # Guarda en resultados/Etapa_1/e1_nombreImagen.png
    nombre_base = Path(nombre_archivo).stem
    ruta_salida = carpeta / f"e1_{nombre_base}.png"
    fig.savefig(ruta_salida, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return ruta_salida


if __name__ == "__main__":
    from etapas.e0_normalizacion_roi import cargar_imagen_rgb

    carpeta_rois = Path("datos/ROIs")
    archivos_roi = sorted(list(carpeta_rois.glob("*.jpg")) + list(carpeta_rois.glob("*.png")))

    if not archivos_roi:
        print("No hay imágenes en datos/ROIs/. Guarda algunas desde la GUI primero.")
    else:
        print(f"Procesando {len(archivos_roi)} ROIs -> Guardando resultados en resultados/Etapa_1/\n")
        for ruta in archivos_roi:
            img_rgb = cargar_imagen_rgb(ruta)
            res_e1 = procesar_etapa1(img_rgb)
            guardar_visualizacion_e1(res_e1, ruta.name)
        print("\n¡Listo! Revisa la carpeta resultados/Etapa_1/.")