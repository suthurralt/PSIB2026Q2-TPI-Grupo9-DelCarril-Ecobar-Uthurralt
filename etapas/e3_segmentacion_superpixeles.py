"""
Módulo: e3_segmentacion_superpixeles.py
=======================================
Etapa 3: Segmentación de la lesión por graph cut (GrabCut) en Lab.

Método semiautomático: la ROI recortada a mano en el paso 0 está centrada en la lesión,
y GrabCut se inicializa con un prior geométrico sobre esa ROI:
  - centro (30% del radio)         -> lesión segura
  - elipse (80% del radio)         -> probable lesión
  - esquinas y marco de 3%         -> fondo seguro
  - resto                          -> probable fondo
El color (Lab) decide el contorno dentro de la franja "probable".
"""

from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes


def _grabcut_lesion(rgb, iters=5):
    h, w = rgb.shape[:2]
    yy, xx = np.mgrid[:h, :w]
    d = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)

    m = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    m[d < 0.80] = cv2.GC_PR_FGD
    m[d < 0.30] = cv2.GC_FGD
    m[d > 1.05] = cv2.GC_BGD            
    b = int(0.03 * min(h, w))
    m[:b, :] = m[-b:, :] = m[:, :b] = m[:, -b:] = cv2.GC_BGD

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(lab, m, None, bgd, fgd, iters, cv2.GC_INIT_WITH_MASK)
    return (m == cv2.GC_FGD) | (m == cv2.GC_PR_FGD)


def procesar_etapa3(e1, e2):
    rgb = e1["rgb"]
    mascara_piel = e2["mascara"]

    res = _grabcut_lesion(rgb) & mascara_piel
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    res = cv2.morphologyEx(res.astype(np.uint8), cv2.MORPH_OPEN, k)

    n, lab_cc, st, _ = cv2.connectedComponentsWithStats(res)
    if n > 1:
        res = lab_cc == (np.argmax(st[1:, cv2.CC_STAT_AREA]) + 1)

    return {"mascara_lesion": binary_fill_holes(res.astype(bool))}


def guardar_visualizacion_e3(e1, e3, nombre_archivo, carpeta_salida="resultados/Etapa_3"):
    carpeta = Path(carpeta_salida)
    carpeta.mkdir(parents=True, exist_ok=True)

    rgb = e1["rgb"]
    mascara = e3["mascara_lesion"]

    contorno = rgb.copy()
    cs, _ = cv2.findContours(mascara.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(contorno, cs, -1, (255, 255, 255), 2)

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(rgb)
    ax[0].set_title("ROI Original (512x512)")
    ax[1].imshow(mascara, cmap="gray")
    ax[1].set_title("Lesión (GrabCut)")
    ax[2].imshow(contorno)
    ax[2].set_title("Contorno sobre la ROI")
    for a in ax:
        a.axis("off")

    fig.tight_layout()
    ruta_salida = carpeta / f"e3_{Path(nombre_archivo).stem}.png"
    fig.savefig(ruta_salida, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return ruta_salida