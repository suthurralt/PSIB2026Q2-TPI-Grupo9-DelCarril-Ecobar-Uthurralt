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

# Se genera la máscara del fondo blanco
def mascara_fondo(rgb, v_min=V_MIN_FONDO, s_max=S_MAX_FONDO,
                  area_min=AREA_MIN_FONDO, margen=MARGEN_PX):
    """Devuelve una máscara booleana con True en los píxeles de fondo blanco."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    candidatos = ((hsv[..., 2] >= v_min) & (hsv[..., 1] <= s_max)).astype(np.uint8)  # candidatos a fondo (cumplen con V alto y S bajo)

    # Cierre chico para unir el fondo cortado por ruido/artefactos JPEG
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    candidatos = cv2.morphologyEx(candidatos, cv2.MORPH_CLOSE, k3)

    # Etiquetado de componentes conexas: solo las que tocan el borde son fondo
    alto, ancho = candidatos.shape
    n, etiquetas, stats, _ = cv2.connectedComponentsWithStats(candidatos, connectivity=8)
    area_min_px = area_min * alto * ancho
    fondo = np.zeros((alto, ancho), dtype=np.uint8)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        toca_borde = x == 0 or y == 0 or x + w == ancho or y + h == alto
        if toca_borde and area >= area_min_px:
            fondo[etiquetas == i] = 1

    if margen > 0 and fondo.any():
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margen + 1, 2 * margen + 1))
        fondo = cv2.dilate(fondo, k)
    return fondo.astype(bool)


def mascara_piel(rgb, **kwargs):
    """Máscara booleana con True en los píxeles válidos (piel). Complemento del fondo."""
    return ~mascara_fondo(rgb, **kwargs)


def rgb_a_lab(rgb):
    """Convierte RGB uint8 a CIELAB en float32.

    Se convierte en punto flotante (RGB/255) para que OpenCV devuelva valores en
    unidades reales: L* en [0, 100], a* y b* aprox. en [-128, 127], con 0 = neutro.
    (Con uint8, OpenCV reescala L a 0-255 y suma 128 a a* y b*.)
    """
    return cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)


def aplicar_mascara(canal, mascara, relleno=np.nan):
    """Copia del canal con los píxeles fuera de la máscara reemplazados por `relleno`."""
    salida = canal.astype(np.float32).copy()
    salida[~mascara] = relleno
    return salida

# IMPORTANTE: devuelve la máscara y L a b sin enmascarar
def procesar_etapa1(rgb, **kwargs_mascara):
    """Aplica la etapa 1 a una imagen RGB.

    Devuelve un dict con: rgb, mascara (True = piel), L, a, b (float32, sin
    enmascarar) y frac_fondo (fracción de la imagen descartada).
    """
    mascara = mascara_piel(rgb, **kwargs_mascara)
    lab = rgb_a_lab(rgb)
    
    return {
        "rgb": rgb,
        "mascara": mascara,
        "L": lab[..., 0],
        "a": lab[..., 1],
        "b": lab[..., 2],
        "frac_fondo": float(1.0 - mascara.mean()),
    }


def mostrar_etapa1(res, titulo="", rango_a=(0, 30)):
    """Figura de control: RGB con el fondo marcado, máscara, L* y a* enmascarados."""
    rgb_marcada = res["rgb"].copy()
    rgb_marcada[~res["mascara"]] = (0, 120, 255)  # fondo descartado en azul

    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    ax[0].imshow(res["rgb"])
    ax[0].set_title(f"{titulo}\noriginal")
    ax[1].imshow(rgb_marcada)
    ax[1].set_title(f"fondo descartado: {100 * res['frac_fondo']:.1f}%")
    im_l = ax[2].imshow(aplicar_mascara(res["L"], res["mascara"]), cmap="gray", vmin=0, vmax=100)
    ax[2].set_title("L* (luminosidad)")
    fig.colorbar(im_l, ax=ax[2], fraction=0.046)
    im_a = ax[3].imshow(aplicar_mascara(res["a"], res["mascara"]), cmap="magma",
                        vmin=rango_a[0], vmax=rango_a[1])
    ax[3].set_title("a* (rojo)")
    fig.colorbar(im_a, ax=ax[3], fraction=0.046)
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Guardado / carga de resultados
# --------------------------------------------------------------------------
def _escribir_png(ruta, imagen):
    """cv2.imwrite no soporta rutas con acentos en Windows: se codifica y se escribe con numpy."""
    ok, buf = cv2.imencode(".png", imagen)
    if not ok:
        raise IOError(f"No se pudo codificar {ruta}")
    buf.tofile(str(ruta))


def guardar_etapa1(res, id_, carpeta):
    """Guarda los resultados de una imagen en `carpeta` y devuelve las rutas.

    Archivos (`id_` = nombre de la imagen original, p. ej. "TC (156)"):
      - <id>_mascara.png      máscara de piel (blanco = piel, negro = fondo)
      - <id>_enmascarada.png  imagen RGB con el fondo en negro (para mirar)
      - <id>_lab.npz          L, a, b (float32) y mascara (bool): es lo que
                              lee la etapa siguiente, sin pérdida de precisión
    """
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    rutas = {
        "mascara": carpeta / f"{id_}_mascara.png",
        "enmascarada": carpeta / f"{id_}_enmascarada.png",
        "lab": carpeta / f"{id_}_lab.npz",
    }
    _escribir_png(rutas["mascara"], res["mascara"].astype(np.uint8) * 255)
    enmascarada = res["rgb"].copy()
    enmascarada[~res["mascara"]] = 0
    _escribir_png(rutas["enmascarada"], cv2.cvtColor(enmascarada, cv2.COLOR_RGB2BGR))
    np.savez_compressed(rutas["lab"], L=res["L"], a=res["a"], b=res["b"], mascara=res["mascara"])
    return rutas


def cargar_etapa1(ruta_npz):
    """Lee un <id>_lab.npz guardado por guardar_etapa1. Devuelve dict con L, a, b y mascara."""
    with np.load(ruta_npz) as d:
        return {k: d[k] for k in ("L", "a", "b", "mascara")}
