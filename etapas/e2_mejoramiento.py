"""
Etapa 2: Mejoramiento.
 
Entrada: diccionario de la Etapa 1 (canales L* y a* de la ROI de 512x512 y la
máscara válida, que en la práctica es True en toda la ROI).
 
Qué hace:
  1. Filtro de mediana de 7x7 sobre a* y sobre L* (elimina ruido impulsivo sin
     difuminar los bordes de la lesión).
  2. Ecualización del histograma del a* filtrado, calculada solo con los
     píxeles de la máscara. Resultado en [0, 1]. Sirve para visualizar y para
     los métodos de segmentación por umbral.
  3. Eritema relativo a la piel periférica: a*_diff = max(0, a*_filtrado -
     mediana de a*_filtrado en un marco de 30 px en el perímetro de la ROI).
     Es el exceso de rojo respecto de la piel de esa misma foto.
  4. L* se conserva sin filtrar ('L') para el análisis de textura de la
     Etapa 5, porque la mediana borra las escamas y pápulas que se quieren
     medir. También se devuelve L* filtrado ('L_suave').
 
Salida: diccionario con 'mascara', 'a_suave', 'a_eq', 'a_diff', 'L' (sin
filtrar) y 'L_suave'.
"""
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter

K_MEDIANA = 7

# Filtro de mediana k x k sobre un canal (L* o a*). Los píxeles fuera de la
# máscara se reemplazan por la mediana de los píxeles válidos antes de filtrar,
# para no mezclar fondo con piel. Como la máscara de la Etapa 1 es True en toda
# la ROI, en la práctica no se reemplaza ningún píxel.
def mediana_con_mascara(canal, mascara, k=K_MEDIANA):
    """Filtro de mediana k x k que no mezcla el fondo con la piel."""
    canal = canal.astype(np.float32)
    relleno = np.where(mascara, canal, np.median(canal[mascara]))
    filtrado = median_filter(relleno, size=k, mode="reflect")
    return filtrado, mascara


def ecualizar_con_mascara(canal, mascara, n_niveles=256):
    """Ecualización del histograma calculada SOLO con los píxeles de piel."""
    valores = canal[mascara]
    vmin, vmax = float(valores.min()), float(valores.max())
    if vmax <= vmin:
        return np.zeros_like(canal, dtype=np.float32)

    niveles = np.clip(((canal - vmin) / (vmax - vmin) * (n_niveles - 1)).round(), 0, n_niveles - 1).astype(int)
    hist = np.bincount(niveles[mascara], minlength=n_niveles)
    cdf = np.cumsum(hist) / hist.sum()
    salida = cdf[niveles].astype(np.float32)
    salida[~mascara] = 0.0
    return salida


def obtener_mascara_periferia(shape, ancho_borde_px=30):
    """Obtiene el marco exterior de la ROI para tomar el tono de piel sana."""
    h, w = shape[:2]
    mascara = np.zeros((h, w), dtype=bool)
    mascara[:ancho_borde_px, :] = True
    mascara[-ancho_borde_px:, :] = True
    mascara[:, :ancho_borde_px] = True
    mascara[:, -ancho_borde_px:] = True
    return mascara


def procesar_etapa2(e1, k=K_MEDIANA):
    """Aplica el mejoramiento a la salida de la Etapa 1 (usa 'L', 'a' y 'mascara').
 
    Devuelve un diccionario con:
      - 'mascara': máscara válida (la misma de la Etapa 1)
      - 'a_suave': a* con filtro de mediana
      - 'a_eq': a* filtrado y ecualizado, en [0, 1]
      - 'a_diff': eritema relativo = max(0, a_suave - mediana de a_suave en la periferia)
      - 'L': L* original sin filtrar (para la textura de la Etapa 5)
      - 'L_suave': L* con filtro de mediana
    """
    a_suave, mascara = mediana_con_mascara(e1["a"], e1["mascara"], k)
    L_suave, _ = mediana_con_mascara(e1["L"], e1["mascara"], k)
    a_eq = ecualizar_con_mascara(a_suave, mascara)

    # Resta de piel sana periférica sobre a*
    mascara_periferia = obtener_mascara_periferia(e1["a"].shape) & mascara
    mediana_a_periferia = np.median(a_suave[mascara_periferia])
    a_diff = np.maximum(0, a_suave - mediana_a_periferia)

    return {
        "mascara": mascara,
        "a_suave": a_suave,
        "a_eq": a_eq,
        "a_diff": a_diff,
        "L": e1["L"].astype(np.float32),  # L* original sin filtrar (para textura posterior en Etapa 5)
        "L_suave": L_suave,
    }


def guardar_visualizacion_e2(e1, e2, nombre_archivo, carpeta_salida="resultados/Etapa_2"):
    """Guarda en resultados/Etapa_2/ la figura de control del mejoramiento."""
    carpeta = Path(carpeta_salida)
    carpeta.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 4, figsize=(16, 4))

    ax[0].imshow(e1["rgb"])
    ax[0].set_title("ROI Original (512x512)")
    ax[0].axis("off")

    ax[1].imshow(e2["a_suave"], cmap="magma")
    ax[1].set_title("a* Mediana (Filtrado)")
    ax[1].axis("off")

    ax[2].imshow(e2["a_eq"], cmap="magma")
    ax[2].set_title("a* Ecualizado (Piel)")
    ax[2].axis("off")

    ax[3].imshow(e2["a_diff"], cmap="magma")
    ax[3].set_title("a* - Piel Periférica")
    ax[3].axis("off")

    fig.tight_layout()

    nombre_base = Path(nombre_archivo).stem
    ruta_salida = carpeta / f"e2_{nombre_base}.png"
    fig.savefig(ruta_salida, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return ruta_salida