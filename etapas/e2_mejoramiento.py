"""
Etapa 2 — Mejoramiento.

Entrada: L*, a* y la máscara de piel de la etapa 1.
Salida: Canales suavizados, ecualizados y con resta de piel periférica.
"""
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter

K_MEDIANA = 7


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
    """Aplica el mejoramiento a la salida de la etapa 1 (dict con L, a, mascara)."""
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