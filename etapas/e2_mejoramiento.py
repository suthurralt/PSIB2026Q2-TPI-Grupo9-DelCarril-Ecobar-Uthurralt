"""Etapa 2 — Mejoramiento.

Entrada: L*, a* y la máscara de piel de la etapa 1.
Salida: versiones de los canales pensadas para cada uso posterior.

    canal       cómo se obtiene                      para qué se usa
    ---------   ----------------------------------   -------------------------------------------
    a_suave     mediana sobre a*                      segmentación y medidas de color (etapas 3 y 5)
    a_eq        ecualización de a_suave (con máscara) SOLO segmentación/visualización: sus valores
                                                      son relativos a cada imagen, no se comparan
    L           L* original, sin tocar                textura (GLCM, etapa 5)
    L_suave     mediana sobre L*                      segmentación (pelos y pliegues finos borrados)

Decisiones (probadas sobre Entrenamiento):
  * Mediana en a*: el pelo casi no aparece en a* (es oscuro, no rojo); lo que sí hay
    es el patrón en bloques del JPEG, que guarda el color a menor resolución que el
    brillo. La mediana lo suaviza y conserva los bordes de la lesión (un gaussiano
    los desdibujaría).
  * L* sin filtrar para textura: la mediana borra estructuras más finas que el
    kernel, y eso incluye la descamación que la GLCM tiene que medir. Una prueba
    con black-hat para detectar pelos marcaba también escamas y pliegues
    (hasta 24% de la imagen en CD 56), así que no se usa.
  * Ecualizar a* y no L*: ver ecualizar_con_mascara().
"""
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter

K_MEDIANA = 7   # lado del kernel (px). Borra líneas de ~3 px (pelos finos) y conserva el anillo de TC (>10 px)


def mediana_con_mascara(canal, mascara, k=K_MEDIANA):
    """Filtro de mediana k x k que no mezcla el fondo con la piel.

    Antes de filtrar, el fondo se rellena con la mediana de la piel (si no, el
    blanco del fondo "entraría" en los píxeles de piel cercanos). Devuelve el canal
    filtrado y la máscara erosionada k//2 px, porque en esa franja del borde la
    ventana todavía toca el relleno.
    """
    canal = canal.astype(np.float32)
    relleno = np.where(mascara, canal, np.median(canal[mascara]))
    filtrado = median_filter(relleno, size=k, mode="reflect")
    if (~mascara).any():
        ee = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * (k // 2) + 1,) * 2)
        mascara = cv2.erode(mascara.astype(np.uint8), ee).astype(bool)
    return filtrado, mascara


def ecualizar_con_mascara(canal, mascara, n_niveles=256):
    """Ecualización del histograma calculada SOLO con los píxeles de piel.

    Pasos (los de la clase): histograma -> normalizar -> histograma acumulado (CDF)
    -> cada píxel se reemplaza por el valor de la CDF en su nivel. Resultado en [0, 1];
    el fondo queda en 0.

    Por qué con máscara: con cv2.equalizeHist el fondo blanco (hasta 36% de la
    imagen, todo con a* ~ 0) ocuparía buena parte del histograma y se
    "llevaría" el contraste.

    Por qué a* sí: a* es lo que separa lesión de piel, pero varía en un rango
    angosto: en Entrenamiento, dentro de cada imagen el 98% de la piel ocupa
    ~12 unidades (mediana), de un rango posible de ±128, y la diferencia
    lesión/piel es de pocas unidades. Ecualizar estira ese rango y la lesión se
    separa mejor de la piel sana.

    Por qué L* no: en L* las diferencias grandes son iluminación, sombras, tono de
    piel y escamas, no "lesión vs. piel". Ecualizar L* amplifica sombras y ruido
    JPEG y cambia las relaciones de gris que después mide la GLCM.

    OJO: la ecualización es por imagen, así que un mismo valor ecualizado no
    significa lo mismo en dos imágenes. Sirve para segmentar o mirar, nunca para
    medir características que se comparan entre imágenes.
    """
    valores = canal[mascara]
    vmin, vmax = float(valores.min()), float(valores.max())
    if vmax <= vmin:
        return np.zeros_like(canal, dtype=np.float32)
    # 1. Cuantizar a n_niveles y calcular el histograma (solo piel)
    niveles = np.clip(((canal - vmin) / (vmax - vmin) * (n_niveles - 1)).round(), 0, n_niveles - 1).astype(int)
    hist = np.bincount(niveles[mascara], minlength=n_niveles)
    # 2-3. Normalizar y acumular
    cdf = np.cumsum(hist) / hist.sum()
    # 4. Mapear cada nivel a su valor de CDF
    salida = cdf[niveles].astype(np.float32)
    salida[~mascara] = 0.0
    return salida


def procesar_etapa2(e1, k=K_MEDIANA):
    """Aplica la etapa 2 a la salida de la etapa 1 (dict con L, a, mascara)."""
    a_suave, mascara = mediana_con_mascara(e1["a"], e1["mascara"], k)
    L_suave, _ = mediana_con_mascara(e1["L"], e1["mascara"], k)
    return {
        "mascara": mascara,                      # erosionada k//2 en el borde con el fondo
        "a_suave": a_suave,
        "a_eq": ecualizar_con_mascara(a_suave, mascara),
        "L": e1["L"].astype(np.float32),         # sin filtrar, para textura
        "L_suave": L_suave,
    }


# --------------------------------------------------------------------------
# Visualización
# --------------------------------------------------------------------------
def _en_mascara(x, m):
    return np.where(m, x, np.nan)


def mostrar_etapa2(e1, e2, titulo="", rango_a=(0, 30)):
    """a* original / mediana / ecualizado, y L* original / suavizado."""
    m = e2["mascara"]
    paneles = [
        (e1["rgb"] if "rgb" in e1 else None, "original", None, None),
        (_en_mascara(e1["a"], m), "a* original", "magma", rango_a),
        (_en_mascara(e2["a_suave"], m), f"a* mediana {K_MEDIANA}x{K_MEDIANA}", "magma", rango_a),
        (_en_mascara(e2["a_eq"], m), "a* ecualizado (solo segmentar)", "magma", (0, 1)),
        (_en_mascara(e2["L"], m), "L* sin filtrar (textura)", "gray", (0, 100)),
        (_en_mascara(e2["L_suave"], m), f"L* mediana {K_MEDIANA}x{K_MEDIANA}", "gray", (0, 100)),
    ]
    paneles = [p for p in paneles if p[0] is not None]
    fig, ax = plt.subplots(1, len(paneles), figsize=(3.6 * len(paneles), 3.8))
    for a, (img, t, cmap, rango) in zip(ax, paneles):
        if cmap is None:
            a.imshow(img)
        else:
            a.imshow(img, cmap=cmap, vmin=rango[0], vmax=rango[1])
        a.set_title(t, fontsize=10)
        a.axis("off")
    ax[0].set_title(f"{titulo}\n{ax[0].get_title()}", fontsize=10)
    fig.tight_layout()
    return fig


def mostrar_efecto_mascara_en_ecualizacion(a, mascara, titulo=""):
    """Compara ecualizar a* con cv2.equalizeHist (toda la imagen, fondo incluido) vs. solo con piel.

    `a` es el a* de la etapa 1 (sin filtrar), donde el fondo blanco vale a* ~ 0.
    """
    a8 = np.clip((a + 10) / 40 * 255, 0, 255).astype(np.uint8)  # a* en [-10, 30] -> [0, 255]
    sin_mascara = cv2.equalizeHist(a8) / 255.0
    con_mascara = ecualizar_con_mascara(a, mascara)

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    bins = np.linspace(-5, 30, 71)
    ax[0].hist(a.ravel(), bins=bins, color="0.6", label="toda la imagen")
    ax[0].hist(a[mascara], bins=bins, color="tab:red", alpha=0.7, label="solo piel")
    ax[0].set_title(f"{titulo}\nhistograma de a*")
    ax[0].set_xlabel("a*")
    ax[0].legend()
    ax[1].imshow(_en_mascara(sin_mascara, mascara), cmap="magma", vmin=0, vmax=1)
    ax[1].set_title("ecualizado con el fondo")
    ax[2].imshow(_en_mascara(con_mascara, mascara), cmap="magma", vmin=0, vmax=1)
    ax[2].set_title("ecualizado solo con piel")
    for x in ax[1:]:
        x.axis("off")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Guardado / carga
# --------------------------------------------------------------------------
def guardar_etapa2(e2, id_, carpeta):
    """Guarda <id>_e2.npz (todos los canales + máscara) y <id>_a_eq.png (para mirar)."""
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta_npz = carpeta / f"{id_}_e2.npz"
    np.savez_compressed(ruta_npz, **e2)
    ok, buf = cv2.imencode(".png", (e2["a_eq"] * 255).astype(np.uint8))
    buf.tofile(str(carpeta / f"{id_}_a_eq.png"))
    return ruta_npz


def cargar_etapa2(ruta_npz):
    with np.load(ruta_npz) as d:
        return {k: d[k] for k in d.files}
