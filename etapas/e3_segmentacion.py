"""Etapa 3 — Segmentación de la lesión.

Entrada: salida de la etapa 2 (a_suave, a_eq, L_suave, mascara).
Salida: máscara binaria de la lesión (True = lesión), INCLUYENDO su interior.

Todos los métodos trabajan sobre el eritema (a*), porque la lesión se define por ser
más roja que la piel sana. Cada método da una máscara "cruda" que después pasa por
el mismo posprocesamiento (morfología + etiquetado), para compararlos en igualdad
de condiciones.

Importante para TC: en la tiña el centro suele ser casi del color de la piel, así
que un umbral sobre a* encuentra solo el ANILLO. El posprocesamiento rellena los
huecos para recuperar la lesión entera (anillo + centro), que es lo que necesita la
etapa 4 para comparar borde vs. centro.

Conclusión de la comparación (sobre Entrenamiento): el histograma de a* tiene un
solo pico (la piel) y la lesión es una cola, por lo que los umbrales globales
(Otsu, Li, Triangle), k-means y Chan-Vese parten la piel sana al medio. El método
elegido es el crecimiento de región con semillas definidas respecto de la piel de
cada imagen (mediana + k·MAD).
"""
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import filters, segmentation


# --------------------------------------------------------------------------
# Métodos (cada uno devuelve una máscara booleana cruda dentro de la piel)
# --------------------------------------------------------------------------
def _umbral_global(canal, mascara, funcion):
    t = funcion(canal[mascara])
    return (canal > t) & mascara


def otsu(e2, canal="a_suave"):
    """Otsu: el umbral que maximiza la varianza ENTRE clases (lesión vs. piel)."""
    return _umbral_global(e2[canal], e2["mascara"], filters.threshold_otsu)


def li(e2, canal="a_suave"):
    """Li: el umbral que minimiza la entropía cruzada entre la imagen y su versión binaria."""
    return _umbral_global(e2[canal], e2["mascara"], filters.threshold_li)


def triangle(e2, canal="a_suave"):
    """Triangle: pensado para histogramas con un pico grande (la piel) y una cola (la lesión)."""
    return _umbral_global(e2[canal], e2["mascara"], filters.threshold_triangle)


def adaptativo(e2, canal="a_suave", bloque=101, offset=-1.0):
    """Umbral local: cada píxel se compara con la media de su vecindario (bloque x bloque).

    offset < 0 exige que el píxel sea al menos |offset| unidades de a* más rojo que su entorno.
    """
    a = np.where(e2["mascara"], e2[canal], np.median(e2[canal][e2["mascara"]]))
    t = filters.threshold_local(a, block_size=bloque, offset=offset)
    return (a > t) & e2["mascara"]


def kmeans(e2, k=2):
    """k-means sobre (a*, L*) suavizados y estandarizados; la lesión es el grupo más rojo.

    Con k=2 y un solo canal sería equivalente a Otsu; acá se agrega L* para que el
    brillo también participe.
    """
    m = e2["mascara"]
    X = np.stack([e2["a_suave"][m], e2["L_suave"][m]], axis=1).astype(np.float32)
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    criterio = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3)
    cv2.setRNGSeed(0)
    _, etiquetas, centros = cv2.kmeans(X, k, None, criterio, 5, cv2.KMEANS_PP_CENTERS)
    grupo_lesion = int(np.argmax(centros[:, 0]))        # columna 0 = a*
    salida = np.zeros_like(m)
    salida[m] = etiquetas.ravel() == grupo_lesion
    return salida


def _estadisticas_piel(a, m):
    """Mediana y dispersión robusta (MAD escalada) de a* en la piel de la imagen.

    Como en casi todas las fotos la lesión ocupa una parte menor de la piel, la
    mediana representa a la piel sana. La MAD, a diferencia del desvío estándar,
    casi no se ve afectada por los píxeles de la lesión.
    """
    v = a[m]
    med = float(np.median(v))
    mad = 1.4826 * float(np.median(np.abs(v - med)))
    return med, max(mad, 1e-3)


K_SEMILLA = 2.5     # semillas: a* > mediana + 2.5 MAD (claramente más rojo que la piel)
K_CRECIMIENTO = 1.5 # crecimiento: se agregan vecinos con a* > mediana + 1.5 MAD


def umbral_piel(e2, k=K_SEMILLA, canal="a_suave"):
    """Umbral 'manual' definido estadísticamente: lesión = a* atípicamente alto para esa piel."""
    a, m = e2[canal], e2["mascara"]
    med, mad = _estadisticas_piel(a, m)
    return (a > med + k * mad) & m


def region_growing(e2, k_semilla=K_SEMILLA, k_crecimiento=K_CRECIMIENTO, canal="a_suave"):
    """Crecimiento de región con semillas.

    - Semillas: píxeles claramente más rojos que la piel (a* > mediana + k_semilla MAD).
    - Criterio de similitud: un vecino (8-vecindad) se agrega a la región si su a*
      supera mediana + k_crecimiento MAD.
    - La región crece desde cada semilla hasta que no quedan vecinos que cumplan.

    Equivale a una umbralización con histéresis: un píxel moderadamente rojo solo
    entra si está conectado a una semilla.
    """
    a, m = e2[canal], e2["mascara"]
    med, mad = _estadisticas_piel(a, m)
    semillas = (a > med + k_semilla * mad) & m
    candidatos = (a > med + k_crecimiento * mad) & m
    etiquetas, _ = ndi.label(candidatos, structure=np.ones((3, 3)))
    con_semilla = np.unique(etiquetas[semillas])
    return np.isin(etiquetas, con_semilla[con_semilla > 0])


def contorno_activo(e2, canal="a_eq", iteraciones=60):
    """Contorno activo sin bordes (Chan-Vese morfológico).

    Ajusta iterativamente una curva para que adentro y afuera queden lo más
    homogéneos posible, con un término que mantiene el contorno suave.
    """
    img = np.where(e2["mascara"], e2[canal], 0).astype(float)
    ls = segmentation.morphological_chan_vese(img, num_iter=iteraciones,
                                              init_level_set="checkerboard", smoothing=3)
    ls = ls.astype(bool)
    # quedarse con la fase más roja
    m = e2["mascara"]
    if img[ls & m].mean() < img[~ls & m].mean():
        ls = ~ls
    return ls & m


METODO_ELEGIDO = "Region growing (semillas)"

METODOS = {
    "Otsu (a* suave)": otsu,
    "Otsu (a* ecualizado)": lambda e2: otsu(e2, "a_eq"),
    "Li": li,
    "Triangle": triangle,
    "Adaptativo": adaptativo,
    "k-means (a*, L*)": kmeans,
    "Umbral piel + k·MAD": umbral_piel,
    "Region growing (semillas)": region_growing,
    "Contorno activo": contorno_activo,
}


# --------------------------------------------------------------------------
# Posprocesamiento común: morfología + etiquetado
# --------------------------------------------------------------------------
R_APERTURA = 3      # px: borra puntitos sueltos
R_CIERRE = 15       # px: une tramos cercanos del anillo antes de rellenar
FRAC_MIN_COMPONENTE = 0.2   # se conservan las componentes con >= 20% del área de la mayor
AREA_MIN_PIEL = 0.005       # ... y con al menos 0,5% de la piel


def _disco(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def rellenar_huecos(binaria, mascara):
    """Rellena huecos interiores. El fondo blanco cuenta como 'afuera': un hueco que
    toca el fondo recortado no es interior de la lesión."""
    afuera = ~binaria
    etiquetas, _ = ndi.label(afuera)
    borde = np.zeros_like(binaria)
    borde[[0, -1], :] = True
    borde[:, [0, -1]] = True
    abiertas = set(np.unique(etiquetas[(borde | ~mascara) & afuera])) - {0}
    huecos = afuera & ~np.isin(etiquetas, list(abiertas))
    return binaria | huecos


def posprocesar(cruda, mascara):
    """apertura -> cierre -> rellenar huecos -> etiquetado (se descartan componentes chicas)."""
    b = cruda.astype(np.uint8)
    b = cv2.morphologyEx(b, cv2.MORPH_OPEN, _disco(R_APERTURA))
    b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, _disco(R_CIERRE))
    b = b.astype(bool) & mascara
    b = rellenar_huecos(b, mascara)
    etiquetas, n = ndi.label(b, structure=np.ones((3, 3)))
    if n == 0:
        return b
    areas = ndi.sum(b, etiquetas, index=np.arange(1, n + 1))
    conservar = np.flatnonzero((areas >= FRAC_MIN_COMPONENTE * areas.max())
                               & (areas >= AREA_MIN_PIEL * mascara.sum())) + 1
    return np.isin(etiquetas, conservar)


def segmentar(e2, metodo=METODO_ELEGIDO):
    """Aplica un método y el posprocesamiento. Devuelve (cruda, final)."""
    cruda = METODOS[metodo](e2)
    return cruda, posprocesar(cruda, e2["mascara"])


# --------------------------------------------------------------------------
# Medidas para comparar métodos (no hay segmentación "verdadera" de referencia)
# --------------------------------------------------------------------------
def describir(final, e2):
    """Fracción de piel segmentada y contraste de a* lesión vs. resto de la piel."""
    m = e2["mascara"]
    frac = final[m].mean()
    dentro, fuera = e2["a_suave"][final & m], e2["a_suave"][~final & m]
    if len(dentro) < 50 or len(fuera) < 50:
        return {"frac_lesion": frac, "contraste_a": np.nan}
    disp = np.sqrt((dentro.var() + fuera.var()) / 2) + 1e-6
    return {"frac_lesion": frac, "contraste_a": (np.median(dentro) - np.median(fuera)) / disp}


# --------------------------------------------------------------------------
# Visualización
# --------------------------------------------------------------------------
def superponer(rgb, mascara_piel, lesion, color=(0, 255, 0)):
    """RGB con el contorno de la lesión dibujado y el fondo descartado en gris."""
    img = rgb.copy()
    img[~mascara_piel] = (img[~mascara_piel] * 0.3 + 120).astype(np.uint8)
    contornos, _ = cv2.findContours(lesion.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, contornos, -1, color, 3)
    return img


def mostrar_histogramas(ids, resultados_e2, canal="a_suave"):
    """Histograma de a* (solo piel) con los umbrales de Otsu, Li, Triangle y mediana + k·MAD."""
    fig, ax = plt.subplots(1, len(ids), figsize=(4.5 * len(ids), 3.4))
    for x, id_ in zip(np.atleast_1d(ax), ids):
        a, m = resultados_e2[id_][canal], resultados_e2[id_]["mascara"]
        v = a[m]
        med, mad = _estadisticas_piel(a, m)
        x.hist(v, bins=100, color="0.6")
        umbrales = {"Otsu": filters.threshold_otsu(v), "Li": filters.threshold_li(v),
                    "Triangle": filters.threshold_triangle(v),
                    f"mediana + {K_SEMILLA}·MAD": med + K_SEMILLA * mad}
        for (nombre, t), c in zip(umbrales.items(), ["tab:blue", "tab:orange", "tab:green", "tab:red"]):
            x.axvline(t, color=c, lw=2, label=f"{nombre}: {t:.1f}")
        x.set_title(id_)
        x.set_xlabel("a*")
        x.legend(fontsize=8)
    fig.tight_layout()
    return fig


def mostrar_metodo(nombre, ejemplos, resultados_e1, resultados_e2):
    """Para cada imagen de ejemplo: cruda (lo que da el método) y final (con posprocesamiento)."""
    fig, ax = plt.subplots(2, len(ejemplos), figsize=(3.6 * len(ejemplos), 7.4))
    for j, id_ in enumerate(ejemplos):
        e2 = resultados_e2[id_]
        cruda, final = segmentar(e2, nombre)
        rgb_cruda = resultados_e1[id_]["rgb"].copy()
        rgb_cruda[cruda] = (0.5 * rgb_cruda[cruda] + [0, 127, 0]).astype(np.uint8)
        ax[0, j].imshow(rgb_cruda)
        ax[0, j].set_title(f"{id_}\ncruda", fontsize=10)
        ax[1, j].imshow(superponer(resultados_e1[id_]["rgb"], e2["mascara"], final))
        ax[1, j].set_title(f"final ({100 * final[e2['mascara']].mean():.0f}% de la piel)", fontsize=10)
    for a in ax.flat:
        a.axis("off")
    fig.suptitle(nombre, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def grilla_segmentaciones(ids, lesiones, resultados_e1, resultados_e2, etiquetas=None, cols=6):
    """Todas las imágenes con el contorno final de la lesión."""
    filas = int(np.ceil(len(ids) / cols))
    fig, axs = plt.subplots(filas, cols, figsize=(2.7 * cols, 2.9 * filas))
    for ax, id_ in zip(axs.flat, ids):
        ax.imshow(superponer(resultados_e1[id_]["rgb"], resultados_e2[id_]["mascara"], lesiones[id_]))
        ax.set_title(id_ if etiquetas is None else f"{id_} · {etiquetas[id_]}", fontsize=9)
    for ax in axs.flat:
        ax.axis("off")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Guardado / carga
# --------------------------------------------------------------------------
def guardar_etapa3(lesion, id_, carpeta):
    """Guarda <id>_lesion.png (blanco = lesión). Es lo que lee la etapa 4."""
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / f"{id_}_lesion.png"
    ok, buf = cv2.imencode(".png", lesion.astype(np.uint8) * 255)
    buf.tofile(str(ruta))
    return ruta


def cargar_etapa3(ruta):
    return cv2.imdecode(np.fromfile(str(ruta), dtype=np.uint8), cv2.IMREAD_GRAYSCALE) > 0
