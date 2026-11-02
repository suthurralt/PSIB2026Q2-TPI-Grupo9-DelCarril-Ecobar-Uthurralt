"""Etapa 4 — Morfología: zonas de la lesión y eritema.

Entrada: la lesión (máscara RELLENA: anotación manual o segmentación automática) y la
salida de la etapa 2 (a_suave, máscara de piel).

Salida, por imagen:
  - lesion           la lesión completa (rellena), recortada a la piel.
  - borde / centro   la lesión partida por EROSIÓN: centro = lesión erosionada,
                     borde = lesión − centro (una banda de ancho proporcional al tamaño).
  - perilesional     piel sana alrededor: DILATACIÓN de la lesión − lesión. Es la
                     referencia de color de esa piel (tono, iluminación de esa foto).
  - eritema          la máscara "SIN RELLENAR": los píxeles de la lesión claramente más
                     rojos que la piel perilesional, limpiados con
                     apertura -> cierre -> componentes conexas -> área mínima.
  - eritema_relleno  la misma con los huecos rellenos.

Idea: en una TC el eritema se concentra en el BORDE y deja un hueco en el centro
(anillo); en una placa homogénea el eritema ocupa todo. Eso se mide en la etapa 5
comparando borde vs. centro y eritema vs. eritema relleno.

Diferencias con la limpieza de la etapa 3:
  - NO se elige "la lesión principal" del eritema: dentro de una lesión ya delimitada,
    un anillo cortado son varias componentes y todas son lesión. Quedarse con la mayor
    borraría justamente los tramos del anillo.
  - El cierre es chico (une tramos cercanos), porque cerrar de más rellena el centro y
    borra la diferencia que se quiere medir.
"""
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

FRAC_BORDE = 0.30        # ancho del borde = 30% del radio equivalente de la lesión
ANCHO_MIN = 3            # px
ANCHO_PERILESIONAL = 20  # px de piel alrededor de la lesión que se usan como referencia
PIXELES_MIN_REF = 500    # si la piel perilesional tiene menos, se usa toda la piel fuera de la lesión
K_ERITEMA = 1.5          # eritema: a* > mediana_ref + 1.5 · MAD_ref
R_APERTURA = 2           # px: borra puntitos rojos sueltos
R_CIERRE = 5             # px: une tramos cercanos del anillo (chico, para no rellenar el centro)
FRAC_AREA_MIN = 0.005    # componentes de eritema con menos de 0,5% del área de la lesión se descartan
AREA_MIN_PX = 30


def _disco(r):
    r = max(int(round(r)), 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def radio_equivalente(mascara):
    """Radio del círculo de igual área: sqrt(área / π)."""
    return float(np.sqrt(mascara.sum() / np.pi))


# --------------------------------------------------------------------------
# Zonas
# --------------------------------------------------------------------------
def zonas(lesion, piel, frac_borde=FRAC_BORDE, ancho_peri=ANCHO_PERILESIONAL):
    """Parte la lesión en borde y centro, y arma la banda de piel perilesional.

    La erosión trata como "lesión" lo que no es piel (fondo blanco) y lo que queda
    afuera de la foto: un corte de la foto o del fondo NO es un borde real de la lesión,
    así que ahí no se genera banda de borde.
    """
    lesion = lesion & piel
    vacio = np.zeros_like(lesion)
    if not lesion.any():
        return {"lesion": lesion, "borde": vacio, "centro": vacio, "perilesional": vacio, "ancho_borde": 0.0}

    ancho = max(ANCHO_MIN, frac_borde * radio_equivalente(lesion))
    extendida = (lesion | ~piel).astype(np.uint8)
    centro = cv2.erode(extendida, _disco(ancho), borderType=cv2.BORDER_CONSTANT, borderValue=1).astype(bool) & lesion
    borde = lesion & ~centro
    perilesional = cv2.dilate(lesion.astype(np.uint8), _disco(ancho_peri)).astype(bool) & ~lesion & piel
    return {"lesion": lesion, "borde": borde, "centro": centro, "perilesional": perilesional, "ancho_borde": ancho}


def referencia_piel(a, z, piel):
    """Mediana y MAD de a* en la piel de referencia.

    Orden de preferencia: perilesional -> toda la piel fuera de la lesión -> la propia
    lesión (lesión difusa que ocupa toda la piel: el eritema queda relativo a sí misma).
    """
    for nombre, region in [("perilesional", z["perilesional"]),
                           ("piel fuera de la lesión", piel & ~z["lesion"]),
                           ("lesión", z["lesion"])]:
        if region.sum() >= PIXELES_MIN_REF:
            v = a[region]
            med = float(np.median(v))
            mad = 1.4826 * float(np.median(np.abs(v - med)))
            return med, max(mad, 0.5), nombre
    return np.nan, np.nan, "ninguna"


# --------------------------------------------------------------------------
# Eritema (máscara sin rellenar) + limpieza morfológica
# --------------------------------------------------------------------------
def limpiar(cruda, lesion):
    """apertura -> cierre -> componentes conexas -> filtro por área mínima (sin elegir una principal)."""
    b = cv2.morphologyEx(cruda.astype(np.uint8), cv2.MORPH_OPEN, _disco(R_APERTURA))
    b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, _disco(R_CIERRE)).astype(bool) & lesion
    etiquetas, n = ndi.label(b, structure=np.ones((3, 3)))
    if n == 0:
        return b
    areas = ndi.sum(b, etiquetas, index=np.arange(1, n + 1))
    minimo = max(AREA_MIN_PX, FRAC_AREA_MIN * lesion.sum())
    return np.isin(etiquetas, np.flatnonzero(areas >= minimo) + 1)


def rellenar(mascara):
    """Relleno de huecos interiores (los que no tocan el borde de la imagen)."""
    return ndi.binary_fill_holes(mascara)


def procesar_etapa4(lesion, e2, k=K_ERITEMA):
    """Aplica la etapa 4. `lesion`: máscara rellena; `e2`: salida de la etapa 2."""
    piel, a = e2["mascara"], e2["a_suave"]
    z = zonas(lesion, piel)
    med, mad, fuente_ref = referencia_piel(a, z, piel)
    if np.isnan(med):
        cruda = np.zeros_like(z["lesion"])
    else:
        cruda = (a > med + k * mad) & z["lesion"]
    eritema = limpiar(cruda, z["lesion"])
    return {**z, "eritema_crudo": cruda, "eritema": eritema, "eritema_relleno": rellenar(eritema) & z["lesion"],
            "ref_mediana": med, "ref_mad": mad, "ref_fuente": fuente_ref}


# --------------------------------------------------------------------------
# Visualización
# --------------------------------------------------------------------------
COLORES = {"perilesional": (90, 160, 255), "borde": (255, 170, 0), "centro": (120, 220, 120)}


def pintar_zonas(rgb, r, alfa=0.45):
    img = rgb.astype(np.float32).copy()
    for nombre, color in COLORES.items():
        m = r[nombre]
        img[m] = (1 - alfa) * img[m] + alfa * np.array(color, np.float32)
    return img.astype(np.uint8)


def pintar_mascara(rgb, m, color=(255, 0, 255), alfa=0.55):
    img = rgb.astype(np.float32).copy()
    img[m] = (1 - alfa) * img[m] + alfa * np.array(color, np.float32)
    return img.astype(np.uint8)


def _contorno(img, m, color=(255, 255, 255), grosor=2):
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, cs, -1, color, grosor)
    return img


def mostrar_etapa4(rgb, r, titulo=""):
    """original con la lesión | zonas | eritema crudo | eritema limpio (sin rellenar) | eritema relleno."""
    paneles = [
        (_contorno(rgb.copy(), r["lesion"]), f"{titulo}\nlesión (contorno)"),
        (pintar_zonas(rgb, r), "zonas: perilesional / borde / centro"),
        (pintar_mascara(rgb, r["eritema_crudo"]), f"eritema crudo (a* > ref + {K_ERITEMA}·MAD)"),
        (_contorno(pintar_mascara(rgb, r["eritema"]), r["lesion"]), "eritema limpio (sin rellenar)"),
        (_contorno(pintar_mascara(rgb, r["eritema_relleno"]), r["lesion"]), "eritema relleno"),
    ]
    fig, ax = plt.subplots(1, len(paneles), figsize=(3.4 * len(paneles), 3.8))
    for x, (img, t) in zip(ax, paneles):
        x.imshow(img)
        x.set_title(t, fontsize=9)
        x.axis("off")
    fig.tight_layout()
    return fig


def grilla_etapa4(ids, res_e4, resultados_e1, etiquetas=None, cols=6):
    """Por imagen: zonas pintadas + eritema limpio en magenta + contorno de la lesión."""
    filas = int(np.ceil(len(ids) / cols))
    fig, axs = plt.subplots(filas, cols, figsize=(2.7 * cols, 2.9 * filas))
    for ax, id_ in zip(axs.flat, ids):
        r, rgb = res_e4[id_], resultados_e1[id_]["rgb"]
        img = pintar_mascara(pintar_zonas(rgb, r, alfa=0.3), r["eritema"], alfa=0.6)
        ax.imshow(_contorno(img, r["lesion"]))
        ax.set_title(id_ if etiquetas is None else f"{id_} · {etiquetas[id_]}", fontsize=9)
    for ax in axs.flat:
        ax.axis("off")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Guardado / carga
# --------------------------------------------------------------------------
MASCARAS = ("lesion", "borde", "centro", "perilesional", "eritema", "eritema_relleno")


def guardar_etapa4(r, id_, carpeta, rgb=None):
    """<id>_e4.npz con todas las máscaras y la referencia; <id>_e4.png para mirar (si se pasa rgb)."""
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / f"{id_}_e4.npz"
    np.savez_compressed(ruta, **{k: r[k] for k in MASCARAS},
                        ref_mediana=r["ref_mediana"], ref_mad=r["ref_mad"], ref_fuente=r["ref_fuente"],
                        ancho_borde=r["ancho_borde"])
    if rgb is not None:
        img = _contorno(pintar_mascara(pintar_zonas(rgb, r, alfa=0.3), r["eritema"], alfa=0.6), r["lesion"])
        ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        buf.tofile(str(carpeta / f"{id_}_e4.png"))
    return ruta


def cargar_etapa4(ruta):
    with np.load(ruta) as d:
        return {k: (d[k].item() if d[k].ndim == 0 else d[k]) for k in d.files}
