"""Etapa 4 — Morfología: zonas de la lesión.

Entrada: la lesión (máscara RELLENA: edición manual o segmentación automática de la
etapa 3) y la salida de la etapa 2 (a_suave, máscara de piel).

Salida, por imagen:
  - lesion           la lesión completa (rellena), recortada a la piel.
  - borde / centro   la lesión partida por EROSIÓN: centro = lesión erosionada,
                     borde = lesión − centro (una banda de ancho proporcional al tamaño).
  - perilesional     piel sana alrededor: DILATACIÓN de la lesión − lesión. Es la
                     referencia de color de esa piel (tono e iluminación de esa foto).
  - ref_mediana / ref_mad / ref_fuente
                     mediana y MAD de a* en la piel de referencia, para normalizar
                     por foto en la etapa 5.

La etapa 5 mide números continuos por zona (contraste de a* borde vs. centro,
textura de L* en borde y centro, etc.) usando estas máscaras.
"""
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

FRAC_BORDE = 0.30        # ancho del borde = 30% del radio equivalente de la lesión
ANCHO_MIN = 3            # px
ANCHO_PERILESIONAL = 20  # px de piel alrededor de la lesión que se usan como referencia
PIXELES_MIN_REF = 500    # si la piel perilesional tiene menos, se usa toda la piel fuera de la lesión


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

    La erosión trata como "lesión" lo que no es piel (fondo) y lo que queda afuera de
    la foto: un corte de la foto o del fondo NO es un borde real de la lesión, así que
    ahí no se genera banda de borde.
    """
    lesion = lesion & piel
    vacio = np.zeros_like(lesion)
    if not lesion.any():
        return {"lesion": lesion, "borde": vacio, "centro": vacio,
                "perilesional": vacio, "ancho_borde": 0.0}

    ancho = max(ANCHO_MIN, frac_borde * radio_equivalente(lesion))
    extendida = (lesion | ~piel).astype(np.uint8)
    centro = cv2.erode(extendida, _disco(ancho), borderType=cv2.BORDER_CONSTANT,
                       borderValue=1).astype(bool) & lesion
    borde = lesion & ~centro
    perilesional = cv2.dilate(lesion.astype(np.uint8), _disco(ancho_peri)).astype(bool) & ~lesion & piel
    return {"lesion": lesion, "borde": borde, "centro": centro,
            "perilesional": perilesional, "ancho_borde": ancho}


def referencia_piel(a, z, piel):
    """Mediana y MAD de a* en la piel de referencia.

    Orden de preferencia: perilesional -> toda la piel fuera de la lesión -> la propia
    lesión (lesión difusa que ocupa toda la piel: queda relativa a sí misma).
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


def procesar_etapa4(e3, e2):
    """Parte la lesión (de E3 o editada a mano) en borde / centro / perilesional."""
    if isinstance(e3, dict):
        lesion = e3.get("mascara_lesion", e3.get("mascara_lesion_cruda"))
    else:
        lesion = e3

    piel = np.asarray(e2["mascara"], dtype=bool)
    lesion = np.asarray(lesion, dtype=bool)

    z = zonas(lesion, piel)
    med, mad, fuente_ref = referencia_piel(e2["a_suave"], z, piel)
    return {**z, "ref_mediana": med, "ref_mad": mad, "ref_fuente": fuente_ref}

def exportar_e4(e1, e2, res_e4, nombre_archivo, carpeta_salida="resultados/Etapa_4"):
    """Guarda en un .npz todo lo que necesita la etapa 5 (texturas) sin rehacer E1-E4."""
    carpeta = Path(carpeta_salida)
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / f"e4_{Path(nombre_archivo).stem}.npz"

    np.savez_compressed(
        ruta,
        rgb=e1["rgb"],
        L=e1["L"],
        a=e1["a"],
        a_suave=e2["a_suave"],
        lesion=res_e4["lesion"].astype(bool),
        borde=res_e4["borde"].astype(bool),
        centro=res_e4["centro"].astype(bool),
        perilesional=res_e4["perilesional"].astype(bool),
        ref_mediana=np.float32(res_e4["ref_mediana"]),
        ref_mad=np.float32(res_e4["ref_mad"]),
        ref_fuente=np.str_(res_e4["ref_fuente"]),
        ancho_borde=np.float32(res_e4["ancho_borde"]),
    )
    return ruta


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


def _contorno(img, m, color=(255, 255, 255), grosor=2):
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, cs, -1, color, grosor)
    return img


def guardar_visualizacion_e4(e1, res_e4, nombre_archivo, carpeta_salida="resultados/Etapa_4"):
    carpeta = Path(carpeta_salida)
    carpeta.mkdir(parents=True, exist_ok=True)

    rgb = e1["rgb"]
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))

    ax[0].imshow(_contorno(rgb.copy(), res_e4["lesion"]))
    ax[0].set_title("Lesión Delimitada")

    ax[1].imshow(pintar_zonas(rgb, res_e4))
    ax[1].set_title("Perilesional / Borde / Centro")

    for a in ax:
        a.axis("off")

    fig.tight_layout()
    ruta_salida = carpeta / f"e4_{Path(nombre_archivo).stem}.png"
    fig.savefig(ruta_salida, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return ruta_salida