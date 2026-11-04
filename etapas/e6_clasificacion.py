"""
Etapa 6 — Clasificación tiña / no tiña a partir de los descriptores de textura.

Uso desde la interfaz (después de la Etapa 4):

    from etapas.e6_clasificacion import clasificar
    resultado = clasificar(res_e1["L"], res_e4)
    resultado["veredicto"]   # "tiña" o "no tiña"

`res_e4` es el dict de procesar_etapa4 (tiene las máscaras lesion / borde / centro /
perilesional). También sirve cualquier dict o .npz con esas claves.

Cómo decide:
  * Para cada zona se calculan GLCM (e5_descriptores) y LBP (e5_lbp) sobre L*.
  * Se arman descriptores relativos de la misma foto:
        dif_BC__X = X(borde) − X(centro)
        dif_LP__X = X(lesión) − X(perilesional)
  * El notebook etapa5_texturas.ipynb elige unos pocos descriptores, con un umbral
    y un sentido cada uno, y los guarda en resultados/Etapa_5/clasificador.json.
  * Cada descriptor elegido vota "tiña" si su valor cae del lado de las tiñas del
    umbral. Si vota tiña más de la mitad, el veredicto es "tiña".

Para cambiar la regla no se toca este archivo: se vuelve a correr el notebook.
"""
import json
from pathlib import Path

import numpy as np

from etapas import e5_descriptores as glcm
from etapas import e5_lbp as lbp

ZONAS = ["lesion", "borde", "centro", "perilesional"]
DESC_GLCM = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "entropy"]
DESC_LBP = [f"lbp_{i}" for i in range(lbp.P + 2)] + ["lbp_entropia"]
NOMBRES = [f"glcm_{d}" for d in DESC_GLCM] + DESC_LBP

RUTA_CLASIFICADOR = Path(__file__).resolve().parents[1] / "resultados" / "Etapa_5" / "clasificador.json"


# --------------------------------------------------
# Descriptores de una imagen
# --------------------------------------------------

def descriptores_por_zona(L, zonas):
    """{zona: {descriptor: valor}} con GLCM + LBP de cada zona (NaN si la zona no alcanza)."""
    salida = {}
    for zona in ZONAS:
        mascara = np.asarray(zonas[zona], dtype=bool)
        d = glcm.calcular_descriptores(L, mascara) or {}
        r = lbp.calcular_lbp(L, mascara) or {}
        valores = {f"glcm_{k}": d.get(k, np.nan) for k in DESC_GLCM}
        valores.update({k: r.get(k, np.nan) for k in DESC_LBP})
        salida[zona] = valores
    return salida


def descriptores_imagen(L, zonas):
    """Diccionario plano: 'zona__descriptor' + relativos 'dif_BC__descriptor' y 'dif_LP__descriptor'."""
    por_zona = descriptores_por_zona(L, zonas)
    fila = {f"{z}__{n}": por_zona[z][n] for z in ZONAS for n in NOMBRES}
    for n in NOMBRES:
        fila[f"dif_BC__{n}"] = por_zona["borde"][n] - por_zona["centro"][n]
        fila[f"dif_LP__{n}"] = por_zona["lesion"][n] - por_zona["perilesional"][n]
    return fila


# --------------------------------------------------
# Clasificación
# --------------------------------------------------

def cargar_clasificador(ruta=RUTA_CLASIFICADOR):
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


def clasificar(L, zonas, clasificador=None):
    """
    Devuelve:
      veredicto   "tiña" / "no tiña"
      votos_tina  cuántos descriptores votaron tiña
      validos     cuántos descriptores se pudieron calcular
      detalle     lista con descriptor, valor, umbral, sentido y voto de cada uno
    """
    if clasificador is None:
        clasificador = cargar_clasificador()

    fila = descriptores_imagen(L, zonas)
    detalle, votos, validos = [], 0, 0
    for regla in clasificador["descriptores"]:
        valor = fila[regla["descriptor"]]
        if np.isnan(valor):
            voto = None                      # zona vacía (p. ej. lesión difusa sin borde)
        else:
            voto = valor > regla["umbral"] if regla["sentido"] == "mayor" else valor < regla["umbral"]
            validos += 1
            votos += int(voto)
        detalle.append({**regla, "valor": float(valor), "voto_tina": voto})

    es_tina = validos > 0 and votos > validos / 2
    return {"veredicto": "tiña" if es_tina else "no tiña", "votos_tina": votos,
            "validos": validos, "detalle": detalle}


def clasificar_npz(ruta_npz, clasificador=None):
    """Atajo para un archivo e4_*.npz guardado por exportar_e4."""
    with np.load(ruta_npz) as d:
        zonas = {z: d[z] for z in ZONAS}
        return clasificar(d["L"], zonas, clasificador)
