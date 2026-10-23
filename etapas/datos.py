"""Carga del dataset.

Estructura esperada en disco:

    Dataset_reducido/
        Entrenamiento/*.jpg
        Testeo/*.jpg

La clase se toma del nombre de archivo (formato SkinDisNet: "<SUBTIPO> (<n>).jpg"):
    TC -> clase "TC" (tiña corporis)
    CD, EC -> clase "Otras" (dermatitis de contacto, eczema)
"""
from pathlib import Path
import re

import cv2
import numpy as np
import pandas as pd

CONJUNTOS = ("Entrenamiento", "Testeo")
SUBTIPOS_TC = {"TC"}
SUBTIPOS_OTRAS = {"CD", "EC"}
_PATRON_NOMBRE = re.compile(r"^(?P<subtipo>[A-Za-z]+)\s*\((?P<num>\d+)\)$")


def listar_dataset(ruta_dataset, conjunto):
    """Devuelve un DataFrame (índice = nombre de la imagen) con las imágenes de UN conjunto.

    `conjunto` es "Entrenamiento" o "Testeo". Se carga solo ese conjunto a
    propósito: el pipeline se desarrolla con Entrenamiento y Testeo se usa
    únicamente en la evaluación final.

    Columnas: ruta, conjunto, subtipo (TC/CD/EC), clase (TC/Otras), etiqueta (1 = TC).
    """
    if conjunto not in CONJUNTOS:
        raise ValueError(f"conjunto debe ser uno de {CONJUNTOS}, no {conjunto!r}")
    carpeta = Path(ruta_dataset) / conjunto
    if not carpeta.is_dir():
        raise FileNotFoundError(f"No existe la carpeta {carpeta.resolve()}")

    filas = []
    for ruta in sorted(carpeta.iterdir()):
        if ruta.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        m = _PATRON_NOMBRE.match(ruta.stem)
        subtipo = m["subtipo"].upper() if m else None
        if subtipo in SUBTIPOS_TC:
            clase = "TC"
        elif subtipo in SUBTIPOS_OTRAS:
            clase = "Otras"
        else:
            raise ValueError(f"No se reconoce la clase de {ruta.name} "
                             f"(se espera 'TC (n)', 'CD (n)' o 'EC (n)')")
        filas.append({"id": ruta.stem, "ruta": ruta, "conjunto": conjunto,
                      "subtipo": subtipo, "clase": clase, "etiqueta": int(clase == "TC")})
    if not filas:
        raise FileNotFoundError(f"No hay imágenes en {carpeta.resolve()}")
    return pd.DataFrame(filas).set_index("id")


def verificar_separacion(ruta_dataset):
    """Chequea que ninguna imagen esté a la vez en Entrenamiento y Testeo (por nombre)."""
    nombres = {c: {p.stem for p in (Path(ruta_dataset) / c).glob("*.jpg")} for c in CONJUNTOS}
    repetidas = nombres["Entrenamiento"] & nombres["Testeo"]
    if repetidas:
        raise ValueError(f"Imágenes en ambos conjuntos: {sorted(repetidas)}")
    return True


def cargar_rgb(ruta):
    """Lee una imagen y la devuelve en RGB uint8 (OpenCV lee en BGR).

    Usa np.fromfile + imdecode para soportar rutas con espacios o acentos en Windows.
    """
    datos = np.fromfile(str(ruta), dtype=np.uint8)
    bgr = cv2.imdecode(datos, cv2.IMREAD_COLOR)
    if bgr is None:
        raise IOError(f"No se pudo leer {ruta}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
