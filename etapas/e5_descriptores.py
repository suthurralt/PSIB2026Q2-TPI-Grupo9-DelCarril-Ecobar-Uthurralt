
from pathlib import Path
import numpy as np
import pandas as pd
import json

# --------------------------------------------------
# Parámetros de extracción
# --------------------------------------------------

NIVELES = 32
DISTANCIAS = [1, 2]
ANGULOS = [0, 45, 90, 135]

ZONAS = ["lesion", "borde", "centro", "perilesional"]

MIN_PIXELES = 100


# --------------------------------------------------
# Cuantización de L*
# --------------------------------------------------

def cuantizar_L(L, niveles=NIVELES):
    """
    Convierte L* a niveles enteros entre 0 y niveles-1.

    Admite L* expresado entre 0-100 o codificado entre 0-255.
    """

    L = np.asarray(L, dtype=float)

    if np.nanmax(L) > 100.5:
        L = L / 255.0 * 100.0

    L = np.clip(L, 0, 100)

    Lq = np.rint(L / 100 * (niveles - 1)).astype(np.uint8)

    return Lq


# --------------------------------------------------
# Construcción de GLCM respetando la máscara
# --------------------------------------------------

def glcm_enmascarada(imagen, mascara, niveles, distancia, angulo):
    """
    Construye una GLCM utilizando solamente pares de píxeles
    que pertenezcan a la región indicada por la máscara.

    Se consideran ambas direcciones de cada par (GLCM simétrica).
    """

    alto, ancho = imagen.shape

    # Desplazamiento según distancia y orientación
    rad = np.deg2rad(angulo)

    dr = int(round(-np.sin(rad) * distancia))
    dc = int(round(np.cos(rad) * distancia))

    # Coordenadas de los píxeles que forman cada par
    r1_ini = max(0, -dr)
    r1_fin = min(alto, alto - dr)

    c1_ini = max(0, -dc)
    c1_fin = min(ancho, ancho - dc)

    r2_ini = r1_ini + dr
    r2_fin = r1_fin + dr

    c2_ini = c1_ini + dc
    c2_fin = c1_fin + dc

    img1 = imagen[r1_ini:r1_fin, c1_ini:c1_fin]
    img2 = imagen[r2_ini:r2_fin, c2_ini:c2_fin]

    m1 = mascara[r1_ini:r1_fin, c1_ini:c1_fin]
    m2 = mascara[r2_ini:r2_fin, c2_ini:c2_fin]

    validos = m1 & m2

    if not validos.any():
        return None, 0

    valores1 = img1[validos].astype(int)
    valores2 = img2[validos].astype(int)

    # Construcción de la matriz de coocurrencia
    indices = valores1 * niveles + valores2

    conteos = np.bincount(
        indices,
        minlength=niveles * niveles
    )

    matriz = conteos.reshape(niveles, niveles).astype(float)

    # Simetría y normalización
    matriz = matriz + matriz.T

    total = matriz.sum()

    if total == 0:
        return None, 0

    matriz /= total

    return matriz, len(valores1)


# --------------------------------------------------
# Cálculo de descriptores
# --------------------------------------------------

def calcular_descriptores(L, mascara):

    from skimage.feature import graycoprops

    Lq = cuantizar_L(L)

    mascara = np.asarray(mascara, dtype=bool)

    if mascara.sum() < MIN_PIXELES:
        return None

    resultados = {
        "n_pixeles": int(mascara.sum()),
        "n_pares": 0
    }

    descriptores = [
        "contrast",
        "dissimilarity",
        "homogeneity",
        "energy",
        "correlation"
    ]

    valores = {nombre: [] for nombre in descriptores}
    entropias = []

    for distancia in DISTANCIAS:
        for angulo in ANGULOS:

            matriz, n_pares = glcm_enmascarada(
                Lq, mascara, NIVELES, distancia, angulo
            )

            if matriz is None:
                continue

            resultados["n_pares"] += n_pares

            for nombre in descriptores:
                valor = graycoprops(
                    matriz[:, :, np.newaxis, np.newaxis],
                    nombre
                )[0, 0]

                valores[nombre].append(float(valor))

            # Entropía de la distribución de coocurrencias
            p = matriz[matriz > 0]
            entropia = -np.sum(p * np.log2(p))

            entropias.append(float(entropia))

    if not entropias:
        return None

    # Promedio entre distancias y orientaciones
    for nombre in descriptores:
        resultados[nombre] = float(np.mean(valores[nombre]))

    resultados["entropy"] = float(np.mean(entropias))

    return resultados


# --------------------------------------------------
# Procesamiento de todos los archivos NPZ
# --------------------------------------------------

def procesar_todos_npz(
    carpeta_npz="../resultados/Etapa_4",
    carpeta_salida="../resultados/Etapa_5"
):

    carpeta_npz = Path(carpeta_npz)
    carpeta_salida = Path(carpeta_salida)

    carpeta_salida.mkdir(parents=True, exist_ok=True)

    archivos = sorted(carpeta_npz.glob("e4_roi_TC*.npz"))

    if not archivos:
        print("No se encontraron archivos NPZ.")
        return None

    filas = []

    for ruta in archivos:

        print(f"Procesando: {ruta.name}")

        with np.load(ruta) as datos:

            L = datos["L"]

            for zona in ZONAS:

                mascara = datos[zona]

                descriptores = calcular_descriptores(
                    L, mascara
                )

                fila = {
                    "imagen": ruta.stem.replace("e4_", ""),
                    "zona": zona
                }

                if descriptores is None:
                    fila["estado"] = "Región insuficiente"
                else:
                    fila.update(descriptores)
                    fila["estado"] = "OK"

                filas.append(fila)

    # Tabla consolidada
    df = pd.DataFrame(filas)

    ruta_csv = carpeta_salida / "descriptores_texturales.csv"

    df.to_csv(ruta_csv, index=False, decimal=".")

    # Guardar configuración para reproducibilidad
    configuracion = {
        "canal": "L*",
        "niveles": NIVELES,
        "distancias": DISTANCIAS,
        "angulos_grados": ANGULOS,
        "glcm_simetrica": True,
        "min_pixeles": MIN_PIXELES
    }

    with open(
        carpeta_salida / "configuracion.json",
        "w",
        encoding="utf-8"
    ) as archivo:
        json.dump(configuracion, archivo, indent=4)

    print("\nProcesamiento terminado.")
    print(f"Resultados guardados en: {ruta_csv}")

    return df
