"""
Etapa 5 (parte GLCM): descriptores de textura de la matriz de co-ocurrencia de niveles
de gris (GLCM) sobre el canal L* SIN filtrar, calculados por zona de la lesión.

Entrada: los archivos resultados/Etapa_4/e4_*.npz, que traen el canal L* y las máscaras
booleanas de las cuatro zonas (lesion, borde, centro y perilesional).

Qué hace:
  1. Cuantiza L* a 32 niveles enteros (acepta L* en 0-100 o codificado en 0-255).
  2. Arma la GLCM a mano (glcm_enmascarada), usando solo los pares de píxeles que caen
     dentro de la máscara de la zona, porque graycomatrix de skimage no permite enmascarar.
     La matriz es simétrica (se suma la traspuesta) y está normalizada.
  3. La calcula para las distancias 1 y 2 y los ángulos 0, 45, 90 y 135 grados, y promedia
     los resultados entre las 8 combinaciones.
  4. De cada GLCM obtiene contraste, disimilitud, homogeneidad, energía y correlación
     (con graycoprops de skimage) y la entropía de Shannon en bits (calculada acá, porque
     graycoprops no la ofrece).

Salida de calcular_descriptores: diccionario con 'contrast', 'dissimilarity',
'homogeneity', 'energy', 'correlation', 'entropy', 'n_pixeles' y 'n_pares', o None si la
zona tiene menos de MIN_PIXELES (100) píxeles o ninguna GLCM válida.

procesar_todos_npz recorre todos los .npz, calcula los descriptores de las cuatro zonas
y guarda en la carpeta de salida:
  - descriptores_texturales.csv: una fila por imagen y zona (con una columna 'estado')
  - configuracion.json: parámetros usados (canal, niveles, distancias, ángulos, etc.)
Las rutas por defecto ("../resultados/...") asumen que se ejecuta desde una subcarpeta
del repositorio (por ejemplo, desde etapas/); desde la raíz hay que pasar las rutas.

Este módulo calcula los descriptores por zona. Las diferencias entre zonas
(borde − centro y lesión − perilesional) no se calculan acá.

Además incluye funciones para visualización: glcm_promedio (la GLCM de una región,
promediada en los 4 ángulos, con distancia 1), mapas_glcm_locales (mapas de contraste, disimilitud y
homogeneidad con ventana de 15 px, calculados exactos con un filtro de media porque solo
dependen de la diferencia i − j) y mapa_entropia_glcm (mapa de entropía con ventana de
21 px cada 6 px, interpolado al tamaño de la imagen).
"""

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

    # Si la matriz es de 255, la convertimos a 0-100, porque sino quedaría una matriz gigante
    # al dope
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
    
    Se arma a mano porque la función de skimage graycomatrix no permite enmascarar la imagen.
    """

    alto, ancho = imagen.shape

    # Desplazamiento según distancia y orientación
    rad = np.deg2rad(angulo)

    # Convierte los ángulos en un desplazamiento
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

    # Se queda con los píxeles que estén dentro de la máscara
    validos = m1 & m2

    if not validos.any():
        return None, 0

    valores1 = img1[validos].astype(int)
    valores2 = img2[validos].astype(int)

    # Construcción de la matriz de coocurrencia
    indices = valores1 * niveles + valores2

    # Con bincount se arma la matriz de coocurrencia
    conteos = np.bincount(
        indices,
        minlength=niveles * niveles
    )

    matriz = conteos.reshape(niveles, niveles).astype(float)

    # Simetría (se le suma la traspuesta) y normalización
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

    # Cuantiza L* a niveles enteros, si hay menos de 100 píxeles en la máscara, devuelve None
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

    # Para cada distancia y ángulo, se llama a la función anterior (glcm_enmascarada) 
    # y se calcula la GLCM. Luego, se calculan los descriptores y la entropía de la distribución de coocurrencias.
    for distancia in DISTANCIAS:
        for angulo in ANGULOS:

            matriz, n_pares = glcm_enmascarada(
                Lq, mascara, NIVELES, distancia, angulo
            )

            if matriz is None:
                continue

            resultados["n_pares"] += n_pares

            for nombre in descriptores:
                # Se calcula el descriptor usando graycoprops, que espera una matriz 4D, por eso se agrega un par de dimensiones nuevas.
                valor = graycoprops(
                    matriz[:, :, np.newaxis, np.newaxis],
                    nombre
                )[0, 0]

                valores[nombre].append(float(valor))

            # Entropía de la distribución de coocurrencias
            p = matriz[matriz > 0]
            # Entropía se calcula aparte, ya que graycoprops no la calcula directamente
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
# Visualización: GLCM promedio y mapas locales
# --------------------------------------------------

def desplazamiento(distancia, angulo):
    """(dr, dc) del par de píxeles para una distancia y un ángulo en grados."""
    rad = np.deg2rad(angulo)
    return int(round(-np.sin(rad) * distancia)), int(round(np.cos(rad) * distancia))


def glcm_promedio(L, mascara, distancia=1, niveles=NIVELES):
    """
    GLCM normalizada de una región, promediada en los 4 ángulos.
    Es la matriz que "ven" los descriptores (para graficarla).
    """
    Lq = cuantizar_L(L, niveles)
    mascara = np.asarray(mascara, dtype=bool)
    matrices = []
    for angulo in ANGULOS:
        m, _ = glcm_enmascarada(Lq, mascara, niveles, distancia, angulo)
        if m is not None:
            matrices.append(m)
    if not matrices:
        return None
    return np.mean(matrices, axis=0)


def mapas_glcm_locales(L, ventana=15, distancia=1, niveles=NIVELES):
    """
    Mapas locales de contraste, disimilitud y homogeneidad GLCM.

    Estas tres propiedades solo dependen de la diferencia (i - j) de cada par,
    así que su valor en una ventana es el promedio de f(i - j) sobre los pares
    de la ventana: se calcula exacto con un filtro de media (rápido), promediando
    los 4 ángulos como en calcular_descriptores.
    """
    from scipy.ndimage import uniform_filter

    Lq = cuantizar_L(L, niveles).astype(float)
    alto, ancho = Lq.shape
    funciones = {
        "contrast": lambda d: d ** 2,
        "dissimilarity": np.abs,
        "homogeneity": lambda d: 1.0 / (1.0 + d ** 2),
    }
    suma = {k: np.zeros_like(Lq) for k in funciones}
    peso = np.zeros_like(Lq)

    for angulo in ANGULOS:
        dr, dc = desplazamiento(distancia, angulo)
        diferencia = np.zeros_like(Lq)
        valido = np.zeros_like(Lq)
        r0, r1 = max(0, -dr), min(alto, alto - dr)
        c0, c1 = max(0, -dc), min(ancho, ancho - dc)
        diferencia[r0:r1, c0:c1] = Lq[r0:r1, c0:c1] - Lq[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
        valido[r0:r1, c0:c1] = 1.0
        n = uniform_filter(valido, ventana, mode="constant")
        for nombre, f in funciones.items():
            suma[nombre] += uniform_filter(f(diferencia) * valido, ventana, mode="constant") / np.maximum(n, 1e-9)
        peso += 1.0

    return {k: v / peso for k, v in suma.items()}


def mapa_entropia_glcm(L, ventana=21, paso=6, distancia=1, niveles=NIVELES):
    """
    Mapa local de entropía GLCM: la GLCM se arma en ventanas de
    ventana x ventana (4 ángulos) cada `paso` píxeles y el resultado
    se interpola al tamaño de la imagen.
    """
    import cv2

    Lq = cuantizar_L(L, niveles)
    alto, ancho = Lq.shape
    mitad = ventana // 2
    filas = np.arange(mitad, alto - mitad, paso)
    columnas = np.arange(mitad, ancho - mitad, paso)
    mapa = np.zeros((len(filas), len(columnas)))
    todo = np.ones((ventana, ventana), dtype=bool)

    for i, r in enumerate(filas):
        for j, c in enumerate(columnas):
            parche = Lq[r - mitad:r + mitad + 1, c - mitad:c + mitad + 1]
            matrices = [glcm_enmascarada(parche, todo, niveles, distancia, a)[0] for a in ANGULOS]
            p = np.mean(matrices, axis=0)
            p = p[p > 0]
            mapa[i, j] = -np.sum(p * np.log2(p))

    return cv2.resize(mapa, (ancho, alto), interpolation=cv2.INTER_LINEAR)


# --------------------------------------------------
# Procesamiento de todos los archivos NPZ
# --------------------------------------------------
# Devuelve lo que se armó en las funciones anteriores en un DataFrame y lo guarda en un CSV.
def procesar_todos_npz(
    carpeta_npz="../resultados/Etapa_4",
    carpeta_salida="../resultados/Etapa_5",
    patron="e4_*.npz"
):

    carpeta_npz = Path(carpeta_npz)
    carpeta_salida = Path(carpeta_salida)

    carpeta_salida.mkdir(parents=True, exist_ok=True)

    archivos = sorted(carpeta_npz.glob(patron))

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
