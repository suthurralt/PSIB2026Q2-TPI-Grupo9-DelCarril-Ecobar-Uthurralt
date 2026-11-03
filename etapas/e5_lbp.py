import numpy as np
import pandas as pd

from pathlib import Path
from scipy.ndimage import binary_erosion
from skimage.feature import local_binary_pattern


# Parámetros LBP
P = 8
R = 3
METODO = "uniform"

ZONAS = ["lesion", "borde", "centro", "perilesional"]


def calcular_lbp(L, mascara, P=P, R=R):
    """
    Calcula descriptores LBP para una región de la imagen.
    """

    # Convertir L* de rango 0-100 a uint8
    L_uint8 = np.clip(L * 255 / 100, 0, 255).astype(np.uint8)

    # Calcular imagen LBP
    imagen_lbp = local_binary_pattern(
        L_uint8,
        P,
        R,
        method=METODO
    )

    # Evitar píxeles cercanos al borde de la máscara
    estructura = np.ones((2 * R + 1, 2 * R + 1), dtype=bool)

    mascara_interior = binary_erosion(
        mascara.astype(bool),
        structure=estructura
    )

    valores = imagen_lbp[mascara_interior]

    if len(valores) == 0:
        return None

    # Histograma de los códigos LBP
    hist, _ = np.histogram(
        valores,
        bins=np.arange(P + 3) - 0.5
    )

    hist = hist / hist.sum()

    # Entropía del histograma
    hist_no_cero = hist[hist > 0]

    entropia = -np.sum(
        hist_no_cero * np.log2(hist_no_cero)
    )

    resultado = {
        f"lbp_{i}": hist[i]
        for i in range(P + 2)
    }

    resultado["lbp_entropia"] = entropia
    resultado["n_pixeles"] = len(valores)

    return resultado


def procesar_todos_npz(
    carpeta_entrada="../resultados/Etapa_4",
    carpeta_salida="../resultados/Etapa_5"
):
    """
    Calcula descriptores LBP para todas las imágenes ROI
    y sus cuatro regiones.
    """

    carpeta_entrada = Path(carpeta_entrada)
    carpeta_salida = Path(carpeta_salida)

    carpeta_salida.mkdir(parents=True, exist_ok=True)

    archivos = sorted(
        carpeta_entrada.glob("e4_roi_*.npz")
    )

    filas = []

    for archivo in archivos:

        datos = np.load(archivo)

        L = datos["L"]

        nombre_imagen = archivo.stem.replace("e4_", "", 1)

        for zona in ZONAS:

            mascara = datos[zona]

            resultado = calcular_lbp(L, mascara)

            fila = {
                "imagen": nombre_imagen,
                "zona": zona
            }

            if resultado is None:
                fila["estado"] = "SIN_PIXELES"

            else:
                fila.update(resultado)
                fila["estado"] = "OK"

            filas.append(fila)

        datos.close()

    df = pd.DataFrame(filas)

    ruta_salida = carpeta_salida / "descriptores_lbp.csv"

    df.to_csv(ruta_salida, index=False)

    print(f"Imágenes procesadas: {len(archivos)}")
    print(f"Registros generados: {len(df)}")
    print(f"Archivo guardado en: {ruta_salida}")

    return df