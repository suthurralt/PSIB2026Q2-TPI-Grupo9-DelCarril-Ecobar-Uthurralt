import numpy as np
import pandas as pd

from pathlib import Path
from scipy.ndimage import binary_erosion
from skimage.feature import local_binary_pattern


# Parámetros LBP
P = 8
R = 1
METODO = "uniform"

ZONAS = ["lesion", "borde", "centro", "perilesional"]


def imagen_lbp(L, P=P, R=R):
    """
    Devuelve la imagen de códigos LBP ("uniform") de L* completo.

    Con P=8 cada píxel queda con un código entre 0 y P+1:
      0..P  -> patrones uniformes (cantidad de vecinos >= centro)
      P+1   -> patrones no uniformes
    """

    # Convertir L* de rango 0-100 a uint8
    L_uint8 = np.clip(L * 255 / 100, 0, 255).astype(np.uint8)

    # A cada píxel de la imagen L* se le asigna un código LBP 
    # basado en sus vecinos. El código es 1 si el vecino es mayor o igual que el 
    # píxel central, y 0 si es menor --> representa el patrón 
    # de intensidad local alrededor del píxel.
    return local_binary_pattern(
        L_uint8,
        P,
        R,
        method=METODO
    )


def calcular_lbp(L, mascara, P=P, R=R):
    """
    Calcula descriptores LBP para una región de la imagen.
    """

    # Calcular imagen LBP (ver imagen_lbp)
    imagen_lbp_ = imagen_lbp(L, P, R)

    # Evitar píxeles cercanos al borde de la máscara --> a cada píxel de la máscara 
    # se le aplica una erosión con un elemento estructurante circular de radio R. 
    # Esto asegura que solo se consideren los píxeles que tienen vecinos válidos para el cálculo del LBP.
    estructura = np.ones((2 * R + 1, 2 * R + 1), dtype=bool)

    mascara_interior = binary_erosion(
        mascara.astype(bool),
        structure=estructura
    )

    valores = imagen_lbp_[mascara_interior]

    if len(valores) == 0:
        return None

    # Histograma de los códigos LBP --> suman 1, porque se normaliza dividiendo por el total de píxeles considerados.
    hist, _ = np.histogram(
        valores,
        bins=np.arange(P + 3) - 0.5
    )

    hist = hist / hist.sum()

    # Entropía del histograma --> si hay alta entropia, hay una textura más heterogénea
    # entre los píxeles de la región, mientras que una baja entropía indica una textura más uniforme.
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

# Busca los resultados de la etapa 4, calcula los descriptores LBP para cada imagen y sus regiones,
# y guarda los resultados en un archivo CSV.
def procesar_todos_npz(
    carpeta_entrada="../resultados/Etapa_4",
    carpeta_salida="../resultados/Etapa_5",
    patron="e4_*.npz"
):
    """
    Calcula descriptores LBP para todas las imágenes ROI
    y sus cuatro regiones.
    """

    carpeta_entrada = Path(carpeta_entrada)
    carpeta_salida = Path(carpeta_salida)

    carpeta_salida.mkdir(parents=True, exist_ok=True)

    archivos = sorted(
        carpeta_entrada.glob(patron)
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