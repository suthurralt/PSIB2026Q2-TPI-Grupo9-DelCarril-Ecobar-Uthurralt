"""
Etapa 5 (parte LBP): descriptores de textura con Local Binary Patterns (LBP) sobre el
canal L* SIN filtrar, calculados por zona de la lesión.

Entrada: los archivos resultados/Etapa_4/e4_*.npz, que traen el canal L* y las máscaras
booleanas de las cuatro zonas (lesion, borde, centro y perilesional).

Qué hace:
  1. Convierte L* (rango 0-100) a uint8 y calcula la imagen de códigos LBP con skimage:
     P = 8 vecinos, radio R = 1, método "uniform".
  2. Con el método "uniform", el código de cada píxel es:
       0..P  -> patrón uniforme (a lo sumo 2 transiciones); el valor es la cantidad de
                vecinos con L* mayor o igual que el del píxel central
       P + 1 -> patrón no uniforme (textura desordenada o ruidosa)
     Es decir, 10 códigos posibles (lbp_0 a lbp_9). Por ejemplo, lbp_0 es un píxel más
     claro que todos sus vecinos y lbp_8 es un píxel más oscuro o igual a todos ellos
     (punto oscuro o zona plana).
  3. Para cada zona, arma el histograma normalizado de códigos (suma 1) usando solo los
     píxeles de la máscara cuyos 8 vecinos también están en la máscara (la máscara se
     erosiona con un cuadrado de 3x3). Así el patrón de cada píxel se calcula sin salir
     de la zona.
  4. Calcula la entropía de Shannon del histograma, en bits (máximo log2(10) ≈ 3,32).
     Una entropía alta indica una textura más heterogénea; una baja, más uniforme.

Salida de calcular_lbp: diccionario con 'lbp_0' ... 'lbp_9' (fracción de píxeles con cada
código), 'lbp_entropia' y 'n_pixeles' (píxeles usados, después de la erosión), o None si
no queda ningún píxel. A diferencia del módulo de GLCM, no hay un mínimo de píxeles.

procesar_todos_npz recorre todos los .npz, calcula los descriptores de las cuatro zonas y
guarda descriptores_lbp.csv en la carpeta de salida: una fila por imagen y zona, con una
columna 'estado' ("OK" o "SIN_PIXELES"). Las rutas por defecto ("../resultados/...")
asumen que se ejecuta desde una subcarpeta del repositorio (por ejemplo, etapas/); desde la
raíz hay que pasar las rutas.

Este módulo calcula los descriptores por zona. Las diferencias entre zonas
(borde − centro y lesión − perilesional) se calculan en e6_clasificacion.py. El clasificador
final usa dos de estos descriptores en sus versiones relativas: dif_BC__lbp_9 y dif_LP__lbp_7.
"""



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

    # Calcular imagen LBP 
    imagen_lbp_ = imagen_lbp(L, P, R)

    # Se erosiona la máscara con un cuadrado de 3x3 (los 8 vecinos) para quedarse solo con
   # los píxeles cuyo patrón LBP se calcula íntegramente dentro de la zona.
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
