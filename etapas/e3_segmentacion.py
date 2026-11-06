"""
Etapa 3: Segmentación de la lesión por graph cut (GrabCut) en Lab.
 
Método semiautomático: la ROI recortada a mano en la Etapa 0 está centrada en la
lesión, así que GrabCut se inicializa con un prior geométrico sobre esa ROI. El
prior se define con la distancia elíptica normalizada al centro de la ROI,
d = sqrt(((x - W/2)/(W/2))^2 + ((y - H/2)/(H/2))^2):
 
  - d < 0,30                        -> lesión segura
  - 0,30 <= d < 0,80                -> lesión probable
  - 0,80 <= d <= 1,05               -> fondo probable
  - d > 1,05 (esquinas) y marco
    del 3 % del lado                -> fondo seguro
 
GrabCut no cambia las etiquetas seguras; el color (Lab) decide el contorno dentro
de la franja de etiquetas "probables". Se usan 5 iteraciones y los modelos de
mezcla de gaussianas por defecto de OpenCV (5 componentes por clase).
 
Posprocesamiento sobre el resultado de GrabCut:
  1. Intersección con la máscara válida de la Etapa 2 (en la práctica, toda la ROI).
  2. Apertura con un elemento estructurante elíptico de 9x9 (elimina regiones
     pequeñas y sueltas).
  3. Se conserva solo la componente conexa de mayor área (se descartan lesiones de alrededor).
  4. Relleno de huecos, para incorporar el centro aclarado de las lesiones anulares.
 
Salida de procesar_etapa3: {'mascara_lesion': máscara booleana rellena de la lesión}.
 
La corrección manual de la máscara (pincel y borrador) no está en este módulo: se
hace desde la interfaz gráfica (gui/editor_mascara.py) y la máscara corregida
reemplaza a la automática en las etapas siguientes.
"""

from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes


def _grabcut_lesion(rgb, iters=5): # Esta funcion aplica el algoritmo GrabCut para segmentar la lesión en la imagen RGB proporcionada.
    h, w = rgb.shape[:2]
    yy, xx = np.mgrid[:h, :w]
    # d es la distancia de cada píxel al centro de la imagen, normalizada por el tamaño de la imagen. Se utiliza para definir las regiones de fondo y primer plano.
    d = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)

    # Inicialmente, todos los píxeles se consideran posible fondo.
    m = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    m[d < 0.80] = cv2.GC_PR_FGD # Los píxeles donde la d < 0,8 se consideran probables fondo (probable foreground).
    m[d < 0.30] = cv2.GC_FGD # Los píxeles donde la d < 0,3 se consideran primer plano seguro (foreground).
    m[d > 1.05] = cv2.GC_BGD # Los píxeles donde la d > 1,05 se consideran fondo seguro (background).
    b = int(0.03 * min(h, w))
    m[:b, :] = m[-b:, :] = m[:, :b] = m[:, -b:] = cv2.GC_BGD

    # Convierte la imag RGB a Lab para que el algoritmo GrabCut pueda trabajar en el espacio de color Lab, que es más adecuado para la segmentación basada en color.
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    
    # Se aplica grabCut sobre la imagen Lab, con la máscara que indica las regiones de fondo y primer plano, y se actualizan los modelos de fondo y primer plano.
    cv2.grabCut(lab, m, None, bgd, fgd, iters, cv2.GC_INIT_WITH_MASK)
    return (m == cv2.GC_FGD) | (m == cv2.GC_PR_FGD)


def procesar_etapa3(e1, e2):
    rgb = e1["rgb"]
    mascara_piel = e2["mascara"]

    res = _grabcut_lesion(rgb) & mascara_piel
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    res = cv2.morphologyEx(res.astype(np.uint8), cv2.MORPH_OPEN, k)

    n, lab_cc, st, _ = cv2.connectedComponentsWithStats(res)
    if n > 1:
        res = lab_cc == (np.argmax(st[1:, cv2.CC_STAT_AREA]) + 1)

    return {"mascara_lesion": binary_fill_holes(res.astype(bool))}


def guardar_visualizacion_e3(e1, e3, nombre_archivo, carpeta_salida="resultados/Etapa_3"):
    carpeta = Path(carpeta_salida)
    carpeta.mkdir(parents=True, exist_ok=True)

    rgb = e1["rgb"]
    mascara = e3["mascara_lesion"]

    contorno = rgb.copy()
    cs, _ = cv2.findContours(mascara.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(contorno, cs, -1, (255, 255, 255), 2)

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(rgb)
    ax[0].set_title("ROI Original (512x512)")
    ax[1].imshow(mascara, cmap="gray")
    ax[1].set_title("Lesión (GrabCut)")
    ax[2].imshow(contorno)
    ax[2].set_title("Contorno sobre la ROI")
    for a in ax:
        a.axis("off")

    fig.tight_layout()
    ruta_salida = carpeta / f"e3_{Path(nombre_archivo).stem}.png"
    fig.savefig(ruta_salida, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return ruta_salida