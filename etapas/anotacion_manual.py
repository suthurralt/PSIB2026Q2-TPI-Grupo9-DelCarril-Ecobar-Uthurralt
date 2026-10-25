"""Anotación manual del contorno de la lesión (opción TEMPORAL a la etapa 3).

Sirve para dos cosas:
  1. Seguir con las etapas 4-6 usando lesiones bien recortadas mientras se corrige la
     segmentación automática.
  2. Tener una referencia para medir la segmentación automática (Dice) cuando se la mejore.

Uso desde la terminal (en la carpeta del repo):
    python -m etapas.anotacion_manual Entrenamiento
    python -m etapas.anotacion_manual Testeo
    python -m etapas.anotacion_manual Entrenamiento --rehacer "TC (156)"
    python -m etapas.anotacion_manual Entrenamiento --escala 0.8 --modo envolvente

o desde el notebook:  am.anotar_imagenes(df["ruta"], CARPETA_MANUAL)

Qué marcar (el MISMO criterio para todas, como si no supieras el diagnóstico):
    - El borde EXTERNO de la lesión: todo lo que se ve como lesión queda adentro.
      En una lesión con anillo, el anillo entero queda adentro (no se marca el centro aparte).
    - Anillo incompleto: seguí el borde hasta donde se ve y cerrá el contorno por el
      camino más corto, sin inventar lesión afuera de la foto.
    - Varias lesiones: la más grande o la más central.
    - Si los límites son dudosos, confirmá con "d" en vez de ENTER (queda anotada como dudosa).
    - Lesión DIFUSA que cubre toda la piel visible, sin un borde para seguir (pápulas
      dispersas, eritema o descamación en toda la foto): tecla "t". La lesión pasa a ser
      toda la piel de la imagen.
    - No se ve ninguna lesión que se pueda delimitar: tecla "n" (queda registrada sin lesión).

    OJO: que la lesión sea homogénea o no tenga anillo NO es motivo para usar "t" o "n".
    Una placa homogénea con borde se contornea igual que cualquier otra: si tiene anillo
    o no es justamente lo que mide la etapa 4. "t" y "n" dependen solo de si hay un
    BORDE que seguir, no de la forma de la lesión.

Mouse y teclas:
    - Click izquierdo: agregar un punto · mantener apretado y arrastrar: dibujar a mano alzada
    - Click derecho o BACKSPACE: borrar el último punto · r: borrar todo
    - ENTER o ESPACIO: cerrar el contorno y confirmar · d: confirmar como dudosa
    - t: lesión difusa (toda la piel) · n: sin lesión delimitable
    - s: saltear la imagen (queda pendiente) · q o ESC: salir (lo anotado ya quedó guardado)

Por cada imagen se guarda en la carpeta de anotaciones:
    <id>_mascara.png   máscara de la lesión (blanco = lesión), mismo tamaño que la imagen
    <id>_puntos.json   puntos del contorno (coordenadas de la imagen original) y el tipo de
                       anotación: "contorno", "dudosa", "difusa" o "sin_lesion"
    <id>_vista.png     la imagen con el contorno dibujado, para revisar
"""
import argparse
import json
from pathlib import Path
import random

import cv2
import numpy as np

TEXTO_AYUDA = ["click: punto  der/BKSP: borrar  r: reset",
               "ENTER: ok  d: dudosa  t: difusa  n: sin lesion  s: saltear  q: salir"]
TIPOS = ("contorno", "dudosa", "difusa", "sin_lesion")
DIST_MIN_ARRASTRE = 4      # px de pantalla entre puntos al dibujar a mano alzada


# --------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------
def escala_automatica(alto_img):
    """Escala para que la imagen ocupe ~70% del alto de la pantalla (entre 0.6x y 1.6x)."""
    try:
        import ctypes
        alto_pantalla = ctypes.windll.user32.GetSystemMetrics(1)   # solo Windows
    except Exception:
        alto_pantalla = 900
    return float(np.clip(0.7 * alto_pantalla / alto_img, 0.6, 1.6))


def cargar_bgr(ruta):
    return cv2.imdecode(np.fromfile(str(ruta), dtype=np.uint8), cv2.IMREAD_COLOR)


def _escribir_png(ruta, imagen):
    ok, buf = cv2.imencode(".png", imagen)
    buf.tofile(str(ruta))


# --------------------------------------------------------------------------
# Interfaz de anotación
# --------------------------------------------------------------------------
def anotar_contorno(img_bgr, titulo, escala=None):
    """Abre una ventana para dibujar el contorno.

    Devuelve (tipo, puntos):
      ("contorno" o "dudosa", array N x 2 de int en coordenadas de la imagen original)
      ("difusa", None)      lesión difusa: toda la piel
      ("sin_lesion", None)  no hay lesión delimitable
      ("saltear", None) / ("salir", None)
    """
    escala = escala or escala_automatica(img_bgr.shape[0])
    interp = cv2.INTER_AREA if escala < 1 else cv2.INTER_LINEAR
    base = cv2.resize(img_bgr, None, fx=escala, fy=escala, interpolation=interp)
    estado = {"puntos": [], "arrastrando": False}

    def mouse(evento, x, y, flags, param):
        pts = estado["puntos"]
        if evento == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))
            estado["arrastrando"] = True
        elif evento == cv2.EVENT_MOUSEMOVE and estado["arrastrando"]:
            if np.hypot(x - pts[-1][0], y - pts[-1][1]) >= DIST_MIN_ARRASTRE:
                pts.append((x, y))
        elif evento == cv2.EVENT_LBUTTONUP:
            estado["arrastrando"] = False
        elif evento == cv2.EVENT_RBUTTONDOWN and pts:
            pts.pop()

    cv2.namedWindow(titulo, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(titulo, mouse)
    tam = 0.42 * max(escala, 0.8)
    try:
        while True:
            vista = base.copy()
            pts = estado["puntos"]
            if len(pts) >= 2:
                cv2.polylines(vista, [np.array(pts, np.int32)], False, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.line(vista, pts[-1], pts[0], (0, 200, 255), 1, cv2.LINE_AA)  # cierre (provisorio)
            for p in pts[-1:]:
                cv2.circle(vista, p, 4, (0, 255, 0), -1)
            for k, linea in enumerate(TEXTO_AYUDA):
                yy = int(18 + k * 22 * max(escala, 0.8))
                cv2.putText(vista, linea, (6, yy), cv2.FONT_HERSHEY_SIMPLEX, tam, (255, 255, 255), 3, cv2.LINE_AA)
                cv2.putText(vista, linea, (6, yy), cv2.FONT_HERSHEY_SIMPLEX, tam, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.imshow(titulo, vista)

            tecla = cv2.waitKey(20) & 0xFF
            if cv2.getWindowProperty(titulo, cv2.WND_PROP_VISIBLE) < 1 or tecla in (ord("q"), 27):
                return "salir", None
            if tecla == ord("s"):
                return "saltear", None
            if tecla == ord("t"):
                return "difusa", None
            if tecla == ord("n"):
                return "sin_lesion", None
            if tecla == ord("r"):
                pts.clear()
            if tecla == 8 and pts:                      # BACKSPACE
                pts.pop()
            if tecla in (13, 32, ord("d")) and len(pts) >= 3:
                alto, ancho = img_bgr.shape[:2]
                originales = np.round(np.array(pts, float) / escala).astype(int)
                originales[:, 0] = np.clip(originales[:, 0], 0, ancho - 1)
                originales[:, 1] = np.clip(originales[:, 1], 0, alto - 1)
                return ("dudosa" if tecla == ord("d") else "contorno"), originales
    finally:
        cv2.destroyWindow(titulo)


def puntos_a_mascara(puntos, forma, modo="poligono"):
    """Convierte el contorno en máscara booleana.

    modo="poligono": el polígono tal cual se dibujó.
    modo="envolvente": la envolvente convexa de los puntos (más tolerante a contornos
    dibujados rápido, pero rellena las concavidades de la lesión).
    """
    pts = np.asarray(puntos, np.int32).reshape(-1, 1, 2)
    if modo == "envolvente":
        pts = cv2.convexHull(pts)
    elif modo != "poligono":
        raise ValueError(f"modo debe ser 'poligono' o 'envolvente', no {modo!r}")
    mascara = np.zeros(forma[:2], np.uint8)
    cv2.fillPoly(mascara, [pts], 1)
    return mascara.astype(bool)


# --------------------------------------------------------------------------
# Guardado / carga
# --------------------------------------------------------------------------
def rutas_anotacion(carpeta, id_):
    carpeta = Path(carpeta)
    return {"mascara": carpeta / f"{id_}_mascara.png",
            "puntos": carpeta / f"{id_}_puntos.json",
            "vista": carpeta / f"{id_}_vista.png"}


def guardar_anotacion(carpeta, id_, puntos, mascara, img_bgr, tipo="contorno", modo="poligono"):
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    r = rutas_anotacion(carpeta, id_)
    _escribir_png(r["mascara"], mascara.astype(np.uint8) * 255)
    with open(r["puntos"], "w", encoding="utf-8") as f:
        json.dump({"id": id_, "modo": modo, "tipo": tipo, "dudosa": tipo == "dudosa",
                   "puntos": [] if puntos is None else np.asarray(puntos).tolist()}, f)
    vista = img_bgr.copy()
    contornos, _ = cv2.findContours(mascara.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(vista, contornos, -1, (0, 255, 0), 2)
    if tipo in ("difusa", "sin_lesion"):
        texto = "DIFUSA (toda la piel)" if tipo == "difusa" else "SIN LESION"
        cv2.putText(vista, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(vista, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
    _escribir_png(r["vista"], vista)


def esta_anotada(carpeta, id_):
    return rutas_anotacion(carpeta, id_)["mascara"].exists()


def cargar_mascara(carpeta, id_):
    """Máscara booleana de la lesión anotada a mano (None si no está anotada)."""
    ruta = rutas_anotacion(carpeta, id_)["mascara"]
    if not ruta.exists():
        return None
    return cv2.imdecode(np.fromfile(str(ruta), dtype=np.uint8), cv2.IMREAD_GRAYSCALE) > 0


def tipo_anotacion(carpeta, id_):
    """"contorno", "dudosa", "difusa", "sin_lesion" o None si no está anotada."""
    ruta = rutas_anotacion(carpeta, id_)["puntos"]
    if not ruta.exists():
        return None
    with open(ruta, encoding="utf-8") as f:
        d = json.load(f)
    return d.get("tipo", "dudosa" if d.get("dudosa") else "contorno")   # compatible con las primeras anotaciones


def es_dudosa(carpeta, id_):
    t = tipo_anotacion(carpeta, id_)
    return None if t is None else t == "dudosa"


# --------------------------------------------------------------------------
# Bucle principal
# --------------------------------------------------------------------------
def anotar_imagenes(rutas, carpeta, modo="poligono", escala=None, rehacer=False, mezclar=True):
    """Anota una lista de imágenes. Saltea las ya anotadas salvo rehacer=True.

    rutas: rutas de las imágenes (p. ej. df["ruta"]). El id es el nombre sin extensión.
    mezclar: orden aleatorio (fijo) y sin mostrar el nombre, para anotar sin saber la clase.
    """
    carpeta = Path(carpeta)
    rutas = [Path(r) for r in rutas]
    pendientes = [r for r in rutas if rehacer or not esta_anotada(carpeta, r.stem)]
    if mezclar:
        random.Random(0).shuffle(pendientes)
    print(f"{len(rutas) - len(pendientes)} ya anotadas, {len(pendientes)} por anotar.")

    for n, ruta in enumerate(pendientes, start=1):
        img = cargar_bgr(ruta)
        titulo = f"Anotar lesion - {n} de {len(pendientes)}" if mezclar else ruta.stem
        tipo, puntos = anotar_contorno(img, titulo, escala=escala)
        if tipo == "salir":
            print("Salida. Lo anotado quedó guardado.")
            break
        if tipo == "saltear":
            print(f"  {n}: salteada")
            continue
        if tipo == "difusa":
            mascara = np.ones(img.shape[:2], bool)       # toda la imagen; se recorta con la máscara de piel
        elif tipo == "sin_lesion":
            mascara = np.zeros(img.shape[:2], bool)
        else:
            mascara = puntos_a_mascara(puntos, img.shape, modo=modo)
        guardar_anotacion(carpeta, ruta.stem, puntos, mascara, img, tipo=tipo, modo=modo)
        detalle = f"{len(puntos)} puntos" if puntos is not None else ""
        print(f"  {n}: guardada ({tipo}{', ' + detalle if detalle else ''})")

    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass
    hechas = sum(esta_anotada(carpeta, r.stem) for r in rutas)
    print(f"\n{hechas} de {len(rutas)} anotadas en {carpeta.resolve()}")


def dice(a, b):
    """Coeficiente de Dice entre dos máscaras: 2|A∩B| / (|A|+|B|). 1 = idénticas, 0 = sin superposición."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    total = a.sum() + b.sum()
    return np.nan if total == 0 else 2 * (a & b).sum() / total


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("conjunto", choices=["Entrenamiento", "Testeo"])
    p.add_argument("--rehacer", help='nombre de una imagen para volver a anotarla, p. ej. "TC (156)"')
    p.add_argument("--escala", type=float, help="tamaño en pantalla (1 = 512 px); por defecto se ajusta a la pantalla")
    p.add_argument("--modo", default="poligono", choices=["poligono", "envolvente"])
    p.add_argument("--dataset", default="Dataset_reducido")
    p.add_argument("--carpeta", default="anotaciones", help="carpeta base donde se guardan (se agrega /<conjunto>)")
    args = p.parse_args()

    rutas = sorted((Path(args.dataset) / args.conjunto).glob("*.jpg"))
    if not rutas:
        raise SystemExit(f"No hay imágenes en {(Path(args.dataset) / args.conjunto).resolve()}")
    carpeta = Path(args.carpeta) / args.conjunto
    if args.rehacer:
        rutas = [r for r in rutas if r.stem == args.rehacer]
        if not rutas:
            raise SystemExit(f"No existe {args.rehacer}.jpg")

    print(__doc__[__doc__.index("Qué marcar"):__doc__.index("Por cada imagen")])
    anotar_imagenes(rutas, carpeta, modo=args.modo, escala=args.escala,
                    rehacer=bool(args.rehacer), mezclar=not args.rehacer)


if __name__ == "__main__":
    main()
