"""Herramienta para marcar a mano un rectángulo (ROI) alrededor de la lesión.

Uso (desde la carpeta del repo):
    python marcar_roi.py Entrenamiento
    python marcar_roi.py Testeo
    python marcar_roi.py Entrenamiento --rehacer "TC (156)"   # volver a marcar una imagen
    python marcar_roi.py Entrenamiento --escala 0.8           # mostrar más chico

Qué marcar (el MISMO criterio para todas, como si no supieras el diagnóstico):
    - El rectángulo más chico que contenga todo lo que se ve como lesión (borde e
      interior), más un margen chico de piel sana. No hace falta que haya anillo: el
      rectángulo solo le dice al programa DÓNDE está la lesión; si tiene forma de anillo
      o no es lo que se mide después.
    - Lesión sin anillo (placa, mancha, costra, zona descamada): marcá toda la zona afectada.
    - Anillo incompleto (un tramo se desvanece o sale de la foto): marcá la zona que el
      anillo encierra hasta donde se ve, sin "completarlo" afuera de la imagen.
    - Varias lesiones: la más grande o la más central.
    - Si la lesión ocupa casi toda la foto, marcala igual (el programa usa el borde de la
      foto como piel de referencia).
    - Si la lesión se ve muy poco o sus límites son dudosos, confirmá con "d" en vez de
      ENTER: queda marcada igual, pero anotada como dudosa.

Teclas:
    - Arrastrar con el mouse: dibujar (volver a arrastrar lo reemplaza).
    - ENTER o ESPACIO: confirmar · d: confirmar como dudosa
    - s: saltear (queda sin marcar; se vuelve a ofrecer la próxima vez) · q o ESC: salir

Tamaño: la imagen se ajusta a la pantalla. Si queda grande o chica: --escala 0.8 (o 1.2, etc.).

Las imágenes aparecen en orden aleatorio y sin el nombre del archivo, para marcar sin
saber la clase (importante en Testeo). Los rectángulos se guardan en rois.csv a medida
que se marcan; si se corta, al volver a correrlo sigue con las que faltan.
"""
import argparse
import csv
from pathlib import Path
import random

import cv2
import numpy as np

RUTA_DATASET = Path("Dataset_reducido")
RUTA_CSV = Path("rois.csv")
CAMPOS = ["id", "conjunto", "x", "y", "w", "h", "dudosa"]
TEXTO_AYUDA = "ENTER: ok  d: dudosa  s: saltear  q: salir"


def escala_automatica(alto_img):
    """Escala para que la imagen ocupe ~70% del alto de la pantalla (entre 0.6x y 1.6x)."""
    try:
        import ctypes
        alto_pantalla = ctypes.windll.user32.GetSystemMetrics(1)   # solo Windows
    except Exception:
        alto_pantalla = 900
    return float(np.clip(0.7 * alto_pantalla / alto_img, 0.6, 1.6))


def leer_rois():
    if not RUTA_CSV.exists():
        return {}
    with open(RUTA_CSV, newline="", encoding="utf-8") as f:
        return {(r["conjunto"], r["id"]): {**r, "dudosa": r.get("dudosa", "0")} for r in csv.DictReader(f)}


def guardar_rois(rois):
    with open(RUTA_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS)
        w.writeheader()
        for clave in sorted(rois):
            w.writerow({k: rois[clave][k] for k in CAMPOS})


def cargar_bgr(ruta):
    return cv2.imdecode(np.fromfile(str(ruta), dtype=np.uint8), cv2.IMREAD_COLOR)


def marcar(img_bgr, titulo, escala):
    """Muestra la imagen y deja dibujar un rectángulo con el mouse.

    Devuelve ("ok" o "dudosa", (x, y, w, h)) en coordenadas de la imagen original,
    ("saltear", None) o ("salir", None).
    """
    grande = cv2.resize(img_bgr, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA if escala < 1 else cv2.INTER_LINEAR)
    estado = {"p0": None, "p1": None, "arrastrando": False}

    def mouse(evento, x, y, flags, param):
        if evento == cv2.EVENT_LBUTTONDOWN:
            estado.update(p0=(x, y), p1=(x, y), arrastrando=True)
        elif evento == cv2.EVENT_MOUSEMOVE and estado["arrastrando"]:
            estado["p1"] = (x, y)
        elif evento == cv2.EVENT_LBUTTONUP:
            estado.update(p1=(x, y), arrastrando=False)

    cv2.namedWindow(titulo, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(titulo, mouse)
    try:
        while True:
            vista = grande.copy()
            if estado["p0"] is not None:
                cv2.rectangle(vista, estado["p0"], estado["p1"], (0, 255, 0), 2)
            tam = 0.5 * max(escala, 0.8)
            cv2.putText(vista, TEXTO_AYUDA, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, tam, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(vista, TEXTO_AYUDA, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, tam, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.imshow(titulo, vista)
            tecla = cv2.waitKey(20) & 0xFF
            if cv2.getWindowProperty(titulo, cv2.WND_PROP_VISIBLE) < 1:
                return "salir", None                       # cerraron la ventana con la X
            if tecla in (ord("q"), 27):
                return "salir", None
            if tecla == ord("s"):
                return "saltear", None
            if tecla in (13, 32, ord("d")) and estado["p0"] is not None:
                (x0, y0), (x1, y1) = estado["p0"], estado["p1"]
                x, y = min(x0, x1) / escala, min(y0, y1) / escala
                w, h = abs(x1 - x0) / escala, abs(y1 - y0) / escala
                if w >= 5 and h >= 5:
                    alto, ancho = img_bgr.shape[:2]
                    x, y = max(0, round(x)), max(0, round(y))
                    w, h = min(round(w), ancho - x), min(round(h), alto - y)
                    return ("dudosa" if tecla == ord("d") else "ok"), (x, y, w, h)
    finally:
        cv2.destroyWindow(titulo)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("conjunto", choices=["Entrenamiento", "Testeo"])
    p.add_argument("--rehacer", help='nombre de una imagen para volver a marcarla, p. ej. "TC (156)"')
    p.add_argument("--escala", type=float, help="tamaño de la imagen en pantalla (1 = 512 px); por defecto se ajusta a la pantalla")
    args = p.parse_args()

    carpeta = RUTA_DATASET / args.conjunto
    rutas = sorted(carpeta.glob("*.jpg"))
    if not rutas:
        raise SystemExit(f"No hay imágenes en {carpeta.resolve()}")

    rois = leer_rois()
    if args.rehacer:
        pendientes = [r for r in rutas if r.stem == args.rehacer]
        if not pendientes:
            raise SystemExit(f"No existe {args.rehacer}.jpg en {carpeta}")
    else:
        pendientes = [r for r in rutas if (args.conjunto, r.stem) not in rois]
        random.Random(0).shuffle(pendientes)   # orden aleatorio (fijo) para no ir por clase

    print(f"{args.conjunto}: {len(rutas) - len(pendientes)} ya marcadas, {len(pendientes)} por marcar.")
    print(__doc__[__doc__.index("Qué marcar"):__doc__.index("Tamaño:")])

    for n, ruta in enumerate(pendientes, start=1):
        img = cargar_bgr(ruta)
        escala = args.escala or escala_automatica(img.shape[0])
        accion, rect = marcar(img, f"Marcar lesion - {n} de {len(pendientes)}", escala)
        if accion == "salir":
            print("Salida. Lo marcado quedó guardado.")
            break
        if accion == "saltear":
            print(f"  {n}: salteada")
            continue
        x, y, w, h = rect
        rois[(args.conjunto, ruta.stem)] = {"id": ruta.stem, "conjunto": args.conjunto,
                                            "x": x, "y": y, "w": w, "h": h,
                                            "dudosa": int(accion == "dudosa")}
        guardar_rois(rois)
        print(f"  {n}: guardada" + (" (dudosa)" if accion == "dudosa" else ""))

    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass
    marcadas = sum(1 for (c, _) in rois if c == args.conjunto)
    print(f"\n{args.conjunto}: {marcadas} de {len(rutas)} marcadas en {RUTA_CSV.resolve()}")


if __name__ == "__main__":
    main()
