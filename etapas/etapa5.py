"""Descriptores de forma: circularidad y dimensión fractal del borde.

Entrada: la máscara de la lesión (RELLENA, anotación manual o automática —
la misma que usa la etapa 4), no la máscara de piel ni el eritema.

Salida por imagen:
  - area_px          cantidad de píxeles de la lesión.
  - perimetro_px     perímetro del contorno externo (cv2.arcLength).
  - circularidad      4·π·Área / Perímetro². 1 = círculo perfecto; cuanto más
                      bajo, más alejada de un círculo (alargada, con anillo
                      incompleto, bordes festoneados, etc.).
  - dim_fractal_borde dimensión fractal del contorno, estimada por conteo de
                      cajas (box-counting) sobre el borde de 1 píxel de la
                      máscara. Un contorno liso da un valor cercano a 1; un
                      contorno más irregular/festoneado da un valor más alto
                      (el límite teórico para una curva que llena el plano es 2).

Por qué estos dos:
  - La circularidad es sensible a la forma GLOBAL (un anillo incompleto o una
    lesión alargada baja el valor aunque el borde en sí sea liso).
  - La dimensión fractal es sensible a la irregularidad LOCAL del contorno
    (cuántos "dientes" tiene el borde), independientemente de la forma global.
  Las dos juntas separan "forma irregular pero borde liso" de "forma regular
  pero borde festoneado", que es justamente la clase de distinción que separa
  un anillo de TC (borde activo, descamado) de una placa de Otras.

Limitación conocida: el perímetro de un contorno pixelado (cv2.arcLength)
sobreestima el perímetro real de formas chicas, por el efecto "escalera" del
borde discreto. No es un problema para COMPARAR imágenes entre sí (todas
tienen el mismo sesgo), pero sí para comparar con valores de la literatura
calculados sobre contornos continuos.

Estas funciones NO dependen del resto del pipeline: reciben una máscara
booleana (o la cargan desde un PNG) y no necesitan los dict de las etapas 1-4.

--------------------------------------------------------------------------
Estructura radial: ratio de eritema y perfil de gradiente radial
--------------------------------------------------------------------------
A diferencia de lo anterior, estas dos SÍ necesitan color (a*) además de la
forma, y se apoyan en las zonas que arma la etapa 4 (borde, centro):

  - ratio_eritema (R_e) = media(a* en el borde) / media(a* en el centro).
    >1: el borde es más rojo que el centro (anillo, esperable en TC).
    ~1: borde y centro parecidos (placa homogénea).
  - perfil_gradiente_radial: intensidad media de a* en bandas concéntricas de
    distancia al centroide de la lesión, de r=0 (centroide) a r=1 (punto más
    alejado dentro de la lesión). Es la versión continua de la misma idea que
    R_e: en una TC se espera que suba hacia r=1; en una placa homogénea,
    que quede plano.

A diferencia de circularidad/dim_fractal_borde, a* puede ser negativo (eje
verde-rojo de CIELAB), así que si la media del centro da ~0 el cociente R_e
pierde sentido (se devuelve NaN); en ese caso conviene mirar el perfil
completo en vez de un solo número.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Circularidad
# --------------------------------------------------------------------------
def _contorno_externo(mascara):
    """Contorno externo más grande de la máscara (None si está vacía)."""
    contornos, _ = cv2.findContours(mascara.astype(np.uint8), cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_NONE)
    if not contornos:
        return None
    return max(contornos, key=cv2.contourArea)


def circularidad(mascara):
    """4·π·Área / Perímetro² del contorno externo de la lesión.

    Usa el área y el perímetro del MISMO contorno (no el conteo de píxeles de
    la máscara) para que la fórmula sea consistente. Si la lesión está vacía
    o el contorno es degenerado (perímetro 0, p. ej. una lesión de 1-2 px),
    devuelve NaN.
    """
    mascara = np.asarray(mascara, bool)
    contorno = _contorno_externo(mascara)
    if contorno is None:
        return np.nan
    area = cv2.contourArea(contorno)
    perimetro = cv2.arcLength(contorno, True)
    if perimetro <= 0:
        return np.nan
    return float(4 * np.pi * area / perimetro ** 2)


# --------------------------------------------------------------------------
# Dimensión fractal del borde (box-counting)
# --------------------------------------------------------------------------
def _pixeles_borde(mascara):
    """Coordenadas (x, y) del borde de 1 píxel de la máscara (lesión − erosión)."""
    m = mascara.astype(np.uint8)
    erosionada = cv2.erode(m, np.ones((3, 3), np.uint8))
    borde = (m - erosionada).astype(bool)
    ys, xs = np.nonzero(borde)
    return np.stack([xs, ys], axis=1)


def _tamanos_caja(puntos, n_min=2):
    """Tamaños de caja (potencias de 2) acotados por el tamaño del borde.

    Entre 2 px y la mitad de la extensión mayor del borde. Con muy pocos
    tamaños el ajuste lineal no es confiable (se descarta más abajo).
    """
    extension = max(puntos[:, 0].max() - puntos[:, 0].min(),
                     puntos[:, 1].max() - puntos[:, 1].min()) + 1
    exp_max = int(np.floor(np.log2(max(extension, 4))))
    exponentes = range(1, max(exp_max - 1, n_min))
    return [2 ** e for e in exponentes]


def dimension_fractal_borde(mascara, tamanos=None):
    """Dimensión fractal del contorno por conteo de cajas (box-counting).

    Para cada tamaño de caja s, se cuenta cuántas celdas de una grilla s×s
    contienen al menos un píxel del borde (N(s)). La dimensión fractal es la
    pendiente de log(N(s)) vs. log(1/s) (ajuste por cuadrados mínimos).

    Devuelve NaN si el borde tiene muy pocos píxeles o si, para los tamaños
    de caja disponibles, quedan menos de 3 puntos para el ajuste (lesión
    demasiado chica para que la medida sea confiable).
    """
    puntos = _pixeles_borde(mascara)
    if len(puntos) < 10:
        return np.nan

    if tamanos is None:
        tamanos = _tamanos_caja(puntos)

    usados, conteos = [], []
    for s in tamanos:
        celdas = puntos // s
        n = len(np.unique(celdas, axis=0))
        if n > 0:
            usados.append(s)
            conteos.append(n)

    if len(usados) < 3:
        return np.nan

    log_inv_s = np.log(1.0 / np.asarray(usados, dtype=float))
    log_n = np.log(np.asarray(conteos, dtype=float))
    pendiente, _ = np.polyfit(log_inv_s, log_n, 1)
    return float(pendiente)


# --------------------------------------------------------------------------
# Por imagen / por dataset
# --------------------------------------------------------------------------
def descriptores_forma(mascara):
    """Calcula todos los descriptores de forma de una máscara de lesión.

    Devuelve un dict: area_px, perimetro_px, circularidad, dim_fractal_borde.
    """
    mascara = np.asarray(mascara, bool)
    area_px = int(mascara.sum())
    if area_px == 0:
        return {"area_px": 0, "perimetro_px": np.nan,
                "circularidad": np.nan, "dim_fractal_borde": np.nan}

    contorno = _contorno_externo(mascara)
    perimetro = float(cv2.arcLength(contorno, True)) if contorno is not None else np.nan

    return {
        "area_px": area_px,
        "perimetro_px": perimetro,
        "circularidad": circularidad(mascara),
        "dim_fractal_borde": dimension_fractal_borde(mascara),
    }


def procesar_dataset(mascaras):
    """Aplica descriptores_forma a un dict {id: máscara booleana}.

    Devuelve un DataFrame indexado por id, una fila por imagen.
    """
    filas = {id_: descriptores_forma(m) for id_, m in mascaras.items()}
    df = pd.DataFrame.from_dict(filas, orient="index")
    df.index.name = "id"
    return df


def guardar_csv(df, ruta):
    """Guarda el DataFrame de descriptores en `ruta` (crea la carpeta si falta)."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ruta, encoding="utf-8")
    return ruta


def calcular_y_guardar(mascaras, ruta_csv):
    """Atajo: procesar_dataset + guardar_csv. Devuelve el DataFrame."""
    df = procesar_dataset(mascaras)
    guardar_csv(df, ruta_csv)
    return df


# --------------------------------------------------------------------------
# Estructura radial: ratio de eritema
# --------------------------------------------------------------------------
def ratio_eritema(a, borde, centro):
    """R_e = media(a* en el borde) / media(a* en el centro) (ver cabecera del módulo).

    `a` es el canal a* (p. ej. `a_suave` de la etapa 2, el mismo que usa la
    etapa 4 para la referencia de color). `borde` y `centro` son las máscaras
    booleanas que arma `e4.zonas()`.

    Devuelve NaN si el borde o el centro no tienen píxeles (lesión chica, sin
    banda de borde o completamente erosionada), o si la media del centro es 0
    (el cociente no tiene sentido: ver nota al final de la cabecera del módulo).
    """
    a = np.asarray(a, dtype=float)
    borde, centro = np.asarray(borde, bool), np.asarray(centro, bool)
    if not borde.any() or not centro.any():
        return np.nan
    media_centro = float(np.mean(a[centro]))
    if media_centro == 0:
        return np.nan
    return float(np.mean(a[borde]) / media_centro)


# --------------------------------------------------------------------------
# Estructura radial: perfil de gradiente radial
# --------------------------------------------------------------------------
def centroide(mascara):
    """Centroide (fila, columna) de una máscara booleana. (NaN, NaN) si está vacía."""
    ys, xs = np.nonzero(mascara)
    if len(xs) == 0:
        return np.nan, np.nan
    return float(ys.mean()), float(xs.mean())


def perfil_gradiente_radial(a, lesion, n_bins=10):
    """Perfil radial I(r): media de a* por banda de distancia al centroide.

    r=0 es el centroide de la lesión, r=1 es su punto más alejado DENTRO de la
    lesión. Cada imagen tiene su propio radio máximo (el de su propia lesión),
    así que el eje r es siempre una fracción [0, 1] de la extensión propia de
    esa lesión -- comparable entre lesiones chicas y grandes, igual que el
    criterio de `e4.zonas()` para el ancho del borde.

    Devuelve (r, intensidad): dos arrays de longitud n_bins. r[i] es el centro
    del bin i; intensidad[i] es la media de a* de los píxeles de la lesión
    cuya distancia (normalizada) al centroide cae en ese bin, o NaN si el bin
    no tiene píxeles (puede pasar con n_bins grande y lesiones chicas).

    En una TC con anillo se espera que intensidad suba hacia r=1 (borde más
    rojo); en una placa homogénea el perfil debería quedar más plano.
    """
    lesion = np.asarray(lesion, bool)
    r_centros = np.array([(i + 0.5) / n_bins for i in range(n_bins)])
    ys, xs = np.nonzero(lesion)
    if len(xs) < 10:
        return r_centros, np.full(n_bins, np.nan)

    cy, cx = ys.mean(), xs.mean()
    d = np.hypot(xs - cx, ys - cy)
    d_max = d.max()
    if d_max <= 0:
        return r_centros, np.full(n_bins, np.nan)
    r_norm = d / d_max

    valores_a = np.asarray(a, dtype=float)[ys, xs]
    indices_bin = np.clip((r_norm * n_bins).astype(int), 0, n_bins - 1)
    intensidad = np.full(n_bins, np.nan)
    for i in range(n_bins):
        en_bin = indices_bin == i
        if en_bin.any():
            intensidad[i] = float(valores_a[en_bin].mean())
    return r_centros, intensidad


def descriptores_estructura_radial(a, lesion, borde, centro, n_bins=10):
    """Ratio de eritema + perfil de gradiente radial, para una imagen.

    Devuelve un dict: ratio_eritema, y gradiente_r0 .. gradiente_r{n_bins-1}
    (la intensidad de cada banda radial de perfil_gradiente_radial, en orden
    de r=0 a r=1 -- quedan como columnas separadas para que entren en un CSV).
    """
    _, intensidad = perfil_gradiente_radial(a, lesion, n_bins=n_bins)
    salida = {"ratio_eritema": ratio_eritema(a, borde, centro)}
    salida.update({f"gradiente_r{i}": v for i, v in enumerate(intensidad)})
    return salida


def procesar_dataset_radial(canales_a, lesiones, bordes, centros, n_bins=10):
    """Aplica descriptores_estructura_radial a un dataset completo.

    `canales_a`, `lesiones`, `bordes`, `centros`: dicts {id: array/máscara},
    con las mismas claves (p. ej. resultados_e2[id]["a_suave"], res_e4[id]["lesion"],
    res_e4[id]["borde"], res_e4[id]["centro"]). Devuelve un DataFrame indexado
    por id, usable con guardar_csv como el de procesar_dataset.
    """
    filas = {id_: descriptores_estructura_radial(canales_a[id_], lesiones[id_],
                                                 bordes[id_], centros[id_], n_bins=n_bins)
             for id_ in lesiones}
    df = pd.DataFrame.from_dict(filas, orient="index")
    df.index.name = "id"
    return df


# --------------------------------------------------------------------------
# Carga de máscaras desde disco (compatible con etapa3 y anotación manual)
# --------------------------------------------------------------------------
def cargar_mascara_png(ruta):
    """Lee una máscara binaria desde un PNG (blanco = lesión)."""
    datos = cv2.imdecode(np.fromfile(str(ruta), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if datos is None:
        raise IOError(f"No se pudo leer {ruta}")
    return datos > 0


def cargar_mascaras_carpeta(carpeta, patron="*_lesion.png", sufijo="_lesion"):
    """Carga todas las máscaras de una carpeta que matcheen `patron`.

    Sirve tanto para resultados/etapa3/<conjunto>/ (patron="*_lesion.png",
    sufijo="_lesion") como para anotaciones/<conjunto>/ (patron="*_mascara.png",
    sufijo="_mascara"). El id de cada imagen es el nombre de archivo sin el sufijo.

    Devuelve un dict {id: máscara booleana}.
    """
    carpeta = Path(carpeta)
    mascaras = {}
    for ruta in sorted(carpeta.glob(patron)):
        id_ = ruta.stem
        if sufijo and id_.endswith(sufijo):
            id_ = id_[: -len(sufijo)]
        mascaras[id_] = cargar_mascara_png(ruta)
    return mascaras


"""Descriptores de textura: entropía y contraste de la GLCM, borde vs. centro.
 
Entrada: el canal L* SIN FILTRAR de la etapa 2 (`e2["L"]`, no `L_suave` -- la
mediana de la etapa 2 borra justamente las estructuras finas, escamas y
pápulas, que es lo que la GLCM tiene que medir) y las zonas `borde`/`centro`
de la etapa 4 (`e4.zonas()`).
 
GLCM (matriz de co-ocurrencia de niveles de gris): P(i, j) es, para un
desplazamiento (distancia, ángulo) fijo, la frecuencia con la que un píxel de
nivel i tiene un vecino de nivel j en esa dirección. Acá se usan 4 ángulos
habituales (0°, 45°, 90°, 135°) con distancia 1 px, sumados y normalizados
juntos -- la textura de un anillo no tiene una orientación fija dentro de la
foto, así que no conviene depender de una sola dirección.
 
Salida por imagen (ver `descriptores_textura_glcm`):
  - ent_borde, ent_centro, ent_diff   entropía GLCM (ENT) de cada zona y su diferencia
  - con_borde, con_centro, con_diff   contraste GLCM (CON) de cada zona y su diferencia
 
    ENT = -sum_{i,j} P(i,j)·log2(P(i,j))   (0·log2(0) := 0)
    CON =  sum_{i,j} (i-j)²·P(i,j)
 
Nota sobre el signo de ENT: la fórmula se suele escribir sin el signo "-"
delante en algunas tablas, pero sin él la suma da NEGATIVA (log2(P) <= 0 para
P en [0,1]) y "más desorganizado = más negativo" es confuso. Acá se aplica el
signo "-" de la definición estándar de entropía de Shannon, así que ENT >= 0
y más alto = más desorganizado, como dice la interpretación ("alta
desorganización en el anillo").
 
Por qué una GLCM "enmascarada" y no skimage.feature.graycomatrix directo:
borde y centro son regiones IRREGULARES (una banda, un blob con agujero), no
rectángulos. Si se le pasa la imagen completa a graycomatrix, cuenta pares de
píxeles que cruzan el límite de la región (p. ej. borde con centro, o con
fondo) como si fueran textura de esa región. Acá solo se cuenta un par
(p, vecino) si LOS DOS píxeles están dentro de la misma máscara.
 
Cuantización: L* se reescala linealmente, dentro de cada región, a
`niveles` niveles de gris (16 por defecto). Menos niveles = matriz menos
dispersa en regiones chicas (bandas de borde finas); es un parámetro, no un
valor validado contra la literatura.
 
Esta función depende de `etapas.forma` solo para reutilizar `guardar_csv`
(no hay nada de "forma" en los cálculos de acá).
"""

NIVELES_GLCM = 16
DISTANCIA_GLCM = 1
PIXELES_MIN_GLCM = 20   # por debajo de esto la matriz queda demasiado dispersa para confiar
 
 
def _angulos_offsets(distancia=DISTANCIA_GLCM):
    """Desplazamientos (dy, dx) para 0°, 45°, 90° y 135° a la distancia dada.
 
    Convención de imagen (fila crece hacia abajo): "arriba" es dy negativo.
    """
    d = distancia
    return [(0, d), (-d, d), (-d, 0), (-d, -d)]
 
 
# --------------------------------------------------------------------------
# GLCM enmascarada
# --------------------------------------------------------------------------
def _glcm_enmascarada(L, mascara, niveles=NIVELES_GLCM, angulos=None):
    """GLCM normalizada y simétrica de L dentro de `mascara`.
 
    Promedia (en realidad: acumula y después normaliza junto) los desplazamientos
    de `angulos`. Simétrica: cada par contado cuenta también como (j, i), que es
    la convención habitual (así ENT y CON no dependen de qué lado del par se mire).
 
    Devuelve None si la región tiene muy pocos píxeles o es de intensidad
    constante (no hay textura que medir).
    """
    angulos = angulos or _angulos_offsets()
    mascara = np.asarray(mascara, bool)
    if mascara.sum() < PIXELES_MIN_GLCM:
        return None
 
    valores = L[mascara]
    vmin, vmax = float(valores.min()), float(valores.max())
    if vmax <= vmin:
        return None
    Lq = np.clip(((L - vmin) / (vmax - vmin) * (niveles - 1)).round(), 0, niveles - 1).astype(np.int32)
 
    alto, ancho = L.shape
    P = np.zeros((niveles, niveles), dtype=np.float64)
    ys0, xs0 = np.nonzero(mascara)
    for dy, dx in angulos:
        ys2, xs2 = ys0 + dy, xs0 + dx
        validos = (ys2 >= 0) & (ys2 < alto) & (xs2 >= 0) & (xs2 < ancho)
        ys, xs, ys2v, xs2v = ys0[validos], xs0[validos], ys2[validos], xs2[validos]
        en_mascara = mascara[ys2v, xs2v]
        ys, xs, ys2v, xs2v = ys[en_mascara], xs[en_mascara], ys2v[en_mascara], xs2v[en_mascara]
        if len(ys) == 0:
            continue
        i, j = Lq[ys, xs], Lq[ys2v, xs2v]
        np.add.at(P, (i, j), 1)
        np.add.at(P, (j, i), 1)
 
    total = P.sum()
    if total == 0:
        return None
    return P / total
 
 
# --------------------------------------------------------------------------
# Propiedades de la GLCM
# --------------------------------------------------------------------------
def entropia_glcm(P):
    """ENT = -sum P(i,j)·log2(P(i,j)), con 0·log2(0) := 0. Más alto = más desorganizado."""
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(P > 0, P * np.log2(P), 0.0)
    return float(-term.sum())
 
 
def contraste_glcm(P):
    """CON = sum (i-j)²·P(i,j). Más alto = más contraste local de grises."""
    n = P.shape[0]
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    return float(np.sum((i - j) ** 2 * P))
 
 
# --------------------------------------------------------------------------
# Por imagen / por dataset
# --------------------------------------------------------------------------
def descriptores_textura_glcm(L, borde, centro, niveles=NIVELES_GLCM, distancia=DISTANCIA_GLCM):
    """Entropía y contraste GLCM de borde y centro, y su diferencia (borde − centro).
 
    `L`: canal L* sin filtrar (e2["L"]). `borde`, `centro`: máscaras booleanas
    de res_e4 (etapa 4).
 
    Devuelve un dict: ent_borde, ent_centro, ent_diff, con_borde, con_centro,
    con_diff. NaN en una zona si esa zona tiene muy pocos píxeles o intensidad
    constante (ver _glcm_enmascarada); en ese caso la diferencia también es NaN.
    """
    angulos = _angulos_offsets(distancia)
    P_borde = _glcm_enmascarada(L, borde, niveles=niveles, angulos=angulos)
    P_centro = _glcm_enmascarada(L, centro, niveles=niveles, angulos=angulos)
 
    ent_b = entropia_glcm(P_borde) if P_borde is not None else np.nan
    ent_c = entropia_glcm(P_centro) if P_centro is not None else np.nan
    con_b = contraste_glcm(P_borde) if P_borde is not None else np.nan
    con_c = contraste_glcm(P_centro) if P_centro is not None else np.nan
 
    return {
        "ent_borde": ent_b, "ent_centro": ent_c,
        "ent_diff": ent_b - ent_c if not (np.isnan(ent_b) or np.isnan(ent_c)) else np.nan,
        "con_borde": con_b, "con_centro": con_c,
        "con_diff": con_b - con_c if not (np.isnan(con_b) or np.isnan(con_c)) else np.nan,
    }
 
 
def procesar_dataset_textura(canales_L, bordes, centros, niveles=NIVELES_GLCM, distancia=DISTANCIA_GLCM):
    """Aplica descriptores_textura_glcm a un dataset completo.
 
    `canales_L`, `bordes`, `centros`: dicts {id: array/máscara}, p. ej.
    resultados_e2[id]["L"], res_e4[id]["borde"], res_e4[id]["centro"].
    Devuelve un DataFrame indexado por id, usable con guardar_csv.
    """
    filas = {id_: descriptores_textura_glcm(canales_L[id_], bordes[id_], centros[id_],
                                            niveles=niveles, distancia=distancia)
             for id_ in bordes}
    df = pd.DataFrame.from_dict(filas, orient="index")
    df.index.name = "id"
    return df
 
 
def calcular_y_guardar(canales_L, bordes, centros, ruta_csv, niveles=NIVELES_GLCM, distancia=DISTANCIA_GLCM):
    """Atajo: procesar_dataset_textura + guardar_csv. Devuelve el DataFrame."""
    df = procesar_dataset_textura(canales_L, bordes, centros, niveles=niveles, distancia=distancia)
    guardar_csv(df, ruta_csv)
    return df
 