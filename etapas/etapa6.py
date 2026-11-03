"""Etapa 6 (estadística descriptiva) — resumen de los descriptores por clase.
 
No decide nada todavía (esa es `etapas/decision.py`): esto solo resume, para
cada descriptor de forma y de textura, cómo se comparan sus valores entre TC
y Otras en Entrenamiento -- n, media, desvío, mediana, mínimo y máximo de
cada uno, por clase.
"""
import numpy as np
import pandas as pd
 
 
def resumen_por_clase(df, columnas, clase_col="clase"):
    """Estadística descriptiva de `columnas` de `df`, separada por clase.
 
    `df`: DataFrame indexado por id, con una columna `clase_col` ("TC"/"Otras")
    y las columnas numéricas de `columnas` (p. ej. descriptores de forma o de
    textura, ya calculados).
 
    Devuelve un DataFrame con una fila por descriptor y columnas en dos
    niveles: (clase, estadística), con estadística en
    {n, media, desvio, mediana, minimo, maximo}. `n` cuenta los valores no
    NaN de cada descriptor en esa clase (puede ser menor que el total de
    imágenes: ver etapas/textura.py, PIXELES_MIN_GLCM).
    """
    filas = {}
    for col in columnas:
        fila = {}
        for clase, grupo in df.groupby(clase_col)[col]:
            valores = grupo.dropna()
            fila[(clase, "n")] = int(valores.size)
            fila[(clase, "media")] = float(valores.mean()) if valores.size else np.nan
            fila[(clase, "desvio")] = float(valores.std()) if valores.size else np.nan
            fila[(clase, "mediana")] = float(valores.median()) if valores.size else np.nan
            fila[(clase, "minimo")] = float(valores.min()) if valores.size else np.nan
            fila[(clase, "maximo")] = float(valores.max()) if valores.size else np.nan
        filas[col] = fila
 
    tabla = pd.DataFrame.from_dict(filas, orient="index")
    tabla.columns = pd.MultiIndex.from_tuples(tabla.columns, names=["clase", "estadistica"])
    tabla = tabla.sort_index(axis=1, level=0)
    tabla.index.name = "descriptor"
    return tabla
 
 
def guardar_csv(tabla, ruta_csv):
    """Guarda el resumen (con sus dos niveles de columna) en un CSV."""
    from pathlib import Path
    ruta_csv = Path(ruta_csv)
    ruta_csv.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(ruta_csv)
    return ruta_csv
import numpy as np
import pandas as pd
 
MEDIANA_CIRCULARIDAD = 0.8251249074899892
MEDIANA_DIM_FRACTAL_BORDE = 1.0088363436071974
 
CRITERIOS = (
    "circularidad_alta",
    "dim_fractal_borde_alta",
    "ratio_eritema_mayor_1",
    "perfil_borde_mayor_centro",
    "ent_borde_mayor_centro",
    "con_borde_mayor_centro",
)
 
 
def construir_criterios(df, mediana_circularidad=MEDIANA_CIRCULARIDAD,
                        mediana_dim_fractal=MEDIANA_DIM_FRACTAL_BORDE):
    """Evalúa los 6 criterios para cada fila de `df`.
 
    `df` necesita las columnas: circularidad, dim_fractal_borde, ratio_eritema,
    gradiente_r0, gradiente_r9 (bandas extremas del perfil radial; r0 = banda
    más interna/centro, r9 = banda más externa/borde -- ver
    `forma.perfil_gradiente_radial`), ent_diff, con_diff.
 
    Devuelve un DataFrame booleano (mismo índice que `df`), una columna por
    criterio (ver CRITERIOS). NaN en una columna de origen produce False en
    ese criterio (no "no se sabe"): se trata como que ese voto no se ganó.
    """
    return pd.DataFrame({
        "circularidad_alta": df["circularidad"] > mediana_circularidad,
        "dim_fractal_borde_alta": df["dim_fractal_borde"] > mediana_dim_fractal,
        "ratio_eritema_mayor_1": df["ratio_eritema"] > 1,
        "perfil_borde_mayor_centro": (df["gradiente_r9"] - df["gradiente_r0"]) > 0,
        "ent_borde_mayor_centro": df["ent_diff"] > 0,
        "con_borde_mayor_centro": df["con_diff"] > 0,
    }, index=df.index).fillna(False)
 
 
def contar_cumplidos(df):
    """Cuántos de los 6 criterios cumple cada fila de `df` (ver construir_criterios)."""
    return construir_criterios(df).sum(axis=1)
 
 
UMBRAL_VOTOS = 4  # cumplidos >= este valor -> "es tiña"
 
 
def veredicto(cumplidos, umbral=UMBRAL_VOTOS):
    """Traduce una cuenta de 0 a 6 al veredicto ("es tiña" si cumplidos >= umbral, si no "no es")."""
    return "es tiña" if cumplidos >= umbral else "no es"
 
 
def clasificar_dataset(df, umbral=UMBRAL_VOTOS):
    """Aplica la regla completa a un DataFrame con las columnas de construir_criterios.
 
    Devuelve un DataFrame con: cumplidos (0-6) y veredicto ("es tiña"/"no es"),
    más las 6 columnas booleanas de cada criterio, para poder ver por qué.
    """
    criterios = construir_criterios(df)
    salida = criterios.copy()
    salida["cumplidos"] = criterios.sum(axis=1)
    salida["veredicto"] = salida["cumplidos"].apply(lambda c: veredicto(c, umbral))
    return salida
 
 
def evaluar(veredictos, real):
    """Tabla cruzada clase real x veredicto, y accuracy/sensibilidad/especificidad.
 
    `veredictos`: Series de "es tiña"/"no es" (p. ej. la columna 'veredicto'
    de clasificar_dataset). `real`: Series de "TC"/"Otras", mismo índice.
    """
    comunes = real.index.intersection(veredictos.index)
    real, veredictos = real.loc[comunes], veredictos.loc[comunes]
 
    tabla = pd.crosstab(real, veredictos)
    for col in ("es tiña", "no es"):
        if col not in tabla.columns:
            tabla[col] = 0
    tabla = tabla[["es tiña", "no es"]]
 
    pred_tc = veredictos == "es tiña"
    real_tc = real == "TC"
    tp = int((pred_tc & real_tc).sum())
    tn = int((~pred_tc & ~real_tc).sum())
    fp = int((pred_tc & ~real_tc).sum())
    fn = int((~pred_tc & real_tc).sum())
    n = tp + tn + fp + fn
 
    return {
        "tabla": tabla,
        "accuracy": (tp + tn) / n if n else np.nan,
        "sensibilidad_tc": tp / (tp + fn) if (tp + fn) else np.nan,
        "especificidad_otras": tn / (tn + fp) if (tn + fp) else np.nan,
        "n_evaluadas": n,
    }