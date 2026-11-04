"""
Módulo: e0_normalizacion_roi.py
===============================
Etapa 0 del Pipeline Biomédico para Tinea Corporis.

Responsabilidades:
1. Ingesta de imágenes en espacio de color RGB.
2. Normalización espacial estandarizada a 512x512 píxeles.
3. Persistencia opcional de imágenes recortadas (prefijo 'roi_') en disco.
4. Registro de metadatos geométricos en 'rois.csv'.
"""
import csv
import os
from pathlib import Path
import cv2
import numpy as np

# Ruta por defecto a la carpeta de imágenes externas
RUTA_EXTERNAS_DEFECTO = Path("datos/Externas")
RUTA_ROIS_DEFECTO = Path("datos/ROIs")
RUTA_ROIS_CSV = Path("rois.csv")


def cargar_imagen_rgb(ruta_o_nombre):
    """
    Carga una imagen desde disco y la convierte al espacio de color RGB.
    Acepta rutas absolutas, relativas o nombres de archivo contenidos en 'datos/Externas/'.
    """
    ruta = Path(ruta_o_nombre)
    
    # Si solo pasaron el nombre de archivo, lo busca en la carpeta datos/Externas/
    if not ruta.is_file() and not ruta.is_absolute():
        ruta = RUTA_EXTERNAS_DEFECTO / ruta_o_nombre

    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró la imagen en: {ruta.resolve()}")

    # Lectura robusta para Windows (rutas con espacios o tildes)
    datos = np.fromfile(str(ruta), dtype=np.uint8)
    bgr = cv2.imdecode(datos, cv2.IMREAD_COLOR)
    
    if bgr is None:
        raise IOError(f"No se pudo decodificar la imagen: {ruta}")

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def normalizar_espacialmente(corte_array, tamano_destino=(512, 512)):
    """
    Normaliza el arreglo recortado a una dimensión fija de (512, 512, 3) 
    usando interpolación INTER_AREA para preservar las texturas de la piel.
    Devuelve un bloque contiguo en memoria de tipo uint8.
    """
    if corte_array is None or corte_array.size == 0:
        raise ValueError("El recorte ingresado está vacío y no se puede normalizar.")
    
    # Acá INTER_AREA se usa para reducir el tamaño de la imagen sin perder calidad.
    # Calcula los píxeles como el promedio de los píxeles vecinos, lo que es ideal para reducir imágenes.
    roi_norm = cv2.resize(corte_array, tamano_destino, interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(roi_norm, dtype=np.uint8)

def guardar_imagen_roi(roi_array, nombre_original, carpeta_salida="datos/ROIs"):
    """Guarda la matriz de 512x512 como una imagen física en 'datos/ROIs/' 
    agregando el prefijo 'roi_' al nombre original.."""
    os.makedirs(carpeta_salida, exist_ok=True)
    
    nombre_base = os.path.basename(nombre_original)
    nombre_roi = f"roi_{nombre_base}"
    ruta_destino = os.path.join(carpeta_salida, nombre_roi)
    
    # OpenCV guarda en BGR, así que convertimos RGB -> BGR antes de guardar
    bgr = cv2.cvtColor(roi_array, cv2.COLOR_RGB2BGR)
    cv2.imwrite(ruta_destino, bgr)
    
    print(f"[Etapa 0] Imagen ROI guardada en: {ruta_destino}")
    return ruta_destino

def registrar_roi_csv(nombre_archivo, centro, tamano, angulo, csv_path=RUTA_ROIS_CSV):
    """
    Escribe o actualiza la geometría del rectángulo recortado en rois.csv.
    """
    cx, cy = centro
    w, h = tamano
    
    file_exists = Path(csv_path).exists()

    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["archivo", "centro_x", "centro_y", "ancho", "alto", "angulo"])
        
        writer.writerow([
            os.path.basename(nombre_archivo),
            round(cx, 2),
            round(cy, 2),
            round(w, 2),
            round(h, 2),
            round(angulo, 2)
        ])
    print(f"[Etapa 0] Registro guardado en {csv_path} para {nombre_archivo}")