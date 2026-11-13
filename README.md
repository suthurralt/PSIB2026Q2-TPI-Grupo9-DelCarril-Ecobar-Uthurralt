# Detección de tiña corporis en imágenes clínicas mediante metodologías de procesamiento de imágenes clásicas

Trabajo Práctico Integrador de **16.63 – Procesamiento de Señales e Imágenes Biomédicas** - Grupo 9.

**Integrantes:** Francisco Del Carril (65229) · Victoria Nazarena Escobar (65435) · Sofía Uthurralt (65579)

## Tema y objetivo

La tiña corporis (TC) es una infección fúngica de la piel con múltiples imitadoras clínicas tales como el eczema numular y la dermatitis de contacto. Estas tres afecciones se tratan de forma opuesta: la tiña con antifúngicos, y las otras con corticoides tópicos. Tratar tiña con corticoides lleva a la aparición de la tiña incógnita, que enmascara el cuadro y favorece su extensión.


Este proyecto implementa un **pipeline de procesamiento de imágenes clásico, sin aprendizaje automático**, que recibe una fotografía clínica de una lesión y decide si es **tiña corporis** en estado de placa anular (placa rojiza, descamativa y pruriginosa, generalmente con aclaramiento central) o una **lesión inflamatoria** (eczema o dermatitis de contacto, agrupadas como *Otras*).

La idea central: aislar la lesión, dividirla en **borde** y **centro**, y comparar ambas zonas entre sí y con la piel en la **zona perilesional** de la misma foto, bajo la hipótesis de que en la tiña el borde activo descamado presenta mayor variabilidad que el centro, mientras que en las lesiones inflamatorias la placa es más homogénea.
La comparación se realiza a partir de la extracción de descriptores que surgen del análisis de texturas por GLCM (Gray-Level Co-Ocurrence Matrix) y LBP (Local Binary Patterns).

## Datos

- **SkinDisNet** ([Mendeley Data](https://data.mendeley.com/datasets/yj3md44hxg/2)): fotografías clínicas con smartphone de dos hospitales de Bangladesh. Se usaron las clases tiña corporis, eczema y dermatitis de contacto. La partición Entrenamiento/Testeo se hizo verificando que ningún paciente quedara en ambos conjuntos.
- **Imágenes externas**: imágenes de tiña corporis de PubMed Central, DermNet y otras fuentes de referencia médica.
- Las ROI generadas se guardan en `datos/ROIs/` con el prefijo `roi_`.

## Herramientas
El sistema se implementó en Python, usando NumPy, OpenCV, SciPy, scikit-image, pandas y Matplotlib para la lógica y el pipeline de procesamiento, y PyQt6 y pyqtgraph para la interfaz gráfica.

## Pipeline

| Etapa | Nombre | Entrada → salida | Módulo |
|---|---|---|---|
| 0 | ROI y normalización | Foto RGB → ROI RGB 512×512 | `e0_normalizacion_roi.py` |
| 1 | Espacio de color | ROI RGB → canales L\*, a\*, b\* y máscara válida | `e1_mascara_lab.py` |
| 2 | Mejoramiento | Mediana 7×7, ecualización de a\*, eritema relativo a la piel; L\* sin filtrar para textura | `e2_mejoramiento.py` |
| 3 | Segmentación | ROI → máscara binaria rellena de la lesión (GrabCut con prior geométrico + corrección manual de ser necesario) | `e3_segmentacion.py` |
| 4 | Morfología | Máscara → zonas borde, centro y perilesional | `e4_morfologia.py` |
| 5 | Descriptores | Textura GLCM y LBP sobre L\* por zona; descriptores relativos (borde − centro, lesión − perilesional) | `e5_descriptores.py` y `e5_lbp.py` |
| 6 | Decisión | Regla de votos con 3 descriptores y umbrales aprendidos | `e6_clasificacion.py`|
| 7 | Testeo | Evaluación sobre el conjunto de Testeo | — |

### Decisiones de diseño principales

- **ROI semiautomática:** el operador marca un rectángulo (trasladable, redimensionable y rotable); se remuestrea a 512×512 con `cv2.INTER_AREA`. La geometría queda registrada en `rois.csv`.
- **CIELAB en punto flotante:** a\* mide el eritema; L\* se reserva para la textura.
- **Segmentación:** se compararon Otsu, Li, Triangle, umbral adaptativo, k-means, umbral relativo a la piel, crecimiento de regiones y Chan-Vese. Ninguno funcionó de forma confiable con SkinDisNet, por lo que se eligió **GrabCut** inicializado con la distancia elíptica al centro de la ROI.
- **Zonas:** el ancho del borde es `w = max(3, 0,30·r_eq)`; centro = lesión erosionada, borde = lesión \ centro, perilesional = dilatación de 20 px de la lesión ∩ piel.
- **Descriptores relativos:** como casi todas las TC provienen de fuentes externas y las *Otras* de SkinDisNet, se priorizan descriptores que comparan zonas de la misma foto (borde − centro, lesión − perilesional) para reducir el sesgo de cámara e iluminación.
- **Regla de decisión:** se ordenan los descriptores relativos (los que comparan zonas de la misma foto) por capacidad de separación entre TC y Otras, medida como |AUC − 0,5|. Se eligen los N mejores, descartando los que tengan correlación de Spearman > 0,8 con alguno ya elegido. Cada descriptor vota según un umbral que maximiza sensibilidad + especificidad en Entrenamiento, y la imagen se clasifica como tiña si más de la mitad vota tiña. Se usó **N = 3** (con N = 3 se obtuvo el mejor desempeño en Entrenamiento: 25/25).

Descriptores finales (guardados en `clasificador.json`):

| Descriptor | Vota tiña si es… | Umbral | AUC |
|---|---|---|---|
| `dif_BC__lbp_9` | menor a | −0,000906 | 0,087 |
| `dif_LP__lbp_7` | menor a | −0,003686 | 0,113 |
| `dif_LP__glcm_correlation` | menor a | −0,023342 | 0,160 |

## Resultados

Las filas de las matrices son la clase real y las columnas la predicción.

**Entrenamiento (25 imágenes: 10 TC y 15 Otras).** Como estas imágenes también se usaron para elegir descriptores y umbrales, el resultado es una **cota optimista**: 25 aciertos sobre 25.

| | no tiña | tiña |
|---|---|---|
| **Otras** | 15 | 0 |
| **TC** | 0 | 10 |

**Testeo (8 imágenes: 3 TC y 5 Otras).** Sensibilidad 66,7 % (2 de 3 TC) y especificidad 100 % (5 de 5 Otras).

| | no tiña | tiña |
|---|---|---|
| **Otras** | 5 | 0 |
| **TC** | 1 | 2 |

El único error es un falso negativo (`TC_08_wmc.jpg`), que es el error más grave en este contexto. La imagen tiene bordes difusos y menos iluminación que las de Entrenamiento. Con solo 3 TC en Testeo, estos porcentajes son muy poco estables.

## Estructura del repositorio
El repositorio está organizado en módulos que separan la interfaz gráfica, las etapas de procesamiento, los datos de entrada y los resultados generados por el pipeline.

```
.
├── main.py                          # Punto de entrada del sistema 
│
├── gui/                             # Componentes de la interfaz gráfica (PyQt6)
│   ├── __init__.py
│   ├── main_window.py               # Ventana principal
│   └── editor_mascara.py            # Editor de máscaras de segmentación automática (agregar máscara / borrar fondo)
│
├── etapas/                          # Implementación de las etapas del pipeline
│   ├── __init__.py
│   ├── e0_normalizacion_roi.py      # Normalización de ROIs a 512x512 (centrada en la lesión con algo de borde de piel sana)
│   ├── e1_mascara_lab.py            # Transformación al espacio CIELAB
│   ├── e2_mejoramiento.py           # Mejoramiento de imágenes
│   ├── e3_segmentacion.py           # Segmentación de lesiones
│   ├── e4_morfologia.py             # Análisis morfológico y delimitación de regiones
│   ├── e5_descriptores.py           # Extracción de descriptores texturales (GLCM)
│   ├── e5_lbp.py                    # Extracción de descriptores LBP      
│   ├── e6_clasificacion.py          # Clasificación 
│   └── análisis_para_elegir_descriptores.ipynb # Se detalla el análisis para la detección de descriptores.
│
├── datos/
│   ├── Dataset_reducido/            # Imágenes de TC y Otras (eczema, dermatitis de contacto)
│   │   ├── Entrenamiento/           # Imágenes utilizadas para la etapa de Entrenamiento (25).
│   │   └── Testeo/                  # Imágenes utilizadas para la etapa de Testeo (8)
│   │
│   └── ROIs/                        # Regiones de interés normalizadas a 512 × 512 píxeles, alrededor de la lesión con una parte de piel sana.
│
├── resultados/                      # Se genera al correr el pipeline.
│   ├── Etapa_1/                     # Resultados de la transformación a CIELAB
│   ├── Etapa_2/                     # Resultados del mejoramiento
│   ├── Etapa_3/                     # Resultados de la segmentación
│   ├── Etapa_4/                     # Resultados morfológicos y archivos .npz a ser utilizados por la Etapa 5
│   └── Etapa_5/                     # Resultados del análisis textural y extracción de descriptores
│       ├── clasificador.json        # Configuración del clasificador
│       ├── descriptores_lbp.csv     # Descriptores LBP
│       └── descriptores_texturales.csv  # Descriptores GLCM
│
├── rois.csv                         # Geometría de las ROI (centro, ancho, alto y ángulo)
├── requirements.txt                 # Dependencias del proyecto
└── .gitignore                       # Archivos excluidos del control de versiones
```


## Instalación
### 1. Clonar en la terminal
```bash
git clone https://github.com/suthurralt/PSIB2026Q2-TPI-Grupo9-DelCarril-Ecobar-Uthurralt.git
cd PSIB2026Q2-TPI-Grupo9-DelCarril-Ecobar-Uthurralt
```
### 2. Crear un entorno virtual propio
```bash
python -m venv env
```
### 3. Activarlo

Windows (PowerShell):
```bash
.\env\Scripts\Activate.ps1
```
Windows (cmd):
```bash
env\Scripts\activate.bat
```

Linux / macOS:
```bash
source env/bin/activate
```

### 4. Instalar dependencias
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```
## Requisitos principales
```
numpy==2.5.4
scipy==1.18.1
pandas==3.1.0
matplotlib==3.11.2
opencv-python==5.0.0.93
scikit-image==0.26.0
PyQt6==6.12.0
pyqtgraph==0.14.0
Python 3.12 o superior (desarrollado y probado con Python 3.14.6)
```

## Modo de ejecución
Ejecutar en la terminal, desde la raíz del repositorio con el entorno activado:
```bash
python main.py
```

En la interfaz:

1. Cargar una carpeta de imágenes en `1. Cargar Carpeta de Datos` y marcar las que se quieren analizar.
2. Marcar la ROI de cada imagen (Etapa 0) manipulando el rectángulo (permite rotar y cambiar su tamaño) y recorrer el pipeline etapa por etapa, presionando el botón de `Ir a ...` para avanzar a la etapa siguiente.
3. En la etapa de segmentación, si la segmentación automática no es correcta, retocarla con el editor de máscara (pincel y borrador de radio 12px).
4. Al finalizar la Etapa 4, que divide la lesión en centro, borde y zona perilesional, la interfaz ejecuta `e6_clasificacion.clasificar()` y muestra el veredicto y la cantidad de votos a favor de tiña.

Por cada imagen se guarda un `.npz` con la ROI, los canales L\*, a\* y a\* filtrado, las cuatro máscaras (lesión, borde, centro, perilesional), el ancho del borde y la referencia de piel. Es con estos datos que se avanza hacia las etapas 5 y 6 de descriptores y clasificación.

## Limitaciones

- **Dependencia del operador:** la ROI y de ser necesario, la máscara son manuales.
- **Deformación al normalizar:** el remuestreo a 512×512 no conserva la relación de aspecto, por lo que se descartó la circularidad como descriptor.
- **Supuesto de lesión centrada:** el prior de GrabCut falla con lesiones excéntricas, múltiples o que ocupan toda la ROI; además se descartan lesiones satélite.
- **Lesiones difusas:** en algunos eczemas la división borde/centro pierde sentido.
- **Ancho de borde fijo relativo** (30 % de `r_eq`), mientras que el anillo activo real de una tiña varía.
- **Conjunto de datos pequeño**.
-  **Sesgo de origen:** casi todas las TC provienen de fuentes externas y todas las *Otras* de SkinDisNet, por lo que un descriptor podría separar las clases por diferencias de cámara, iluminación o resolución y no por la textura de la lesión. Los descriptores relativos (comparación entre zonas de la misma foto) mitigan este problema pero no lo eliminan. Por eso los resultados, en especial los de Testeo, deben considerarse optimistas.
- **Descriptores no validados por la literatura:** no se encontraron trabajos específicos sobre descriptores para tiña corporis, así que se eligieron empíricamente con pocas imágenes.

## Trabajo futuro

- Umbralización de descriptores con técnicas más avanzadas (k-means, árboles de decisión).
- Segmentación automática más robusta, para depender menos de la ROI y la corrección manual.
- Más imágenes de entrenamiento.
- Analizar nuevos descriptores (por ejemplo, circularidad, conservando la relación de aspecto).
- Encontrar descriptores validados por la literatura para tiña corporis. No se encontraron trabajos suficientemente específicos, por lo que los descriptores se eligieron comparando las imágenes de Entrenamiento. Ese método depende mucho del conjunto de entrenamiento y puede variar de un dataset a otro, sobre todo con pocas imágenes.

> El código fue desarrollado con asistencia de IA (principalmente Claude).