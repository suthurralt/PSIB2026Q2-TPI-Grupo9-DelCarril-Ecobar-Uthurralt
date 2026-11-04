import json
import numpy as np
from pathlib import Path

class EvaluadorTinea:
    def __init__(self, ruta_json_referencia="referencia_tinea.json"):
        """
        Carga la configuración y los rangos robustos de referencia desde el JSON.
        """
        self.ruta_json = Path(ruta_json_referencia)
        self.cargar_referencias()
        
    def cargar_referencias(self):
        if not self.ruta_json.exists():
            raise FileNotFoundError(f"No se encontró el archivo de referencia en: {self.ruta_json}")
            
        with open(self.ruta_json, "r", encoding="utf-8") as f:
            self.datos_ref = json.load(f)
            
        self.version = self.datos_ref.get("version", "desconocida")
        self.umbral = self.datos_ref.get("umbral_aceptacion", 0.75)
        self.features_ref = self.datos_ref.get("features", {})
        
    def evaluar_features(self, features_nueva_imagen):
        """
        Evalúa un diccionario de características relativas obtenidas de una nueva imagen.
        
        Param:
            features_nueva_imagen (dict): p. ej. {
                'dif_borde_centro__contrast': 0.08,
                'dif_borde_peri__homogeneity': -0.03,
                ...
            }
            
        Returns:
            dict: Informe detallado con score global, clasificación y estado por métrica.
        """
        cumplidas = 0
        total_evaluadas = 0
        detalle = {}
        
        for feat_name, limites in self.features_ref.items():
            if feat_name in features_nueva_imagen:
                val = features_nueva_imagen[feat_name]
                lo = limites["lo"]
                hi = limites["hi"]
                
                # Verificación si el valor de la nueva foto está dentro del rango [lo, hi]
                esta_dentro = bool(lo <= val <= hi)
                
                if esta_dentro:
                    cumplidas += 1
                total_evaluadas += 1
                
                detalle[feat_name] = {
                    "valor_medido": round(val, 5),
                    "rango_referencia": (round(lo, 5), round(hi, 5)),
                    "cumple_rango": esta_dentro
                }
            else:
                # Caso en que una métrica de referencia no fue calculada en la nueva imagen
                detalle[feat_name] = {
                    "valor_medido": None,
                    "rango_referencia": (limites["lo"], limites["hi"]),
                    "cumple_rango": False,
                    "error": "Métrica faltante en la imagen evaluada"
                }
        
        # Cálculo del porcentaje de coincidencia
        score = (cumplidas / total_evaluadas) if total_evaluadas > 0 else 0.0
        es_compatible = score >= self.umbral
        
        return {
            "es_compatible": es_compatible,
            "score_porcentaje": round(score * 100, 2),
            "umbral_requerido": round(self.umbral * 100, 2),
            "metricas_cumplidas": f"{cumplidas}/{total_evaluadas}",
            "detalle_por_metrica": detalle,
            "nota": self.datos_ref.get("nota", "")
        }


# ==========================================
#  EJEMPLO DE USO / PRUEBA DEL SCRIPT
# ==========================================
if __name__ == "__main__":
    # 1. Instanciar el evaluador con el JSON generado por tu notebook
    try:
        evaluador = EvaluadorTinea("referencia_tinea.json")
        
        # 2. Simular métricas extraídas de una lesión de un paciente nuevo
        features_paciente_ejemplo = {
            "dif_borde_centro__contrast": 0.083,       # Cae en [-0.393, 0.560] -> Sí
            "dif_borde_peri__homogeneity": -0.034,     # Cae en [-0.056, -0.012] -> Sí
            "dif_borde_peri__energy": -0.013,          # Cae en [-0.075, 0.048] -> Sí
            "dif_borde_centro__lbp_entropia": -0.004   # Cae en [-0.051, 0.042] -> Sí
        }

        # 3. Correr la evaluación
        resultado = evaluador.evaluar_features(features_paciente_ejemplo)

        # 4. Mostrar informe en consola
        print("\n--- RESULTADO DE EVALUACIÓN DE COMPATIBILIDAD ---")
        print(f"¿Compatible con Tinea Corporis?: {'SÍ' if resultado['es_compatible'] else 'NO'}")
        print(f"Porcentaje de coincidencia: {resultado['score_porcentaje']}% (Umbral: {resultado['umbral_requerido']}%)")
        print(f"Métricas en rango: {resultado['metricas_cumplidas']}\n")
        
        print("Detalle por característica:")
        for feat, info in resultado["detalle_por_metrica"].items():
            estado = "✓ OK" if info["cumple_rango"] else "✗ FUERA"
            print(f" - {feat}: {info['valor_medido']} en {info['rango_referencia']} -> {estado}")

    except Exception as e:
        print(f"Error al ejecutar la evaluación: {e}")