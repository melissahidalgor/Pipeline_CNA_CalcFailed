import pandas as pd
import numpy as np

class PIACEFeatureEngineering:
    """
    Módulo modular de Feature Engineering que replica las reglas de negocio
    del código VB.NET usando los nombres de Tags reales de PI.
    """
    def __init__(self, thresholds_dict=None):
        # Parámetros leídos de ModuleDB en el código VB.NET
        self.thresholds = thresholds_dict or {
            'TempMin': -5.0, 'TempMax': 50.0, 'TempMDif': 0.3,
            'HumedadMin': 0.0, 'HumedadMax': 100.0, 'HumedadMDif': 2.0,
            'PresionMin': 0.92, 'PresionMax': 1.1, 'PresionMDif': 0.005,
            'G1_FD_INTENS_Min': 40.0,
            'G2_FD_INTENS_Min': 40.0,
            'G3_FD_INTENS_Min': 40.0,
            'G4_FD_INTENS_Min': 40.0
        }

    def create_quality_features(self, df):
        """Genera banderas binarias: 1 si la entrada es válida, 0 si es nula/inválida."""
        df_feats = df.copy()
        for col in df.columns:
            # Detecta si el resampleo dejó valores nulos (equivale a BAD/No Numérico en PI)
            df_feats[f'{col}_IsValid'] = df[col].notna().astype(int)
        return df_feats

    def create_inter_sensor_features(self, df_feats):
        """
        Replica la lógica de GetAmbiental: evalúa discrepancias y rangos entre
        las mediciones triples (A, B, C) de Temperatura, Humedad y Presión.
        """
        # 1. Temperatura (Temp_A, Temp_B, Temp_C)
        if all(c in df_feats.columns for c in ['Temp_A', 'Temp_B', 'Temp_C']):
            df_feats['Diff_Temp_AB'] = (df_feats['Temp_A'] - df_feats['Temp_B']).abs()
            df_feats['Diff_Temp_AC'] = (df_feats['Temp_A'] - df_feats['Temp_C']).abs()
            df_feats['Diff_Temp_BC'] = (df_feats['Temp_B'] - df_feats['Temp_C']).abs()
            
            df_feats['Temp_Exceeds_Dif'] = (
                (df_feats['Diff_Temp_AB'] > self.thresholds['TempMDif']) &
                (df_feats['Diff_Temp_AC'] > self.thresholds['TempMDif']) &
                (df_feats['Diff_Temp_BC'] > self.thresholds['TempMDif'])
            ).astype(int)

            df_feats['Temp_Out_of_Range'] = (
                (df_feats['Temp_A'] < self.thresholds['TempMin']) | (df_feats['Temp_A'] > self.thresholds['TempMax']) |
                (df_feats['Temp_B'] < self.thresholds['TempMin']) | (df_feats['Temp_B'] > self.thresholds['TempMax']) |
                (df_feats['Temp_C'] < self.thresholds['TempMin']) | (df_feats['Temp_C'] > self.thresholds['TempMax'])
            ).astype(int)

        # 2. Humedad (Hum_A, Hum_B, Hum_C)
        if all(c in df_feats.columns for c in ['Hum_A', 'Hum_B', 'Hum_C']):
            df_feats['Diff_Hum_AB'] = (df_feats['Hum_A'] - df_feats['Hum_B']).abs()
            df_feats['Diff_Hum_AC'] = (df_feats['Hum_A'] - df_feats['Hum_C']).abs()
            df_feats['Diff_Hum_BC'] = (df_feats['Hum_B'] - df_feats['Hum_C']).abs()
            
            df_feats['Hum_Exceeds_Dif'] = (
                (df_feats['Diff_Hum_AB'] > self.thresholds['HumedadMDif']) &
                (df_feats['Diff_Hum_AC'] > self.thresholds['HumedadMDif']) &
                (df_feats['Diff_Hum_BC'] > self.thresholds['HumedadMDif'])
            ).astype(int)

            df_feats['Hum_Out_of_Range'] = (
                (df_feats['Hum_A'] < self.thresholds['HumedadMin']) | (df_feats['Hum_A'] > self.thresholds['HumedadMax'])
            ).astype(int)

        # 3. Presión (Pres_A, Pres_B, Pres_C)
        if all(c in df_feats.columns for c in ['Pres_A', 'Pres_B', 'Pres_C']):
            df_feats['Diff_Pres_AB'] = (df_feats['Pres_A'] - df_feats['Pres_B']).abs()
            df_feats['Diff_Pres_AC'] = (df_feats['Pres_A'] - df_feats['Pres_C']).abs()
            df_feats['Diff_Pres_BC'] = (df_feats['Pres_B'] - df_feats['Pres_C']).abs()
            
            df_feats['Pres_Exceeds_Dif'] = (
                (df_feats['Diff_Pres_AB'] > self.thresholds['PresionMDif']) &
                (df_feats['Diff_Pres_AC'] > self.thresholds['PresionMDif']) &
                (df_feats['Diff_Pres_BC'] > self.thresholds['PresionMDif'])
            ).astype(int)

            df_feats['Pres_Out_of_Range'] = (
                (df_feats['Pres_A'] < self.thresholds['PresionMin']) | (df_feats['Pres_A'] > self.thresholds['PresionMax'])
            ).astype(int)

        return df_feats

    # No se usa en esta aplicacion
    def create_flame_and_operational_features(self, df_feats):
        """
        Replica GetLlamaTGx para detectar si las turbinas de gas están encendidas o apagadas.
        """
        for i in range(1, 5):
            flame_cols = [f'G{i}_DFlama1', f'G{i}_DFlama2', f'G{i}_DFlama3', f'G{i}_DFlama4']
            thresh = self.thresholds.get(f'G{i}_FD_INTENS_Min', 5.0)
            
            if all(c in df_feats.columns for c in flame_cols):
                df_feats[f'G{i}_Flama_Active'] = (
                    (df_feats[flame_cols[0]] > thresh) |
                    (df_feats[flame_cols[1]] > thresh) |
                    (df_feats[flame_cols[2]] > thresh) |
                    (df_feats[flame_cols[3]] > thresh)
                ).astype(int)

        return df_feats

    def create_domain_ratios_and_counters(self, df_feats):
        """
        Detecta condiciones propensas a división entre cero o valores negativos en contadores.
        """
        # Banderas de Densidad de Gas (Evita división entre cero en Poder Calorífico)
        if 'DENSIDAD_GAS' in df_feats.columns:
            df_feats['DENSIDAD_GAS_Zero_Flag'] = (df_feats['DENSIDAD_GAS'] <= 0).astype(int)
        if 'DENSIDAD_GAS_TAM' in df_feats.columns:
            df_feats['DENSIDAD_GAS_TAM_Zero_Flag'] = (df_feats['DENSIDAD_GAS_TAM'] <= 0).astype(int)

        # Monitoreo de saltos negativos en contadores acumulados de energía
        energy_cols = [c for c in df_feats.columns if '_ACTIVA_' in c or '_REACTIVA_' in c]
        for col in energy_cols:
            df_feats[f'{col}_Is_Negative'] = (df_feats[col] < 0).astype(int)

        return df_feats

    def transform(self, df_shifted):
        """Ejecuta el flujo completo de transformación manteniendo la modularidad."""
        df_proc = self.create_quality_features(df_shifted)
        df_proc = self.create_inter_sensor_features(df_proc)
        df_proc = self.create_flame_and_operational_features(df_proc)
        df_proc = self.create_domain_ratios_and_counters(df_proc)
        
        # Rellenar nulos con 0 para que los clasificadores (Random Forest) puedan leer la matriz
        df_proc.fillna(0, inplace=True)
        return df_proc
# --- Ejemplo de Uso ---
# fe = PIACEFeatureEngineering()
# X_processed = fe.transform(df_final_features)