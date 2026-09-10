import json
import os
import pandas as pd
import numpy as np

class PIACEFeatureEngineering:
    """
    Módulo modular de Feature Engineering que replica las reglas de negocio
    y umbrales operativos de ModuleDB para la Central Termoeléctrica.
    """
    def __init__(self, config_path="config/moduledb.json", thresholds_dict=None):
        if thresholds_dict is not None:
            self.thresholds = thresholds_dict
        else:
            self.thresholds = self._load_config(config_path)

    def _load_config(self, config_path):
        """Carga y aplana los umbrales de configuración desde un archivo JSON."""
        defaults = {
            'TempMin': -5.0, 'TempMax': 50.0, 'TempMDif': 0.3,
            'HumedadMin': 0.0, 'HumedadMax': 100.0, 'HumedadMDif': 2.0,
            'PresionMin': 0.92, 'PresionMax': 1.1, 'PresionMDif': 0.005,
            'G1_FD_INTENS_Min': 40.0, 'G2_FD_INTENS_Min': 40.0,
            'G3_FD_INTENS_Min': 40.0, 'G4_FD_INTENS_Min': 40.0
        }
        if not os.path.exists(config_path):
            return defaults

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            return {
                'TempMin': float(cfg.get('temperatura', {}).get('min', -5.0)),
                'TempMax': float(cfg.get('temperatura', {}).get('max', 50.0)),
                'TempMDif': float(cfg.get('temperatura', {}).get('max_dif', 0.3)),
                'HumedadMin': float(cfg.get('humedad', {}).get('min', 0.0)),
                'HumedadMax': float(cfg.get('humedad', {}).get('max', 100.0)),
                'HumedadMDif': float(cfg.get('humedad', {}).get('max_dif', 2.0)),
                'PresionMin': float(cfg.get('presion', {}).get('min', 0.92)),
                'PresionMax': float(cfg.get('presion', {}).get('max', 1.1)),
                'PresionMDif': float(cfg.get('presion', {}).get('max_dif', 0.005)),
                'G1_FD_INTENS_Min': float(cfg.get('turbinas', {}).get('G1_FD_INTENS_Min', 40.0)),
                'G2_FD_INTENS_Min': float(cfg.get('turbinas', {}).get('G2_FD_INTENS_Min', 40.0)),
                'G3_FD_INTENS_Min': float(cfg.get('turbinas', {}).get('G3_FD_INTENS_Min', 40.0)),
                'G4_FD_INTENS_Min': float(cfg.get('turbinas', {}).get('G4_FD_INTENS_Min', 40.0))
            }
        except Exception:
            return defaults

    def create_quality_features(self, df):
        """
        Genera una variable de estado de calidad (QualityState) de 3 estados para cada sensor:
          0: Correcto / Normal (Dato numérico presente y sin eventos BAD en la ventana)
          1: Error de Hardware / BAD (Al menos una lectura BAD o inválida en los datos crudos)
          2: Sin lecturas / Vacío (Ventana vacía / pérdida de telemetría en esos 5 minutos)
        """
        df_feats = df.copy()
        
        # Identificar columnas base de valor (excluyendo metadatos existentes)
        val_cols = [
            c for c in df.columns 
            if not c.endswith('_Has_Bad') and not c.endswith('_QualityState') and not c.endswith('_IsValid')
        ]

        has_bad_cols_to_drop = []
        for col in val_cols:
            bad_col = f'{col}_Has_Bad'
            if bad_col in df_feats.columns:
                has_bad = df_feats[bad_col] == 1
                is_na = df_feats[col].isna()

                # Vectorizado: 1 si hubo BAD, 2 si es NA sin BAD, 0 si todo correcto
                state = np.where(has_bad, 1, np.where(is_na, 2, 0))
                df_feats[f'{col}_QualityState'] = state
                has_bad_cols_to_drop.append(bad_col)
            else:
                # Respaldo si no vino _Has_Bad: 2 si es NA, 0 si tiene valor
                df_feats[f'{col}_QualityState'] = np.where(df_feats[col].isna(), 2, 0)

        # Eliminar las columnas intermedias _Has_Bad para evitar redundancia en la matriz de features
        df_feats.drop(columns=has_bad_cols_to_drop, inplace=True, errors='ignore')

        return df_feats

    def create_inter_sensor_features(self, df_feats):
        """
        Replica la lógica de GetAmbiental: evalúa discrepancias y rangos entre
        las mediciones triples (A, B, C) de Temperatura, Humedad y Presión
        usando los umbrales de ModuleDB.
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
                (df_feats['Hum_A'] < self.thresholds['HumedadMin']) | (df_feats['Hum_A'] > self.thresholds['HumedadMax']) |
                (df_feats['Hum_B'] < self.thresholds['HumedadMin']) | (df_feats['Hum_B'] > self.thresholds['HumedadMax']) |
                (df_feats['Hum_C'] < self.thresholds['HumedadMin']) | (df_feats['Hum_C'] > self.thresholds['HumedadMax'])
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
                (df_feats['Pres_A'] < self.thresholds['PresionMin']) | (df_feats['Pres_A'] > self.thresholds['PresionMax']) |
                (df_feats['Pres_B'] < self.thresholds['PresionMin']) | (df_feats['Pres_B'] > self.thresholds['PresionMax']) |
                (df_feats['Pres_C'] < self.thresholds['PresionMin']) | (df_feats['Pres_C'] > self.thresholds['PresionMax'])
            ).astype(int)

        return df_feats

    def create_flame_and_operational_features(self, df_feats):
        """
        Replica GetLlamaTGx para detectar si las turbinas de gas están encendidas o apagadas.
        """
        for i in range(1, 5):
            flame_cols = [f'G{i}_DFlama1', f'G{i}_DFlama2', f'G{i}_DFlama3', f'G{i}_DFlama4']
            thresh = self.thresholds.get(f'G{i}_FD_INTENS_Min', 40.0)

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
        energy_cols = [
            c for c in df_feats.columns 
            if ('_ACTIVA_' in c or '_REACTIVA_' in c) 
            and not c.endswith('_Has_Bad') 
            and not c.endswith('_IsValid')
            and not c.endswith('_QualityState')
        ]
        for col in energy_cols:
            df_feats[f'{col}_Is_Negative'] = (df_feats[col] < 0).astype(int)

        return df_feats

    def transform(self, df, fill_na=True):
        """Ejecuta el flujo completo de transformación manteniendo la modularidad."""
        df_proc = self.create_quality_features(df)
        df_proc = self.create_inter_sensor_features(df_proc)
        df_proc = self.create_flame_and_operational_features(df_proc)
        df_proc = self.create_domain_ratios_and_counters(df_proc)

        if fill_na:
            # Rellenar nulos con 0 para que clasificadores (Random Forest) puedan leer la matriz
            df_proc.fillna(0, inplace=True)

        return df_proc
