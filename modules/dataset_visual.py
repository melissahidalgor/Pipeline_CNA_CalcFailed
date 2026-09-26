import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class DatasetVisualizer:
    """
    Módulo de visualización y exploración preliminar de datos (EDA)
    para el análisis de Capacidad Neta Disponible (CNA).
    Permite diagnosticar la estructura, calidad, atípicos y comparar
    el comportamiento de las señales antes y después del resampleo.
    """
    def __init__(self, config_path="config/moduledb.json"):
        self.config_path = config_path
        self.thresholds = self._load_thresholds(config_path)
        # Configuración estética base para gráficos
        sns.set_theme(style="whitegrid", palette="muted")
        plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans', 'Arial', 'Liberation Sans', 'sans-serif'],'font.size': 10})
      
    def _load_thresholds(self, config_path):
        """Carga umbrales de alarma desde el archivo JSON de configuración."""
        defaults = {
            'TempMDif': 0.3, 'HumedadMDif': 2.0, 'PresionMDif': 0.005,
            'TempMin': -5.0, 'TempMax': 50.0,
            'PresionMin': 0.92, 'PresionMax': 1.1
        }
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    cfg = json.load(f)
                return {
                    'TempMDif': float(cfg.get('temperatura', {}).get('max_dif', 0.3)),
                    'HumedadMDif': float(cfg.get('humedad', {}).get('max_dif', 2.0)),
                    'PresionMDif': float(cfg.get('presion', {}).get('max_dif', 0.005)),
                    'TempMin': float(cfg.get('temperatura', {}).get('min', -5.0)),
                    'TempMax': float(cfg.get('temperatura', {}).get('max', 50.0)),
                    'PresionMin': float(cfg.get('presion', {}).get('min', 0.92)),
                    'PresionMax': float(cfg.get('presion', {}).get('max', 1.1)),
                }
            except Exception:
                return defaults
        return defaults

    def summarize_dataset(self, df, name="Dataset"):
        """
        Genera un reporte estadístico y estructural detallado:
        tipo de dato, conteo no nulo, porcentaje de vacíos, duplicados,
        estadísticas descriptivas y conteo de valores atípicos (IQR).
        """
        records = []
        n_rows = len(df)

        for col in df.columns:
            series = df[col]
            dtype = str(series.dtype)
            null_count = int(series.isna().sum())
            null_pct = round((null_count / n_rows) * 100, 2)
            non_null = n_rows - null_count
            n_unique = int(series.nunique())

            # Estadísticas si es numérico
            num_s = pd.to_numeric(series, errors='coerce')
            if num_s.notna().sum() > 0:
                mean_val = round(float(num_s.mean()), 4)
                std_val = round(float(num_s.std()), 4)
                min_val = round(float(num_s.min()), 4)
                q25 = round(float(num_s.quantile(0.25)), 4)
                median_val = round(float(num_s.median()), 4)
                q75 = round(float(num_s.quantile(0.75)), 4)
                max_val = round(float(num_s.max()), 4)

                # Detección de atípicos por Rango Intercuartílico (IQR)
                iqr = q75 - q25
                lower_bound = q25 - 1.5 * iqr
                upper_bound = q75 + 1.5 * iqr
                outliers_count = int(((num_s < lower_bound) | (num_s > upper_bound)).sum())
            else:
                mean_val = std_val = min_val = q25 = median_val = q75 = max_val = np.nan
                outliers_count = 0

            records.append({
                'Columna': col,
                'Tipo': dtype,
                'No_Nulos': non_null,
                'Nulos': null_count,
                '%_Nulos': null_pct,
                'Valores_Unicos': n_unique,
                'Media': mean_val,
                'Desv_Std': std_val,
                'Min': min_val,
                'Q1_25%': q25,
                'Mediana': median_val,
                'Q3_75%': q75,
                'Max': max_val,
                'Atipicos_IQR': outliers_count
            })

        summary_df = pd.DataFrame(records)
        return summary_df

    def compare_pre_post(self, df_raw, df_resampled):
        """
        Genera una tabla comparativa ejecutiva de las características
        del dataset ANTES vs. DESPUÉS del resampleo.
        """
        # Calcular timestamps en crudo (usando Time_CNA como referencia temporal)
        raw_rows = len(df_raw)
        raw_cols = len(df_raw.columns)
        res_rows = len(df_resampled)
        res_cols = len(df_resampled.columns)

        time_range_res = f"{df_resampled.index.min()} a {df_resampled.index.max()}"

        comp_data = {
            'Métrica': [
                'Total de Registros (Filas)',
                'Total de Columnas',
                'Tipo de Intervalo Temporal',
                'Rango de Fechas',
                'Alineación entre Sensores',
                'Trazabilidad de Errores BAD'
            ],
            'Antes (Dataset Crudo)': [
                f"{raw_rows:,} registros dispersos",
                f"{raw_cols} columnas (pares Time/Valor)",
                "Irregular (por excepción, segundos/minutos)",
                "Desfasado entre sensores",
                "Desalineado (múltiples columnas de tiempo)",
                "Textos de error mezclados ('Bad', 'Invalid Data')"
            ],
            'Después (Resampleado a 5 min)': [
                f"{res_rows:,} intervalos regulares",
                f"{res_cols} señales unificadas",
                "Regular y continuo (exactamente 5 minutos)",
                time_range_res,
                "Sincronizado en un solo Timestamp común",
                "Separado en Estados de Calidad (0, 1, 2)"
            ]
        }
        return pd.DataFrame(comp_data)

    def plot_resampling_comparison(self, df_raw, df_resampled, tag="Temp_A", sample_hours=8, start_time=None):
        """
        Visualización 1: Compara la serie temporal ANTES y DESPUÉS del resampleo.
        Muestra la dispersión de datos por excepción frente a la grilla regular de 5 min.
        """
        t_col = f"Time_{tag}"
        v_col = tag

        # Extraer ventana de tiempo para visualización limpia
        if start_time is None:
            start_dt = pd.to_datetime('2026-07-01 00:00:00')
        else:
            start_dt = pd.to_datetime(start_time)
        end_dt = start_dt + pd.Timedelta(hours=sample_hours)

        # 1. Datos Crudos
        df_sub_raw = df_raw[[t_col, v_col]].dropna(subset=[t_col]).copy()
        dt_raw = pd.to_datetime(df_sub_raw[t_col], format='%d-%b-%y %H:%M:%S', errors='coerce')
        df_sub_raw['dt'] = dt_raw
        df_sub_raw['val'] = pd.to_numeric(df_sub_raw[v_col], errors='coerce')
        mask_raw = (df_sub_raw['dt'] >= start_dt) & (df_sub_raw['dt'] <= end_dt)
        raw_window = df_sub_raw[mask_raw].sort_values(by='dt')

        # 2. Datos Resampleados
        res_window = df_resampled.loc[(df_resampled.index >= start_dt) & (df_resampled.index <= end_dt), v_col]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

        # Gráfica Cruda
        ax1.plot(raw_window['dt'], raw_window['val'], color='#e74c3c', marker='.', linestyle='-', alpha=0.6, markersize=5, label='Lecturas Crudas (Muestreo irregular por excepción)')
        ax1.set_title(f"ANTES: Señal Cruda '{tag}' (Registros asíncronos y dispersos)", fontsize=12, fontweight='bold')
        ax1.set_ylabel("Valor")
        ax1.legend(loc='upper right')
        ax1.grid(True, linestyle='--', alpha=0.6)

        # Gráfica Resampleada
        ax2.plot(res_window.index, res_window.values, color='#2980b9', marker='o', linestyle='-', markersize=4, label='Serie Resampleada (Promedio regular cada 5 min)')
        ax2.set_title(f"DESPUÉS: Señal '{tag}' en Grilla Regular de 5 Minutos (Sincronizada con CNA)", fontsize=12, fontweight='bold')
        ax2.set_xlabel("Tiempo")
        ax2.set_ylabel("Valor Promedio")
        ax2.legend(loc='upper right')
        ax2.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        return fig, (ax1, ax2)

    def plot_quality_distribution(self, X_features, top_n=20):
        """
        Visualización 2: Gráfico de barras apiladas con la distribución de los 3 estados de calidad:
          Estado 0: Correcto / Normal (Verde)
          Estado 1: Falla de Instrumento / BAD (Rojo)
          Estado 2: Sin Lecturas / Vacío (Naranja)
        """
        qs_cols = [c for c in X_features.columns if c.endswith('_QualityState')]
        if not qs_cols:
            raise ValueError("No se encontraron columnas terminadas en '_QualityState' en la matriz proporcionada.")

        records = []
        n_total = len(X_features)

        for col in qs_cols:
            base_name = col.replace('_QualityState', '')
            counts = X_features[col].value_counts()
            c0 = counts.get(0, 0)
            c1 = counts.get(1, 0)
            c2 = counts.get(2, 0)

            records.append({
                'Señal': base_name,
                'Normal_%': (c0 / n_total) * 100,
                'BAD_%': (c1 / n_total) * 100,
                'Vacio_%': (c2 / n_total) * 100,
                'Total_Problemas': c1 + c2
            })

        df_dist = pd.DataFrame(records).sort_values(by='Total_Problemas', ascending=False).head(top_n)
        df_dist = df_dist.sort_values(by='Total_Problemas', ascending=True)

        fig, ax = plt.subplots(figsize=(12, max(6, len(df_dist) * 0.35)))

        y_pos = np.arange(len(df_dist))
        bars_normal = ax.barh(y_pos, df_dist['Normal_%'], color='#2ecc71', label='Estado 0: Correcto (Normal)')
        bars_vacio = ax.barh(y_pos, df_dist['Vacio_%'], left=df_dist['Normal_%'], color='#f39c12', label='Estado 2: Sin Lectura (Vacío/Telemetría)')
        bars_bad = ax.barh(y_pos, df_dist['BAD_%'], left=df_dist['Normal_%'] + df_dist['Vacio_%'], color='#e74c3c', label='Estado 1: Falla Sensor (BAD)')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_dist['Señal'], fontsize=9)
        ax.set_xlabel("Porcentaje de Intervalos de 5 Minutos (%)", fontsize=11, fontweight='bold')
        ax.set_title(f"Distribución de Calidad por Sensor (Top {top_n} Señales con Incidencias)", fontsize=13, fontweight='bold')
        ax.set_xlim(0, 100)
        ax.legend(loc='lower left', frameon=True)
        ax.grid(axis='x', linestyle='--', alpha=0.7)

        plt.tight_layout()
        return fig, ax

    def plot_triple_redundancy(self, df_resampled, variable="Temp", threshold=None,
                               sample_hours=8, start_time=None):
        """
        Visualización 3: Compara mediciones triples (A, B, C) y grafica las diferencias
        inter-sensor frente al umbral límite de alarma de ModuleDB.

        Args:
            df_resampled: DataFrame con la grilla regular de 5 minutos.
            variable: Prefijo de la variable redundante ('Temp', 'Hum', 'Pres').
            threshold: Umbral manual de discrepancia (si None usa config/moduledb.json).
            sample_hours: Número de horas a visualizar (None = rango completo del dataset).
            start_time: Timestamp de inicio de la ventana. Si None usa el primer registro.
        """
        cols = [f"{variable}_A", f"{variable}_B", f"{variable}_C"]
        for c in cols:
            if c not in df_resampled.columns:
                raise ValueError(f"No se encontró la columna requerida '{c}' en el DataFrame.")

        thresh = threshold or self.thresholds.get(f"{variable}MDif", 0.3)

        # Recorte temporal si se especifica ventana
        if sample_hours is not None:
            if start_time is None:
                start_dt = pd.to_datetime('2026-07-01 00:00:00')
            else:
                start_dt = pd.to_datetime(start_time)
            end_dt = start_dt + pd.Timedelta(hours=sample_hours)
            df_sub = df_resampled.loc[(df_resampled.index >= start_dt) & (df_resampled.index <= end_dt), cols].dropna()
        else:
            df_sub = df_resampled[cols].dropna()

        diff_ab = (df_sub[f"{variable}_A"] - df_sub[f"{variable}_B"]).abs()
        diff_ac = (df_sub[f"{variable}_A"] - df_sub[f"{variable}_C"]).abs()
        diff_bc = (df_sub[f"{variable}_B"] - df_sub[f"{variable}_C"]).abs()
        max_diff = pd.concat([diff_ab, diff_ac, diff_bc], axis=1).max(axis=1)

        time_range_label = ""
        if sample_hours is not None:
            time_range_label = f" ({df_sub.index.min().strftime('%d-%b %H:%M')} — {df_sub.index.max().strftime('%d-%b %H:%M')})"

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        # Panel 1: Curvas de los 3 sensores
        ax1.plot(df_sub.index, df_sub[f"{variable}_A"], label=f"{variable}_A", color='#3498db', alpha=0.8, linewidth=1)
        ax1.plot(df_sub.index, df_sub[f"{variable}_B"], label=f"{variable}_B", color='#9b59b6', alpha=0.8, linewidth=1)
        ax1.plot(df_sub.index, df_sub[f"{variable}_C"], label=f"{variable}_C", color='#1abc9c', alpha=0.8, linewidth=1)
        ax1.set_title(f"Monitoreo Redundante Triple: {variable} (A, B y C){time_range_label}", fontsize=12, fontweight='bold')
        ax1.set_ylabel("Magnitud")
        ax1.legend(loc='upper right')
        ax1.grid(True, linestyle='--', alpha=0.6)

        # Panel 2: Máxima discrepancia vs. Umbral ModuleDB
        ax2.plot(max_diff.index, max_diff.values, color='#e67e22', linewidth=1, label='Discrepancia Máxima observada')
        ax2.axhline(y=thresh, color='#c0392b', linestyle='--', linewidth=1.5, label=f'Umbral ModuleDB ({thresh})')

        # Resaltar instantes donde se excede el umbral
        exceed_mask = max_diff > thresh
        n_exceed = int(exceed_mask.sum())
        if exceed_mask.any():
            ax2.scatter(max_diff.index[exceed_mask], max_diff[exceed_mask],
                        color='red', s=15, zorder=5, label=f'Exceso de Discrepancia ({n_exceed} intervalos)')

        ax2.set_title(f"Discrepancia Inter-Sensor frente al Límite ModuleDB ({variable}MDif = {thresh})", fontsize=12, fontweight='bold')
        ax2.set_xlabel("Tiempo")
        ax2.set_ylabel("Delta Máximo")
        ax2.legend(loc='upper right')
        ax2.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        return fig, (ax1, ax2)

    def plot_target_behavior(self, df_resampled, target_col="CNA"):
        """
        Visualización 4: Diagnóstico de la variable objetivo (Calc Failed):
        - Panel izquierdo: Distribución temporal de fallas a lo largo del día (por hora) y del mes.
        - Panel derecho: Conteo de estados de cálculo (Normal vs Calc Failed) con desbalance de clases.
        """
        if target_col not in df_resampled.columns:
            raise ValueError(f"No se encontró la columna objetivo '{target_col}' en el DataFrame.")

        cna_series = df_resampled[target_col]
        has_bad = df_resampled.get(f"{target_col}_Has_Bad", pd.Series(0, index=df_resampled.index))
        calc_failed = (has_bad == 1) | (cna_series.isna())

        fail_index = df_resampled.index[calc_failed]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # 1. Distribución temporal de fallas: Conteo de Calc Failed por hora del día
        fail_hours = fail_index.hour
        hour_counts = pd.Series(fail_hours).value_counts().sort_index()
        all_hours = pd.Series(0, index=range(24))
        all_hours.update(hour_counts)

        ax1.bar(all_hours.index, all_hours.values, color='#e74c3c', alpha=0.85, width=0.8)
        ax1.set_xticks(range(24))
        ax1.set_xticklabels([f"{h:02d}:00" for h in range(24)], rotation=45, ha='right', fontsize=8)
        ax1.set_xlabel("Hora del Día", fontsize=11, fontweight='bold')
        ax1.set_ylabel("Número de Eventos Calc Failed", fontsize=10)
        ax1.set_title(
            f"Distribución Horaria de Fallas de Cálculo ({target_col})\n"
            f"{int(calc_failed.sum()):,} eventos totales a lo largo del mes",
            fontsize=11, fontweight='bold'
        )
        ax1.grid(axis='y', linestyle='--', alpha=0.6)

        # 2. Desbalance de clases (Normal vs Calc Failed)
        labels = ['Normal (OK)', 'Calc Failed']
        counts = [int((~calc_failed).sum()), int(calc_failed.sum())]
        pcts = [(c / len(df_resampled)) * 100 for c in counts]

        bars = ax2.bar(labels, counts, color=['#2ecc71', '#e74c3c'], width=0.5)
        ax2.set_title(
            f"Desbalance de Clases — Variable Objetivo\n"
            f"(Calc Failed: {counts[1]:,} = {pcts[1]:.2f}% | Normal: {counts[0]:,} = {pcts[0]:.2f}%)",
            fontsize=11, fontweight='bold'
        )
        ax2.set_ylabel("Número de Intervalos de 5 min")
        ax2.grid(axis='y', linestyle='--', alpha=0.6)

        for bar, count, pct in zip(bars, counts, pcts):
            yval = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, yval + (max(counts) * 0.02),
                     f"{count:,}\n({pct:.2f}%)", ha='center', va='bottom', fontweight='bold', fontsize=10)

        ax2.set_ylim(0, max(counts) * 1.15)

        plt.tight_layout()
        return fig, (ax1, ax2)
