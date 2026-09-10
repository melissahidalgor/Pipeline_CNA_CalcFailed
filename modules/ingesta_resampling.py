import pandas as pd
import numpy as np

class PIACEIngestionPipeline:
    """
    Módulo de Ingesta y Resampling para simular la ventana 
    de cálculo de PI ACE a partir de registros dispersos.
    Permite procesar señales individuales o múltiples pares de señales
    (Time_<Variable>, <Variable>) alineándolas a una grilla regular de 5 minutos
    que coincide con la estampa de tiempo de CNA.
    """
    def __init__(self, freq_minutes=5):
        self.freq_minutes = freq_minutes
        self.freq_str = f"{freq_minutes}min"
        
    def load_data(self, file_path):
        """Lee el archivo CSV o Excel preservando textos originales para trazabilidad de errores."""
        path_str = str(file_path)
        if path_str.endswith('.csv'):
            return pd.read_csv(file_path, low_memory=False)
        else:
            return pd.read_excel(file_path)

    @staticmethod
    def _parse_datetime(series):
        """
        Convierte series de tiempo mixtas de manera robusta:
        soporta cadenas '%d-%b-%y %H:%M:%S' y valores seriales numéricos de Excel (ej. 46233.98).
        """
        s = series.dropna()
        dt_str = pd.to_datetime(s, format='%d-%b-%y %H:%M:%S', errors='coerce') # Convierte formato dby hmd a fechas

        # Estrategia de respaldo 1 - numeros seriales tipo Excel
        mask_bad = dt_str.isna() # Mascara de conversiones fallidas
        if mask_bad.any():
            num = pd.to_numeric(s[mask_bad], errors='coerce') # Intenta convertir a númerico
            dt_num = pd.to_datetime(num, unit='D', origin='1899-12-30').dt.round('min') # Transforma los números seriales estilo Excel a fechas
            dt_str = dt_str.fillna(dt_num)

        # Estrategia de respaldo 2 -
            still_na = dt_str.isna() # Mascara de conversiones fallidas
            if still_na.any():
                dt_str = dt_str.fillna(pd.to_datetime(s[still_na], errors='coerce', format='mixed')) # Permite que Pandas intente adivinar dinámicamente el formato de texto

        return dt_str.dt.round('min')
        # Devuelve la serie con todas las fechas unificadas como objetos datetime64 con redondeo al minuto más cercano

    def extract_tag_pairs(self, df_raw):
        """
        Detecta automáticamente las parejas (columna_tiempo, columna_valor).
        Identifica pares con prefijo 'Time_' o columnas adyacentes en el dataset.
        """
        pairs = [] # Lista vacía para guardar las parejas encontradas en forma de tuplas
        cols = list(df_raw.columns) # Extrae los nombres de todas las columnas
        for col in cols:
            if col.startswith('Time_'):
                val_col = col.replace('Time_', '', 1) # Extrae nombre tag
                if val_col in cols:
                    pairs.append((col, val_col))

        # Estrategia de respaldo
        if not pairs and len(cols) >= 2:
            for i in range(0, len(cols) - 1, 2):
                pairs.append((cols[i], cols[i + 1])) # Empareja la columna en la posición actual con la adyacente
                
        return pairs

    def process_tag_series(
        self, 
        df_raw, 
        time_col=None, 
        val_col=None, 
        tag_pairs=None, 
        reference_time_col='Time_CNA',
        include_quality_flags=True,
        interpolate_limit=None
    ):
        """
        Estandariza y resamplea a 5 minutos, preservando la visibilidad de datos BAD.
        
        Args:
            df_raw: DataFrame crudo leído con load_data.
            time_col, val_col: Opcionales para procesar una sola señal.
            tag_pairs: Opcional, lista de tuplas [(col_tiempo, col_valor), ...].
            reference_time_col: Columna para la grilla maestra de 5 min (Time_CNA).
            include_quality_flags: Si es True, incluye columnas '<Tag>_Has_Bad' que indican
                                   si hubo algún valor BAD o no numérico en esa ventana de 5 min.
            interpolate_limit: Límite para interpolación temporal suave (None por defecto).
        """
        # Modo 1: Señal individual
        if time_col is not None and val_col is not None:
            df_tag = df_raw[[time_col, val_col]].dropna(subset=[time_col]).copy()
            dt = self._parse_datetime(df_tag[time_col])
            raw_vals = df_tag[val_col]
            num_vals = pd.to_numeric(raw_vals, errors='coerce')
            is_bad = num_vals.isna() & raw_vals.notna()

            s_val = pd.Series(num_vals.values, index=dt).sort_index()
            s_bad = pd.Series(is_bad.astype(int).values, index=dt).sort_index()

            val_res = s_val.resample(self.freq_str).mean()
            bad_res = s_bad.resample(self.freq_str).max().fillna(0).astype(int)

            if interpolate_limit:
                val_res = val_res.interpolate(method='time', limit=interpolate_limit)

            val_res.name = val_col
            if include_quality_flags:
                bad_res.name = f"{val_col}_Has_Bad"
                res_df = pd.concat([val_res, bad_res], axis=1)
            else:
                res_df = val_res.to_frame()
            res_df.index.name = 'Timestamp'
            return res_df

        # Modo 2: Múltiples señales (o todas automáticamente)
        if tag_pairs is None:
            tag_pairs = self.extract_tag_pairs(df_raw)

        # Construir grilla maestra de 5 minutos usando la columna de referencia (Time_CNA)
        ref_grid = None
        if reference_time_col and reference_time_col in df_raw.columns:
            ref_dt = self._parse_datetime(df_raw[reference_time_col])
            ref_grid = pd.date_range(start=ref_dt.min(), end=ref_dt.max(), freq=self.freq_str)

        resampled_series = []
        for t_col, v_col in tag_pairs:
            df_sub = df_raw[[t_col, v_col]].dropna(subset=[t_col])
            dt = self._parse_datetime(df_sub[t_col])
            raw_vals = df_sub[v_col]
            num_vals = pd.to_numeric(raw_vals, errors='coerce')

            # Detectar si hubo valores BAD o no numéricos en los registros crudos de la ventana
            is_bad = num_vals.isna() & raw_vals.notna()

            s_val = pd.Series(num_vals.values, index=dt).sort_index()
            s_bad = pd.Series(is_bad.astype(int).values, index=dt).sort_index()

            val_res = s_val.resample(self.freq_str).mean()
            bad_res = s_bad.resample(self.freq_str).max().fillna(0).astype(int)

            # Alinear a la grilla maestra si está disponible
            if ref_grid is not None:
                val_res = val_res.reindex(ref_grid)
                bad_res = bad_res.reindex(ref_grid).fillna(0).astype(int)

            val_res.name = v_col
            resampled_series.append(val_res)

            if include_quality_flags:
                bad_res.name = f"{v_col}_Has_Bad"
                resampled_series.append(bad_res)

        df_aligned = pd.concat(resampled_series, axis=1)
        df_aligned.index.name = 'Timestamp'

        if interpolate_limit:
            val_cols_to_interp = [c for c in df_aligned.columns if not c.endswith('_Has_Bad')]
            df_aligned[val_cols_to_interp] = df_aligned[val_cols_to_interp].interpolate(
                method='time', limit=interpolate_limit
            )

        return df_aligned