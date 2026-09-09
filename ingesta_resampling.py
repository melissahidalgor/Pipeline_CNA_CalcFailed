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
        """Lee el archivo CSV o Excel sustituyendo estados de error por NaN."""
        path_str = str(file_path)
        na_vals = ['Bad', 'Invalid Data', 'Calc Failed', 'I/O Timeout']
        if path_str.endswith('.csv'):
            return pd.read_csv(file_path, na_values=na_vals, low_memory=False)
        else:
            return pd.read_excel(file_path, na_values=na_vals)

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
        interpolate_limit=None
    ):
        """
        Estandariza y resamplea a 5 minutos.
        
        Puede usarse de dos formas:
        1. Señal individual:
           process_tag_series(df_raw, 'Time_Temp_A', 'Temp_A')
        2. Múltiples señales (o todas automáticamente):
           process_tag_series(df_raw) # Procesa automáticamente todos los pares (Time_X, X)
           process_tag_series(df_raw, tag_pairs=[('Time_Temp_A', 'Temp_A'), ...])
           
        Alinea todas las señales a una grilla regular de 5 minutos basada en reference_time_col (Time_CNA).
        """
        # Modo 1: Señal individual
        if time_col is not None and val_col is not None:
            df_tag = df_raw[[time_col, val_col]].dropna(subset=[time_col]).copy()
            df_tag[time_col] = self._parse_datetime(df_tag[time_col])
            df_tag[val_col] = pd.to_numeric(df_tag[val_col], errors='coerce')
            df_tag = df_tag.sort_values(by=time_col).set_index(time_col)
            df_resampled = df_tag.resample(self.freq_str).mean()
            if interpolate_limit:
                df_resampled = df_resampled.interpolate(method='time', limit=interpolate_limit)
            return df_resampled

        # Modo 2: Múltiples señales
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
            val = pd.to_numeric(df_sub[v_col], errors='coerce')
            
            s = pd.Series(val.values, index=dt).sort_index()
            s_res = s.resample(self.freq_str).mean()
            
            # Alinear a la grilla de CNA si está disponible
            if ref_grid is not None:
                s_res = s_res.reindex(ref_grid)
                
            s_res.name = v_col
            resampled_series.append(s_res)

        df_aligned = pd.concat(resampled_series, axis=1)
        df_aligned.index.name = 'Timestamp'

        if interpolate_limit:
            df_aligned = df_aligned.interpolate(method='time', limit=interpolate_limit)

        return df_aligned