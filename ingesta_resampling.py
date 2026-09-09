import pandas as pd
import numpy as np

class PIACEIngestionPipeline:
    """
    Módulo de Ingesta, Resampling para simular la ventana 
    de cálculo de PI ACE a partir de registros dispersos.
    """
    def __init__(self, freq_minutes=5):
        self.freq_minutes = freq_minutes
        self.freq_str = f"{freq_minutes}min"
        
    def load_data(self, file_path):
        """Lee el archivo CSV y extrae las series de tiempo"""
        # Pandas lee el archivo completo en memoria antes de determinar el tipo de dato de cada columna
        df_raw = pd.read_csv(file_path, low_memory=False)
        return df_raw


    def process_tag_series(self, df_raw, time_col, val_col):
        """Estandariza una señal individual: convierte timestamp, limpia y resamplea."""
        df_tag = df_raw[[time_col, val_col]].copy()
        df_tag.dropna(subset=[time_col], inplace=True)
        
        # Convertir a datetime y forzar valores numéricos (reemplazando BAD/CNA/Calc Failed por NaN)
        df_tag[time_col] = pd.to_datetime(df_tag[time_col])
        df_tag[val_col] = pd.to_numeric(df_tag[val_col], errors='coerce')
        
        # Ordenar e indexar por tiempo
        df_tag = df_tag.sort_values(by=time_col).set_index(time_col)
        
        # Resamplear a grilla regular (promedio en la ventana de 5 min)
        df_resampled = df_tag.resample(self.freq_str).mean()
        return df_resampled

    def align_and_shift(self, dict_tags):
        """
        Unifica todos los tags en un solo DataFrame temporal y aplica 
        el desplazamiento (lag) de 15 minutos para simular sEscritura.
        """
        # Unir todas las series en un solo DataFrame mediante alineación de índices
        df_aligned = pd.concat(dict_tags.values(), axis=1)
        
        # Interpolar suavemente para cubrir vacíos breves de lectura por excepción
        df_aligned = df_aligned.interpolate(method='time', limit=2)
        
        # Aplicar Lag de 15 minutos (desplazar datos de entrada hacia adelante) - NO APLICA
        # Esto alinea X(t-15m) con Y(t)
       # shift_periods = self.lag_minutes // self.freq_minutes
       # df_shifted = df_aligned.shift(shift_periods)
        
        return df_aligned

# --- Ejemplo de Uso ---
# pipeline = PIACEIngestionPipeline(freq_minutes=5, lag_minutes=15)
# raw_data = pipeline.load_data('muestraIN.xlsx')
# tag1 = pipeline.process_tag_series(raw_data, 'Time1', 'Tag1')
# tag2 = pipeline.process_tag_series(raw_data, 'Time2', 'Tag2')
# df_final_features = pipeline.align_and_shift({'Tag1': tag1, 'Tag2': tag2})