# Predicción de Fallas de Cálculo (Calc Failed) — CNA

Este repositorio contiene la solución modular de Machine Learning para predecir eventos de **Calc Failed** en el cálculo de **Capacidad Neta Disponible (CNA)** de una Central Termoeléctrica. 

El proyecto integra la telemetría irregular de sensores, la lógica de negocio heredada del antiguo motor de cálculo **AVEVA PI System (PI ACE)** y umbrales operativos de **ModuleDB**, empaquetado dentro de un pipeline robusto de Scikit-Learn.

---

## 📌 Contexto del Problema

El cálculo de CNA depende de múltiples variables ambientales (Temperatura, Humedad, Presión) y operativas (calidad de llama en turbinas, densidad de gas, contadores de energía). Cuando los sensores fallan o la discrepancia entre sensores redundantes supera los límites operativos de ModuleDB, el sistema genera un estado **Calc Failed** (6.42% de las observaciones en el dataset de prueba).

**Objetivo:** Predecir de manera temprana y automática los fallos de cálculo minimizando falsos negativos mediante modelos de clasificación binaria.

---

## 🛠️ Arquitectura del Proyecto

```text
RandomForest_CNA/
├── config/
│   ├── moduledb.json        # Umbrales y límites operativos (ModuleDB)
│   └── requirements.txt     # Librerías y dependencias
├── data/
│   └── dataset.csv          # Datos de telemetría de planta
├── modules/
│   ├── __init__.py          # Exportación de clases principales
│   ├── ingesta_resampling.py # PIACEIngestionPipeline (Carga y resampleo temporal a 5 min)
│   ├── feature_engineering.py# PIACEFeatureEngineering (Reglas ModuleDB y QualityState 0/1/2)
│   └── dataset_visual.py    # DatasetVisualizer (EDA y diagnósticos de discrepancias)
├── main_exp.ipynb           # Notebook de Exploración y Experimentos (EDA)
├── main_prod.ipynb          # Notebook de Producción (Pipeline Encapsulado)
└── README.md                # Documentación del proyecto
```

---

## ⚙️ Descripción de Módulos

### 1. Ingesta y Resampleo (`modules/ingesta_resampling.py`)
* `PIACEIngestionPipeline`: Procesa series de tiempo irregulares (muestreo por excepción) y las alinea en una grilla regular de 5 minutos referenciada a `Time_CNA`.
* Preserva la visibilidad de lecturas erróneas (`'Bad'` / `'Invalid Data'`) generando banderas `<Tag>_Has_Bad` antes de aplicar agregaciones (`mean`).

### 2. Ingeniería de Características (`modules/feature_engineering.py`)
* `PIACEFeatureEngineering`: Carga umbrales de `config/moduledb.json`.
* **Codificación de Calidad (3 Estados):** Genera `<Tag>_QualityState` (0: Normal, 1: Error de Hardware/BAD, 2: Sin Telemetría/Vacío).
* **Evaluación de Discrepancias Redundantes:** Calcula deltas máximos entre sensores A, B y C (`Diff_Temp_AB`, etc.) y banderas de exceso frente a ModuleDB (`Temp_Exceeds_Dif`).
* **Banderas de Dominio:** Detección de pérdida de flama (`G1_FD_INTENS`), densidad de gas cero y deltas negativos en acumuladores de energía.

### 3. Visualización y Diagnóstico (`modules/dataset_visual.py`)
* `DatasetVisualizer`: Herramientas gráficas y analíticas para:
  * Comparación de señales crudas vs. resampleadas en ventanas de tiempo configurables (`plot_resampling_comparison`).
  * Distribución de estados de calidad por sensor (`plot_quality_distribution`).
  * Monitoreo redundante triple vs. umbrales ModuleDB (`plot_triple_redundancy`).
  * Distribución temporal horaria de eventos *Calc Failed* y desbalance de clases (`plot_target_behavior`).

---

## 📓 Notebooks: Exploración vs. Producción

El proyecto está estructurado en dos entornos de trabajo diferenciados para separar la fase analítica de la puesta en producción:

### 1. `main_exp.ipynb` (Entorno de Exploración y Análisis - EDA)
* **Propósito:** Análisis exploratorio preliminar, validación de hipótesis de negocio y diagnóstico visual.
* **Contenido:**
  * Resúmenes estadísticos comparativos (antes y después del resampleo).
  * Gráficas de comportamiento de sensores redundantes y calidad de señal.
  * Pruebas iterativas de algoritmos y ajuste manual de hiperparámetros.
  * Inspección gráfica de desbalance de clases y curvas de entrenamiento.

### 2. `main_prod.ipynb` (Entorno de Producción y Pipeline Encapsulado)
* **Propósito:** Simulación de un entorno de producción mediante un flujo de trabajo hermético en Scikit-Learn que evita la fuga de datos (*data leakage*).
* **Arquitectura del Pipeline:**
  ```python
  full_pipeline = Pipeline(steps=[
      ('feature_engineering', FunctionTransformer(fe.transform, validate=False)),
      ('classifier', xgb.XGBClassifier(scale_pos_weight=scale_pos, ...))
  ])
  ```
* **Ventajas:**
  * **Cero Data Leakage:** El pipeline ejecuta la ingeniería de características y el escalado (`RobustScaler`) de manera aislada dentro de cada fold durante la Validación Cruzada (`cross_validate`).
  * **Inferencia Directa:** Permite pasar datos crudos alineados (`X_raw`) directamente a `full_pipeline.predict(X_raw)`.

---

## 🚀 Requisitos e Instalación

1. **Clonar el repositorio:**
   ```bash
   git clone <URL_DEL_REPOSITORIO>
   cd RandomForest_CNA
   ```

2. **Crear y activar entorno virtual:**
   ```bash
   python -m venv config/venv
   # En Windows:
   .\config\venv\Scripts\activate
   ```

3. **Instalar dependencias:**
   ```bash
   pip install -r config/requirements.txt
   ```

---

## 📊 Ejecución

Para reproducir el pipeline completo de producción:
1. Abrir `main_prod.ipynb` en Jupyter Notebook o VS Code.
2. Seleccionar el kernel del entorno virtual (`config/venv`).
3. Ejecutar todas las celdass.
