import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import numpy as np
from datetime import timedelta
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer 
from scipy.stats import f_oneway
import numpy as np
import statsmodels.api as sm
import joblib

# Título visible en la página
st.title("Metodología de exploración")

## 1 Hipótesis del Análisis Exploratorio de Datos
st.subheader("Hipótesis del Análisis Exploratorio de Datos")
st.write("""
Las métricas de interacción en redes sociales (número de interacciones, likes, número de comentarios, etc.) reflejan de cerca los cambios en la intención de voto para cada candidato a lo largo del período electoral.  
Este análisis evaluará la correlación entre estas métricas y las intenciones de voto, probando su significancia estadística para determinar si estas métricas reflejan cambios en el apoyo público según lo indicado por los datos de encuestas.
""")

## 2 Datos Adicionales
st.write("""
### Datos de Encuestas
Para realizar este análisis, fue necesario recopilar datos históricos de encuestas para realizar comparaciones. Tras evaluar varias fuentes, se seleccionó el [Barómetro Electoral de Bloomberg](https://www.bloomberg.com/graphics/Mexico-Encuestas-Presidenciales-2024-ventaja-sheinbaum-galvez-veda/) como referencia.  
Este sitio web, publicado por Bloomberg en México, recopila las principales encuestas del país y las pondera para obtener un promedio de intención de voto en fechas específicas. Los datos fueron recopilados manualmente, copiándolos y pegándolos en un archivo CSV que luego fue procesado a través de este notebook.
""")
st.image("images/barometro_ejemplo.png", caption="Visualización de los datos recopilados.")

## 1.5 Metodología Empleada
st.subheader("Metodología Empleada")
st.write("""
Se siguió una estrategia en forma de embudo que comenzó comparando las métricas más generales en todas las plataformas y avanzó hacia las métricas más específicas de cada plataforma. Este enfoque permitió descubrir y probar múltiples combinaciones de métricas que mejor se alinearon con las intenciones de voto. Los pasos de la metodología se ilustran en la siguiente figura:
""")
st.image("images/methodology_funnel.png", caption="Representación del enfoque de embudo seguido por el equipo.")










