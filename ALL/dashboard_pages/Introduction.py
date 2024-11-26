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


st.title("Introducción")

# Introducción
st.header("Bienvenido")
st.markdown(
    """
    Este dashboard ofrece un análisis integral de las métricas de redes sociales y su vínculo con las intenciones de voto en el contexto electoral. A través de esta herramienta, profundizaremos en la limpieza de datos, la generación de estadísticas descriptivas, la realización de pruebas de hipótesis, el desarrollo de análisis de regresión y la creación de visualizaciones interactivas para interpretar los resultados de manera efectiva.
    """
)

# Imagen representativa
st.image(
    "ALL/OMD.jpeg",  # Cambia esta URL por tu imagen
    caption="Análisis de métricas en redes sociales.",
    use_column_width=True,
)

# Propósito
st.subheader("Propósito")
st.markdown(
    """
    - **Hipótesis:** "¿Existe una correlación significativa entre las métricas de redes sociales (interacciones, alcance, etc.) y las intenciones de voto de los usuarios?"
    - **Objetivo:** Explorar cómo las métricas de redes sociales reflejan las intenciones de voto.
    - **Contexto:** Utilizamos datos de múltiples plataformas y candidatos para descubrir patrones significativos.
    - **Metodología:** Limpieza de datos, análisis estadísticos y gráficos interactivos.
    """
)

# Contexto
st.subheader("Contexto")
st.markdown(
    """
    Este análisis se realizó usando un conjunto de datos que incluye:
    - Interacciones sociales (likes, comentarios, visualizaciones).
    - Métricas por plataforma (Facebook, Instagram, YouTube).
    - Comparaciones con encuestas de intención de voto.
    """
)