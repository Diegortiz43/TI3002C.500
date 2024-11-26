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

st.title("Metodología de exploración - Extendido")

## 1.1 Hipótesis del Análisis Exploratorio de Datos
st.subheader("1.1 Hipótesis del Análisis Exploratorio de Datos")
st.write("""
Las métricas de interacción en redes sociales (número de interacciones, likes, número de comentarios, etc.) reflejan de cerca los cambios en la intención de voto para cada candidato a lo largo del período electoral.  
Este análisis evaluará la correlación entre estas métricas y las intenciones de voto, probando su significancia estadística para determinar si estas métricas reflejan cambios en el apoyo público según lo indicado por los datos de encuestas (ver la sección [1.3 Datos Adicionales](#13-datos-adicionales) para más información).
""")

## 1.2 Objetivo de la Hipótesis
st.subheader("1.2 Objetivo de la Hipótesis")
st.write("""
Una vez seleccionadas las principales métricas de redes sociales según las mejores correlaciones de Pearson, se utilizará una regresión lineal simple para evaluar las métricas más influyentes basándose en los resultados del estadístico F.  
Finalmente, después de haber seleccionado la mejor métrica predictora para cada candidato, se realizará una proyección hacia adelante para los datos de encuestas y cada métrica por candidato. Esto permitirá medir la discrepancia entre los resultados proyectados tanto de los datos de las encuestas como de las métricas de redes sociales frente a los resultados oficiales de votación (del [Programa de Resultados Electorales Preliminares](https://prep2024.ine.mx/publicacion/nacional/presidencia/nacional/candidatura)).
""")

## 1.3 Datos Adicionales
st.subheader("1.3 Datos Adicionales")
st.write("""
### Datos de Encuestas
Para realizar este análisis, fue necesario recopilar datos históricos de encuestas para realizar comparaciones. Tras evaluar varias fuentes, se seleccionó el Barómetro Electoral de Bloomberg como referencia.  
Este sitio web, publicado por Bloomberg en México, recopila las principales encuestas del país y las pondera para obtener un promedio de intención de voto en fechas específicas. Los datos fueron recopilados manualmente, copiándolos y pegándolos en un archivo CSV que luego fue procesado a través de este notebook.
""")
st.image("images/barometro_ejemplo.png", caption="Visualización de los datos recopilados.")

## 1.4 Proporciones de Métricas por Candidato
st.subheader("1.4 Proporciones de Métricas por Candidato")
st.write("""
Es importante notar que los datos de las encuestas solo tienen 52 fechas utilizables (desde enero hasta el 28 de mayo, un día antes del cierre de campañas). Esto significa que solo hay 52 puntos de datos, y no son continuos. Las diferencias entre las fechas de las encuestas recopiladas por Bloomberg varían. Esto se ejemplifica mejor en las siguientes imágenes:
""")
st.image("images/candidates_metrics_proportion_1.png", caption="Ejemplo de un punto de datos de intención de voto en abril.")
st.image("images/candidates_metrics_proportion_2.png", caption="Diferencia de 7 días entre puntos de datos consecutivos.")

st.write("""
Para alinear estas fechas y valores con nuestros datos, calculamos la suma acumulada hasta el punto de datos de las encuestas y la dividimos por la suma acumulada total de la métrica específica en ese día. Esto permitió obtener la proporción de likes, comentarios, vistas e interacciones en los puntos de datos presentes en las encuestas.

**¿Por qué usar la suma acumulada?**  
El uso de sumas acumuladas captura la evolución de las interacciones a lo largo del tiempo en lugar de analizar solo instantáneas diarias. Este enfoque suaviza las fluctuaciones diarias y refleja el crecimiento gradual del apoyo a los candidatos, facilitando la identificación de tendencias a largo plazo y correlaciones con las intenciones de voto.
""")

## 1.5 Metodología Empleada
st.subheader("1.5 Metodología Empleada")
st.write("""
Se siguió una estrategia en forma de embudo que comenzó comparando las métricas más generales en todas las plataformas y avanzó hacia las métricas más específicas de cada plataforma. Este enfoque permitió descubrir y probar múltiples combinaciones de métricas que mejor se alinearon con las intenciones de voto. Los pasos de la metodología se ilustran en la siguiente figura:
""")
st.image("images/methodology_funnel.png", caption="Representación del enfoque de embudo seguido por el equipo.")

st.markdown("""
### Pasos de la Metodología
- **Análisis de Interacciones Consolidadas en Todas las Plataformas**  
Este paso analiza el número total de interacciones en todas las plataformas de redes sociales combinadas (e.g., Facebook, Instagram, X, YouTube). El objetivo es entender cómo las interacciones agregadas (likes, comentarios, vistas, etc.) se correlacionan con las intenciones de voto.

- **Análisis de Interacciones Consolidadas por Plataforma**  
En este paso, se analizan las interacciones en cada plataforma por separado. En lugar de agregar todas las plataformas, el enfoque se centra en las interacciones totales dentro de cada plataforma individual para identificar tendencias específicas y su relación con las intenciones de voto.

- **Análisis Métrico por Plataforma**  
Aquí, el análisis se desglosa aún más para estudiar métricas individuales (e.g., likes, comentarios, vistas) en cada plataforma por separado. Esto proporciona una comprensión más granular de cómo los tipos específicos de interacción contribuyen a las intenciones de voto por plataforma.

- **Selección Final de Métricas**  
Este paso evalúa los resultados de los análisis previos para seleccionar las métricas más relevantes. Las métricas elegidas son aquellas que mejor se correlacionan con las intenciones de voto, formando la base para modelar y realizar predicciones adicionales.
""")