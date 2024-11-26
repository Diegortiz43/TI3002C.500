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

st.title("📊 Exploración de datos")
st.write("Esta página muestra la proporción de interacciones por candidato en función de la plataforma seleccionada.")

# Cargar base
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import timedelta

df = pd.read_csv("ALL/final_datasets/all_together_no_duplicates_no_missing_filtered.csv")
df['datetime'] = pd.to_datetime(df['datetime'])

plataformas = st.selectbox("Selecciona una Plataforma:", df['platform'].unique())


# Filtrar los datos según la plataforma seleccionada
df_filtered = df[df["platform"] == plataformas]

# Cálculo de proporciones
proportions = df_filtered.groupby("candidate_name")["num_interaction"].sum()
proportions = proportions / proportions.sum() * 100  # Convertir a porcentajes
proportions = proportions.reset_index()
proportions.columns = ["Candidato", "Proporción"]

candidate_plot_colors = {
'Claudia Sheinbaum': '#741D23',      
'Jorge Álvarez Máynez': '#FF8300',      
'Xóchitl Gálvez': '#1E75BC'       
}

# Gráfico de pastel
st.subheader(f"Proporción de Interacciones en {plataformas}")
fig = px.pie(
proportions,
values="Proporción",
names="Candidato",
title=f"Proporción de Interacciones en {plataformas}",
color="Candidato",  # Columna que define los colores
color_discrete_map=candidate_plot_colors  # Aplicar el diccionario de colores
)
st.plotly_chart(fig)


# Linea de Tendencia

# Cargar datos
pollster_data = pd.read_csv("ALL/final_datasets/Pollster_vs_Metrics_Predicted.csv")
pollster_data["date"] = pd.to_datetime(pollster_data["date"])

# Diccionario para renombrar métricas a nombres más amigables
metric_labels = {
    "claudia_voting_intention": "Claudia's Voting Intention",
    "xochitl_voting_intention": "Xóchitl's Voting Intention",
    "maynez_voting_intention": "Máynez's Voting Intention",
}

# Selección de métricas para el candidato actual
selected_metrics = ["claudia_voting_intention", "xochitl_voting_intention", "maynez_voting_intention"]

# Filtrar datos relevantes (asegurando crear una copia explícita)
filtered_data = pollster_data[["date"] + selected_metrics].copy()

# Filtrar datos antes de 2024-05-28
filtered_data = filtered_data[filtered_data["date"] < "2024-05-28"]

# Renombrar las métricas con nombres amigables
melted_data = filtered_data.melt(id_vars="date", var_name="Métrica", value_name="Valor").copy()
melted_data["Métrica"] = melted_data["Métrica"].map(metric_labels)

candidate_plot_colors = {
"Claudia's Voting Intention": '#741D23',      
"Máynez's Voting Intention": '#FF8300',      
"Xóchitl's Voting Intention": '#1E75BC'       
}

# Crear la gráfica con px.line
st.subheader("Tendencia de la Intención de Voto")
st.write("Con la intención de tener un parámetro de comparativa relacionado completamente con las elecciones, se importó el [Barómetro de Bloomberg](https://www.bloomberg.com/graphics/Mexico-Encuestas-Presidenciales-2024-ventaja-sheinbaum-galvez-veda/), que indica la **Intención de Voto** para cada candidato a partir de una ponderación de todas las encuestas realizadas a la población Mexicana.")
fig = px.line(
melted_data,
x="date",
y="Valor",
color="Métrica",
title=f"Tendencia de la Intención de Voto",
labels={"date": "Fecha", "Valor": "Valor", "Métrica": "Métrica"},
template="plotly_white",
color_discrete_map=candidate_plot_colors  # Apply the color dictionary
)

# Mostrar el gráfico
st.plotly_chart(fig)
