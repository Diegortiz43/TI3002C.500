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


st.title("📊 Resultados y Predicciones Clave (detalles)")
st.write("Visualiza las métricas seleccionadas agrupadas por intención de voto, interacciones generales o plataforma específica.")

# Cargar datos (reemplaza con tu archivo CSV real)
@st.cache_data
def load_pollster_data():
    file_path = "ALL/final_datasets/Pollster_vs_Metrics_Predicted.csv"  # Cambia esta línea por la ubicación real
    data = pd.read_csv(file_path)
    data["date"] = pd.to_datetime(data["date"])  # Aseguramos que la columna `date` esté en formato fecha
    return data

pollster_data = load_pollster_data()

# Configuración inicial
st.sidebar.header("Filtros dinámicos")

# Lista inicial de todas las métricas
all_metrics = pollster_data.columns.tolist()

# 1. Filtro de candidatos
st.sidebar.subheader("Selecciona Candidatos")
candidates_categories = {
    "Candidato - Claudia Sheinbaum": [col for col in all_metrics if "claudia_" in col],
    "Candidato - Xochitl Gálvez": [col for col in all_metrics if "xochitl_" in col],
    "Candidato - Jorge Máynez": [col for col in all_metrics if "maynez_" in col],
}
selected_candidates = st.sidebar.multiselect(
    "Candidatos Disponibles",
    options=candidates_categories.keys(),
    default=list(candidates_categories.keys()),  # Todos seleccionados por defecto
)

# Actualizar métricas restantes según los candidatos seleccionados
filtered_metrics_by_candidates = []
for candidate in selected_candidates:
    filtered_metrics_by_candidates.extend(candidates_categories[candidate])

# 2. Filtro de plataformas
st.sidebar.subheader("Selecciona Plataformas")
platform_categories = {
    "Plataforma - X": [col for col in filtered_metrics_by_candidates if "x_" in col],
    "Plataforma - YouTube": [col for col in filtered_metrics_by_candidates if "youtube_" in col],
    "Plataforma - Facebook": [col for col in filtered_metrics_by_candidates if "facebook_" in col],
    "Plataforma - Instagram": [col for col in filtered_metrics_by_candidates if "instagram_" in col],
}
selected_platforms = st.sidebar.multiselect(
    "Plataformas Disponibles",
    options=platform_categories.keys(),
    default=list(platform_categories.keys()),  # Todas seleccionadas por defecto
)

# Actualizar métricas restantes según las plataformas seleccionadas
filtered_metrics_by_platforms = []
for platform in selected_platforms:
    filtered_metrics_by_platforms.extend(platform_categories[platform])

# 3. Filtro de métricas generales
st.sidebar.subheader("Selecciona Métricas")
metrics_categories = {
    "Métrica - Voting Intention": [col for col in filtered_metrics_by_platforms if "voting_intention" in col],
    "Métrica - Total de Interacciones": [col for col in filtered_metrics_by_platforms if "num_interaction_share" in col],
    "Métrica - Like": [col for col in filtered_metrics_by_platforms if "like_" in col],
    "Métrica - Comment": [col for col in filtered_metrics_by_platforms if "comment_" in col],
}
selected_metrics = st.sidebar.multiselect(
    "Métricas Disponibles",
    options=metrics_categories.keys(),
    default=list(metrics_categories.keys()),  # Todas seleccionadas por defecto
)

# Extraer métricas finales
final_metrics = []
for metric in selected_metrics:
    final_metrics.extend(metrics_categories[metric])

# Siempre incluir Voting Intention (pero filtrado por candidato)
voting_intention_metrics = [col for col in filtered_metrics_by_candidates if "voting_intention" in col]
final_metrics = list(set(final_metrics + voting_intention_metrics))

# Verificar selección de métricas
if not final_metrics:
    st.warning("Por favor selecciona al menos una métrica para visualizar.")
else:
    # Filtrar datos según las métricas seleccionadas
    filtered_data = pollster_data[["date"] + final_metrics]

    # Transformar datos para visualización
    melted_data = filtered_data.melt(id_vars="date", value_vars=final_metrics, var_name="Métrica", value_name="Valor")

    # Gráfico de líneas
    st.subheader("Tendencia de Métricas Seleccionadas")
    fig = px.line(
        melted_data,
        x="date",
        y="Valor",
        color="Métrica",
        title="Gráfico de Tendencias",
        labels={"date": "Fecha", "Valor": "Valor", "Métrica": "Métrica"},
        template="plotly_white"
    )
    # # Agregar una línea vertical estática en la fecha 28 de mayo
    # fig.add_vline(
    #     x="2024-05-28",  # Fecha de la línea
    #     line_width=2,  # Grosor de la línea
    #     line_dash="dash",  # Estilo de la línea (puede ser "dash", "solid", etc.)
    #     line_color="red",  # Color de la línea
    #     annotation_text="Predicted Data",  # Etiqueta de la línea
    #     annotation_position="top left"  # Posición de la etiqueta
    # )

    # Mostrar el gráfico en Streamlit
    st.plotly_chart(fig)