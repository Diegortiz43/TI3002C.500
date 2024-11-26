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


st.title("📊 Resultados y Predicciones Clave")

# Cargar datos
pollster_data = pd.read_csv("ALL/final_datasets/Pollster_vs_Metrics_Predicted.csv")
pollster_data["date"] = pd.to_datetime(pollster_data["date"])

# Configuración del slicer
candidates = ["Claudia Sheinbaum", "Xóchitl Gálvez", "Jorge Álvarez Máynez"]
selected_candidate = st.selectbox("Selecciona un candidato", candidates)

# Configuración de métricas por candidato
candidate_metrics = {
    "Claudia Sheinbaum": ["claudia_instagram_like_count_share", "claudia_voting_intention"],
    "Xóchitl Gálvez": ["xochitl_instagram_like_count_share", "xochitl_voting_intention"],
    "Jorge Álvarez Máynez": ["maynez_num_interaction_share", "maynez_voting_intention", "maynez_x_num_interaction_share"],
}

# Diccionario para renombrar métricas a nombres más amigables
metric_labels = {
    "claudia_instagram_like_count_share": "Proporción de Likes en Instagram",
    "claudia_voting_intention": "Intención de Voto",
    "xochitl_instagram_like_count_share": "Proporción de Likes en Instagram",
    "xochitl_voting_intention": "Intención de Voto",
    "maynez_num_interaction_share": "Interacciones totales",
    "maynez_voting_intention": "Intención de Voto",
    "maynez_x_num_interaction_share": "Proporción de Interacciones totales (en X)",
}

# Selección de métricas para el candidato actual
selected_metrics = candidate_metrics[selected_candidate]

# Filtrar datos relevantes
filtered_data = pollster_data[["date"] + selected_metrics]

# Asegurar que la columna "date" esté en formato datetime
filtered_data["date"] = pd.to_datetime(filtered_data["date"])

# Aplicar filtro condicional para Claudia y Xóchitl
if selected_candidate in ["Claudia Sheinbaum", "Xóchitl Gálvez"]:
    filtered_data = filtered_data[filtered_data["date"] >= "2024-04-01"]

# Convertir la fecha de la línea vertical a datetime
predicted_date = pd.Timestamp("2024-05-28")

# Renombrar las métricas con nombres amigables
melted_data = filtered_data.melt(id_vars="date", var_name="Métrica", value_name="Valor")
melted_data["Métrica"] = melted_data["Métrica"].map(metric_labels)

color_map = {
    "Proporción de Likes en Instagram": "#E1306C",             # Instagram branding color
    "Intención de Voto": "black",                              # General intention color
    "Proporción de Interacciones en Youtube": "#FF0000",       # YouTube branding color
    "Interacciones totales": "#FF8300",                            # Jorge Álvarez Máynez
    "Proporción de Interacciones totales": "#1DA1F2",          # Twitter (X) branding color
}

# Crear la gráfica con px.line
st.subheader(f"Tendencia de Métricas: {selected_candidate}")
st.write("A continuación se pueden observar la proporción de las **Interacciones** para cada candidato comparadas con la **Intención de Voto** recolectada por Bloomberg de las encuestas realizadas durante la campaña electoral.")

fig = px.line(
    melted_data,
    x="date",
    y="Valor",
    color="Métrica",
    title=f"Tendencia de Métricas - {selected_candidate}",
    labels={"date": "Fecha", "Valor": "Valor", "Métrica": "Métrica"},
    template="plotly_white",
    color_discrete_map=color_map  # Apply the custom color map
)

# Agregar la línea vertical con go.Figure (usamos add_shape para mayor control)
fig.update_layout(
    shapes=[
        dict(
            type="line",
            x0=predicted_date,
            x1=predicted_date,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line=dict(color="red", width=2, dash="dash"),
        )
    ]
)

# Añadir texto para la línea vertical
fig.add_annotation(
    x=predicted_date,
    y=1.05,  # Posición ligeramente fuera del gráfico
    text="Veda Electoral",
    showarrow=False,
    font=dict(color="red", size=12),
    xref="x",
    yref="paper",
)

# Mostrar el gráfico
st.plotly_chart(fig)




st.subheader(f"Predicción de Conteo Final: {selected_candidate}")
st.write("A partir de los coeficientes derivados de las regresiones lineales entre la **Intención de Voto** y las **Interacciones con mejor correlacion** de los candidatos, realizamos la predicción del conteo final de votos. Esta gráfica muestra la comparativa de la intencion de voto recaudada por Bloomberg, los datos recaudados por el OMD (proporción de la métrica final) y nuestra predicción calculada.")

# Definir métricas y resultados oficiales según el candidato
if selected_candidate == "Claudia Sheinbaum":
    metrics = {
        "Último Valor en Encuestas": "claudia_voting_intention",
        "Proyección de Intención de Voto": "claudia_voting_intention",
        "Proporción de Interacción Final": "claudia_instagram_like_count_share"
    }
    prep_result = 59.36
elif selected_candidate == "Xóchitl Gálvez":
    metrics = {
        "Último Valor en Encuestas": "xochitl_voting_intention",
        "Proyección de Intención de Voto": "xochitl_voting_intention",
        "Proporción de Interacción Final": "xochitl_instagram_like_count_share"
    }
    prep_result = 27.91
else:  # Jorge Álvarez Maynez
    metrics = {            
        "Último Valor en Encuestas": "maynez_voting_intention",
        "Proyección de Intención de Voto": "maynez_voting_intention",
        "Proporción de Interacción Final": "maynez_num_interaction_share"
    }
    prep_result = 10.42

# Filtrar datos para las fechas relevantes
filtered_data = pollster_data[pollster_data["date"].isin(["2024-05-28", "2024-06-02"])]
filtered_data = filtered_data[["date"] + list(metrics.values())].set_index("date")

# Asegurarnos de que filtered_data tiene valores únicos en la fecha seleccionada
def safe_get_value(data, date, metric):
    try:
        value = data.loc[date, metric]
        # Si es una serie con un único valor, convertir a flotante
        if isinstance(value, pd.Series):
            value = value.iloc[0]  # Tomar el primer valor
        return float(value)  # Convertir a escalar
    except KeyError:
        return 0  # Si no existe el valor, devolver 0

# Extraer valores específicos para el gráfico
values = {
    "Último Valor en Encuestas": safe_get_value(filtered_data, "2024-05-28", metrics["Último Valor en Encuestas"]),
    "Proyección de Intención de Voto": safe_get_value(filtered_data, "2024-06-02", metrics["Proyección de Intención de Voto"]),
    "Proporción de Interacción Final": safe_get_value(filtered_data, "2024-05-28", metrics["Proporción de Interacción Final"])        
}

# Crear gráfica de barras verticales
fig = go.Figure()

fig.add_trace(
go.Bar(
    x=list(values.keys()),
    y=list(values.values()),
    text=[f"{v:.2f}%" for v in values.values()],
    textposition="inside",  # Ensure text is outside the bar
    textfont=dict(size=12),  # Adjust text font size if needed
    insidetextanchor='middle',  # Anchor text inside bars
    marker_color=["black", "#ABDFFA", "#3A719B"],  # Colores para cada barra
    textangle=0  # Optional: rotate text if needed
)
)


# Añadir línea horizontal con los resultados oficiales
official_results = {
    "Claudia Sheinbaum": 59.36,
    "Xóchitl Gálvez": 27.91,
    "Jorge Álvarez Máynez": 10.42,
}
fig.add_hline(
    y=official_results[selected_candidate],
    line_dash="dash",
    annotation_text=str(f"Resultado PREP Oficial: {prep_result:.2f}%"),
    annotation_position="top",  # This positions it on the top of the graph
    annotation_yshift = 7,      # This moves the annotation text 10 units down
    line_color="red",
    annotation=dict(
    font=dict(
        color="red"  # Change annotation text color to black
    )
    )
)

# Mostrar gráfico en Streamlit
st.plotly_chart(fig, use_container_width=True)

