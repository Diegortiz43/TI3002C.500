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

# Configuración inicial de la sesión
if 'page' not in st.session_state:
    st.session_state['page'] = "Introducción"  # Página predeterminada al iniciar

# Configuración general del Dashboard
st.set_page_config(
    page_title="Análisis de Métricas de Redes Sociales",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    st.image(
        "ALL/OMD_logo.png",  # Cambia esta URL por tu imagen
        use_column_width=False,
    )
    st.title("Navegación")
    selected_page = st.radio(
        "Páginas",
        options=[
            "Introducción",
            "Metodología de exploración",
            "Metodología de exploración +",
            "Exploración de datos",
            "Resultados y Predicciones Clave",
            "Resultados y Predicciones Clave (detalles)",
            "Análisis de sentimiento",
            "Conclusiones",
            "Sobre nosotros"
        ],
        index=["Introducción", "Metodología de exploración", "Metodología de exploración +","Exploración de datos",
               "Resultados y Predicciones Clave", "Resultados y Predicciones Clave (detalles)","Análisis de sentimiento", "Conclusiones", "Sobre nosotros"].index(st.session_state['page']),
    )

    # Actualizar la página en st.session_state cuando se selecciona en el sidebar
    st.session_state['page'] = selected_page

# Usar el estado actualizado para mostrar la página correspondiente
page = st.session_state['page']

# Página: Introduction
if page == "Introducción":
    st.title("📊 Introducción")
    
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

    # Botón para avanzar
    if st.button("Ir a la siguiente página"):
        st.session_state['page'] = "Metodología de exploración"





























### Metodología de exploración
# Página: Metodología de exploración
elif page == "Metodología de exploración":
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

    st.write("Para una descripción más detallada de el procedimiento que siguió el equipo:")
    # Botón para avanzar
    if st.button("Más información"):
        st.session_state['page'] = "Metodología de exploración +"


    st.write("De lo contrario, continúa hacia Exploración de datos")

    # Botón para avanzar
    if st.button("Siguiente sección"):
        st.session_state['page'] = "Exploración de datos"
























### Metodología de exploración +
# Página: Metodología de exploración +
elif page == "Metodología de exploración +":
    st.title("Metodología de exploración +")

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

    if st.button("Siguiente sección"):
        st.session_state['page'] = "Exploración de datos"






















        ### Descriptive Statistics

        # Página: Descriptive Statistics
elif page == "Exploración de datos":
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



    # Botón para avanzar
    if st.button("Ir a la siguiente página"):
        st.session_state['page'] = "Datos de encuestadora"





















# Página: Resultados y Predicciones Clave
elif page == "Resultados y Predicciones Clave":
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
        "maynez_num_interaction_share": "Interaction Share",
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
        "Interaction Share": "#FF8300",                            # Jorge Álvarez Máynez
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
        annotation_text=str(f"Official PREP Result: {prep_result:.2f}%"),
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


    st.write("A continuación les brindamos la oportunidad de experimentar con **todas las metricas independientes** por **Plataforma** y **Candidato**:")
        # Botón para avanzar
    if st.button("Más información"):
        st.session_state['page'] = "Datos de encuestadora (detalles)"


    st.write("De lo contrario, puede acceder a la siguiente sección dando click en el botón:")
    if st.button("Ir a la siguiente página"):
        st.session_state['page'] = "Análisis de sentimiento"


















# Página: Resultados y Predicciones Clave (detalles)
if page == "Resultados y Predicciones Clave (detalles)":
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






















# Página: Análisis de sentimiento
elif page == "Análisis de sentimiento":
    st.title("📊 Análisis de sentimiento")
    st.write("A partir de un modelo SVM Linear realizamos un análisis de sentimiento que categorizó cada publicación con un sentimiento **Positivo**, **Neutral** o **Negativo**. La siguiente gráfica muestra las proporciones de sentimientos en cada categoría o candidato seleccionado.")

    df = pd.read_csv("ALL/final_datasets/SA_df.csv")

    def selectDataByDate(df, start_date, end_date):
        # Convert the 'datetime' column to datetime format, ignoring invalid values
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')

        # Adjust the time to Mexico timezone by subtracting 6 hours
        df['datetime'] = df['datetime'] - timedelta(hours=6)

        # Check how many values were converted to NaT (invalid values)
        invalid_dates = df['datetime'].isna().sum()
        print(f"We found {invalid_dates} invalid values in the 'datetime' column.")

        # Check the range of valid dates
        valid_dates = df['datetime'].dropna()
        min_date = valid_dates.min()
        max_date = valid_dates.max()
        print(f"Range of valid dates: {min_date} to {max_date}")

        # Ensure that the provided date range overlaps with the data's date range
        if end_date < min_date or start_date > max_date:
            print("Warning: The specified date range does not overlap with the data's date range.")
            return pd.DataFrame()  # Return an empty DataFrame if there's no overlap

        # Filter the data within the specified date range
        filtered_df = df[(df['datetime'] >= start_date) & (df['datetime'] <= end_date)]
        print(f"Number of rows in the filtered DataFrame: {len(filtered_df)}")

        return filtered_df

    start_date= pd.to_datetime('2024-01-01 00:00:00')
    end_date= pd.to_datetime('2024-07-31 23:59:59')

    df = selectDataByDate(df, start_date, end_date)

    #Encode sentiment
    df['Sentiment'] = df['Sentiment'].map({'Negativo': -1, 'Neutral': 0, 'Positivo': 1})

    # Define the candidates to include
    candidates = ['Xóchitl Gálvez', 'Claudia Sheinbaum', 'Jorge Álvarez Máynez']

# ------------------ New Section for Pie Chart ------------------

    st.subheader("Análisis Proporcional de Sentimientos")
    pie_mode = st.radio(
        "Elige el tipo de análisis:",
        options=[
            "Ver proporción de sentimientos de un candidato",
            "Ver proporción de un sentimiento entre candidatos"
        ],
        index=0
    )

    if pie_mode == "Ver proporción de sentimientos de un candidato":
        selected_candidate = st.selectbox("Selecciona un candidato:", candidates)
        candidate_data = df[df['candidate_name'] == selected_candidate]
        sentiment_proportion = candidate_data['Sentiment'].value_counts(normalize=True) * 100
        sentiment_proportion = sentiment_proportion.rename({-1: 'Negativo', 0: 'Neutral', 1: 'Positivo'})
        fig = px.pie(
            sentiment_proportion,
            values=sentiment_proportion.values,
            names=sentiment_proportion.index,
            title=f"Proporción de Sentimientos - {selected_candidate}",
            color=sentiment_proportion.index,
            color_discrete_map={
                "Negativo": "#CF2E23",
                "Neutral": "#A59492",
                "Positivo": "#42A537"
            }
        )
        fig.update_traces(
            textfont_size=14,  # Size of labels inside the pie
            textinfo='percent+label'  # Display percentage and label
        )
    else:
        selected_sentiment = st.selectbox("Selecciona un sentimiento:", ['Negativo', 'Neutral', 'Positivo'])
        sentiment_mapping = {'Negativo': -1, 'Neutral': 0, 'Positivo': 1}
        sentiment_value = sentiment_mapping[selected_sentiment]
        sentiment_data = df[df['Sentiment'] == sentiment_value]
        sentiment_by_candidate = sentiment_data['candidate_name'].value_counts(normalize=True) * 100
        fig = px.pie(
            sentiment_by_candidate,
            values=sentiment_by_candidate.values,
            names=sentiment_by_candidate.index,
            title=f"Proporción de {selected_sentiment} entre los candidatos",
            color=sentiment_by_candidate.index,
            color_discrete_map={
                'Claudia Sheinbaum': '#741D23',
                'Xóchitl Gálvez': '#1E75BC',
                'Jorge Álvarez Máynez': '#FF8300'
            }
        )
        fig.update_traces(
            textfont_size=14,  # Size of labels inside the pie
            textinfo='percent+label'  # Display percentage and label
        )

    fig.update_layout(
        legend=dict(
            font=dict(size=16)  # Size of labels outside the pie (legend)
        )
    )

    st.plotly_chart(fig)


# -------------------------- End of pie chart --------------------------

    # Filter data for the selected candidates
    df_filtered = df[df['candidate_name'].isin(candidates)]

    # Group by date and candidate, calculating the mean sentiment values
    sentiment_per_day = df_filtered.groupby(
        [df_filtered['datetime'].dt.date, 'candidate_name']
    )['Sentiment'].mean().unstack()

    # Calculate total sentiment per day across all candidates
    total_sentiment_per_day = sentiment_per_day.sum(axis=1)

    # Define colors for each candidate
    colors = {
        'Claudia Sheinbaum': '#741D23',
        'Xóchitl Gálvez': '#1E75BC',
        'Jorge Álvarez Máynez': '#FF8300'
    }

    st.subheader("Tendencia de Sentimiento Promedio")
    st.write("A continuación se presenta una gráfica de sentimiento promedio, importante recordar que 1 es positivo y -1 negativo. Recuerde que puede hacer click al nombre de algún candidato para desactivar/activar su línea temporal")

    # Create a plotly figure
    fig = go.Figure()

    # Plot lines for each candidate
    for candidate in sentiment_per_day.columns:
        fig.add_trace(go.Scatter(
            x=sentiment_per_day.index,
            y=sentiment_per_day[candidate],
            mode='lines',
            line=dict(color=colors.get(candidate, 'gray'), width=2),
            name=candidate
        ))

    # Update layout
    fig.update_layout(
        title="Tendencia de sentimientos (diario)",
        title_x=0.5,
        xaxis=dict(
            title='Fecha',
            tickformat='%b %Y',
            type='category',
            showgrid=False
        ),
        yaxis=dict(
            title='Sentimiento promedio',
            showgrid=True
        ),
        showlegend=True,
        plot_bgcolor='white',
        hovermode="x unified"
    )

    # Show the plot in Streamlit
    st.plotly_chart(fig)


### Texto Sentiment
    st.subheader("Prueba Nuestro Modelo")
    st.write("Introduce una oración personalizada para probar el funcionamiento del modelo de Sentiment Analysis.")


    # Cargar el modelo preentrenado
    my_best_model_loaded = joblib.load("ALL/sa_modeling/best_model.joblib")  # Cambia la ruta a tu modelo

    # Configuración de la página en Streamlit
    st.title("Análisis de Sentimiento")
    st.write("Ingrese un texto para analizar el sentimiento utilizando el modelo entrenado.")

    # Entrada de texto del usuario
    user_input = st.text_area("Escribe tu texto aquí:", "")

    # Si el usuario ha ingresado texto
    if user_input.strip():
        try:
            # Configurar el modelo TF-IDF (sin un tokenizer personalizado)
            max_Features = 200
            TFIDF = TfidfVectorizer(
                stop_words=[],  # Lista de stopwords (vacía en este caso)
                ngram_range=(1, 3),  # N-grams
                max_features=max_Features  # Máximo número de características
            )
            
            # Dato importante:
            # Si usaste un TF-IDF específico durante el entrenamiento, deberás cargar ese vectorizador.
            # Aquí estamos ajustando uno nuevo, lo cual funcionará siempre que el modelo pueda manejar las mismas características.

            # Ajustar y transformar el texto ingresado por el usuario
            user_tfidf = TFIDF.fit_transform([user_input])  # Ajuste y transformación del texto
            
            # Realizar la predicción
            predicted_class = my_best_model_loaded.predict(user_tfidf)
            
            # Determinar el sentimiento
            if predicted_class[0] == 0:
                sentiment = "Negativo"
            elif predicted_class[0] == 1:
                sentiment = "Neutral"
            else:
                sentiment = "Positivo"
            
            # Mostrar el resultado al usuario
            st.subheader("Resultado del análisis de sentimiento:")
            st.success(f"Sentimiento del texto: **{sentiment}**")
        
        except Exception as e:
            st.error(f"Error al analizar el texto: {e}")
    else:
        st.write("Por favor, ingrese un texto para comenzar el análisis.")




























### Conclusiones
elif page == "Conclusiones":
    st.title("📊 Conclusiones")

    st.write("""
    ### 8. Conclusión

    Este proyecto ha proporcionado una demostración detallada y exitosa del uso de métricas de redes sociales como una herramienta predictiva para anticipar la intención de voto en contextos electorales. A través de un análisis exhaustivo que combinó datos históricos y modelos de regresión lineal simple, se logró establecer una conexión sólida entre indicadores clave de redes sociales —como la proporción de interacciones, likes y comentarios— y la intención de voto reflejada en encuestas. Este enfoque permitió no solo interpretar las tendencias en redes sociales, sino también proyectar posibles resultados electorales con un grado notable de precisión.

    #### Principales hallazgos por candidato:

    - **Claudia Sheinbaum**:  
    Se identificó una correlación robusta entre las métricas de Instagram y la intención de voto. En particular, los datos sugieren que el volumen y la naturaleza de las interacciones en esta plataforma reflejan consistentemente cambios en el apoyo público hacia la candidata. Las proyecciones realizadas basadas en estas métricas se alinearon estrechamente con los resultados oficiales, destacando la capacidad del modelo para capturar las fluctuaciones en el sentimiento del electorado a lo largo del periodo de análisis.

    - **Xóchitl Gálvez**:  
    Los resultados indicaron una fuerte conexión entre el engagement en Instagram y las intenciones de voto para esta candidata. A través de las métricas seleccionadas, se logró identificar patrones de comportamiento en redes sociales que resultaron ser predictores significativos de las preferencias reales del electorado. Este análisis subraya el impacto de la actividad en redes sociales en la percepción pública y su utilidad como herramienta para medir el apoyo popular.

    - **Álvarez Máynez**:  
    Para este candidato, los modelos destacaron un aumento notable en la intención de voto por cada incremento en la proporción de interacciones totales. Este hallazgo pone de manifiesto el papel clave que juega el engagement digital como un indicador temprano de crecimiento en apoyo electoral. Además, las métricas utilizadas para este análisis destacaron la capacidad de las redes sociales para reflejar de manera precisa la evolución de la popularidad de candidatos emergentes.

    ---

    #### Implicaciones metodológicas y prácticas:

    El uso de redes sociales como fuente de datos para estudios electorales no solo es innovador, sino que también representa una herramienta poderosa para complementar las encuestas tradicionales. Este proyecto demostró cómo los datos de interacción digital pueden proporcionar información adicional que, cuando se utiliza correctamente, mejora la capacidad de anticipar los resultados electorales. Las siguientes conclusiones destacan los aportes más significativos del análisis:

    1. **Correlación significativa entre métricas digitales y comportamiento electoral:**  
    Los resultados obtenidos mostraron una fuerte relación entre indicadores clave de redes sociales y la intención de voto, lo que valida la relevancia de estas métricas en el análisis político moderno.

    2. **Visualización de valores reales vs. predichos:**  
    Las proyecciones realizadas no solo fueron precisas, sino que también ofrecieron visualizaciones claras e intuitivas que permitieron interpretar los resultados de manera efectiva. Este enfoque facilita el análisis tanto para equipos técnicos como para actores políticos.

    3. **Validación con datos oficiales:**  
    Al comparar las proyecciones con resultados oficiales, se pudo confirmar la precisión de los modelos en general, lo que respalda la confiabilidad del enfoque utilizado.

    4. **Potencial para futuras mejoras:**  
    Si bien los modelos produjeron resultados sólidos, se identificaron áreas de mejora. Por ejemplo, la incorporación de relaciones no lineales o el uso de variables externas, como el análisis de sentimientos y la cobertura mediática, podrían aumentar la precisión y profundidad del análisis.

    ---

    #### Desafíos y oportunidades para el futuro:

    El análisis reveló algunos desafíos inherentes al uso de métricas de redes sociales como indicadores de comportamiento electoral. Uno de los principales desafíos es la variabilidad en los datos de redes sociales, que pueden estar influenciados por factores externos, como campañas mediáticas o eventos virales. Sin embargo, estos desafíos también representan oportunidades para expandir el alcance del análisis. Por ejemplo:

    - **Incorporación de aprendizaje automático:**  
    La utilización de modelos más avanzados podría mejorar la capacidad predictiva, permitiendo la identificación de patrones complejos en los datos.

    - **Integración de nuevas fuentes de datos:**  
    Añadir datos de búsquedas en Google, menciones en noticias y análisis de sentimientos podría enriquecer aún más el modelo y proporcionar una visión más integral.

    - **Estudios longitudinales:**  
    Realizar análisis similares en diferentes contextos electorales y comparar los resultados podría validar aún más el enfoque y generalizar su aplicabilidad.

    ---

    #### Reflexión final:

    En conclusión, este proyecto ofrece un marco integral para el uso de datos de redes sociales en estudios electorales. Al demostrar el potencial predictivo de las métricas digitales, abre nuevas posibilidades para estrategias de campaña, mejora de la precisión en encuestas y monitoreo de la opinión pública. Los resultados obtenidos subrayan la importancia de integrar tecnologías digitales y análisis estadísticos para comprender mejor el comportamiento electoral en el siglo XXI. Con mejoras y refinamientos futuros, este enfoque puede convertirse en una herramienta estándar para investigadores, analistas y estrategas políticos.
    """)




























elif page == "Sobre nosotros":
    st.title("Sobre nosotros")

    # Introduction
    st.markdown("""
    Somos el **Equipo 5**, un grupo de estudiantes multidiverso cursando el séptimo semestre en el Instituto Tecnológico y de Estudios Superiores Monterrey. 
    Este proyecto refleja nuestro esfuerzo conjunto por aplicar nuestras habilidades en análisis de datos, 
    visualización y machine learning para abordar un tema de impacto social.
    """)

    # Display team members and images
    st.subheader("Nuestro Equipo")

    team_members = [
        {"name": "Diego Ortiz Puente", "image": "images/diego.jpg", "description": "Estudiante de la Licenciatura en Finanzas."},
        {"name": "Mateo Lizárraga Alcocer", "image": "images/mateo.jpeg", "description": "Estudiante de Ingeniería Industrial."},
        {"name": "Paola Fernanda García Álvarez", "image": "images/pao.jpeg", "description": "Estudiante de la Licenciatura en Mercadotecnia."},
        {"name": "Alonso Viveros Hernández", "image": "images/alonso.png", "description": "Estudiante de Ingeniería en Biotecnología."},
        {"name": "Michelle Pascal Morales", "image": "images/mich.png", "description": "Estudiante de Ingeniería Industrial."},
        {"name": "Artemio Cancino Labastida", "image": "images/artemio.png", "description": "Estudiante de la Licenciatura en Finanzas."},
    ]

    # Loop to display each team member
    cols = st.columns(3)  # Use 3 columns to display images side by side
    for i, member in enumerate(team_members):
        with cols[i % 3]:  # Cycle through columns
            st.image(member["image"], caption=member["name"], use_column_width=True)
            st.markdown(f"**{member['name']}**")
            st.write(member["description"])

    # Closing remarks
    st.markdown("""
    Nos apasiona el análisis de datos y la búsqueda de soluciones innovadoras para problemas complejos. 
    Agradecemos tu interés en nuestro proyecto y estamos emocionados de compartir nuestro trabajo contigo.
    """)

