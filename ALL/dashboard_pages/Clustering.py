import streamlit as st
from PIL import Image

# Configuración de la página
st.title("Clustering de Temas: Campaña Presidencial")
st.write(
    "En esta página se muestran los resultados del análisis de clustering "
    "de temas realizado sobre los posts relacionados con la campaña presidencial antes y después de cada debate."
)

# Explicación del contexto
st.subheader("Contexto del Clustering")
st.write("""
Se utilizaron técnicas avanzadas de procesamiento de texto y análisis de datos para agrupar los temas más relevantes en los días previos y posteriores a cada uno de los tres debates presidenciales. 
Cada análisis produjo cuatro clusters temáticos representativos. A continuación, puedes explorar estos clusters visualmente y observar las palabras más importantes asociadas a cada grupo.
""")

# Filtros
st.sidebar.title("Filtros")
debate_filter = st.sidebar.selectbox(
    "Selecciona el Debate",
    options=["Debate 1", "Debate 2", "Debate 3"],
    index=0
)

time_filter = st.sidebar.selectbox(
    "Selecciona el Momento",
    options=["Antes", "Después"],
    index=0
)

centroids_filter = st.sidebar.selectbox(
    "Visualización de Centroides",
    options=["Con Centroides", "Sin Centroides"],
    index=0
)

# Mapeo de imágenes (asegúrate de tener estas imágenes en tu carpeta de proyecto)
image_paths = {
    "Debate 1": {
        "Antes": {
            "Con Centroides": "images/clustering/debate_1_antes_centroides.png",
            "Sin Centroides": "images/clustering/debate_1_antes.png"
        },
        "Después": {
            "Con Centroides": "images/clustering/debate_1_despues_centroides.png",
            "Sin Centroides": "images/clustering/debate_1_despues.png"
        }
    },
    "Debate 2": {
        "Antes": {
            "Con Centroides": "images/clustering/debate_2_antes_centroides.png",
            "Sin Centroides": "images/clustering/debate_2_antes.png"
        },
        "Después": {
            "Con Centroides": "images/clustering/debate_2_despues_centroides.png",
            "Sin Centroides": "images/clustering/debate_2_despues.png"
        }
    },
    "Debate 3": {
        "Antes": {
            "Con Centroides": "images/clustering/debate_3_antes_centroides.png",
            "Sin Centroides": "images/clustering/debate_3_antes.png"
        },
        "Después": {
            "Con Centroides": "images/clustering/debate_3_despues_centroides.png",
            "Sin Centroides": "images/clustering/debate_3_despues.png"
        }
    }
}

# Mostrar imagen seleccionada
selected_image_path = image_paths[debate_filter][time_filter][centroids_filter]
st.subheader(f"Visualización del Clustering ({debate_filter} - {time_filter} - {centroids_filter})")

try:
    image = Image.open(selected_image_path)
    st.image(image, caption=f"{debate_filter} - {time_filter} - {centroids_filter}", use_column_width=True)
except FileNotFoundError:
    st.error(f"No se encontró la imagen en la ruta: {selected_image_path}. Verifica que el archivo exista.")

# Mostrar palabras clave para los clusters
st.subheader("Palabras más importantes por Cluster")
words = {
    "Debate 1": {
        "Antes": [
            "ecuador, poder, hijo, político, domingo, mañana, presidenta...",
            "soldado, callate, vas, borrachin, falta, payaso, cuanto...",
            "presidenta, mañana, apoyar, pais, propuesta, seguir, político...",
            "sheinbaum, claudia, ganar, mañana, buscar, voto, madre..."
        ],
        "Después": [
            "gano, gobierno, xochitl, campaña, millón, poder, mujer, encuesta...",
            "gano, ganar, xochitl, poder, claudia sheinbaum, presidenta, mentir...",
            "burro, lenguaje, estupidez, final, esquirol, obvio, sorprender...",
            "claudia, presidenta, ganar, mujer, corrupción, salud, gobierno..."
        ]
    },
    "Debate 2": {
        "Antes": [
            "voto, poder, xochitl, ganar, presidenta, mujer, segundo, político...",
            "voto, xochitl, millón, pendejo, video, corrupción, gobierno...",
            "jajaja, rata, chario, completo, ajiu, payaso, detener, pobre, olvidar...",
            "votar, presidenta, segundo, mañana, junio, gente, domingo, gobierno..."
        ],
        "Después": [
            "decir, ganar, xochitl, segundo, mexico, candidata...",
            "callate, jajaja, metio, ciclista, accidente, ratota, rudo, pendeja...",
            "debate, xochitl, claudia, hacer, corrupción, ayer, segundo, casa...",
            "mexico, ver, votar, claudia, candidato, xochitl, ser, mejor, ganar..."
        ]
    },
    "Debate 3": {
        "Antes": [
            "votar, junio, gobierno, ganar, nuevo, poder, creer, encuesta, hijo...",
            "rosa, xochitl, votar, ganar, politico, presidenta, poder, voto, junio...",
            "domingo, gobierno, tercer, mañana, partido, encuesta, color, mayo...",
            "chillar, callar, miente, jajajaj, uy, suficiente, pobre, cobarde, empate, libro..."
        ],
        "Después": [
            "tercer, ganar, presidenta, xochitl, votar, lugar, ultimo...",
            "bloqueaste, pensar, madre, pendejo, confirmar, vaya, borraco, mucho...",
            "xochitl, presidenta, gano, poder, politico, tercer, pais, seguridad...",
            "claudia, gano, tercer, junio, morena, maynez, seguir, historia, proxima..."
        ]
    }
}

selected_words = words[debate_filter][time_filter]
for cluster_num, word_list in enumerate(selected_words):
    st.markdown(f"**Cluster {cluster_num}:** {word_list}")
