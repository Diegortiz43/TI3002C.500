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


# ### Texto Sentiment
# st.subheader("Prueba Nuestro Modelo")
# st.write("Introduce una oración personalizada para probar el funcionamiento del modelo de Sentiment Analysis.")


# # Cargar el modelo preentrenado
# my_best_model_loaded = joblib.load("ALL/sa_modeling/best_model.joblib")  # Cambia la ruta a tu modelo

# # Configuración de la página en Streamlit
# st.title("Análisis de Sentimiento")
# st.write("Ingrese un texto para analizar el sentimiento utilizando el modelo entrenado.")

# # Entrada de texto del usuario
# user_input = st.text_area("Escribe tu texto aquí:", "")

# # Si el usuario ha ingresado texto
# if user_input.strip():
#     try:
#         # Configurar el modelo TF-IDF
#         max_Features = 200
#         TFIDF = TfidfVectorizer(
#             stop_words=[],  # Lista de stopwords (vacía en este caso)
#             ngram_range=(1, 3),  # N-grams
#             max_features=max_Features  # Máximo número de características
#         )

#         # Asegurarnos de que el texto tenga suficientes características para el modelo
#         min_features_needed = max_Features - len(user_input.split())
#         if min_features_needed > 0:
#             filler_text = "yes" * 5
#             # Generar palabras neutrales
#             user_input += filler_text  # Añadirlas al texto ingresado

#         # Ajustar y transformar el texto ingresado por el usuario
#         user_tfidf = TFIDF.fit_transform([user_input])  # Ajuste y transformación del texto
        
#         # Realizar la predicción
#         predicted_class = my_best_model_loaded.predict(user_tfidf)
        
#         # Determinar el sentimiento
#         if predicted_class[0] == 0:
#             sentiment = "Negativo"
#         elif predicted_class[0] == 1:
#             sentiment = "Neutral"
#         else:
#             sentiment = "Positivo"
        
#         # Mostrar el resultado al usuario
#         st.subheader("Resultado del análisis de sentimiento:")
#         st.success(f"Sentimiento del texto: **{sentiment}**")
    
#     except Exception as e:
#         st.error(f"Error al analizar el texto: {e}")
# else:
#     st.write("Por favor, ingrese un texto para comenzar el análisis.")

