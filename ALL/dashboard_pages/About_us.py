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