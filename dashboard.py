import streamlit as st


Intro = st.Page(
    "ALL/dashboard_pages/Introduction.py",
    title="Introducción",
    icon=":material/home:",
    
)

EDAMethodology = st.Page(
    "ALL/dashboard_pages/EDA_Methodology.py",
    title="Metodología de Exploración",
    icon=":material/help:",
    
)

EDAMethodology_ext = st.Page(
    "ALL/dashboard_pages/EDA_Methodology_ext.py",
    title="Metodología de Exploración - Extendido",
    icon=":material/check_circle:",
)


DataExploration = st.Page(
    "ALL/dashboard_pages/Data_Exploration.py",
    title="Exploración de Datos",
    icon=":material/monitoring:",
)

Clustering = st.Page(
    "ALL/dashboard_pages/Clustering.py",
    title="Clustering por Tema",
    icon=":material/data_usage:",
)

PollsterData = st.Page(
    "ALL/dashboard_pages/Pollster_Data.py",
    title="Resultados y Predicciones Clave",
    icon=":material/how_to_vote:",
)

PollsterData_ext = st.Page(
    "ALL/dashboard_pages/Pollster_Data_ext.py",
    title="Resultados y Predicciones Clave - Extendido",
    icon=":material/how_to_vote:",
)

SentimentAnalysis  = st.Page(         
    "ALL/dashboard_pages/Sentiment_Analysis.py",
    title="Análisis de Sentimiento",
    icon=":material/sentiment_satisfied:",
)

Conclusion  = st.Page(         
    "ALL/dashboard_pages/Conclusion.py",
    title="Conclusión",
    icon=":material/mountain_flag:",
)

AboutUs  = st.Page(         
    "ALL/dashboard_pages/About_us.py",
    title="Sobre Nosotros",
    icon=":material/group:",
)

intro_pages = [Intro,AboutUs]
methodology_pages = [EDAMethodology, EDAMethodology_ext, DataExploration, Clustering]
pollster_pages = [PollsterData, PollsterData_ext]
machinelearning_pages = [SentimentAnalysis]
conclusion_pages = [Conclusion]

st.logo("ALL/OMD_logo.png", icon_image="ALL/OMD_logo.png")

page_dict = {}

page_dict["Inicio"] =  intro_pages
page_dict["Análisis Exploratorio y Descriptivo"] = methodology_pages
page_dict["Resultados Predictivos"] = pollster_pages
page_dict["Machine Learning"] = machinelearning_pages
page_dict["Conclusión"] = conclusion_pages




pg = st.navigation(page_dict)
pg.run()