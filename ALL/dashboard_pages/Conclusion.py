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

st.title("📊 Conclusiones")

st.write("""
### Conclusión

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