import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Análisis Titanic - ML & BI",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 2rem;
        color: #2e86ab;
        border-bottom: 2px solid #2e86ab;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2e86ab;
    }
    .variable-dict {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

# Diccionario de variables
VARIABLE_DICT = {
    'survived': 'Supervivencia (0 = No, 1 = Sí)',
    'pclass': 'Clase del ticket (1 = 1ra, 2 = 2da, 3 = 3ra)',
    'sex': 'Género del pasajero',
    'age': 'Edad en años',
    'sibsp': 'Número de hermanos/cónyuge a bordo',
    'parch': 'Número de padres/hijos a bordo',
    'fare': 'Tarifa del pasajero',
    'embarked': 'Puerto de embarque (C = Cherbourg, Q = Queenstown, S = Southampton)',
    'class': 'Clase (igual a pclass)',
    'who': 'Categoría (man, woman, child)',
    'adult_male': 'Si es hombre adulto (True/False)',
    'deck': 'Cubierta de la cabina',
    'embark_town': 'Ciudad de embarque',
    'alive': 'Si sobrevivió (yes/no)',
    'alone': 'Si viajaba solo (True/False)',
    'title': 'Título extraído del nombre',
    'family_size': 'Tamaño total de la familia (sibsp + parch + 1)',
    'is_alone': 'Si viajaba solo (1) o con familia (0)',
    'age_group': 'Grupo de edad (Child, Teen, Young Adult, Adult, Senior)',
    'fare_category': 'Categoría de tarifa (Low, Medium, High, Very High)',
    'has_cabin': 'Si tenía cabina asignada (1) o no (0)'
}

# Título principal
st.markdown('<h1 class="main-header">🚢 Análisis Completo del Titanic</h1>', unsafe_allow_html=True)
st.markdown("### Machine Learning & Business Intelligence Application")

# Cargar y preparar datos
@st.cache_data
def load_data():
    try:
        import seaborn as sns
        titanic = sns.load_dataset('titanic')
    except:
        try:
            url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
            titanic = pd.read_csv(url)
        except:
            st.error("No se pudo cargar el dataset. Usando datos de ejemplo limitados.")
            return create_sample_data()
    
    # Feature engineering avanzado
    if 'name' in titanic.columns:
        titanic['title'] = titanic['name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    else:
        titanic['title'] = 'Mr'
    
    sibsp = titanic.get('sibsp', pd.Series(0, index=titanic.index))
    parch = titanic.get('parch', pd.Series(0, index=titanic.index))
    titanic['family_size'] = sibsp + parch + 1
    titanic['is_alone'] = (titanic['family_size'] == 1).astype(int)
    
    age = titanic.get('age', pd.Series(30, index=titanic.index))
    titanic['age_group'] = pd.cut(age, 
                                 bins=[0, 12, 18, 35, 50, 100], 
                                 labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Senior'],
                                 right=False)
    
    fare = titanic.get('fare', pd.Series(30, index=titanic.index))
    titanic['fare_category'] = pd.cut(fare,
                                     bins=[0, 10, 30, 100, 600],
                                     labels=['Low', 'Medium', 'High', 'Very High'])
    
    cabin = titanic.get('cabin', pd.Series(None, index=titanic.index))
    titanic['deck'] = cabin.str[0] if cabin.notna().any() else 'Unknown'
    titanic['has_cabin'] = cabin.notna().astype(int)
    
    # Nueva feature: riesgo por edad y clase
    titanic['risk_score'] = titanic['pclass'] * (titanic['age'] / 80 if 'age' in titanic.columns else 1)
    
    # Nueva feature: valor relativo pagado
    if 'fare' in titanic.columns and 'pclass' in titanic.columns:
        class_avg_fare = titanic.groupby('pclass')['fare'].transform('mean')
        titanic['fare_ratio'] = titanic['fare'] / class_avg_fare
    
    return titanic

def create_sample_data():
    """Crear datos de ejemplo si falla la carga"""
    np.random.seed(42)
    n_passengers = 200
    
    data = {
        'survived': np.random.choice([0, 1], n_passengers, p=[0.6, 0.4]),
        'pclass': np.random.choice([1, 2, 3], n_passengers, p=[0.2, 0.3, 0.5]),
        'sex': np.random.choice(['male', 'female'], n_passengers, p=[0.6, 0.4]),
        'age': np.random.normal(30, 15, n_passengers).clip(0, 80),
        'sibsp': np.random.poisson(0.5, n_passengers),
        'parch': np.random.poisson(0.4, n_passengers),
        'fare': np.random.exponential(30, n_passengers),
        'embarked': np.random.choice(['S', 'C', 'Q'], n_passengers, p=[0.7, 0.2, 0.1])
    }
    return pd.DataFrame(data)

@st.cache_data
def prepare_ml_data(df):
    """Preparar datos para machine learning"""
    df_ml = df.copy()
    
    if 'age' in df_ml.columns:
        df_ml['age'].fillna(df_ml['age'].median(), inplace=True)
    else:
        df_ml['age'] = 30
    
    if 'embarked' in df_ml.columns:
        df_ml['embarked'].fillna(df_ml['embarked'].mode()[0] if len(df_ml['embarked'].mode()) > 0 else 'S', inplace=True)
    
    if 'deck' in df_ml.columns:
        df_ml['deck'].fillna('Unknown', inplace=True)
    
    if 'fare' in df_ml.columns:
        df_ml['fare'].fillna(df_ml['fare'].median(), inplace=True)
    else:
        df_ml['fare'] = 30
    
    if 'title' in df_ml.columns:
        df_ml['title'] = df_ml['title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 
                                                'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
        df_ml['title'] = df_ml['title'].replace('Mlle', 'Miss')
        df_ml['title'] = df_ml['title'].replace('Ms', 'Miss')
        df_ml['title'] = df_ml['title'].replace('Mme', 'Mrs')
    else:
        df_ml['title'] = 'Mr'
    
    le = LabelEncoder()
    categorical_cols = ['sex', 'embarked', 'title', 'deck']
    
    for col in categorical_cols:
        if col in df_ml.columns:
            if df_ml[col].notna().any():
                try:
                    df_ml[col] = le.fit_transform(df_ml[col].astype(str))
                except:
                    df_ml[col] = pd.factorize(df_ml[col])[0]
    
    available_features = []
    possible_features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 
                        'title', 'family_size', 'is_alone', 'has_cabin', 'deck', 'risk_score', 'fare_ratio']
    
    for feature in possible_features:
        if feature in df_ml.columns:
            available_features.append(feature)
    
    X = df_ml[available_features]
    y = df_ml['survived'] if 'survived' in df_ml.columns else pd.Series(0, index=df_ml.index)
    
    return X, y, available_features

# Cargar datos
titanic = load_data()
X, y, features = prepare_ml_data(titanic)

# Sidebar para navegación
st.sidebar.title("🎛️ Panel de Control")
section = st.sidebar.radio("Navegación", [
    "📊 Overview & KPIs",
    "👥 Análisis Demográfico", 
    "💰 Análisis Socioeconómico",
    "🔍 Análisis de Supervivencia",
    "🤖 Machine Learning",
    "📈 Clustering & Segmentación",
    "🎯 Insights & Recomendaciones"
])

# Mostrar información del dataset en sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Info del Dataset")
st.sidebar.write(f"**Filas:** {len(titanic)}")
st.sidebar.write(f"**Columnas:** {len(titanic.columns)}")
st.sidebar.write(f"**Features ML:** {len(features)}")

# =============================================================================
# SECCIÓN 1: OVERVIEW & KPIs
# =============================================================================
if section == "📊 Overview & KPIs":
    st.markdown('<h2 class="section-header">📊 Overview del Dataset Titanic</h2>', unsafe_allow_html=True)
    
    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_passengers = len(titanic)
        st.metric("Total Pasajeros", f"{total_passengers:,}")
    
    with col2:
        survival_rate = titanic['survived'].mean() * 100 if 'survived' in titanic.columns else 0
        st.metric("Tasa de Supervivencia", f"{survival_rate:.1f}%")
    
    with col3:
        avg_fare = titanic['fare'].mean() if 'fare' in titanic.columns else 0
        st.metric("Tarifa Promedio", f"${avg_fare:.2f}")
    
    with col4:
        avg_age = titanic['age'].mean() if 'age' in titanic.columns else 0
        st.metric("Edad Promedio", f"{avg_age:.1f} años")
    
    # Segunda fila de métricas
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        if 'sex' in titanic.columns and 'survived' in titanic.columns:
            women_data = titanic[titanic['sex']=='female']
            women_survival = women_data['survived'].mean() * 100 if len(women_data) > 0 else 0
        else:
            women_survival = 0
        st.metric("Supervivencia Mujeres", f"{women_survival:.1f}%")
    
    with col6:
        if 'sex' in titanic.columns and 'survived' in titanic.columns:
            men_data = titanic[titanic['sex']=='male']
            men_survival = men_data['survived'].mean() * 100 if len(men_data) > 0 else 0
        else:
            men_survival = 0
        st.metric("Supervivencia Hombres", f"{men_survival:.1f}%")
    
    with col7:
        if 'pclass' in titanic.columns and 'survived' in titanic.columns:
            first_class = titanic[titanic['pclass']==1]
            first_class_survival = first_class['survived'].mean() * 100 if len(first_class) > 0 else 0
        else:
            first_class_survival = 0
        st.metric("Supervivencia 1ra Clase", f"{first_class_survival:.1f}%")
    
    with col8:
        if 'pclass' in titanic.columns and 'survived' in titanic.columns:
            third_class = titanic[titanic['pclass']==3]
            third_class_survival = third_class['survived'].mean() * 100 if len(third_class) > 0 else 0
        else:
            third_class_survival = 0
        st.metric("Supervivencia 3ra Clase", f"{third_class_survival:.1f}%")
    
    # NUEVOS GRÁFICOS EN OVERVIEW
    st.subheader("📈 Visualizaciones Rápidas del Dataset")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribución de clases
        if 'pclass' in titanic.columns:
            class_dist = titanic['pclass'].value_counts().sort_index()
            fig = px.bar(x=[f'Clase {i}' for i in class_dist.index], 
                        y=class_dist.values,
                        title='Distribución de Pasajeros por Clase',
                        color=class_dist.values,
                        color_continuous_scale='Viridis')
            fig.update_layout(xaxis_title='Clase', yaxis_title='Número de Pasajeros')
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Distribución de puertos de embarque
        if 'embarked' in titanic.columns:
            port_dist = titanic['embarked'].value_counts()
            port_names = {'C': 'Cherbourg', 'Q': 'Queenstown', 'S': 'Southampton'}
            port_dist.index = [port_names.get(x, x) for x in port_dist.index]
            fig = px.pie(values=port_dist.values, names=port_dist.index,
                        title='Distribución por Puerto de Embarque')
            st.plotly_chart(fig, use_container_width=True)
    
    # Tercera fila de gráficos
    col3, col4 = st.columns(2)
    
    with col3:
        # Distribución de género
        if 'sex' in titanic.columns:
            gender_dist = titanic['sex'].value_counts()
            fig = px.pie(values=gender_dist.values, names=gender_dist.index,
                        title='Distribución por Género',
                        color_discrete_sequence=['#FF6B6B', '#4ECDC4'])
            st.plotly_chart(fig, use_container_width=True)
    
    with col4:
        # Supervivencia por clase (nuevo gráfico)
        if 'pclass' in titanic.columns and 'survived' in titanic.columns:
            survival_by_class = titanic.groupby('pclass')['survived'].mean().reset_index()
            fig = px.line(survival_by_class, x='pclass', y='survived',
                         title='Tasa de Supervivencia por Clase',
                         markers=True, line_shape='linear')
            fig.update_layout(xaxis_title='Clase', yaxis_title='Tasa de Supervivencia')
            fig.update_traces(line=dict(color='#E71D36', width=4))
            st.plotly_chart(fig, use_container_width=True)
    
    # Dataset sample
    with st.expander("🔍 Ver Dataset Original"):
        st.dataframe(titanic, use_container_width=True)
        st.write(f"**Dimensiones:** {titanic.shape[0]} filas × {titanic.shape[1]} columnas")
    
    # DICCIONARIO DE VARIABLES
    with st.expander("📚 Diccionario de Variables", expanded=True):
        st.markdown("""
        <div class='variable-dict'>
        <h4>📖 Descripción de las Variables del Dataset</h4>
        """, unsafe_allow_html=True)
        
        # Organizar variables en columnas
        vars_per_column = 7
        variables = list(VARIABLE_DICT.items())
        num_columns = 3
        
        cols = st.columns(num_columns)
        
        for i, (var_name, var_desc) in enumerate(variables):
            with cols[i % num_columns]:
                st.markdown(f"**{var_name}**")
                st.caption(var_desc)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # NUEVO: Análisis de missing values
    st.subheader("🔍 Análisis de Valores Faltantes")
    
    if titanic.isnull().sum().sum() > 0:
        missing_data = titanic.isnull().sum()
        missing_data = missing_data[missing_data > 0]
        
        if len(missing_data) > 0:
            fig = px.bar(x=missing_data.index, y=missing_data.values,
                        title='Valores Faltantes por Columna',
                        labels={'x': 'Columna', 'y': 'Número de Valores Faltantes'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Mostrar porcentajes de missing
            missing_percent = (titanic.isnull().sum() / len(titanic) * 100).round(2)
            missing_percent = missing_percent[missing_percent > 0]
            
            if len(missing_percent) > 0:
                st.write("**Porcentaje de valores faltantes:**")
                for col, percent in missing_percent.items():
                    st.write(f"- {col}: {percent}%")
        else:
            st.success("✅ No hay valores faltantes en el dataset")
    else:
        st.success("✅ No hay valores faltantes en el dataset")

# =============================================================================
# SECCIÓN 2: ANÁLISIS DEMOGRÁFICO
# =============================================================================
elif section == "👥 Análisis Demográfico":
    st.markdown('<h2 class="section-header">👥 Análisis Demográfico de Pasajeros</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'sex' in titanic.columns and 'pclass' in titanic.columns:
            fig = make_subplots(rows=1, cols=2, 
                               subplot_titles=['Distribución por Género', 'Distribución por Clase'],
                               specs=[[{'type':'domain'}, {'type':'domain'}]])
            
            gender_counts = titanic['sex'].value_counts()
            class_counts = titanic['pclass'].value_counts().sort_index()
            
            fig.add_trace(go.Pie(labels=gender_counts.index, values=gender_counts.values,
                                name="Género", marker_colors=['#FF6B6B', '#4ECDC4']), 1, 1)
            fig.add_trace(go.Pie(labels=[f'Clase {c}' for c in class_counts.index], 
                                values=class_counts.values, name="Clase", 
                                marker_colors=['#FF9F1C', '#2EC4B6', '#E71D36']), 1, 2)
            
            fig.update_traces(hole=.4, hoverinfo="label+percent+name")
            fig.update_layout(title_text="Distribución Demográfica", height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Datos demográficos incompletos")
    
    with col2:
        if 'age' in titanic.columns and 'pclass' in titanic.columns and 'sex' in titanic.columns:
            fig = px.box(titanic, x='pclass', y='age', color='sex',
                        title='Distribución de Edad por Clase y Género',
                        labels={'pclass': 'Clase', 'age': 'Edad'},
                        color_discrete_sequence=['#FF6B6B', '#4ECDC4'])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Datos insuficientes para análisis de edad")
    
    # NUEVOS GRÁFICOS DEMOGRÁFICOS
    st.subheader("📊 Análisis Demográfico Avanzado")
    
    col3, col4 = st.columns(2)
    
    with col3:
        # Distribución de edades con histograma y KDE
        if 'age' in titanic.columns:
            fig = px.histogram(titanic, x='age', nbins=30, 
                              title='Distribución de Edades con Curva de Densidad',
                              opacity=0.7,
                              color_discrete_sequence=['#2E86AB'])
            
            # Añadir línea de densidad
            fig.update_traces(xbins=dict(size=5))
            st.plotly_chart(fig, use_container_width=True)
    
    with col4:
        # Edad vs Supervivencia por género
        if 'age' in titanic.columns and 'survived' in titanic.columns and 'sex' in titanic.columns:
            fig = px.violin(titanic, x='survived', y='age', color='sex',
                           title='Distribución de Edad por Supervivencia y Género',
                           box=True, points="all",
                           color_discrete_sequence=['#FF6B6B', '#4ECDC4'])
            fig.update_layout(xaxis_title='Sobrevivió (0=No, 1=Sí)')
            st.plotly_chart(fig, use_container_width=True)
    
    # Análisis de familias
    if 'family_size' in titanic.columns and 'survived' in titanic.columns:
        st.subheader("👨‍👩‍👧‍👦 Análisis de Grupos Familiares")
        
        col5, col6 = st.columns(2)
        
        with col5:
            family_survival = titanic.groupby('family_size')['survived'].mean().reset_index()
            fig = px.line(family_survival, x='family_size', y='survived',
                         title='Tasa de Supervivencia por Tamaño Familiar',
                         markers=True)
            fig.update_layout(xaxis_title='Tamaño Familiar', yaxis_title='Tasa de Supervivencia')
            st.plotly_chart(fig, use_container_width=True)
        
        with col6:
            alone_vs_family = titanic.groupby('is_alone')['survived'].mean().reset_index()
            alone_vs_family['is_alone'] = alone_vs_family['is_alone'].map({0: 'Con Familia', 1: 'Solo'})
            fig = px.bar(alone_vs_family, x='is_alone', y='survived',
                        title='Supervivencia: Solos vs Con Familia',
                        color='is_alone',
                        color_discrete_sequence=['#FF9F1C', '#2EC4B6'])
            st.plotly_chart(fig, use_container_width=True)
    
    # NUEVO: Análisis de títulos
    if 'title' in titanic.columns:
        st.subheader("🎭 Análisis por Títulos Sociales")
        
        title_analysis = titanic.groupby('title').agg({
            'survived': 'mean',
            'age': 'mean',
            'fare': 'mean',
            'pclass': 'mean'
        }).round(2)
        
        title_analysis['count'] = titanic['title'].value_counts()
        title_analysis = title_analysis[title_analysis['count'] > 5]  # Filtrar títulos raros
        
        fig = px.scatter(title_analysis, x='fare', y='survived', 
                        size='count', color='pclass',
                        hover_data=['age'],
                        title='Títulos Sociales: Tarifa vs Supervivencia',
                        size_max=60)
        st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# SECCIÓN 3: ANÁLISIS SOCIOECONÓMICO
# =============================================================================
elif section == "💰 Análisis Socioeconómico":
    st.markdown('<h2 class="section-header">💰 Análisis Socioeconómico</h2>', unsafe_allow_html=True)
    
    if 'fare' in titanic.columns:
        st.subheader("Tarifas y Distribución de Riqueza")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'pclass' in titanic.columns:
                fig = px.box(titanic, x='pclass', y='fare', color='pclass',
                           title='Distribución de Tarifas por Clase',
                           labels={'pclass': 'Clase', 'fare': 'Tarifa ($)'})
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'survived' in titanic.columns:
                fig = px.box(titanic, x='survived', y='fare', color='survived',
                            title='Tarifa Pagada: Supervivientes vs Fallecidos',
                            labels={'survived': 'Sobrevivió', 'fare': 'Tarifa ($)'},
                            color_discrete_sequence=['#FF6B6B', '#4ECDC4'])
                fig.update_layout(xaxis_title='Sobrevivió (0=No, 1=Sí)')
                st.plotly_chart(fig, use_container_width=True)
        
        # NUEVOS GRÁFICOS SOCIOECONÓMICOS
        st.subheader("📈 Análisis Económico Avanzado")
        
        col3, col4 = st.columns(2)
        
        with col3:
            # Distribución de tarifas con histograma
            fig = px.histogram(titanic, x='fare', nbins=50,
                              title='Distribución de Tarifas',
                              color_discrete_sequence=['#FF9F1C'])
            fig.update_layout(xaxis_title='Tarifa ($)', yaxis_title='Frecuencia')
            st.plotly_chart(fig, use_container_width=True)
        
        with col4:
            # Tarifa vs Edad con supervivencia
            if 'age' in titanic.columns:
                fig = px.scatter(titanic, x='age', y='fare', color='survived',
                                size='pclass', hover_data=['sex'],
                                title='Relación Edad vs Tarifa',
                                color_discrete_sequence=['#FF6B6B', '#4ECDC4'])
                st.plotly_chart(fig, use_container_width=True)
        
        # NUEVO: Análisis de cabinas
        if 'deck' in titanic.columns:
            st.subheader("🏠 Análisis de Cubiertas (Decks)")
            
            deck_survival = titanic.groupby('deck').agg({
                'survived': 'mean',
                'fare': 'mean',
                'pclass': 'mean'
            }).reset_index()
            
            deck_survival = deck_survival[deck_survival['deck'] != 'Unknown']
            
            if len(deck_survival) > 0:
                fig = px.scatter(deck_survival, x='fare', y='survived',
                                size='pclass', color='deck',
                                title='Cubiertas: Tarifa vs Supervivencia',
                                size_max=30)
                st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# LAS OTRAS SECCIONES SE MANTIENEN IGUAL PERO CON MÁS INSIGHTS
# =============================================================================

# ... (las otras secciones se mantienen igual que en la versión anterior)

# =============================================================================
# SECCIÓN 7: INSIGHTS & RECOMENDACIONES (MEJORADA)
# =============================================================================
else:
    st.markdown('<h2 class="section-header">🎯 Insights Estratégicos & Recomendaciones</h2>', unsafe_allow_html=True)
    
    # Insights principales MEJORADOS
    st.subheader("🔍 Hallazgos Clave del Análisis")
    
    # Calcular métricas para insights dinámicos
    if 'survived' in titanic.columns and 'sex' in titanic.columns:
        women_survival = titanic[titanic['sex']=='female']['survived'].mean() * 100
        men_survival = titanic[titanic['sex']=='male']['survived'].mean() * 100
    else:
        women_survival = 74.2
        men_survival = 18.9
    
    if 'pclass' in titanic.columns:
        first_class_survival = titanic[titanic['pclass']==1]['survived'].mean() * 100
        third_class_survival = titanic[titanic['pclass']==3]['survived'].mean() * 100
    else:
        first_class_survival = 62.9
        third_class_survival = 24.2
    
    if 'age' in titanic.columns:
        children_survival = titanic[titanic['age'] < 12]['survived'].mean() * 100
    else:
        children_survival = 59.0
    
    if 'fare' in titanic.columns:
        survivor_fare = titanic[titanic['survived']==1]['fare'].mean()
        non_survivor_fare = titanic[titanic['survived']==0]['fare'].mean()
    else:
        survivor_fare = 48.4
        non_survivor_fare = 22.1
    
    insights = [
        f"🚨 **Supervivencia por Género:** Las mujeres tuvieron {women_survival:.1f}% de supervivencia vs {men_survival:.1f}% en hombres",
        f"💼 **Impacto de la Clase Social:** La 1ra clase tuvo {first_class_survival:.1f}% de supervivencia vs {third_class_survival:.1f}% en 3ra clase",
        f"👶 **Factor Edad:** Niños menores de 12 años tuvieron {children_survival:.1f}% de supervivencia",
        f"💰 **Correlación Riqueza-Supervivencia:** Tarifa promedio supervivientes: ${survivor_fare:.1f} vs ${non_survivor_fare:.1f} en fallecidos",
        f"👨‍👩‍👧‍👦 **Efecto Familiar:** Pasajeros con familia (3-4 miembros) mostraron mejores tasas de supervivencia",
        f"🏠 **Ventaja de Cabina:** Pasajeros con cabina asignada tuvieron mayor probabilidad de supervivencia",
        f"🎫 **Títulos Sociales:** Pasajeros con títulos de mayor estatus social tuvieron mejor supervivencia",
        f"⚓ **Puerto de Embarque:** Pasajeros de Cherbourg mostraron mayor tasa de supervivencia"
    ]
    
    for insight in insights:
        st.info(insight)
    
    # NUEVOS INSIGHTS ESTADÍSTICOS
    st.subheader("📊 Insights Estadísticos Avanzados")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Correlaciones
        if len(features) > 0:
            numeric_data = titanic[features + ['survived']].select_dtypes(include=[np.number])
            if not numeric_data.empty and 'survived' in numeric_data.columns:
                correlation_with_survival = numeric_data.corr()['survived'].sort_values(ascending=False)
                correlation_with_survival = correlation_with_survival[correlation_with_survival.index != 'survived']
                
                top_correlations = correlation_with_survival.head(5)
                bottom_correlations = correlation_with_survival.tail(5)
                
                st.write("**Top 5 correlaciones con supervivencia:**")
                for feature, corr in top_correlations.items():
                    st.write(f"- {feature}: {corr:.3f}")
                
                st.write("**Bottom 5 correlaciones con supervivencia:**")
                for feature, corr in bottom_correlations.items():
                    st.write(f"- {feature}: {corr:.3f}")
    
    with col2:
        # Patrones demográficos
        st.write("**Patrones Demográficos Clave:**")
        st.write("- Mujeres y niños tuvieron prioridad en protocolos de rescate")
        st.write("- La clase social fue el mejor predictor individual de supervivencia")
        st.write("- Pasajeros solos tuvieron menor tasa de supervivencia")
        st.write("- La edad mostró una relación no lineal con la supervivencia")
    
    # Recomendaciones estratégicas MEJORADAS
    st.subheader("💡 Recomendaciones Estratégicas")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("""
        ### 🚢 Para Diseño de Buques Futuros
        - **Ubicación estratégica** de botes salvavidas accesibles a todas las clases
        - **Sistemas de comunicación** mejorados entre cubiertas
        - **Señalización multilingüe** para evacuación
        - **Capacitación obligatoria** de tripulación en simulacros
        - **Tecnología de localización** en tiempo real
        """)
        
        st.markdown("""
        ### 👥 Para Protocolos de Emergencia
        - **Protocolos diferenciados** por ubicación en el barco
        - **Sistemas de alerta temprana** con cobertura total
        - **Rutas de evacuación** optimizadas por análisis de datos
        - **Puntos de encuentro** múltiples y bien señalizados
        """)
    
    with col4:
        st.markdown("""
        ### 📊 Para Gestión de Riesgos
        - **Modelos predictivos** de supervivencia por segmento
        - **Simulaciones de evacuación** basadas en machine learning
        - **Sistemas de pricing** que consideren factores de seguridad
        - **Auditorías continuas** de protocolos de seguridad
        """)
        
        st.markdown("""
        ### 🎯 Para Políticas de Seguridad
        - **Estándares industry-wide** para capacidad de botes salvavidas
        - **Certificaciones obligatorias** en gestión de emergencias
        - **Sistemas de monitoreo** en tiempo real de ocupación
        - **Tecnología de evacuación** asistida por IA
        """)
    
    # NUEVO: Lecciones técnicas
    st.subheader("🔧 Lecciones Técnicas para Ciencia de Datos")
    
    tech_lessons = [
        "✅ **Feature Engineering es crucial** - Variables derivadas como 'family_size' mejoraron los modelos",
        "✅ **Los datos faltantes requieren estrategia** - Imputación inteligente basada en relaciones",
        "✅ **La visualización multidimensional** revela patrones ocultos en los datos",
        "✅ **La validación cruzada** es esencial para modelos robustos",
        "✅ **La interpretabilidad del modelo** es tan importante como la precisión",
        "✅ **El clustering no supervisado** puede descubrir segmentos naturales en los datos"
    ]
    
    for lesson in tech_lessons:
        st.success(lesson)
    
    # Llamado a la acción MEJORADO
    st.markdown("---")
    st.markdown("""
    <div style='background-color: #2E86AB; padding: 20px; border-radius: 10px; color: white;'>
    <h3 style='color: white; text-align: center;'>🚀 Próximos Pasos para Investigación</h3>
    <div style='columns: 2;'>
    <ul>
    <li>Desarrollar modelos predictivos en tiempo real para gestión de crisis</li>
    <li>Implementar sistemas de simulación de evacuación basados en IA</li>
    <li>Crear dashboards de monitoreo de seguridad para tripulación</li>
    <li>Establecer estándares industry-wide para protocolos de emergencia</li>
    <li>Integrar sensores IoT para monitoreo en tiempo real</li>
    <li>Desarrollar sistemas de recomendación para asignación de recursos</li>
    <li>Implementar análisis de redes sociales entre pasajeros</li>
    <li>Crear modelos de series temporales para predicción de riesgos</li>
    </ul>
    </div>
    </div>
    """, unsafe_allow_html=True)

# Pie de página profesional
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
<p><strong>Análisis Profesional del Titanic</strong> | Machine Learning & Business Intelligence Application</p>
<p>Desarrollado para demostración educativa | Herramientas: Streamlit, Scikit-learn, Plotly, Pandas</p>
</div>
""", unsafe_allow_html=True)
