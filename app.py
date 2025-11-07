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
</style>
""", unsafe_allow_html=True)

# Título principal
st.markdown('<h1 class="main-header">🚢 Análisis Completo del Titanic</h1>', unsafe_allow_html=True)
st.markdown("### Machine Learning & Business Intelligence Application")

# Cargar y preparar datos
@st.cache_data
def load_data():
    try:
        # Intentar cargar desde diferentes fuentes
        import seaborn as sns
        titanic = sns.load_dataset('titanic')
    except:
        try:
            # Fallback: cargar desde URL
            url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
            titanic = pd.read_csv(url)
        except:
            # Último fallback: datos hardcodeados básicos
            st.error("No se pudo cargar el dataset. Usando datos de ejemplo limitados.")
            return create_sample_data()
    
    # Feature engineering avanzado
    if 'name' in titanic.columns:
        titanic['title'] = titanic['name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    else:
        titanic['title'] = 'Mr'  # Valor por defecto
    
    # Calcular family_size de manera segura
    sibsp = titanic.get('sibsp', pd.Series(0, index=titanic.index))
    parch = titanic.get('parch', pd.Series(0, index=titanic.index))
    titanic['family_size'] = sibsp + parch + 1
    titanic['is_alone'] = (titanic['family_size'] == 1).astype(int)
    
    # Crear age_group de manera segura
    age = titanic.get('age', pd.Series(30, index=titanic.index))  # Valor por defecto 30
    titanic['age_group'] = pd.cut(age, 
                                 bins=[0, 12, 18, 35, 50, 100], 
                                 labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Senior'],
                                 right=False)
    
    # Fare category
    fare = titanic.get('fare', pd.Series(30, index=titanic.index))
    titanic['fare_category'] = pd.cut(fare,
                                     bins=[0, 10, 30, 100, 600],
                                     labels=['Low', 'Medium', 'High', 'Very High'])
    
    # Cabin information
    cabin = titanic.get('cabin', pd.Series(None, index=titanic.index))
    titanic['deck'] = cabin.str[0] if cabin.notna().any() else 'Unknown'
    titanic['has_cabin'] = cabin.notna().astype(int)
    
    return titanic

def create_sample_data():
    """Crear datos de ejemplo si falla la carga"""
    data = {
        'survived': [0, 1, 1, 0, 1, 0, 0, 1, 1, 0],
        'pclass': [3, 1, 2, 3, 1, 3, 2, 1, 2, 3],
        'sex': ['male', 'female', 'female', 'male', 'female', 'male', 'male', 'female', 'female', 'male'],
        'age': [22, 38, 26, 35, 35, 28, 54, 2, 27, 14],
        'sibsp': [1, 1, 0, 0, 1, 0, 0, 1, 0, 0],
        'parch': [0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
        'fare': [7.25, 71.28, 10.5, 8.05, 53.1, 8.46, 21.0, 151.55, 15.74, 7.85],
        'embarked': ['S', 'C', 'S', 'S', 'S', 'Q', 'S', 'S', 'S', 'S']
    }
    return pd.DataFrame(data)

@st.cache_data
def prepare_ml_data(df):
    """Preparar datos para machine learning"""
    df_ml = df.copy()
    
    # Handle missing values de manera segura
    if 'age' in df_ml.columns:
        df_ml['age'].fillna(df_ml['age'].median(), inplace=True)
    else:
        df_ml['age'] = 30  # Valor por defecto
    
    if 'embarked' in df_ml.columns:
        df_ml['embarked'].fillna(df_ml['embarked'].mode()[0] if len(df_ml['embarked'].mode()) > 0 else 'S', inplace=True)
    
    if 'deck' in df_ml.columns:
        df_ml['deck'].fillna('Unknown', inplace=True)
    
    if 'fare' in df_ml.columns:
        df_ml['fare'].fillna(df_ml['fare'].median(), inplace=True)
    else:
        df_ml['fare'] = 30  # Valor por defecto
    
    # Feature engineering for ML
    if 'title' in df_ml.columns:
        df_ml['title'] = df_ml['title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 
                                                'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
        df_ml['title'] = df_ml['title'].replace('Mlle', 'Miss')
        df_ml['title'] = df_ml['title'].replace('Ms', 'Miss')
        df_ml['title'] = df_ml['title'].replace('Mme', 'Mrs')
    else:
        df_ml['title'] = 'Mr'  # Valor por defecto
    
    # Encode categorical variables de manera segura
    le = LabelEncoder()
    categorical_cols = ['sex', 'embarked', 'title', 'deck']
    
    for col in categorical_cols:
        if col in df_ml.columns:
            # Verificar que la columna existe y tiene datos
            if df_ml[col].notna().any():
                try:
                    df_ml[col] = le.fit_transform(df_ml[col].astype(str))
                except:
                    # Si falla el encoding, usar valores numéricos simples
                    df_ml[col] = pd.factorize(df_ml[col])[0]
    
    # Select features for ML (solo las que existen)
    available_features = []
    possible_features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 
                        'title', 'family_size', 'is_alone', 'has_cabin', 'deck']
    
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
    
    # Visualización de distribución general
    col1, col2 = st.columns(2)
    
    with col1:
        if 'survived' in titanic.columns:
            survived_counts = titanic['survived'].value_counts()
            fig = px.pie(values=survived_counts.values, 
                        names=['No Sobrevivió', 'Sobrevivió'],
                        title='Distribución de Supervivencia',
                        color_discrete_sequence=['#FF6B6B', '#4ECDC4'])
            fig.update_traces(textinfo='percent+label', pull=[0.1, 0])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Datos de supervivencia no disponibles")
    
    with col2:
        if 'age' in titanic.columns:
            fig = px.histogram(titanic, x='age', nbins=30, title='Distribución de Edades',
                              color_discrete_sequence=['#2E86AB'])
            fig.update_layout(xaxis_title='Edad', yaxis_title='Frecuencia')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Datos de edad no disponibles")
    
    # Dataset sample
    with st.expander("🔍 Ver Dataset Original"):
        st.dataframe(titanic, use_container_width=True)
        st.write(f"**Dimensiones:** {titanic.shape[0]} filas × {titanic.shape[1]} columnas")

# =============================================================================
# SECCIÓN 2: ANÁLISIS DEMOGRÁFICO
# =============================================================================
elif section == "👥 Análisis Demográfico":
    st.markdown('<h2 class="section-header">👥 Análisis Demográfico de Pasajeros</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'sex' in titanic.columns and 'pclass' in titanic.columns:
            # Distribución por género y clase
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
            # Edad por género y clase
            fig = px.box(titanic, x='pclass', y='age', color='sex',
                        title='Distribución de Edad por Clase y Género',
                        labels={'pclass': 'Clase', 'age': 'Edad'},
                        color_discrete_sequence=['#FF6B6B', '#4ECDC4'])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Datos insuficientes para análisis de edad")
    
    # Análisis de familias
    if 'family_size' in titanic.columns and 'survived' in titanic.columns:
        st.subheader("👨‍👩‍👧‍👦 Análisis de Grupos Familiares")
        
        col3, col4 = st.columns(2)
        
        with col3:
            family_survival = titanic.groupby('family_size')['survived'].mean().reset_index()
            fig = px.line(family_survival, x='family_size', y='survived',
                         title='Tasa de Supervivencia por Tamaño Familiar',
                         markers=True)
            fig.update_layout(xaxis_title='Tamaño Familiar', yaxis_title='Tasa de Supervivencia')
            st.plotly_chart(fig, use_container_width=True)
        
        with col4:
            alone_vs_family = titanic.groupby('is_alone')['survived'].mean().reset_index()
            alone_vs_family['is_alone'] = alone_vs_family['is_alone'].map({0: 'Con Familia', 1: 'Solo'})
            fig = px.bar(alone_vs_family, x='is_alone', y='survived',
                        title='Supervivencia: Solos vs Con Familia',
                        color='is_alone',
                        color_discrete_sequence=['#FF9F1C', '#2EC4B6'])
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
                # Distribución de tarifas por clase
                fig = px.box(titanic, x='pclass', y='fare', color='pclass',
                           title='Distribución de Tarifas por Clase',
                           labels={'pclass': 'Clase', 'fare': 'Tarifa ($)'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Datos de clase no disponibles")
        
        with col2:
            if 'survived' in titanic.columns:
                # Tarifa vs Supervivencia
                fig = px.box(titanic, x='survived', y='fare', color='survived',
                            title='Tarifa Pagada: Supervivientes vs Fallecidos',
                            labels={'survived': 'Sobrevivió', 'fare': 'Tarifa ($)'},
                            color_discrete_sequence=['#FF6B6B', '#4ECDC4'])
                fig.update_layout(xaxis_title='Sobrevivió (0=No, 1=Sí)')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Datos de supervivencia no disponibles")

# =============================================================================
# SECCIÓN 4: ANÁLISIS DE SUPERVIVENCIA
# =============================================================================
elif section == "🔍 Análisis de Supervivencia":
    st.markdown('<h2 class="section-header">🔍 Análisis Detallado de Supervivencia</h2>', unsafe_allow_html=True)
    
    if 'survived' in titanic.columns and 'pclass' in titanic.columns and 'sex' in titanic.columns:
        # Heatmap de supervivencia
        st.subheader("Mapa de Calor de Factores de Supervivencia")
        
        # Preparar datos para heatmap
        survival_pivot = titanic.pivot_table(values='survived', 
                                            index='pclass', 
                                            columns='sex', 
                                            aggfunc='mean')
        
        fig = px.imshow(survival_pivot, 
                       title='Tasa de Supervivencia por Clase y Género',
                       color_continuous_scale='RdYlBu',
                       aspect='auto')
        fig.update_layout(xaxis_title='Género', yaxis_title='Clase')
        st.plotly_chart(fig, use_container_width=True)
        
        # Análisis multivariable
        st.subheader("Análisis Multivariable")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'fare' in titanic.columns and 'family_size' in titanic.columns:
                # Sunburst chart
                fig = px.sunburst(titanic, path=['pclass', 'sex', 'survived'],
                                 values='fare', color='survived',
                                 color_continuous_scale='Blues',
                                 title='Sunburst: Clase → Género → Supervivencia')
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'age' in titanic.columns:
                # Edad vs Tarifa con supervivencia
                fig = px.scatter(titanic, x='age', y='fare', color='survived',
                                size='family_size', hover_data=['pclass', 'sex'],
                                title='Edad vs Tarifa (Tamaño: Familia)',
                                color_discrete_sequence=['#FF6B6B', '#4ECDC4'])
                st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# SECCIÓN 5: MACHINE LEARNING
# =============================================================================
elif section == "🤖 Machine Learning":
    st.markdown('<h2 class="section-header">🤖 Modelos de Machine Learning</h2>', unsafe_allow_html=True)
    
    if len(X) > 0 and len(y) > 0:
        st.subheader("Configuración del Modelo")
        
        col1, col2 = st.columns(2)
        
        with col1:
            model_choice = st.selectbox(
                "Seleccionar Modelo:",
                ["Random Forest", "Gradient Boosting", "Logistic Regression", "SVM"]
            )
        
        with col2:
            test_size = st.slider("Tamaño del Conjunto de Test:", 0.1, 0.4, 0.2, 0.05)
        
        # Entrenar modelo seleccionado
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Escalar características
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        models = {
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
            "Logistic Regression": LogisticRegression(random_state=42),
            "SVM": SVC(probability=True, random_state=42)
        }
        
        model = models[model_choice]
        
        # Entrenar modelo
        with st.spinner(f'Entrenando modelo {model_choice}...'):
            if model_choice in ["Logistic Regression", "SVM"]:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        accuracy = accuracy_score(y_test, y_pred)
        
        # Mostrar resultados
        col3, col4 = st.columns(2)
        
        with col3:
            st.metric("Accuracy del Modelo", f"{accuracy:.3f}")
            
            # Validación cruzada
            with st.spinner('Realizando validación cruzada...'):
                if model_choice in ["Logistic Regression", "SVM"]:
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
                else:
                    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            
            st.metric("Accuracy Validación Cruzada", f"{cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
        
        with col4:
            # Matriz de confusión
            cm = confusion_matrix(y_test, y_pred)
            fig = px.imshow(cm, 
                           labels=dict(x="Predicho", y="Real", color="Count"),
                           x=['No Sobrevivió', 'Sobrevivió'],
                           y=['No Sobrevivió', 'Sobrevivió'],
                           title='Matriz de Confusión',
                           color_continuous_scale='Blues',
                           text_auto=True)
            st.plotly_chart(fig, use_container_width=True)
        
        # Curva ROC si está disponible
        if y_pred_proba is not None:
            st.subheader("Curva ROC y Métricas")
            
            col5, col6 = st.columns(2)
            
            with col5:
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                roc_auc = auc(fpr, tpr)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f'ROC curve (AUC = {roc_auc:.3f})'))
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
                fig.update_layout(title='Curva ROC',
                                 xaxis_title='False Positive Rate',
                                 yaxis_title='True Positive Rate')
                st.plotly_chart(fig, use_container_width=True)
            
            with col6:
                # Importancia de características
                if hasattr(model, 'feature_importances_'):
                    feature_importance = pd.DataFrame({
                        'feature': features,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=True)
                    
                    fig = px.bar(feature_importance, x='importance', y='feature',
                                title='Importancia de Características',
                                orientation='h')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("La importancia de características no está disponible para este modelo")
        
        # Reporte de clasificación
        st.subheader("Reporte de Clasificación Detallado")
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.background_gradient(cmap='Blues'), use_container_width=True)
    
    else:
        st.error("No hay suficientes datos para entrenar modelos de Machine Learning")

# =============================================================================
# SECCIÓN 6: CLUSTERING & SEGMENTACIÓN
# =============================================================================
elif section == "📈 Clustering & Segmentación":
    st.markdown('<h2 class="section-header">📈 Segmentación de Pasajeros con Clustering</h2>', unsafe_allow_html=True)
    
    # Preparar datos para clustering
    clustering_features = ['age', 'fare', 'pclass']
    available_clustering_features = [f for f in clustering_features if f in titanic.columns]
    
    if len(available_clustering_features) >= 2:
        clustering_data = titanic[available_clustering_features].copy()
        clustering_data = clustering_data.fillna(clustering_data.median())
        
        # Normalizar datos
        scaler = StandardScaler()
        clustering_scaled = scaler.fit_transform(clustering_data)
        
        # Determinar número óptimo de clusters
        st.subheader("Determinación del Número Óptimo de Clusters")
        
        inertia = []
        k_range = range(1, 11)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(clustering_scaled)
            inertia.append(kmeans.inertia_)
        
        fig = px.line(x=list(k_range), y=inertia, 
                     title='Método del Codo para Determinar K Óptimo',
                     labels={'x': 'Número de Clusters', 'y': 'Inercia'})
        fig.update_traces(mode='lines+markers')
        st.plotly_chart(fig, use_container_width=True)
        
        # Aplicar K-means con k óptimo
        optimal_k = st.slider("Seleccionar número de clusters:", 2, 6, 3)
        
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        clusters = kmeans.fit_predict(clustering_scaled)
        
        clustering_data['cluster'] = clusters
        if 'survived' in titanic.columns:
            clustering_data['survived'] = titanic['survived'].values
        
        # Visualización de clusters
        st.subheader("Visualización de Clusters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'age' in clustering_data.columns and 'fare' in clustering_data.columns:
                fig = px.scatter(clustering_data, x='age', y='fare', color='cluster',
                                title='Clusters: Edad vs Tarifa',
                                color_continuous_scale='Viridis')
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'survived' in clustering_data.columns:
                cluster_survival = clustering_data.groupby('cluster')['survived'].mean().reset_index()
                fig = px.bar(cluster_survival, x='cluster', y='survived',
                            title='Tasa de Supervivencia por Cluster',
                            color='survived',
                            color_continuous_scale='RdYlBu')
                st.plotly_chart(fig, use_container_width=True)
        
        # Análisis de perfiles de clusters
        st.subheader("📊 Perfiles de Clusters")
        
        cluster_profiles = clustering_data.groupby('cluster').mean().round(2)
        st.dataframe(cluster_profiles.style.background_gradient(cmap='YlOrBr'), use_container_width=True)
    
    else:
        st.error("No hay suficientes características para realizar clustering")

# =============================================================================
# SECCIÓN 7: INSIGHTS & RECOMENDACIONES
# =============================================================================
else:
    st.markdown('<h2 class="section-header">🎯 Insights Estratégicos & Recomendaciones</h2>', unsafe_allow_html=True)
    
    # Insights principales
    st.subheader("🔍 Hallazgos Clave del Análisis")
    
    insights = [
        "🚨 **Supervivencia por Género:** Las mujeres tuvieron significativamente mayor tasa de supervivencia",
        "💼 **Impacto de la Clase Social:** La clase social fue un factor determinante en la supervivencia", 
        "👶 **Factor Edad:** Los niños tuvieron prioridad en los protocolos de rescate",
        "👨‍👩‍👧‍👦 **Efecto Familiar:** Los pasajeros con familia mostraron diferentes patrones de supervivencia",
        "💰 **Correlación Riqueza-Supervivencia:** Mayor tarifa correlaciona con mayor supervivencia",
        "🏠 **Ventaja de Cabina:** Los pasajeros con cabina asignada tuvieron mejor supervivencia"
    ]
    
    for insight in insights:
        st.info(insight)
    
    # Recomendaciones estratégicas
    st.subheader("💡 Recomendaciones Estratégicas")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🚢 Para Diseño de Buques
        - **Priorizar botes salvavidas accesibles** a todas las clases
        - **Mejorar señalización** a zonas de evacuación
        - **Sistemas de alerta temprana** más efectivos
        - **Capacitación de tripulación** en procedimientos de emergencia
        """)
        
        st.markdown("""
        ### 👥 Para Protocolos de Evacuación
        - **Protocolos claros** de evacuación por segmentos
        - **Rutas de evacuación** optimizadas
        - **Sistemas de comunicación** efectivos
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Para Análisis de Riesgo
        - **Modelos predictivos** de supervivencia
        - **Simulaciones de evacuación** basadas en datos
        - **Sistemas de monitoreo** en tiempo real
        """)
        
        st.markdown("""
        ### 🎯 Para Políticas de Seguridad
        - **Auditorías regulares** de protocolos
        - **Entrenamiento obligatorio** de evacuación
        - **Tecnología de seguridad** moderna
        """)
    
    # Lecciones aprendidas
    st.subheader("📚 Lecciones Aprendidas para la Industria Naviera")
    
    lessons = [
        "✅ **La seguridad no debe ser un lujo** - Equidad en protocolos de seguridad",
        "✅ **Los datos salvan vidas** - Análisis predictivo para optimizar recursos", 
        "✅ **La preparación es clave** - Protocolos claros salvan vidas",
        "✅ **La tecnología como aliada** - Sistemas modernos previenen tragedias",
        "✅ **La equidad como principio** - Igual oportunidad de supervivencia para todos"
    ]
    
    for lesson in lessons:
        st.success(lesson)

# Pie de página profesional
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
<p><strong>Análisis Profesional del Titanic</strong> | Machine Learning & Business Intelligence Application</p>
<p>Desarrollado para demostración educativa | Herramientas: Streamlit, Scikit-learn, Plotly, Pandas</p>
</div>
""", unsafe_allow_html=True)
