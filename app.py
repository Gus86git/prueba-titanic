import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
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

# Título principal con estilo
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
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🚢 Análisis Completo del Titanic</h1>', unsafe_allow_html=True)
st.markdown("### Machine Learning & Business Intelligence Application")

# Cargar y preparar datos
@st.cache_data
def load_data():
    # Cargar dataset
    try:
        titanic = sns.load_dataset('titanic')
    except:
        # Fallback si no carga desde seaborn
        url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
        titanic = pd.read_csv(url)
    
    # Feature engineering avanzado
    titanic['title'] = titanic['name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    titanic['family_size'] = titanic['sibsp'] + titanic['parch'] + 1
    titanic['is_alone'] = (titanic['family_size'] == 1).astype(int)
    titanic['age_group'] = pd.cut(titanic['age'], 
                                 bins=[0, 12, 18, 35, 50, 100], 
                                 labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Senior'])
    
    # Precio del ticket categorizado
    titanic['fare_category'] = pd.cut(titanic['fare'],
                                     bins=[0, 10, 30, 100, 600],
                                     labels=['Low', 'Medium', 'High', 'Very High'])
    
    # Cabin information
    titanic['deck'] = titanic['cabin'].str[0]
    titanic['has_cabin'] = titanic['cabin'].notna().astype(int)
    
    return titanic

@st.cache_data
def prepare_ml_data(df):
    """Preparar datos para machine learning"""
    df_ml = df.copy()
    
    # Handle missing values
    df_ml['age'].fillna(df_ml['age'].median(), inplace=True)
    df_ml['embarked'].fillna(df_ml['embarked'].mode()[0], inplace=True)
    df_ml['deck'].fillna('Unknown', inplace=True)
    
    # Feature engineering for ML
    df_ml['title'] = df_ml['title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 
                                            'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df_ml['title'] = df_ml['title'].replace('Mlle', 'Miss')
    df_ml['title'] = df_ml['title'].replace('Ms', 'Miss')
    df_ml['title'] = df_ml['title'].replace('Mme', 'Mrs')
    
    # Encode categorical variables
    le = LabelEncoder()
    categorical_cols = ['sex', 'embarked', 'title', 'deck']
    for col in categorical_cols:
        df_ml[col] = le.fit_transform(df_ml[col].astype(str))
    
    # Select features for ML
    features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 
                'title', 'family_size', 'is_alone', 'has_cabin', 'deck']
    
    X = df_ml[features]
    y = df_ml['survived']
    
    return X, y, features

# Cargar datos
titanic = load_data()
X, y, features = prepare_ml_data(titanic)

# Sidebar para navegación
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/f/fd/RMS_Titanic_3.jpg/800px-RMS_Titanic_3.jpg", 
                 use_column_width=True)
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
        survival_rate = titanic['survived'].mean() * 100
        st.metric("Tasa de Supervivencia", f"{survival_rate:.1f}%")
    
    with col3:
        avg_fare = titanic['fare'].mean()
        st.metric("Tarifa Promedio", f"${avg_fare:.2f}")
    
    with col4:
        avg_age = titanic['age'].mean()
        st.metric("Edad Promedio", f"{avg_age:.1f} años")
    
    # Segunda fila de métricas
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        women_survival = titanic[titanic['sex']=='female']['survived'].mean() * 100
        st.metric("Supervivencia Mujeres", f"{women_survival:.1f}%")
    
    with col6:
        men_survival = titanic[titanic['sex']=='male']['survived'].mean() * 100
        st.metric("Supervivencia Hombres", f"{men_survival:.1f}%")
    
    with col7:
        first_class_survival = titanic[titanic['pclass']==1]['survived'].mean() * 100
        st.metric("Supervivencia 1ra Clase", f"{first_class_survival:.1f}%")
    
    with col8:
        third_class_survival = titanic[titanic['pclass']==3]['survived'].mean() * 100
        st.metric("Supervivencia 3ra Clase", f"{third_class_survival:.1f}%")
    
    # Visualización de distribución general
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.pie(titanic, names='survived', title='Distribución de Supervivencia',
                    color_discrete_sequence=['#FF6B6B', '#4ECDC4'])
        fig.update_traces(textinfo='percent+label', pull=[0.1, 0])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(titanic, x='age', nbins=30, title='Distribución de Edades',
                          color_discrete_sequence=['#2E86AB'])
        fig.update_layout(xaxis_title='Edad', yaxis_title='Frecuencia')
        st.plotly_chart(fig, use_container_width=True)
    
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
        # Distribución por género y clase
        fig = make_subplots(rows=1, cols=2, 
                           subplot_titles=['Distribución por Género', 'Distribución por Clase'],
                           specs=[[{'type':'domain'}, {'type':'domain'}]])
        
        gender_counts = titanic['sex'].value_counts()
        class_counts = titanic['pclass'].value_counts().sort_index()
        
        fig.add_trace(go.Pie(labels=gender_counts.index, values=gender_counts.values,
                            name="Género"), 1, 1)
        fig.add_trace(go.Pie(labels=[f'Clase {c}' for c in class_counts.index], 
                            values=class_counts.values, name="Clase"), 1, 2)
        
        fig.update_traces(hole=.4, hoverinfo="label+percent+name")
        fig.update_layout(title_text="Distribución Demográfica")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Edad por género y clase
        fig = px.box(titanic, x='pclass', y='age', color='sex',
                    title='Distribución de Edad por Clase y Género',
                    labels={'pclass': 'Clase', 'age': 'Edad'},
                    color_discrete_sequence=['#FF6B6B', '#4ECDC4'])
        st.plotly_chart(fig, use_container_width=True)
    
    # Análisis de familias
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
    
    st.subheader("Tarifas y Distribución de Riqueza")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribución de tarifas por clase
        fig = px.violin(titanic, x='pclass', y='fare', color='pclass',
                       title='Distribución de Tarifas por Clase',
                       labels={'pclass': 'Clase', 'fare': 'Tarifa ($)'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Tarifa vs Supervivencia
        fig = px.box(titanic, x='survived', y='fare', color='survived',
                    title='Tarifa Pagada: Supervivientes vs Fallecidos',
                    labels={'survived': 'Sobrevivió', 'fare': 'Tarifa ($)'},
                    color_discrete_sequence=['#FF6B6B', '#4ECDC4'])
        fig.update_layout(xaxis_title='Sobrevivió (0=No, 1=Sí)')
        st.plotly_chart(fig, use_container_width=True)
    
    # Análisis de cabinas
    st.subheader("🏠 Análisis de Cabinas y Ubicación")
    
    col3, col4 = st.columns(2)
    
    with col3:
        deck_survival = titanic.groupby('deck')['survived'].mean().reset_index()
        deck_survival = deck_survival[deck_survival['deck'].notna()]
        fig = px.bar(deck_survival, x='deck', y='survived',
                    title='Tasa de Supervivencia por Cubierta (Deck)',
                    color='survived',
                    color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)
    
    with col4:
        cabin_survival = titanic.groupby('has_cabin')['survived'].mean().reset_index()
        cabin_survival['has_cabin'] = cabin_survival['has_cabin'].map({0: 'Sin Cabina', 1: 'Con Cabina'})
        fig = px.pie(cabin_survival, values='survived', names='has_cabin',
                    title='Supervivencia: Pasajeros con/sin Cabina')
        st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# SECCIÓN 4: ANÁLISIS DE SUPERVIVENCIA
# =============================================================================
elif section == "🔍 Análisis de Supervivencia":
    st.markdown('<h2 class="section-header">🔍 Análisis Detallado de Supervivencia</h2>', unsafe_allow_html=True)
    
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
        # Supervivencia por clase y género
        fig = px.sunburst(titanic, path=['pclass', 'sex', 'survived'],
                         values='fare', color='survived',
                         color_continuous_scale='Blues',
                         title='Sunburst: Clase → Género → Supervivencia')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Edad vs Tarifa con supervivencia
        fig = px.scatter(titanic, x='age', y='fare', color='survived',
                        size='family_size', hover_data=['pclass', 'sex'],
                        title='Edad vs Tarifa (Tamaño: Familia)',
                        color_discrete_sequence=['#FF6B6B', '#4ECDC4'])
        st.plotly_chart(fig, use_container_width=True)
    
    # Análisis de títulos
    st.subheader("🎭 Análisis por Título Social")
    
    title_survival = titanic.groupby('title')['survived'].agg(['mean', 'count']).reset_index()
    title_survival = title_survival[title_survival['count'] > 5]  # Filtrar títulos raros
    
    fig = px.bar(title_survival, x='title', y='mean',
                title='Tasa de Supervivencia por Título Social',
                hover_data=['count'],
                color='mean',
                color_continuous_scale='Viridis')
    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# SECCIÓN 5: MACHINE LEARNING
# =============================================================================
elif section == "🤖 Machine Learning":
    st.markdown('<h2 class="section-header">🤖 Modelos de Machine Learning</h2>', unsafe_allow_html=True)
    
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
    
    if model_choice in ["Logistic Regression", "SVM"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    
    # Mostrar resultados
    col3, col4 = st.columns(2)
    
    with col3:
        st.metric("Accuracy del Modelo", f"{accuracy:.3f}")
        
        # Validación cruzada
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
    
    # Curva ROC
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

# =============================================================================
# SECCIÓN 6: CLUSTERING & SEGMENTACIÓN
# =============================================================================
elif section == "📈 Clustering & Segmentación":
    st.markdown('<h2 class="section-header">📈 Segmentación de Pasajeros con Clustering</h2>', unsafe_allow_html=True)
    
    # Preparar datos para clustering
    clustering_data = titanic[['age', 'fare', 'pclass', 'sibsp', 'parch']].copy()
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
    clustering_data['survived'] = titanic['survived'].values
    
    # Visualización de clusters
    st.subheader("Visualización de Clusters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.scatter(clustering_data, x='age', y='fare', color='cluster',
                        size='pclass', hover_data=['sibsp', 'parch'],
                        title='Clusters: Edad vs Tarifa',
                        color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        cluster_survival = clustering_data.groupby('cluster')['survived'].mean().reset_index()
        fig = px.bar(cluster_survival, x='cluster', y='survived',
                    title='Tasa de Supervivencia por Cluster',
                    color='survived',
                    color_continuous_scale='RdYlBu')
        st.plotly_chart(fig, use_container_width=True)
    
    # Análisis de perfiles de clusters
    st.subheader("📊 Perfiles de Clusters")
    
    cluster_profiles = clustering_data.groupby('cluster').agg({
        'age': 'mean',
        'fare': 'mean', 
        'pclass': 'mean',
        'sibsp': 'mean',
        'parch': 'mean',
        'survived': 'mean'
    }).round(2)
    
    st.dataframe(cluster_profiles.style.background_gradient(cmap='YlOrBr'), use_container_width=True)
    
    # Interpretación de clusters
    st.subheader("🎯 Interpretación de Segmentos")
    
    for cluster in range(optimal_k):
        with st.expander(f"📋 Perfil del Cluster {cluster}"):
            cluster_data = cluster_profiles.loc[cluster]
            st.write(f"""
            - **Edad promedio:** {cluster_data['age']} años
            - **Tarifa promedio:** ${cluster_data['fare']:.2f}
            - **Clase social promedio:** {cluster_data['pclass']:.1f}
            - **Tamaño familiar promedio:** {cluster_data['sibsp'] + cluster_data['parch']:.1f}
            - **Tasa de supervivencia:** {cluster_data['survived']*100:.1f}%
            """)

# =============================================================================
# SECCIÓN 7: INSIGHTS & RECOMENDACIONES
# =============================================================================
else:
    st.markdown('<h2 class="section-header">🎯 Insights Estratégicos & Recomendaciones</h2>', unsafe_allow_html=True)
    
    # Insights principales
    st.subheader("🔍 Hallazgos Clave del Análisis")
    
    insights = [
        "🚨 **Supervivencia por Género:** Las mujeres tuvieron una tasa de supervivencia del 74.2% vs 18.9% en hombres",
        "💼 **Impacto de la Clase Social:** La 1ra clase tuvo 62.9% de supervivencia vs 24.2% en 3ra clase",
        "👶 **Factor Edad:** Niños menores de 12 años tuvieron mayor tasa de supervivencia (59.0%)",
        "👨‍👩‍👧‍👦 **Efecto Familiar:** Pasajeros con familia (3-4 miembros) tuvieron mejor supervivencia",
        "💰 **Correlación Riqueza-Supervivencia:** Tarifa promedio supervivientes: $48.4 vs $22.1 en fallecidos",
        "🏠 **Ventaja de Cabina:** Pasajeros con cabina asignada: 66.7% supervivencia vs 30.0% sin cabina"
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
        - **Protocolos claros** de "mujeres y niños primero"
        - **Rutas de evacuación** optimizadas por ubicación de cabinas
        - **Sistemas de comunicación** multilingües
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Para Análisis de Riesgo
        - **Modelos predictivos** de supervivencia por segmento
        - **Simulaciones de evacuación** basadas en datos históricos
        - **Sistemas de pricing dinámico** que consideren factores de seguridad
        """)
        
        st.markdown("""
        ### 🎯 Para Políticas de Seguridad
        - **Auditorías regulares** de protocolos de seguridad
        - **Entrenamiento obligatorio** de evacuación para pasajeros
        - **Tecnología de localización** en tiempo real durante emergencias
        """)
    
    # Lecciones aprendidas
    st.subheader("📚 Lecciones Aprendidas para la Industria Naviera")
    
    lessons = [
        "✅ **La seguridad no debe ser un lujo** - Las disparidades entre clases son inaceptables",
        "✅ **Los datos salvan vidas** - El análisis predictivo puede optimizar recursos",
        "✅ **La preparación es clave** - Protocolos claros salvan más vidas que la improvisación",
        "✅ **La tecnología como aliada** - Sistemas modernos pueden prevenir tragedias",
        "✅ **La equidad como principio** - Todos los pasajeros merecen igual oportunidad de supervivencia"
    ]
    
    for lesson in lessons:
        st.success(lesson)
    
    # Llamado a la acción
    st.markdown("---")
    st.markdown("""
    <div style='background-color: #2E86AB; padding: 20px; border-radius: 10px; color: white;'>
    <h3 style='color: white; text-align: center;'>🚀 Próximos Pasos para Investigación</h3>
    <ul>
    <li>Desarrollar modelos predictivos en tiempo real para gestión de crisis</li>
    <li>Implementar sistemas de simulación de evacuación basados en IA</li>
    <li>Crear dashboards de monitoreo de seguridad para tripulación</li>
    <li>Establecer estándares industry-wide para protocolos de emergencia</li>
    </ul>
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
