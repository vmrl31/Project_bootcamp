import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from kmodes.kprototypes import KPrototypes
from sklearn.preprocessing import MinMaxScaler
import matplotlib.ticker as mticker
from sklearn.manifold import TSNE

# Cargar los datos
df = pd.read_csv("archivo_limpio.csv")

st.set_page_config(page_title="Dashboard de felicidad mundial", layout="wide")
st.sidebar.title("Módulos del dashboard")
modulo = st.sidebar.radio("Selecciona una sección:", [
    "Evolución de la felicidad",
    "Comparación por continente",
    "Correlaciones",
    "Clustering de países",
    "Métricas y rankign"
])

# Variables numéricas
variables = ['pib_per_capita', 'apoyo_social', 'vida_saludable', 'libertad', 'generosidad', 'corrupcion']

# Módulo 1: Evolución de la felicidad
def modulo_1():

    st.title("Evolución de la felicidad entre 2015 y 2019")
    
    st.markdown("### Evolución global del puntaje de felicidad")

    promedio_anual = df.groupby('año')['puntaje_de_felicidad'].mean().reset_index()
    promedio_anual['año'] = promedio_anual['año'].astype(int)

    fig_line = px.line(promedio_anual,
                       x='año',
                       y='puntaje_de_felicidad',
                       markers=True,
                       title="Evolución del puntaje de felicidad global (2015–2019)",
                       labels={"año": "Año", "puntaje_de_felicidad": "Puntaje promedio"})
    fig_line.update_traces(line=dict(width=3))
    st.plotly_chart(fig_line)

    paises = st.multiselect("Selecciona países:", df["pais"].unique(), default=["colombia"])
    df_filtrado = df[df["pais"].isin(paises)]
    fig = px.line(df_filtrado, x="año", y="puntaje_de_felicidad", color="pais",
                  markers=True, title="Evolución del puntaje de felicidad por país")
    st.plotly_chart(fig, use_container_width=True)

# Módulo 2: Comparación por continente
def modulo_2():
    
    st.title("Comparación de felicidad por continente")
    tab1, tab2 = st.tabs(["Factores importantes", "Promedio por continente"])

    with tab1:

        st.markdown("### Evolución global de los factores clave de la felicidad (2015–2019)")
        variables = ['pib_per_capita', 'apoyo_social', 'vida_saludable', 'libertad', 'generosidad', 'corrupcion']

        promedios_variables = df.groupby('año')[variables].mean().reset_index()
        fig_evolucion_global, ax = plt.subplots(figsize=(12, 6))

        for var in variables:
            sns.lineplot(data=promedios_variables, x="año", y=var, label=var, ax=ax)

        ax.set_title("Evolución global de variables relacionadas con la felicidad")
        ax.set_xlabel("Año")
        ax.set_ylabel("Promedio global")
        ax.legend(title="Variable")
        ax.grid(True)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        st.pyplot(fig_evolucion_global)

        st.subheader("Comportamiento de factores importantes por continente")

        df_continent = df.groupby(['continente', 'año'])[variables].mean().reset_index()
        continentes = sorted(df['continente'].unique())
        selected = st.multiselect("Selecciona continentes:", continentes, default=continentes)

        for continente in selected:
            st.markdown(f"### {continente}")
            data = df_continent[df_continent['continente'] == continente]
            fig = px.line(data, x='año', y=variables, title=f"Evolución de variables en {continente}")
            st.plotly_chart(fig, use_container_width=True)
        
    with tab2:
        anio = st.selectbox("Selecciona un año:", sorted(df["año"].unique()))
        df_anio = df[df["año"] == anio]
        df_mean = df_anio.groupby("continente")["puntaje_de_felicidad"].mean().reset_index()
        fig2 = px.bar(df_mean, x="continente", y="puntaje_de_felicidad", color="continente",
                      title=f"Promedio de felicidad por continente en {anio}")
        st.plotly_chart(fig2, use_container_width=True)

# Módulo 3: correlaciones
def modulo_3():
    st.title("Análisis de correlaciones")

    columnas_numericas = df.select_dtypes(include='number').columns
    fig_corr_full, ax = plt.subplots(figsize=(10, 8))
    corr_matriz = df[columnas_numericas].corr()
    sns.heatmap(corr_matriz, annot=True, cmap='coolwarm', fmt=".2f", square=True, ax=ax)
    ax.set_title('Matriz de correlación entre variables')
    st.pyplot(fig_corr_full)

    anio = st.selectbox("Selecciona un año:", sorted(df["año"].unique()), key="corr")
    df_corr = df[df["año"] == anio].select_dtypes(include="number")
    corr_matrix = df_corr.corr()
    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                    title=f"Matriz de correlación de variables numéricas en {anio}")
    st.plotly_chart(fig, use_container_width=True)

# Módulo 4: clustering de países con KPrototypes + t-SNE
def modulo_4():
    st.title("Clustering de países según el perfil de felicidad")

    features = ['pais', 'continente', 'puntaje_de_felicidad', 'pib_per_capita','apoyo_social', 'vida_saludable', 'libertad', 'generosidad', 'corrupcion']
    df_kproto = df[features].copy()

    num_cols = ['puntaje_de_felicidad', 'pib_per_capita', 'apoyo_social','vida_saludable', 'libertad', 'generosidad', 'corrupcion']
    cat_cols = ['pais', 'continente']

    scaler = MinMaxScaler()
    df_kproto[num_cols] = scaler.fit_transform(df_kproto[num_cols])

    # Conversión a matriz para K-Prototypes
    matrix = df_kproto.to_numpy()

    # Entrenamiento del modelo K-Prototypes
    kproto = KPrototypes(n_clusters=4, init='Cao', verbose=0, random_state=42)
    clusters = kproto.fit_predict(matrix, categorical=[0, 1])
    df_kproto['cluster'] = clusters

    # Reducción de dimensionalidad con t-SNE para visualización
    tsne = TSNE(n_components=2, random_state=42, perplexity=40)
    tsne_result = tsne.fit_transform(df_kproto[num_cols])
    df_kproto['tsne_1'] = tsne_result[:, 0]
    df_kproto['tsne_2'] = tsne_result[:, 1]

    # Visualización de clusters en pyplot
    st.subheader("Visualización de Clusters con K-Prototypes y t-SNE")
    fig, ax = plt.subplots(figsize=(10, 6))

    for cluster in sorted(df_kproto['cluster'].unique()):
        subset = df_kproto[df_kproto['cluster'] == cluster]
        ax.scatter(subset['tsne_1'], subset['tsne_2'], label=f"Cluster {cluster}", alpha=0.7)

    ax.set_title('Clusters de países según perfil de felicidad (K-Prototypes + t-SNE)')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.legend(title='Cluster')
    plt.tight_layout()

    st.pyplot(fig)

    # Tabla de países por cluster
    st.subheader("Países agrupados por cluster")
    st.dataframe(df_kproto[['pais', 'continente', 'cluster']].drop_duplicates().sort_values('cluster'))

def modulo_5():
    st.subheader("Ranking de felicidad promedio por país")

    tipo = st.radio("¿Qué ranking desea ver?", ["Top 10 países más felices", "Top 10 países menos felices"])

    # Datos: ranking promedio de felicidad
    ranking_promedio = df.groupby('pais')['rango_de_felicidad'].mean()

    # Datos: Cambio entre 2015 y 2019
    puntaje = df[df['año'].isin([2015, 2019])][['pais', 'año', 'rango_de_felicidad']]
    puntaje = puntaje.pivot(index='pais', columns='año', values='rango_de_felicidad')
    puntaje['cambio'] = puntaje[2019] - puntaje[2015]

    # Visualización 
    if tipo == "Top 10 países más felices":
        top10_ranking = ranking_promedio.sort_values().head(10)[::-1]
        color_scale = "viridis"
        plotly_title = "Top 10 países con mejor ranking de felicidad (2015–2019)"

        # Gráfico de cambio
        top10_cambio = puntaje.sort_values('cambio', ascending=True).head(10)
        color_pyplot = "Greens_r"
        pyplot_title = "Top 10 países que más mejoraron (2015–2019)"

    else:
        top10_ranking = ranking_promedio.sort_values(ascending=False).head(10).sort_values()
        color_scale = "viridis"
        plotly_title = "Top 10 países con peor ranking de felicidad (2015–2019)"

        # Gráfico de cambio
        top10_cambio = puntaje.sort_values('cambio', ascending=False).head(10)
        color_pyplot = "Reds"
        pyplot_title = "Top 10 países que más empeoraron (2015–2019)"

    # Gráfico 1: ranking promedio
    fig1 = px.bar(
        top10_ranking,
        x=top10_ranking.values,
        y=top10_ranking.index,
        orientation='h',
        color=top10_ranking.values,
        color_continuous_scale=color_scale,
        title=plotly_title,
        labels={"x": "Ranking promedio", "y": "País"}
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Gráfico 2: Cambio entre 2015 y 2019
    fig2, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(y=top10_cambio.index, x=top10_cambio['cambio'], palette=color_pyplot, ax=ax)
    ax.set_title(pyplot_title)
    ax.set_xlabel("Cambio en rango de felicidad (2019 - 2015)")
    ax.set_ylabel("País")
    plt.tight_layout()
    st.pyplot(fig2)


# Render o carga del módulo seleccionadoo
if modulo == "Evolución de la felicidad":
    modulo_1()
elif modulo == "Comparación por continente":
    modulo_2()
elif modulo == "Correlaciones":
    modulo_3()
elif modulo == "Clustering de países":
    modulo_4()
else:
    modulo_5()