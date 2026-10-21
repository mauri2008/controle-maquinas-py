import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# Configuração da página do Streamlit
st.set_page_config(
    page_title="Dashboard de Manutenção Preditiva",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ Dashboard de Manutenção Preditiva (DBSCAN)")
st.markdown("""
Este dashboard simula dados de sensores em um chão de fábrica (`Frequência de Vibração` e `Pico de Temperatura`) 
e utiliza o algoritmo **DBSCAN** para identificar padrões normais de operação e detectar anomalias (outliers).
""")

# --- GERAGÃO DE DADOS ---
@st.cache_data
def generate_factory_data(seed=42):
    np.random.seed(seed)
    
    # 1. Grupo Normal 1: Operação Padrão
    vib_1 = np.random.normal(loc=50, scale=5, size=200)
    temp_1 = np.random.normal(loc=65, scale=4, size=200)
    
    # 2. Grupo Normal 2: Operação de Alta Carga
    vib_2 = np.random.normal(loc=80, scale=6, size=150)
    temp_2 = np.random.normal(loc=85, scale=5, size=150)
    
    # 3. Outliers / Anomalias intencionais
    vib_outliers = np.random.uniform(low=20, high=120, size=20)
    temp_outliers = np.random.uniform(low=40, high=110, size=20)
    
    # Unindo os dados
    vibracao = np.concatenate([vib_1, vib_2, vib_outliers])
    temperatura = np.concatenate([temp_1, temp_2, temp_outliers])
    
    # Criando identificadores para as máquinas
    ids_maquinas = [f"MAQ-{i+1:03d}" for i in range(len(vibracao))]
    
    df = pd.DataFrame({
        'ID_Maquina': ids_maquinas,
        'Frequencia_Vibracao': vibracao,
        'Pico_Temperatura': temperatura
    })
    
    return df

df = generate_factory_data()

# --- BARRA LATERAL (PARÂMETROS DO DBSCAN) ---
st.sidebar.header("⚙️ Parâmetros do DBSCAN")
st.sidebar.markdown("Ajuste os parâmetros do modelo para reagrupar os pontos:")

eps = st.sidebar.slider(
    "eps (Distância máxima)",
    min_value=1.0,
    max_value=15.0,
    value=5.0,
    step=0.5,
    help="Distância máxima entre dois pontos para que sejam considerados vizinhos."
)

min_samples = st.sidebar.slider(
    "min_samples (Mínimo de Amostras)",
    min_value=2,
    max_value=20,
    value=5,
    step=1,
    help="Número mínimo de pontos em uma vizinhança para formar uma região densa (cluster)."
)

# --- PROCESSAMENTO DBSCAN ---
X = df[['Frequencia_Vibracao', 'Pico_Temperatura']].values
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
df['Cluster'] = dbscan.fit_predict(X)

# Flag de Anomalia (-1 no DBSCAN é ruído/outlier)
df['Status'] = df['Cluster'].apply(lambda x: 'Anomalia' if x == -1 else 'Normal')

# --- VISUALIZAÇÃO COM SEABORN ---
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("📊 Agrupamento de Sensores e Anomalias")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Definindo paleta personalizada: preto para anomalias (-1), cores do Seaborn para os clusters válidos
    unique_clusters = df['Cluster'].unique()
    n_clusters = len([c for c in unique_clusters if c != -1])
    
    palette = {}
    if -1 in unique_clusters:
        palette[-1] = "#000000"  # Preto para Outliers
        
    base_palette = sns.color_palette("tab10", n_colors=max(n_clusters, 1))
    cluster_idx = 0
    for c in sorted(unique_clusters):
        if c != -1:
            palette[c] = base_palette[cluster_idx]
            cluster_idx += 1

    # Gráfico de Dispersão
    sns.scatterplot(
        data=df,
        x='Frequencia_Vibracao',
        y='Pico_Temperatura',
        hue='Cluster',
        palette=palette,
        style='Status',
        markers={'Normal': 'o', 'Anomalia': 'X'},
        s=80,
        ax=ax
    )

    ax.set_title("Identificação de Clusters e Outliers via DBSCAN", fontsize=14)
    ax.set_xlabel("Frequência de Vibração (Hz)", fontsize=12)
    ax.set_ylabel("Pico de Temperatura (°C)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    
    st.pyplot(fig)

with col2:
    st.subheader("📈 Resumo da Análise")
    total_maquinas = len(df)
    anomalias_count = (df['Cluster'] == -1).sum()
    clusters_count = n_clusters

    st.metric("Total de Máquinas Monitoradas", total_maquinas)
    st.metric("Clusters Encontrados", clusters_count)
    st.metric("Máquinas em Anomalia", anomalias_count, delta_color="inverse")

# --- TABELA DE MÁQUINAS ANÔMALAS ---
st.divider()
st.subheader("🚨 Tabela de Máquinas Consideradas Anômalas")

df_anomalias = df[df['Cluster'] == -1][['ID_Maquina', 'Frequencia_Vibracao', 'Pico_Temperatura']]

if not df_anomalias.empty:
    st.warning(f"Atenção: {len(df_anomalias)} máquinas apresentaram padrões discrepantes e necessitam de vistoria.")
    st.dataframe(
        df_anomalias.style.format({
            'Frequencia_Vibracao': '{:.2f} Hz',
            'Pico_Temperatura': '{:.2f} °C'
        }),
        use_container_width=True
    )
else:
    st.success("Nenhuma anomalia foi identificada com os parâmetros atuais do DBSCAN.")
