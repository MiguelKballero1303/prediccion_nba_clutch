import streamlit as st 
import pandas as pd
import numpy as np
from nba_api.stats.endpoints import scoreboardv2, leaguedashplayerstats
from nba_api.stats.static import teams
from sklearn.preprocessing import StandardScaler
import plotly.express as px

def calcular_clutch_score(df):
    weights = {
        'PTS': 0.35, 'FG_PCT': 0.25, 'FT_PCT': 0.15,
        'AST': 0.15, 'FG3_PCT': 0.10, 'TOV': -0.25
    }
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[list(weights.keys())])
    clutch_scores = np.dot(scaled_data, list(weights.values()))
    df_clutch = df[['PLAYER_NAME', 'TEAM_ABBREVIATION']].copy()
    df_clutch['CLUTCH_SCORE'] = clutch_scores
    df_clutch['CLUTCH_RANK'] = df_clutch['CLUTCH_SCORE'].rank(ascending=False)
    return df_clutch.sort_values('CLUTCH_SCORE', ascending=False).reset_index(drop=True)

def obtener_partidos():
    try:
        scoreboard = scoreboardv2.ScoreboardV2()
        data = scoreboard.get_normalized_dict()
        if not data['GameHeader']:
            st.warning("No hay partidos programados para hoy.")
            return {}
        partidos = {}
        for game in data['GameHeader']:
            home_team = teams.find_team_name_by_id(game['HOME_TEAM_ID'])['nickname']
            away_team = teams.find_team_name_by_id(game['VISITOR_TEAM_ID'])['nickname']
            key = f"{home_team} vs {away_team}"
            partidos[key] = {
                'home_abbrev': teams.find_team_name_by_id(game['HOME_TEAM_ID'])['abbreviation'],
                'away_abbrev': teams.find_team_name_by_id(game['VISITOR_TEAM_ID'])['abbreviation'],
                'status': game['GAME_STATUS_TEXT']
            }
        return partidos
    except Exception as e:
        st.error(f"Error al obtener partidos: {str(e)}")
        return {}

def obtener_datos_jugadores(abreviaturas):
    try:
        stats = leaguedashplayerstats.LeagueDashPlayerStats(
            per_mode_detailed='PerGame', 
            last_n_games=10
        ).get_data_frames()[0]
        
        relevant_stats = stats[
            ['PLAYER_NAME', 'TEAM_ABBREVIATION', 'FG_PCT', 
             'FG3_PCT', 'FT_PCT', 'AST', 'TOV', 'PTS']
        ].dropna()
        
        return relevant_stats[relevant_stats['TEAM_ABBREVIATION'].isin(abreviaturas)]
    except Exception as e:
        st.error(f"Error en datos de jugadores: {str(e)}")
        return pd.DataFrame()

st.set_page_config(page_title="NBA Clutch Predictor", layout="wide")
st.title("🏀 NBA Clutch Player Predictor")
st.markdown("### 🔥 Descubre quién brilla en los momentos decisivos")

# Sidebar para controles
with st.sidebar:
    partidos = obtener_partidos()
    if partidos:
        partido_seleccionado = st.selectbox("Selecciona un partido:", list(partidos.keys()))
        info = partidos[partido_seleccionado]
        abreviaturas = [info['home_abbrev'], info['away_abbrev']]
    else:
        st.stop()

# Contenido principal en pestañas
tab1, tab2 = st.tabs(["📈 Análisis del Partido", "📚 Información Técnica"])

with tab1:
    if partidos:
        jugadores = obtener_datos_jugadores(abreviaturas)
        
        if not jugadores.empty:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("### 🏅 Top 5 Jugadores Clutch")
                df_clutch = calcular_clutch_score(jugadores)
                top_players = df_clutch.head(5)
                
                # Gráfico de barras interactivo
                fig = px.bar(top_players, 
                            x='CLUTCH_SCORE', 
                            y='PLAYER_NAME',
                            orientation='h',
                            color='CLUTCH_SCORE',
                            labels={'CLUTCH_SCORE': 'Puntuación Clutch', 'PLAYER_NAME': 'Jugador'},
                            height=400)
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("### 📊 Ranking Clutch")
                st.dataframe(
                    top_players.style
                    .format({'CLUTCH_SCORE': '{:.2f}'})
                    .bar(subset=['CLUTCH_SCORE'], color='#FF4B4B')
                    .applymap(lambda x: 'color: green' if x == 1 else '', subset=['CLUTCH_RANK']),
                    height=35 + 35 * len(top_players),  # 35px por fila + header
                    hide_index=True
                )

            st.markdown("---")
            st.markdown("### 📋 Estadísticas Detalladas")
            st.dataframe(
                jugadores.style.format({
                    'FG_PCT': '{:.1%}', 
                    'FG3_PCT': '{:.1%}', 
                    'FT_PCT': '{:.1%}', 
                    'PTS': '{:.1f}'
                }),
                height=35 + 35 * len(jugadores),
                use_container_width=True,
                hide_index=True
            )
with tab2:
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        with st.expander("📖 Métrica Clutch", expanded=True):
            st.markdown("""
            **Fórmula del Clutch Score:**
            - 35% Puntos por partido (PTS)
            - 25% Eficacia en tiros de campo (FG%)
            - 15% Eficacia en tiros libres (FT%)
            - 15% Asistencias (AST)
            - 10% Eficacia en triples (3P%)
            - 25% Penalización por pérdidas (TOV)
            """)
    
    with col_info2:
        with st.expander("📚 Glosario Técnico", expanded=True):
            st.markdown("""
            - **PTS**: Puntos por partido
            - **FG%**: Porcentaje de tiros de campo
            - **3P%**: Porcentaje de triples
            - **FT%**: Porcentaje de tiros libres
            - **AST**: Asistencias
            - **TOV**: Pérdidas de balón
            """)

if not partidos:
    st.error("No hay partidos programados para hoy")