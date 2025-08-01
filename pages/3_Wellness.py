#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 09:32:55 2025

@author: fcvmathieu
"""

import streamlit as st
import pandas as pd
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
import os
import pickle
import ast
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import plotly.express as px
import plotly.graph_objects as go
import numpy as np


# Constants for Google Sheets
SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
TOKEN_FILE = 'token_well.pickle'
SPREADSHEET_ID = '1tiCkE28kdrP4BOyUHCSo83WYRvwLdlALRuPiv-cDsHU'  # Replace with your actual Spreadsheet ID
RANGE_NAME = 'Feuille 1'

st.set_page_config(layout='wide')

# Display the club logo from GitHub at the top right
logo_url = 'https://raw.githubusercontent.com/FC-Versailles/wellness/main/logo.png'
col1, col2 = st.columns([9, 1])
with col1:
    st.title("Wellness | FC Versailles")
with col2:
    st.image(logo_url, use_container_width=True)
    
# Add a horizontal line to separate the header
st.markdown("<hr style='border:1px solid #ddd' />", unsafe_allow_html=True)


# Function to get Google Sheets credentials
def get_credentials():
    creds = None
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'client_secret.json', SCOPES
            )
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, 'wb') as token:
            pickle.dump(creds, token)
    return creds

# Function to fetch data from Google Sheet
def fetch_google_sheet(spreadsheet_id, range_name):
    creds = get_credentials()
    service = build('sheets', 'v4', credentials=creds)
    sheet = service.spreadsheets()
    result = sheet.values().get(spreadsheetId=spreadsheet_id, range=range_name).execute()
    values = result.get('values', [])
    if not values:
        st.error("No data found in the specified range.")
        return pd.DataFrame()
    header = values[0]
    data = values[1:]
    max_columns = len(header)
    adjusted_data = [
        row + [None] * (max_columns - len(row)) if len(row) < max_columns else row[:max_columns]
        for row in data
    ]
    return pd.DataFrame(adjusted_data, columns=header)

# Define the gradient function
def color_gradient(value):
    colors = ["green", "yellow", "orange", "red"]
    cmap = LinearSegmentedColormap.from_list("smooth_gradient", colors)

    # Handle NaN values by assigning a default gradient (e.g., green)
    if pd.isna(value):
        value = 1

    # Normalize the value to fit between 0 and 1 for the gradient
    norm_value = (value - 1) / (5 - 1)
    return f"background-color: rgba({','.join(map(str, [int(c * 255) for c in cmap(norm_value)[:3]]))}, 0.6)"


# Add a button to refresh the data
if st.button("Actualiser les données"):
    st.cache_data.clear()  # Clear the cache to fetch new data
    st.success("Data refreshed successfully!")


# Fetch Google Sheet data
@st.cache_data
def load_data(ttl=60):
    return fetch_google_sheet(SPREADSHEET_ID, RANGE_NAME)


data = load_data()

# Rename columns as requested
data = data.rename(columns={
    'Humeur post-entrainement': 'Humeur-Post',
    'Plaisir entrainement': 'Plaisir-Post',
    'RPE': 'RPE', 'Progression entrainement':'Progression'
})




# Ensure the "Date" column is in datetime format
if "Date" in data.columns:
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    
    
# Sidebar for navigation
st.sidebar.title("Wellness")
page = st.sidebar.selectbox("Select Page", ["Pre-entrainement","Post-entrainement","Joueurs"])

if page == "Pre-entrainement":
    st.header("État de l'équipe")
    
   # Filter data for "Quand ?" == "pre-entrainement"
    pre_training_data = data[data['Quand ?'] == "pre-entrainement"]
    
    # Define the full list of players
    all_players = [
        "Doucouré","Basque","Ben Brahim","Calvet","Chadet","Cisse","Adehoumi",
        "Fischer", "Kalai","Koffi","M'bone",  "Barbet",
        "Moussadek", "Odzoumo", "Kouassi","Renaud","Renot","Yavorsky", "Guillaume",
        "Santini","Zemoura","Tchato","Ouchen","Traoré"
     
    ]

    # Date selection
    date_min = pre_training_data['Date'].min()
    date_max = pre_training_data['Date'].max()
    selected_date = st.sidebar.date_input(
        "Choisir la date:", 
        min_value=date_min, 
        max_value=date_max
    )

    # Filter data by the selected date
    filtered_data = pre_training_data[pre_training_data['Date'] == pd.Timestamp(selected_date)]

    # Convert relevant columns to numeric to avoid TypeError
    columns_to_convert = ['Sommeil','Stress', 'Fatigue', 'Courbature', 'Humeur','Alimentation']
    for col in columns_to_convert:
        filtered_data[col] = pd.to_numeric(filtered_data[col], errors='coerce')
        
   # Get the list of players who filled the questionnaire
    players_filled = filtered_data['Nom'].dropna().unique()
    players_not_filled = list(set(all_players) - set(players_filled))     

    # Drop unnecessary columns for display
    columns_to_display = ['Nom', 'Sommeil','Stress', 'Fatigue', 'Courbature', 'Humeur','Alimentation']
    filtered_data_display = filtered_data[columns_to_display]

    # Display the filtered data with gradient
    if not filtered_data_display.empty:


        # Apply color gradient
        def apply_gradient(df):
            return df.style.applymap(color_gradient, subset=['Sommeil','Stress', 'Fatigue', 'Courbature', 'Humeur','Alimentation'])
        
        st.dataframe(apply_gradient(filtered_data_display), use_container_width=True)

        # Calculate averages
        averages = filtered_data.groupby('Nom')[columns_to_convert].mean().reset_index()

        # Display players with high scores
        high_scores = filtered_data[(filtered_data['Sommeil'] > 3) |
                                    (filtered_data['Stress'] > 3) | 
                                     (filtered_data['Fatigue'] > 3) | 
                                     (filtered_data['Courbature'] > 3) | 
                                     (filtered_data['Humeur'] > 3) | 
                                     (filtered_data['Alimentation'] > 3)]
        if not high_scores.empty:
            st.write("##### Joueurs avec des scores supérieurs à 3:")
            for index, row in high_scores.iterrows():
                st.write(f"- {row['Nom']}: Sommeil {row['Sommeil']}- Stress {row['Stress']} - Fatigue {row['Fatigue']} - Courbature {row['Courbature']} - Humeur {row['Alimentation']} - Humeur {row['Alimentation']}")
        else:
            st.write("Aucun joueur avec des scores élevés pour aujourd'hui.")
    else:
        st.write(f"Aucune donnée disponible pour le {selected_date}.")

    # Display players who did not fill the questionnaire
    st.write("Joueurs n'ayant pas rempli le questionnaire:")
    if players_not_filled:
        for player in sorted(players_not_filled):
            st.write(f"- {player}")
    else:
        st.write("Tous les joueurs ont rempli le questionnaire.")
        
    st.markdown("### 🤕 Douleurs Déclarées")
    
    
    # Filter data by the selected date
    filtered_data = data[data['Date'] == pd.Timestamp(selected_date)]
    
    # Further filter for players who reported "Oui" for "Douleurs"
    players_with_pain = filtered_data[filtered_data['Douleurs'] == "Oui"]
    
    # Display an overview
    if not players_with_pain.empty:
        
        # Display a table of players and details
        columns_to_display = [
            'Nom', 'Identifie l\'emplacement de la douleur', 'Intensité de la douleur'
        ]
        st.dataframe(players_with_pain[columns_to_display])
    
    else:
        st.write(f"Aucun joueur n'a signalé de douleurs le {selected_date.strftime('%d-%m-%Y')}.")
        
        
    st.markdown("### ⚠️ Alertes")
    
        # --- Préparation pour les alertes : construire df_player sur la date sélectionnée ---
    # Ici on prend tous les joueurs qui ont répondu à la date sélectionnée (pré-entrainement dans ton flow)
    df_player = filtered_data.copy()  # filtered_data vient de la date + pré-entrainement
    
    # Normaliser le nom
    df_player['Nom'] = df_player['Nom'].str.strip().str.title()
    
    # Extraire numériquement les métriques si elles sont sous forme de string/list
    def extract_first_numeric(value):
        try:
            v = ast.literal_eval(value)
            if isinstance(v, list) and v:
                return float(v[0])
            return float(v)
        except (ValueError, SyntaxError, TypeError):
            try:
                return float(value)
            except:
                return float('nan')
    
    for col in ['Sommeil', 'Stress', 'Fatigue', 'Courbature', 'Humeur']:
        if col in df_player.columns:
            df_player.loc[:, col] = df_player[col].apply(extract_first_numeric)
        else:
            df_player.loc[:, col] = float('nan')
    
    # On garde uniquement les lignes complètes sur les 5 métriques
    df_player = df_player.dropna(subset=['Sommeil', 'Stress', 'Fatigue', 'Courbature', 'Humeur']).sort_values('Date').copy()
    
    # Moyennes mobiles 7 jours pour chaque composante
    for col in ['Sommeil', 'Stress', 'Fatigue', 'Courbature', 'Humeur']:
        df_player[f'{col}_7j'] = df_player[col].rolling(window=7, min_periods=1).mean()
    
    # Score Bien-être (plus bas = mieux)
    df_player['Score Bien-être'] = (
        df_player['Sommeil']
        + df_player['Stress']
        + df_player['Fatigue']
        + df_player['Courbature']
        + df_player['Humeur']
    ) / 5
    df_player['Score_7j'] = df_player['Score Bien-être'].rolling(window=7, min_periods=1).mean()

    
    # --- Paramètres d'alerte ---
    thresholds = {
        'Sommeil': 4.0,
        'Stress': 4.0,
        'Fatigue': 4.0,
        'Courbature': 4.0,
        'Humeur': 4.0,
        'Score Bien-être': 3.5
    }
    z_alert_zscore = 1.5
    window_baseline = 14
    sudden_threshold = 1.0  # variation rapide vs moyenne 7j
    
    # Joueurs à considérer (ceux qui ont répondu ce jour)
    players = filtered_data['Nom'].dropna().str.strip().str.title().unique()
    
    # Stocker alertes par joueur
    player_alerts_summary = []
    
    def safe_extract_numeric(val):
        try:
            v = ast.literal_eval(val)
            if isinstance(v, list) and v:
                return float(v[0])
            return float(v)
        except:
            try:
                return float(val)
            except:
                return float('nan')
    
    for player in sorted(players):
        # Série temporelle complète du joueur jusqu'à la date sélectionnée
        df_player = data[
            (data['Nom'].str.strip().str.title() == player) &
            (data['Date'] <= pd.Timestamp(selected_date))
        ].sort_values('Date').copy()
    
        # Nettoyage des métriques
        for col in ['Sommeil', 'Stress', 'Fatigue', 'Courbature', 'Humeur']:
            if col in df_player.columns:
                df_player.loc[:, col] = df_player[col].apply(safe_extract_numeric)
            else:
                df_player.loc[:, col] = float('nan')
        df_player = df_player.dropna(subset=['Sommeil', 'Stress', 'Fatigue', 'Courbature', 'Humeur'])
        if df_player.empty:
            continue  # pas assez de données
    
        # Rolling 7j pour chaque composante
        for col in ['Sommeil', 'Stress', 'Fatigue', 'Courbature', 'Humeur']:
            df_player[f'{col}_7j'] = df_player[col].rolling(window=7, min_periods=1).mean()
    
        # Score bien-être
        df_player['Score Bien-être'] = (
            df_player['Sommeil']
            + df_player['Stress']
            + df_player['Fatigue']
            + df_player['Courbature']
            + df_player['Humeur']
        ) / 5
        df_player['Score_7j'] = df_player['Score Bien-être'].rolling(window=7, min_periods=1).mean()
    
        # On prend le dernier point disponible (dernier jour <= selected_date)
        latest_idx = -1
        latest_row = df_player.iloc[latest_idx]
    
        # Collecte des alertes composantes
        comp_alerts = {}
        for var in ['Sommeil', 'Stress', 'Fatigue', 'Courbature', 'Humeur']:
            alert_flag = False
            reasons = []
    
            rolling_mean = df_player[var].rolling(window=window_baseline, min_periods=7).mean()
            rolling_std = df_player[var].rolling(window=window_baseline, min_periods=7).std()
    
            latest = latest_row[var]
            mean_latest = rolling_mean.iloc[latest_idx] if len(rolling_mean) >= abs(latest_idx) else np.nan
            std_latest = rolling_std.iloc[latest_idx] if len(rolling_std) >= abs(latest_idx) else np.nan
    
            # A. seuil absolu
            if latest >= thresholds[var]:
                alert_flag = True
                reasons.append(f"{var} ≥ seuil ({latest:.1f} ≥ {thresholds[var]})")
    
            # B. déviation baseline (z-score)
            if pd.notna(std_latest) and std_latest > 0 and pd.notna(mean_latest):
                zscore = (latest - mean_latest) / std_latest
                if zscore > z_alert_zscore:
                    alert_flag = True
                    reasons.append(f"{var} z-score élevé ({zscore:.2f})")
            else:
                zscore = 0
    
            # C. changement brusque vs 7j
            mm7 = df_player[f'{var}_7j'].iloc[latest_idx]
            delta = latest - mm7
            if delta > sudden_threshold:
                alert_flag = True
                reasons.append(f"Augmentation rapide vs 7j (Δ={delta:.2f})")
    
            comp_alerts[var] = {
                'alert': alert_flag,
                'latest': latest,
                'reasons': reasons
            }
    
        # Score Bien-être
        latest_score = latest_row['Score Bien-être']
        rolling_mean_score = df_player['Score Bien-être'].rolling(window=window_baseline, min_periods=7).mean()
        rolling_std_score = df_player['Score Bien-être'].rolling(window=window_baseline, min_periods=7).std()
        mean_score_latest = rolling_mean_score.iloc[latest_idx] if len(rolling_mean_score) >= abs(latest_idx) else np.nan
        std_score_latest = rolling_std_score.iloc[latest_idx] if len(rolling_std_score) >= abs(latest_idx) else np.nan
    
        score_alert = False
        score_reasons = []
    
        if latest_score >= thresholds['Score Bien-être']:
            score_alert = True
            score_reasons.append(f"Score ≥ seuil ({latest_score:.2f} ≥ {thresholds['Score Bien-être']})")
    
        if pd.notna(std_score_latest) and std_score_latest > 0 and pd.notna(mean_score_latest):
            score_z = (latest_score - mean_score_latest) / std_score_latest
            if score_z > z_alert_zscore:
                score_alert = True
                score_reasons.append(f"Score z-score élevé ({score_z:.2f})")
        else:
            score_z = 0
    
        # Tendance adverse : remontée soudaine
        if len(df_player) >= 4:
            prev_3 = df_player['Score Bien-être'].iloc[-4:-1].mean()
            if latest_score > prev_3 + 0.5:
                score_alert = True
                score_reasons.append(f"Tendance adverse vs récente (dernier {latest_score:.2f} > prev {prev_3:.2f})")
    
        # Composite
        comps_in_alert = [v for v in comp_alerts.values() if v['alert']]
        num_comps = len(comps_in_alert)
    
        if score_alert and num_comps >= 1:
            level = "ROUGE"
            summary = f"Score et au moins une composante en alerte."
        elif num_comps >= 2:
            level = "ROUGE"
            summary = "Plusieurs composantes en alerte."
        elif score_alert:
            level = "ORANGE"
            summary = "Score Bien-être dégradé."
        elif num_comps == 1:
            level = "ORANGE"
            summary = "Une seule composante en zone trouble."
        else:
            # amélioration si score nettement plus bas que baseline
            if pd.notna(std_score_latest) and pd.notna(mean_score_latest) and latest_score < (mean_score_latest - std_score_latest):
                level = "VERT"
                summary = "Amélioration significative."
            else:
                level = "OK"
                summary = "Stable."
    
        # Collecte pour affichage trié ensuite
        player_alerts_summary.append({
            'player': player,
            'level': level,
            'summary': summary,
            'score': latest_score,
            'score_reasons': score_reasons,
            'components': comp_alerts
        })
    
    # --- Affichage des joueurs à surveiller triés par sévérité ---
    priority_order = {'ROUGE': 0, 'ORANGE': 1, 'VERT': 2, 'OK': 3}
    player_alerts_summary.sort(key=lambda x: (priority_order.get(x['level'], 99), -x['score']))
    
    for info in player_alerts_summary:
        name = info['player']
        level = info['level']
        summary = info['summary']
        score = info['score']
        if level == "ROUGE":
            st.error(f"{name} – {level} : {summary} (Score={score:.2f})")
        elif level == "ORANGE":
            st.warning(f"{name} – {level} : {summary} (Score={score:.2f})")
        elif level == "VERT":
            st.success(f"{name} – {level} : {summary} (Score={score:.2f})")
        else:
            st.info(f"{name} – {level} : {summary} (Score={score:.2f})")
    
        with st.expander(f"Détails {name}"):
            if info['score_reasons']:
                st.write("Score Bien-être :")
                for r in info['score_reasons']:
                    st.write(f"  • {r}")
            st.write("Composantes :")
            for var, detail in info['components'].items():
                status = "⚠️" if detail['alert'] else "OK"
                line = f"  - {var}: {status}, valeur={detail['latest']:.2f}"
                st.write(line)
                for r in detail['reasons']:
                    st.write(f"      • {r}")
                    
      # Explication déroulable pour le head of performance
    with st.expander("📘 Comment fonctionnent les alertes (cliquer pour développer)"):
        st.markdown("""
        **1. Objectif général**  
        Le système produit des alertes individualisées pour chaque joueur, en évaluant à la fois les composantes subjectives de bien-être (Sommeil, Stress, Fatigue, Courbature, Humeur) et un score agrégé de bien-être. L’idée est de repérer rapidement les joueurs dont l’état se dégrade ou qui sortent de leur norme, pour prioriser les interventions.
    
        **2. Logique d’alerte (A à D)**  
        **A. Seuils absolus**  
        Pour chaque métrique et pour le score global, il y a des seuils définis au-delà desquels on considère que la valeur est préoccupante :  
        Exemples : Stress ≥ 4.0, Fatigue ≥ 4.0, Humeur ≥ 4.0, Score Bien-être ≥ 3.5.  
        Si une métrique dépasse son seuil, cela déclenche une alerte de type « dépassement absolu ».
    
        **B. Déviation par rapport à la baseline individuelle**  
        Chaque joueur a sa propre norme historique calculée sur une fenêtre roulante (14 jours) :  
        On calcule la moyenne et l’écart-type de chaque métrique sur cette période.  
        Si la valeur la plus récente dépasse sa moyenne habituelle de plus de 1.5 écarts-types (z-score > 1.5), c’est considéré comme une dérive significative par rapport à sa normale personnelle.
    
        **C. Changement brusque / tendance**  
        On compare la valeur la plus récente de chaque métrique à sa moyenne mobile sur 7 jours.  
        Si l’augmentation est rapide (delta > seuil, ici 1.0), cela signale un changement abrupt (par exemple une fatigue soudaine ou une augmentation du stress).  
        Pour le score de bien-être, une remontée récente par rapport à la période immédiate précédente est interprétée comme une tendance adverse.
    
        **D. Règles composites et priorisation**  
        Les alertes sont agrégées pour chaque joueur et classées :  
        - **Alerte rouge** : Score Bien-être en alerte et au moins une composante aussi, ou plusieurs composantes simultanément en alerte.  
        - **Alerte orange** : Une seule composante en zone trouble, ou le score de bien-être seul est dégradé.  
        - **Vert / reprise** : Le score baisse significativement par rapport à sa baseline (amélioration).  
        - **Stable** : Pas d’alerte majeure.
    
        **3. Ce que le reporting affiche**  
        Pour chaque joueur (triés par sévérité) :  
        - **Niveau global** : ROUGE / ORANGE / VERT / OK avec un résumé court (ex. : “Score et une composante en alerte”, “Une seule composante problématique”).  
        - **Score Bien-être** : valeur récente, indication de dépassement de seuil, z-score, tendance.  
        - **Composantes individuelles** : pour Sommeil, Stress, Fatigue, Courbature et Humeur :  
            - Valeur actuelle,  
            - Si elles sont en alerte,  
            - Et les motivations : dépassement de seuil, z-score élevé, augmentation rapide vs moyenne mobile 7j.  
        - **Détail dépliable** : permet de voir la granularité des causes pour chaque joueur.
        """)                  

elif page == "Post-entrainement":
    st.header("État de l'équipe")

    # Filter data for "Quand ?" == "post-entrainement"
    post_training_data = data[data['Quand ?'] == "post-entrainement"]

    # Date selection
    date_min = post_training_data['Date'].min()
    date_max = post_training_data['Date'].max()
    selected_date = st.sidebar.date_input(
        "Choisir une date:", 
        min_value=date_min, 
        max_value=date_max
    )

    # Filter data by the selected date
    filtered_data = post_training_data[post_training_data['Date'] == pd.Timestamp(selected_date)]

    # Columns to display
    columns_to_display = ['Nom', 'Humeur-Post', 'Plaisir-Post', 'Progression']

    if not filtered_data.empty:
        # Convert columns to numeric if necessary and cast to integers
        for col in ['Humeur-Post', 'Plaisir-Post', 'Progression']:
            filtered_data[col] = pd.to_numeric(filtered_data[col], errors='coerce').fillna(0).astype(int)

        # Drop unnecessary columns for display
        filtered_data_display = filtered_data[columns_to_display]

        # Apply color gradient to numeric columns
        def apply_gradient(df):
            return df.style.applymap(color_gradient, subset=['Humeur-Post', 'Plaisir-Post', 'Progression'])

        # Display the table with gradients applied
        st.write(f"#### {selected_date.strftime('%d-%m-%Y')}")
        st.dataframe(apply_gradient(filtered_data_display), use_container_width=True)

        # Calculate averages
        averages = filtered_data[['Humeur-Post', 'Plaisir-Post', 'Progression']].mean().round(0).astype(int)

        # Convert averages to DataFrame with column names "Métrique" and "Moyenne"
        averages_df = pd.DataFrame({"Métrique": averages.index, "Moyenne": averages.values})

        # Display averages as "Score moyen du jour"
        st.write("#### Score moyen du jour")
        st.dataframe(averages_df, use_container_width=True)

        # Check for players who didn't fill the form
        all_players = [
             "Doucouré","Basque","Ben Brahim","Calvet","Chadet","Cisse","Adehoumi",
             "Fischer", "Kalai","Koffi","M'bone",  "Barbet", 
             "Moussadek", "Odzoumo", "Kouassi","Renaud","Renot","Yavorsky", "Guillaume",
             "Santini","Zemoura","Tchato","Ouchen","Traoré"
          
        ]

        players_filled = filtered_data['Nom'].dropna().unique()
        players_not_filled = list(set(all_players) - set(players_filled))

        st.write("Joueurs n'ayant pas rempli le questionnaire:")
        if players_not_filled:
            for player in sorted(players_not_filled):
                st.write(f"- {player}")
        else:
            st.write("Tous les joueurs ont rempli le questionnaire.")
    else:
        st.write(f"Aucune donnée disponible pour le {selected_date.strftime('%d-%m-%Y')}.")

    # # Filter data for "Quand ?" == "post-entrainement"
    # post_training_data = data[data['Quand ?'] == "post-entrainement"]
    
    # # Sidebar selection for variable
    # selected_variable = st.sidebar.selectbox(
    #     "Choisir une variable:", 
    #     ["Humeur-Post", "Plaisir-Post", "RPE"]
    # )
    
    # # Sidebar date range selection
    # date_min = post_training_data['Date'].min()
    # date_max = post_training_data['Date'].max()
    # selected_date_range = st.sidebar.date_input(
    #     "Choisir une plage de dates:", 
    #     [date_min, date_max], 
    #     min_value=date_min, 
    #     max_value=date_max
    # )
    
    # # Ensure correct format for date selection
    # if isinstance(selected_date_range, tuple):
    #     start_date, end_date = selected_date_range
    # else:
    #     start_date, end_date = date_min, date_max  # Fallback in case of error
    
    # # Filter data by the selected date range
    # filtered_data = post_training_data[
    #     (post_training_data['Date'] >= pd.Timestamp(start_date)) & 
    #     (post_training_data['Date'] <= pd.Timestamp(end_date))
    # ]
    
    # # Convert selected variable to numeric if necessary
    # filtered_data[selected_variable] = pd.to_numeric(filtered_data[selected_variable], errors='coerce')
    
    # # Interactive plot
    # if not filtered_data.empty:
    #     fig = px.line(
    #         filtered_data, 
    #         x="Date", 
    #         y=selected_variable, 
    #         color="Nom",
    #         markers=True, 
    #         title=f"Évolution de {selected_variable}"
    #     )
    
    #     # Customize layout
    #     fig.update_layout(
    #         xaxis_title="Date",
    #         yaxis_title=selected_variable,
    #         hovermode="closest"
    #     )
    
    #     # Display interactive plot
    #     st.plotly_chart(fig, use_container_width=True)
    
    #     # Display mean per player
    #     mean_per_player = filtered_data.groupby("Nom")[selected_variable].mean().reset_index()
    #     mean_per_player.columns = ["Nom", f"Moyenne {selected_variable}"]
    
    #     st.write(f"### Moyenne de {selected_variable} par joueur")
    #     st.dataframe(mean_per_player, use_container_width=True)
    
    # else:
    #     st.write("Aucune donnée disponible pour la plage de dates sélectionnée.")
    


elif page == "Joueurs":

    st.header("Joueur")
    
    # --- Normalisation des noms ---
    data['Nom'] = data['Nom'].str.strip().str.title()
    
    # Liste canonique de joueurs et sélection
    all_players = [
        "Doucouré","Basque","Ben Brahim","Calvet","Chadet","Cisse","Adehoumi",
        "Fischer", "Kalai","Koffi","M'bone", "Barbet",
        "Moussadek", "Odzoumo", "Kouassi","Renaud","Renot","Yavorsky", "Guillaume",
        "Santini","Zemoura","Tchato","Ouchen","Traoré"
    ]
    available_players = sorted([p for p in all_players if p in data['Nom'].unique()])
    selected_name = st.selectbox("Choisir un nom:", options=available_players)
    
    # Filtrage et copie
    df_player = data.loc[data['Nom'] == selected_name].copy()
    
    # --- utilitaire pour extraire un nombre depuis potentiellement une liste ---
    def extract_first_numeric(value):
        try:
            value = ast.literal_eval(value)
            if isinstance(value, list) and value:
                return float(value[0])
            return float(value)
        except (ValueError, SyntaxError, TypeError):
            return float('nan')
    
    # --- Nettoyage des colonnes d'intérêt ---
    for col in ['Sommeil', 'Stress', 'Fatigue', 'Courbature', 'Humeur']:
        if col in df_player.columns:
            df_player.loc[:, col] = df_player[col].apply(extract_first_numeric)
        else:
            df_player.loc[:, col] = float('nan')
    
    # On garde uniquement les lignes complètes sur les 5 métriques
    required = ['Sommeil', 'Stress', 'Fatigue', 'Courbature', 'Humeur']
    df_player = df_player.dropna(subset=required).sort_values('Date').copy()
    
    # --- Moyennes mobiles 7 jours pour chaque variable ---
    for col in ['Sommeil', 'Stress', 'Fatigue', 'Courbature', 'Humeur']:
        df_player[f'{col}_7j'] = df_player[col].rolling(window=7, min_periods=1).mean()
    
    # --- Score bien-être (plus bas = mieux) ---
    df_player['Score Bien-être'] = (
        df_player['Sommeil']
        + df_player['Stress']
        + df_player['Fatigue']
        + df_player['Courbature']
        + df_player['Humeur']
    ) / 5
    
    df_player['Score_3j'] = df_player['Score Bien-être'].rolling(window=3, min_periods=1).mean()
    df_player['Score_7j'] = df_player['Score Bien-être'].rolling(window=7, min_periods=1).mean()
    
    # --- Tendance (baisse du score = amélioration) ---
    if len(df_player) >= 4:
        latest = df_player.iloc[-1]['Score Bien-être']
        previous_mean = df_player.iloc[-4:-1]['Score Bien-être'].mean()
        if latest < previous_mean:
            trend = "📈 Le bien-être du joueur s'améliore (score baisse)."
        elif latest > previous_mean:
            trend = "📉 Le bien-être semble se dégrader (score augmente)."
        else:
            trend = "⏸️ Score stable."
    else:
        trend = "ℹ️ Pas assez de données pour évaluer la tendance."
    
    # --- Sélecteur de métriques (Score en premier) ---
    metrics_to_show = st.multiselect(
        "Choisir les métriques à afficher",
        options=[
            'Score Bien-être',
            'Sommeil',
            'Stress',
            'Fatigue',
            'Courbature',
            'Humeur'
        ],
        default=[
            'Score Bien-être'
        ]
    )
    
    # --- Graphe ---
    fig = go.Figure()
    
    # Score bien-être (toujours brut, sans moyenne mobile superposée ici)
    if 'Score Bien-être' in metrics_to_show:
        fig.add_trace(go.Scatter(
            x=df_player['Date'],
            y=df_player['Score Bien-être'],
            mode='lines+markers',
            name='Score Bien-être',
            line=dict(color='black', width=4, dash='dash')
        ))
    
    # Pour chaque variable sélectionnée, afficher la série et sa moyenne mobile 7j
    var_config = {
        'Sommeil': 'blue',
        'Stress': 'purple',
        'Fatigue': 'red',
        'Courbature': 'orange',
        'Humeur': 'green'
    }
    
    for var, color in var_config.items():
        if var in metrics_to_show:
            # valeur brute
            fig.add_trace(go.Scatter(
                x=df_player['Date'],
                y=df_player[var],
                mode='lines+markers',
                name=var,
                line=dict(color=color)
            ))
            # moyenne mobile 7j en trait pointillé
            fig.add_trace(go.Scatter(
                x=df_player['Date'],
                y=df_player[f'{var}_7j'],
                mode='lines',
                name=f'{var} 7j',
                line=dict(color=color, dash='dash')
            ))
    
    fig.update_layout(
        title=f"Métriques de bien-être pour {selected_name} (plus bas = mieux)",
        xaxis_title="Date",
        yaxis=dict(
        title="Valeur (plus bas = meilleur)",
            range=[1, 5]  # force l'échelle de 1 à 5
        ),
        hovermode="x unified",
        template="simple_white",
        legend=dict(orientation="h", y=-0.3),
        height=550
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f"**{trend}**")
    
    
    
    
    
