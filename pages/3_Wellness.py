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
        st.write(f"#### {selected_date.strftime('%d-%m-%Y')}")

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
            st.write("#### Joueurs avec des scores supérieurs à 3:")
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
        
    st.header("Douleurs déclarées")
    
    
    # Filter data by the selected date
    filtered_data = data[data['Date'] == pd.Timestamp(selected_date)]
    
    # Further filter for players who reported "Oui" for "Douleurs"
    players_with_pain = filtered_data[filtered_data['Douleurs'] == "Oui"]
    
    # Display an overview
    if not players_with_pain.empty:
        st.write(f"### {selected_date.strftime('%d-%m-%Y')}")
        
        # Display a table of players and details
        columns_to_display = [
            'Nom', 'Identifie l\'emplacement de la douleur', 'Intensité de la douleur'
        ]
        st.dataframe(players_with_pain[columns_to_display])
    
    else:
        st.write(f"Aucun joueur n'a signalé de douleurs le {selected_date.strftime('%d-%m-%Y')}.")
           

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
    
    # Nettoyage des noms dans le dataframe
    data['Nom'] = data['Nom'].str.strip().str.title()
    
    # Liste fixe des joueurs (normalisée aussi)
    all_players = [
         "Doucouré","Basque","Ben Brahim","Calvet","Chadet","Cisse","Adehoumi",
         "Fischer", "Kalai","Koffi","M'bone",  "Barbet",
         "Moussadek", "Odzoumo", "Kouassi","Renaud","Renot","Yavorsky", "Guillaume",
         "Santini","Zemoura","Tchato","Ouchen","Traoré"
      
    ]
    
    # Ne garder que ceux présents dans les données
    available_players = sorted([p for p in all_players if p in data['Nom'].unique()])
    selected_name = st.sidebar.selectbox("Choisir un nom:", options=available_players)

    # Filter data for the selected player
    data_filtered_by_name = data[data['Nom'] == selected_name]
    
    def extract_first_numeric(value):
        try:
            value = ast.literal_eval(value)
            if isinstance(value, list) and value:
                return float(value[0])
            return float(value)
        except (ValueError, SyntaxError, TypeError):
            return float('nan')
        
        # Nettoyage de la colonne RPE
    data_filtered_by_name['RPE'] = pd.to_numeric(data_filtered_by_name['RPE'], errors='coerce')
    rpe_data = data_filtered_by_name[['Date', 'RPE']].dropna().sort_values('Date')
    
    if not rpe_data.empty:
        rpe_data['RPE_7j'] = rpe_data['RPE'].rolling(window=7, min_periods=1).mean()
        rpe_data['RPE_28j'] = rpe_data['RPE'].rolling(window=28, min_periods=1).mean()
    
        fig_rpe = go.Figure()
        
        # Barres pour les RPE journaliers
        fig_rpe.add_trace(go.Bar(
            x=rpe_data['Date'],
            y=rpe_data['RPE'],
            name="RPE quotidien",
            marker_color='rgba(55, 128, 191, 0.7)'
        ))
    
        # Moyenne 7 jours
        fig_rpe.add_trace(go.Scatter(
            x=rpe_data['Date'],
            y=rpe_data['RPE_7j'],
            mode='lines+markers',
            name='Moyenne 7 jours',
            line=dict(color='orange')
        ))
    
        # Moyenne 28 jours
        fig_rpe.add_trace(go.Scatter(
            x=rpe_data['Date'],
            y=rpe_data['RPE_28j'],
            mode='lines+markers',
            name='Moyenne 28 jours',
            line=dict(color='green')
        ))
    
        fig_rpe.update_layout(
            title=f"Suivi du RPE pour {selected_name}",
            xaxis_title="Date",
            yaxis_title="RPE",
            barmode='overlay',
            template='simple_white',
            height=400
        )
        
        st.plotly_chart(fig_rpe, use_container_width=True)
    else:
        st.info(f"Aucune donnée RPE disponible pour {selected_name}.")    
        
  
    
    # Nettoyage des colonnes
    for col in ['Sommeil', 'Fatigue', 'Courbature', 'Humeur']:
        data_filtered_by_name[col] = data_filtered_by_name[col].apply(extract_first_numeric)
    
    data_filtered_by_name = data_filtered_by_name.dropna(subset=['Sommeil', 'Fatigue', 'Courbature', 'Humeur'])
    data_filtered_by_name = data_filtered_by_name.sort_values("Date")
    
    # Score bien-être personnalisé
    def compute_score(df):
        return (df['Sommeil'] + (7 - df['Fatigue']) + (7 - df['Courbature']) + df['Humeur']) / 4
    
    data_filtered_by_name['Score Bien-être'] = data_filtered_by_name.apply(compute_score, axis=1)
    
    # Moyennes glissantes
    data_filtered_by_name['Score_3j'] = data_filtered_by_name['Score Bien-être'].rolling(window=3, min_periods=1).mean()
    data_filtered_by_name['Score_7j'] = data_filtered_by_name['Score Bien-être'].rolling(window=7, min_periods=1).mean()
    
    # Détection de tendance
    latest_score = data_filtered_by_name.iloc[-1]['Score Bien-être']
    prev_score = data_filtered_by_name.iloc[-4:-1]['Score Bien-être'].mean()
    
    if latest_score > prev_score:
        trend = "📈 Le bien-être du joueur est en amélioration ces derniers jours."
    elif latest_score < prev_score:
        trend = "📉 Le bien-être du joueur semble se dégrader récemment."
    else:
        trend = "⏸️ Le bien-être du joueur est stable."
    
    # GRAPHIQUE PLOTLY
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data_filtered_by_name['Date'],
        y=data_filtered_by_name['Sommeil'],
        mode='lines+markers',
        name='Sommeil',
        line=dict(color='blue')
    ))
    
    fig.add_trace(go.Scatter(
        x=data_filtered_by_name['Date'],
        y=7 - data_filtered_by_name['Fatigue'],  # inversé pour cohérence visuelle
        mode='lines+markers',
        name='(7 - Fatigue)',
        line=dict(color='red')
    ))
    
    fig.add_trace(go.Scatter(
        x=data_filtered_by_name['Date'],
        y=7 - data_filtered_by_name['Courbature'],  # inversé pour cohérence visuelle
        mode='lines+markers',
        name='(7 - Courbature)',
        line=dict(color='orange')
    ))
    
    fig.add_trace(go.Scatter(
        x=data_filtered_by_name['Date'],
        y=data_filtered_by_name['Humeur'],
        mode='lines+markers',
        name='Humeur',
        line=dict(color='green')
    ))
    
    fig.add_trace(go.Scatter(
        x=data_filtered_by_name['Date'],
        y=data_filtered_by_name['Score Bien-être'],
        mode='lines+markers',
        name='Score Bien-être',
        line=dict(color='purple', width=4, dash='dash')
    ))
    
    # Layout
    fig.update_layout(
        title=f"Métriques de bien-être pour {selected_name}",
        xaxis_title="Date",
        yaxis_title="Échelle 1-7",
        hovermode="x unified",
        template="simple_white",
        legend=dict(orientation="h", y=-0.3),
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f"**{trend}**")
