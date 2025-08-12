#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 21:09:07 2025

@author: fcvmathieu
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import os
import pickle
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
import squarify
import seaborn as sns
from datetime import date

# Constants for Google Sheets
SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
TOKEN_FILE = 'token_ent.pickle'  # Replace with your credentials file path
SPREADSHEET_ID = '15n4XkQHrUpSPAW61vmR_Rk1kibd5zcmVqgHA40szlPg'  # Replace with your actual Spreadsheet ID
RANGE_NAME = 'Feuille 1'  # Replace with your range name

st.set_page_config(layout='wide')

# Display the club logo from GitHub at the top right
logo_url = 'https://raw.githubusercontent.com/FC-Versailles/care/main/logo.png'
col1, col2 = st.columns([9, 1])
with col1:
    st.title("Entrainement | FC Versailles")
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
# Streamlit cache for loading data
@st.cache_data(ttl=60)
def load_data():
    data = fetch_google_sheet(SPREADSHEET_ID, RANGE_NAME)
    data = data[~data['Type'].isin(['Salle', 'Dev Individuel'])]
    return data

# Effacer le cache si nécessaire
st.cache_data.clear()

# Streamlit App
def main():

    # --- Bouton "Questionnaire" (lien externe) ---
    questionnaire_url = "https://tally.so/forms/3ql17d/share"
    st.markdown(
        f'''
        <a href="{questionnaire_url}" target="_blank" style="
            display: inline-block;
            padding: 10px 18px;
            background-color: #2563eb;
            color: white;
            text-decoration: none;
            border-radius: 8px;
            font-weight: 600;
            margin-bottom: 12px;
        ">
            Questionnaire
        </a>
        ''',
        unsafe_allow_html=True
    )

    data = load_data()
    
    if not data.empty:
        # Convert "Date" column to datetime format
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
        
        # Sidebar filters
        st.sidebar.header("Filtres")
        session_types = st.sidebar.multiselect("Type de séance", data['Type'].unique(), default=data['Type'].unique())
        
        start_default = date(2025, 7, 1)
        end_default = data['Date'].max().date()
        # Si tu veux empêcher l'utilisateur de sélectionner avant le 01/07/2025, ajoute min_value=start_default
        date_range = st.sidebar.date_input(
            "Sélectionner une période",
            [start_default, end_default],
            min_value=start_default,
            max_value=end_default
        )
        
        
        # Apply filters
        filtered_data = data[(data['Date'] >= pd.to_datetime(date_range[0])) &
                             (data['Date'] <= pd.to_datetime(date_range[1])) &
                             (data['Type'].isin(session_types))]

        # Display filtered data


        
        # Treemap Visualization
        st.write("### Répartition des activités")
        activity_columns = [col for col in filtered_data.columns if col.startswith('Temps')]
        if activity_columns:
            activity_data = filtered_data.melt(id_vars=[], value_vars=activity_columns, var_name='Temps', value_name='Activité').dropna()
            activity_counts = activity_data[(activity_data['Activité'] != 'RAS') & (activity_data['Activité'] != 'Prévention')]['Activité'].value_counts()
            
            labels = [f"{label}\n{count}" for label, count in zip(activity_counts.index, activity_counts.values)]
            
            fig = go.Figure()
            fig.add_trace(go.Treemap(labels=labels, parents=['']*len(activity_counts), values=activity_counts.values))
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Stacked bar chart by session type
        st.write("### Répartition des procédés par type d'entraînement")
        melted_data = filtered_data.melt(id_vars=["Type"], value_vars=activity_columns, var_name="Temps", value_name="Activité").dropna()
        activity_by_type = melted_data[melted_data['Activité'] != 'RAS'].groupby(['Type', 'Activité']).size().unstack(fill_value=0)
        
        fig = go.Figure()
        for col in activity_by_type.columns:
            fig.add_trace(go.Bar(y=activity_by_type.index, x=activity_by_type[col], name=col, orientation='h'))
        fig.update_layout(barmode='stack', xaxis_title='Nombre d\'apparitions', yaxis_title='Activité')
        fig.update_layout(legend_title_text='Type', legend=dict(x=1.05, y=1))
        st.plotly_chart(fig, use_container_width=True)
            # Scatter plot (square layout)
        st.write("### Répartition des contenus par date")
        scatter_data = filtered_data.melt(id_vars=['Date'], value_vars=activity_columns, var_name='Temps', value_name='Activité').dropna()
        scatter_data = scatter_data[scatter_data['Activité'] != 'RAS']
        
        scatter_data['Date'] = pd.to_datetime(scatter_data['Date'], format='%d-%m-%y')
        fig = px.scatter(scatter_data, x='Date', y='Activité', color='Activité', size_max=10)
        fig.update_traces(marker=dict(symbol='square', size=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Aucune donnée chargée. Veuillez vérifier votre connexion à Google Sheets.")

if __name__ == "__main__":
    main()


