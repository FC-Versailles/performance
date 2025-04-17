#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 16:07:28 2025

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
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from matplotlib.backends.backend_pdf import PdfPages
import plotly.express as px

# Constants for Google Sheets
SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
TOKEN_FILE = 'token.pickle'
SPREADSHEET_ID = '1CBWB1XKc4FIHFd6w7z8QW5SgLKJBwpvBrcEgDLbPxqI'  # Replace with your actual Spreadsheet ID
RANGE_NAME = 'Feuille 1'

st.set_page_config(layout='wide')

# Display the club logo from GitHub at the top right
logo_url = 'https://raw.githubusercontent.com/FC-Versailles/nutrition/main/logo.png'
col1, col2 = st.columns([9, 1])
with col1:
    st.title("Nutrition | FC Versailles")
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


# Load data from Google Sheets
@st.cache_data(ttl=60)
def load_data():
    return fetch_google_sheet(SPREADSHEET_ID, RANGE_NAME)


# Load your data
data = load_data()

data = data[~data['Nom'].isin(['Agoro', 'Bangoura', 'Mbala'])]

# Data preprocessing
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
data['Poids'] = data['Poids'].str.replace(',', '.')
data['MG %'] = data['MG %'].str.replace(',', '.')

# Replace empty strings with NaN
data.replace('', np.nan, inplace=True)

# Drop rows with NaN values in 'Poids' and 'MG %'
data.dropna(subset=['Poids', 'MG %'], inplace=True)

# Convert columns to float
data['Poids'] = data['Poids'].astype(float)
data['MG %'] = data['MG %'].astype(float)

# Sidebar navigation
st.sidebar.title("Nutrition")
page = st.sidebar.selectbox("Select Page", ["Equipe", "Joueurs"])

if page == "Equipe":
    st.title("Etat de l'équipe")
    st.markdown("#### Choisir la date")
    if not data.empty:
        min_date = data['Date'].min().date()
        max_date = data['Date'].max().date()
        available_dates = data['Date'].dt.date.unique()
        available_dates.sort()

        # Set the most recent date as default
        if available_dates.size > 0:
            default_date = available_dates[-1]
            selected_date = st.selectbox("Date:", options=available_dates, index=len(available_dates) - 1)

        filtered_data = data[data['Date'].dt.date == selected_date]

        if not filtered_data.empty:
            # Select relevant columns
            final_table = filtered_data[['Nom', 'Poids', 'MG %']]

            # Define the highlighting function for 'MG %'
            def highlight_mg(val):
                if val < 10:
                    return 'background-color: limegreen; text-align: center;'
                elif 10 <= val <= 12:
                    return 'background-color: lemonchiffon; text-align: center;'
                else:
                    return 'background-color: tomato; text-align: center;'

            # Apply the highlighting function to the 'MG %' column and center-align text
            styled_table = final_table.style.applymap(highlight_mg, subset=['MG %']).format({'Poids': '{:.2f}', 'MG %': '{:.2f}'})
            # Center-align all table cells (headers and data)
            styled_table = styled_table.set_table_styles([
                {'selector': 'th', 'props': [('text-align', 'center'), ('vertical-align', 'middle')]},
                {'selector': 'td', 'props': [('text-align', 'center'), ('vertical-align', 'middle')]}
            ])

            # Calculate dynamic height for the table based on the number of rows
            num_rows = len(final_table)
            row_height = 35  # Approximate pixel height per row
            table_height = max(400, num_rows * row_height)  # Minimum height of 400px

            # Display the styled table
            st.markdown(f"#### Suivi des Poids et MG % ({selected_date})")
            st.dataframe(styled_table, use_container_width=True, height=table_height)
        else:
            st.warning("No data available for the selected date.")
    else:
        st.warning("No available dates with data.")

elif page == "Joueurs":
    st.title("Fiche Joueur")
    # Filter by player name (Nom)
    player_names = data['Nom'].dropna().unique()
    selected_player = st.selectbox("Select a player:", options=player_names)

    # Filter data for the selected player
    player_data = data[data['Nom'] == selected_player].sort_values(by='Date')
    
    # Remove rows where Nom is in the specified lis
    if not player_data.empty:
        # Plot interactive graph: Date vs Poids
        st.markdown(f"### Evolution du poids")
        fig = px.line(
            player_data,
            x='Date',
            y='Poids',
            title=f"{selected_player}",
            labels={"Date": "Date", "Poids": "Poids (kg)"},
        )
        fig.update_traces(mode="lines+markers", connectgaps=False)  # Ensure line connects only between valid dates
        fig.update_layout(xaxis_title="Date", yaxis_title="Poids (kg)", title_x=0.5)

        # Display the plot
        st.plotly_chart(fig, use_container_width=True)

        # Display the raw data for the player
        st.markdown("### Données du joueurs")
        st.dataframe(player_data[['Date', 'Poids', 'MG %','Remarque']].assign(Date=player_data['Date'].dt.strftime('%d-%m-%y')),use_container_width=False)
    else:
        st.warning(f"No data available for {selected_player}.")
