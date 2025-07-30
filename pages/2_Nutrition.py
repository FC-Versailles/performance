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
from datetime import date

# Constants for Google Sheets
SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
TOKEN_FILE = 'token_nut.pickle'
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
data['Saison'] = data['Saison'].astype(str).str.strip()
data = data[data['Saison'] == '2526']
data['Date'] = data['Date'].astype(str).str.strip()
data['Date'] = data['Date'].replace('', np.nan)
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')


data['Poids'] = data['Poids'].str.replace(',', '.')
data['MG %'] = data['MG %'].str.replace(',', '.')

# Replace empty strings with NaN
data.replace('', np.nan, inplace=True)

# Drop rows with NaN values in 'Poids' and 'MG %'
data.dropna(subset=['Poids'], inplace=True)

# Convert columns to float
data['Poids'] = data['Poids'].astype(float)
data['MG %'] = data['MG %'].astype(float)

# Sidebar navigation
st.sidebar.title("Nutrition")
page = st.sidebar.selectbox("Select Page", ["Equipe", "Joueurs"])

if page == "Equipes":
    st.title("Etat de l'équipe")
    st.markdown("#### Choisir la date")
    if not data.empty:
        available_dates = data['Date'].dropna().dt.date.unique()
        available_dates = sorted(available_dates)
    else:
        available_dates = []

    if len(available_dates) > 0:
        default_date = available_dates[-1]
        selected_date = st.date_input(
            "Date :",
            value=default_date,
            min_value=min(available_dates),
            max_value=max(available_dates),
            format="YYYY-MM-DD"
        )

        filtered_data = data[data['Date'].dt.date == selected_date]

        if not filtered_data.empty:
            # Poids ref pour chaque joueur
            poids_ref_dict = data.groupby('Nom')['Poids'].min().to_dict()
            final_table = filtered_data[['Nom', 'Poids', 'MG %']].copy()
            final_table['Poids ref'] = final_table['Nom'].map(poids_ref_dict)

            # Highlight MG %
            def highlight_mg(val):
                if val < 10:
                    return 'background-color: limegreen; text-align: center;'
                elif 10 <= val <= 12:
                    return 'background-color: lemonchiffon; text-align: center;'
                else:
                    return 'background-color: tomato; text-align: center;'

            # Highlight Poids (rouge si +2kg du ref)
            def highlight_poids(row):
                try:
                    if (row['Poids'] - row['Poids ref']) >= 2:
                        return ['background-color: tomato; text-align: center;' if col == 'Poids' else '' for col in final_table.columns]
                    else:
                        return ['text-align: center;' for _ in final_table.columns]
                except Exception:
                    return ['text-align: center;' for _ in final_table.columns]

            # Styling combiné
            styled_table = (
                final_table.style
                    .applymap(highlight_mg, subset=['MG %'])
                    .apply(highlight_poids, axis=1)
                    .format({'Poids': '{:.2f}', 'MG %': '{:.2f}', 'Poids ref': '{:.2f}'})
                    .set_table_styles([
                        {'selector': 'th', 'props': [('text-align', 'center'), ('vertical-align', 'middle')]},
                        {'selector': 'td', 'props': [('text-align', 'center'), ('vertical-align', 'middle')]}
                    ])
            )

            num_rows = len(final_table)
            row_height = 35
            table_height = max(400, num_rows * row_height)
            st.dataframe(styled_table, use_container_width=True, height=table_height)
        else:
            st.warning("No data available for the selected date.")
    else:
        st.warning("No available dates with data.")
        

elif page == "Equipe":
    st.title("Etat de l'équipe")
    st.markdown("#### Choisir la semaine")
    
    # build list of Mondays
    mondays = sorted({
        (d.date() - timedelta(days=d.weekday()))
        for d in data['Date'].dropna()
    })
    if not mondays:
        st.warning("No available weeks with data.")
    else:
        # week selector (default = latest)
        sel_idx = len(mondays) - 1
        selected_monday = st.selectbox(
            "Semaine:",
            mondays,
            index=sel_idx,
            format_func=lambda d: d.strftime("%d-%m-%Y")
        )
    
        # key dates
        monday    = pd.to_datetime(selected_monday)
        tuesday   = monday + timedelta(days=1)
        wednesday = monday + timedelta(days=2)
        # prior calendar week range (for “Poids Semaine”)
        last_week_start = monday - timedelta(days=7)
        last_week_end   = last_week_start + timedelta(days=6)
    
        # assemble data
        rows = []
        for nom, grp in data.groupby('Nom'):
            grp = grp.set_index('Date')
            def get_p(dt):
                try:
                    return float(grp.at[pd.to_datetime(dt), 'Poids'])
                except:
                    return np.nan
    
            j_p3 = get_p(monday)
            j_m3 = get_p(tuesday)
            j_m2 = get_p(wednesday)
    
            # pick the most recent J-day we actually have
            if not np.isnan(j_m2):
                ref_day = wednesday
            elif not np.isnan(j_m3):
                ref_day = tuesday
            elif not np.isnan(j_p3):
                ref_day = monday
            else:
                ref_day = None
    
            # J-7 = 7 days before that ref_day
            j7 = get_p(ref_day - timedelta(days=7)) if ref_day else np.nan
    
            week_mean = grp.loc[
                (grp.index >= last_week_start) &
                (grp.index <= last_week_end),
                'Poids'
            ].mean()
    
            rows.append({
                'Nom':            nom,
                'Poids J+3':      j_p3,
                'Poids J-3':      j_m3,
                'Poids J-2':      j_m2,
                'Poids J-7':      j7,
                'Poids Semaine':  week_mean
            })
    
        final = pd.DataFrame(rows)
    
        # styling: highlight J‑columns based on the max–min range
        def highlight_diff(row):
            vals = []
            for c in ['Poids J+3','Poids J-3','Poids J-2']:
                if pd.notna(row[c]):
                    vals.append(row[c])
            diff = max(vals) - min(vals) if len(vals) >= 2 else 0
            style_list = []
            for col in final.columns:
                if col in ['Poids J+3','Poids J-3','Poids J-2']:
                    if diff >= 2:
                        style_list.append('background-color: lightcoral; text-align: center;')
                    elif diff >= 1.5:
                        style_list.append('background-color: lightsalmon; text-align: center;')
                    elif diff >= 1:
                        style_list.append('background-color: lightyellow; text-align: center;')
                    else:
                        style_list.append('text-align: center;')
                else:
                    style_list.append('text-align: center;')
            return style_list
    
        styled = (
            final.style
                 .apply(highlight_diff, axis=1)
                 .format(
                     {
                         'Poids J+3': '{:.2f}',
                         'Poids J-3': '{:.2f}',
                         'Poids J-2': '{:.2f}',
                         'Poids J-7': '{:.2f}',
                         'Poids Semaine': '{:.2f}',
                     },
                     na_rep='-'
                 )
                 .set_table_styles([
                     {'selector': 'th', 'props': [('text-align','center')]},
                     {'selector': 'td', 'props': [('text-align','center')]}
                 ])
        )
    
        st.dataframe(
            styled,
            use_container_width=True,
            height=max(400, len(final) * 35)
        )


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
