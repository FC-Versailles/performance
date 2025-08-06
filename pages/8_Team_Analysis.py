#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 21:11:53 2025

@author: fcvmathieu
"""

import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(layout='wide')
col1, col2 = st.columns([9,1])
with col1:
    st.title("Analyse Equipe | FC Versailles")
with col2:
    st.image(
        'https://raw.githubusercontent.com/FC-Versailles/wellness/main/logo.png',
        use_container_width=True
    )
st.markdown("<hr style='border:1px solid #ddd' />", unsafe_allow_html=True)

# ── Constants ────────────────────────────────────────────────────────────────
SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
TOKEN_FILE_PTS = 'token_pts.pickle'
SPREADSHEET_ID = '19Sear5ZiqBk_dzqqMZsS27CHECqgz7fHC4LlgsqKkZY'
SHEET_NAME = 'Feuille 1'
RANGE_NAME = f"'{SHEET_NAME}'!A1:Z1000"  # Adjust range as needed

# ── Authentication & Data Fetching ──────────────────────────────────────────
def get_pts_credentials():
    creds = None
    if os.path.exists(TOKEN_FILE_PTS):
        with open(TOKEN_FILE_PTS, 'rb') as f:
            creds = pickle.load(f)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('client_secret.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE_PTS, 'wb') as f:
            pickle.dump(creds, f)
    return creds


def fetch_google_sheet_pts(spreadsheet_id: str, range_name: str) -> pd.DataFrame:
    """
    Fetches values from a Google Sheet and returns a pandas DataFrame.
    Converts numeric columns automatically.
    """
    creds = get_pts_credentials()
    service = build('sheets', 'v4', credentials=creds)
    result = service.spreadsheets().values().get(
        spreadsheetId=spreadsheet_id,
        range=range_name,
        valueRenderOption='FORMATTED_VALUE'
    ).execute()
    rows = result.get('values', [])
    if not rows:
        raise ValueError(f"No data found in range {range_name}.")

    header, *data_rows = rows
    df = pd.DataFrame(data_rows, columns=header)
    # Attempt conversion of all columns to numeric where possible
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='ignore')
    return df

# ── Load Data ────────────────────────────────────────────────────────────────
df = fetch_google_sheet_pts(SPREADSHEET_ID, RANGE_NAME)

# X = n° de match
import plotly.graph_objects as go
import streamlit as st

# Données
x_values = list(range(1, len(df) + 1))
green      = "#00FF00"
dark_blue  = "#00008B"
light_blue = "#87CEFA"
orange     = "orange"

# Nuances de gris pour les anciennes saisons
num_hist = len(df.columns[:-2])
greys = [
    f'#{255 - i*15:02x}{255 - i*15:02x}{255 - i*15:02x}' 
    for i in range(num_hist)
]

# Création de la figure
fig = go.Figure()

# Anciennes saisons (gris ou vert)
for idx, col in enumerate(df.columns[:-2]):
    color = green if idx in [4,5,6,7,8] else greys[idx]
    fig.add_trace(go.Scatter(
        x=x_values,
        y=df[col],
        mode='markers',
        marker=dict(color=color, size=8, opacity=0.7),
        name=col,
        showlegend=False
    ))

# Saisons clés
fig.add_trace(go.Scatter(
    x=x_values, y=df['col12'],
    mode='markers',
    marker=dict(color=dark_blue, size=10, opacity=0.7),
    name='FCV 24/25'
))
fig.add_trace(go.Scatter(
    x=x_values, y=df['col13'],
    mode='markers',
    marker=dict(color=light_blue, size=10, opacity=0.7),
    name='FCV 23/24'
))
fig.add_trace(go.Scatter(
    x=x_values, y=df['Le Mans'],
    mode='markers',
    marker=dict(color=orange, size=10, opacity=0.7),
    name='Le Mans 24/25'
))
fig.add_trace(go.Scatter(
    x=x_values, y=df['col14'],
    mode='markers',
    marker=dict(color='black', size=10, opacity=0.9),
    name='FCV 25/26'
))

# Lignes-cibles
fig.add_shape(type="line", x0=1, x1=len(df), y0=34, y1=34,
              line=dict(color="black", width=1))
fig.add_shape(type="line", x0=1, x1=len(df), y0=44, y1=44,
              line=dict(color="black", width=1))

# Annotations
fig.add_annotation(x=2, y=35, text="Cible maintien : 34 pts",
                   showarrow=False,
                   font=dict(size=12, color="darkblue", family="Arial Black"))
fig.add_annotation(x=2, y=45, text="Cible Top 8 : 44 pts",
                   showarrow=False,
                   font=dict(size=12, color="darkblue", family="Arial Black"))

# Mise à jour du layout
fig.update_layout(
    width=1600,
    height=900,
    title=dict(
        text="Route vers le top 8 | 44 pts",
        x=0.02, xanchor="left",
        font=dict(size=20, color="darkblue", family="Arial Black")
    ),
    xaxis=dict(
        title=dict(text="Matchs",
                   font=dict(size=14, color="darkblue", family="Arial Black")),
        tickmode="array",
        tickvals=list(range(1, 33)),
        tickfont=dict(size=12,color="darkblue")
    ),
    yaxis=dict(
        title=dict(text="Points",
                   font=dict(size=14, color="darkblue", family="Arial Black")),
        tickmode="array",
        tickvals=list(range(0, 66, 5)),
        tickfont=dict(size=12,color="darkblue")
    ),
    legend=dict(
        orientation="v",
        x=1, y=0,
        xanchor="right", yanchor="bottom",
        bgcolor="rgba(0,0,0,0)"
    ),
    plot_bgcolor="white",
    margin=dict(l=50, r=50, t=80, b=50),
)

# Grille pointillée
# Grille en pointillés
fig.update_xaxes(
    showgrid=True,
    gridcolor="lightgrey",
    gridwidth=0.5,
    griddash="dash",   # <- pointillés
    zeroline=False
)
fig.update_yaxes(
    showgrid=True,
    gridcolor="lightgrey",
    gridwidth=0.5,
    griddash="dash",   # <- pointillés
    zeroline=False
)

# Affichage Streamlit
st.plotly_chart(fig, width=1600, height=900)




