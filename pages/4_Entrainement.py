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
from datetime import date

# ─────────────────────────── Config ───────────────────────────
st.set_page_config(page_title="Analyse | FC Versailles", layout='wide')

logo_url = 'https://raw.githubusercontent.com/FC-Versailles/care/main/logo.png'

# Google Sheets constants
SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
TOKEN_FILE_EN = 'token_ent.pickle'
SPREADSHEET_ID_EN  = '15n4XkQHrUpSPAW61vmR_Rk1kibd5zcmVqgHA40szlPg'   # Entrainement
SPREADSHEET_ID_EN2 = '1fY6624a0xdu7g8Hm59Qm9Jw0xglBRoH-fBErQtQoRfc'    # Daily
RANGE_NAME = 'Feuille 1'

# ─────────────────────── Auth & Fetch helpers ───────────────────────
def get_en_credentials():
    creds = None
    if os.path.exists(TOKEN_FILE_EN):
        with open(TOKEN_FILE_EN, 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('client_secret.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE_EN, 'wb') as token:
            pickle.dump(creds, token)
    return creds

def fetch_google_sheet(spreadsheet_id, range_name):
    def make_unique_columns(cols):
        seen = {}
        out = []
        for c in cols:
            key = c if c is not None else ""
            if key in seen:
                seen[key] += 1
                out.append(f"{key}__{seen[key]}")  # e.g. "Commentaire (a)__1"
            else:
                seen[key] = 0
                out.append(key)
        return out

    creds = get_en_credentials()
    service = build('sheets', 'v4', credentials=creds)
    result = service.spreadsheets().values().get(
        spreadsheetId=spreadsheet_id, range=range_name
    ).execute()

    values = result.get('values', [])
    if not values:
        return pd.DataFrame()

    header = values[0]
    rows = values[1:]
    max_cols = len(header)
    rows = [r + [None]*(max_cols - len(r)) if len(r) < max_cols else r[:max_cols] for r in rows]

    df = pd.DataFrame(rows, columns=header)
    df.columns = make_unique_columns(list(df.columns))  # <- ensure uniqueness
    return df

# ───────────────────────── Cache loaders ─────────────────────────
@st.cache_data(ttl=60)
def load_data_entrainement():
    df = fetch_google_sheet(SPREADSHEET_ID_EN, RANGE_NAME)
    if not df.empty and 'Type' in df.columns:
        df = df[~df['Type'].isin(['Salle', 'Dev Individuel'])]
    return df

@st.cache_data(ttl=60)
def load_data_daily():
    df = fetch_google_sheet(SPREADSHEET_ID_EN2, RANGE_NAME)

    # Colonnes à retirer
    cols_to_remove = [
        'Submission ID', 'Respondent ID', 'Submitted at',
        'Moment', 'Type'
    ]

    # Supprimer uniquement celles qui existent vraiment
    df = df.drop(columns=[c for c in cols_to_remove if c in df.columns], errors='ignore')

    return df

# ───────────────────────── UI Header ─────────────────────────
col1, col2 = st.columns([9, 1])
with col1:
    st.title("FC Versailles | Analyses")
with col2:
    st.image(logo_url, use_container_width=True)
st.markdown("<hr style='border:1px solid #ddd' />", unsafe_allow_html=True)

# ───────────────────────── Page selector (sidebar OK) ─────────────────────────
# Sidebar page selector (dropdown instead of radio)
page = st.sidebar.selectbox("Page", ["Analyse entrainement", "Analyse daily"], index=0)


# ───────────────────────── Common helpers ─────────────────────────
def coerce_date(df, colname="Date"):
    if colname in df.columns:
        df[colname] = pd.to_datetime(df[colname], errors='coerce')
    return df

def plot_treemap_from_activity_cols(filtered_df):
    activity_cols = [c for c in filtered_df.columns if str(c).startswith('Temps')]
    if not activity_cols:
        return
    melted = filtered_df.melt(id_vars=[], value_vars=activity_cols,
                              var_name='Temps', value_name='Activité').dropna()
    if 'Activité' not in melted.columns:
        return
    counts = melted[(melted['Activité'] != 'RAS') & (melted['Activité'] != 'Prévention')]['Activité'].value_counts()
    if counts.empty:
        return
    labels = [f"{lab}\n{val}" for lab, val in zip(counts.index, counts.values)]
    st.plotly_chart(go.Figure(go.Treemap(labels=labels, parents=['']*len(counts), values=counts.values)),
                    use_container_width=True)

def plot_stacked_by_type(filtered_df):
    activity_cols = [c for c in filtered_df.columns if str(c).startswith('Temps')]
    if not activity_cols or 'Type' not in filtered_df.columns:
        return
    melted = filtered_df.melt(id_vars=["Type"], value_vars=activity_cols,
                              var_name="Temps", value_name="Activité").dropna()
    melted = melted[melted['Activité'] != 'RAS']
    if melted.empty:
        return
    pivot = melted.groupby(['Type', 'Activité']).size().unstack(fill_value=0)
    fig = go.Figure()
    for col in pivot.columns:
        fig.add_trace(go.Bar(y=pivot.index, x=pivot[col], name=col, orientation='h'))
    fig.update_layout(barmode='stack', xaxis_title="Nombre d'apparitions", yaxis_title="Type",
                      legend_title_text='Activité', legend=dict(x=1.02, y=1))
    st.plotly_chart(fig, use_container_width=True)

def plot_scatter_date_content(filtered_df):
    if 'Date' not in filtered_df.columns:
        return
    activity_cols = [c for c in filtered_df.columns if str(c).startswith('Temps')]
    if not activity_cols:
        return
    melted = filtered_df.melt(id_vars=['Date'], value_vars=activity_cols,
                              var_name='Temps', value_name='Activité').dropna()
    melted = melted[melted['Activité'] != 'RAS']
    if melted.empty:
        return
    fig = px.scatter(melted, x='Date', y='Activité', color='Activité', size_max=10)
    fig.update_traces(marker=dict(symbol='square', size=10))
    st.plotly_chart(fig, use_container_width=True)

# ───────────────────────── Page: Analyse entrainement ─────────────────────────
def render_entrainement():
    # Bouton Questionnaire

    data = load_data_entrainement()
    if data.empty:
        st.error("Aucune donnée d'entraînement chargée.")
        return

    data = coerce_date(data, "Date")

    # ── Filtres DANS LA PAGE (pas dans la sidebar)
    with st.container():
        st.subheader("Filtres")
        c1, c2 = st.columns([2, 3])

        # Type(s) de séance
        if 'Type' in data.columns:
            with c1:
                all_types = sorted([t for t in data['Type'].dropna().unique().tolist()])
                session_types = st.multiselect(
                    "Type de séance",
                    options=all_types,
                    default=all_types
                )
        else:
            session_types = None

        # Période
        with c2:
            start_default = date(2025, 7, 1)
            if data['Date'].dropna().empty:
                st.warning("Pas de dates valides dans les données.")
                return
            end_default = data['Date'].max().date()
            date_range = st.date_input(
                "Période",
                [start_default, end_default],
                min_value=start_default,
                max_value=end_default
            )

    # Application des filtres
    filtered = data.copy()
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        filtered = filtered[(filtered['Date'] >= pd.to_datetime(date_range[0])) &
                            (filtered['Date'] <= pd.to_datetime(date_range[1]))]
    if session_types is not None:
        filtered = filtered[filtered['Type'].isin(session_types)]

    # Vue table + visuels
    st.markdown("### Données filtrées (Entraînement)")
    st.dataframe(filtered, use_container_width=True)

    st.markdown("### Répartition des activités")
    plot_treemap_from_activity_cols(filtered)

    st.markdown("### Répartition des procédés par type d'entraînement")
    plot_stacked_by_type(filtered)

    st.markdown("### Répartition des contenus par date")
    plot_scatter_date_content(filtered)

# ───────────────────────── Page: Analyse daily ─────────────────────────
def render_daily():
    
    st.markdown(
        '''
        <a href="https://tally.so/r/3ql17d" target="_blank" style="
            display: inline-block; padding: 10px 18px; background-color: #2563eb;
            color: white; text-decoration: none; border-radius: 8px; font-weight: 600; margin-bottom: 12px;">
            Questionnaire
        </a>
        ''', unsafe_allow_html=True
    )

    data2 = load_data_daily()
    if data2.empty:
        st.error("Aucune donnée 'daily' chargée.")
        return

    # Find candidate date columns (e.g., 'Date', 'Date__1', etc.)
    date_cols = [c for c in data2.columns if str(c).strip().lower().startswith('date')]
    if not date_cols:
        st.error("Aucune colonne 'Date' trouvée dans les données daily.")
        return

    # Coerce all candidate date columns to datetime, pick the one with most non-null values
    for c in date_cols:
        data2[c] = pd.to_datetime(data2[c], errors='coerce')

    primary_date_col = max(date_cols, key=lambda c: data2[c].notna().sum())
    # Alias chosen column to 'Date' for downstream consistency
    if primary_date_col != 'Date':
        data2['Date'] = data2[primary_date_col]

    valid_dates = data2['Date'].dropna()
    if valid_dates.empty:
        st.warning("Pas de dates valides dans les données daily.")
        return

    # ── UNIQUE FILTRE : Date unique
    st.subheader("Filtre date (Daily)")
    min_d = valid_dates.min().date()
    max_d = valid_dates.max().date()
    sel_date = st.date_input("Date", value=max_d, min_value=min_d, max_value=max_d)

    filtered = data2[data2['Date'].dt.date == sel_date].copy()
    if filtered.empty:
        st.info("Aucune ligne pour la date sélectionnée.")
        return

    st.markdown(f"### Données (Daily) — {sel_date.isoformat()}")
    st.dataframe(filtered, use_container_width=True)

    # Visuels optionnels si colonnes 'Temps*' présentes
    has_temps_cols = any(str(c).startswith('Temps') for c in filtered.columns)
    if has_temps_cols:
        st.markdown("### Répartition des activités (Daily)")
        plot_treemap_from_activity_cols(filtered)

        st.markdown("### Répartition par Type (Daily)")
        plot_stacked_by_type(filtered)

        st.markdown("### Contenus par date (Daily)")
        plot_scatter_date_content(filtered)


if page == "Analyse entrainement":
    render_entrainement()
else:
    render_daily()
