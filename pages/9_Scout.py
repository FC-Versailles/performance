#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 27 13:16:14 2025

@author: fcvmathieu
"""

import streamlit as st
import pandas as pd
import datetime
import os
import pickle
from io import BytesIO
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import plotly.graph_objects as go
from urllib.parse import quote
from statsbombpy import sb
import requests_cache
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
import plotly.express as px
import seaborn as sns
import base64
import uuid
import json
from pathlib import Path
import requests
from fpdf import FPDF
import tempfile
import base64
import unicodedata
from scipy.stats import zscore


def check_scout_login():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        st.title("FC Versailles | Accès Sécurisé")
        id_input = st.text_input("Identifiant :", type="password")
        if st.button("Connexion"):
            if id_input == "FCVscouting78!":
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Identifiant incorrect. Essayez encore.")
        st.stop()  # Stop rest of script if not logged in

check_scout_login()

st.set_page_config(layout='wide')

# Display the club logo from GitHub at the top right
logo_url = 'https://raw.githubusercontent.com/FC-Versailles/scouting/main/logo.png'
col1, col2 = st.columns([9, 1])
with col1:
    st.title("Recrutement | FC Versailles")
with col2:
    st.image(logo_url, use_container_width=True)
    
st.markdown("<hr style='border:1px solid #ddd' />", unsafe_allow_html=True)

# ---- Statsbomb ----


DEFAULT_CREDS = {
    "user": "mathieu.feigean@fcversailles.com",
    "passwd": "uVBxDK5X",
}

# Looking at all competitions to search for comp and season id
comp = sb.competitions(creds = DEFAULT_CREDS)

# Disable caching to avoid SQLite errors
sb.CACHE_ENABLED = False  
session = requests_cache.CachedSession(backend="memory")

df1 = sb.player_season_stats(competition_id=129, season_id=317,creds = DEFAULT_CREDS)
df2 = sb.player_season_stats(competition_id=1035, season_id=317,creds = DEFAULT_CREDS)
df3 = sb.player_season_stats(competition_id=8, season_id=317,creds = DEFAULT_CREDS)
df4 = sb.player_season_stats(competition_id=177, season_id=317,creds = DEFAULT_CREDS)
df5 = sb.player_season_stats(competition_id=63, season_id=317,creds = DEFAULT_CREDS)


data = pd.concat([df1, df2,df3,df4,df5], ignore_index=True)

cols_to_drop = [
    'account_id', 'player_id', 'team_id', 'competition_id', 'season_id', 
    'country_id', 'player_female', 'player_first_name', 'player_last_name', 'player_known_name'
]

existing_cols = [col for col in cols_to_drop if col in data.columns]

if missing := list(set(cols_to_drop) - set(existing_cols)):
    st.warning(f"Colonnes absentes ignorées : {missing}")

data = data.drop(columns=existing_cols)



# Remove the "player_season_" prefix from applicable column names
updated_columns = {col: col.replace("player_season_", "") for col in data.columns if col.startswith("player_season_")}
data.rename(columns=updated_columns, inplace=True)

data = data.dropna(axis=1, how='all')

data['birth_date1'] = pd.to_datetime(data['birth_date'], errors='coerce').dt.year
data['birth_date1'] = data['birth_date1'].astype(float).astype('Int64')

# Create age column
current_year = datetime.datetime.now().year  # Use full module reference
timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
data['age'] = current_year - data['birth_date1']
data = data.drop(columns=['birth_date1'])

column_order = (
    ['player_name', 'primary_position', 'secondary_position', 
     'team_name', 'competition_name','season_name', 
     'birth_date','age', 'player_weight', 
     'player_height', 'minutes', 'starting_appearances', 'appearances', 'average_minutes', 'most_recent_match', '90s_played'] + 
    [col for col in data.columns if col not in [
    'player_name', 'primary_position', 'secondary_position', 
    'team_name', 'competition_name','season_name',
    'birth_date','age', 'player_weight', 
    'player_height', 'minutes', 'starting_appearances', 'appearances', 'average_minutes', 'most_recent_match', '90s_played']]
)

data = data[column_order]

# ---- GOOGLE SHEETS CONFIGURATION ----
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
TOKEN_FILE_SCOUT = 'token_scout.pickle'
SPREADSHEET_ID_SCOUT = '1bqVJ5zSBJJsZe_PsH5lzspFKA6P0l3Mfc4jta00Jh9k'
DATABASE_RANGE_SCOUT = 'Feuille 1'

def get_scout_credentials():
    creds = None
    if os.path.exists(TOKEN_FILE_SCOUT):
        with open(TOKEN_FILE_SCOUT, 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'client_secret.json', SCOPES
            )
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE_SCOUT, 'wb') as token:
            pickle.dump(creds, token)
    return creds

def fetch_scout_google_sheet(spreadsheet_id, range_name):
    creds = get_scout_credentials()
    service = build('sheets', 'v4', credentials=creds)
    sheet = service.spreadsheets()
    result = sheet.values().get(spreadsheetId=spreadsheet_id, range=range_name).execute()
    values = result.get('values', [])
    if not values:
        st.error("No data found in the specified range.")
        return pd.DataFrame()
    header = values[0]
    data = values[1:]
    return pd.DataFrame(data, columns=header)

@st.cache_data(ttl=60)
def load_data():
    return fetch_scout_google_sheet(SPREADSHEET_ID_SCOUT, DATABASE_RANGE_SCOUT)

df = load_data()
df = df.loc[:, ~df.columns.duplicated()]

df['Date de naissance'] = pd.to_numeric(df['Date de naissance'], errors='coerce')
current_year = datetime.datetime.now().year
df['Age'] = current_year - df['Date de naissance']

params = st.query_params
default_page = params.get("page", "FCV Database")
PAGES = ["FCV Database", "Chercher Joueurs","Joueur à regarder","Statsbomb","Short List"]
page = st.sidebar.selectbox("Select Page", PAGES, index=PAGES.index(default_page))

#####################################################################################################################

if page == "FCV Database":
    st.markdown('<h2 style="color:#0031E3; margin-bottom: 20px;">📂 Scouting Database</h2>', unsafe_allow_html=True)


    # Create Filter Columns
    f1_col1, f1_col2, f1_col3, f1_col4, f1_col5 = st.columns(5)
    f2_col1, f2_col2, f2_col3, f2_col4, f2_col5 = st.columns(5)

    # 🔹 Position Filter
    positions = ["AILL", "ATT", "DC", "DD", "DG", "GB", "MC", "MO"]
    selected_position = f1_col1.multiselect("🔍 Le poste", options=positions, default=[])

    # 🔹 Championship Filter
    championships = df['Championnat'].dropna().unique().tolist()
    selected_championship = f1_col2.multiselect("🏆 Le championnat", options=championships, default=[])
    
    
    # 🔹 Footedness Filter
    pied_options = df['Pied'].dropna().unique().tolist()
    selected_pied = f1_col3.multiselect("🦶 Pied", options=pied_options, default=[])
    
    # 🔹 Contract End Filter
    contract_dates = df['Fin de contrat'].dropna().unique().tolist()
    selected_contract = f1_col4.multiselect("📅 Fin de contrat", options=contract_dates, default=[])
    

    # ✅ **Convert "Submitted at" to Datetime Format**
    df["Submitted at"] = pd.to_datetime(df["Submitted at"], errors="coerce").dt.tz_localize(None)

    # ✅ Ensure Submission Date is not empty
    if df["Submitted at"].notna().sum() > 0:
        min_date = df["Submitted at"].min()
        max_date = df["Submitted at"].max()
    else:
        min_date = datetime.datetime.now() - datetime.timedelta(days=30)
        max_date = datetime.datetime.now()


    # ✅ Submission Date Slider (Use datetime.date() to fix TypeError)
    selected_start_date, selected_end_date = f1_col5.slider(
        "📆 Date d'ajout",
        min_value=min_date.date(),
        max_value=max_date.date(),
        value=(min_date.date(), max_date.date()),
        format="MM-YY"
    )

    # ✅ Convert selected dates back to datetime for filtering
    selected_start_date = datetime.datetime.combine(selected_start_date, datetime.time.min)
    selected_end_date = datetime.datetime.combine(selected_end_date, datetime.time.max)


    # 🔹 Convert Birth Year
    df["Date de naissance"] = pd.to_numeric(df["Date de naissance"], errors="coerce")

    # 🔹 Ensure Valid Birth Years Exist
    if df["Date de naissance"].notna().sum() > 0:
        min_birth_year = int(df["Date de naissance"].min(skipna=True))
        max_birth_year = int(df["Date de naissance"].max(skipna=True))
    else:
        min_birth_year, max_birth_year = 1980, datetime.datetime.now().year

    # ✅ Birth Year Slider
    selected_birth_year = f2_col1.slider(
        "📅 Date de Naissance",
        min_birth_year,
        max_birth_year,
        (min_birth_year, max_birth_year)
    )

    # 🔹 Player Search Box
    search_query = f2_col5.text_input("🔎 Recherche joueur", "")
    
        # 🔹 Type de joueur Filter
    type_options = df['Type de joueur'].dropna().unique().tolist()
    selected_type = f2_col3.multiselect("🎯 Type de joueur", options=type_options, default=[])
    
    # 🔹 Potential Filter
    potential_options = df['Potential'].dropna().unique().tolist()
    selected_potential = f2_col4.multiselect("💎 Potential", options=potential_options, default=[])
    
        # 🔹 Profil Filter
    profil_options = [
    "Initiateur", "Agresseur", "Facilitateur", "Défensif", "Progresseur", "Overlapper", "Directeur",
    "Linebreaker", "Recuperateur", "Createur", "Catalyseur", "Box-to-box", "Explorateur",
    "Détonateur", "Libérateur", "Box Killer", "Mobile finisher", "Target man"]
    selected_profil = f2_col2.multiselect("🧬 Profil", options=profil_options, default=[])



    # 🔹 Apply Filters
    filtered_df = df.copy()
    if selected_position:
        filtered_df = filtered_df[
            filtered_df['Poste'].apply(
                lambda postes: any(pos in postes for pos in selected_position) if isinstance(postes, str) else False
            )
        ]

    if selected_championship:
        filtered_df = filtered_df[filtered_df['Championnat'].isin(selected_championship)]
    if selected_pied:
        filtered_df = filtered_df[filtered_df['Pied'].isin(selected_pied)]
    if selected_contract:
        filtered_df = filtered_df[filtered_df['Fin de contrat'].isin(selected_contract)]
    if search_query:
        filtered_df = filtered_df[filtered_df['Player'].str.contains(search_query, case=False, na=False)]
    if selected_type:
        filtered_df = filtered_df[filtered_df['Type de joueur'].isin(selected_type)]
    if selected_potential:
        filtered_df = filtered_df[filtered_df['Potential'].isin(selected_potential)]
    if selected_profil:
        filtered_df = filtered_df[
            filtered_df['Profil'].apply(
                lambda profils: any(p in profils for p in selected_profil) if isinstance(profils, str) else False
            )
        ]
    

    # ✅ **Filter by Submission Date Range**
    filtered_df = filtered_df[
        (filtered_df["Submitted at"] >= selected_start_date) & 
        (filtered_df["Submitted at"] <= selected_end_date)
    ]

    # ✅ **Filter by Birth Year**
    filtered_df = filtered_df[
        (filtered_df["Date de naissance"] >= selected_birth_year[0]) &
        (filtered_df["Date de naissance"] <= selected_birth_year[1])
    ]

    # ✅ **Filter by Player Name (Case Insensitive)**
    if search_query:
        filtered_df = filtered_df[filtered_df['Player'].str.contains(search_query, case=False, na=False)]
    
  
    # Sort by most recent submission
    filtered_df = filtered_df.sort_values(by="Submitted at", ascending=False)
    
    full_filtered_df = filtered_df.copy()
    # Columns to Display
    columns_to_display = ["Prénom", "Player", "Date de naissance", "Pied", "Taille", "Poste", "Championnat", "Club", "Fin de contrat", "Profil", "Type de joueur", "Potential"]
    filtered_df_for_display = full_filtered_df[columns_to_display]

    
        # Apply CSS for styling
    st.markdown("""
    <style>
        table {
            width: 100% !important;
            border-collapse: collapse !important;
        }
        th {
            background-color: #beb245 !important;
            text-align: center !important; /* Center align header text */
            padding: 10px !important;
        }
        td {
            text-align: center !important;
            vertical-align: top !important;
            white-space: pre-wrap !important;
            word-wrap: break-word !important;
            padding: 8px !important;
            border: 1px solid #ddd !important;
        }
    </style>
    """, unsafe_allow_html=True)

    
    # Display table without index
    st.write(filtered_df_for_display.head(500).to_html(index=False, escape=False), unsafe_allow_html=True)

    
    # Export to Excel
    def convert_df_to_excel(df):
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Scouting Data')
        return output.getvalue()
    
    excel_data = convert_df_to_excel(filtered_df)
    st.download_button(
        label="📂 Télécharger la liste",
        data=excel_data,
        file_name="FCV_Scouting_Data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    
    # Display Reports (détaillés pour les 5 premiers joueurs filtrés)
    st.subheader("📄 Liste des joueurs")
    
    top_players_df = full_filtered_df.head(10)
    for _, player_data in top_players_df.iterrows():
        player_name = player_data['Player']
        link = f"[📎 Voir la fiche complète](/?page=Chercher%20Joueurs&player={player_name})"
        encoded_name = quote(player_name)
        st.markdown(f"""
        <h3 style='color:#444;'>📋 Rapport pour {player_name}</h3>
        <a href='/?page=Chercher%20Joueurs&player={encoded_name}' style='text-decoration:none;'>
            <button style='background-color:#0043a4; color:white; padding:6px 12px; border:none; border-radius:5px;'>
                📎 Voir la fiche complète
            </button>
        </a>
        """, unsafe_allow_html=True)
    
        col1, col2 = st.columns(2)
    
        with col1:
            st.markdown(f"**Prénom :** {player_data.get('Prénom', '')}")
            st.markdown(f"**Âge :** {player_data.get('Age', '')}")
            st.markdown(f"**Taille :** {player_data.get('Taille', '')}")
            st.markdown(f"**Pied :** {player_data.get('Pied', '')}")
            st.markdown(f"**Poste :** {player_data.get('Poste', '')}")
            st.markdown(f"**Profil :** {player_data.get('Profil', '')}")
            st.markdown(f"**Type de joueur :** {player_data.get('Type de joueur', '')}")
    
        with col2:
            st.markdown(f"**Soumis le :** {player_data.get('Submitted at', '')}")
            st.markdown(f"**Date de naissance :** {player_data.get('Date de naissance', '')}")
            st.markdown(f"**Club :** {player_data.get('Club', '')}")
            st.markdown(f"**Championnat :** {player_data.get('Championnat', '')}")
            st.markdown(f"**Fin de contrat :** {player_data.get('Fin de contrat', '')}")
            st.markdown(f"**Transfermarkt :** {player_data.get('Transfermarkt', '')}")
            st.markdown(f"**Potential :** {player_data.get('Potential', '')}")
    
        rapport = str(player_data.get("Rapport", "")).strip()
        if rapport:
            st.subheader("📝 Commentaire du scout")
            st.markdown(f"<div style='white-space: pre-wrap;'>{rapport}</div>", unsafe_allow_html=True)
        else:
            st.warning("Aucun rapport disponible pour ce joueur.")
    

        st.markdown("<hr style='border:1px solid #ddd' />", unsafe_allow_html=True)
####################################################################################################################################################################################### 
#######################################################################################################################################################################################
 
if page == "Chercher Joueurs":
    st.markdown('<h2 style="color:#0031E3; margin-bottom: 20px;">\U0001F50E Chercher un Joueur</h2>', unsafe_allow_html=True)

    default_player = params.get("player", "")
    search_input = st.text_input("\U0001F50E Nom du joueur", value=default_player)

    if search_input:
        st.query_params.update({"page": "Chercher Joueurs", "player": search_input})
    else:
        st.query_params.update({"page": "Chercher Joueurs"})

    matched_players = df[df['Player'].str.contains(search_input, case=False, na=False)] if search_input else pd.DataFrame()

    if not matched_players.empty:
        for _, player_data in matched_players.iterrows():
            st.markdown(f"<h3 style='color:#444;'>\U0001F4CB Rapport pour {player_data['Player']}</h3>", unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                for field in ["Prénom", "Âge", "Taille", "Pied", "Poste", "Profil", "Type de joueur"]:
                    st.markdown(f"**{field} :** {player_data.get(field, '')}")
            with col2:
                for field in ["Submitted at", "Date de naissance", "Club", "Championnat", "Fin de contrat", "Transfermarkt", "Potential"]:
                    st.markdown(f"**{field} :** {player_data.get(field, '')}")

            rapport = str(player_data.get("Rapport", "")).strip()
            if rapport:
                st.subheader("\U0001F4DD Commentaire du scout")
                st.markdown(f"<div style='white-space: pre-wrap;'>{rapport}</div>", unsafe_allow_html=True)
            else:
                st.warning("Aucun rapport disponible pour ce joueur.")

            radar_sets = [
                ("\U0001F4CA Physical Skills", ["Physiquement fort", "Intensité des courses","Vitesse", "Volume des courses"], 'rgba(0, 48, 135, 0.7)'),
                ("\U0001F3AF Contribution au jeu", ["Conserver", "Progresser", "Créer du danger", "Contribuer"], 'rgba(255, 111, 0, 0.7)'),
                ("\U0001F6E1\ufe0f Défensive", ["Implication défensive", "Duels et interceptions", "Chasseur"], 'rgba(0, 135, 91, 0.7)')
            ]

            radar_col1, radar_col2, radar_col3 = st.columns(3)

            for radar_title, fields, color in radar_sets:
                if all(field in player_data and str(player_data[field]).strip() not in ["", "NA", "N/A"] for field in fields):
                    try:
                        values = [float(player_data[field]) for field in fields]
                        fig = go.Figure()
                        fig.add_trace(go.Scatterpolar(
                            r=values,
                            theta=fields,
                            fill='toself',
                            name=radar_title,
                            marker=dict(color=color)
                        ))
                        fig.update_layout(
                            polar=dict(
                                radialaxis=dict(
                                    visible=True,
                                    range=[0, 5],
                                    tickvals=[0, 1, 2, 3, 4, 5],
                                    ticktext=["0", "1", "2", "3", "4", "5"]
                                )
                            ),
                            showlegend=False,
                            title=radar_title
                        )
                        if radar_title == "\U0001F4CA Physical Skills":
                            radar_col1.plotly_chart(fig, use_container_width=True)
                        elif radar_title == "\U0001F3AF Contribution au jeu":
                            radar_col2.plotly_chart(fig, use_container_width=True)
                        else:
                            radar_col3.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.info("⚠️ Erreur dans la génération du radar.")
                        st.write(e)
                else:
                    st.info(f"⚠️ Données insuffisantes pour afficher le graphique : {radar_title}")

            st.markdown("---")

    elif search_input:
        st.info("Aucun joueur trouvé avec ce nom.")
        
####################################################################################################################################################################################### 
####################################################################################################################################################################################### 

if page == "Joueur à regarder":

    st.markdown('<h2 style="color:#0031E3; margin-bottom: 20px;">\U0001F50E Joueurs à regarder</h2>', unsafe_allow_html=True)


    # 🔧 Vérifie combien de joueurs n'ont pas de rapport
    if "Rapport" in df.columns:
        st.write("👀 Nombre de joueurs sans rapport :", df['Rapport'].isna().sum() + (df['Rapport'].str.strip() == "").sum())

        # Filtrer les joueurs pour lesquels le champ "Rapport" est vide
        watchlist_df = df[df['Rapport'].isna() | (df['Rapport'].str.strip() == "")]
     

        if watchlist_df.empty:
            st.success("🎉 Tous les joueurs ont un rapport !")
        else:
            # Colonnes à afficher pour la liste de joueurs sans rapport
            cols_to_show = [
                "Submitted at","Prénom", "Player", "Poste", 
                "Championnat", "Club", "Fin de contrat"
            ]

            st.markdown("Voici la liste des joueurs sans rapport de scout :")
            watchlist_df = watchlist_df.sort_values(by="Submitted at", ascending=False)
            st.dataframe(watchlist_df[cols_to_show], use_container_width=True)

            # Ajout d'un bouton pour télécharger la liste
            def convert_watchlist_to_excel(df):
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df.to_excel(writer, index=False, sheet_name='Watchlist')
                return output.getvalue()

    else:
        st.error("❌ La colonne 'Rapport' est introuvable dans le DataFrame. Vérifiez votre Google Sheet.")

####################################################################################################################################################################################### 
####################################################################################################################################################################################### 


# # --- GitHub Settings from Streamlit Secrets ---
# GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
# REPO_NAME = st.secrets["REPO_NAME"]
# BRANCH = st.secrets["BRANCH"]
# SHORTLIST_PATH = st.secrets["SHORTLIST_PATH"]

# GITHUB_API_URL = f"https://api.github.com/repos/{REPO_NAME}/contents/{SHORTLIST_PATH}"
# HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"}

# # --- Load shortlist from GitHub ---
# def load_shortlist():
#     response = requests.get(GITHUB_API_URL, headers=HEADERS)
#     if response.status_code == 200:
#         content = response.json()
#         decoded = base64.b64decode(content['content']).decode('utf-8')
#         return json.loads(decoded), content['sha']
#     else:
#         default = {pos: [] for pos in [
#             'GK', 'RB', 'RCB', 'LCB', 'LB',
#             'RCM', 'CM', 'LCM',
#             'RW', 'ST', 'LW']}
#         return default, None

# # --- Save shortlist to GitHub ---
# def save_shortlist(shortlist_data, sha=None):
#     content = json.dumps(shortlist_data, indent=2)
#     encoded = base64.b64encode(content.encode('utf-8')).decode('utf-8')
#     data = {
#         "message": "update shortlist",
#         "content": encoded,
#         "branch": BRANCH
#     }
#     if sha:
#         data["sha"] = sha
#         response = requests.put(GITHUB_API_URL, headers=HEADERS, json=data)
#     else:
#         response = requests.put(GITHUB_API_URL, headers=HEADERS, json=data)
#     return response.status_code in [200, 201]

# shortlist_data, shortlist_sha = load_shortlist()

# if page == "Short List":
#     st.markdown('<h2 style="color:#0031E3; margin-bottom: 20px;"> Short List - Shadow Team (1-4-3-3)</h2>', unsafe_allow_html=True)

#     available_players = df['Player'].dropna().unique().tolist()

#     def render_position_select(position):
#         st.markdown(f"### {position}")

#         current_list = shortlist_data.get(position, [])
#         if not isinstance(current_list, list):
#             current_list = []

#         selected = st.multiselect(
#             f"Ajouter des joueurs à {position} :",
#             [p for p in available_players if p not in current_list],
#             key=f"multi_{position}"
#         )

#         updated = False
#         for player in selected:
#             if player not in current_list and len(current_list) < 5:
#                 current_list.append(player)
#                 updated = True

#         if updated:
#             shortlist_data[position] = current_list
#             save_shortlist(shortlist_data, shortlist_sha)

#         if current_list:
#             st.markdown("**Joueurs sélectionnés :**")
#             for player in current_list:
#                 col1, col2 = st.columns([4, 1])
#                 with col1:
#                     st.write(player)
#                 with col2:
#                     if st.button(f"❌", key=f"remove_{position}_{player}"):
#                         current_list.remove(player)
#                         shortlist_data[position] = current_list
#                         save_shortlist(shortlist_data, shortlist_sha)

#     with st.container():
#         st.markdown("#### 🧤 Défense")
#         def1, def2, def3, def4, def5 = st.columns(5)
#         with def1: render_position_select("GK")
#         with def2: render_position_select("RB")
#         with def3: render_position_select("RCB")
#         with def4: render_position_select("LCB")
#         with def5: render_position_select("LB")

#     with st.container():
#         st.markdown("#### 🎯 Milieu")
#         mid1, mid2, mid3 = st.columns(3)
#         with mid1: render_position_select("RCM")
#         with mid2: render_position_select("CM")
#         with mid3: render_position_select("LCM")

#     with st.container():
#         st.markdown("#### 🔥 Attaque")
#         att1, att2, att3 = st.columns(3)
#         with att1: render_position_select("RW")
#         with att2: render_position_select("ST")
#         with att3: render_position_select("LW")

#     if st.button("🗑️ Réinitialiser toute la Shortlist"):
#         shortlist_data = {pos: [] for pos in shortlist_data}
#         save_shortlist(shortlist_data, shortlist_sha)
#         st.success("Shortlist réinitialisée.")

#     # Vue terrain 1-4-3-3
#     st.markdown("---")
#     st.subheader("📷 Vue terrain 1-4-3-3")

#     fig, ax = plt.subplots(figsize=(10, 7))
#     ax.set_xlim(0, 100)
#     ax.set_ylim(0, 100)
#     ax.axis('off')

#     position_coords = {
#         "GK": (50, 10),
#         "RB": (80, 25), "RCB": (65, 30), "LCB": (35, 30), "LB": (20, 25),
#         "RCM": (70, 50), "CM": (50, 55), "LCM": (30, 50),
#         "RW": (80, 75), "ST": (50, 80), "LW": (20, 75)
#     }

#     for pos, (x, y) in position_coords.items():
#         players = shortlist_data.get(pos, [])
#         if isinstance(players, list) and players:
#             label = f"{pos}\n" + "\n".join(players[:5])
#         else:
#             label = pos
#         ax.text(x, y, label, ha='center', va='center', fontsize=9,
#                 bbox=dict(facecolor='#0031E3', alpha=0.7, boxstyle='round,pad=0.5'), color='white')

#     st.pyplot(fig)


    
####################################################################################################################################################################################### 
####################################################################################################################################################################################### 


elif page == "Statsbomb":
    st.title("Statsbomb")

    def plot_scatter(df, players, x_col, y_col, x_label, y_label, title):
        x_median = df[x_col].median()
        y_median = df[y_col].median()
    
        # Séparer joueurs sélectionnés et non sélectionnés
        df_highlight = df[df["Name"].isin(players)]
        df_normal = df[~df["Name"].isin(players)]
    
        fig = go.Figure()
    
        # 🔵 Points normaux (par compétition)
        for comp in df_normal["competition_name"].unique():
            df_comp = df_normal[df_normal["competition_name"] == comp]
            fig.add_trace(go.Scatter(
                x=df_comp[x_col],
                y=df_comp[y_col],
                mode='markers',
                name=comp,
                marker=dict(size=8, opacity=0.7),
                text=df_comp["Name"],
                hovertemplate=(
                    f"<b>%{{text}}</b><br>{x_label}: %{{x:.2f}}<br>{y_label}: %{{y:.2f}}<extra></extra>"
                )
            ))
    
        # ⚫ Joueurs sélectionnés
        fig.add_trace(go.Scatter(
            x=df_highlight[x_col],
            y=df_highlight[y_col],
            mode='markers+text',
            name="Joueurs sélectionnés",
            marker=dict(size=12, color="black", line=dict(width=1, color="white")),
            text=df_highlight["Name"].apply(lambda x: x.split()[-1]),  # affiche juste le nom
            textposition="top center",
            hovertemplate=(
                f"<b>%{{text}}</b><br>{x_label}: %{{x:.2f}}<br>{y_label}: %{{y:.2f}}<extra></extra>"
            )
        ))
    
        # ➕ Lignes médianes
        fig.add_vline(
            x=x_median,
            line=dict(color="gray", dash="dash", width=1),
            annotation_text=f"Médiane {x_label}: {x_median:.2f}",
            annotation_position="top left"
        )
        fig.add_hline(
            y=y_median,
            line=dict(color="gray", dash="dash", width=1),
            annotation_text=f"Médiane {y_label}: {y_median:.2f}",
            annotation_position="bottom right"
        )
    
        # Mise en page
        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title=y_label,
            legend_title="Compétition",
            height=700,
            template="simple_white"
        )
    
        return fig
    
    # Dictionnaire des compétences
    competences_dict = {
        "Création des occasions": ("np_xg_90", "xa_90"),
        "Qualité de Dribble": ("dribbles_90", "dribble_ratio"),
        "Etat de confiance": ("npxgxa_90", "over_under_performance_90"),
        "Qualité de tirs": ("obv_shot_90", "np_psxg_90"),
        "Faire progresseur le jeu vers l'avant": ("carry_length", "deep_progressions_90"),
        "Agir le plus proche possible du but": ("op_passes_into_and_touches_inside_box_90", "deep_completions_90"),
        "Capacité à conserver le ballon": ("change_in_passing_ratio", "turnovers_90"),
        "Création de danger": ("obv_dribble_carry_90", "obv_pass_90"),
        "Intensité sans ballon": ("average_x_pressure", "counterpressures_90"),
        "Pressing": ("padj_pressures_90", "pressure_regains_90"),
        "Récupérer des ballons": ("ball_recoveries_90", "padj_interceptions_90"),
        "Duels Aérien": ("aerial_wins_90", "aerial_ratio")
    }

    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        competition = st.multiselect("Compétition", data["competition_name"].dropna().unique())
    
    with col2:
        ordered_positions = ['Goalkeeper','Centre Back','Right Centre Back','Left Centre Back','Right Back','Left Back','Left Wing Back','Right Wing Back', 'Centre Defensive Midfielder',
                             'Right Defensive Midfielder','Left Defensive Midfielder','Right Centre Midfielder','Left Centre Midfielder', 'Right Midfielder','Left Midfielder',
                             'Centre Attacking Midfielder','Right Attacking Midfielder','Left Attacking Midfielder','Left Wing','Right Wing', 
                             'Centre Forward','Right Centre Forward','Left Centre Forward']

        available_positions = [pos for pos in ordered_positions if pos in data["primary_position"].unique()]
        position = st.multiselect("Position principale", available_positions)

    with col3:
        age = st.slider(
            "Âge",
            min_value=int(data["age"].min()),
            max_value=int(data["age"].max()),
            value=(int(data["age"].min()), int(data["age"].max()))
        )
    
    with col4:
        minutes = st.slider(  # ⚠️ ici on appelle la variable "minutes" (pas min_minutes)
            "Minutes",
            min_value=int(data["minutes"].min()),
            max_value=int(data["minutes"].max()),
            value=(int(data["minutes"].min()), int(data["minutes"].max()))
        )
    

    filtered_data = data.copy()
    if competition:
        filtered_data = filtered_data[filtered_data["competition_name"].isin(competition)]
    if position:
        filtered_data = filtered_data[filtered_data["primary_position"].isin(position)]

    filtered_data = filtered_data[
        (filtered_data["age"] >= age[0]) &
        (filtered_data["age"] <= age[1]) &
        (filtered_data["minutes"] >= minutes[0]) &
        (filtered_data["minutes"] <= minutes[1])
    ]
    
       # Liste des joueurs disponibles après filtre
    available_players = filtered_data["player_name"].unique()
    
    # Session state pour conserver la sélection des joueurs
    if "player_selection" not in st.session_state:
        st.session_state.player_selection = []
    
    # Liste fusionnée : joueurs disponibles + ceux déjà sélectionnés
    merged_players = list(set(available_players).union(set(st.session_state.player_selection)))
    
    # Multiselect avec persistance
    player_selection = st.multiselect(
        "Choisis les joueurs à mettre en valeur",
        options=merged_players,
        default=st.session_state.player_selection
    )
    
    # Mise à jour de la session
    st.session_state.player_selection = player_selection


    selected_competence = st.selectbox("Choisis une compétence à analyser", list(competences_dict.keys()))
    x_col, y_col = competences_dict[selected_competence]


    if not filtered_data.empty:
        df_for_plot = filtered_data.rename(columns={"player_name": "Name"})  # plus besoin de Season
        fig = plot_scatter(df_for_plot, player_selection, x_col, y_col, x_col, y_col, selected_competence)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Aucune donnée pour les filtres sélectionnés.")
        
        
    chosen_variables = [
        "counterpressures_90", "aggressive_actions_90",
        "aerial_wins_90", "aerial_ratio",
        "change_in_passing_ratio", "turnovers_90",
        "padj_tackles_90", "padj_interceptions_90",
        "blocks_per_shot", "padj_clearances_90",
        "passing_ratio", "dispossessions_90",
        "carries_90", "carry_length",
        "forward_pass_proportion", "obv_pass_90",
        "dribbled_past_90", "dribble_faced_ratio",
        "crosses_90", "xa_90",
        "shot_touch_ratio", "touches_inside_box_90",
        "np_psxg_90", "obv_shot_90",
        "total_dribbles_90", "obv_dribble_carry_90",
        "defensive_action_regains_90", "pressure_regains_90",
        "np_xg_90", "np_xg_per_shot","xs_ratio","sp_xa_90","sp_key_passes_90","gsaa_90","obv_defensive_action_90","ball_recoveries_90","deep_progressions_90","op_f3_passes_90"
    ]
    
    column_mappings = {
        "Agresseur": ["counterpressures_90", "aggressive_actions_90"],
        "Header": [ "aerial_wins_90", "aerial_ratio"],
        "Technicien": ["change_in_passing_ratio", "turnovers_90"],
        "Defender": ["padj_tackles_90", "padj_interceptions_90"],
        "Protecteur": ["blocks_per_shot", "padj_clearances_90"],
        "Annihilateur": ["obv_defensive_action_90", "ball_recoveries_90"],
        "Conserver": ["passing_ratio", "dispossessions_90"],
        "Progress": ["carries_90", "carry_length"],
        "Pass": ["forward_pass_proportion", "obv_pass_90"],
        "1v1": ["dribbled_past_90", "dribble_faced_ratio"], 
        "Assist": ["crosses_90", "xa_90"],     
        "Box": ["shot_touch_ratio", "touches_inside_box_90"],
        "Tireur": ["np_psxg_90", "obv_shot_90"],
        "Percuteur": ["total_dribbles_90", "obv_dribble_carry_90"],
        "Recuperateur": ["defensive_action_regains_90", "pressure_regains_90"],
        "Striker": ["np_xg_90", "np_xg_per_shot"],
        "GK": ["xs_ratio", "gsaa_90"],
        "Set Pieces": ["sp_xa_90","sp_key_passes_90"],
        "Last third": ["deep_progressions_90","op_f3_passes_90"]
    }
    
     
    
    # 🛑 Variables pour lesquelles un score élevé est négatif
    negatively_correlated = [
        "turnovers_90", "dispossessions_90", "dribbled_past_90"
    ]
    
    zscore_df = filtered_data[['player_name', 'primary_position']].copy()
    for var in chosen_variables:
        if var in filtered_data.columns:
            values = filtered_data[var]
            if var in negatively_correlated:
                zscore_df[var] = -zscore(values, nan_policy="omit")
            else:
                zscore_df[var] = zscore(values, nan_policy="omit")
        else:
            st.warning(f"⚠️ Colonne introuvable : {var}")
        
    # 🧮 Création des profils
    aggregated_df = zscore_df[['player_name', 'primary_position']].copy()
    for profile, cols in column_mappings.items():
        if cols[0] in zscore_df.columns and cols[1] in zscore_df.columns:
            aggregated_df[profile] = zscore_df[cols[0]] + zscore_df[cols[1]]
    
    aggregated_df["Total Score"] = aggregated_df[list(column_mappings.keys())].sum(axis=1)
    
    # ✅ Renommer pour affichage
    aggregated_df = aggregated_df.rename(columns={
        "player_name": "Name",
        "primary_position": "Position"
    })
    
    # ➕ Ajout de la compétition pour styliser
    aggregated_df = aggregated_df.merge(
        filtered_data[["player_name", "competition_name"]],
        how="left",
        left_on="Name",
        right_on="player_name"
    )
    aggregated_df.drop(columns=["player_name"], inplace=True)
    
        # 📋 Choix des profils à afficher
        # 📋 Choix des profils à afficher
    selected_profiles = st.multiselect(
        "Selectionner compétences:",
        options=list(column_mappings.keys()),
        default=["Agresseur", "Defender", "Striker"]
    )

    if selected_profiles:
        # Coefficients personnalisés pour chaque profil sélectionné
        st.markdown("##### Pondération des profils sélectionnés")
        profile_weights = {}
        cols = st.columns(5)
        for idx, profile in enumerate(selected_profiles):
            with cols[idx % 5]:
                profile_weights[profile] = st.number_input(
                    f"{profile} :", min_value=0.0, max_value=10.0, value=1.0, step=0.1
                )
    
        # 💡 Calcul du score pondéré
        aggregated_df["Profile Score"] = sum(
            aggregated_df[profile] * weight for profile, weight in profile_weights.items()
        )
    
        # 📄 Colonnes dans l’ordre souhaité
        columns_to_display = (
            ['Name', 'Position'] +
            selected_profiles +
            ['Profile Score', 'Total Score']
        )
    
        # Création de la table finale
        table_df = (
            aggregated_df[columns_to_display]
            .drop_duplicates(subset=["Name"])
            .sort_values(by='Profile Score', ascending=False)
            .head(30)
        )
    
        # 🎨 Stylisation
        def highlight_name(val, comp):
            color_map = {
                c: plt.cm.tab10(i / 10) for i, c in enumerate(aggregated_df["competition_name"].dropna().unique())
            }
            rgba = color_map.get(comp, (1, 1, 1, 1))  # blanc par défaut
            r, g, b = [int(x * 255) for x in rgba[:3]]
            return f"color: rgb({r},{g},{b})"
    
        def style_table(df, full_df):
            styled = df.style
            cmap_cols = selected_profiles + ["Profile Score", "Total Score"]
            styled = styled.background_gradient(cmap="YlGnBu", subset=cmap_cols)
            styled = styled.apply(
                lambda row: [
                    highlight_name(row['Name'], full_df.loc[full_df['Name'] == row['Name'], 'competition_name'].values[0])
                    if col == 'Name' else ''
                    for col in df.columns
                ],
                axis=1
            )
            return styled
    
        st.subheader("Liste des joueurs")
        st.dataframe(style_table(table_df, aggregated_df), use_container_width=True)
    
        # 🔧 Nettoie les caractères spéciaux pour PDF
        def clean_text(text):
            if not isinstance(text, str):
                text = str(text)
            return unicodedata.normalize('NFKD', text).encode('latin-1', 'ignore').decode('latin-1')
    
        # 📄 Classe PDF simple avec Arial
        class PDF(FPDF):
            def header(self):
                self.set_font("Arial", "B", 12)
                self.cell(0, 10, "Player Profile Table", 0, 1, "C")
                self.ln(5)
    
        pdf = PDF()
        pdf.add_page()
        pdf.set_font("Arial", "", 8)
    
        # 🧱 Dimensions
        col_width = 25
        row_height = 6
    
        # En-têtes
        for col in table_df.columns:
            pdf.cell(col_width, row_height, clean_text(col), border=1)
        pdf.ln(row_height)
    
        # Lignes du tableau
        for _, row in table_df.iterrows():
            for item in row:
                text = clean_text(round(item, 2)) if isinstance(item, float) else clean_text(item)
                pdf.cell(col_width, row_height, text, border=1)
            pdf.ln(row_height)
    
        # 📤 Export en PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
            pdf.output(tmpfile.name)
            with open(tmpfile.name, "rb") as f:
                base64_pdf = base64.b64encode(f.read()).decode("utf-8")
            download_link = f'<a href="data:application/pdf;base64,{base64_pdf}" download="player_table.pdf">📄 Télécharger la table en PDF</a>'
            st.markdown(download_link, unsafe_allow_html=True)
    
    else:
        st.info("Sélectionne au moins un profil pour afficher la table et générer le PDF.")
    
    with st.expander("📘 Glossaire des profils (clique pour afficher)"):
        st.markdown("""
        <style>
        .profil-label {
            font-weight: bold;
            color: #2c3e50;
        }
        .profil-desc {
            margin-bottom: 10px;
        }
        </style>
    
        <div class='profil-desc'><span class='profil-label'>🧨 Agresseur :</span> Capacité à effectuer des contre-pressings et des actions agressives immédiatement après la perte du ballon.</div>
        <div class='profil-desc'><span class='profil-label'>🪖 Header :</span> Capacité à remporter les duels aériens et à être efficace dans les airs.</div>
        <div class='profil-desc'><span class='profil-label'>🎩 Technicien :</span> Capacité à limiter les pertes de balle et à améliorer la précision des passes.</div>
        <div class='profil-desc'><span class='profil-label'>🛡 Defender :</span> Aptitude à réaliser des tacles et des interceptions efficaces.</div>
        <div class='profil-desc'><span class='profil-label'>🧱 Protecteur :</span> Capacité à bloquer les tirs et à dégager proprement la zone défensive.</div>
        <div class='profil-desc'><span class='profil-label'>🚫 Annihilateur :</span> Capacité à détruire les actions adverses et récupérer des ballons clés.</div>
        <div class='profil-desc'><span class='profil-label'>🔒 Conserver :</span> Capacité à conserver la possession, éviter les pertes inutiles.</div>
        <div class='profil-desc'><span class='profil-label'>➡️ Progress :</span> Capacité à faire avancer le jeu via des courses ou transmissions verticales.</div>
        <div class='profil-desc'><span class='profil-label'>🎯 Pass :</span> Capacité à effectuer des passes vers l’avant créatrices de valeur.</div>
        <div class='profil-desc'><span class='profil-label'>⚔️ 1v1 :</span> Capacité à résister aux dribbles adverses et à défendre en un contre un.</div>
        <div class='profil-desc'><span class='profil-label'>🅰️ Assist :</span> Capacité à créer des occasions via des centres et des passes décisives.</div>
        <div class='profil-desc'><span class='profil-label'>📦 Box :</span> Présence et efficacité dans la surface adverse.</div>
        <div class='profil-desc'><span class='profil-label'>🎯 Tireur :</span> Qualité des tirs, dangerosité générée.</div>
        <div class='profil-desc'><span class='profil-label'>🏃 Percuteur :</span> Capacité à dribbler et porter le ballon pour déséquilibrer.</div>
        <div class='profil-desc'><span class='profil-label'>♻️ Recuperateur :</span> Capacité à récupérer la possession suite à des actions défensives.</div>
        <div class='profil-desc'><span class='profil-label'>🎯 Striker :</span> Efficacité devant le but, qualité des occasions et des tirs.</div>
        <div class='profil-desc'><span class='profil-label'>🧤 GK :</span> Efficacité du gardien à stopper les tirs et surpasser les attentes.</div>
        <div class='profil-desc'><span class='profil-label'>🎯 Set Pieces :</span> Qualité des passes sur coups de pied arrêtés et création d’occasions.</div>
        """, unsafe_allow_html=True)


st.markdown("""
    <style>
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: #f8f9fa;
            padding: 10px;
            text-align: center;
            font-size: 14px;
            color: #333;
        }
    </style>
    <div class="footer">
        <p><strong>M.Feigean</strong> - Football Development</p>
    </div>
    """, unsafe_allow_html=True)
