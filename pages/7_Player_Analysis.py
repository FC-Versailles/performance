#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 21:11:53 2025

@author: fcvmathieu
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import pandas as pd
import numpy as np
import streamlit as st

from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from statsbombpy import sb

import requests
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go

# ---------------- UI ----------------
st.set_page_config(layout="wide")

# ---------------- Config ----------------
DEFAULT_CREDS = {
    "user": "mathieu.feigean@fcversailles.com",
    "passwd": "uVBxDK5X",
}
COMPETITION_ID = 129
SEASON_ID = 318
FOCUS_TEAM = "Versailles"

SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
TOKEN_FILE = 'token_player.pickle'

# Unique Google Sheet (toutes infos joueur)
SPREADSHEET_ID = '1Q_asy-W9DMpJ0KspQEYrqtsVJBhviNhiYr6k2jaretI'
RANGE_NAME = 'Feuille 1'

# ---------- En-tête ----------
col1, col2 = st.columns([9, 1])
with col1:
    st.title("⚽ Performance Individuelle | FC Versailles")
with col2:
    st.image(
        "https://raw.githubusercontent.com/FC-Versailles/wellness/main/logo.png",
        use_container_width=True,
    )
st.markdown("<hr style='border:1px solid #ddd' />", unsafe_allow_html=True)

# ---------------- Helpers ----------------
def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out

def _fmt(value, default="—"):
    return default if value is None or str(value).strip() == "" else value

def _coerce_id(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")

def _pick_id_and_name(cols: pd.Index):
    s = set([str(c).strip() for c in cols])
    id_col = next((c for c in ["player_id", "statsbomb_id", "playerId", "id"] if c in s), None)
    name_col = next((c for c in ["player_name", "player"] if c in s), None)
    return id_col, name_col

def _safe_select(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    keep = [c for c in cols if c in df.columns]
    return df[keep] if keep else df

# ---------------- Google Sheet (OAuth) ----------------
def get_credentials():
    creds = None
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('client_secret.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, 'wb') as token:
            pickle.dump(creds, token)
    return creds

def fetch_sheet(spreadsheet_id: str, range_name: str) -> pd.DataFrame:
    creds = get_credentials()
    service = build('sheets', 'v4', credentials=creds)
    resp = service.spreadsheets().values().get(
        spreadsheetId=spreadsheet_id,
        range=range_name
    ).execute()
    values = resp.get('values', [])
    if not values:
        return pd.DataFrame()
    header = [str(h).strip() for h in values[0]]
    rows = values[1:]
    max_cols = len(header)
    adjusted = [
        row + [None] * (max_cols - len(row)) if len(row) < max_cols else row[:max_cols]
        for row in rows
    ]
    return pd.DataFrame(adjusted, columns=header)

@st.cache_data(ttl=60)
def load_sheet() -> pd.DataFrame:
    return _norm_cols(fetch_sheet(SPREADSHEET_ID, RANGE_NAME))

# ---------------- StatsBomb ----------------
@st.cache_data(show_spinner="Chargement StatsBomb…")
def load_player_season_stats(competition_id: int, season_id: int, creds: dict) -> pd.DataFrame:
    df = sb.player_season_stats(
        competition_id=competition_id,
        season_id=season_id,
        creds=creds
    )
    return _norm_cols(df)

# ---------------- Data ingest & harmonisation ----------------
# 1) StatsBomb : toute la compétition
player_season_all = load_player_season_stats(COMPETITION_ID, SEASON_ID, DEFAULT_CREDS)
player_season_all = _norm_cols(player_season_all)

# Vérification colonne équipe
if "team_name" not in player_season_all.columns:
    st.error(f"'team_name' manquant dans StatsBomb. Colonnes: {list(player_season_all.columns)}")
    st.stop()

# Sous-ensemble Versailles pour l’app (mais on garde player_season_all si besoin plus tard)
player_season = (
    player_season_all[
        player_season_all["team_name"].astype(str).str.strip().eq(FOCUS_TEAM)
    ]
    .reset_index(drop=True)
)

# 2) Google Sheet
df_sheet = load_sheet()
df_sheet = _norm_cols(df_sheet)

# 3) Harmonisation clés StatsBomb (Versailles)
id_col_sb, name_col_sb = _pick_id_and_name(player_season.columns)
if id_col_sb is None or name_col_sb is None:
    st.error(f"StatsBomb doit exposer un id et un nom. Colonnes: {list(player_season.columns)}")
    st.stop()

sb_players = (
    player_season
    .rename(columns={id_col_sb: "statsbomb_id", name_col_sb: "sb_player_name"})
    .assign(statsbomb_id=lambda x: _coerce_id(x["statsbomb_id"]))
)

# 4) Harmonisation clés feuille Google
sheet_id_col = next(
    (c for c in df_sheet.columns if str(c).strip().lower() in {"statsbomb_id", "player_id", "sb_id"}),
    None
)
sheet_name_col = next(
    (c for c in df_sheet.columns if str(c).strip().lower() in {"player_name", "name", "player"}),
    None
)

if sheet_id_col is None:
    df_sheet["statsbomb_id"] = pd.NA
else:
    if sheet_id_col != "statsbomb_id":
        df_sheet = df_sheet.rename(columns={sheet_id_col: "statsbomb_id"})
    df_sheet["statsbomb_id"] = _coerce_id(df_sheet["statsbomb_id"])

if sheet_name_col and sheet_name_col != "player_name":
    df_sheet = df_sheet.rename(columns={sheet_name_col: "player_name"})

# ---------------- Index joueurs (union SB + Sheet) ----------------
players_index = (
    sb_players[["statsbomb_id", "sb_player_name"]]
    .dropna(subset=["statsbomb_id"])
    .drop_duplicates()
    .merge(
        _safe_select(df_sheet, ["statsbomb_id", "player_name"]),
        on="statsbomb_id",
        how="outer"
    )
)

players_index["display_name"] = players_index["player_name"].fillna(players_index["sb_player_name"])
players_index = (
    players_index
    .dropna(subset=["statsbomb_id"])
    .sort_values("display_name", na_position="last")
)

# ---------------- Sélecteur ----------------
st.sidebar.header("Sélection joueur")
if players_index.empty:
    st.sidebar.info("Aucun joueur détecté. Vérifiez la présence de 'statsbomb_id'.")
    st.stop()

player_options = [
    (int(pid), str(name) if pd.notna(name) else f"ID {int(pid)}")
    for pid, name in zip(players_index["statsbomb_id"], players_index["display_name"])
]
selected_name = st.sidebar.selectbox(
    "Joueur",
    options=[name for _, name in player_options],
    index=0
)
name_to_id = {name: pid for pid, name in player_options}
selected_id = name_to_id.get(selected_name)

# ---------------- Slices ----------------
sb_slice = sb_players.loc[sb_players["statsbomb_id"] == selected_id].copy()
sheet_slice = (
    df_sheet.loc[df_sheet.get("statsbomb_id").eq(selected_id)]
    if "statsbomb_id" in df_sheet.columns
    else pd.DataFrame()
)

# Une seule ligne de référence (si doublons)
sheet_row = (
    sheet_slice.sort_index(ascending=False).head(1).to_dict(orient="records")
)
sheet_row = sheet_row[0] if sheet_row else {}


# ---------------- Aliases de champs (Google Sheet) ----------------
ALIAS = {
    # Identité
    "numero": ["Numéro", "numero", "number", "dorsal", "shirt_number"],
    "photo": ["photo", "Photo", "img_url", "image", "avatar"],
    "age": ["Age", "age"],
    "nationalite": ["Nationalité", "nationalite", "nationality"],
    "prenom": ["Prénom", "prenom", "first_name", "forename"],

    # Information contractuelle
    "debut_contrat": ["Debut Contrat", "Début Contrat", "debut_contrat", "contract_start"],
    "fin_contrat": ["Fin de contrat", "Fin Contrat", "fin_contrat", "contract_end"],
    "type": ["Type", "type", "player_type", "contract_type"],

    # Caractéristique du joueur
    "poste": ["Poste", "poste", "position", "role"],
    "pied": ["Pied", "pied", "foot", "strong_foot", "preferred_foot"],
    "profil": ["Profil", "profil", "player_profile"],
    "joueur_reference": ["Joueur Référence", "Joueur Reference", "joueur_reference", "reference_player"],
    "performance": ["Performance", "performance"],
    "development": ["Development", "Développement", "development", "developpement"],

    # Auto-évaluation
    "objectifs": ["Objectifs", "objectifs", "goals", "targets"],
    "plan_reussite_personnel": ["Préparation Physique a-Plan de réussite Personnel", "Plan réussite Personnel", "plan_reussite_personnel"],
    "plan_reussite_tactique": ["b-Plan de réussite Tactique", "Plan réussite Tactique", "plan_reussite_tactique"],
    "plan_reussite_technique": ["c-Plan de réussite Technique", "Plan réussite Technique", "plan_reussite_technique"],
    "plan_reussite_mental": ["d-Plan de réussite Mental", "Plan réussite Mental", "plan_reussite_mental"],
    "plan_reussite_groupe": ["e-Plan de réussite Groupe", "Plan réussite Groupe", "plan_reussite_groupe"],

    # Développement du joueur
    "theme_progression": ["Thème de progression", "theme_progression", "areas_to_improve", "weaknesses"],
    "plan_developpement": ["Plan de Developpement", "Plan de Développement", "plan_developpement", "development_plan"],

    # Profil Athlétique
    "taille": ["Taille", "taille", "height", "Ht", "Height"],
    "poids": ["Poids", "poids", "weight", "Wt", "Weight"],
    "mg_percent": ["MG%", "mg%", "MG_percent", "MG_Pourcentage", "mg_percent", "body_fat"],
    "distance": ["Distance", "distance", "distance_max", "max_distance"],
    "vmax": ["Vmax", "vmax", "max_speed", "vitesse_max"],
    "hi": ["HI", "hi", "high_intensity", "high_intensity_distance"],
    "thi": ["THI", "thi", "time_high_intensity"],
    "sprint": ["Sprint", "sprint", "sprints", "N° Sprints", "nb_sprints"],
    "test_iso": ["Test ISO", "test_iso", "iso_test"],
}

def get_field(row: dict, key: str):
    aliases = ALIAS.get(key, [key])
    for a in aliases:
        if a in row and row[a] not in [None, ""]:
            return row[a]
    return None

# ---------------- Rapport individuel ----------------

# Header avec photo + identité
def load_google_drive_image(url: str):
    """Convert Drive link → download link and fetch bytes."""
    if not isinstance(url, str):
        return None
    if "drive.google.com" not in url:
        return url
    try:
        if "id=" in url:
            file_id = url.split("id=")[1].split("&")[0]
        elif "/d/" in url:
            file_id = url.split("/d/")[1].split("/")[0]
        else:
            return None
        direct_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        r = requests.get(direct_url, stream=True, timeout=15)
        if r.status_code == 200 and "image" in r.headers.get("Content-Type", ""):
            return BytesIO(r.content)
        return None
    except Exception as e:
        st.warning(f"Error loading image: {e}")
        return None

col_photo, col_id = st.columns([1,2])

photo_url = get_field(sheet_row, "photo")
if photo_url is None and not sheet_slice.empty:
    for k in ["photo", "Photo", "img_url", "image", "avatar"]:
        if k in sheet_slice.columns:
            s = sheet_slice[k].dropna()
            if not s.empty:
                photo_url = str(s.iloc[0])
                break

with col_photo:
    if photo_url:
        img_bytes = load_google_drive_image(photo_url)
        if img_bytes:
            st.image(img_bytes, width=150)
        else:
            st.write("❌ Image non accessible.")
    else:
        st.write("No photo available")

with col_id:
    numero = _fmt(get_field(sheet_row, "numero"))
    age = _fmt(get_field(sheet_row, "age"))
    nat = _fmt(get_field(sheet_row, "nationalite"))
    pren = _fmt(get_field(sheet_row, "prenom"))
    st.markdown(f"**Nom**: {selected_name}")
    st.markdown(f"**Prénom**: {pren}")
    st.markdown(f"**Numéro**: {numero}")
    st.markdown(f"**Âge**: {age}")
    st.markdown(f"**Nationalité**: {nat}")


st.markdown("#### 📊 Stats du joueur")

stats_fields = {
    "Titularisation": ["Titularisation", "Titulaire", "Titularisations"],
    "Entrant": ["Entrant", "Entrées", "Remplaçant"],
    "Minutes": ["Minutes", "Min", "Temps de jeu"],
    "Carton": ["Carton", "Cartons", "Cartons reçus"]
}

def _find_stat_value(row: dict, keys: list[str]):
    for k in keys:
        if k in row and row[k] not in [None, ""]:
            try:
                return int(float(str(row[k]).replace(",", ".")))
            except Exception:
                return row[k]
    return "—"

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Titularisation", _find_stat_value(sheet_row, stats_fields["Titularisation"]))
with c2:
    st.metric("Entrant", _find_stat_value(sheet_row, stats_fields["Entrant"]))
with c3:
    st.metric("Minutes", _find_stat_value(sheet_row, stats_fields["Minutes"]))
with c4:
    st.metric("Carton", _find_stat_value(sheet_row, stats_fields["Carton"]))

st.divider()

# ===================== Information contractuelle =====================
with st.expander("", expanded=True):
    st.markdown("<h3 style='color:#0031E3; font-weight:800; font-size:22px;'>Information Contractuelle</h3>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f"**Début de contrat**: {_fmt(get_field(sheet_row, 'debut_contrat'))}")
    with c2: st.markdown(f"**Fin de contrat**: {_fmt(get_field(sheet_row, 'fin_contrat'))}")
    with c3:
        team_disp = sb_slice['team_name'].iloc[0] if not sb_slice.empty and 'team_name' in sb_slice.columns else FOCUS_TEAM
        st.markdown(f"**Équipe**: {_fmt(team_disp)}")
    with c4:
        st.markdown(f"**Type**: {_fmt(get_field(sheet_row, 'type'))}")

# ===================== Caractéristique du joueur =====================
with st.expander("", expanded=True):
    st.markdown(
        "<h3 style='color:#0031E3; font-weight:800; font-size:22px;'>Caractéristique du joueur</h3>",
        unsafe_allow_html=True
    )

    # --- Première ligne ---
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f"**Poste**: {_fmt(get_field(sheet_row, 'poste'))}")
    with c2: st.markdown(f"**Pied**: {_fmt(get_field(sheet_row, 'pied'))}")
    with c3: st.markdown(f"**Profil**: {_fmt(get_field(sheet_row, 'profil'))}")
    with c4: st.markdown(f"**Joueur Référence**: {_fmt(get_field(sheet_row, 'joueur_reference'))}")

    # --- Deuxième ligne ---
    d1, d2 = st.columns(2)
    with d1: st.markdown(f"**Performance**: {_fmt(get_field(sheet_row, 'performance'))}")
    with d2: st.markdown(f"**Développement**: {_fmt(get_field(sheet_row, 'Developpement'))}")


# ===================== Auto-Évaluation =====================
with st.expander("", expanded=True):
    st.markdown("<h3 style='color:#0031E3; font-weight:800; font-size:22px;'>Auto-Evaluation</h3>", unsafe_allow_html=True)

    st.markdown("**Objectifs**")
    st.write(_fmt(get_field(sheet_row, "objectifs")))
    a, b, c, d, e = st.columns(5)
    with a:
        st.markdown("**Plan réussite | Personnel**")
        st.write(_fmt(get_field(sheet_row, "plan_reussite_personnel")))
    with b:
        st.markdown("**Plan réussite | Tactique**")
        st.write(_fmt(get_field(sheet_row, "plan_reussite_tactique")))
    with c:
        st.markdown("**Plan réussite | Technique**")
        st.write(_fmt(get_field(sheet_row, "plan_reussite_technique")))
    with d:
        st.markdown("**Plan réussite | Mental**")
        st.write(_fmt(get_field(sheet_row, "plan_reussite_mental")))
    with e:
        st.markdown("**Plan réussite | Groupe**")
        st.write(_fmt(get_field(sheet_row, "plan_reussite_groupe")))

# ===================== Développement du joueur =====================
with st.expander("", expanded=True):
    st.markdown("<h3 style='color:#0031E3; font-weight:800; font-size:22px;'>Développement du joueur</h3>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Thème de progression**")
        st.write(_fmt(get_field(sheet_row, "theme_progression")))
    with c2:
        st.markdown("**Plan de Développement**")
        st.write(_fmt(get_field(sheet_row, "plan_developpement")))

# ===================== Profil Athlétique =====================
with st.expander("", expanded=True):
    st.markdown("<h3 style='color:#0031E3; font-weight:800; font-size:22px;'>Profil Athlétique</h3>", unsafe_allow_html=True)

    k1, k2, k3, k4, k5, k6, k7, k8, k9 = st.columns(9)
    k1.metric("Taille", _fmt(get_field(sheet_row, "taille")))
    k2.metric("Poids", _fmt(get_field(sheet_row, "poids")))
    k3.metric("MG %", _fmt(get_field(sheet_row, "mg_percent")))
    k4.metric("Distance", _fmt(get_field(sheet_row, "distance")))
    k5.metric("Vmax", _fmt(get_field(sheet_row, "vmax")))
    k6.metric("HI", _fmt(get_field(sheet_row, "hi")))
    k7.metric("THI", _fmt(get_field(sheet_row, "thi")))
    k8.metric("Sprint", _fmt(get_field(sheet_row, "sprint")))
    k9.metric("Test ISO", _fmt(get_field(sheet_row, "test_iso")))


st.divider()

# ---------------- Tables brutes ----------------
st.subheader("Raw tables")

st.dataframe(
    player_season if not player_season.empty else pd.DataFrame(),
    use_container_width=True,
    height=300
)

st.divider()


# ---------------- KPIs rapides StatsBomb ----------------
st.subheader("Indicateur de performance")

# Vérification minimale
if sb_players.empty:
    st.info("Pas de données StatsBomb disponibles.")
    st.stop()

# Liste des colonnes numériques potentielles
num_cols = sorted([
    c for c in sb_players.columns
    if sb_players[c].dtype in ["int64", "float64", "Int64"]
])

if len(num_cols) < 2:
    st.info("Pas assez de colonnes numériques pour tracer un scatter.")
    st.stop()

# Sélecteurs latéraux
col_x, col_y = st.columns(2)
with col_x:
    x_var = st.selectbox("Variable X", num_cols, index=0)
with col_y:
    y_var = st.selectbox("Variable Y", num_cols, index=1)

# Le DataFrame complet compétition permet d'ajouter du contexte
plot_df = player_season_all.copy()

# On normalise et force numérique
plot_df[x_var] = pd.to_numeric(plot_df[x_var], errors="coerce")
plot_df[y_var] = pd.to_numeric(plot_df[y_var], errors="coerce")

# Couleur par défaut
plot_df["color"] = "#1f77b4"    # bleu
plot_df["size"] = 9

# Highlight joueur sélectionné
plot_df.loc[
    plot_df["player_id"].astype("Int64") == selected_id,
    ["color", "size"]
] = ["red", 18]

# ---------------- Plotly scatter ----------------
fig = px.scatter(
    plot_df,
    x=x_var,
    y=y_var,
    color=plot_df["color"],         # override colors
    size=plot_df["size"],
    hover_data=["player_name", "team_name", "player_id"],
)

fig.update_traces(marker=dict(line=dict(width=1, color="black")))

fig.update_layout(
    height=480,
    showlegend=False,
    title=f"Scatter {x_var} vs {y_var} — joueurs compétition (Versailles en rouge)",
)

st.plotly_chart(fig, use_container_width=True)


