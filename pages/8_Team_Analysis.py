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
from statsbombpy import sb
import pandas as pd
import numpy as np
from matplotlib.patches import Polygon
import streamlit as st
import re


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

# --- Credentials ---
DEFAULT_CREDS = {
    "user": "mathieu.feigean@fcversailles.com",
    "passwd": "uVBxDK5X",
}

# --- Load Matches ---
matches = sb.matches(competition_id=129, season_id=318, creds=DEFAULT_CREDS)

if matches.empty:
    raise ValueError("No match data available for the specified competition and season.")

# Keep only available matches
matches = matches[matches["match_status"] == "available"]

# --- Filter only Versailles matches ---
versailles_matches = matches[
    (matches["home_team"] == "Versailles") | (matches["away_team"] == "Versailles")
].copy()

# Sort by date (newest first)
versailles_matches["match_date"] = pd.to_datetime(versailles_matches["match_date"])
versailles_matches = versailles_matches.sort_values("match_date", ascending=False)

# --- Optional: Add last game if missing ---
# Check last match from StatsBomb
latest_match = versailles_matches.head(1)
# Append if not already in df (in case you're storing old df in Streamlit)
# versailles_matches = pd.concat([versailles_matches, latest_match]).drop_duplicates(subset=["match_id"])

# --- Get Team Match Stats for each match ---
team_match_stats_list = []
for match_id in versailles_matches["match_id"].unique():
    team_stats = sb.team_match_stats(match_id, creds=DEFAULT_CREDS)
    team_match_stats_list.append(team_stats)

statsbomb_df = pd.concat(team_match_stats_list, ignore_index=True)

selected_columns = [
    "match_id",
    "team_name",
    "team_match_possession",
    "opposition_name",
    "team_match_goals",
    "team_match_goals_conceded",
    "team_match_np_xg",
    "team_match_passes_inside_box",
    "team_match_deep_progressions",
    "team_match_obv",
    "team_match_obv_shot",
    "team_match_obv_shot_nconceded",
    "team_match_np_xg_conceded",
    "team_match_ppda",
    "team_match_aggression",
    "team_match_fhalf_pressures"
]

# Filter the DataFrame to keep only the selected columns
statsbomb_df = statsbomb_df[selected_columns]


SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
TOKEN_FILE_PTS = 'token_pts.pickle'
CLIENT_SECRET_FILE = 'client_secret.json'   # ← si vous utilisez 'client_secret_v1.json', changez ce nom

SPREADSHEET_ID = '19Sear5ZiqBk_dzqqMZsS27CHECqgz7fHC4LlgsqKkZY'
SHEET_NAME = 'Feuille 1'
RANGE_NAME = f"'{SHEET_NAME}'!A1:Z1000"

SPREADSHEET_ID_2 = '1T9jaldVN6fSFwVsBQmAyChTEWuHeoIXmTeoOR81fZhs'
SHEET_NAME_2 = 'Feuille 1'
RANGE_2 = f"'{SHEET_NAME_2}'!A1:AF1000"

# ── Auth ────────────────────────────────────────────────────────────────────
def get_credentials(token_file: str, client_secret_file: str) -> object:
    creds = None
    if os.path.exists(token_file):
        with open(token_file, 'rb') as f:
            creds = pickle.load(f)

    if not creds or not creds.valid:
        if creds and creds.expired and getattr(creds, "refresh_token", None):
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(client_secret_file, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_file, 'wb') as f:
            pickle.dump(creds, f)
    return creds

# ── Fetch ───────────────────────────────────────────────────────────────────
def fetch_google_sheet(spreadsheet_id: str, range_name: str,
                       token_file: str = TOKEN_FILE_PTS,
                       client_secret_file: str = CLIENT_SECRET_FILE) -> pd.DataFrame:
    """
    Lit une plage Google Sheets et renvoie un DataFrame.
    - Utilise la première ligne comme en-têtes.
    - Convertit automatiquement les colonnes numériques lorsque possible.
    """
    creds = get_credentials(token_file, client_secret_file)
    service = build('sheets', 'v4', credentials=creds)
    result = service.spreadsheets().values().get(
        spreadsheetId=spreadsheet_id,
        range=range_name,
        valueRenderOption='FORMATTED_VALUE'
    ).execute()

    rows = result.get('values', [])
    if not rows:
        raise ValueError(f"Aucune donnée trouvée dans la plage {range_name}.")

    header, *data_rows = rows

    # Normalise la longueur des lignes si nécessaire (padding/troncature)
    n_cols = len(header)
    fixed_rows = []
    for r in data_rows:
        if len(r) < n_cols:
            r = r + [''] * (n_cols - len(r))
        elif len(r) > n_cols:
            r = r[:n_cols]
        fixed_rows.append(r)

    df = pd.DataFrame(fixed_rows, columns=header)

    # Conversion numérique "best-effort"
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='ignore')

    return df

# ── Load Data ───────────────────────────────────────────────────────────────
df  = fetch_google_sheet(SPREADSHEET_ID,  RANGE_NAME)   # Feuille 1
df2 = fetch_google_sheet(SPREADSHEET_ID_2, RANGE_2)  


########################################################################################################
########################################################################################################

# Données
x_values = list(range(1, len(df) + 1))
green      = "#00FF00"
dark_blue  = "#0031E3"
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
        marker=dict(color=color, size=8, opacity=0.4),
        name=col,
        showlegend=False
    ))

# Saisons clés
fig.add_trace(go.Scatter(
    x=x_values, y=df['col12'],
    mode='markers',
    marker=dict(color=dark_blue, size=10, opacity=0.2),
    name='FCV 24/25'
))
fig.add_trace(go.Scatter(
    x=x_values, y=df['col13'],
    mode='markers',
    marker=dict(color=light_blue, size=10, opacity=0.2),
    name='FCV 23/24'
))
fig.add_trace(go.Scatter(
    x=x_values, y=df['Le Mans'],
    mode='markers',
    marker=dict(color=orange, size=10, opacity=0.2),
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
fig.add_annotation(x=2.2, y=35, text="Cible maintien : 34 pts",
                   showarrow=False,
                   font=dict(size=10, color="#0031E3", family="Arial Black"))
fig.add_annotation(x=2, y=45, text="Cible Top 8 : 44 pts",
                   showarrow=False,
                   font=dict(size=10, color="#0031E3", family="Arial Black"))

# Mise à jour du layout
fig.update_layout(
    width=1600,
    height=900,
    title=dict(
        text="Route vers le top 8 | 44 pts",
        x=0.02, xanchor="left",
        font=dict(size=20, color="#0031E3", family="Arial Black")
    ),
    xaxis=dict(
        title=dict(text="Matchs",
                   font=dict(size=14, color="#0031E3", family="Arial Black")),
        tickmode="array",
        tickvals=list(range(1, 33)),
        tickfont=dict(size=12,color="#0031E3")
    ),
    yaxis=dict(
        title=dict(text="Points",
                   font=dict(size=14, color="#0031E3", family="Arial Black")),
        tickmode="array",
        tickvals=list(range(0, 66, 5)),
        tickfont=dict(size=12,color="#0031E3")
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


######################################################################################


# ----------------- Helpers -----------------
def to_float_series(s: pd.Series) -> pd.Series:
    """Convert cleanly to float (handles '3,25' -> 3.25)."""
    if s.dtype.kind in "biufc":
        return pd.to_numeric(s, errors="coerce")
    return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False), errors="coerce")

def to_pct(series: pd.Series) -> pd.Series:
    """
    Convert a series to percentage [0..100].
    - if most values ≤ 1.2 -> assume 0–1 and *100
    - else assume already in %
    """
    x = to_float_series(series)
    nonan = x.dropna()
    if nonan.empty:
        return x
    if (nonan <= 1.2).mean() >= 0.8:
        x = x * 100.0
    return x

def odds3_to_probs_pct(win_odds: pd.Series, draw_odds: pd.Series, lose_odds: pd.Series) -> pd.DataFrame:
    """
    Accept decimal odds OR probabilities (0–1) OR percentages.
    Returns fair probabilities (%) for win/draw/lose that sum to 100 (row-wise).
    """
    w = to_float_series(win_odds).to_numpy()
    d = to_float_series(draw_odds).to_numpy()
    l = to_float_series(lose_odds).to_numpy()

    n = len(w)
    pW = np.full(n, np.nan, float)
    pD = np.full(n, np.nan, float)
    pL = np.full(n, np.nan, float)

    valid = np.isfinite(w) & np.isfinite(d) & np.isfinite(l) & (w > 0) & (d > 0) & (l > 0)
    if not np.any(valid):
        return pd.DataFrame({'win prob': pW, 'draw prob': pD, 'lose prob': pL})

    s = w + d + l

    # Already probabilities (sum≈1)
    is_prob01 = valid & (s >= 0.95) & (s <= 1.05)
    pW[is_prob01] = (w[is_prob01] / s[is_prob01]) * 100.0
    pD[is_prob01] = (d[is_prob01] / s[is_prob01]) * 100.0
    pL[is_prob01] = (l[is_prob01] / s[is_prob01]) * 100.0

    # Already percentages (sum≈100)
    is_pct = valid & (s >= 95.0) & (s <= 105.0)
    pW[is_pct] = (w[is_pct] / s[is_pct]) * 100.0
    pD[is_pct] = (d[is_pct] / s[is_pct]) * 100.0
    pL[is_pct] = (l[is_pct] / s[is_pct]) * 100.0

    # Decimal odds -> implied probs ∝ 1/odds, then normalize (de-vig)
    is_odds = valid & ~(is_prob01 | is_pct)
    if np.any(is_odds):
        invW = 1.0 / w[is_odds]
        invD = 1.0 / d[is_odds]
        invL = 1.0 / l[is_odds]
        invS = invW + invD + invL
        pW[is_odds] = (invW / invS) * 100.0
        pD[is_odds] = (invD / invS) * 100.0
        pL[is_odds] = (invL / invS) * 100.0

    return pd.DataFrame({'win prob': pW, 'draw prob': pD, 'lose prob': pL})

def expected_points_from_pct(win_pct: pd.Series, draw_pct: pd.Series) -> pd.Series:
    """EP = 3*(win%/100) + 1*(draw%/100)."""
    return 3.0 * (to_float_series(win_pct) / 100.0) + 1.0 * (to_float_series(draw_pct) / 100.0)

# ----------------- Validate required cols -----------------
required = ['Game day','Team','Win odds','Draw odds','Lose odds','Win chance','Draw chance','Points']
missing = [c for c in required if c not in df2.columns]
if missing:
    st.error(f"Missing columns: {missing}")
    st.stop()

# ----------------- Compute probs, xpts, dpts -----------------
df2 = df2.copy()
probs = odds3_to_probs_pct(df2['Win odds'], df2['Draw odds'], df2['Lose odds'])
df2 = df2.join(probs)

# Optional sanity check column (should be ~100; float rounding may give 99.99/100.01)
df2['prob_sum'] = df2['win prob'] + df2['draw prob'] + df2['lose prob']

# xPTS from odds-based probs
df2['xpts'] = expected_points_from_pct(df2['win prob'], df2['draw prob'])

# dPTS from your "chance" columns (auto 0–1 vs %)
df2['dpts'] = expected_points_from_pct(to_pct(df2['Win chance']), to_pct(df2['Draw chance']))

# Harmonize basics
df2['day']    = pd.to_numeric(df2['Game day'], errors='coerce')
df2['Team']   = df2['Team'].astype(str).str.strip()
df2['Points'] = to_float_series(df2['Points'])

# ----------------- UI & filtering -----------------
teams = sorted(df2['Team'].dropna().unique().tolist())
default_idx = teams.index("Versailles") if "Versailles" in teams else 0
team_sel = st.selectbox("Choisissez l’équipe", teams, index=default_idx)

df_team = df2[df2['Team'].str.casefold() == team_sel.casefold()].sort_values('day').reset_index(drop=True)
if df_team.empty:
    st.warning("Aucun match pour ce filtre.")
    st.stop()

# ----------------- Derived KPIs -----------------
df_team['cumul_pts']  = df_team['Points'].cumsum()
df_team['cumul_xpts'] = df_team['xpts'].cumsum()
df_team['cumul_dpts'] = df_team['dpts'].fillna(0).cumsum()

df_team['mean_pts']  = df_team['Points'].shift().rolling(3, min_periods=1).mean()
df_team['mean_xpts'] = df_team['xpts'].shift().rolling(3, min_periods=1).mean()
df_team['mean_dpts'] = df_team['dpts'].shift().rolling(3, min_periods=1).mean()

df_team['luck'] = df_team['Points'] - df_team['dpts']
df_team['perf'] = df_team['dpts']   - df_team['xpts']

# ----------------- Plots -----------------
COL_XPTS = '#B1B4B2'  # gris
COL_DPTS = '#CFB013'  # or
COL_PTS  = '#0031E3'  # bleu

# Plot 1: cumulative
st.markdown("### Dynamique de l'équipe")
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=df_team['day'], y=df_team['cumul_xpts'], mode='lines+markers',
                          name='xPTS (odds→prob)', line=dict(color=COL_XPTS, width=2),
                          marker=dict(size=6, opacity=0.85)))
fig1.add_trace(go.Scatter(x=df_team['day'], y=df_team['cumul_dpts'], mode='lines+markers',
                          name='dPTS (win/draw chance)', line=dict(color=COL_DPTS, width=2),
                          marker=dict(size=6, opacity=0.85)))
fig1.add_trace(go.Scatter(x=df_team['day'], y=df_team['cumul_pts'], mode='lines+markers',
                          name='PTS (réels)', line=dict(color=COL_PTS, width=2),
                          marker=dict(size=6, opacity=0.9)))
fig1.update_layout(title='', xaxis_title='Journée', yaxis_title='Cumulative Points',
                   hovermode='x unified', legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0),
                   margin=dict(l=30, r=20, t=50, b=30))
fig1.update_xaxes(range=[0.5, 10], dtick=1, showgrid=True, gridcolor='lightgrey')
fig1.update_yaxes(range=[0, 10],showgrid=True, gridcolor='lightgrey')
st.plotly_chart(fig1, use_container_width=True)


# Plot 3: Luck & Perf
st.markdown("### Performance & Points")
mean_luck = df_team['luck'].mean()
mean_perf = df_team['perf'].mean()
max_luck = float(np.nanmax(np.abs(df_team['luck']))) if len(df_team) else 1.0
max_perf = float(np.nanmax(np.abs(df_team['perf']))) if len(df_team) else 1.0
max_luck = max(max_luck, 0.5)
max_perf = max(max_perf, 0.5)

fig3 = go.Figure()
fig3.add_trace(go.Scatter(
    x=df_team['luck'], y=df_team['perf'],
    mode='markers', marker=dict(size=12, color=COL_PTS, opacity=0.9),
    text=df_team['Team'],
    hovertemplate="Luck: %{x:.2f}<br>Perf: %{y:.2f}<br>Équipe: %{text}<extra></extra>",
    name="Matches"
))
fig3.add_trace(go.Scatter(
    x=[mean_luck], y=[mean_perf],
    mode='markers', marker=dict(size=8, color='black', symbol='x'),
    name='Moyenne'
))
fig3.add_shape(type="line", x0=-3, x1=3, y0=0, y1=0, line=dict(color="black", dash="dash"))
fig3.add_shape(type="line", x0=0, x1=0, y0=-3, y1=3, line=dict(color="black", dash="dash"))

fig3.add_annotation(x=0.2, y=2.95, text="Performance", showarrow=False, font=dict(size=12), yshift=10)
fig3.add_annotation(x=2.95, y=-0.2, text="Points", showarrow=False, font=dict(size=12), xshift=20)
fig3.update_layout(title="",
                   xaxis_title="Chance", yaxis_title="Over/under perf",
                   hovermode='closest', margin=dict(l=30, r=30, t=50, b=30), height=600)
fig3.update_xaxes(range=[-3, 3], showgrid=True, gridcolor='lightgrey')
fig3.update_yaxes(range=[-3, 3], showgrid=True, gridcolor='lightgrey')
st.plotly_chart(fig3, use_container_width=True)

# Optional: quick check the probs sum to 100
# st.write(df_team[['win prob','draw prob','lose prob','prob_sum']].head(10))

# Plot 2: rolling means (3 games)
st.markdown("### Dynamique — moyennes (3 matchs)")
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=df_team['day'], y=df_team['mean_xpts'], mode='lines+markers',
                          name='xPTS (odds→prob)', line=dict(color=COL_XPTS, width=2),
                          marker=dict(size=6, opacity=0.85)))
fig2.add_trace(go.Scatter(x=df_team['day'], y=df_team['mean_dpts'], mode='lines+markers',
                          name='dPTS (win/draw chance)', line=dict(color=COL_DPTS, width=2),
                          marker=dict(size=6, opacity=0.85)))
fig2.add_trace(go.Scatter(x=df_team['day'], y=df_team['mean_pts'], mode='lines+markers',
                          name='PTS (réels)', line=dict(color=COL_PTS, width=2),
                          marker=dict(size=6, opacity=0.9)))
fig2.update_layout(title="",
                   xaxis_title='Journée', yaxis_title='Moyenne Points par Match',
                   hovermode='x unified', legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0),
                   margin=dict(l=30, r=20, t=50, b=30))
fig2.update_xaxes(range=[0.5, 10],dtick=1, showgrid=True, gridcolor='lightgrey')
fig2.update_yaxes(range=[0, 3], dtick=0.25, showgrid=True, gridcolor='lightgrey')
st.plotly_chart(fig2, use_container_width=True)




###################################################################################

st.markdown("### Pyramide des KPI")

# --- KPI Targets ---
targets = {
    "Goals_Diff": 0.99,
    "team_match_goals": 1.5,
    "team_match_np_xg": 1.25,
    "team_match_passes_inside_box": 4,
    "team_match_deep_progressions": 40,
    "team_match_obv": 1.6,
    "team_match_obv_shot": 0.01,
    "team_match_goals_conceded": 0.8,           # lower is better
    "team_match_np_xg_conceded": 0.90,          # lower is better
    "team_match_ppda": 8,                       # lower is better
    "team_match_aggression": 0.24,
    "team_match_fhalf_pressures": 75,
    "team_match_obv_shot_nconceded": 0.5
}



# --- Filtre principal : uniquement Versailles ---
versailles_df = statsbomb_df[statsbomb_df["team_name"] == "Versailles"].copy()

# --- Liste des adversaires avec option "All" ---
opponents = sorted(versailles_df["opposition_name"].dropna().unique().tolist())
opponents = opponents

# --- Sélecteurs sur deux colonnes ---
col1, col2 = st.columns(2)

with col1:
    selected_opponent = st.selectbox("Select Opponent", opponents)

if selected_opponent != "All":
    versailles_df = versailles_df[versailles_df["opposition_name"] == selected_opponent]

with col2:
    match_ids = versailles_df["match_id"].tolist()
    selected_match = st.selectbox("Select Match", match_ids)

# --- Colonnes auxiliaires ---
versailles_df["Goals_Diff"] = versailles_df["team_match_goals"] - versailles_df["team_match_goals_conceded"]
versailles_df["Pts"] = versailles_df["Goals_Diff"].apply(lambda gd: 3 if gd > 0 else (1 if gd == 0 else 0))

# --- Validation des KPI ---
def check_kpi(row):
    results = {}
    for kpi, target in targets.items():
        if "conceded" in kpi or kpi == "team_match_ppda":  # KPI défensifs : plus bas = mieux
            results[kpi] = row.get(kpi, np.nan) <= target
        else:  # KPI offensifs : plus haut = mieux
            results[kpi] = row.get(kpi, np.nan) >= target
    return results

versailles_df["kpi_results"] = versailles_df.apply(check_kpi, axis=1)
versailles_df["kpi_validated_count"] = versailles_df["kpi_results"].apply(lambda res: sum(res.values()))
versailles_df["total_kpi"] = versailles_df["kpi_results"].apply(len)

# --- Ligne du match sélectionné ---
row = versailles_df[versailles_df["match_id"] == selected_match].iloc[0]
kpis_ok = row["kpi_results"]



# Sécurisation du champ OBV shot concédé (nom variable parfois différent)
obv_shot_nc = row.get("team_match_obv_shot_nconceded", row.get("team_match_obv_shot", np.nan))

vals = {
    "pts": row["Pts"],
    "gd": row["Goals_Diff"],
    "goals": row["team_match_goals"],
    "goals_conceded": row["team_match_goals_conceded"],
    "xg": row["team_match_np_xg"],
    "obv": row["team_match_obv"],
    "obv_shot": row["team_match_obv_shot"],
    "xg_conceded": row["team_match_np_xg_conceded"],
    "obv_shot_conceded": obv_shot_nc,
    "pib": row["team_match_passes_inside_box"],
    "deep": row["team_match_deep_progressions"],
    "ppda": row["team_match_ppda"],
    "aggression": row["team_match_aggression"],
    "pressures": row["team_match_fhalf_pressures"],
}
# --- Helper: couleur selon validation KPI ---
def kpi_color(kpi_key, default="lightgrey"):
    ok = kpis_ok.get(kpi_key, None)
    if ok is None:
        return default
    return "#CFB013" if ok else "#B1B4B2"

# --- IMPORTS AFFICHAGE SÛR ---

def draw_pyramid(vals, kpis_ok):
    """
    vals: dict avec les valeurs affichées (keys: 'pts','goals','gd','goals_conceded','xg','obv','obv_shot',
                                           'obv_shot_conceded','xg_conceded','pib','deep','ppda','aggression','pressures')
    kpis_ok: dict booléen par KPI StatsBomb (keys: 'team_match_goals','Goals_Diff','team_match_goals_conceded',
                                              'team_match_np_xg','team_match_obv_shot','team_match_obv_shot_nconceded',
                                              'team_match_np_xg_conceded','team_match_passes_inside_box',
                                              'team_match_deep_progressions','team_match_obv','team_match_ppda',
                                              'team_match_aggression','team_match_fhalf_pressures')
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    # ---------- Géométrie équilatérale ----------
    S = 1.0                          # côté (uniforme)
    H = S * np.sqrt(3) / 2.0         # hauteur
    # helpers ancrés sur lignes horizontales
    def tri_up_base(cx, y_base, s=S):
        h = s * np.sqrt(3) / 2.0
        return [(cx - s/2, y_base), (cx + s/2, y_base), (cx, y_base + h)]

    def tri_down_top(cx, y_top, s=S):
        h = s * np.sqrt(3) / 2.0
        return [(cx - s/2, y_top), (cx + s/2, y_top), (cx, y_top - h)]

    # ---------- Style ----------
    EDGE_W = 0.8     # bord fin pour éviter l'impression d'espace
    EDGE   = "#111"
    TEXT   = dict(ha="center", va="center",color="#0031E3", fontsize=9, fontweight="bold")

    def fmt(x, n=2, as_int=False):
        if x is None:
            return "-"
        if isinstance(x, float):
            if np.isnan(x) or np.isinf(x):
                return "-"
        return f"{int(x)}" if as_int else f"{x:.{n}f}"

    def col_color(kpi_key, default="lightgrey"):
        if kpi_key is None:
            return "white"
        ok = kpis_ok.get(kpi_key, None)
        if ok is None:
            return default
        return "#CFB013" if ok else "#B1B4B2"

    def add_triangle(coords, face, label=None, value=None):
        ax.add_patch(Polygon(coords, closed=True, facecolor=face,
                             edgecolor=EDGE, linewidth=EDGE_W, joinstyle="round"))
        if label is not None:
            cx = sum(p[0] for p in coords) / 3.0
            cy = sum(p[1] for p in coords) / 3.0
            ax.text(cx, cy, f"{label}" if value is None else f"{label}\n{value}", **TEXT)

    # ---------- Espacement horizontal compact (dépend de l'orientation) ----------
    def centers_for_row(orientations):
        """
        orientations: liste 'up'/'down' pour la rangée.
        Règle:
         - même orientation adjacente -> +S
         - orientation différente -> +S/2
        Puis centrage global autour de 0.
        """
        xs = [0.0]
        for i in range(1, len(orientations)):
            prev, curr = orientations[i-1], orientations[i]
            dx = S if prev == curr else (S / 2.0)
            xs.append(xs[-1] + dx)
        # centrage
        offset = (xs[0] + xs[-1]) / 2.0
        return [x - offset for x in xs]

    # ---------- Définitions des rangées ----------
    # Rangée 1 (sommet)
    row1 = [("Pts", None, "up", "pts")]
    ori1 = ["up"]

    # Rangée 2
    row2 = [("Goals", "team_match_goals", "up", "goals"),
            ("GD",    "Goals_Diff",      "down","gd"),
            ("GC",    "team_match_goals_conceded","up","goals_conceded")]
    ori2 = ["up","down","up"]

    # Rangée 3
    row3 = [("xG",        "team_match_np_xg",           "up",  "xg"),
            ("OBV Shot",  "team_match_obv_shot",        "down","obv_shot"),
            ("OBV Shot C","team_match_obv_shot_nconceded","down","obv_shot_conceded"),
            ("xGC",       "team_match_np_xg_conceded",  "up",  "xg_conceded")]
    ori3 = ["up","down","down","up"]

    # Rangée 4
    row4 = [("PIB",   "team_match_passes_inside_box","up",   "pib"),
            ("Deep P","team_match_deep_progressions","down", "deep"),
            ("OBV",   "team_match_obv",             "up",   "obv"),
            (None,    None,                          "down", None),   # séparateur blanc inversé
            ("PPDA",  "team_match_ppda",            "up",   "ppda"),
            ("Agg",   "team_match_aggression",      "down", "aggression"),
            ("Press", "team_match_fhalf_pressures", "up",   "pressures")]
    ori4 = ["up","down","up","down","up","down","up"]

    # ---------- X-centers par rangée (compaction horizontale) ----------
    xs_r1 = centers_for_row(ori1)
    xs_r2 = centers_for_row(ori2)
    xs_r3 = centers_for_row(ori3)
    xs_r4 = centers_for_row(ori4)

    # ---------- Y-lignes (compaction verticale pyramidale) ----------
    # Pour chaque rangée r, on a:
    #  - y_top[r]   = ligne haute (où se posent les DOWN)
    #  - y_base[r]  = y_top[r] - H (où se posent les UP)
    # Et on chaîne: y_top[r] = y_base[r-1]
    y_top1  = 3 * H
    y_base1 = y_top1 - H

    y_top2  = y_base1
    y_base2 = y_top2 - H

    y_top3  = y_base2
    y_base3 = y_top3 - H

    y_top4  = y_base3
    y_base4 = y_top4 - H

    # ---------- Figure ----------
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.axis("off")
    ax.axis("off")

    # Décalage horizontal pour les étiquettes à gauche
    label_x = min(xs_r4) - S * 1.2  # un peu à gauche de la pyramide
    label_kwargs = dict(ha="right", va="center",color="#0031E3" ,fontsize=8, fontweight="bold")
    
    # Row 1 : Points
    ax.text(label_x, (y_top1 + y_base1) / 2, "Points", **label_kwargs)
    
    # Row 2 : Buts
    ax.text(label_x, (y_top2 + y_base2) / 2, "Buts", **label_kwargs)
    
    # Row 3 : Occasions créées et concédées
    ax.text(label_x, (y_top3 + y_base3) / 2, "Occasions créées\net concédées", **label_kwargs)
    
    # Row 4 : Modèle de jeu et Performance
    ax.text(label_x, (y_top4 + y_base4) / 2, "Modèle de jeu\net Performance", **label_kwargs)

    # --- Rangée 1 ---
    (lab, kpi, ori, vkey) = row1[0]
    add_triangle(tri_up_base(xs_r1[0], y_base1),
                 face="white", label=lab, value=fmt(vals[vkey], as_int=True))

    # --- Rangée 2 ---
# --- Rangée 2 ---
    for (cx, (lab, kpi, ori, vkey)) in zip(xs_r2, row2):
        if ori == "up":
            coords = tri_up_base(cx, y_base2)
        else:
            coords = tri_down_top(cx, y_top2)
        face = col_color(kpi, "lightgrey")
    
        # Forcer affichage entier pour Goals, GD, GC
        if vkey in ("goals", "gd", "goals_conceded"):
            value = fmt(vals[vkey], as_int=True)
        else:
            value = fmt(vals[vkey]) if vkey else None
    
        add_triangle(coords, face, lab, value)

    # --- Rangée 3 ---
    for (cx, (lab, kpi, ori, vkey)) in zip(xs_r3, row3):
        if ori == "up":
            coords = tri_up_base(cx, y_base3)
        else:
            coords = tri_down_top(cx, y_top3)
        face = col_color(kpi, "lightgrey")
        value = fmt(vals[vkey]) if vkey else None
        add_triangle(coords, face, lab, value)

    # --- Rangée 4 ---
    for (cx, (lab, kpi, ori, vkey)) in zip(xs_r4, row4):
        if ori == "up":
            coords = tri_up_base(cx, y_base4)
        else:
            coords = tri_down_top(cx, y_top4)
        face = "white" if (lab is None and kpi is None) else col_color(kpi, "lightgrey")
        value = (fmt(vals[vkey], as_int=True) if vkey in ("pib","deep","pressures")
                 else (fmt(vals[vkey]) if vkey else None))
        add_triangle(coords, face, lab, value)

    # Cadre & aspect
    x_min, x_max = min(xs_r4) - S, max(xs_r4) + S
    y_min, y_max = y_base4 - 0.1*H, y_top1 + 0.1*H
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    return fig


# st.write({
#     "rows_Versailles": len(versailles_df),
#     "match_ids": len(versailles_df.get("match_id", [])),
#     "selected_match": selected_match if 'selected_match' in locals() else None
# })

# --- RENDU FIGURE ROBUSTE ---
try:
    fig = draw_pyramid(vals, kpis_ok)          # crée la figure
    st.pyplot(fig, use_container_width=True, clear_figure=True)
    plt.close(fig)                        # évite conflits de figures
except Exception as e:
    st.error(f"Échec rendu matplotlib: {e}")
    # Fallback image (utile si Streamlit/pyplot a un glitch CSS)
    try:
        import io
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
        buf.seek(0)
        st.image(buf, caption="Fallback PNG")
    except Exception as e2:
        st.error(f"Fallback image échoué: {e2}")
        
        
        
        
        
        