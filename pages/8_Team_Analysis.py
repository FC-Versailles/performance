#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 21:11:53 2025

@author: fcvmathieu
"""

import os
import pickle
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

import streamlit as st
import plotly.graph_objects as go

from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from statsbombpy import sb


st.set_page_config(layout="wide")

col1, col2 = st.columns([9, 1])
with col1:
    st.title("Analyse Equipe | FC Versailles")
with col2:
    st.image(
        "https://raw.githubusercontent.com/FC-Versailles/wellness/main/logo.png",
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

matches = matches[matches["match_status"] == "available"].copy()

versailles_matches = matches[
    (matches["home_team"] == "Versailles") | (matches["away_team"] == "Versailles")
].copy()

versailles_matches["match_date"] = pd.to_datetime(versailles_matches["match_date"], errors="coerce")
versailles_matches = versailles_matches.sort_values("match_date", ascending=False)

team_match_stats_list = []
for match_id in versailles_matches["match_id"].dropna().unique():
    team_stats = sb.team_match_stats(match_id, creds=DEFAULT_CREDS)
    if team_stats is not None and not team_stats.empty:
        team_match_stats_list.append(team_stats)

if team_match_stats_list:
    statsbomb_df = pd.concat(team_match_stats_list, ignore_index=True)
else:
    statsbomb_df = pd.DataFrame()

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
    "team_match_fhalf_pressures",
]

if not statsbomb_df.empty:
    existing_selected_columns = [c for c in selected_columns if c in statsbomb_df.columns]
    statsbomb_df = statsbomb_df[existing_selected_columns].copy()

SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
TOKEN_FILE_PTS = "token_pts.pickle"
CLIENT_SECRET_FILE = "client_secret.json"

SPREADSHEET_ID = "19Sear5ZiqBk_dzqqMZsS27CHECqgz7fHC4LlgsqKkZY"
SHEET_NAME = "Feuille 1"
RANGE_NAME = f"'{SHEET_NAME}'!A1:Z1000"

SPREADSHEET_ID_2 = "1T9jaldVN6fSFwVsBQmAyChTEWuHeoIXmTeoOR81fZhs"
SHEET_NAME_2 = "Feuille 1"
RANGE_2 = f"'{SHEET_NAME_2}'!A1:AF1000"


# ── Auth ────────────────────────────────────────────────────────────────────
def get_credentials(token_file: str, client_secret_file: str):
    creds = None
    if os.path.exists(token_file):
        with open(token_file, "rb") as f:
            creds = pickle.load(f)

    if not creds or not creds.valid:
        if creds and creds.expired and getattr(creds, "refresh_token", None):
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(client_secret_file, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_file, "wb") as f:
            pickle.dump(creds, f)

    return creds


# ── Fetch ───────────────────────────────────────────────────────────────────
def fetch_google_sheet(
    spreadsheet_id: str,
    range_name: str,
    token_file: str = TOKEN_FILE_PTS,
    client_secret_file: str = CLIENT_SECRET_FILE,
) -> pd.DataFrame:
    """
    Lit une plage Google Sheets et renvoie un DataFrame.
    - Utilise la première ligne comme en-têtes.
    - NE convertit PAS automatiquement toutes les colonnes en numérique.
      Les conversions sont faites plus tard, colonne par colonne.
    """
    creds = get_credentials(token_file, client_secret_file)
    service = build("sheets", "v4", credentials=creds)

    result = service.spreadsheets().values().get(
        spreadsheetId=spreadsheet_id,
        range=range_name,
        valueRenderOption="FORMATTED_VALUE",
    ).execute()

    rows = result.get("values", [])
    if not rows:
        raise ValueError(f"Aucune donnée trouvée dans la plage {range_name}.")

    header, *data_rows = rows

    n_cols = len(header)
    fixed_rows = []
    for r in data_rows:
        if len(r) < n_cols:
            r = r + [""] * (n_cols - len(r))
        elif len(r) > n_cols:
            r = r[:n_cols]
        fixed_rows.append(r)

    df = pd.DataFrame(fixed_rows, columns=header)

    # Keep raw sheet data as strings to avoid destroying text columns like Team
    for col in df.columns:
        df[col] = df[col].astype("string")

    return df


# ── Helpers ─────────────────────────────────────────────────────────────────
def to_float_series(s: pd.Series) -> pd.Series:
    """Convert cleanly to float (handles '3,25' -> 3.25)."""
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")

    s = s.astype("string")
    s = s.str.replace("\u202f", "", regex=False)  # narrow no-break space
    s = s.str.replace("\u00a0", "", regex=False)  # no-break space
    s = s.str.replace(" ", "", regex=False)
    s = s.str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


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
        return pd.DataFrame({"win prob": pW, "draw prob": pD, "lose prob": pL})

    s = w + d + l

    is_prob01 = valid & (s >= 0.95) & (s <= 1.05)
    pW[is_prob01] = (w[is_prob01] / s[is_prob01]) * 100.0
    pD[is_prob01] = (d[is_prob01] / s[is_prob01]) * 100.0
    pL[is_prob01] = (l[is_prob01] / s[is_prob01]) * 100.0

    is_pct = valid & (s >= 95.0) & (s <= 105.0)
    pW[is_pct] = (w[is_pct] / s[is_pct]) * 100.0
    pD[is_pct] = (d[is_pct] / s[is_pct]) * 100.0
    pL[is_pct] = (l[is_pct] / s[is_pct]) * 100.0

    is_odds = valid & ~(is_prob01 | is_pct)
    if np.any(is_odds):
        invW = 1.0 / w[is_odds]
        invD = 1.0 / d[is_odds]
        invL = 1.0 / l[is_odds]
        invS = invW + invD + invL
        pW[is_odds] = (invW / invS) * 100.0
        pD[is_odds] = (invD / invS) * 100.0
        pL[is_odds] = (invL / invS) * 100.0

    return pd.DataFrame({"win prob": pW, "draw prob": pD, "lose prob": pL})


def expected_points_from_pct(win_pct: pd.Series, draw_pct: pd.Series) -> pd.Series:
    return 3.0 * (to_float_series(win_pct) / 100.0) + 1.0 * (to_float_series(draw_pct) / 100.0)


# ── Load Data ───────────────────────────────────────────────────────────────
df = fetch_google_sheet(SPREADSHEET_ID, RANGE_NAME)
df2 = fetch_google_sheet(SPREADSHEET_ID_2, RANGE_2)

########################################################################################################
########################################################################################################

# --- Ensure col14 is numeric ---
if "col14" in df.columns:
    df["col14"] = pd.to_numeric(df["col14"], errors="coerce")
else:
    st.error("La colonne 'col14' est absente du premier Google Sheet.")
    st.stop()

num_days = df["col14"].notna().sum()
last_points = df["col14"].dropna().iloc[-1] if num_days > 0 else 0

pct_objective = (last_points / 44) * 100
pct_played = (num_days / 32) * 100

if pct_objective < 0.1 * pct_played:
    emoji = "🔴"
elif pct_objective == 0:
    emoji = "🟠"
else:
    emoji = "🟢"

st.markdown("### 🔋 Work in progress")
st.markdown(
    f"""
    <div style="font-size:24px; font-weight:bold; color:#0031E3; font-family:Arial Black;">
        {emoji} % de l'objectif atteint : {pct_objective:.1f}%  
        | % du championnat joué : {pct_played:.1f}%
    </div>
    """,
    unsafe_allow_html=True,
)

x_values = list(range(1, len(df) + 1))
green = "#00FF00"
dark_blue = "#0031E3"
light_blue = "#87CEFA"
orange = "orange"

num_hist = len(df.columns[:-2])
greys = [
    f"#{255 - i*15:02x}{255 - i*15:02x}{255 - i*15:02x}"
    for i in range(num_hist)
]

fig = go.Figure()

for col_name, color, size, opacity, label in [
    ("col8", "Yellow", 8, 0.8, "59 pts"),
    ("col12", dark_blue, 10, 0.8, "FCV 24/25"),
    ("col13", light_blue, 10, 0.8, "FCV 23/24"),
    ("Le Mans", orange, 8, 0.8, "Le Mans 24/25"),
    ("Nancy", "red", 8, 0.9, "Nancy"),
    ("Boulogne", "green", 8, 0.9, "Boulogne"),
    ("col14", "black", 10, 0.9, "FCV 25/26"),
]:
    if col_name in df.columns:
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=to_float_series(df[col_name]),
                mode="markers",
                marker=dict(color=color, size=size, opacity=opacity),
                name=label,
            )
        )

fig.add_shape(type="line", x0=1, x1=len(df), y0=34, y1=34, line=dict(color="black", width=1))
fig.add_shape(type="line", x0=1, x1=len(df), y0=44, y1=44, line=dict(color="black", width=1))

fig.add_annotation(
    x=2.2, y=35, text="Cible maintien : 34 pts",
    showarrow=False,
    font=dict(size=10, color="#0031E3", family="Arial Black")
)
fig.add_annotation(
    x=2, y=45, text="Cible Top 8 : 44 pts",
    showarrow=False,
    font=dict(size=10, color="#0031E3", family="Arial Black")
)

fig.update_layout(
    width=1600,
    height=900,
    title=dict(
        text="Route vers le top 8 | 44 pts",
        x=0.02,
        xanchor="left",
        font=dict(size=16, color="#0031E3", family="Arial Black"),
    ),
    xaxis=dict(
        title=dict(text="Matchs", font=dict(size=14, color="#0031E3", family="Arial Black")),
        tickmode="array",
        tickvals=list(range(1, 33)),
        tickfont=dict(size=12, color="#0031E3"),
    ),
    yaxis=dict(
        title=dict(text="Points", font=dict(size=14, color="#0031E3", family="Arial Black")),
        tickmode="array",
        tickvals=list(range(0, 66, 5)),
        tickfont=dict(size=12, color="#0031E3"),
    ),
    legend=dict(
        orientation="v",
        x=1,
        y=0,
        xanchor="right",
        yanchor="bottom",
        bgcolor="rgba(0,0,0,0)",
    ),
    plot_bgcolor="white",
    margin=dict(l=50, r=50, t=80, b=50),
)

fig.update_xaxes(showgrid=True, gridcolor="lightgrey", gridwidth=0.5, zeroline=False)
fig.update_yaxes(showgrid=True, gridcolor="lightgrey", gridwidth=0.5, zeroline=False)

st.plotly_chart(fig)

######################################################################################

required = ["Game day", "Team", "Win odds", "Draw odds", "Lose odds", "Win chance", "Draw chance", "Points"]
missing = [c for c in required if c not in df2.columns]
if missing:
    st.error(f"Missing columns: {missing}")
    st.stop()

df2 = df2.copy()

# Explicit column-by-column conversion only where needed
df2["day"] = pd.to_numeric(df2["Game day"], errors="coerce")
df2["Team"] = df2["Team"].astype("string").str.strip()
df2["Points"] = to_float_series(df2["Points"])

probs = odds3_to_probs_pct(df2["Win odds"], df2["Draw odds"], df2["Lose odds"])
df2 = df2.join(probs)

df2["prob_sum"] = df2["win prob"] + df2["draw prob"] + df2["lose prob"]
df2["xpts"] = expected_points_from_pct(df2["win prob"], df2["draw prob"])
df2["dpts"] = expected_points_from_pct(to_pct(df2["Win chance"]), to_pct(df2["Draw chance"]))

teams = sorted([t for t in df2["Team"].dropna().unique().tolist() if str(t).strip() != ""])
if not teams:
    st.error("Aucune équipe disponible dans la colonne 'Team'.")
    st.stop()

default_idx = teams.index("Versailles") if "Versailles" in teams else 0
team_sel = st.selectbox("Choisissez l’équipe", teams, index=default_idx)

df_team = (
    df2.loc[df2["Team"].str.casefold() == str(team_sel).casefold()]
    .sort_values("day")
    .reset_index(drop=True)
)

if df_team.empty:
    st.warning("Aucun match pour ce filtre.")
    st.stop()

df_team["cumul_pts"] = df_team["Points"].cumsum()
df_team["cumul_xpts"] = df_team["xpts"].cumsum()
df_team["cumul_dpts"] = df_team["dpts"].fillna(0).cumsum()

df_team["mean_pts"] = df_team["Points"].shift().rolling(3, min_periods=1).mean()
df_team["mean_xpts"] = df_team["xpts"].shift().rolling(3, min_periods=1).mean()
df_team["mean_dpts"] = df_team["dpts"].shift().rolling(3, min_periods=1).mean()

df_team["luck"] = df_team["Points"] - df_team["dpts"]
df_team["perf"] = df_team["dpts"] - df_team["xpts"]

COL_XPTS = "#B1B4B2"
COL_DPTS = "#CFB013"
COL_PTS = "#0031E3"

pts_match = df_team["Points"].mean()

if pts_match > 2:
    emoji = "🚀"
elif pts_match > 1.5:
    emoji = "🟢"
elif pts_match >= 1.06:
    emoji = "🟠"
else:
    emoji = "🔴"

st.markdown(f"### Dynamique de l'équipe | Pts/match : {pts_match:.2f} {emoji}")

fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=df_team.index + 1, y=df_team["cumul_xpts"],
    mode="lines+markers", name="xPTS (odds→prob)",
    line=dict(color=COL_XPTS, width=2),
    marker=dict(size=6, opacity=0.85)
))
fig1.add_trace(go.Scatter(
    x=df_team.index + 1, y=df_team["cumul_dpts"],
    mode="lines+markers", name="dPTS (win/draw chance)",
    line=dict(color=COL_DPTS, width=2),
    marker=dict(size=6, opacity=0.85)
))
fig1.add_trace(go.Scatter(
    x=df_team.index + 1, y=df_team["cumul_pts"],
    mode="lines+markers", name="PTS (réels)",
    line=dict(color=COL_PTS, width=2),
    marker=dict(size=6, opacity=0.9)
))

fig1.update_layout(
    title="",
    xaxis_title="Match n°",
    yaxis_title="Points cumulés",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    margin=dict(l=30, r=20, t=50, b=30),
)
fig1.update_xaxes(range=[1, 32], dtick=1, showgrid=True, gridcolor="lightgrey")
fig1.update_yaxes(range=[0, 65], dtick=5, showgrid=True, gridcolor="lightgrey")
st.plotly_chart(fig1, use_container_width=True)

mean_luck = df_team["luck"].mean()
mean_perf = df_team["perf"].mean()

st.markdown(
    f"### Performance & Points | Over/under Perf : {mean_perf:.2f} & Chance : {mean_luck:.2f}"
)

fig3 = go.Figure()
fig3.add_trace(go.Scatter(
    x=df_team["luck"],
    y=df_team["perf"],
    mode="markers+text",
    marker=dict(size=12, color=COL_PTS, opacity=0.9),
    text=df_team["day"],
    textposition="top center",
    hovertemplate=(
        "Game day: %{text}<br>"
        "Luck: %{x:.2f}<br>"
        "Perf: %{y:.2f}<extra></extra>"
    ),
    name="Matches",
))
fig3.add_trace(go.Scatter(
    x=[mean_luck], y=[mean_perf],
    mode="markers",
    marker=dict(size=8, color="black", symbol="x"),
    name="Moyenne",
))
fig3.add_shape(type="line", x0=-3, x1=3, y0=0, y1=0, line=dict(color="black", dash="dash"))
fig3.add_shape(type="line", x0=0, x1=0, y0=-3, y1=3, line=dict(color="black", dash="dash"))

fig3.add_annotation(x=0.2, y=2.95, text="Performance", showarrow=False, font=dict(size=12), yshift=10)
fig3.add_annotation(x=2.95, y=-0.2, text="Points", showarrow=False, font=dict(size=12), xshift=20)
fig3.update_layout(
    title="",
    xaxis_title="Chance",
    yaxis_title="Over/under perf",
    hovermode="closest",
    margin=dict(l=30, r=30, t=50, b=30),
    height=600,
)
fig3.update_xaxes(range=[-2, 2], dtick=0.5, showgrid=True, gridcolor="lightgrey")
fig3.update_yaxes(range=[-2, 2], dtick=0.5, showgrid=True, gridcolor="lightgrey")
st.plotly_chart(fig3, use_container_width=True)

sub_col = next((c for c in df_team.columns if c.strip().lower() in {"submitted_at", "submitted at"}), None)
if sub_col is None:
    st.warning("Colonne 'submitted_at' manquante.")
    st.stop()

df_team["submitted_at"] = pd.to_datetime(df_team[sub_col], errors="coerce")
if df_team["submitted_at"].notna().sum() == 0:
    st.warning("La colonne 'submitted_at' existe mais aucune date valide n'a pu être parsée.")
    st.stop()

day_first_submit = (
    df_team.groupby("day", as_index=False)["submitted_at"]
    .min()
    .rename(columns={"submitted_at": "day_first_submit"})
)
df_team = df_team.merge(day_first_submit, on="day", how="left")

valid_submit = df_team["day_first_submit"].notna()
if not valid_submit.any():
    st.warning("Impossible de construire l'ordre réel joué à partir de 'submitted_at'.")
    st.stop()

df_team["played_order"] = pd.NA
df_team.loc[valid_submit, "played_order"] = (
    df_team.loc[valid_submit, "day_first_submit"]
    .rank(method="dense", ascending=True)
    .astype("Int64")
)
df_team = df_team.dropna(subset=["played_order"]).copy()
df_team["played_order"] = df_team["played_order"].astype(int)

df_team = df_team.sort_values(["played_order", "day"]).reset_index(drop=True)
df_team["Δdpts"] = df_team["dpts"].diff()
df_team["Δxpts"] = df_team["xpts"].diff()

mean_d = df_team["Δdpts"].mean(skipna=True)
mean_x = df_team["Δxpts"].mean(skipna=True)
st.markdown(f"### Dynamique par match | 📈 dPTS : {mean_d:+.2f}% - xPTS : {mean_x:+.2f}%")

df_team["Points_mm3"] = df_team["Points"].rolling(3, min_periods=1).mean()
df_team["dpts_mm3"] = df_team["dpts"].rolling(3, min_periods=1).mean()
df_team["xpts_mm3"] = df_team["xpts"].rolling(3, min_periods=1).mean()

fig2 = go.Figure()

def add_scatter(y, name, color, symbol, group):
    fig2.add_trace(go.Scatter(
        x=df_team["played_order"],
        y=df_team[y],
        mode="markers",
        name=name,
        legendgroup=group,
        marker=dict(color=color, size=10, opacity=0.85, symbol=symbol),
        customdata=np.stack([df_team["day"]], axis=-1),
        hovertemplate="Ordre joué: %{x}<br>Journée: %{customdata[0]}<br>" + name + ": %{y:.2f}<extra></extra>",
    ))

add_scatter("xpts", "xPTS (odds→prob)", COL_XPTS, "circle", "xpts")
add_scatter("dpts", "dPTS (win/draw chance)", COL_DPTS, "diamond", "dpts")
add_scatter("Points", "PTS (réels)", COL_PTS, "square", "pts")

fig2.add_trace(go.Scatter(
    x=df_team["played_order"], y=df_team["xpts_mm3"],
    mode="lines", name="xPTS (MM3)", legendgroup="xpts",
    line=dict(color=COL_XPTS, width=2)
))
fig2.add_trace(go.Scatter(
    x=df_team["played_order"], y=df_team["dpts_mm3"],
    mode="lines", name="dPTS (MM3)", legendgroup="dpts",
    line=dict(color=COL_DPTS, width=2)
))
fig2.add_trace(go.Scatter(
    x=df_team["played_order"], y=df_team["Points_mm3"],
    mode="lines", name="PTS (MM3)", legendgroup="pts",
    line=dict(color=COL_PTS, width=2)
))

fig2.update_layout(
    xaxis_title="Ordre réel joué",
    yaxis_title="Points par Match",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    margin=dict(l=30, r=20, t=50, b=30),
)
fig2.update_xaxes(dtick=1, showgrid=True, gridcolor="lightgrey")
fig2.update_yaxes(showgrid=True, gridcolor="lightgrey")
st.plotly_chart(fig2, use_container_width=True)

# ================== KPI PYRAMID • SEASON MEAN ==================
st.markdown("### Pyramide des KPI — Moyenne saison")

targets = {
    "Goals_Diff": 0.99,
    "team_match_goals": 1.5,
    "team_match_np_xg": 1.25,
    "team_match_passes_inside_box": 4,
    "team_match_deep_progressions": 40,
    "team_match_obv": 1.6,
    "team_match_obv_shot": 0.01,
    "team_match_goals_conceded": 0.8,
    "team_match_np_xg_conceded": 0.90,
    "team_match_ppda": 8,
    "team_match_aggression": 0.24,
    "team_match_fhalf_pressures": 75,
    "team_match_obv_shot_nconceded": 0.5,
}

def _check_kpi_mean(d: dict, targets: dict) -> dict:
    out = {}
    for kpi, tgt in targets.items():
        val = pd.to_numeric(d.get(kpi, np.nan), errors="coerce")
        out[kpi] = (val <= tgt) if (("conceded" in kpi) or (kpi == "team_match_ppda")) else (val >= tgt)
    return out


def _draw_pyramid(vals: dict, kpis_ok: dict):
    S = 1.0
    H = S * np.sqrt(3) / 2.0

    def tri_up_base(cx, yb, s=S):
        h = s * np.sqrt(3) / 2.0
        return [(cx - s/2, yb), (cx + s/2, yb), (cx, yb + h)]

    def tri_down_top(cx, yt, s=S):
        h = s * np.sqrt(3) / 2.0
        return [(cx - s/2, yt), (cx + s/2, yt), (cx, yt - h)]

    EDGE_W, EDGE = 0.8, "#111"
    TEXT = dict(ha="center", va="center", color="#0031E3", fontsize=9, fontweight="bold")

    def fmt(x, n=2):
        if x is None:
            return "-"
        if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
            return "-"
        return f"{float(x):.{n}f}"

    def col_color(k, default="lightgrey"):
        if k is None:
            return "white"
        ok = kpis_ok.get(k, None)
        return default if ok is None else ("#CFB013" if ok else "#B1B4B2")

    def add_triangle(ax, coords, face, label=None, value=None):
        ax.add_patch(Polygon(coords, closed=True, facecolor=face, edgecolor=EDGE, linewidth=EDGE_W, joinstyle="round"))
        if label is not None:
            cx = sum(p[0] for p in coords) / 3.0
            cy = sum(p[1] for p in coords) / 3.0
            ax.text(cx, cy, f"{label}" if value is None else f"{label}\n{value}", **TEXT)

    def centers_for_row(oris):
        xs = [0.0]
        for i in range(1, len(oris)):
            xs.append(xs[-1] + (S if oris[i - 1] == oris[i] else S / 2))
        off = (xs[0] + xs[-1]) / 2.0
        return [x - off for x in xs]

    row1 = [("Pts", None, "up", "pts")]
    ori1 = ["up"]
    row2 = [
        ("Goals", "team_match_goals", "up", "goals"),
        ("GD", "Goals_Diff", "down", "gd"),
        ("Goals C", "team_match_goals_conceded", "up", "goals_conceded"),
    ]
    ori2 = ["up", "down", "up"]
    row3 = [
        ("xG", "team_match_np_xg", "up", "xg"),
        ("OBV Shot", "team_match_obv_shot", "down", "obv_shot"),
        ("OBV Shot C", "team_match_obv_shot_nconceded", "down", "obv_shot_conceded"),
        ("xGC", "team_match_np_xg_conceded", "up", "xg_conceded"),
    ]
    ori3 = ["up", "down", "down", "up"]
    row4 = [
        ("PIB", "team_match_passes_inside_box", "up", "pib"),
        ("Deep P", "team_match_deep_progressions", "down", "deep"),
        ("OBV", "team_match_obv", "up", "obv"),
        (None, None, "down", None),
        ("PPDA", "team_match_ppda", "up", "ppda"),
        ("Agg", "team_match_aggression", "down", "aggression"),
        ("Press", "team_match_fhalf_pressures", "up", "pressures"),
    ]
    ori4 = ["up", "down", "up", "down", "up", "down", "up"]

    xs1, xs2, xs3, xs4 = map(centers_for_row, [ori1, ori2, ori3, ori4])
    yT1 = 3 * H
    yB1 = yT1 - H
    yT2 = yB1
    yB2 = yT2 - H
    yT3 = yB2
    yB3 = yT3 - H
    yT4 = yB3
    yB4 = yT4 - H

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.axis("off")

    label_x = min(xs4) - S * 1.2
    labkw = dict(ha="right", va="center", color="#0031E3", fontsize=8, fontweight="bold")
    ax.text(label_x, (yT1 + yB1) / 2, "Points", **labkw)
    ax.text(label_x, (yT2 + yB2) / 2, "Buts", **labkw)
    ax.text(label_x, (yT3 + yB3) / 2, "Occasions créées\net concédées", **labkw)
    ax.text(label_x, (yT4 + yB4) / 2, "Modèle de jeu\net Performance", **labkw)

    (lab, kpi, ori, vk) = row1[0]
    add_triangle(ax, tri_up_base(xs1[0], yB1), "white", lab, fmt(vals[vk]))

    for cx, (lab, kpi, ori, vk) in zip(xs2, row2):
        coords = tri_up_base(cx, yB2) if ori == "up" else tri_down_top(cx, yT2)
        add_triangle(ax, coords, col_color(kpi), lab, fmt(vals[vk]))

    for cx, (lab, kpi, ori, vk) in zip(xs3, row3):
        coords = tri_up_base(cx, yB3) if ori == "up" else tri_down_top(cx, yT3)
        add_triangle(ax, coords, col_color(kpi), lab, fmt(vals[vk]))

    for cx, (lab, kpi, ori, vk) in zip(xs4, row4):
        coords = tri_up_base(cx, yB4) if ori == "up" else tri_down_top(cx, yT4)
        face = "white" if (lab is None and kpi is None) else col_color(kpi)
        val = None if vk is None else fmt(vals.get(vk))
        add_triangle(ax, coords, face, lab, val)

    x_min, x_max = min(xs4) - S, max(xs4) + S
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(yB4 - 0.1 * H, yT1 + 0.1 * H)
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    return fig


versailles_df = statsbomb_df[statsbomb_df["team_name"] == "Versailles"].copy() if not statsbomb_df.empty else pd.DataFrame()

if versailles_df.empty:
    st.info("Aucune donnée team_match_stats pour Versailles.")
else:
    versailles_df["team_match_goals"] = pd.to_numeric(versailles_df["team_match_goals"], errors="coerce")
    versailles_df["team_match_goals_conceded"] = pd.to_numeric(versailles_df["team_match_goals_conceded"], errors="coerce")
    versailles_df["Goals_Diff"] = versailles_df["team_match_goals"] - versailles_df["team_match_goals_conceded"]
    versailles_df["Pts_match"] = versailles_df["Goals_Diff"].apply(lambda gd: 3 if gd > 0 else (1 if gd == 0 else 0))

    def mean_safe(c):
        return pd.to_numeric(versailles_df.get(c, np.nan), errors="coerce").mean()

    mean_dict = {
        k: mean_safe(k) for k in [
            "team_match_goals",
            "team_match_goals_conceded",
            "team_match_np_xg",
            "team_match_np_xg_conceded",
            "team_match_passes_inside_box",
            "team_match_deep_progressions",
            "team_match_obv",
            "team_match_obv_shot",
            "team_match_obv_shot_nconceded",
            "team_match_ppda",
            "team_match_aggression",
            "team_match_fhalf_pressures",
        ]
    }
    mean_dict["Goals_Diff"] = versailles_df["Goals_Diff"].mean()
    mean_dict["Pts"] = versailles_df["Pts_match"].mean()

    vals = {
        "pts": mean_dict["Pts"],
        "gd": mean_dict["Goals_Diff"],
        "goals": mean_dict["team_match_goals"],
        "goals_conceded": mean_dict["team_match_goals_conceded"],
        "xg": mean_dict["team_match_np_xg"],
        "obv": mean_dict["team_match_obv"],
        "obv_shot": mean_dict["team_match_obv_shot"],
        "xg_conceded": mean_dict["team_match_np_xg_conceded"],
        "obv_shot_conceded": mean_dict["team_match_obv_shot_nconceded"],
        "pib": mean_dict["team_match_passes_inside_box"],
        "deep": mean_dict["team_match_deep_progressions"],
        "ppda": mean_dict["team_match_ppda"],
        "aggression": mean_dict["team_match_aggression"],
        "pressures": mean_dict["team_match_fhalf_pressures"],
    }

    kpis_ok = _check_kpi_mean(mean_dict, targets)
    fig = _draw_pyramid(vals, kpis_ok)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ================== KPI TRENDLINES (rolling) — Pyramide KPIs ==================
st.markdown("### Tendances KPI • fenêtre glissante")

BAR_COLOR = "#0031E3"
LINE_COLOR = "#000000"
COMP_ID = 129
SEASONS = [318]
TEAM = "Versailles"
DEFAULT_WINDOW = 5

PYRAMID_KPIS = [
    "team_match_np_xg",
    "team_match_np_xg_conceded",
    "team_match_passes_inside_box",
    "team_match_deep_progressions",
    "team_match_obv",
    "team_match_obv_shot",
    "team_match_obv_shot_nconceded",
    "team_match_ppda",
    "team_match_aggression",
    "team_match_fhalf_pressures",
    "team_match_goals",
    "team_match_goals_conceded",
]

@st.cache_data(show_spinner=False)
def load_team_match_stats_multi(comp_id: int, seasons: list[int], team: str, creds: dict) -> pd.DataFrame:
    frames = []
    for s in seasons:
        m = sb.matches(competition_id=comp_id, season_id=s, creds=creds)
        if m is None or m.empty:
            continue

        m = m[
            (m["match_status"] == "available") &
            ((m["home_team"] == team) | (m["away_team"] == team))
        ].copy()
        if m.empty:
            continue

        m["match_id"] = pd.to_numeric(m["match_id"], errors="coerce").astype("Int64")
        m["match_date"] = pd.to_datetime(m["match_date"], errors="coerce")

        def _oppo(r):
            return r["away_team"] if r["home_team"] == team else r["home_team"]

        m["opposition_name"] = m.apply(_oppo, axis=1)

        tms_list = []
        for mid in m["match_id"].dropna().astype(int).tolist():
            tms = sb.team_match_stats(match_id=int(mid), creds=creds)
            if tms is None or tms.empty:
                continue
            tms = tms[tms["team_name"] == team].copy()
            if tms.empty:
                continue
            tms_list.append(tms)

        if not tms_list:
            continue

        df_tmp = pd.concat(tms_list, ignore_index=True)
        df_tmp["match_id"] = pd.to_numeric(df_tmp["match_id"], errors="coerce").astype("Int64")

        df_tmp = df_tmp.merge(
            m[["match_id", "match_date", "home_team", "away_team", "opposition_name"]],
            on="match_id",
            how="left",
        )
        frames.append(df_tmp)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values("match_date").reset_index(drop=True)

    if "opposition_name" not in out.columns or out["opposition_name"].isna().all():
        if {"home_team", "away_team", "team_name"}.issubset(out.columns):
            out["opposition_name"] = np.where(
                out["home_team"] == team, out["away_team"], out["home_team"]
            )
        else:
            out["opposition_name"] = "Opposition"

    return out


tms_all = load_team_match_stats_multi(COMP_ID, SEASONS, TEAM, DEFAULT_CREDS)

if tms_all.empty:
    st.info("Aucune donnée `team_match_stats`.")
else:
    available_metrics = [m for m in PYRAMID_KPIS if m in tms_all.columns]
    if not available_metrics:
        st.warning("Aucune des métriques de la pyramide n’est disponible.")
    else:
        c1, c2 = st.columns([3, 1])
        with c1:
            metrics_sel = st.multiselect("KPIs", available_metrics, default=available_metrics[:3])
        with c2:
            window_sel = st.number_input("Fenêtre rolling", min_value=2, max_value=15, value=DEFAULT_WINDOW, step=1)

        for metric in metrics_sel:
            df_plot = tms_all[["match_date", "home_team", "away_team", "team_name", "opposition_name", metric]].copy()
            df_plot = df_plot.sort_values("match_date").reset_index(drop=True)

            if "opposition_name" in df_plot.columns:
                opp = df_plot["opposition_name"].astype("string")
            else:
                opp = pd.Series([""] * len(df_plot), dtype="string")

            deduced = np.where(df_plot["home_team"] == TEAM, df_plot["away_team"], df_plot["home_team"])
            df_plot["opponent"] = np.where(opp.fillna("").str.len() > 0, opp.fillna(""), deduced)

            y = pd.to_numeric(df_plot[metric], errors="coerce")
            rolling = y.rolling(window_sel, min_periods=1).mean()
            avg = y.mean()

            fig, ax = plt.subplots(figsize=(18, 10))
            x_pos = np.arange(len(df_plot))
            ax.bar(x_pos, y, color=BAR_COLOR, alpha=0.95)
            ax.plot(x_pos, rolling, lw=10, color=BAR_COLOR, zorder=9)
            ax.plot(x_pos, rolling, lw=8, color=LINE_COLOR, zorder=10, label=f"Moyenne glissante {window_sel} matchs")
            ax.axhline(avg, color="black", linestyle="--", lw=3, label="Moyenne saison")

            for s in ["top", "right"]:
                ax.spines[s].set_visible(False)
            ax.grid(True, axis="y", color="grey", lw=1, alpha=0.5)

            ax.set_xticks(x_pos)
            ax.set_xticklabels(df_plot["opponent"], rotation=90, ha="center")

            metric_label = metric.replace("team_match_", "").replace("_", " ")
            ax.set_ylabel(metric_label)
            ax.set_xlabel("Adversaire")
            ax.legend(loc="upper left", fontsize=12)
            fig.text(0.125, 0.95, f"{TEAM} — {metric_label} trendline", fontsize=20, fontweight="bold")

            st.pyplot(fig, use_container_width=True)
            plt.close(fig)