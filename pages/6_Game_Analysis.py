#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Game Analysis | FC Versailles
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import streamlit as st

# Optional dependency; guard import for cloud
try:
    from highlight_text import fig_text
except Exception:
    fig_text = None

from statsbombpy import sb

# =======================
# ---- CONSTANTS --------
# =======================
FOCUS_TEAM = "Versailles"
COMP_ID = 129
SEASON_ID = 318

BGCOLOR = "white"
FOCUS_COLOR = "#0031E3"
OPPO_COLOR = "grey"
BAR_COLOR = "black"

USER = os.getenv("SB_USER", "mathieu.feigean@fcversailles.com")
PASSWORD = os.getenv("SB_PASS", "uVBxDK5X")
CREDS = {"user": USER, "passwd": PASSWORD}

st.set_page_config(layout="wide")
col1, col2 = st.columns([9, 1])
with col1:
    st.title("Game Analysis | FC Versailles")
with col2:
    st.image(
        "https://raw.githubusercontent.com/FC-Versailles/wellness/main/logo.png",
        use_container_width=True,
    )
st.markdown("<hr style='border:1px solid #ddd' />", unsafe_allow_html=True)
st.caption("Filtre adversaire + date. Si les événements ne sont pas publiés, le match apparaît mais le rapport est bloqué.")

# =======================
# ---- HELPERS ----------
# =======================
@st.cache_data(show_spinner=False)
def load_matches(comp_id: int, season_id: int, creds: dict) -> pd.DataFrame:
    m = sb.matches(competition_id=comp_id, season_id=season_id, creds=creds)
    if m.empty:
        return m
    m = m[(m["home_team"] == FOCUS_TEAM) | (m["away_team"] == FOCUS_TEAM)].copy()

    # Normalize
    m["match_date"] = pd.to_datetime(m["match_date"], errors="coerce").dt.date
    m["match_status"] = m["match_status"].astype(str)
    m["has_events"] = m["match_status"].str.lower().eq("available")

    def opponent(row):
        return row["away_team"] if row["home_team"] == FOCUS_TEAM else row["home_team"]

    def score_label(row):
        hs = int(row.get("home_score", 0) or 0)
        as_ = int(row.get("away_score", 0) or 0)
        return f"{row['home_team']} {hs}–{as_} {row['away_team']}"

    m["opponent"] = m.apply(opponent, axis=1)
    m["label"] = m.apply(score_label, axis=1)

    m = m.sort_values(by=["match_date", "match_id"], ascending=[False, False]).reset_index(drop=True)
    return m


@st.cache_data(show_spinner=False)
def load_events(match_id: int, creds: dict) -> pd.DataFrame:
    ev = sb.events(match_id=match_id, creds=creds)
    if ev.empty:
        return ev
    for c in ["period", "minute", "second"]:
        if c in ev.columns:
            ev[c] = pd.to_numeric(ev[c], errors="coerce")
    return ev


def group_to_5min(minute_series: pd.Series) -> pd.Series:
    x = pd.to_numeric(minute_series, errors="coerce").fillna(0)
    return ((x // 5) + 1) * 5


def game_flow_ax(data: pd.DataFrame, period: int, ax, focus_team: str, opposition: str):
    df = data[data["period"] == period].copy()
    ax.set_facecolor(BGCOLOR)

    if df.empty:
        ax.set_ylim(-30, 30)
        ax.set_yticklabels([])
        ax.grid(True, color="lightgrey", lw=1, alpha=0.6)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        return 0.0

    df = df.sort_values(["minute", "second"]).reset_index(drop=True)
    df["grouped_min"] = group_to_5min(df["minute"])

    # Possession proxy: completed passes
    poss = df[df["type"] == "Pass"].copy()
    teamposs = poss[(poss["pass_outcome"].isna()) & (poss["team"] == focus_team)]
    oppoposs = poss[(poss["pass_outcome"].isna()) & (poss["team"] == opposition)]
    teamP = teamposs.groupby("grouped_min").size().rename("team_poss")
    oppoP = oppoposs.groupby("grouped_min").size().rename("oppo_poss")
    matchposs = pd.concat([teamP, oppoP], axis=1).fillna(0).reset_index()

    timelist = matchposs["grouped_min"].tolist()
    teamP_list = gaussian_filter1d(matchposs["team_poss"].to_numpy(), sigma=1)
    oppoP_list = gaussian_filter1d(matchposs["oppo_poss"].to_numpy(), sigma=1)
    poss_diff = (teamP_list - oppoP_list).tolist()

    # OBV differential (Pass/Carry/Dribble)
    total_net_obv = 0.0
    obv_diff = []
    timelistOBV = []
    if "obv_total_net" in df.columns:
        obv = df[df["type"].isin(["Pass", "Carry", "Dribble"])].dropna(subset=["obv_total_net"]).copy()
        if not obv.empty:
            teamOBV = obv[obv["team"] == focus_team].groupby("grouped_min")["obv_total_net"].sum()
            oppoOBV = obv[obv["team"] == opposition].groupby("grouped_min")["obv_total_net"].sum()
            matchobv = pd.concat([teamOBV.rename("team"), oppoOBV.rename("oppo")], axis=1).fillna(0).reset_index()
            timelistOBV = matchobv["grouped_min"].tolist()
            teamOBV_list = gaussian_filter1d(matchobv["team"].to_numpy(), sigma=1)
            oppoOBV_list = gaussian_filter1d(matchobv["oppo"].to_numpy(), sigma=1)
            obv_diff = (teamOBV_list - oppoOBV_list) * 50  # scale for visual
            total_net_obv = float(teamOBV_list.sum() - oppoOBV_list.sum())

    # Goals markers
    team_goals = df[
        ((df["team"] == focus_team) & (df["shot_outcome"] == "Goal"))
        | ((df["type"] == "Own Goal For") & (df["team"] == focus_team))
    ]
    oppo_goals = df[
        ((df["team"] == opposition) & (df["shot_outcome"] == "Goal"))
        | ((df["type"] == "Own Goal For") & (df["team"] == opposition))
    ]

    ax.plot(timelist, poss_diff, "lightgrey")

    y_pos = (np.asarray(poss_diff) + 1e-7) > 0
    y_neg = (np.asarray(poss_diff) - 1e-7) < 0
    ax.fill_between(timelist, poss_diff, where=y_pos, interpolate=True, color=FOCUS_COLOR, alpha=0.95,
                    label=f"{focus_team} possession")
    ax.fill_between(timelist, poss_diff, where=y_neg, interpolate=True, color=OPPO_COLOR, alpha=0.95,
                    label=f"{opposition} possession")

    if len(timelistOBV) and len(obv_diff):
        ax.bar(timelistOBV, obv_diff, width=3, color=BAR_COLOR, alpha=0.95, zorder=10, label="Danger (OBV)")

    if not team_goals.empty:
        ax.scatter(team_goals["minute"], np.arange(len(team_goals)), color=FOCUS_COLOR,
                   edgecolor="white", linewidth=2.5, marker="o", s=450, zorder=12, label=f"{focus_team} buts")
    if not oppo_goals.empty:
        ax.scatter(oppo_goals["minute"], -np.arange(1, len(oppo_goals) + 1), color=OPPO_COLOR,
                   edgecolor="black", linewidth=2.0, marker="o", s=450, zorder=12, label=f"{opposition} buts")

    ax.grid(True, color="grey", lw=1, alpha=0.5)
    ax.set_ylim(-30, 30)
    ax.set_yticklabels([])
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    return total_net_obv


def plot_xg_cumulative(data: pd.DataFrame, team_name: str, oppo_name: str,
                       focus_color: str, oppo_color: str):
    events_shots = data[data["type"] == "Shot"].copy()
    team_shots = events_shots[events_shots["team"] == team_name]
    oppo_shots = events_shots[events_shots["team"] == oppo_name]

    def build_stairs(df):
        mins = [0.0]
        vals = [0.0]
        for _, r in df.iterrows():
            m = float(r["minute"]) + float(r["second"]) / 60.0
            mins.append(m)
            vals.append(float(r["shot_statsbomb_xg"]) + vals[-1])
        max_time = max(float(data["minute"].max() or 90) + float(data["second"].max() or 0) / 60.0, 90.0)
        mins.append(max_time)
        return mins, vals

    min_team, xg_team = build_stairs(team_shots)
    min_oppo, xg_oppo = build_stairs(oppo_shots)

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.stairs(values=xg_team, edges=min_team, linewidth=5, label=team_name, color=focus_color)
    ax.stairs(values=xg_oppo, edges=min_oppo, linewidth=5, label=oppo_name, color=oppo_color)

    # Target lines
    ax.axhline(y=1.69, xmin=0.0, xmax=1.0, linewidth=2, color=focus_color, ls="--")
    if fig_text:
        fig_text(0.13, 0.61, "Target occasion créée", color=focus_color, fontweight="bold", fontsize=12, backgroundcolor="0.85")
    ax.axhline(y=1.07, xmin=0.0, xmax=1.0, linewidth=2, color="grey", ls="--")
    if fig_text:
        fig_text(0.13, 0.35, "Target occasion concédée", color="grey", fontweight="bold", fontsize=12, backgroundcolor="0.85")

    ymax = max(xg_team[-1], xg_oppo[-1]) + 0.5
    ax.set_ylim(-0.02, ymax)
    ax.set_yticks([0, 0.7, 1.4, 2.1, 2.8])
    ax.set_xticks([0, 15, 30, 45, 60, 75, 90])

    if fig_text:
        fig_text(0.13, 0.80, s="Mi-temps 1\n", fontsize=12, fontweight="bold", color="black")
        fig_text(0.51, 0.80, s="Mi-temps 2\n", fontsize=12, fontweight="bold", color="black")

    ax.text(min_team[-1] + 1, xg_team[-1], f"{xg_team[-1]:.2f}", color=focus_color)
    ax.text(min_oppo[-1] + 1, xg_oppo[-1], f"{xg_oppo[-1]:.2f}", color=oppo_color)

    ax.set_xlabel("Minute de Jeu", fontsize=14, fontweight="bold", color="black", labelpad=10)
    ax.set_ylabel("Valeur de xG", fontsize=14, fontweight="bold", color="black")
    fig.text(0.5, 0.92, "Expected Goals", ha="center", fontsize=22, fontweight="bold", color="black")
    ax.legend(frameon=True, loc="upper right")
    st.pyplot(fig, use_container_width=True)


# =======================
# ------- UI ------------
# =======================
with st.sidebar:
    if st.button("🔄 Refresh data"):
        st.cache_data.clear()
        st.experimental_rerun()

matches = load_matches(COMP_ID, SEASON_ID, CREDS)
if matches.empty:
    # Auto-try next season if data moved
    fallback = load_matches(COMP_ID, SEASON_ID + 1, CREDS)
    matches = fallback

if matches.empty:
    st.error("Aucun match trouvé pour Versailles sur la saison sélectionnée.")
    st.stop()

with st.sidebar:
    st.header("Filtres")
    show_pending = st.checkbox("Inclure les matchs sans événements (pré-match / live)", value=True)

if not show_pending:
    matches = matches[matches["has_events"]]

if matches.empty:
    st.info("Aucun match avec les filtres actuels.")
    st.stop()

with st.sidebar:
    opponents = sorted(matches["opponent"].unique().tolist())
    sel_oppo = st.selectbox("Adversaire", opponents, index=0)

m_oppo = matches[matches["opponent"] == sel_oppo].copy()
dates_for_oppo = sorted(m_oppo["match_date"].unique().tolist(), reverse=True)

with st.sidebar:
    sel_date = st.selectbox("Date du match", dates_for_oppo, index=0, format_func=lambda d: d.strftime("%Y-%m-%d"))

m_pick = m_oppo[m_oppo["match_date"] == sel_date].copy()
if m_pick.empty:
    st.warning("Aucun match pour cet adversaire et cette date.")
    st.stop()

if len(m_pick) > 1:
    labels = [f"{r['label']}  (id={r['match_id']})" for _, r in m_pick.iterrows()]
    with st.sidebar:
        chosen_label = st.selectbox("Sélectionnez le match exact", labels, index=0)
    match_id = int(chosen_label.split("id=")[-1].strip(")"))
    match_row = m_pick[m_pick["match_id"] == match_id].iloc[0]
else:
    match_row = m_pick.iloc[0]
    match_id = int(match_row["match_id"])

home_team = match_row["home_team"]
away_team = match_row["away_team"]
home_score = int(match_row.get("home_score", 0) or 0)
away_score = int(match_row.get("away_score", 0) or 0)
opposition = match_row["opponent"]
status = match_row["match_status"]
has_events = bool(match_row.get("has_events", False))

# Header
title_col1, title_col2 = st.columns([3, 1])
with title_col1:
    st.subheader(f"{home_team} {home_score}–{away_score} {away_team} • {sel_date.strftime('%Y-%m-%d')}  • {status}")
with title_col2:
    st.metric("Match ID", match_id)

# If events not yet available, block plots cleanly
if not has_events:
    st.info("Rapport indisponible: événements non publiés pour ce match. Réessayez après publication.")
    st.stop()

# =======================
# ---- LOAD EVENTS ------
# =======================
events = load_events(match_id, CREDS)
if events.empty:
    st.info("Aucun événement retourné par l’API pour ce match.")
    st.stop()

# =======================
# ---- PLOT: FLOW -------
# =======================
fig = plt.figure(figsize=(24, 10))
fig.set_facecolor(BGCOLOR)
gs = fig.add_gridspec(nrows=1, ncols=2, wspace=0.15)

ax1 = fig.add_subplot(gs[0])
total_net_obv_1 = game_flow_ax(events, 1, ax1, FOCUS_TEAM, opposition)
ax1.set_ylabel("1ère mi-temps", fontsize=14, color="black")
ax1.legend(fontsize=10, loc="upper right")

ax2 = fig.add_subplot(gs[1])
total_net_obv_2 = game_flow_ax(events, 2, ax2, FOCUS_TEAM, opposition)
ax2.set_ylabel("2ème mi-temps", fontsize=14, color="black")
ax2.legend(fontsize=10, loc="upper right")

fig.text(0.5, 0.94, "Domination par la possession et le danger (OBV)", ha="center",
         fontsize=22, fontweight="bold", color="black")
st.pyplot(fig, use_container_width=True)

# =======================
# ---- PLOT: xG ---------
# =======================
plot_xg_cumulative(
    data=events,
    team_name=FOCUS_TEAM,
    oppo_name=opposition,
    focus_color=FOCUS_COLOR,
    oppo_color=OPPO_COLOR,
)

# =======================
# ---- NOTES ------------
# =======================
with st.expander("Notes techniques"):
    st.markdown(
        """
- Possession = proxy par passes complétées (bins 5’, lissage gaussien).
- Danger (OBV) = différentiel de `obv_total_net` Pass/Carry/Dribble.
- xG cumulés = escaliers par équipe + lignes cibles 1.69 / 1.07.
- Cache activé sur matches et events. Bouton Refresh pour forcer la MAJ.
- Affichage des matchs non disponibles pour anticiper la publication.
        """
    )
