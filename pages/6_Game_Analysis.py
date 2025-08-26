#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 21:11:52 2025

@author: fcvmathieu
"""
#events.to_csv('events.csv')
# app.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.ndimage import gaussian_filter1d
from statsbombpy import sb
from scipy.ndimage import gaussian_filter

# Optionnel: mplsoccer pour les tracés joueurs
from mplsoccer import VerticalPitch
from mplsoccer import Pitch

# ---------- Paramètres ----------
st.set_page_config(layout="wide")
FOCUS_TEAM = "Versailles"
COMP_ID = 129
SEASON_ID = 318

# Couleurs
BGCOLOR = "white"
FOCUS_COLOR = "#0031E3"
OPPO_COLOR = "grey"
BAR_COLOR = "black"

# Identifiants SB
USER = os.getenv("SB_USER", "mathieu.feigean@fcversailles.com")
PASSWORD = os.getenv("SB_PASS", "uVBxDK5X")
CREDS = {"user": USER, "passwd": PASSWORD}

# ---------- En-tête ----------
col1, col2 = st.columns([9, 1])
with col1:
    st.title("Game Analysis | FC Versailles")
with col2:
    st.image(
        "https://raw.githubusercontent.com/FC-Versailles/wellness/main/logo.png",
        use_container_width=True,
    )
st.markdown("<hr style='border:1px solid #ddd' />", unsafe_allow_html=True)

# ---------- Helpers SB ----------
@st.cache_data(show_spinner=False)
def load_matches(comp_id: int, season_id: int, creds: dict) -> pd.DataFrame:
    m = sb.matches(competition_id=comp_id, season_id=season_id, creds=creds)
    if m.empty:
        return m
    m = m[(m["home_team"] == FOCUS_TEAM) | (m["away_team"] == FOCUS_TEAM)]
    m = m[m["match_status"] == "available"].copy()
    m["match_date"] = pd.to_datetime(m["match_date"]).dt.date

    def opponent(row):
        return row["away_team"] if row["home_team"] == FOCUS_TEAM else row["home_team"]

    def score_label(row):
        hs = int(row["home_score"]); as_ = int(row["away_score"])
        return f"{row['home_team']} {hs}–{as_} {row['away_team']}"

    m["opponent"] = m.apply(opponent, axis=1)
    m["label"] = m.apply(score_label, axis=1)
    m = m.sort_values(by=["match_date", "match_id"], ascending=[False, False]).reset_index(drop=True)
    return m

@st.cache_data(show_spinner=False)
def load_events(match_id: int, creds: dict) -> pd.DataFrame:
    ev = sb.events(match_id=match_id, creds=creds)
    for c in ["period", "minute", "second"]:
        if c in ev.columns:
            ev[c] = pd.to_numeric(ev[c], errors="coerce")
    return ev

# ---------- SB events -> df1 for player plots ----------
def sb_events_to_df1(ev: pd.DataFrame) -> pd.DataFrame:
    def _xy(loc):
        if isinstance(loc, (list, tuple)) and len(loc) >= 2:
            return float(loc[0]), float(loc[1])
        return np.nan, np.nan

    def _end_xy(row):
        t = row.get("type")
        if t == "Pass":
            loc = row.get("pass_end_location")
        elif t == "Carry":
            loc = row.get("carry_end_location")
        else:
            loc = row.get("location")
        return _xy(loc)

    df = ev.copy()
    out = pd.DataFrame({
        "event_type_name": df["type"].astype(str),
        "team_name": df["team"].astype(str),
        "player_name": df["player"].astype(str),
        "obv_total_net": df.get("obv_total_net", np.nan),
    })
    out["location_x"], out["location_y"] = zip(*df["location"].map(_xy))
    ex, ey = zip(*df.apply(_end_xy, axis=1))
    out["end_location_x"], out["end_location_y"] = ex, ey

    # align with your original labels
    out["event_type_name"] = out["event_type_name"].replace({
        "Carry": "Carries",
        "Shot": "shot",
    })
    return out





def group_to_5min(minute_series: pd.Series) -> pd.Series:
    x = pd.to_numeric(minute_series, errors="coerce").fillna(0)
    return ((x // 5) + 1) * 5

# ---------- Graphiques Equipe ----------
def game_flow_ax(data: pd.DataFrame, period: int, ax, focus_team: str, opposition: str):
    df = data[data["period"] == period].copy()
    if df.empty:
        ax.set_facecolor(BGCOLOR)
        ax.set_ylim(-30, 30)
        ax.set_yticklabels([])
        ax.grid(True, color="lightgrey", lw=1, alpha=0.6)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        return 0.0

    df = df.sort_values(["minute", "second"]).reset_index(drop=True)
    df["grouped_min"] = group_to_5min(df["minute"])

    # Possession proxy
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

    # OBV diff par bins
    obv = df.dropna(subset=["obv_total_net"]) if "obv_total_net" in df.columns else pd.DataFrame()
    if not obv.empty:
        obv = obv[obv["type"].isin(["Pass", "Carry", "Dribble"])].copy()
        teamOBV = obv[obv["team"] == focus_team].groupby("grouped_min")["obv_total_net"].sum()
        oppoOBV = obv[obv["team"] == opposition].groupby("grouped_min")["obv_total_net"].sum()
        matchobv = pd.concat([teamOBV.rename("team"), oppoOBV.rename("oppo")], axis=1).fillna(0).reset_index()
        timelistOBV = matchobv["grouped_min"].tolist()
        teamOBV_list = gaussian_filter1d(matchobv["team"].to_numpy(), sigma=1)
        oppoOBV_list = gaussian_filter1d(matchobv["oppo"].to_numpy(), sigma=1)
        obv_diff = (teamOBV_list - oppoOBV_list) * 50  # scaling visuel
    else:
        timelistOBV, obv_diff = [], []

    # Buts
    team_goals = df[((df["team"] == focus_team) & (df["shot_outcome"] == "Goal")) | ((df["type"] == "Own Goal For") & (df["team"] == focus_team))]
    oppo_goals = df[((df["team"] == opposition) & (df["shot_outcome"] == "Goal")) | ((df["type"] == "Own Goal For") & (df["team"] == opposition))]

    ax.set_facecolor(BGCOLOR)
    ax.plot(timelist, poss_diff, "lightgrey")
    y_pos = (np.asarray(poss_diff) + 1e-7) > 0
    y_neg = (np.asarray(poss_diff) - 1e-7) < 0
    ax.fill_between(timelist, poss_diff, where=y_pos, interpolate=True, color=FOCUS_COLOR, alpha=0.95, label=f"{focus_team} possession")
    ax.fill_between(timelist, poss_diff, where=y_neg, interpolate=True, color=OPPO_COLOR, alpha=0.95, label=f"{opposition} possession")

    if len(timelistOBV) and len(obv_diff):
        ax.bar(timelistOBV, obv_diff, width=3, color=BAR_COLOR, alpha=0.95, zorder=10, label="Danger (OBV diff)")

    if not team_goals.empty:
        ax.scatter(team_goals["minute"], np.zeros(len(team_goals))+20, color=FOCUS_COLOR, edgecolor="white", linewidth=2.0, marker="o", s=300, zorder=12, label=f"{focus_team} buts")
    if not oppo_goals.empty:
        ax.scatter(oppo_goals["minute"], np.zeros(len(oppo_goals))-20, color=OPPO_COLOR, edgecolor="black", linewidth=2.0, marker="o", s=300, zorder=12, label=f"{opposition} buts")

    ax.grid(True, color="grey", lw=1, alpha=0.5)
    ax.set_ylim(-30, 30)
    ax.set_yticklabels([])
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)

def plot_xg_cumulative(data: pd.DataFrame, team_name: str, oppo_name: str):
    events_shots = data[data["type"] == "Shot"].copy()
    team_shots = events_shots[events_shots["team"] == team_name]
    oppo_shots = events_shots[events_shots["team"] == oppo_name]

    def build_stairs(df):
        mins = [0.0]; vals = [0.0]
        for _, r in df.iterrows():
            m = float(r["minute"]) + float(r["second"]) / 60.0
            mins.append(m); vals.append(float(r["shot_statsbomb_xg"]) + vals[-1])
        return mins, vals

    min_team, xg_team = build_stairs(team_shots)
    min_oppo, xg_oppo = build_stairs(oppo_shots)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot cumulative xG lines
    ax.step(min_team, xg_team, where="post", linewidth=3, label=team_name, color=FOCUS_COLOR)
    ax.step(min_oppo, xg_oppo, where="post", linewidth=3, label=oppo_name, color=OPPO_COLOR)

    # Annotate final values
    if len(min_team) > 0:
        ax.text(min_team[-1]+1, xg_team[-1], f"{xg_team[-1]:.2f}", 
                color=FOCUS_COLOR, fontsize=11, va="center", fontweight="bold")
    if len(min_oppo) > 0:
        ax.text(min_oppo[-1]+1, xg_oppo[-1], f"{xg_oppo[-1]:.2f}", 
                color=OPPO_COLOR, fontsize=11, va="center", fontweight="bold")

    # Style
    ymax = max(xg_team[-1] if xg_team else 0, xg_oppo[-1] if xg_oppo else 0) + 0.5
    ax.set_ylim(0, ymax)
    ax.set_xlim(0, 95)

    ax.set_xticks([0, 15, 30, 45, 60, 75, 90])
    ax.set_xlabel("Minute")
    ax.set_ylabel("xG cumulés")

    # Beautify
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color="lightgrey", lw=0.8, linestyle="--", alpha=0.6)

    ax.legend(loc="upper right")
    st.pyplot(fig, use_container_width=True)

def plot_shot_map_combined(events: pd.DataFrame, team1: str, team2: str):
    # Build (x,y, outcome, player) from SB events
    shots = events[events["type"] == "Shot"].copy()
    shots["x"] = shots["location"].apply(lambda v: v[0] if isinstance(v, (list, tuple)) and len(v) >= 2 else np.nan)
    shots["y"] = shots["location"].apply(lambda v: v[1] if isinstance(v, (list, tuple)) and len(v) >= 2 else np.nan)
    shots = shots.dropna(subset=["x", "y"])
    shots["team_name"] = shots["team"]
    shots["player_name"] = shots["player"]
    shots["outcome_name"] = shots.get("shot_outcome", np.nan)
    shots["xg"] = shots.get("shot_statsbomb_xg", 0.0).fillna(0.0)

    # Create pitch canvas
    pitch = Pitch(pitch_type="statsbomb", line_color="black")
    fig, ax = pitch.grid(grid_height=0.92, title_height=0.06, axis=False,
                         endnote_height=0.02, title_space=0, endnote_space=0)
    axp = ax["pitch"]

    # Team1 shots (left→right)
    mask_t1 = shots["team_name"] == team1
    df_t1 = shots.loc[mask_t1, ["x","y","outcome_name","player_name","xg"]]
    for _, r in df_t1.iterrows():
        is_goal = r["outcome_name"] == "Goal"
        s = 600 * float(r["xg"]) if r["xg"] > 0 else 120  # minimum visible size
        pitch.scatter(r["x"], r["y"], alpha=1.0 if is_goal else 0.25,
                      s=s, color=FOCUS_COLOR, edgecolors="white", linewidths=0.7, ax=axp, zorder=3)
        if is_goal:
            pitch.annotate(str(r["player_name"]), (r["x"]+1, r["y"]-2), ax=axp, fontsize=10, color=FOCUS_COLOR)

    # Team2 shots mirrored to attack left→right
    mask_t2 = shots["team_name"] == team2
    df_t2 = shots.loc[mask_t2, ["x","y","outcome_name","player_name","xg"]]
    for _, r in df_t2.iterrows():
        mx, my = 120 - r["x"], 80 - r["y"]
        is_goal = r["outcome_name"] == "Goal"
        s = 600 * float(r["xg"]) if r["xg"] > 0 else 120
        pitch.scatter(mx, my, alpha=1.0 if is_goal else 0.25,
                      s=s, color=OPPO_COLOR, edgecolors="black", linewidths=0.7, ax=axp, zorder=3)
        if is_goal:
            pitch.annotate(str(r["player_name"]), (mx+1, my-2), ax=axp, fontsize=10, color=OPPO_COLOR)

    fig.suptitle(f"{team1} vs {team2}", fontsize=18, y=0.98)
    st.pyplot(fig, use_container_width=True)


def plot_obv_zones(events, team_name, oppo_name, bins_x=12, bins_y=8,
                   min_actions=6, smooth_sigma=0.0, vlim_pct=98, mode="diff"):
    """
    mode: 'diff' (team - opp) | 'for' | 'against' | 'per_action'
    """
    if "obv_total_net" not in events.columns:
        st.info("OBV non disponible.")
        return

    ev = events[events["type"].isin(["Pass","Carry","Dribble"])].copy()

    def _end_xy(r):
        if r["type"] == "Pass":  p = r.get("pass_end_location")
        elif r["type"] == "Carry": p = r.get("carry_end_location")
        else: p = r.get("location")
        return (p[0], p[1]) if isinstance(p, (list,tuple)) and len(p)>=2 else (np.nan,np.nan)

    ev["x_end"], ev["y_end"] = zip(*ev.apply(_end_xy, axis=1))
    ev = ev.dropna(subset=["x_end","y_end","obv_total_net"])

    def _grid(df):
        H, xe, ye = np.histogram2d(df["x_end"], df["y_end"],
                                   bins=[bins_x,bins_y],
                                   range=[[0,120],[0,80]],
                                   weights=df["obv_total_net"])
        C, _, _ = np.histogram2d(df["x_end"], df["y_end"],
                                 bins=[bins_x,bins_y],
                                 range=[[0,120],[0,80]])
        return H.T, C.T  # transpose for display

    H_for,  C_for  = _grid(ev[ev["team"]==team_name])
    H_opp,  C_opp  = _grid(ev[ev["team"]==oppo_name])

    if mode == "for":
        Z, C = H_for, C_for
        title = f"OBV for • {team_name}"
    elif mode == "against":
        Z, C = H_opp, C_opp
        title = f"OBV against • {team_name}"
    elif mode == "per_action":
        Z = np.divide(H_for - H_opp, (C_for + C_opp), out=np.zeros_like(H_for), where=(C_for+C_opp)>0)
        C = C_for + C_opp
        title = f"OBV diff par action • {team_name}"
    else:  # 'diff'
        Z, C = H_for - H_opp, C_for + C_opp
        title = f"Zones OBV {team_name} – {oppo_name})"

    # filter low-volume cells
    Z = np.where(C >= min_actions, Z, 0.0)

    # optional smoothing
    if smooth_sigma > 0:
        Z = gaussian_filter(Z, sigma=smooth_sigma)

    # robust color scale
    vmax = np.nanpercentile(np.abs(Z), vlim_pct) or 1.0

    fig, ax = plt.subplots(figsize=(12, 6))
    pitch = Pitch(pitch_type='statsbomb', line_color='black')
    pitch.draw(ax=ax)
    im = ax.imshow(np.flipud(Z), extent=[0,120,0,80], aspect="auto",
                   cmap="RdYlGn", vmin=-vmax, vmax=vmax, alpha=0.8)

    # thin grid
    ax.set_xticks(np.linspace(0,120,bins_x+1)); ax.set_yticks(np.linspace(0,80,bins_y+1))
    ax.grid(True, which="both", color="lightgrey", lw=0.5, alpha=0.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("OBV (vert = +  FVC)")

    # annotate top hot/cold cells
    k = 4
    flat = Z.flatten()
    idx_hot  = flat.argsort()[-k:][::-1]
    idx_cold = flat.argsort()[:k]
    gx = np.linspace(0,120,bins_x+1); gy = np.linspace(0,80,bins_y+1)

    def _annot(i, color):
        r, c = divmod(i, Z.shape[1])
        # flipud in display → row index maps from top
        r_disp = Z.shape[0]-1-r
        x0, x1 = gx[c], gx[c+1]; y0, y1 = gy[r_disp], gy[r_disp+1]
        ax.add_patch(plt.Rectangle((x0,y0), x1-x0, y1-y0, fill=False, ec=color, lw=1.5))
    for i in idx_hot:  _annot(i, "green")
    for i in idx_cold: _annot(i, "red")

    ax.set_title(title)
    st.pyplot(fig, use_container_width=True)

@st.cache_data(show_spinner=False)
def aggregate_team_matches(comp_id: int, season_id: int, focus_team: str, creds: dict) -> pd.DataFrame:
    matches = sb.matches(competition_id=comp_id, season_id=season_id, creds=creds)
    matches = matches[matches.match_status == "available"].copy()
    out_rows = []
    for _, m in matches.iterrows():
        mid = int(m["match_id"])
        ev = sb.events(match_id=mid, creds=creds)

        home = m["home_team"]; away = m["away_team"]
        date = pd.to_datetime(m["match_date"]).date()
        hs = int(m["home_score"]); as_ = int(m["away_score"])
        match_label = f"{home} {hs}–{as_} {away}"

        shots = ev[ev["type"] == "Shot"].copy()
        for t in [home, away]:
            opp = away if t == home else home
            sub = shots[shots["team"] == t]
            xg = sub["shot_statsbomb_xg"].fillna(0).sum()
            op_oppo = shots[(shots["team"] == opp) & (shots["play_pattern"] == "Regular Play")]["shot_statsbomb_xg"].fillna(0).sum()
            xga = shots[shots["team"] == opp]["shot_statsbomb_xg"].fillna(0).sum()

            if t == home: goals, goals_conc = hs, as_
            else: goals, goals_conc = as_, hs

            has_obv = "obv_total_net" in ev.columns
            if has_obv:
                obv_team = ev[(ev["team"] == t) & (ev["type"].isin(["Pass", "Carry", "Dribble"]))]["obv_total_net"].fillna(0).sum()
                obv_opp  = ev[(ev["team"] == opp) & (ev["type"].isin(["Pass", "Carry", "Dribble"]))]["obv_total_net"].fillna(0).sum()
            else:
                obv_team, obv_opp = np.nan, np.nan

            passes = ev[(ev["type"] == "Pass") & (ev["pass_outcome"].isna())].copy()
            poss_t = len(passes[passes["team"] == t]); poss_o = len(passes[passes["team"] == opp])
            possession = 100.0 * poss_t / max(1, poss_t + poss_o)

            row = {
                "Match_ID": mid, "Match": match_label, "Date": date,
                "Home": home, "Away": away, "Team": t, "Opponent": opp,
                "Home/Away": "Home" if t == home else "Away",
                "Goals": goals, "Goals Conceded": goals_conc, "GD": goals - goals_conc,
                "xG": xg, "xGA": xga, "xGD": xg - xga,
                "Possession": possession,
                "OBV": obv_team, "OBV Against": obv_opp, "OBV Difference": obv_team - obv_opp
            }
            out_rows.append(row)

    dfm = pd.DataFrame(out_rows).sort_values("Date").reset_index(drop=True)
    condW = dfm["GD"] > 0; condL = dfm["GD"] < 0
    dfm["Result"] = np.select([condW, condL], ["W", "L"], default="D")
    dfm["Pts"] = np.select([condW, condL], [3, 0], default=1)
    return dfm

# ---------- Sélection match ----------
matches = load_matches(COMP_ID, SEASON_ID, CREDS)
if matches.empty:
    st.error("Aucun match disponible.")
    st.stop()

with st.sidebar:
    st.header("Filtres")
    opponents = sorted(matches["opponent"].unique().tolist())
    sel_oppo = st.selectbox("Adversaire", opponents, index=0)
    m_oppo = matches[matches["opponent"] == sel_oppo].copy()
    dates_for_oppo = sorted(m_oppo["match_date"].unique().tolist(), reverse=True)
    sel_date = st.selectbox("Date du match", dates_for_oppo, index=0, format_func=lambda d: d.strftime("%Y-%m-%d"))

m_pick = m_oppo[m_oppo["match_date"] == sel_date].copy()
if m_pick.empty:
    st.warning("Aucun match trouvé pour cet adversaire et cette date.")
    st.stop()

if len(m_pick) > 1:
    labels = [f"{r['label']}  (id={r['match_id']})" for _, r in m_pick.iterrows()]
    chosen_label = st.selectbox("Sélection du match exact", labels, index=0)
    match_id = int(chosen_label.split("id=")[-1].strip(")"))
    match_row = m_pick[m_pick["match_id"] == match_id].iloc[0]
else:
    match_row = m_pick.iloc[0]
    match_id = int(match_row["match_id"])

home_team = match_row["home_team"]; away_team = match_row["away_team"]
home_score = int(match_row["home_score"]); away_score = int(match_row["away_score"])
opposition = match_row["opponent"]
events = load_events(match_id, CREDS)
if events.empty:
    st.error("Pas d'événements pour ce match.")
    st.stop()

# ---------- Onglets principaux ----------
tab_equipe, tab_joueurs = st.tabs(["Equipe", "Joueurs"])

with tab_equipe:

    # Flux (deux mi-temps sur une figure)
    fig = plt.figure(figsize=(24, 8))
    fig.set_facecolor(BGCOLOR)
    gs = fig.add_gridspec(nrows=1, ncols=2, wspace=0.15)
    ax1 = fig.add_subplot(gs[0]); game_flow_ax(events, 1, ax1, FOCUS_TEAM, opposition)
    ax1.set_ylabel("1ère mi-temps", fontsize=12); ax1.legend(fontsize=9, loc="upper right")
    ax2 = fig.add_subplot(gs[1]); game_flow_ax(events, 2, ax2, FOCUS_TEAM, opposition)
    ax2.set_ylabel("2ème mi-temps", fontsize=12); ax2.legend(fontsize=9, loc="upper right")
    fig.text(0.5, 0.96, "Domination: possession relative + danger (OBV)", ha="center", fontsize=18, fontweight="bold", color="black")
    st.pyplot(fig, use_container_width=True)

    # xG cumulés
    st.markdown("### xG cumulés")
    plot_xg_cumulative(events, FOCUS_TEAM, opposition)

    # Shot map
    st.markdown("### Shot Map")
    plot_shot_map_combined(events, FOCUS_TEAM, opposition)

    # Zones OBV
    st.markdown("### Zones OBV")
    plot_obv_zones(events, FOCUS_TEAM, opposition, bins_x=12, bins_y=8)


# ---------- Onglet Joueurs : intégration de ton code adapté ----------
def draw_players_ball_receipts(df1: pd.DataFrame, df2: pd.DataFrame, team_name: str = "Versailles"):
    df1 = df1.copy()
    df1.dropna(subset=["player_name"], inplace=True)
    players = df1.loc[df1["team_name"] == team_name, "player_name"].dropna().unique().tolist()
    if not players:
        st.info("Aucun joueur trouvé pour l'équipe sélectionnée.")
        return

    n = len(players)
    ncols = 6
    nrows = int(np.ceil(n / ncols))

    pitch = VerticalPitch(line_zorder=2, line_color="black", linewidth=1, pitch_type="statsbomb")
    fig, axes = pitch.draw(nrows=nrows, ncols=ncols, figsize=(18, 3*nrows+2))
    axes = np.atleast_2d(axes)

    for i, player_name in enumerate(players):
        p1 = df1[df1["player_name"] == player_name]
        p2 = df2[df2["player_name"] == player_name] if df2 is not None else pd.DataFrame()

        pos = p1[p1["event_type_name"].isin(["Ball Receipt*", "Pass", "Duel"])]
        miss = p1[p1["event_type_name"].isin(["Miscontrol", "Dispossessed"])]

        r = i // ncols
        c = i % ncols
        ax = axes[r, c]

        # KDE si dispo
        try:
            pitch.kdeplot(pos["location_x"], pos["location_y"], ax=ax, cmap="Blues", fill=True, levels=100)
        except Exception:
            pass
        pitch.scatter(pos["location_x"], pos["location_y"], alpha=1, s=12, color="black", ax=ax)
        pitch.scatter(miss["location_x"], miss["location_y"], alpha=1, s=12, color="red", ax=ax)
        ax.set_title(player_name, fontsize=11, fontweight="bold")

    fig.suptitle("Ballons touchés — Versailles", fontsize=18, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

def draw_players_pass_carry_dribble(df1: pd.DataFrame, team_name: str = "Versailles"):
    sub = df1[df1["team_name"] == team_name].copy()
    players = sub["player_name"].dropna().unique().tolist()
    if not players:
        st.info("Aucun joueur trouvé pour l'équipe sélectionnée.")
        return

    n = len(players)
    ncols = 6
    nrows = int(np.ceil(n / ncols))

    pitch = VerticalPitch(line_color="black", linewidth=1)
    fig, axes = pitch.draw(nrows=nrows, ncols=ncols, figsize=(18, 3*nrows+2))
    axes = np.atleast_2d(axes)

    for i, player_name in enumerate(players):
        dfp = sub[sub["player_name"] == player_name]
        df_pass = dfp[dfp["event_type_name"] == "Pass"][["location_x","location_y","end_location_x","end_location_y"]]
        df_carry = dfp[dfp["event_type_name"] == "Carries"][["location_x","location_y","end_location_x","end_location_y"]]
        df_drib = dfp[dfp["event_type_name"] == "Dribble"][["location_x","location_y"]]

        ax = axes.flat[i]
        # Passes
        for _, row in df_pass.iterrows():
            pitch.arrows(row["location_x"], row["location_y"],
                         row["end_location_x"], row["end_location_y"],
                         width=1.5, headwidth=3, headlength=3, color="grey", ax=ax)
        # Carries
        for _, row in df_carry.iterrows():
            pitch.arrows(row["location_x"], row["location_y"],
                         row["end_location_x"], row["end_location_y"],
                         width=1.5, headwidth=3, headlength=3, linestyle="--", color="black", ax=ax)
        # Dribbles
        pitch.scatter(df_drib["location_x"], df_drib["location_y"], alpha=0.9, s=20, color="blue", ax=ax)

        ax.axhline(y=80, xmin=0, xmax=100, color="grey", linestyle="--")
        ax.set_title(player_name, fontsize=11, fontweight="bold")

    fig.suptitle("Passes • Conduites • Dribbles — Versailles", fontsize=18, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

def draw_players_obv_bar(df1: pd.DataFrame, team_name: str = "Versailles"):
    cols_needed = {"timestamp","event_type_name","team_name","player_name","location_x","location_y","obv_total_net"}
    if not cols_needed.issubset(set(df1.columns)):
        st.info("Colonnes OBV manquantes pour le barplot (obv_total_net...).")
        return

    pos = df1[df1["event_type_name"].isin(["Carries","Pass","shot"])].copy()
    pos = pos[pos["team_name"] == team_name]
    if pos.empty:
        st.info("Aucune action pour OBV joueur.")
        return

    total_net = pos.groupby("player_name")["obv_total_net"].agg(list).reset_index()
    total_net["pos"] = total_net["obv_total_net"].apply(lambda v: sum(x for x in v if x >= 0))
    total_net["neg"] = total_net["obv_total_net"].apply(lambda v: sum(x for x in v if x < 0))
    total_net = total_net[["player_name","neg","pos"]].sort_values("pos", ascending=False)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(total_net["player_name"], total_net["neg"], color="red", edgecolor="black", linewidth=0.7)
    ax.bar(total_net["player_name"], total_net["pos"], bottom=total_net["neg"], color="green", edgecolor="black", linewidth=0.7)
    ax.set_title("Efficience ballon au pied (OBV + / -)", fontsize=18, fontweight="bold")
    ax.set_xlabel(""); ax.set_ylabel("OBV net")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig, use_container_width=True)

# ---------- Tab Joueurs (API-only) ----------
with tab_joueurs:
    df1 = sb_events_to_df1(events).dropna(subset=["player_name"])
    players = df1.loc[df1["team_name"] == FOCUS_TEAM, "player_name"].unique()

    # 1) Ballons touchés / pertes (3x6)
    nrows, ncols = 3, 6
    pitch = VerticalPitch(line_zorder=2, line_alpha=0.5, goal_alpha=0.3)
    fig, axes = pitch.draw(nrows=nrows, ncols=ncols, figsize=(16, 12))

    for i, player_name in enumerate(players):
        if i >= nrows * ncols: break
        row, col = divmod(i, ncols)
        ax = axes[row, col]
        p = df1[df1["player_name"] == player_name]
        pos  = p[p["event_type_name"].isin(["Ball Receipt*", "Pass", "Duel"])]
        miss = p[p["event_type_name"].isin(["Miscontrol", "Dispossessed"])]

        pch = VerticalPitch(line_zorder=2, line_color='black', linewidth=1, pitch_type='statsbomb')
        try:
            pch.kdeplot(pos.location_x, pos.location_y, ax=ax, cmap='Blues', fill=True, levels=100)
        except Exception:
            pass
        pch.scatter(pos.location_x,  pos.location_y,  s=20, color="black", ax=ax)
        pch.scatter(miss.location_x, miss.location_y, s=20, color="red",   ax=ax)
        ax.set_title(player_name, fontsize=11, fontweight='bold')

    fig.suptitle("Ballons touchés — Versailles Players", fontsize=18, fontweight="bold")
    plt.subplots_adjust(hspace=0.5, wspace=0.3)
    st.pyplot(fig, use_container_width=True)

    # 2) Passes / Carries / Dribbles (3x6)
    pitch = VerticalPitch(line_zorder=2, line_alpha=0.5, goal_alpha=0.3)
    fig, axes = pitch.draw(nrows=nrows, ncols=ncols, figsize=(16, 12))
    axes = np.asarray(axes).flatten()

    for i, player_name in enumerate(players):
        if i >= nrows * ncols: break
        dfp = df1[(df1["player_name"] == player_name) & (df1["team_name"] == FOCUS_TEAM)]
        df_pass  = dfp[dfp["event_type_name"] == "Pass"][["location_x","location_y","end_location_x","end_location_y"]]
        df_carry = dfp[dfp["event_type_name"] == "Carries"][["location_x","location_y","end_location_x","end_location_y"]]
        df_drib  = dfp[dfp["event_type_name"] == "Dribble"][["location_x","location_y"]]

        ax = axes[i]
        pch = VerticalPitch(line_color='black', linewidth=1)

        for _, r in df_pass.iterrows():
            inside = (0 <= r.end_location_x <= 100) and (0 <= r.end_location_y <= 80)
            color = "grey" if inside else "tan"
            pch.arrows(r.location_x, r.location_y, r.end_location_x, r.end_location_y,
                       width=1.5, headwidth=3, headlength=3, color=color, ax=ax)

        for _, r in df_carry.iterrows():
            pch.arrows(r.location_x, r.location_y, r.end_location_x, r.end_location_y,
                       width=1.5, headwidth=3, headlength=3, linestyle="--", color="black", ax=ax)

        pch.scatter(df_drib.location_x, df_drib.location_y, s=30, color="blue", ax=ax)
        ax.axhline(y=80, color='grey', linestyle='--')
        ax.set_title(player_name, fontsize=11, fontweight='bold')

    fig.suptitle("Passes • Carries • Dribbles", fontsize=18, fontweight="bold")
    st.pyplot(fig, use_container_width=True)

    # 3) OBV ± par joueur
    pos_df = df1[
        (df1["team_name"] == FOCUS_TEAM) &
        (df1["event_type_name"].isin(["Carries","Pass","shot"]))
    ].copy()

    neg_total = pos_df[pos_df["obv_total_net"] < 0].groupby('player_name')["obv_total_net"].sum()
    pos_total = pos_df[pos_df["obv_total_net"] >= 0].groupby('player_name')["obv_total_net"].sum()
    total_net = pd.concat([neg_total, pos_total], axis=1).fillna(0)
    total_net.columns = ['obv_total_net_neg','obv_total_net_pos']
    total_net = total_net.sort_values('obv_total_net_pos', ascending=False)

    fig, ax = plt.subplots(figsize=(16, 9), facecolor="white")
    total_net.plot(kind='bar', ax=ax, color=['red', 'green'], ec="black", lw=1, width=0.8)
    ax.set_title('Efficience Ballon au pied (OBV ±)', fontsize=18, fontweight="bold")
    ax.set_xlabel(''); ax.set_ylabel('Niveau de Menace')
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
