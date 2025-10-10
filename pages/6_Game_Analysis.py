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
from scipy.stats import gaussian_kde

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

@st.cache_data(show_spinner=False)
def load_team_match_stats(comp_id: int, season_id: int, focus_team: str, creds: dict) -> pd.DataFrame:
    m = sb.matches(competition_id=comp_id, season_id=season_id, creds=creds)
    m = m[m["match_status"] == "available"]
    m = m[(m["home_team"] == focus_team) | (m["away_team"] == focus_team)]
    if m.empty:
        return pd.DataFrame()
    dfs = []
    for mid in m["match_id"].unique():
        tms = sb.team_match_stats(int(mid), creds=creds)
        dfs.append(tms)
    df = pd.concat(dfs, ignore_index=True)
    # Colonnes dérivées pour cohérence avec team_analysis
    df["Goals_Diff"] = df["team_match_goals"] - df["team_match_goals_conceded"]
    df["Pts"] = df["Goals_Diff"].apply(lambda gd: 3 if gd > 0 else (1 if gd == 0 else 0))
    return df

statsbomb_df = load_team_match_stats(COMP_ID, SEASON_ID, FOCUS_TEAM, CREDS)



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



def plot_obv_kde(events: pd.DataFrame, team_name: str, bw: float = 0.3):
    if "obv_total_net" not in events.columns:
        st.info("OBV column not available.")
        return

    df = events[
        (events["team"] == team_name)
        & (events["type"].isin(["Pass", "Carry"]))   # <-- filter here
    ].copy()

    if df.empty:
        st.info("No OBV events for this team.")
        return

    def _loc(r):
        loc = r.get("location")
        return loc if isinstance(loc, (list, tuple)) and len(loc) >= 2 else [np.nan, np.nan]

    df[["x", "y"]] = df.apply(_loc, axis=1, result_type="expand")
    df = df.dropna(subset=["x", "y", "obv_total_net"])

    xs = np.linspace(0, 120, 300)
    ys = np.linspace(0, 80, 200)
    X, Y = np.meshgrid(xs, ys)

    pos = df[df["obv_total_net"] >= 0]
    neg = df[df["obv_total_net"] < 0]

    Z_pos = np.zeros_like(X)
    Z_neg = np.zeros_like(X)

    if not pos.empty:
        kde_pos = gaussian_kde(np.vstack([pos.x, pos.y]),
                               weights=pos["obv_total_net"], bw_method=bw)
        Z_pos = kde_pos(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

    if not neg.empty:
        kde_neg = gaussian_kde(np.vstack([neg.x, neg.y]),
                               weights=-neg["obv_total_net"], bw_method=bw)
        Z_neg = kde_neg(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

    Z = Z_pos - Z_neg
    vmax = np.nanpercentile(np.abs(Z), 98) or 1.0

    fig, ax = plt.subplots(figsize=(10, 6))
    Pitch(pitch_type="statsbomb", line_color="black").draw(ax=ax)

    im = ax.imshow(
        Z, extent=[0, 120, 0, 80], origin="lower",
        cmap="RdYlGn", vmin=-vmax, vmax=vmax, alpha=0.85, aspect="auto"
    )
    plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02).set_label("OBV (green = +)")
    ax.set_title(f"OBV KDE • {team_name}", fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
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

def plot_shot_scatter(events: pd.DataFrame, team1: str, team2: str):
    shots = events[events["type"] == "Shot"].copy()
    if shots.empty:
        st.info("No shots for this match.")
        return

    x_col = "shot_shot_execution_xg_uplift"
    y_col = "shot_statsbomb_xg"
    shots = shots.dropna(subset=[x_col, y_col, "team", "shot_outcome"])

    fig, ax = plt.subplots(figsize=(9, 7), facecolor="white")

    # background highlight
    ax.axvspan(0.20, shots[x_col].max() + 0.05, facecolor="#F9E79F", alpha=0.3)
    ax.axhspan(0.20, shots[y_col].max() + 0.05, facecolor="#F9E79F", alpha=0.3)

    for team, color in [(team1, FOCUS_COLOR), (team2, OPPO_COLOR)]:
        sub = shots[shots["team"] == team]
        if sub.empty:
            continue
        goals = sub[sub["shot_outcome"] == "Goal"]
        others = sub[sub["shot_outcome"] != "Goal"]

        ax.scatter(
            others[x_col], others[y_col],
            s=60, c=color, alpha=0.7, edgecolor="white", label=f"{team} (no goal)"
        )
        ax.scatter(
            goals[x_col], goals[y_col],
            s=90, c=color, marker="s", edgecolor="black", lw=0.8, label=f"{team} goals"
        )

    ax.set_xlabel("PsxG", fontsize=12)
    ax.set_ylabel("xG", fontsize=12)
    ax.set_title("Shots: PsxG vs xG", fontsize=14, weight="bold")

    # axis style
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_position(("data", 0))

    ax.grid(True, color="lightgrey", lw=0.5, alpha=0.6)

    ax.legend(
        frameon=False,
        fontsize=9,
        loc="lower right",
        bbox_to_anchor=(1, 0)
    )

    st.pyplot(fig, use_container_width=True)
    
# === KPI Pyramid (requires a statsbomb_df avec colonnes team_match_*) ===
KPI_TARGETS = {
    "Goals_Diff": 0.99,
    "team_match_goals": 1.5,
    "team_match_np_xg": 1.25,
    "team_match_passes_inside_box": 4,
    "team_match_deep_progressions": 40,
    "team_match_obv": 1.6,
    "team_match_obv_shot": 0.01,
    "team_match_goals_conceded": 0.8,          # lower is better
    "team_match_np_xg_conceded": 0.90,         # lower is better
    "team_match_ppda": 8,                      # lower is better
    "team_match_aggression": 0.24,
    "team_match_fhalf_pressures": 75,
    "team_match_obv_shot_nconceded": 0.5
}

def _kpi_check(row, targets: dict):
    out = {}
    for kpi, tgt in targets.items():
        if ("conceded" in kpi) or (kpi == "team_match_ppda"):
            out[kpi] = row.get(kpi, np.nan) <= tgt
        else:
            out[kpi] = row.get(kpi, np.nan) >= tgt
    return out

def _draw_kpi_pyramid(vals: dict, kpis_ok: dict):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    S = 1.0
    H = S * np.sqrt(3) / 2.0

    def tri_up_base(cx, y_base, s=S):
        h = s * np.sqrt(3) / 2.0
        return [(cx - s/2, y_base), (cx + s/2, y_base), (cx, y_base + h)]

    def tri_down_top(cx, y_top, s=S):
        h = s * np.sqrt(3) / 2.0
        return [(cx - s/2, y_top), (cx + s/2, y_top), (cx, y_top - h)]

    EDGE_W = 0.8
    EDGE = "#111"
    TEXT = dict(ha="center", va="center", color="#0031E3", fontsize=9, fontweight="bold")

    def fmt(x, n=2, as_int=False):
        if x is None:
            return "-"
        if isinstance(x, float):
            if np.isnan(x) or np.isinf(x):
                return "-"
        return f"{int(x)}" if as_int else f"{x:.{n}f}"

    def col_color(k, default="lightgrey"):
        if k is None:
            return "white"
        ok = kpis_ok.get(k, None)
        if ok is None:
            return default
        return "#CFB013" if ok else "#B1B4B2"

    def add_triangle(ax, coords, face, label=None, value=None):
        ax.add_patch(Polygon(coords, closed=True, facecolor=face, edgecolor=EDGE, linewidth=EDGE_W, joinstyle="round"))
        if label is not None:
            cx = sum(p[0] for p in coords) / 3.0
            cy = sum(p[1] for p in coords) / 3.0
            ax.text(cx, cy, f"{label}" if value is None else f"{label}\n{value}", **TEXT)

    def centers_for_row(oris):
        xs = [0.0]
        for i in range(1, len(oris)):
            dx = S if oris[i-1] == oris[i] else (S/2.0)
            xs.append(xs[-1] + dx)
        off = (xs[0] + xs[-1]) / 2.0
        return [x - off for x in xs]

    row1 = [("Pts", None, "up", "pts")]
    ori1 = ["up"]
    row2 = [("Goals","team_match_goals","up","goals"),
            ("GD","Goals_Diff","down","gd"),
            ("GC","team_match_goals_conceded","up","goals_conceded")]
    ori2 = ["up","down","up"]
    row3 = [("xG","team_match_np_xg","up","xg"),
            ("OBV Shot","team_match_obv_shot","down","obv_shot"),
            ("OBV Shot C","team_match_obv_shot_nconceded","down","obv_shot_conceded"),
            ("xGC","team_match_np_xg_conceded","up","xg_conceded")]
    ori3 = ["up","down","down","up"]
    row4 = [("PIB","team_match_passes_inside_box","up","pib"),
            ("Deep P","team_match_deep_progressions","down","deep"),
            ("OBV","team_match_obv","up","obv"),
            (None,None,"down",None),
            ("PPDA","team_match_ppda","up","ppda"),
            ("Agg","team_match_aggression","down","aggression"),
            ("Press","team_match_fhalf_pressures","up","pressures")]
    ori4 = ["up","down","up","down","up","down","up"]

    xs_r1, xs_r2, xs_r3, xs_r4 = map(centers_for_row, [ori1, ori2, ori3, ori4])

    y_top1  = 3*H; y_base1 = y_top1 - H
    y_top2  = y_base1; y_base2 = y_top2 - H
    y_top3  = y_base2; y_base3 = y_top3 - H
    y_top4  = y_base3; y_base4 = y_top4 - H

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.axis("off")

    label_x = min(xs_r4) - S*1.2
    labkw = dict(ha="right", va="center", color="#0031E3", fontsize=8, fontweight="bold")
    ax.text(label_x, (y_top1+y_base1)/2, "Points", **labkw)
    ax.text(label_x, (y_top2+y_base2)/2, "Buts", **labkw)
    ax.text(label_x, (y_top3+y_base3)/2, "Occasions créées\net concédées", **labkw)
    ax.text(label_x, (y_top4+y_base4)/2, "Modèle de jeu\net Performance", **labkw)

    (lab,kpi,ori,vkey) = row1[0]
    add_triangle(ax, tri_up_base(xs_r1[0], y_base1), "white", lab, fmt(vals[vkey], as_int=True))

    for (cx,(lab,kpi,ori,vkey)) in zip(xs_r2, row2):
        coords = tri_up_base(cx, y_base2) if ori=="up" else tri_down_top(cx, y_top2)
        face = col_color(kpi)
        val = fmt(vals[vkey], as_int=True) if vkey in ("goals","gd","goals_conceded") else fmt(vals[vkey])
        add_triangle(ax, coords, face, lab, val)

    for (cx,(lab,kpi,ori,vkey)) in zip(xs_r3, row3):
        coords = tri_up_base(cx, y_base3) if ori=="up" else tri_down_top(cx, y_top3)
        face = col_color(kpi)
        add_triangle(ax, coords, face, lab, fmt(vals[vkey]))

    for (cx,(lab,kpi,ori,vkey)) in zip(xs_r4, row4):
        coords = tri_up_base(cx, y_base4) if ori=="up" else tri_down_top(cx, y_top4)
        face = "white" if (lab is None and kpi is None) else col_color(kpi)
        val = fmt(vals[vkey], as_int=True) if vkey in ("pib","deep","pressures") else (fmt(vals[vkey]) if vkey else None)
        add_triangle(ax, coords, face, lab, val)

    x_min, x_max = min(xs_r4)-S, max(xs_r4)+S
    y_min, y_max = y_base4 - 0.1*H, y_top1 + 0.1*H
    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max); ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    return fig

def render_kpi_pyramid_from_teamstats(statsbomb_df: pd.DataFrame, match_id: int, focus_team: str = "Versailles"):
    if statsbomb_df.empty:
        st.info("KPI pyramid: aucune donnée team_match_stats.")
        return

    row = statsbomb_df[(statsbomb_df["team_name"] == focus_team) &
                       (statsbomb_df["match_id"] == match_id)]
    if row.empty:
        st.info("KPI pyramid: match non trouvé pour l'équipe.")
        return
    row = row.iloc[0].to_dict()

    # tolérance de nom pour OBV shot concédé
    obv_shot_nc = row.get("team_match_obv_shot_nconceded", row.get("team_match_obv_shot", np.nan))

    vals = {
        "pts": row.get("Pts", np.nan),
        "gd": row.get("Goals_Diff", np.nan),
        "goals": row.get("team_match_goals", np.nan),
        "goals_conceded": row.get("team_match_goals_conceded", np.nan),
        "xg": row.get("team_match_np_xg", np.nan),
        "obv": row.get("team_match_obv", np.nan),
        "obv_shot": row.get("team_match_obv_shot", np.nan),
        "xg_conceded": row.get("team_match_np_xg_conceded", np.nan),
        "obv_shot_conceded": obv_shot_nc,
        "pib": row.get("team_match_passes_inside_box", np.nan),
        "deep": row.get("team_match_deep_progressions", np.nan),
        "ppda": row.get("team_match_ppda", np.nan),
        "aggression": row.get("team_match_aggression", np.nan),
        "pressures": row.get("team_match_fhalf_pressures", np.nan),
    }

    # validation KPI
    kpis_ok = _kpi_check(row, KPI_TARGETS)

    fig = _draw_kpi_pyramid(vals, kpis_ok)
    st.markdown("### Pyramide des KPI")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)



# ---------- Onglets principaux ----------
tab_equipe, tab_joueurs, tab_kpi = st.tabs(["Equipe", "Joueurs", "KPI"])

with tab_equipe:
    render_kpi_pyramid_from_teamstats(statsbomb_df, match_id, focus_team=FOCUS_TEAM)



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
    st.markdown("### OBV Heatmap (Versailles)")
    plot_obv_kde(events, FOCUS_TEAM, bw=0.3)

    st.markdown("### Shot Scatter (xG uplift vs StatsBomb xG)")
    plot_shot_scatter(events, FOCUS_TEAM, opposition)


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
    

# --- KPI helpers (safe) ------------------------------------------------------
def _zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    mu = s.mean()
    sd = s.std(ddof=0)
    if pd.isna(sd) or sd == 0:
        return pd.Series([0.0] * len(s), index=series.index)
    return (s - mu) / sd

def group_by_count(count_df: pd.DataFrame) -> pd.DataFrame:
    if count_df.empty:
        return pd.DataFrame(columns=["team","match_id","value","plot"])
    out = count_df.groupby(["team","match_id"]).size().reset_index(name="value")
    out["plot"] = _zscore(out["value"])
    return out

def group_by_metric(metric_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if metric_df.empty or metric not in metric_df.columns:
        return pd.DataFrame(columns=["team","match_id","value","plot"])
    out = (
        metric_df.groupby(["team","match_id"])[metric]
        .sum().reset_index(name="value")
    )
    out["plot"] = _zscore(out["value"])
    return out

# --- KPI point plotter -------------------------------------------------------
def plot_variable(team: str, plot_data: pd.DataFrame, y_value: float, game_id: int, ax):
    if plot_data.empty:
        return
    is_this = plot_data["match_id"] == game_id
    # visuals
    color_team = FOCUS_COLOR if team == FOCUS_TEAM else OPPO_COLOR
    color_other = OPPO_COLOR if team == FOCUS_TEAM else FOCUS_COLOR

    plot_data = plot_data.copy()
    plot_data["alpha"] = np.where(is_this, 1.0, 0.2)
    plot_data["ec"]    = np.where(is_this, color_team, color_other)
    plot_data["size"]  = np.where(is_this, 5000.0, 500.0)
    plot_data["color"] = np.where(is_this, color_other, color_team)

    x = plot_data["plot"].to_numpy()
    y = np.full_like(x, y_value, dtype=float)
    ax.scatter(x, y, s=plot_data["size"], alpha=plot_data["alpha"],
               c=plot_data["color"], ec=plot_data["ec"], lw=5, zorder=10)

    this_row = plot_data[is_this]
    if not this_row.empty:
        val = float(this_row.iloc[0]["value"])
        x_pos = float(this_row.iloc[0]["plot"])
        ax.annotate(f"{val:.2f}", (x_pos, y_value - 0.05),
                    color="white", fontsize=22, fontweight="bold", ha="center", zorder=11)

    
with tab_kpi:
    st.markdown("## 📊 KPI (benchmark match vs saison)")

    df = events  # already loaded for match_id

    def plot_team_data_simple(team: str, df_events: pd.DataFrame, game_id: int):
        fig, ax = plt.subplots(figsize=(12, 8))
        # Shots For (exclude penalties if column exists)
        shots_for = df_events[(df_events["type"] == "Shot") & (df_events["team"] == team)].copy()
        if "shot_type" in shots_for.columns:
            shots_for = shots_for[shots_for["shot_type"].ne("Penalty")]
        shots_df = group_by_count(shots_for)
        plot_variable(team, shots_df, 0, game_id, ax)

        # xG For
        xg_for = group_by_metric(shots_for, "shot_statsbomb_xg")
        plot_variable(team, xg_for, 1, game_id, ax)

        # Pressures
        pressures = df_events[(df_events["type"] == "Pressure") & (df_events["team"] == team)].copy()
        pressures_df = group_by_count(pressures)
        plot_variable(team, pressures_df, 2, game_id, ax)

        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["Shots", "xG", "Pressures"], fontsize=12)
        ax.axvline(0, c="black", ls="--", lw=2)
        ax.set_xlim(-3, 3)
        ax.set_title(f"KPI Match vs saison — {team}")
        for s in ["top", "right", "left", "bottom"]:
            ax.spines[s].set_visible(False)
        return fig

    fig_kpi = plot_team_data_simple(FOCUS_TEAM, df, match_id)
    st.pyplot(fig_kpi, use_container_width=True)
