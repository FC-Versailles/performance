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
        goals = []
        cum = 0.0

        for _, r in df.iterrows():
            m = float(r["minute"]) + float(r["second"]) / 60.0
            xg = float(r["shot_statsbomb_xg"])
            cum += xg
            mins.append(m)
            vals.append(cum)

            if r.get("shot_outcome") == "Goal":
                goals.append((m, cum))

        return mins, vals, goals

    min_team, xg_team, goals_team = build_stairs(team_shots)
    min_oppo, xg_oppo, goals_oppo = build_stairs(oppo_shots)

    fig, ax = plt.subplots(figsize=(12, 6))

    # --- cumulative xG lines ---
    ax.step(min_team, xg_team, where="post", linewidth=3, label=team_name, color=FOCUS_COLOR)
    ax.step(min_oppo, xg_oppo, where="post", linewidth=3, label=oppo_name, color=OPPO_COLOR)

    # --- compute y limits BEFORE placing texts ---
    ymax = max(xg_team[-1] if xg_team else 0, xg_oppo[-1] if xg_oppo else 0) + 0.6
    ax.set_ylim(0, ymax)
    ax.set_xlim(0, 95)

    # --- xG diff every 15 minutes (team_name - oppo_name) ---
    shots15 = events_shots.copy()
    shots15["m"] = (
        pd.to_numeric(shots15["minute"], errors="coerce").fillna(0)
        + pd.to_numeric(shots15["second"], errors="coerce").fillna(0) / 60.0
    )

    if "shot_type" in shots15.columns:
        shots15 = shots15[shots15["shot_type"].ne("Penalty")].copy()

    shots15["bin15"] = (shots15["m"] // 15).astype(int) * 15
    bins_keep = [0, 15, 30, 45, 60, 75, 90]
    shots15 = shots15[shots15["bin15"].isin(bins_keep)].copy()

    g = (
        shots15.groupby(["bin15", "team"])["shot_statsbomb_xg"]
        .sum()
        .unstack("team")
        .fillna(0.0)
    )
    for t in [team_name, oppo_name]:
        if t not in g.columns:
            g[t] = 0.0

    g = g.sort_index()
    diff = g[team_name] - g[oppo_name]

    # --- place ΔxG texts at the correct x positions (bin centers) ---
    y_text = ymax - 0.03 * ymax  # small offset under the top

    for b in g.index:
        val = float(diff.loc[b])
        if not np.isfinite(val):
            continue

        x_pos = b + 7.5  # center of the 15-min window
        txt = f"+{val:.2f}" if val >= 0 else f"{val:.2f}"
        col = FOCUS_COLOR if val >= 0 else OPPO_COLOR

        ax.text(
            x_pos, y_text,
            txt,
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold",
            color=col
        )

    # --- goal markers (font-safe) ---
    for m, y in goals_team:
        ax.scatter([m], [y], s=180, marker="o",
                   edgecolors="white", linewidths=1.5,
                   color=FOCUS_COLOR, zorder=6)

    for m, y in goals_oppo:
        ax.scatter([m], [y], s=180, marker="o",
                   edgecolors="black", linewidths=1.5,
                   color=OPPO_COLOR, zorder=6)

    # --- annotate final values ---
    if len(min_team) > 0:
        ax.text(min_team[-1] + 1, xg_team[-1], f"{xg_team[-1]:.2f}",
                color=FOCUS_COLOR, fontsize=11, va="center", fontweight="bold")
    if len(min_oppo) > 0:
        ax.text(min_oppo[-1] + 1, xg_oppo[-1], f"{xg_oppo[-1]:.2f}",
                color=OPPO_COLOR, fontsize=11, va="center", fontweight="bold")

    # --- styling ---
    ax.set_xticks([0, 15, 30, 45, 60, 75, 90])
    ax.set_xlabel("Minutes")
    ax.set_ylabel("xG cumulés")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color="lightgrey", lw=0.8, linestyle="--", alpha=0.6)
    ax.legend(loc="lower right")

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)



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

    # --- NEW: uplift col for color (Versailles only) ---
    uplift_col = "shot_shot_execution_xg_uplift"
    if uplift_col in shots.columns:
        shots["uplift"] = pd.to_numeric(shots[uplift_col], errors="coerce")
    else:
        shots["uplift"] = np.nan

    # --- NEW: Versailles legend stats (team1 only) ---
    # total shots / "on target" (shot_outcome != "Off T") + inside box + xG per player
    v = shots[shots["team_name"] == team1].copy()
    total_shots = int(len(v))

    on_target = v["outcome_name"].notna() & (v["outcome_name"] != "Off T")
    n_on_target = int(on_target.sum())

    in_box = (v["x"].between(102, 120)) & (v["y"].between(18, 62))
    n_in_box = int(in_box.sum())

    xg_by_player = (
        v.groupby("player_name")["xg"]
        .sum()
        .sort_values(ascending=False)
    )
    # keep legend compact
    top_n = 10
    xg_lines = [f"{p}: {val:.2f}" for p, val in xg_by_player.head(top_n).items()]
    if len(xg_by_player) > top_n:
        xg_lines.append("…")

    legend_text = (
        f"{team1}\n"
        f"Shots: {total_shots} | On target: {n_on_target}\n"
        f"Inside box: {n_in_box}\n"
        f"xG by player:\n" + "\n".join(xg_lines)
    )

    # --- NEW: color mapping for uplift (team1 only) ---
    # If uplift missing, fallback to 0 (neutral color)
    v_uplift = v["uplift"].copy()
    if v_uplift.notna().sum() == 0:
        vmax = 1.0
    else:
        vmax = np.nanpercentile(np.abs(v_uplift.dropna()), 98)
        vmax = float(vmax) if (vmax and np.isfinite(vmax) and vmax > 0) else 1.0

    norm = mpl.colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.get_cmap("RdYlGn")  # red negative, green positive

    # Create pitch canvas
    pitch = Pitch(pitch_type="statsbomb", line_color="black")
    fig, ax = pitch.grid(grid_height=0.92, title_height=0.01, axis=False,
                         endnote_height=0.02, title_space=0, endnote_space=0)
    axp = ax["pitch"]

    # Team1 shots (left→right)  --- CHANGED: color by uplift, goals only annotated ---
    mask_t1 = shots["team_name"] == team1
    df_t1 = shots.loc[mask_t1, ["x","y","outcome_name","player_name","xg","uplift"]]
    for _, r in df_t1.iterrows():
        is_goal = r["outcome_name"] == "Goal"
        s = 600 * float(r["xg"]) if r["xg"] > 0 else 120  # minimum visible size

        u = r["uplift"]
        if pd.isna(u):
            u = 0.0
        col = cmap(norm(float(u)))

        pitch.scatter(
            r["x"], r["y"],
            alpha=1.0 if is_goal else 0.25,
            s=s, color=col,
            edgecolors="white", linewidths=0.7,
            ax=axp, zorder=3
        )
        if is_goal:
            pitch.annotate(str(r["player_name"]), (r["x"]+1, r["y"]-2),
                           ax=axp, fontsize=10, color="black")

    # Team2 shots mirrored to attack left→right (unchanged styling)
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

    # --- NEW: legend box for team1 (Versailles) ---
    axp.text(
        0.7, 0.88,
        legend_text,
        transform=axp.transAxes,
        ha="center", va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="black", alpha=0.9)
    )

    # --- NEW: colorbar for uplift (team1) ---
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axp, fraction=0.03, pad=0.02)
    cbar.set_label("Sous/sur Performance")

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
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    st.pyplot(fig, use_container_width=True)



import matplotlib as mpl

def plot_progressive_threat_obv(
    events: pd.DataFrame,
    team_name: str,
    min_gain_pct: float = 0.25,   # 25%
):
    """
    Progressive actions (Pass, Carry) for `team_name`:
      progressive if distance-to-goal decreases by >= min_gain_pct of starting distance
      goal center = (120,40)

    EXTRA:
      - Exclude actions that FINISH in the last third (x >= 90)
      - Label each arrow with the player name (near the end point)

    Plot:
      - Pass: full arrow
      - Carry: dashed arrow
      - Arrow color = obv_total_net (RdYlGn, red=negative, green=positive)
      - Exclude incomplete passes (pass_outcome not null)
    """
    if "obv_total_net" not in events.columns:
        st.info("OBV column not available.")
        return

    GOAL_X, GOAL_Y = 120.0, 40.0

    def _xy(v):
        if isinstance(v, (list, tuple)) and len(v) >= 2:
            return float(v[0]), float(v[1])
        return np.nan, np.nan

    def dist_to_goal(x, y):
        return np.sqrt((GOAL_X - x) ** 2 + (GOAL_Y - y) ** 2)

    # --- base filter team + types ---
    df = events[
        (events["team"] == team_name) &
        (events["type"].isin(["Pass", "Carry"]))
    ].copy()

    if df.empty:
        st.info("No Pass/Carry events for this team.")
        return

    # --- build start/end coords ---
    df["sx"], df["sy"] = zip(*df["location"].map(_xy))

    df["end_loc"] = np.where(df["type"].eq("Pass"), df["pass_end_location"], df["carry_end_location"])
    df["ex"], df["ey"] = zip(*df["end_loc"].map(_xy))

    # remove invalid coords / OBV
    df = df.dropna(subset=["sx","sy","ex","ey","obv_total_net"])

    # remove incomplete passes
    if "pass_outcome" in df.columns:
        df = df[~((df["type"] == "Pass") & (df["pass_outcome"].notna()))].copy()

    # --- EXCLUDE actions that finish in last third (x >= 90) ---
    df = df[df["ex"] < 90].copy()

    if df.empty:
        st.info("No Pass/Carry events left after excluding actions ending in last third.")
        return

    # --- progressive definition (>= 25% gain toward goal) ---
    d0 = dist_to_goal(df["sx"].to_numpy(), df["sy"].to_numpy())
    d1 = dist_to_goal(df["ex"].to_numpy(), df["ey"].to_numpy())
    gain_pct = (d0 - d1) / np.maximum(d0, 1e-6)
    df = df[gain_pct >= min_gain_pct].copy()

    if df.empty:
        st.info("No progressive (>=25% gain) passes/carries found (after last-third exclusion).")
        return

    # --- player label ---
    # prefer 'player' (events df) else try 'player_name'
    if "player" in df.columns:
        df["player_label"] = df["player"].astype(str)
    elif "player_name" in df.columns:
        df["player_label"] = df["player_name"].astype(str)
    else:
        df["player_label"] = "?"

    # --- color mapping by OBV ---
    vmax = np.nanpercentile(np.abs(df["obv_total_net"]), 98)
    vmax = float(vmax) if (vmax and np.isfinite(vmax)) else 1.0
    norm = mpl.colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.get_cmap("RdYlGn")

    # --- plot pitch ---
    fig, ax = plt.subplots(figsize=(12, 8))
    pitch = Pitch(pitch_type="statsbomb", line_color="black")
    pitch.draw(ax=ax)

    # last-third line for context (x=90)
    ax.axvline(90, color="black", lw=1.5, ls="--", alpha=0.6)

    # --- draw arrows + labels ---
    for _, r in df.iterrows():
        col = cmap(norm(float(r["obv_total_net"])))
        ls = "-" if r["type"] == "Pass" else "--"

        pitch.arrows(
            r["sx"], r["sy"], r["ex"], r["ey"],
            ax=ax, color=col,
            width=2.2, headwidth=4.5, headlength=4.5,
            alpha=0.95, linestyle=ls
        )

        # label near end point (small offset to reduce overlap)
        ax.text(
            r["ex"] + 0.8,
            r["ey"] + 0.8,
            r["player_label"],
            fontsize=8,
            color="black",
            ha="left",
            va="bottom",
            zorder=20
        )

    # colorbar
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)

    n_pass = int((df["type"] == "Pass").sum())
    n_carry = int((df["type"] == "Carry").sum())

    ax.set_title(
        
        f"Passes: {n_pass} | Carries: {n_carry}",
        fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)



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
    shots = shots.dropna(subset=[x_col, y_col, "team", "shot_outcome", "player", "minute"])

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

        # --- NEW: conditional annotations ---
        to_label = sub[
            (sub[x_col] < -0.10) | (sub[y_col] > 0.20)
        ]

        for _, r in to_label.iterrows():
            label = f"{r['player']} {int(r['minute'])}'"
            ax.annotate(
                label,
                (r[x_col], r[y_col]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=7,
                color="black",
                ha="left",
                va="bottom"
            )

    ax.set_xlabel("PSxG (sous / sur-performance)", fontsize=6)
    ax.set_ylabel("xG", fontsize=6)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_position(("data", 0))

    ax.grid(True, color="lightgrey", lw=0.5, alpha=0.6)

    ax.legend(
        frameon=False,
        fontsize=6,
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

# ---------- Graphiques Equipe ----------
def plot_entrees_zone(
    events: pd.DataFrame,
    focus_team: str = "Versailles",
    title: str = "",
    x_min: float = 102, x_max: float = 120,
    y_min: float = 18,  y_max: float = 62,
):
    """
    Generic entries plot (Regular Play + possession_team==focus_team)
    - Pass (complete): black arrow if pass_end_location inside zone
    - Carry: blue arrow if carry_end_location inside zone
    - Arrow start = location
    Legends:
      1) Passes (N) / Conduites (N) with colors
      2) Players: (Total entries) (Pass+Carry)
    """
    def _xy(v):
        if isinstance(v, (list, tuple)) and len(v) >= 2:
            return float(v[0]), float(v[1])
        return np.nan, np.nan

    def _in_zone(x, y):
        return (x >= x_min) and (x <= x_max) and (y >= y_min) and (y <= y_max)

    # --- filters: Regular Play + possession_team == focus_team ---
    df = events.copy()
    df = df[(df.get("play_pattern") == "Regular Play") & (df.get("possession_team") == focus_team)]

    # --- PASS entries ---
    passes = df[df["type"] == "Pass"].copy()
    if "pass_outcome" in passes.columns:
        passes = passes[passes["pass_outcome"].isna()].copy()

    passes["sx"], passes["sy"] = zip(*passes["location"].map(_xy))
    passes["ex"], passes["ey"] = zip(*passes["pass_end_location"].map(_xy))
    passes = passes.dropna(subset=["sx","sy","ex","ey"])
    passes = passes[passes.apply(lambda r: _in_zone(r["ex"], r["ey"]), axis=1)]
    passes["player_name"] = passes.get("player").astype(str)

    # --- CARRY entries ---
    carries = df[df["type"] == "Carry"].copy()
    carries["sx"], carries["sy"] = zip(*carries["location"].map(_xy))
    carries["ex"], carries["ey"] = zip(*carries["carry_end_location"].map(_xy))
    carries = carries.dropna(subset=["sx","sy","ex","ey"])
    carries = carries[carries.apply(lambda r: _in_zone(r["ex"], r["ey"]), axis=1)]
    carries["player_name"] = carries.get("player").astype(str)

    fig, ax = plt.subplots(figsize=(10, 7))
    pitch = Pitch(pitch_type="statsbomb", line_color="black")
    pitch.draw(ax=ax)

    # highlight zone
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                           fill=False, lw=2, ec="black", alpha=0.5))

    # draw arrows
    if not passes.empty:
        pitch.arrows(
            passes["sx"], passes["sy"], passes["ex"], passes["ey"],
            ax=ax, color="black", width=2.0, headwidth=4, headlength=4, alpha=0.9
        )
    if not carries.empty:
        pitch.arrows(
            carries["sx"], carries["sy"], carries["ex"], carries["ey"],
            ax=ax, color=FOCUS_COLOR, width=2.0, headwidth=4, headlength=4, alpha=0.9
        )

    # legends
    from matplotlib.lines import Line2D
    n_pass = int(len(passes))
    n_carry = int(len(carries))

    handles_type = [
        Line2D([0], [0], color="black", lw=3),
        Line2D([0], [0], color=FOCUS_COLOR, lw=3),
    ]
    labels_type = [
        f"Passes vers zone ({n_pass})",
        f"Conduites vers zone ({n_carry})",
    ]
    leg1 = ax.legend(handles_type, labels_type, frameon=True, loc="upper center", fontsize=6)
    ax.add_artist(leg1)

    entries_all = pd.concat(
        [passes[["player_name"]], carries[["player_name"]]],
        ignore_index=True
    )
    if not entries_all.empty:
        counts = (
            entries_all.groupby("player_name")
            .size()
            .sort_values(ascending=False)
        )
        TOP_N = 15
        handles_players = [Line2D([0], [0], color="none") for _ in range(min(TOP_N, len(counts)))]
        labels_players = [f"{p} : ({int(n)})" for p, n in counts.head(TOP_N).items()]

        ax.legend(
            handles_players,
            labels_players,
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(0.15, 0.92),
            fontsize=6,

        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def plot_entrees_surface(events: pd.DataFrame, focus_team: str = "Versailles"):
    return plot_entrees_zone(
        events,
        focus_team=focus_team,
        x_min=102, x_max=120, y_min=18, y_max=62
    )


def plot_entrees_last_third(events: pd.DataFrame, focus_team: str = "Versailles"):
    """
    Entrées dans le dernier tiers (x 90→120, y 0→80)
    - Regular Play + possession_team == Versailles
    - Pass (complete): black arrows
    - Carry: blue arrows
    - If start location already in last third -> SAME arrow but colored YELLOW (both pass & carry)
    - REMOVE actions that FINISH in the surface/box (x 102→120, y 18→62)

    Legends:
      1) Passes (outside->in) / Conduites (outside->in) + Actions (inside->inside)
      2) Players: (Total entries)
    """
    # last third
    x_min, x_max = 90, 120
    y_min, y_max = 0, 80

    # penalty box / surface area (StatsBomb)
    box_x_min, box_x_max = 102, 120
    box_y_min, box_y_max = 18, 62

    def _xy(v):
        if isinstance(v, (list, tuple)) and len(v) >= 2:
            return float(v[0]), float(v[1])
        return np.nan, np.nan

    def _in_last_third(x, y):
        return (x_min <= x <= x_max) and (y_min <= y <= y_max)

    def _in_box(x, y):
        return (box_x_min <= x <= box_x_max) and (box_y_min <= y <= box_y_max)

    # --- base filters ---
    df = events.copy()
    df = df[(df.get("play_pattern") == "Regular Play") & (df.get("possession_team") == focus_team)]

    # --- PASS entries ---
    passes = df[df["type"] == "Pass"].copy()
    if "pass_outcome" in passes.columns:
        passes = passes[passes["pass_outcome"].isna()].copy()

    passes["sx"], passes["sy"] = zip(*passes["location"].map(_xy))
    passes["ex"], passes["ey"] = zip(*passes["pass_end_location"].map(_xy))
    passes = passes.dropna(subset=["sx", "sy", "ex", "ey"])

    # keep only actions that END in last third
    passes = passes[passes.apply(lambda r: _in_last_third(r["ex"], r["ey"]), axis=1)]
    # remove actions that END in the box
    passes = passes[~passes.apply(lambda r: _in_box(r["ex"], r["ey"]), axis=1)]

    passes["start_in_zone"] = passes.apply(lambda r: _in_last_third(r["sx"], r["sy"]), axis=1)
    passes["player_name"] = passes.get("player").astype(str)

    # --- CARRY entries ---
    carries = df[df["type"] == "Carry"].copy()
    carries["sx"], carries["sy"] = zip(*carries["location"].map(_xy))
    carries["ex"], carries["ey"] = zip(*carries["carry_end_location"].map(_xy))
    carries = carries.dropna(subset=["sx", "sy", "ex", "ey"])

    carries = carries[carries.apply(lambda r: _in_last_third(r["ex"], r["ey"]), axis=1)]
    carries = carries[~carries.apply(lambda r: _in_box(r["ex"], r["ey"]), axis=1)]

    carries["start_in_zone"] = carries.apply(lambda r: _in_last_third(r["sx"], r["sy"]), axis=1)
    carries["player_name"] = carries.get("player").astype(str)

    # split by start location
    pass_out_in  = passes[~passes["start_in_zone"]].copy()
    pass_in_in   = passes[ passes["start_in_zone"]].copy()
    carry_out_in = carries[~carries["start_in_zone"]].copy()
    carry_in_in  = carries[ carries["start_in_zone"]].copy()

    fig, ax = plt.subplots(figsize=(10, 7))
    pitch = Pitch(pitch_type="statsbomb", line_color="black")
    pitch.draw(ax=ax)

    # highlight last third
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                           fill=False, lw=1, ec="black", alpha=0.5))

    # draw arrows
    if not pass_out_in.empty:
        pitch.arrows(pass_out_in["sx"], pass_out_in["sy"], pass_out_in["ex"], pass_out_in["ey"],
                     ax=ax, color="black", width=2.0, headwidth=4, headlength=4, alpha=0.9)
    if not carry_out_in.empty:
        pitch.arrows(carry_out_in["sx"], carry_out_in["sy"], carry_out_in["ex"], carry_out_in["ey"],
                     ax=ax, color=FOCUS_COLOR, width=2.0, headwidth=4, headlength=4, alpha=0.9)

    # inside->inside = YELLOW
    if not pass_in_in.empty:
        pitch.arrows(pass_in_in["sx"], pass_in_in["sy"], pass_in_in["ex"], pass_in_in["ey"],
                     ax=ax, color="#FFD400", width=2.0, headwidth=4, headlength=4, alpha=0.95)
    if not carry_in_in.empty:
        pitch.arrows(carry_in_in["sx"], carry_in_in["sy"], carry_in_in["ex"], carry_in_in["ey"],
                     ax=ax, color="#FFD400", width=2.0, headwidth=4, headlength=4, alpha=0.95)

    # legends
    from matplotlib.lines import Line2D
    handles_type = [
        Line2D([0], [0], color="black", lw=3),
        Line2D([0], [0], color=FOCUS_COLOR, lw=3),
        Line2D([0], [0], color="#FFD400", lw=3),
    ]
    labels_type = [
        f"Passes vers dernier tiers ({len(pass_out_in)})",
        f"Conduites vers dernier tiers ({len(carry_out_in)})",
        f"Actions dans dernier tiers ({len(pass_in_in) + len(carry_in_in)})",
    ]
    leg1 = ax.legend(handles_type, labels_type, frameon=True, loc="upper center", fontsize=6)
    ax.add_artist(leg1)

    # players legend (total entries, pass+carry)
    entries_all = pd.concat([passes[["player_name"]], carries[["player_name"]]], ignore_index=True)
    if not entries_all.empty:
        counts = entries_all.groupby("player_name").size().sort_values(ascending=False)
        TOP_N = 15
        handles_players = [Line2D([0], [0], color="none") for _ in range(min(TOP_N, len(counts)))]
        labels_players = [f"{p} : ({int(n)})" for p, n in counts.head(TOP_N).items()]
        ax.legend(
            handles_players,
            labels_players,
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(0.15, 0.92),
            fontsize=6,
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

 




def plot_possession_versailles_only(events: pd.DataFrame, focus_team: str, opposition: str):
    """
    Plot only Versailles possession %:
      - Bars for H1 0-15, H1 15-30, H1 30-45, H2 0-15, H2 15-30, H2 30-45
      - Uses duration + possession_team
      - Text: possession % for full game + by half
    """
    needed = {"duration", "possession_team", "period", "minute"}
    miss = [c for c in needed if c not in events.columns]
    if miss:
        st.info(f"Missing columns for possession time: {miss}")
        return

    df = events.copy()
    df = df[df["possession_team"].isin([focus_team, opposition])].copy()

    df["dur_s"] = pd.to_numeric(df["duration"], errors="coerce").fillna(0.0).clip(lower=0)
    df["minute"] = pd.to_numeric(df["minute"], errors="coerce").fillna(0).astype(int)
    df["period"] = pd.to_numeric(df["period"], errors="coerce").fillna(0).astype(int)

    # minute within half
    base_min = df.groupby("period")["minute"].transform("min")
    df["min_in_half"] = (df["minute"] - base_min).clip(lower=0)

    # 15-min bins inside each half
    df["bin15"] = (df["min_in_half"] // 15) * 15
    df = df[df["bin15"].isin([0, 15, 30])].copy()

    # --- % possession full match + halves ---
    total_game = df.groupby("possession_team")["dur_s"].sum()
    game_total = float(total_game.sum())
    v_game = float(total_game.get(focus_team, 0.0))
    v_game_pct = 100 * v_game / game_total if game_total > 0 else np.nan

    half_team = df.groupby(["period", "possession_team"])["dur_s"].sum().unstack("possession_team").fillna(0.0)
    for t in [focus_team, opposition]:
        if t not in half_team.columns:
            half_team[t] = 0.0
    half_team["total"] = half_team[focus_team] + half_team[opposition]
    v_h1_pct = 100 * float(half_team.loc[1, focus_team]) / float(half_team.loc[1, "total"]) if (1 in half_team.index and half_team.loc[1, "total"] > 0) else np.nan
    v_h2_pct = 100 * float(half_team.loc[2, focus_team]) / float(half_team.loc[2, "total"]) if (2 in half_team.index and half_team.loc[2, "total"] > 0) else np.nan

    # --- segment possession (Versailles % per 15-min bin) ---
    seg = (
        df.groupby(["period", "bin15", "possession_team"])["dur_s"]
        .sum()
        .unstack("possession_team")
        .fillna(0.0)
        .reset_index()
        .sort_values(["period", "bin15"])
    )
    for t in [focus_team, opposition]:
        if t not in seg.columns:
            seg[t] = 0.0
    seg["total"] = seg[focus_team] + seg[opposition]
    seg["v_pct"] = 100 * seg[focus_team] / seg["total"].replace(0, np.nan)

    seg["label"] = seg["period"].map({1: "H1", 2: "H2"}).fillna("P?") + " " + seg["bin15"].map({
        0: "0–15", 15: "15–30", 30: "30–45"
    })

    # --- text summary ---
    st.markdown(
        f"**Possession FCV :** "
        f"{v_game_pct:.1f}% sur le match | "
        f"H1: {v_h1_pct:.1f}% & H2: {v_h2_pct:.1f}%"
    )

    # --- plot (only Versailles possession %) ---
    # --- plot (only Versailles possession %) ---
    fig, ax = plt.subplots(figsize=(12, 4))
    
    x = np.arange(len(seg))
    values = seg["v_pct"].to_numpy()
    
    bars = ax.bar(x, values, color="#0031E3")
    
    # labels on bars
    for bar in bars:
        height = bar.get_height()
        if np.isnan(height):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(height + 1, 99),     # avoid going above 100
            f"{height:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="black"
        )
    
    ax.set_xticks(x)
    ax.set_xticklabels(seg["label"])
    ax.set_ylim(0, 100)
    ax.set_ylabel("% possession")
    
    # 50% benchmark (correct for possession)
    ax.axhline(y=50, ls="--", lw=0.5, color="black", alpha=0.1)
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    




# ---------- Onglets principaux ----------
tab_equipe, tab_joueurs, tab_kpi = st.tabs(["Match", "Joueurs", "KPI"])

with tab_equipe:

    # Flux (deux mi-temps sur une figure)
    fig = plt.figure(figsize=(24, 8))
    fig.set_facecolor(BGCOLOR)
    gs = fig.add_gridspec(nrows=1, ncols=2, wspace=0.15)
    ax1 = fig.add_subplot(gs[0]); game_flow_ax(events, 1, ax1, FOCUS_TEAM, opposition)
    ax1.set_ylabel("1ère mi-temps", fontsize=12); ax1.legend(fontsize=9, loc="upper right")
    ax2 = fig.add_subplot(gs[1]); game_flow_ax(events, 2, ax2, FOCUS_TEAM, opposition)
    ax2.set_ylabel("2ème mi-temps", fontsize=12); ax2.legend(fontsize=9, loc="upper right")
    st.markdown("### Domination : Possession & Danger")
    st.pyplot(fig, use_container_width=True)
    
    st.markdown("### Possession")
    plot_possession_versailles_only(events, focus_team=FOCUS_TEAM, opposition=opposition)


    # xG cumulés
    st.markdown("### xG cumulés")
    plot_xg_cumulative(events, FOCUS_TEAM, opposition)

    # Shot map
    st.markdown("### Shot Map")
    plot_shot_map_combined(events, FOCUS_TEAM, opposition)

    st.markdown("### Entrées dans la surface")
    plot_entrees_surface(events, focus_team=FOCUS_TEAM)
    
    st.markdown("### Jeu dans le dernier tiers")
    plot_entrees_last_third(events, focus_team=FOCUS_TEAM)

    
    st.markdown("### Menace Progressive - >25%")
    plot_progressive_threat_obv(events, team_name=FOCUS_TEAM, min_gain_pct=0.25)

    st.markdown("### Performance des tirs")
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


def plot_kpi_zscore_benchmark(
    statsbomb_df: pd.DataFrame,
    match_id: int,
    focus_team: str = "Versailles",
):
    """
    Benchmark plot:
      - y = KPI name
      - x = z-score across all matches for `focus_team`
      - grey dots = other matches
      - big blue dot = selected match
      - label on big dot = raw KPI value for that match
    """

    if statsbomb_df is None or statsbomb_df.empty:
        st.info("KPI benchmark: statsbomb_df is empty.")
        return None

    df = statsbomb_df.copy()

    needed = {"match_id", "team_name"}
    if not needed.issubset(df.columns):
        st.info(f"KPI benchmark: missing columns {needed - set(df.columns)}.")
        return None

    df = df[df["team_name"] == focus_team].copy()
    if df.empty:
        st.info("KPI benchmark: no rows for focus team.")
        return None

    KPI_MAP = [
        ("xG", "team_match_np_xg"),
        ("Corner xG", "team_match_corner_xg"),
        ("xG / tir", "team_match_np_xg_per_shot"),
        ("OBV Shot", "team_match_obv_shot"),
      
        ("Tirs", "team_match_np_shots"),
        ("Tirs contre-attaque", "team_match_counter_attacking_shots"),
       
        ("Passes réussies", "team_match_successful_passes"),
        ("Cente réussis", "team_match_successful_crosses_into_box"),
    
        ("Entrées surface", "team_match_touches_in_box"),       
        ("Actions réussies 20m", "team_match_deep_completions"),        
        ("Deep progressions", "team_match_deep_progressions"),        
        ("OBV", "team_match_obv"),
        ("Dribbles réussi", "team_match_completed_dribbles"),        
        
        ("xG concédés", "team_match_np_xg_conceded"),      
        ("PPDA", "team_match_ppda"),    
        ("Pression Porteur", "team_match_aggression"),      
        ("Pressing moitié Adv", "team_match_fhalf_pressures"),
        ("Ballons gagnés", "team_match_defensive_action_regains"),
       
    ]

    # keep only KPIs that exist
    KPI_MAP = [(lab, col) for lab, col in KPI_MAP if col in df.columns]
    if not KPI_MAP:
        st.info("KPI benchmark: none of the KPI columns exist in statsbomb_df.")
        return None

    # long format
    rows = []
    for lab, col in KPI_MAP:
        s = pd.to_numeric(df[col], errors="coerce")
        z = _zscore(s)
        tmp = pd.DataFrame(
            {"match_id": df["match_id"], "kpi_label": lab, "value": s, "z": z}
        )
        rows.append(tmp)

    plot_df = pd.concat(rows, ignore_index=True).dropna(subset=["z"])
    plot_df["is_this"] = plot_df["match_id"].astype(int).eq(int(match_id))

    # y order (top to bottom)
    kpi_order = [lab for lab, _ in KPI_MAP]
    plot_df["kpi_label"] = pd.Categorical(plot_df["kpi_label"], categories=kpi_order, ordered=True)

    y_map = {lab: i for i, lab in enumerate(kpi_order)}
    plot_df["y"] = plot_df["kpi_label"].map(y_map).astype(float)

    # jitter for grey points only
    rng = np.random.default_rng(42)
    plot_df["y_jit"] = plot_df["y"] + np.where(
        plot_df["is_this"], 0.0, rng.normal(0, 0.08, size=len(plot_df))
    )

    fig, ax = plt.subplots(figsize=(10, 0.55 * len(kpi_order) + 1.5), facecolor="white")

    # other matches
    others = plot_df[~plot_df["is_this"]]
    ax.scatter(
        others["z"], others["y_jit"],
        s=25, alpha=0.35, color="grey", edgecolor="none", zorder=1
    )

    # selected match
    thism = plot_df[plot_df["is_this"]]
    ax.scatter(
        thism["z"], thism["y"],
        s=220, alpha=1.0, color="#0031E3",
        edgecolor="white", linewidth=1.5, zorder=3
    )

    # label selected match with raw value
    for _, r in thism.iterrows():
        if pd.isna(r["value"]):
            continue
        val = float(r["value"])
        ax.text(
            float(r["z"]) + 0.08, float(r["y"]),
            f"{val:.2f}",
            va="center", ha="left",
            fontsize=9, color="black", fontweight="bold", zorder=4
        )

    ax.axvline(0, c="black", ls="--", lw=1.2, alpha=0.6)
    ax.set_yticks(range(len(kpi_order)))
    ax.set_yticklabels(kpi_order, fontsize=6)
    ax.invert_yaxis()

    ax.set_xlabel("Z-score (vs saison)", fontsize=6)
    ax.set_xlim(-3.2, 3.2)
    ax.grid(axis="x", color="lightgrey", lw=0.6, alpha=0.6)

    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)

    plt.tight_layout()
    return fig

    
    
with tab_kpi:
    render_kpi_pyramid_from_teamstats(statsbomb_df, match_id, focus_team=FOCUS_TEAM)

    st.markdown("### KPI Match vs Season")
    fig_kpi = plot_kpi_zscore_benchmark(statsbomb_df, match_id, focus_team=FOCUS_TEAM)
    if fig_kpi is not None:
        st.pyplot(fig_kpi, use_container_width=True)
        plt.close(fig_kpi)