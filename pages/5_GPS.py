import os
import pickle
import streamlit as st
import pandas as pd
import plotly.express as px
import streamlit.components.v1 as components
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
import io
from reportlab.lib.pagesizes import landscape, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Image, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.styles import ParagraphStyle
import requests
import re
from bs4 import BeautifulSoup
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import matplotlib
cmap = matplotlib.cm.get_cmap('RdYlGn_r')
cmaps = matplotlib.cm.get_cmap('RdYlGn')
cmapa = matplotlib.cm.get_cmap('Greens')  # ou RdYlGn pour du rouge/vert
import matplotlib.colors as mcolors
import plotly.graph_objects as go
from scipy.stats import zscore

# ── Constants ───────────────────────────────────────"──────────────────────────

st.set_page_config(layout='wide')
col1, col2 = st.columns([9,1])
with col1:
    st.title("GPS | FC Versailles")
with col2:
    st.image(
        'https://raw.githubusercontent.com/FC-Versailles/wellness/main/logo.png',
        use_container_width=True
    )
st.markdown("<hr style='border:1px solid #ddd' />", unsafe_allow_html=True)

# ── Fetch & cache data ────────────────────────────────────────────────────────
SCOPES         = ['https://www.googleapis.com/auth/spreadsheets.readonly']
TOKEN_FILE_GPS     = 'token._gps.pickle'
SPREADSHEET_ID_GPS = '1NfaLx6Yn09xoOHRon9ri6zfXZTkU1dFFX2rfW1kZvmw'
SHEET_NAME     = 'Feuille 1'
RANGE_NAME = 'Feuille 1!A1:AC'

def get_credentials():
    creds = None
    if os.path.exists(TOKEN_FILE_GPS):
        with open(TOKEN_FILE_GPS, 'rb') as f:
            creds = pickle.load(f)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'client_secret.json', SCOPES
            )
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE_GPS, 'wb') as f:
            pickle.dump(creds, f)
    return creds

def fetch_google_sheet(spreadsheet_id, sheet_name):
    creds   = get_credentials()
    service = build('sheets','v4',credentials=creds)
    meta    = service.spreadsheets().get(
        spreadsheetId=spreadsheet_id, includeGridData=True
    ).execute()
    for s in meta['sheets']:
        if s['properties']['title'] == sheet_name:
            data = s['data'][0]['rowData']
            break
    else:
        st.error(f"❌ Feuille {sheet_name} introuvable.")
        return pd.DataFrame()
    rows = []
    for row in data:
        rows.append([cell.get('formattedValue') for cell in row.get('values',[])])
    max_len = max(len(r) for r in rows)
    rows = [r + [None]*(max_len-len(r)) for r in rows]
    header = rows[0]
    return pd.DataFrame(rows[1:], columns=header)

@st.cache_data(ttl=60)
def load_data():
    creds = get_credentials()
    service = build('sheets', 'v4', credentials=creds)

    # === Use the fast values().get() endpoint ===
    result = (
        service.spreadsheets()
               .values()
               .get(spreadsheetId=SPREADSHEET_ID_GPS,
                    range=RANGE_NAME,
                    valueRenderOption='FORMATTED_VALUE')
               .execute()
    )
    rows = result.get('values', [])
    if not rows:
        st.error("❌ Aucune donnée trouvée dans la plage.")
        return pd.DataFrame()

    # first row = header, rest = data
    header, data_rows = rows[0], rows[1:]
    df = pd.DataFrame(data_rows, columns=header)

    # keep only your 24 columns
    expected = [
        "Season","Semaine","HUMEUR","PLAISIR","RPE","ERPE","Date","AMPM","Jour","Type","Ttotal","Teffectif","Name",
        "Duration","Distance","M/min","Distance 15km/h","M/min 15km/h",
        "Distance 15-20km/h","Distance 20-25km/h","Distance 25km/h",
        "Distance 90% Vmax","N° Sprints","Vmax","%Vmax","Acc","Dec","Amax","Dmax"
    ]
    df = df.loc[:, expected]

    # hard-code season
    df = df[df["Season"] == "2526"]

    # downstream processing...
    return df

data = load_data()
data = data[data["Name"] != "BAGHDADI"]

# ── Pre-process common cols ────────────────────────────────────────────────────
# Filter by season

# Duration → int (invalid → 0)
if "Duration" in data.columns:
    # 1) coerce to float (invalid → NaN)
    durations = pd.to_numeric(
        data["Duration"]
            .astype(str)
            .str.replace(",", ".", regex=False),
        errors="coerce"
    )
    # 2) replace NaN with 0 and cast to plain int
    data["Duration"] = durations.fillna(0).astype(int)

# Type → uppercase & stripped
if "Type" in data.columns:
    data["Type"] = data["Type"].astype(str).str.upper().str.strip()

# Name → title-case
if "Name" in data.columns:
    data["Name"] = (
        data["Name"].astype(str)
                 .str.strip()
                 .str.lower()
                 .str.title()
    )

# Semaine → integer
if "Semaine" in data.columns:
    data["Semaine"] = pd.to_numeric(data["Semaine"], errors="coerce").astype("Int64")

# Date → datetime
if "Date" in data.columns:
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    
player_positions = {
    "ADEHOUMI":   "PIS",
    "BEN BRAHIM": "ATT",
    "BENHADDOUD": "M",
    "CALVET":     "DC",
    "CHADET":     "PIS",
    "CISSE":      "DC",
    "DOUCOURE":   "ATT",
    "FISCHER":    "PIS",
    "GAVAL":      "ATT",
    "GUILLAUME":  "ATT",
    "KALAI":      "ATT",
    "MBONE":      "DC",
    "MOUSSADEK":  "DC",
    "OUCHEN":     "M",
    "RENAUD":     "M",
    "SANTINI":    "PIS",
    "TCHATO":     "DC",
    "ZEMOURA":    "ATT",
    "BASQUE":     "M",
    "KOUASSI":    "M",
    "ODZOUMO":    "ATT",
    "TRAORE":     "M",
    "KOFFI":      "ATT",
    "TLILI":      "ATT"
}

# ── Sidebar: page selection ───────────────────────────────────────────────────
pages = ["Entrainement","Match","Best performance","Joueurs","Minutes de jeu"]
page  = st.sidebar.selectbox("Choisissez une page", pages)


# ── PAGE: BEST PERFORMANCE ────────────────────────────────────────────────────
if page == "Best performance":
    st.subheader("🏅 Meilleures performances")

    # === 1) Best-per-game (min > 50) for four core metrics
    cols = ["M/min", "Vmax", "Amax", "Dmax"]
    best_df = data[(data["Type"] == "GAME") & (data["Duration"] > 45)].copy()
    for c in cols:
        if c in best_df.columns:
            best_df[c] = pd.to_numeric(
                best_df[c].astype(str).str.replace(",", "."), errors="coerce"
            )
    best = (
        best_df
        .groupby("Name")[cols]
        .max()
        .reset_index()
        .sort_values("M/min", ascending=False)
    )
    st.dataframe(best, use_container_width=True)

    # Top 3 and Flop 3 for these
    def top_n(df, col, n=3):
        return df.nlargest(n, col)["Name"].tolist()

    def flop_n(df, col, n=3):
        return df.nsmallest(n, col)["Name"].tolist()

    st.markdown(f"**🎖️ Top 3 endurants** : {', '.join(top_n(best, 'M/min'))}")
    st.markdown(f"**🔻Flop 3 endurants** : {', '.join(flop_n(best, 'M/min'))}")

    st.markdown(f"**⚡ Top 3 rapides** : {', '.join(top_n(best, 'Vmax'))}")
    st.markdown(f"**🐢 Flop 3 rapides** : {', '.join(flop_n(best, 'Vmax'))}")

    st.markdown(f"**💥 Top 3 explosifs** : {', '.join(top_n(best, 'Amax'))}")
    st.markdown(f"**🔻 Flop 3 explosifs** : {', '.join(flop_n(best, 'Amax'))}")



    # === 2) Build Référence Match ===
    st.subheader("🏆 Référence Match")

    # A) Pull all GAME rows
    mask = data["Type"] == "GAME"
    match_df = data[mask].copy()

    # B) Define and clean numeric columns
    ref_fields = [
        "Duration", "Distance", "M/min", "Distance 15km/h", "M/min 15km/h",
        "Distance 15-20km/h", "Distance 20-25km/h", "Distance 25km/h",
        "N° Sprints", "Acc", "Dec", "Vmax", "Distance 90% Vmax"
    ]
    for c in ref_fields:
        if c in match_df.columns:
            cleaned = (
                match_df[c]
                       .astype(str)
                       .str.replace(r"[^\d\-,\.]", "", regex=True)
                       .str.replace(",", ".", regex=False)
                       .replace("", pd.NA)
            )
            match_df[c] = pd.to_numeric(cleaned, errors="coerce")
        else:
            match_df[c] = pd.NA

    # C) Aggregate per player
    records = []
    for name, grp in match_df.groupby("Name"):
        rec = {"Name": name}
        # full games ≥ 90′ → max of each
        full = grp[grp["Duration"] >= 90]
        if not full.empty:
            for c in ref_fields:
                rec[c] = full[c].max()
        else:
            # partial → copy M/min, M/min 15km/h, Vmax; scale others
            longest = grp.loc[grp["Duration"].idxmax()]
            orig = longest["Duration"]
            rec["Duration"] = orig
            for c in ref_fields:
                val = longest[c]
                if c in {"Duration","Vmax", "M/min", "M/min 15km/h"} or pd.isna(val) or orig <= 0:
                    rec[c] = val
                else:
                    rec[c] = 90 * val / orig
        records.append(rec)

    Refmatch = pd.DataFrame.from_records(records, columns=["Name"] + ref_fields)
    # D) Round & cast
    for c in ref_fields:
        if c == "Vmax":
            Refmatch[c] = Refmatch[c].round(1)
        else:
            Refmatch[c] = Refmatch[c].round(0).astype("Int64")

    # E) Render as scrollable, styled HTML
    styled = (
        Refmatch.style
                .set_table_attributes('class="centered-table"')
                .format({c: "{:.1f}" if c == "Vmax" else "{:d}" for c in ref_fields})
    )
    html = styled.to_html()
    row_h = 35
    total_h = max(300, len(Refmatch) * row_h)
    wrapper = f"""
    <html><head>
      <style>
        .centered-table {{ border-collapse: collapse; width: 100%; }}
        .centered-table th, .centered-table td {{
          text-align: center; padding: 4px 8px; border: 1px solid #ddd;
        }}
        .centered-table th {{ background-color: #f0f0f0; }}
      </style>
    </head><body>
      <div style="max-height:{total_h}px; overflow-y:auto;">
        {html}
      </div>
    </body></html>
    """
    components.html(wrapper, height=total_h + 20, scrolling=True)

    # === 3) Top & Flop from the Référence Match ===
    def best_and_worst(df, col, label):
        clean = df[df[col].notna()]
        if clean.empty:
            return ""
        top = clean.loc[clean[col].idxmax(), "Name"]
        flop = clean.loc[clean[col].idxmin(), "Name"]
        return f"**{label}** : Top – {top}, Flop – {flop}"

    st.markdown(best_and_worst(Refmatch, "M/min", "Endurance relative"))
    st.markdown(best_and_worst(Refmatch, "Vmax", "Vitesse max relative"))
    st.markdown(best_and_worst(Refmatch, "Acc", "Accélérations max"))
    st.markdown(best_and_worst(Refmatch, "Dec", "Décélérations max"))
    st.markdown(best_and_worst(Refmatch, "Distance", "Distance totale"))


# ── PAGE: ENTRAINEMENT ────────────────────────────────────────────────────────

# --- Your other data loading and pre-processing goes above this ---
elif page == "Entrainement":
    # --- Import PDF libs and set PDF_ENABLED ---
    try:
        from reportlab.lib.pagesizes import landscape, A4
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Image, Paragraph, Spacer
        from reportlab.lib import colors
        from reportlab.lib.colors import HexColor
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        PDF_ENABLED = True
    except ImportError:
        PDF_ENABLED = False


    # ========== OBJECTIVE FIELDS ==========
    objective_fields = [
        "RPE",
        "Duration", "Distance", "Distance 15km/h", "Distance 15-20km/h",
        "Distance 20-25km/h", "Distance 25km/h", "Acc", "Dec", "Vmax", "Distance 90% Vmax"
    ]

    # ========== Reference Match ==========
    mask_match = data["Type"].fillna("").str.upper().str.strip() == "GAME"
    match_df_all = data[mask_match].copy()
    ref_fields = [
        "Duration", "Distance", "M/min", "Distance 15km/h", "M/min 15km/h",
        "Distance 15-20km/h", "Distance 20-25km/h", "Distance 25km/h",
        "N° Sprints", "Acc", "Dec", "Vmax", "Distance 90% Vmax"
    ]
    for c in ref_fields:
        if c in match_df_all:
            cleaned = (
                match_df_all[c].astype(str)
                              .str.replace(r"[^\d\-,\.]", "", regex=True)
                              .str.replace(",", ".", regex=False)
                              .replace("", pd.NA)
            )
            match_df_all[c] = pd.to_numeric(cleaned, errors="coerce")
        else:
            match_df_all[c] = pd.NA

    records = []
    for name, grp in match_df_all.groupby("Name"):
        rec = {"Name": name}
        full = grp[grp["Duration"] >= 90]
        if not full.empty:
            for c in ref_fields:
                rec[c] = full[c].max()
        else:
            longest = grp.loc[grp["Duration"].idxmax()]
            orig = longest["Duration"]
            rec["Duration"] = orig
            for c in ref_fields:
                val = longest[c]
                if c in {"Vmax", "M/min", "M/min 15km/h"} or pd.isna(val) or orig <= 0:
                    rec[c] = val
                else:
                    rec[c] = 90 * val / orig
        records.append(rec)
    Refmatch = pd.DataFrame.from_records(records, columns=["Name"] + ref_fields)
    for c in ref_fields:
        if c == "Vmax":
            Refmatch[c] = Refmatch[c].round(1)
        else:
            Refmatch[c] = Refmatch[c].round(0).astype("Int64")

    # ========== Entrainement Block ==========
    st.markdown("### Entraînement")
    allowed_tasks = [
        "OPTI", "MESO", "DRILLS", "COMPENSATION", "MACRO", "OPPO", 
        "OPTI +", "OPTI J-1", "MICRO", "DEV INDIV","WU + GAME + COMP"
    ]
    train_data = data[data["Type"].isin(allowed_tasks)].copy()
    valid_dates = train_data["Date"].dropna()
    if valid_dates.empty:
        st.warning("Aucune date d'entraînement valide trouvée.")
        st.stop()
    
    min_d, max_d = valid_dates.min().date(), valid_dates.max().date()
    sel_date = st.date_input(
        "Choisissez la date pour les objectifs",
        value=max_d,
        min_value=min_d,
        max_value=max_d
    )
    date_df = train_data[train_data["Date"].dt.date == sel_date].copy()
    
    # ====== AM/PM/Total logic (ultra minimal, robust) ======
# ====== AM/PM/Total logic (ultra minimal, robust) ======
    if "AMPM" in date_df.columns and not date_df["AMPM"].isnull().all():
        ampm_unique = sorted([
            str(x) for x in date_df["AMPM"].dropna().unique()
            if str(x).strip() != "" and str(x).lower() != "nan"
        ])
        
        # Reorder to have AM, PM, then Total
        options = [x for x in ["AM", "PM"] if x in ampm_unique]
        options.append("Total")
    
        sel_ampm = st.selectbox("Sélectionnez la session (AM/PM)", options, key="ampm")
    
        if sel_ampm == "Total":
            # Nettoyage AVANT aggregation
            num_cols = [c for c in date_df.columns if c in [
                "Duration", "Distance", "Distance 15km/h", "Distance 15-20km/h",
                "Distance 20-25km/h", "Distance 25km/h", "Acc", "Dec", "Distance 90% Vmax"
            ]]
            for c in num_cols + ["RPE", "Vmax"]:
                if c in date_df.columns:
                    date_df[c] = pd.to_numeric(
                        date_df[c].astype(str)
                                   .str.replace(r"[^\d\-,\.]", "", regex=True)
                                   .str.replace(",", ".", regex=False)
                                   .replace("", pd.NA),
                        errors="coerce"
                    )
            agg_dict = {c: "sum" for c in num_cols}
            for c in ["RPE", "Vmax"]:
                if c in date_df.columns:
                    agg_dict[c] = "mean"
            filtered_df = date_df.groupby("Name", as_index=False).agg(agg_dict)
        else:
            filtered_df = date_df[date_df["AMPM"] == sel_ampm].copy()
    else:
        filtered_df = date_df.copy()
    
    if filtered_df.empty:
        st.info(f"Aucune donnée d'entraînement pour le {sel_date}.")
        st.stop()
    # --- Résumé JOURNÉE ---
    erpe_col = next((c for c in filtered_df.columns if c.lower() == "rpe"), None)
    if erpe_col is not None:
        session_ERPE = pd.to_numeric(filtered_df[erpe_col], errors="coerce").mean()
    else:
        st.warning("Aucune colonne ERPE trouvée dans la séance sélectionnée.")
        session_ERPE = float("nan")
    
    max_duration = pd.to_numeric(filtered_df["Ttotal"], errors="coerce").max(skipna=True) if "Ttotal" in filtered_df.columns else float("nan")
    max_teffectif = pd.to_numeric(filtered_df["Teffectif"], errors="coerce").max(skipna=True) if "Teffectif" in filtered_df.columns else float("nan")
    
    try:
        if pd.notna(max_duration) and max_duration > 0 and pd.notna(max_teffectif):
            indicateur = float(max_teffectif) * 100 / float(max_duration)
        else:
            indicateur = 0
    except Exception:
        indicateur = 0
    
    st.markdown(
        f"###### Objectifs du {sel_date} &nbsp; | &nbsp; "
        f"Temps total : <b>{max_duration:.0f} min</b> &nbsp; | &nbsp; "
        f"Temps effectif : <b>{max_teffectif:.0f} min</b> &nbsp; | &nbsp; "
        f"RPE estimé : <b>{session_ERPE:.0f}</b> &nbsp; | &nbsp; "
        f"Indicateur : <b>{indicateur:.1f}%</b>",
        unsafe_allow_html=True
    )
    
    # ========== SLIDERS OBJECTIVES POUR % ==========
    objective_fields = [
        "RPE", "Duration", "Distance", "Distance 15km/h", "Distance 15-20km/h",
        "Distance 20-25km/h", "Distance 25km/h", "Acc", "Dec", "Vmax", "Distance 90% Vmax"
    ]
    obj_vars = [c for c in objective_fields if c != "RPE"]
    row1, row2 = obj_vars[:5], obj_vars[5:]
    objectives = {}
    cols5 = st.columns(5)
    for i, stat in enumerate(row1):
        with cols5[i]:
            objectives[stat] = st.slider(f"{stat} (%)", 0, 150, 100, key=f"obj_{stat}")
    cols5b = st.columns(5)
    for i, stat in enumerate(row2):
        with cols5b[i]:
            objectives[stat] = st.slider(f"{stat} (%)", 0, 150, 100, key=f"obj_{stat}")
    
    # ========== TABLE + POURCENTAGE + MOYENNES ==========
    df_ent = filtered_df[["Name"] + [col for col in objective_fields if col in filtered_df.columns]].copy()
    for c in objective_fields:
        if c not in df_ent.columns:
            continue
        cleaned = (
            df_ent[c].astype(str)
                     .str.replace(r"[^\d\-,\.]", "", regex=True)
                     .str.replace(",", ".", regex=False)
                     .replace("", pd.NA)
        )
        num = pd.to_numeric(cleaned, errors="coerce")
        df_ent[c] = num.round(1) if c == "Vmax" else num.round(0).astype("Int64")
    
    # --- build global vmax reference (before percentage loop)
    vmax_all = (
        data.copy()
            .assign(Vmax=pd.to_numeric(
                data["Vmax"].astype(str).str.replace(",", "."),
                errors="coerce"
            ))
            .groupby("Name", as_index=False)["Vmax"]
            .max()
            .rename(columns={"Vmax": "Vmax_best"})
    )
    vmax_idx = vmax_all.set_index("Name")
    
    ref_idx = Refmatch.set_index("Name")
    
    # --- percentage columns ---
    for c in objective_fields:
        if c not in df_ent.columns or c == "RPE":
            continue
        pct_col = f"{c} %"
        df_ent[pct_col] = df_ent.apply(
            lambda r: (
                round(r[c] / vmax_idx.at[r["Name"], "Vmax_best"] * 100, 1)
                if c == "Vmax"
                and r["Name"] in vmax_idx.index
                and pd.notna(r[c])
                and pd.notna(vmax_idx.at[r["Name"], "Vmax_best"])
                and vmax_idx.at[r["Name"], "Vmax_best"] > 0
                else
                round(r[c] / ref_idx.at[r["Name"], c] * 100, 1)
                if r["Name"] in ref_idx.index
                and pd.notna(r[c])
                and pd.notna(ref_idx.at[r["Name"], c])
                and ref_idx.at[r["Name"], c] > 0
                else pd.NA
            ),
            axis=1,
        )
    
    mean_data = {"Name": "Moyenne"}
    for c in objective_fields:
        if c not in df_ent.columns:
            continue
        raw_mean = df_ent[c].mean(skipna=True)
        if pd.isna(raw_mean):
            mean_data[c] = pd.NA
        else:
            mean_data[c] = round(raw_mean, 1) if c == "Vmax" else int(round(raw_mean, 0))
        if c != "RPE":
            pct_col = f"{c} %"
            if pct_col in df_ent.columns:
                pct_mean = df_ent[pct_col].mean(skipna=True)
                mean_data[pct_col] = round(pct_mean, 1) if not pd.isna(pct_mean) else pd.NA
    
    df_ent = pd.concat([df_ent, pd.DataFrame([mean_data])], ignore_index=True)
    
    # ====== Tri, couleurs, affichage ======
    df_ent["Pos"] = df_ent["Name"].str.upper().map(player_positions)
    overall_mean = df_ent[df_ent["Name"] == "Moyenne"].copy()
    players_only = df_ent[df_ent["Name"] != "Moyenne"].copy()
    # Ajoute cette ligne AVANT la boucle des groupes
    players_only["Pos"] = players_only["Pos"].fillna("NC")  # 'NC' = non classé
    
    pos_order = ["DC", "M", "PIS", "ATT", "NC"]  # "NC" à la fin pour capturer les joueurs sans position
    grouped = []
    for pos in pos_order:
        grp = players_only[players_only["Pos"] == pos].sort_values("Name")
        if grp.empty:
            continue
        grouped.append(grp)
        mean_vals = {"Name": f"Moyenne {pos}", "Pos": pos}
        for c in objective_fields:
            if c not in grp.columns: continue
            vals = grp[c]
            moy = vals.mean(skipna=True)
            if pd.isna(moy):
                mean_vals[c] = pd.NA
            else:
                mean_vals[c] = round(moy, 1) if c == "Vmax" else int(round(moy, 0))
            if c != "RPE":
                pct_col = f"{c} %"
                if pct_col in grp.columns:
                    pct_mean = grp[pct_col].mean(skipna=True)
                    mean_vals[pct_col] = round(pct_mean, 1) if not pd.isna(pct_mean) else pd.NA
        grouped.append(pd.DataFrame([mean_vals]))
    
    if not grouped:
        st.info("Aucun joueur dans les positions attendues. Vérifiez vos filtres ou vos données.")
        st.stop()
    df_sorted = pd.concat(grouped, ignore_index=True)
    df_sorted.loc[df_sorted['Name'].str.startswith('Moyenne'), 'Pos'] = ''
    
    display_cols = [
        "RPE", "Name", "Pos",
        "Duration", "Duration %",
        "Distance", "Distance %",
        "Distance 15km/h", "Distance 15km/h %",
        "Distance 15-20km/h", "Distance 15-20km/h %",
        "Distance 20-25km/h", "Distance 20-25km/h %",
        "Distance 25km/h", "Distance 25km/h %",
        "Vmax", "Vmax %",
        "Distance 90% Vmax", "Distance 90% Vmax %",
        "Acc", "Acc %",
        "Dec", "Dec %",
    ]
    display_cols = [c for c in display_cols if c in df_sorted.columns]
    df_display = df_sorted.loc[:, display_cols]
    
    def alternate_colors(row):
        if row['Name'].startswith('Moyenne'): return [''] * len(display_cols)
        color = '#EDE8E8' if row.name % 2 == 0 else 'white'
        return [f'background-color:{color}'] * len(display_cols)
    
    def highlight_moyenne(row):
        if row['Name'] == 'Moyenne':
            return ['background-color:#EDE8E8; color:#0031E3;'] * len(display_cols)
        elif row['Name'].startswith('Moyenne ') and row['Name'] != "Moyenne":
            return ['background-color:#CFB013; color:#000000;'] * len(display_cols)
        return [''] * len(display_cols)
    
    def hl(v, obj):
        if pd.isna(v): return ""
        d = abs(v - obj)
        if d <= 5: return "background-color:#c8e6c9;"
        if d <= 10: return "background-color:#fff9c4;"
        if d <= 15: return "background-color:#ffe0b2;"
        if d <= 100: return "background-color:#ffcdd2;"
        return ""
    
    styled = df_display.style
    styled = styled.apply(alternate_colors, axis=1)
    styled = styled.apply(highlight_moyenne, axis=1)
    style_formats = {}
    for c in objective_fields:
        if f"{c} %" in df_display.columns:
            style_formats[f"{c} %"] = "{:.1f} %"
        if c == "Vmax":
            style_formats[c] = "{:.1f}"
        elif c in df_display.columns:
            style_formats[c] = "{:.0f}"
    styled = styled.format(style_formats)
    
    # === apply custom coloring to % columns using sliders ===
    for stat in obj_vars:
        pct_col = f"{stat} %"
        if pct_col in df_display.columns:
            obj = objectives.get(stat, 100)
            styled = styled.applymap(lambda v, obj=obj: hl(v, obj), subset=[pct_col])
    
    styled = styled.set_table_styles([
        {'selector': 'th', 'props': [('background-color', '#0031E3'), ('color', 'white'), ('white-space', 'nowrap')]},
        {'selector': 'th.row_heading, td.row_heading', 'props': 'display:none;'},
        {'selector': 'th.blank', 'props': 'display:none;'}
    ], overwrite=False)
    styled = styled.set_table_attributes('class="centered-table"')
    
    def rpe_color(val, vmin=1, vmax=10):
        if pd.isna(val): return ""
        try:
            norm = (float(val) - vmin) / (vmax - vmin)
            norm = min(max(norm, 0), 1)
            color = mcolors.rgb2hex(cmap(norm))
            return f"background-color:{color};"
        except:
            return ""
    styled = styled.applymap(rpe_color, subset=["RPE"])
    
    import re
    html_obj = re.sub(r'<th[^>]*>.*?%</th>', '<th>%</th>', styled.to_html())
    
    total_rows = df_sorted.shape[0] + 1
    header_height = 30
    row_height = 28
    iframe_height = header_height + total_rows * row_height
    
    html_template = f"""
    <html>
      <head>
        <style>
          .centered-table {{
            border-collapse: collapse;
            min-width: 1200px;          /* force une largeur minimale */
            white-space: nowrap;        /* empêche le retour à la ligne */
            font-size: 11.5px;             /* texte plus petit */
          }}
          .centered-table th, .centered-table td {{
            text-align: center;
            padding: 2px 4px;           /* padding réduit */
            border: 1px solid #ddd;
          }}
          .centered-table th {{
            background-color: #0031E3;
            color: white;
          }}
        </style>
      </head>
      <body>
        <div style="
             max-height: {iframe_height}px;
             overflow-y: auto;
             overflow-x: auto;
          ">
          {html_obj}
        </div>
      </body>
    </html>
    """
    
    # components.html(
    #     html_template,
    #     height=iframe_height,
    #     width=1500,     # suffisamment large pour déclencher le scroll horizontal
    #     scrolling=True  # active les scrollbars
    # )
    safe = re.sub(r'</div>\s*</body>\s*</html>\s*$', '</div>', html_template, flags=re.I)
    st.markdown(safe, unsafe_allow_html=True)

    #st.markdown(html_template, unsafe_allow_html=True)
    
    # # ── Export PDF with same colored table fit to A4 landscape ───────────────
    # if PDF_ENABLED and st.button("📥 Télécharger le rapport PDF"):
    #     obj = objectives.get("Duration", None) 
    #     buf = io.BytesIO()
    #     doc = SimpleDocTemplate(buf, pagesize=landscape(A4),
    #                             rightMargin=2, leftMargin=2, topMargin=5, bottomMargin=2)
    #     styles = getSampleStyleSheet()
    #     normal = styles["Normal"]

    #     # Header
    #     hdr_style = ParagraphStyle('hdr', parent=normal, fontSize=12, leading=14, textColor=HexColor('#0031E3'))
    #     resp = requests.get("https://raw.githubusercontent.com/FC-Versailles/wellness/main/logo.png")
    #     logo = Image(io.BytesIO(resp.content), width=40, height=40)
    #     hdr_data = [
    #         Paragraph("<b>Données GPS - Séance du :</b>", hdr_style),
    #         Paragraph(sel_date.strftime("%d.%m.%Y"), hdr_style),
    #         logo
    #     ]
    #     hdr_tbl = Table([hdr_data], colWidths=[doc.width/3]*3)
    #     hdr_tbl.setStyle(TableStyle([
    #         ('ALIGN', (0, 0), (0, 0), 'LEFT'),
    #         ('ALIGN', (1, 0), (1, 0), 'CENTER'),
    #         ('ALIGN', (2, 0), (2, 0), 'RIGHT'),
    #         ('BOTTOMPADDING', (0, 0), (-1, -1), 2)
    #     ]))

    #     # Build PDF table data
    #     data_pdf = [list(df_display.columns)]
    #     for _, row in df_display.iterrows():
    #         vals = []
    #         for c in df_display.columns:
    #             val = row[c]
    #             if isinstance(val, float) and c == 'Vmax':
    #                 vals.append(f"{val:.1f}")
    #             elif isinstance(val, float) and c.endswith('%'):
    #                 vals.append(f"{val:.1f} %")
    #             elif isinstance(val, (int, np.integer)):
    #                 vals.append(f"{val:d}")
    #             elif pd.isna(val):
    #                 vals.append("")
    #             else:
    #                 vals.append(str(val))
    #         data_pdf.append(vals)

    #     # Build the cell color matrix, mimicking your Streamlit color logic (but does NOT touch Streamlit table)
    #     cell_styles = []
    #     nrows = len(data_pdf)
    #     ncols = len(data_pdf[0])
    #     for row_idx in range(1, nrows):  # skip header
    #         row = df_display.iloc[row_idx - 1]
    #         for col_idx, col in enumerate(df_display.columns):
    #             cell_color = None
    #             cell_text_color = None
        
    #             # ---------- PRIORITÉ : RPE couleur ----------
    #             if col == "RPE" and pd.notna(row["RPE"]):
    #                 val = float(row["RPE"])
    #                 norm = (val - 1) / (10 - 1)
    #                 norm = min(max(norm, 0), 1)
    #                 color = mcolors.rgb2hex(cmap(norm))
    #                 cell_color = color
    #                 cell_text_color = "#000000"
        
    #             # ---------- Moyenne (SURCHARGE) ----------
    #             elif row['Name'] == 'Moyenne':
    #                 cell_color = '#EDE8E8'
    #                 cell_text_color = '#0031E3'
        
    #             elif row['Name'].startswith('Moyenne ') and row['Name'] != 'Moyenne':
    #                 cell_color = '#CFB013'
    #                 cell_text_color = '#000000'
        
    #             # ---------- % columns (objective coloring) ----------
    #             elif col.endswith('%') and not row['Name'].startswith('Moyenne'):
    #                 stat = col.replace(' %', '')
    #                 val = row[col]
    #                 obj = objectives.get(stat, None)
    #                 if pd.notna(val) and obj is not None:
    #                     d = abs(val - obj)
    #                     if d <= 5:
    #                         cell_color = '#c8e6c9'
    #                     elif d <= 10:
    #                         cell_color = '#fff9c4'
    #                     elif d <= 15:
    #                         cell_color = '#ffe0b2'
    #                     else:
    #                         cell_color = '#ffcdd2'
        
    #             # ---------- Alternance pour tout le reste ----------
    #             if cell_color is None:
    #                 cell_color = '#EDE8E8' if (row_idx - 1) % 2 == 0 else 'white'
    #             # PDF style
    #             try:
    #                 cell_styles.append(('BACKGROUND', (col_idx, row_idx), (col_idx, row_idx), HexColor(cell_color)))
    #             except:
    #                 pass
    #             if cell_text_color:
    #                 try:
    #                     cell_styles.append(('TEXTCOLOR', (col_idx, row_idx), (col_idx, row_idx), HexColor(cell_text_color)))
    #                 except:
    #                     pass
    #             elif row['Name'].startswith('Moyenne ') and row['Name'] != 'Moyenne':
    #                 cell_styles.append(('TEXTCOLOR', (col_idx, row_idx), (col_idx, row_idx), colors.black))



    #     # Header row style
    #     cell_styles += [
    #         ('BACKGROUND', (0, 0), (-1, 0), HexColor('#0031E3')),
    #         ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
    #         ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold')
    #     ]
        
    #     base_styles = [
    #         ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
    #         ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
    #         ('FONTSIZE', (0, 0), (-1, 0), 4),
    #         ('FONTSIZE', (0, 1), (-1, -1), 6),
    #         ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    #         ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    #         ('LEFTPADDING', (0, 0), (-1, -1), 2),
    #         ('RIGHTPADDING', (0, 0), (-1, -1), 2),
    #         ('TOPPADDING', (0, 0), (-1, -1), 2),
    #         ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
    #     ]
        
    #     pdf_tbl = Table(data_pdf, colWidths=[doc.width / ncols] * ncols, repeatRows=1)
    #     pdf_tbl.hAlign = 'CENTER'
    #     pdf_tbl.setStyle(TableStyle(base_styles + cell_styles))
        
    #     elements = [hdr_tbl, Spacer(1, 8), pdf_tbl]
    #     doc.build(elements)
    #     st.download_button(
    #         label="📥 Télécharger le PDF", data=buf.getvalue(),
    #         file_name=f"Entrainement_{sel_date.strftime('%Y%m%d')}.pdf", mime="application/pdf"
    #    )
                                                                     
    # ── 2) PERFORMANCES DÉTAILLÉES (date range + filters) ──────────────────
    

    
    
        
    # === CHARGE DU JOUR : z-score vs moyenne par position (fallback global) ===
    st.markdown("### 🎯 Charge du jour")
    
    charge_metrics = ["Distance", "Distance 15km/h", "Distance 20-25km/h", "Distance 25km/h"]
    
    # 1. Refiltrer la séance choisie (sel_date + sel_ampm)
    train_data = data[data["Type"].isin(allowed_tasks)].copy()
    date_df = train_data[train_data["Date"].dt.date == sel_date].copy()
    
    if "AMPM" in date_df.columns and 'sel_ampm' in locals() and sel_ampm != "Total":
        session_df = date_df[date_df["AMPM"] == sel_ampm].copy()
    else:
        # Agrégation "Total" par joueur
        num_cols = [c for c in date_df.columns if c in [
            "Duration", "Distance", "Distance 15km/h", "Distance 15-20km/h",
            "Distance 20-25km/h", "Distance 25km/h", "Acc", "Dec", "Distance 90% Vmax"
        ]]
        for c in num_cols + ["RPE", "Vmax"]:
            if c in date_df.columns:
                date_df[c] = pd.to_numeric(
                    date_df[c].astype(str)
                               .str.replace(r"[^\d\-,\.]", "", regex=True)
                               .str.replace(",", ".", regex=False)
                               .str.replace("\u202f", "", regex=False),
                    errors="coerce"
                )
        agg_dict = {c: "sum" for c in num_cols}
        for c in ["RPE", "Vmax"]:
            if c in date_df.columns:
                agg_dict[c] = "mean"
        session_df = date_df.groupby("Name", as_index=False).agg(agg_dict)
    
    # 2. Construire charge_df et assigner position
    def clean_numeric_series(s):
        return pd.to_numeric(
            s.astype(str)
             .str.replace(r"[^\d\-,\.]", "", regex=True)
             .str.replace(",", ".", regex=False)
             .str.replace("\u202f", "", regex=False),
            errors="coerce"
        )
    
    charge_df = session_df[["Name"] + [c for c in charge_metrics if c in session_df.columns]].copy()
    for m in charge_metrics:
        if m in charge_df.columns:
            charge_df[m] = clean_numeric_series(charge_df[m])
    charge_df["Pos"] = charge_df["Name"].str.upper().map(player_positions).fillna("NC")
    
    # 3. Stats par position sur la séance (mean/std) avec contrôle (>=2 joueurs)
    pos_stats = {}
    for pos, grp in charge_df.groupby("Pos"):
        pos_stats[pos] = {}
        for metric in charge_metrics:
            vals = grp[metric].dropna().astype(float)
            if len(vals) >= 2:
                pos_stats[pos][f"{metric}_mean"] = vals.mean()
                pos_stats[pos][f"{metric}_std"]  = vals.std(ddof=1)
            else:
                pos_stats[pos][f"{metric}_mean"] = np.nan
                pos_stats[pos][f"{metric}_std"]  = np.nan
    
    # 4. Fallback global (toute séance) si position insuffisante
    global_stats = {}
    for metric in charge_metrics:
        all_vals = charge_df[metric].dropna().astype(float)
        if len(all_vals) >= 2:
            global_stats[f"{metric}_mean"] = all_vals.mean()
            global_stats[f"{metric}_std"]  = all_vals.std(ddof=1)
        else:
            global_stats[f"{metric}_mean"] = np.nan
            global_stats[f"{metric}_std"]  = np.nan
    
    # 5. Calcul des z-scores : positionnels avec fallback
    display = charge_df[["Name", "Pos"]].copy()
    
    def compute_z(row, metric):
        pos = row["Pos"]
        val = row.get(metric)
        if pd.isna(val):
            return np.nan
        mean = pos_stats.get(pos, {}).get(f"{metric}_mean", np.nan)
        std  = pos_stats.get(pos, {}).get(f"{metric}_std", np.nan)
        source = "position"
        if pd.isna(mean) or pd.isna(std) or std == 0:
            mean = global_stats.get(f"{metric}_mean", np.nan)
            std  = global_stats.get(f"{metric}_std", np.nan)
            source = "global"
        if pd.isna(mean) or pd.isna(std) or std == 0:
            return np.nan
        z = (val - mean) / std
        return round(z, 2)
    
    for metric in charge_metrics:
        display[metric] = charge_df.apply(lambda r: compute_z(r, metric), axis=1)
    
    # 6. Affichage coloré
    cmap_z = matplotlib.cm.get_cmap("RdYlGn_r")
    norm = matplotlib.colors.Normalize(vmin=-2, vmax=2, clip=True)
    
    def color_by_z(val):
        if pd.isna(val):
            return ""
        return f"background-color:{mcolors.rgb2hex(cmap_z(norm(val)))};"
    
    styled = (
        display.style
               .format({m: "{:.2f}" for m in charge_metrics})
               .set_table_styles([
                   {"selector": "th", "props": [("background-color", "#0031E3"), ("color", "white"), ("text-align", "center")]},
                   {"selector": "td", "props": [("text-align", "center")]}
               ])
    )
    for metric in charge_metrics:
        styled = styled.applymap(color_by_z, subset=[metric])
    
    
        # calcul dynamique
    n_rows = display.shape[0] + 1  # y compris header
    n_cols = display.shape[1]
    height = min(1000, 40 * n_rows)    # limite à 1000px max
    width  = min(1600, 200 * n_cols)   # ~200px par colonne, cap à 1600px
    
    # --- 6. Listes par métrique (au lieu du tableau coloré) ----------------------
    LOW_THR  = -1.20   # z-score < 1.20
    HIGH_THR = 1.50   # z-score > 1.50
    
    def bullet_names(df, metric, op):
        if metric not in df.columns:
            return None
        m = df[metric].dropna()
        if m.empty:
            return None
        
        if op == "low":
            mask = (df[metric] < LOW_THR)
            sub = df.loc[mask & df[metric].notna(), ["Name", metric]]
            sub = sub.sort_values(metric, ascending=True)
        else:
            mask = (df[metric] > HIGH_THR)
            sub = df.loc[mask & df[metric].notna(), ["Name", metric]]
            sub = sub.sort_values(metric, ascending=False)
        
        if sub.empty:
            return None
        
        lines = [f"• {row['Name']} ({row[metric]:.2f})" for _, row in sub.iterrows()]
        return "\n".join(lines)
    
    
    col_low, col_high = st.columns(2)
    with col_low:
        st.markdown("#### Performances basses — complément nécessaire")
        for metric in charge_metrics:
            res = bullet_names(display, metric, "low")
            if res:  # n'affiche que si non vide
                st.markdown(f"**{metric}**")
                st.markdown(res)
    
    with col_high:
        st.markdown("#### Performances élevées — vigilance")
        for metric in charge_metrics:
            res = bullet_names(display, metric, "high")
            if res:
                st.markdown(f"**{metric}**")
                st.markdown(res)

       
    
    # html = f"""
    # <html>
    #   <head>
    #     <style>
    #       .centered-table {{
    #         border-collapse: collapse;
    #         border-spacing: 1;
    #         width:120%;
    #         margin:2;
    #       }}
    #       .centered-table th, .centered-table td {{
    #         padding:6px;
    #         text-align:center;
    #         border:1px solid #ddd;
    #       }}
    #       .centered-table th{{ background-color:#0031E3; color:white; }}
    #       body {{ margin:1; padding:1; }}
    #     </style>
    #   </head>
    #   <body>
    #     {styled.hide(axis="index").to_html()}
    #   </body>
    # </html>
    # """
    # components.html(html, height=height, width=width, scrolling=True)

        
    st.markdown("### 📊 Analyse Semaine")
    
    # ---- tasks allowed in weekly view ----
    allowed_tasks_week = [
        "WU + GAME + COMP", "WU + GAME",
        "OPTI", "MESO", "DRILLS", "COMPENSATION",
        "MACRO", "OPPO", "OPTI +", "OPTI J-1",
        "MICRO", "COMPENSATION", "AUTRES", "DEV INDIV"
    ]
    
    train_data_week = data[data["Type"].isin(allowed_tasks_week)].copy()
    
    # add Semaine if missing
    if "Semaine" not in train_data_week.columns:
        train_data_week["Semaine"] = train_data_week["Date"].dt.strftime("%Y-%W")
    
    weeks = sorted(train_data_week["Semaine"].dropna().unique())
    selected_weeks = st.multiselect(
        "Sélectionnez une ou plusieurs semaines",
        options=weeks,
        default=weeks[-1:] if weeks else []
    )
    
    week_df = train_data_week[train_data_week["Semaine"].isin(selected_weeks)].copy()
    if week_df.empty:
        st.info("Aucune donnée d'entraînement pour les semaines sélectionnées.")
        st.stop()
    
    # ---- clean numeric columns ----
    df_week = week_df[["Semaine", "Name"] + [c for c in objective_fields if c in week_df.columns]].copy()
    
    for c in objective_fields:
        if c in df_week.columns:
            cleaned = (
                df_week[c].astype(str)
                           .str.replace(r"[^\d\-,\.]", "", regex=True)
                           .str.replace(",", ".", regex=False)
                           .replace("", pd.NA)
            )
            num = pd.to_numeric(cleaned, errors="coerce")
            df_week[c] = num.round(1) if c == "Vmax" else num.round(0).astype("Int64")
    
    # ---- aggregation: pure mean per week across players (Vmax = max) ----
    metric_cols = [c for c in df_week.columns if c not in ["Name", "Semaine"]]
    
    # 1) mean for each player inside week (sessions -> player/week)
    agg_first = {c: "sum" for c in metric_cols}
    if "Vmax" in metric_cols:
        agg_first["Vmax"] = "max"
    
    player_week = df_week.groupby(["Semaine", "Name"], as_index=False).agg(agg_first)
    
    # 2) mean across players for each week (keep Vmax max)
    agg_second = {c: "sum" for c in metric_cols if c != "Vmax"}
    if "Vmax" in metric_cols:
        agg_second["Vmax"] = "max"
    
    df_collective = player_week.groupby("Semaine", as_index=False).agg(agg_second)
    
    # --- table: one row per player/week ---
    df_week_mean = player_week.copy()
    df_week_mean["Pos"] = df_week_mean["Name"].str.upper().map(player_positions)
    players_only = df_week_mean.copy()
    players_only["Pos"] = players_only["Pos"].fillna("NC")
    
    pos_order = ["DC", "M", "PIS", "ATT", "NC"]
    blocks = []
    
    for pos in pos_order:
        part = players_only[players_only["Pos"] == pos].sort_values("Name")
        if part.empty:
            continue
        blocks.append(part)
        avg = {"Name": f"Moyenne {pos}", "Pos": pos}
        for c in objective_fields:
            if c in part.columns:
                m = part[c].mean(skipna=True)
                avg[c] = round(m, 1) if c == "Vmax" else int(round(m, 0)) if pd.notna(m) else pd.NA
        blocks.append(pd.DataFrame([avg]))
    
    df_sorted = pd.concat(blocks, ignore_index=True)
    df_sorted.loc[df_sorted["Name"].str.startswith("Moyenne"), "Pos"] = ""
    
    # ---- display ----
    display_cols = ["RPE", "Name", "Pos"] + [c for c in objective_fields if c != "RPE"]
    display_cols = [c for c in display_cols if c in df_sorted.columns]
    df_display = df_sorted[display_cols]
    
    def alternate_colors(row):
        if row["Name"].startswith("Moyenne"):
            return [""] * len(display_cols)
        color = "#EDE8E8" if row.name % 2 == 0 else "white"
        return [f"background-color:{color}"] * len(display_cols)
    
    def highlight_moyenne(row):
        if row["Name"].startswith("Moyenne"):
            return ["background-color:#CFB013; color:#000000;"] * len(display_cols)
        return [""] * len(display_cols)
    
    styled = df_display.style
    styled = styled.apply(alternate_colors, axis=1)
    styled = styled.apply(highlight_moyenne, axis=1)
    
    fmt = {}
    for c in display_cols:
        if c == "Vmax":
            fmt[c] = "{:.1f}"
        elif c not in ["Name", "Pos", "RPE"]:
            fmt[c] = "{:.0f}"
    styled = styled.format(fmt)
    
    styled = styled.set_table_styles([
        {"selector": "th",
         "props": [("background-color", "#0031E3"),
                   ("color", "white"),
                   ("white-space", "nowrap")]},
        {"selector": "th.row_heading, td.row_heading", "props": "display:none;"},
        {"selector": "th.blank", "props": "display:none;"}
    ], overwrite=False)
    styled = styled.set_table_attributes('class="centered-table"')
    
    html_obj = styled.to_html()
    total_rows = df_sorted.shape[0] + 1
    iframe_height = 30 + total_rows * 28
    
    wrapper = f"""
    <html>
      <head>
        <style>
          body {{ margin:0; padding:0; }}
          .centered-table{{border-collapse:collapse;width:100%;}}
          .centered-table th{{font-size:10px;padding:6px 8px;text-align:center;}}
          .centered-table td{{font-size:10px;padding:4px 6px;text-align:center;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}}
          .centered-table th, .centered-table td{{border:1px solid #ddd;}}
          .centered-table th{{background-color:#0031E3;color:white;}}
        </style>
      </head>
      <body>{html_obj}</body>
    </html>
    """
    st.markdown(wrapper, unsafe_allow_html=True)
        
    
    
    st.markdown("#### 📈 Analyse collective")
    
    # --- Filtres principaux partagés ---
    col1, col2 = st.columns(2)
    with col1:
        semaines = sorted(train_data["Semaine"].dropna().unique())
        sel_sem = st.multiselect("Semaine(s)", semaines)
    with col2:
        types = sorted(train_data["Type"].dropna().unique())
        sel_task = st.multiselect("Tâche(s)", types)
    
    # --- Boucle pour 3 graphiques à la suite ---
    for i in range(1, 4):
        st.markdown(f"###### Graphique {i}")
        colx, coly = st.columns(2)
        with coly:
            agg_options = {"Jour": "day", "Semaine": "week", "Mois": "month"}
            x_axis_mode = st.selectbox(
                f"Regrouper par :", list(agg_options.keys()), key=f"xaxis_{i}"
            )
            agg_mode = agg_options[x_axis_mode]
        with colx:
            YVARS = [
                "Duration", "Distance", "M/min", "Distance 15km/h", "M/min 15km/h",
                "Distance 15-20km/h", "Distance 20-25km/h", "Distance 25km/h",
                "Distance 90% Vmax", "N° Sprints", "Vmax", "%Vmax",
                "Acc", "Dec", "Amax", "Dmax", "HSR", "HSR/min",
                "SPR", "SPR/min", "HSPR", "HSPR/min", "Dist Acc", "Dist Dec"
            ]
            sel_y = st.multiselect(
                f"Variable(s) à afficher (max 2) – Graphique {i}",
                options=[v for v in YVARS if v in train_data.columns],
                default=["Distance"],
                max_selections=2,
                key=f"yvar_{i}"
            )
    
        # --- Filtrage ---
        filt = train_data.copy()
        if sel_sem:
            filt = filt[filt["Semaine"].isin(sel_sem)]
        if sel_task:
            filt = filt[filt["Type"].isin(sel_task)]
        if "Date" in filt.columns:
            filt["Date"] = pd.to_datetime(filt["Date"], errors="coerce")
    
        for col in sel_y:
            if col in filt.columns:
                filt[col] = (
                    filt[col]
                    .replace(["None", "nan", "NaN", ""], np.nan)
                    .astype(str)
                    .str.replace(r"[ \u202f\u00A0]", "", regex=True)
                    .str.replace(",", ".", regex=False)
                    .replace("", np.nan)
                )
                filt[col] = pd.to_numeric(filt[col], errors="coerce")
    
        # --- Regroupement ---
        grp = None
        if sel_y:
            # build grouping column
            if agg_mode == "day" and "Date" in filt.columns:
                filt["XGroup"] = filt["Date"].dt.date
                label_func = lambda d: d.strftime("%d.%m") if not pd.isnull(d) else ""
            elif agg_mode == "week":
                filt["XGroup"] = (
                    filt["Semaine"]
                    if "Semaine" in filt.columns
                    else filt["Date"].dt.strftime("%G-W%V")
                )
                label_func = lambda d: f"S{int(d)}" if pd.notnull(d) and str(d).isdigit() else str(d)
            elif agg_mode == "month" and "Date" in filt.columns:
                filt["XGroup"] = filt["Date"].dt.strftime("%Y-%m")
                label_func = lambda d: str(d)
            else:
                filt["XGroup"] = filt["Date"].dt.date if "Date" in filt.columns else None
                label_func = lambda d: str(d)
    
            if "XGroup" in filt.columns:
                if agg_mode == "day":
                    # simple mean by day
                    grp = (
                        filt.groupby("XGroup")[sel_y]
                        .mean(numeric_only=True)
                        .sort_index()
                    )
                else:
                    # 1️⃣ sum sessions inside each player/period
                    inner = {}
                    for v in sel_y:
                        inner[v] = "max" if v == "Vmax" else "sum"
                    by_player = filt.groupby(["XGroup", "Name"], as_index=False).agg(inner)
    
                    # 2️⃣ average across players (or max for Vmax)
                    outer = {}
                    for v in sel_y:
                        outer[v] = "max" if v == "Vmax" else "mean"
                    grp = (
                        by_player.groupby("XGroup", as_index=True)
                        .agg(outer)
                        .sort_index()
                    )
    
        # --- Plot ---
        if grp is not None and not grp.empty:
            grp_plot = grp.reset_index()
            grp_plot = grp_plot.rename(columns={grp_plot.columns[0]: "XGroup"})
            grp_plot["X_fmt"] = grp_plot["XGroup"].apply(label_func)
            y_arg = sel_y if len(sel_y) > 1 else sel_y[0]
            color_sequence = ["#0031E3", "#CFB013"] if len(sel_y) > 1 else ["#0031E3"]
    
            fig = px.bar(
                grp_plot,
                x="XGroup",
                y=y_arg,
                barmode="group",
                labels={"value": "Valeur collective"},
                title=f"Collectif – {' & '.join(sel_y)} par {x_axis_mode.lower()}",
                text_auto=".0f",
                color_discrete_sequence=color_sequence,
            )
            fig.update_traces(
                textposition="outside",
                textfont_size=10,
                textangle=0,
                cliponaxis=False,
            )
            fig.update_layout(
                xaxis_tickangle=0,
                height=600,
                xaxis_title=x_axis_mode,
                yaxis_title="Valeur collective",
                xaxis=dict(
                    tickmode="array",
                    tickvals=grp_plot["XGroup"],
                    ticktext=grp_plot["X_fmt"],
                ),
                margin=dict(t=40, b=30, l=40, r=30),
            )
            st.plotly_chart(fig, use_container_width=True, key=f"plot_{i}")
        else:
            st.info("Aucune donnée pour ce graphique selon ces filtres ou variable non sélectionnée.")

            
    st.markdown("#### 🏃 Training load")
    
    # --- Data cleaning for Duration & RPE ---
    filt = train_data.copy()
    for col in ["Duration", "RPE"]:
        if col in filt.columns:
            filt[col] = (
                filt[col]
                .replace(["None", "nan", "NaN", ""], np.nan)
                .astype(str)
                .str.replace(r"[ \u202f\u00A0]", "", regex=True)
                .str.replace(",", ".", regex=False)
                .replace("", np.nan)
            )
            filt[col] = pd.to_numeric(filt[col], errors="coerce")
    
    # --- UA calculation ---
    filt["UA"] = filt["Duration"] * filt["RPE"]
    
    # --- Check 'Semaine' column ---
    if "Semaine" not in filt.columns:
        st.warning("La colonne 'Semaine' est manquante.")
        st.stop()
    
    # --- Aggregate UA per week ---
    ua_per_week = (
        filt.groupby("Semaine")["UA"].sum().reset_index()
    )
    # Ensure Semaine is int, if not already
    ua_per_week["Semaine"] = pd.to_numeric(ua_per_week["Semaine"], errors="coerce")
    
    # --- Prepare weeks 1 to 20 for x-axis, merge with actual data (fill missing with 0) ---
    weeks = pd.DataFrame({"Semaine": np.arange(1, 21)})
    ua_per_week = weeks.merge(ua_per_week, on="Semaine", how="left").fillna(0)
    
    # --- Plot ---
    fig = px.bar(
        ua_per_week,
        x="Semaine",
        y="UA",
        labels={"UA": "Charge hebdomadaire (UA)", "Semaine": "Semaine"},
        title="Charge collective hebdomadaire (UA = Duration × RPE)",
        text_auto='.0f',
        color_discrete_sequence=["#0031E3"]
    )
    fig.update_traces(
        textposition='outside',
        textfont_size=10,
        textangle=0,
        cliponaxis=False
    )
    fig.update_layout(
        xaxis=dict(tickmode="linear", tick0=1, dtick=1, range=[0.5, 20.5]),
        height=400,
        xaxis_title="Semaine",
        yaxis_title="Charge hebdomadaire (UA)",
        margin=dict(t=40, b=30, l=40, r=30)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    
    
    st.markdown("#### 📊 Analyse Individuelle")
    
    # --- Filtres individuels ---
    # --- Filtres individuels (sans le filtre joueur) ---
    col1, col2, col3 = st.columns(3)
    with col1:
        semaines = sorted(train_data["Semaine"].dropna().unique())
        sel_sem_i = st.multiselect("Semaine(s) (individuel)", semaines)
    with col2:
        types = sorted(train_data["Type"].dropna().unique())
        sel_task_i = st.multiselect("Tâche(s) (individuel)", types)
    with col3:
        names = sorted(train_data["Name"].dropna().unique()) if "Name" in train_data.columns else []
        sel_name_i = st.multiselect("Joueurs à mettre en couleur", names)  # <-- This is your highlight control
    
    # --- Choix des variables X/Y ---
    colx, coly = st.columns(2)
    YVARS = [
        "Duration", "Distance", "M/min", "Distance 15km/h", "M/min 15km/h",
        "Distance 15-20km/h", "Distance 20-25km/h", "Distance 25km/h",
        "Distance 90% Vmax", "N° Sprints", "Vmax", "%Vmax", "Acc", "Dec", "Amax", "Dmax",
        "HSR", "HSR/min", "SPR", "SPR/min", "HSPR", "HSPR/min", "Dist Acc", "Dist Dec"
    ]
    with colx:
        x_var = st.selectbox("Axe X", options=[v for v in YVARS if v in train_data.columns])
    with coly:
        y_var = st.selectbox("Axe Y", options=[v for v in YVARS if v in train_data.columns])
    
    # --- Filtrage données (SAUF joueur) ---
    filt = train_data.copy()
    if sel_sem_i:
        filt = filt[filt["Semaine"].isin(sel_sem_i)]
    if sel_task_i:
        filt = filt[filt["Type"].isin(sel_task_i)]
    
    # --- Nettoyage X/Y ---
    for col in [x_var, y_var]:
        if col in filt.columns:
            filt[col] = (
                filt[col]
                .replace(["None", "nan", "NaN", ""], np.nan)
                .astype(str)
                .str.replace(r"[ \u202f\u00A0]", "", regex=True)
                .str.replace(",", ".", regex=False)
                .replace("", np.nan)
            )
            filt[col] = pd.to_numeric(filt[col], errors="coerce")
    
    # --- Ajout de la colonne couleur ---
    def assign_color(row):
        if row["Name"] in sel_name_i:
            # Use a color per player, or pick your favorite color
            # Here: unique color per selected, else grey
            return row["Name"]
        else:
            return "Autres"
    
    filt["Couleur"] = filt.apply(assign_color, axis=1)
    
    # Define your color mapping
    color_map = {"Autres": "#A9A9A9"}  # All grey by default
    # Add one color per selected player (using Plotly palette)
    from plotly.colors import qualitative
    for i, n in enumerate(sel_name_i):
        color_map[n] = qualitative.Plotly[i % len(qualitative.Plotly)]
    
    # --- Scatter ---
    if not filt.empty and x_var in filt.columns and y_var in filt.columns:
        fig = px.scatter(
            filt,
            x=x_var,
            y=y_var,
            color="Couleur",
            color_discrete_map=color_map,
            hover_data=["Name", "Semaine", "Type"] if "Name" in filt.columns else None,
            title=f"{y_var} vs {x_var}",
            size_max=16,
        )
        fig.update_traces(marker=dict(size=12))
        fig.update_layout(
            height=700,
            margin=dict(t=60, b=40, l=60, r=40),
            xaxis_title=x_var,
            yaxis_title=y_var,
            legend_title="Joueur(s)"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Aucune donnée ou variables non sélectionnées pour l'analyse individuelle.")





# ── PAGE: MATCH ───────────────────────────────────────────────────────────────
elif page == "Match":
    st.markdown("#### 🧮 Match | Composante athlétique")

    # --- Filter only GAME rows (no Prepa / N3)
    games = (
        data[data["Type"].astype(str).str.upper().str.strip() == "GAME"]
        .loc[~data["Jour"].astype(str).str.lower().eq("prepa")]
        .loc[~data["Jour"].astype(str).str.strip().eq("N3")]
        .copy()
    )
    games["Date"] = pd.to_datetime(games["Date"], errors="coerce")

    # --- Clean numeric columns needed for derived metrics
    base_cols = [
        "Duration", "M/min", "M/min 15km/h",
        "Distance 20-25km/h", "Distance 25km/h",
        "N° Sprints", "Acc"
    ]
    for c in base_cols:
        if c in games.columns:
            games[c] = (
                games[c].astype(str)
                        .str.replace(r"[^\d\-,\.]", "", regex=True)
                        .str.replace(",", ".", regex=False)
                        .replace("", pd.NA)
            )
            games[c] = pd.to_numeric(games[c], errors="coerce")

    # --- Derived variables ---
    # 1) M/min 20-25km/h
    if {"Distance 20-25km/h", "Duration"}.issubset(games.columns):
        games["M/min 20-25km/h"] = np.where(
            (games["Duration"] > 0) & games["Distance 20-25km/h"].notna(),
            games["Distance 20-25km/h"] / games["Duration"],
            np.nan
        )
    # 2) M/min 25km/h  (no “Distance” word)
    if {"Distance 25km/h", "Duration"}.issubset(games.columns):
        games["M/min 25km/h"] = np.where(
            (games["Duration"] > 0) & games["Distance 25km/h"].notna(),
            games["Distance 25km/h"] / games["Duration"],
            np.nan
        )
    # 3) Sprints/min
    if {"N° Sprints", "Duration"}.issubset(games.columns):
        games["Sprints/min"] = np.where(
            (games["Duration"] > 0) & games["N° Sprints"].notna(),
            games["N° Sprints"] / games["Duration"],
            np.nan
        )
    # 4) Acc/min
    if {"Acc", "Duration"}.issubset(games.columns):
        games["Acc/min"] = np.where(
            (games["Duration"] > 0) & games["Acc"].notna(),
            games["Acc"] / games["Duration"],
            np.nan
        )

    # --- One date per Jour for ordering
    jour_dates = (
        games.groupby("Jour", as_index=False)["Date"]
             .min()
             .rename(columns={"Date": "MatchDate"})
    )

    # --- Keep only the wanted columns
    keep_cols = [
        "Jour", "M/min", "M/min 15km/h",
        "M/min 20-25km/h", "M/min 25km/h",
        "Sprints/min", "Acc/min"
    ]
    num_cols = [c for c in keep_cols if c != "Jour" and c in games.columns]

    # --- Mean per Jour
    team_mean = (
        games.groupby("Jour", as_index=False)[num_cols].mean(numeric_only=True)
        .merge(jour_dates, on="Jour", how="left")
        .sort_values("MatchDate")
        .reset_index(drop=True)            # <- important
    )
    
    # --- Rounding
    for c in num_cols:
        team_mean[c] = team_mean[c].astype(float).round(2)
    
    versailles_blue = "#0031E3"
    
    def highlight_last_row(row, last_index):
        return [
            f"background-color:{versailles_blue}; color:white" if row.name == last_index else ""
            for _ in row
        ]
    
    # vue et surlignage de la dernière journée (ex: J4 FCVB)
    df_view  = team_mean[["Jour"] + num_cols]
    last_idx = len(df_view) - 1           # <- index de la dernière ligne visible
    
    styled = (
        df_view.style
            .apply(highlight_last_row, axis=1, last_index=last_idx)
            .format({col: "{:.2f}" for col in num_cols})
            .set_table_styles(
                [
                    {"selector": "thead tr th", "props": [("color", "black"), ("font-weight", "bold")]},
                    {"selector": "th.col_heading", "props": [("color", "black"), ("font-weight", "bold")]},
                ],
                overwrite=True,
            )
    )
    
    # Afficher d’abord le tableau par match (la dernière ligne sera bleue, ex: J4 FCVB)
    st.dataframe(styled, use_container_width=True, hide_index=True)
    
    # Puis seulement les moyennes
    global_mean = (
        games.groupby("Jour", as_index=False)[num_cols]
             .mean(numeric_only=True)[num_cols]
             .mean(numeric_only=True)
             .round(2)
    )
    
    # rattacher les positions
    games["Pos"] = games["Name"].str.upper().map(player_positions).fillna("NC")
    metrics = num_cols
    
    def mean_for(pos):
        sub = games.loc[games["Pos"] == pos, metrics]
        return sub.mean(numeric_only=True).round(2)
    
    mean_team = games[metrics].mean().round(2)
    mean_dc   = mean_for("DC")
    mean_m    = mean_for("M")
    mean_pis  = mean_for("PIS")
    mean_att  = mean_for("ATT")
    
    rows = [
        ("Moyenne équipe", mean_team),
        ("Moyenne DC",     mean_dc),
        ("Moyenne M",      mean_m),
        ("Moyenne PIS",    mean_pis),
        ("Moyenne ATT",    mean_att),
    ]
    summary = pd.DataFrame([r[1] for r in rows], index=[r[0] for r in rows]).reset_index()
    summary = summary.rename(columns={"index": "Ligne"})

    st.write("📌 Moyenne | Postes & équipe")
    st.dataframe(summary, use_container_width=True, hide_index=True)
    
    # 2) Select a game to compare
    sel_jour = st.selectbox("Comparer un match", team_mean["Jour"].tolist())
    
    if sel_jour:
        row = team_mean.loc[team_mean["Jour"] == sel_jour, num_cols].iloc[0]
        pct_var = ((row - global_mean) / global_mean * 100).round(1)
    
        st.markdown(f"**Écart de {sel_jour} par rapport à la moyenne globale :**")
    
        for col, pct in pct_var.items():
            if pd.isna(pct):
                continue
            if pct > 5:
                emoji = "🟢"
            elif pct < -5:
                emoji = "🔴"
            else:
                emoji = "⚪️"
            st.markdown(f"- {col} : {pct:+.1f}% {emoji}")

    # === 📊 Performance athlétique joueurs ===

    st.markdown("<hr style='border:1px solid #ddd' />", unsafe_allow_html=True)  
    st.markdown("#### ⏱️ Période du match")
    
    # 1) GAME ANALYSIS
    df_analysis = data.loc[data["Type"].astype(str).str.upper().str.strip() == "GAME ANALYSIS"].copy()
    if df_analysis.empty:
        st.info("Aucune donnée GAME ANALYSIS.")
        st.stop()
    
    # 2) Sélecteur
    games = sorted(df_analysis["Jour"].dropna().unique())
    sel_game = st.selectbox("Choisissez un match", games, index=len(games)-1)
    df_game = df_analysis[df_analysis["Jour"] == sel_game].copy()
    
    # 3) Nettoyer AMPM
    df_game["AMPM"] = (
        df_game["AMPM"]
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .replace({"nan": np.nan, "": np.nan})
    )
    df_game = df_game.dropna(subset=["AMPM"])
    
    # 4) Colonnes numériques
    num_cols = [
        "Duration","M/min","M/min 15km/h",
        "Distance 20-25km/h","Distance 25km/h","N° Sprints"
    ]
    for c in num_cols:
        if c in df_game.columns:
            df_game[c] = (
                df_game[c]
                .astype(str)
                .str.replace(r"[^\d,.\-]", "", regex=True)
                .str.replace(",", ".", regex=False)
            )
            df_game[c] = pd.to_numeric(df_game[c], errors="coerce")
    
    # 5) Colonnes dérivées
    df_game["M/min 20-25km/h"] = df_game["Distance 20-25km/h"] / df_game["Duration"]
    df_game["M/min >25km/h"]   = df_game["Distance 25km/h"] / df_game["Duration"]
    df_game["Sprints/min"]     = df_game["N° Sprints"] / df_game["Duration"]
    
    # 6) Moyennes par période
    metrics = ["M/min","M/min 15km/h","M/min 20-25km/h","M/min >25km/h","Sprints/min"]
    df_mean = (
        df_game.groupby("AMPM")[metrics]
               .mean(numeric_only=True)
               .reset_index()
    )
    
    mt1 = ["1 MT - 1","1 MT - 2","1 MT - 3"]
    mt2 = ["2 MT - 1","2 MT - 2","2 MT - 3"]
    
    def mean_halves(metric):
        v1 = df_mean.loc[df_mean["AMPM"].isin(mt1), metric].mean(skipna=True)
        v2 = df_mean.loc[df_mean["AMPM"].isin(mt2), metric].mean(skipna=True)
        return v1, v2
    
    titles = {
        "M/min": "M/min par période",
        "M/min 15km/h": "M/min >15 km/h par période",
        "M/min 20-25km/h": "M/min 20–25 km/h par période",
        "M/min >25km/h": "M/min >25 km/h par période",
        "Sprints/min": "Sprints / min par période",
    }
    
    # couleurs pastels demandées
    COLOR_MAP = {
        "M/min": "#77DD77",          # vert clair
        "M/min 15km/h": "#FDFD96",   # jaune clair
        "M/min 20-25km/h": "#B50909",# rouge clair / orange
        "M/min >25km/h": "#990000",  # rouge foncé doux
        "Sprints/min": "#555555",    # noir/gris
    }
    
    def build_fig(metric):
        m1, m2 = mean_halves(metric)
        subtitle = (
            f"(moyenne 1MT : {m1:.2f} & 2MT : {m2:.2f})"
            if pd.notna(m1) and pd.notna(m2) else ""
        )
        fig = px.bar(
            df_mean,
            x="AMPM",
            y=metric,
            text=metric,
            title=f"{titles.get(metric, metric)} {subtitle}",
            color_discrete_sequence=[COLOR_MAP.get(metric, "#888888")],
        )
        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig.update_layout(
            showlegend=False,
            xaxis_title="Période",
            yaxis_title=metric,
            height=420,
            margin=dict(t=40, b=40, l=40, r=20),
        )
        return fig
    
    # === Layout ================================================================
    row1 = st.columns(3)
    for idx, metric in enumerate(["M/min","M/min 15km/h","M/min 20-25km/h"]):
        if metric in df_mean.columns:
            with row1[idx]:
                st.plotly_chart(build_fig(metric), use_container_width=True)
    
    row2 = st.columns(2)
    for idx, metric in enumerate(["M/min >25km/h","Sprints/min"]):
        if metric in df_mean.columns:
            with row2[idx]:
                st.plotly_chart(build_fig(metric), use_container_width=True)

    st.markdown("<hr style='border:1px solid #ddd' />", unsafe_allow_html=True)
    st.markdown("#### 🔋 Charge athlétique joueurs")
    
    # --- filter rows ------------------------------------------------------------
    mask_game = (
        data["Type"].fillna("").astype(str).str.strip().str.upper().eq("GAME")
        & ~data["Jour"].astype(str).str.lower().eq("prepa")
        & ~data["Jour"].astype(str).str.upper().eq("N3")
    )
    match_all = data.loc[mask_game].copy()
    match_all["Date"] = pd.to_datetime(match_all["Date"], errors="coerce")
    
    jours = sorted(match_all["Jour"].dropna().unique())
    if not jours:
        st.info("Aucun match disponible.")
        st.stop()
    
    sel_jour = st.selectbox("Choisissez un match (Jour)", options=jours, index=len(jours)-1)
    match_rows = match_all.loc[match_all["Jour"] == sel_jour].copy()
    
    # --- metrics ----------------------------------------------------------------
    METRICS = [
        "Distance","Distance 15km/h",
        "Distance 15-20km/h","Distance 20-25km/h","Distance 25km/h",
        "N° Sprints","Acc","Dec","Vmax","Distance 90% Vmax"
    ]
    ALL_COLS = ["Name","Duration"] + METRICS
    
    # --- numeric cleaning -------------------------------------------------------
    def to_num(s):
        return pd.to_numeric(
            pd.Series(s, dtype="object")
              .astype(str)
              .str.replace(r"[^\d\-,\.]", "", regex=True)
              .str.replace(",", ".", regex=False)
              .replace("", pd.NA),
            errors="coerce"
        )
    
    # ===== 1) Build player reference from all matches ==========================
    rate_like = {"Vmax"}
    match_all["Duration"] = to_num(match_all["Duration"])
    
    ref_records = []
    for name, grp in match_all.groupby("Name"):
        rec = {"Name": name}
        full = grp.loc[grp["Duration"] >= 90].copy()
    
        gnum = grp.copy()
        for c in METRICS:
            if c in gnum.columns:
                gnum[c] = to_num(gnum[c])
    
        if not full.empty:
            fnum = full.copy()
            for c in METRICS:
                if c in fnum.columns:
                    fnum[c] = to_num(fnum[c])
            rec["Duration"] = float(full["Duration"].max())
            for c in METRICS:
                rec[c] = float(fnum[c].max(skipna=True)) if c in fnum.columns else np.nan
        else:
            idx = gnum["Duration"].idxmax()
            row = gnum.loc[idx]
            dur = float(row.get("Duration")) if pd.notna(row.get("Duration")) else np.nan
            rec["Duration"] = dur
            for c in METRICS:
                val = float(row.get(c)) if c in gnum.columns and pd.notna(row.get(c)) else np.nan
                if c in rate_like:
                    rec[c] = val
                else:
                    rec[c] = 90 * val / dur if pd.notna(val) and pd.notna(dur) and dur > 0 else np.nan
        ref_records.append(rec)
    
    ref_df = pd.DataFrame.from_records(ref_records, columns=["Name","Duration"]+METRICS)
    ref_idx = ref_df.set_index("Name")
    
    # ===== 2) % vs reference for current match =================================
    df = match_rows.loc[:, [c for c in ALL_COLS if c in match_rows.columns]].copy()
    for c in ALL_COLS:
        if c in df.columns and c != "Name":
            df[c] = to_num(df[c])
    
    for c in ["Duration"] + METRICS:
        if c not in df.columns:
            continue
        pct_col = f"{c} (%)"
    
        def pct_func(row):
            val = row.get(c)
            ref_val = ref_idx.at[row["Name"], c] if (row["Name"] in ref_idx.index and c in ref_idx.columns) else np.nan
            if pd.notna(val) and pd.notna(ref_val) and ref_val > 0:
                return round(val / ref_val * 100, 1)
            return np.nan
    
        df[pct_col] = df.apply(pct_func, axis=1)
    
    # ===== 3) sort players by Duration =========================================
    df = df.sort_values("Duration", ascending=False).reset_index(drop=True)
    full_dur = df["Duration"].max(skipna=True)   # best duration
    
    # ===== 4) build HTML with colours ==========================================
    header_html = "".join(f"<th>{col}</th>" for col in ["Name","Duration"]+METRICS)
    rows_html = []
    
    for _, row in df.iterrows():
        is_full = pd.notna(row["Duration"]) and row["Duration"] == full_dur
        row_cells = [f"<td>{row['Name']}</td>"]
    
        # Duration cell (never %)
        dur_text = f"{int(row['Duration'])}" if pd.notna(row["Duration"]) else ""
        row_cells.append(f"<td>{dur_text}</td>")
    
        # metrics
        for col in METRICS:
            val = row.get(col)
            pct = row.get(f"{col} (%)")
            text = ""
            if pd.notna(val) and pd.notna(pct):
                text = f"{val:.1f} ({pct:.0f}%)" if col=="Vmax" else f"{int(val)} ({pct:.0f}%)"
            elif pd.notna(val):
                text = f"{val:.1f}" if col=="Vmax" else f"{int(val)}"
    
            style = ""
            if is_full and pd.notna(pct):
                if pct > 110:
                    style = "background-color: lightgreen;"
                elif pct < 90:
                    style = "background-color: lightcoral;"
    
            row_cells.append(f'<td style="{style}">{text}</td>')
        rows_html.append(f"<tr>{''.join(row_cells)}</tr>")
    
    html = f"""
    <div style="max-height:600px;overflow-y:auto;">
    <table style="border-collapse:collapse;width:100%;font-size:12px;">
    <thead>
    <tr style="background:#0031E3;color:white;">{header_html}</tr>
    </thead>
    <tbody>
    {''.join(rows_html)}
    </tbody>
    </table>
    </div>
    """
    
    st.markdown(html, unsafe_allow_html=True)


# === 📊 Intensité athlétique joueurs =========================================

    st.markdown("#### 💥 Intensité athlétique joueurs")
    
    # 1) Colonnes de base dans match_all
    base_cols = ["Name", "Duration", "M/min", "M/min 15km/h", "Acc", "Distance 25km/h"]
    have = [c for c in base_cols if c in match_rows.columns]
    df_int = match_rows[have].copy()
    
    # 2) Nettoyage numérique
    for c in have:
        if c != "Name":
            df_int[c] = to_num(df_int[c])
    
    # 3) Colonnes dérivées
    if {"Distance 25km/h", "Duration"}.issubset(df_int.columns):
        df_int["M/min 25km/h"] = np.where(
            (df_int["Duration"] > 0) & df_int["Distance 25km/h"].notna(),
            (df_int["Distance 25km/h"] / df_int["Duration"]).round(1),
            np.nan,
        )
    if {"Acc", "Duration"}.issubset(df_int.columns):
        df_int["Acc/min"] = np.where(
            (df_int["Duration"] > 0) & df_int["Acc"].notna(),
            (df_int["Acc"] / df_int["Duration"]).round(2),
            np.nan,
        )
    
    # 4) Construire le référentiel « meilleure perf » à partir de tous les matchs
    metrics_int = ["M/min", "M/min 15km/h", "M/min 25km/h", "Acc/min"]
    ref_records = []
    for name, grp in match_all.groupby("Name"):
        rec = {"Name": name}
        g = grp.copy()
        for c in ["M/min", "M/min 15km/h", "Distance 25km/h", "Acc"]:
            if c in g.columns and c != "Name":
                g[c] = to_num(g[c])
        # dérivées dans le ref
        if {"Distance 25km/h", "Duration"}.issubset(g.columns):
            g["M/min 25km/h"] = np.where(
                (g["Duration"] > 0) & g["Distance 25km/h"].notna(),
                g["Distance 25km/h"] / g["Duration"],
                np.nan,
            )
        if {"Acc", "Duration"}.issubset(g.columns):
            g["Acc/min"] = np.where(
                (g["Duration"] > 0) & g["Acc"].notna(),
                g["Acc"] / g["Duration"],
                np.nan,
            )
        rec["Duration"] = g["Duration"].max(skipna=True)
        for m in metrics_int:
            if m in g.columns:
                rec[m] = g[m].max(skipna=True)
        ref_records.append(rec)
    
    ref_int = pd.DataFrame(ref_records, columns=["Name", "Duration"] + metrics_int)
    ref_idx = ref_int.set_index("Name")
    
    # 5) % vs référence pour le match courant
    cols_show = ["Name", "Duration"] + [c for c in metrics_int if c in df_int.columns]
    df_view = df_int[cols_show].copy().sort_values("Duration", ascending=False).reset_index(drop=True)
    
    # conversion numérique
    for c in metrics_int + ["Duration"]:
        if c in df_view and c != "Name":
            df_view[c] = pd.to_numeric(df_view[c], errors="coerce")
    
    # % vs référence
    for c in metrics_int:
        if c in df_view.columns:
            pct_col = f"{c} (%)"
            def pct_func(row):
                val = row.get(c)
                ref_val = ref_idx.at[row["Name"], c] if row["Name"] in ref_idx.index else np.nan
                if pd.notna(val) and pd.notna(ref_val) and ref_val > 0:
                    return round(val / ref_val * 100, 1)
                return np.nan
            df_view[pct_col] = df_view.apply(pct_func, axis=1)
    
    # --- HTML avec couleurs pour tous les joueurs ---
    header_html = "".join(f"<th>{col}</th>" for col in ["Name", "Duration"] + metrics_int)
    rows_html = []
    
    for _, row in df_view.iterrows():
        row_cells = [f"<td>{row['Name']}</td>"]
        row_cells.append(f"<td>{int(row['Duration']) if pd.notna(row['Duration']) else ''}</td>")
    
        for m in metrics_int:
            val = row.get(m)
            pct = row.get(f"{m} (%)")
            if pd.notna(val) and pd.notna(pct):
                text = f"{val:.1f} ({pct:.0f}%)" if "Acc" not in m else f"{val:.2f} ({pct:.0f}%)"
            elif pd.notna(val):
                text = f"{val:.1f}" if "Acc" not in m else f"{val:.2f}"
            else:
                text = ""
            style = ""
            if pd.notna(pct):
                if pct > 90:
                    style = "background-color: lightgreen;"
                elif pct < 85:
                    style = "background-color: lightcoral;"
            row_cells.append(f'<td style="{style}">{text}</td>')
        rows_html.append(f"<tr>{''.join(row_cells)}</tr>")
    
    html = f"""
    <div style="max-height:600px;overflow-y:auto;">
    <table style="border-collapse:collapse;width:100%;font-size:12px;">
    <thead>
    <tr style="background:#0031E3;color:white;">{header_html}</tr>
    </thead>
    <tbody>
    {''.join(rows_html)}
    </tbody>
    </table>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
        # 5) Bar plots
    metrics_available = [m for m in metrics_int if m in df_int.columns]
    
    for metric in metrics_available:
        plot_df = df_int[["Name", "Duration", metric]].dropna(subset=[metric]).copy()
        if plot_df.empty:
            continue
        plot_df = plot_df.sort_values(metric, ascending=False)
        plot_df["Couleur"] = np.where(plot_df["Duration"] < 40, "#CFB013", "#0031E3")
        plot_df["Label"] = plot_df[metric].apply(lambda v: f"{v:.2f}" if pd.notna(v) else "")
    
        fig_bar = px.bar(
            plot_df, x="Name", y=metric, color="Couleur",
            color_discrete_map="identity", text="Label",
            title=f"{metric} par joueur",
        )
        fig_bar.update_traces(textposition="outside")
        fig_bar.update_layout(
            showlegend=False, xaxis_title="Joueur", yaxis_title=metric,
            height=520, margin=dict(t=40, b=40, l=40, r=20),
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # 6) Z-scores
    metrics_z = metrics_available
    if metrics_z:
        Z = df_int[metrics_z].apply(lambda col: pd.Series(zscore(col, nan_policy="omit"), index=df_int.index))
        names = df_int["Name"]
        vers_blue = "#0031E3"
    
        def list_by_metric(metric, high=True, thr=0.9):
            s = Z[metric].dropna()
            sel = s[s > thr] if high else s[s < -thr]
            if sel.empty:
                return "- Aucun"
            return "\n".join(f"- {names.loc[i]} ({sel.loc[i]:.2f})" for i in sel.index)
    
        high_lines = [f"<span style='color:{vers_blue}'>{m}</span>:<br>{list_by_metric(m, True, 0.9)}"
                      for m in metrics_z]
        low_lines = [f"<span style='color:{vers_blue}'>{m}</span>:<br>{list_by_metric(m, False, 0.9)}"
                     for m in metrics_z]
    
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<b>Performances élevées (Z &gt; 0.9)</b><br>" + "<br><br>".join(high_lines),
                        unsafe_allow_html=True)
        with col2:
            st.markdown("<b>Performances basses (Z &lt; -0.9)</b><br>" + "<br><br>".join(low_lines),
                        unsafe_allow_html=True)
    
    st.markdown("<hr style='border:1px solid #ddd' />", unsafe_allow_html=True)


        
    # ========== 3) RÉFÉRENCE MATCH TABLE (styled) ==========
    st.subheader("🏆 Référence Match")
    
    if ref_df.empty:
        st.info("Aucune référence disponible pour ce match.")
        st.stop()
    
    ref_view = ref_df.copy()
    
    def value_to_color(val, vmin, vmax):
        if pd.isna(val):
            return "background-color:#eee;"
        rng = float(vmax) - float(vmin)
        norm = (float(val) - float(vmin)) / rng if rng > 0 else 0.5
        r, g, b, _ = cmaps(norm)
        return f"background-color:{mcolors.rgb2hex((r, g, b))};"
    
    # style matrix
    cell_styles = []
    for j, col in enumerate(ref_view.columns):
        if col == "Name":
            cell_styles.append([""] * len(ref_view))
            continue
        vals = pd.to_numeric(ref_view[col], errors="coerce")
        vmin, vmax = vals.min(skipna=True), vals.max(skipna=True)
        cell_styles.append([value_to_color(v, vmin, vmax) for v in vals])
    
    styles_per_row = list(map(list, zip(*cell_styles)))
    header_html = "".join(f"<th>{col}</th>" for col in ref_view.columns)
    
    rows_html = []
    for i, row in ref_view.iterrows():
        tds = []
        for j, val in enumerate(row):
            style = styles_per_row[i][j]
            col = ref_view.columns[j]
            if col == "Name":
                disp = str(val) if pd.notna(val) else ""
            elif col.lower().startswith("vmax") and pd.notna(val):
                disp = f"{float(val):.1f}"
            elif pd.notna(val) and isinstance(val, (int, float, np.floating, np.integer)):
                disp = f"{int(float(val))}"
            else:
                disp = "" if pd.isna(val) else str(val)
            tds.append(f'<td style="{style}">{disp}</td>')
        rows_html.append(f"<tr>{''.join(tds)}</tr>")
    
    table_height = max(300, len(ref_view) * 35)
    html = f"""
    <html>
    <head>
      <style>
        .centered-table {{
            border-collapse: collapse;
            width: 100%;
            font-size: 12px;
        }}
        .centered-table th, .centered-table td {{
            text-align: center;
            padding: 4px 8px;
            border: 1px solid #ddd;
        }}
        .centered-table th {{
            background-color: #0031E3;
            color: white;
            font-weight: bold;
            white-space: nowrap;
        }}
      </style>
    </head>
    <body>
      <div style="max-height:{table_height}px; overflow-y:auto;">
        <table class="centered-table">
          <thead><tr>{header_html}</tr></thead>
          <tbody>
            {''.join(rows_html)}
          </tbody>
        </table>
      </div>
    </body>
    </html>
    """
    components.html(html, height=table_height + 40, scrolling=True)


    st.markdown("#### 📊 Match référence – Meilleures perfs (intensité)")
    
    # Colonnes nécessaires dans match_all
    need_cols = ["Name", "Duration", "M/min", "M/min 15km/h", "Distance 25km/h", "Acc"]
    present_cols = [c for c in need_cols if c in match_all.columns]
    ref_df = match_all[present_cols].copy()
    
    # nettoyage numérique
    for c in present_cols:
        if c != "Name":
            ref_df[c] = to_num(ref_df[c])
    
    # Colonnes dérivées
    if {"Distance 25km/h", "Duration"}.issubset(ref_df.columns):
        ref_df["M/min 25km/h"] = np.where(
            (ref_df["Duration"] > 0) & ref_df["Distance 25km/h"].notna(),
            ref_df["Distance 25km/h"] / ref_df["Duration"],
            np.nan,
        )
    
    if {"Acc", "Duration"}.issubset(ref_df.columns):
        ref_df["Acc/min"] = np.where(
            (ref_df["Duration"] > 0) & ref_df["Acc"].notna(),
            ref_df["Acc"] / ref_df["Duration"],
            np.nan,
        )
    
    # On garde uniquement les colonnes finales
    metrics_best = ["M/min", "M/min 15km/h", "M/min 25km/h", "Acc/min"]
    cols_best = ["Name"] + [c for c in metrics_best if c in ref_df.columns]
    
    # Meilleure perf par joueur
    best_perf = (
        ref_df.groupby("Name")[cols_best[1:]]  # toutes sauf Name
              .max(min_count=1)
              .reset_index()
    )
    
    # arrondis
    for c in ["M/min", "M/min 15km/h", "M/min 25km/h"]:
        if c in best_perf:
            best_perf[c] = best_perf[c].round(1)
    if "Acc/min" in best_perf:
        best_perf["Acc/min"] = best_perf["Acc/min"].round(2)
    
    # tri (exemple : M/min décroissant)
    order_col = "M/min" if "M/min" in best_perf.columns else best_perf.columns[1]
    best_perf = best_perf.sort_values(order_col, ascending=False).reset_index(drop=True)
    
    st.dataframe(best_perf, use_container_width=True)


# ── PAGE: PLAYER ANALYSIS ────────────────────────────────────────────────────


elif page == "Joueurs":
    st.subheader("🔎 Analyse d'un joueur")

    # --- Sélection du joueur ---
    players = sorted(data["Name"].dropna().unique())
    sel = st.selectbox("Choisissez un joueur", players)

    p_df = data[data["Name"] == sel].copy()
    p_df["Date"] = pd.to_datetime(p_df["Date"], errors="coerce")

    # --- Colonnes numériques à nettoyer ---
    cols_to_clean = [
        "Acc","Dec", "Distance 90% Vmax", "N° Sprints", "Vmax",
        "Distance 20-25km/h", "Distance 25km/h", "Distance", "Distance 15km/h"
    ]
    for c in cols_to_clean:
        if c in p_df.columns:
            p_df[c] = (
                p_df[c].astype(str)
                      .str.replace(r"[^\d,.-]", "", regex=True)
                      .str.replace(",", ".", regex=False)
            )
            p_df[c] = pd.to_numeric(p_df[c], errors="coerce")

    # --- Agrégation par jour pour éviter les doublons ---
    agg_dict = {c: "sum" for c in cols_to_clean if c != "Vmax" and c in p_df.columns}
    if "Vmax" in p_df.columns:
        agg_dict["Vmax"] = "max"

    p_df = p_df.groupby("Date", as_index=False).agg(agg_dict)

    # --- Créer un calendrier complet pour la période ---
    if not p_df["Date"].dropna().empty:
        date_min, date_max = p_df["Date"].min(), p_df["Date"].max()
        full_dates = pd.date_range(start=date_min, end=date_max, freq='D')

        p_df = p_df.set_index("Date").reindex(full_dates).reset_index()
        p_df = p_df.rename(columns={"index": "Date"})
        for c in cols_to_clean:
            if c in p_df.columns:
                p_df[c] = p_df[c].fillna(0)

    # --- Graphique 1 : Acc par jour ---
    if "Acc" in p_df.columns:
        fig1 = px.bar(
            p_df,
            x="Date",
            y="Acc",
            title=f"{sel} – Accélérations",
            color_discrete_sequence=["#0031E3"],
            text_auto='.0f'
        )
        fig1.update_traces(textposition='outside', cliponaxis=False)
        fig1.update_layout(
            height=400,
            margin=dict(t=40, b=30, l=40, r=30),
            xaxis=dict(dtick="D1", tickformat="%d-%m")
        )
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.info("Pas de données Acc pour ce joueur.")
        
    if "Dec" in p_df.columns:
        fig3 = px.bar(
            p_df,
            x="Date",
            y="Dec",
            title=f"{sel} – Decélérations",
            color_discrete_sequence=["#0031E3"],
            text_auto='.0f'
        )
        fig3.update_traces(textposition='outside', cliponaxis=False)
        fig3.update_layout(
            height=400,
            margin=dict(t=40, b=30, l=40, r=30),
            xaxis=dict(dtick="D1", tickformat="%d-%m")
        )
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Pas de données Acc pour ce joueur.")

    # --- Graphique 2 : Distance 90% Vmax + N° Sprints (bar), Vmax (scatter) ---
    if {"Distance 90% Vmax", "N° Sprints", "Vmax"}.issubset(p_df.columns):
        fig2 = go.Figure()
    
        # --- Bars ---
        fig2.add_trace(go.Bar(
            x=p_df["Date"],
            y=p_df["Distance 90% Vmax"],
            name="Distance 90% Vmax",
            marker_color="#0031E3"
        ))
    
        fig2.add_trace(go.Bar(
            x=p_df["Date"],
            y=p_df["N° Sprints"],
            name="N° Sprints",
            marker_color="#CFB013"
        ))
    
        # --- Red dots for Vmax ---
        fig2.add_trace(go.Scatter(
            x=p_df["Date"],
            y=p_df["Vmax"],
            name="Vmax (km/h)",
            mode="markers+lines",
            marker=dict(color="red", size=8),
            yaxis="y2"
        ))
    
            
        # --- Add annotations above red dots for Vmax > 10 ---
        vmax_points = p_df[p_df["Vmax"] > 10]
        for _, row in vmax_points.iterrows():
            fig2.add_annotation(
                x=row["Date"],
                y=row["Vmax"] + 100,        # Place the text 2 units ABOVE the dot
                text=f"{row['Vmax']:.1f}",  # Show 1 decimal
                showarrow=False,
                xanchor="center",
                font=dict(size=10, color="black", family="Arial")
            )
    
        # --- Layout ---
        fig2.update_layout(
            title=f"{sel} – Distance 90% Vmax, N° Sprints & Vmax",
            barmode="group",
            yaxis=dict(title="Distance / Sprints"),
            yaxis2=dict(title="Vmax (km/h)", overlaying="y", side="right"),
            height=400,
            xaxis=dict(dtick="D1", tickformat="%d-%m"),
            margin=dict(t=40, b=30, l=40, r=30)
        )
    
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Pas de données Distance 90% Vmax, N° Sprints ou Vmax pour ce joueur.")

    # --- Graphique 3 : Distance 20-25km/h et Distance 25km/h ---
    if {"Distance 20-25km/h", "Distance 25km/h"}.issubset(p_df.columns):
        fig3 = px.bar(
            p_df,
            x="Date",
            y=["Distance 20-25km/h", "Distance 25km/h"],
            title=f"{sel} – Distance par zones de vitesse",
            barmode="group",
            color_discrete_sequence=["#0031E3", "#CFB013"],
            text_auto='.0f'
        )
        fig3.update_traces(textposition='outside', cliponaxis=False)
        fig3.update_layout(
            height=400,
            margin=dict(t=40, b=30, l=40, r=30),
            xaxis=dict(dtick="D1", tickformat="%d-%m")
        )
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Pas de données Distance 20-25km/h ou Distance 25km/h pour ce joueur.")
        
        # --- Graphique 4 : Distance et Distance 15km/h ---
    if {"Distance", "Distance 15km/h"}.issubset(p_df.columns):
        fig4 = px.bar(
            p_df,
            x="Date",
            y=["Distance", "Distance 15km/h"],
            title=f"{sel} – Distance totale et Distance >15km/h",
            barmode="group",
            color_discrete_sequence=["#0031E3", "#CFB013"],
            text_auto='.0f'
        )
        fig4.update_traces(textposition='outside', cliponaxis=False)
        fig4.update_layout(
            height=400,
            margin=dict(t=40, b=30, l=40, r=30),
            xaxis=dict(dtick="D1", tickformat="%d-%m")
        )
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("Pas de données Distance ou Distance 15km/h pour ce joueur.") 
        
##########################################################################################################
        
    # Ajouter Semaine avant agrégation par Date
    if "Semaine" in data.columns:
        p_df = data[data["Name"] == sel].copy()
        p_df["Date"] = pd.to_datetime(p_df["Date"], errors="coerce")
        p_df["Semaine"] = data.loc[data["Name"] == sel, "Semaine"].values
    else:
        st.warning("La colonne 'Semaine' n'existe pas dans le DataFrame.")
        st.stop()
    
    # --- Colonnes numériques à nettoyer ---
    cols_to_clean = [
        "Acc", "Dec", "Distance 90% Vmax", "N° Sprints", "Vmax",
        "Distance 20-25km/h", "Distance 25km/h", "Distance", "Distance 15km/h"
    ]
    for c in cols_to_clean:
        if c in p_df.columns:
            p_df[c] = (
                p_df[c].astype(str)
                      .str.replace(r"[^\d,.-]", "", regex=True)
                      .str.replace(",", ".", regex=False)
            )
            p_df[c] = pd.to_numeric(p_df[c], errors="coerce")
    
    # --- Agrégation par jour ---
    agg_dict = {c: "sum" for c in cols_to_clean if c != "Vmax" and c in p_df.columns}
    if "Vmax" in p_df.columns:
        agg_dict["Vmax"] = "max"
    
    p_day = p_df.groupby(["Date", "Semaine"], as_index=False).agg(agg_dict)
    
    # --- Agrégation par semaine ---
    p_week = p_df.groupby("Semaine", as_index=False).agg(agg_dict)
         
    if "Acc" in p_week.columns:
        fig1 = px.bar(
            p_week,
            x="Semaine",
            y="Acc",
            title=f"{sel} – Accélérations",
            color_discrete_sequence=["#0031E3"]
        )
    
        # Add annotation above each bar showing the total accelerations
        for i, row in p_week.iterrows():
            fig1.add_annotation(
                x=row["Semaine"],
                y=row["Acc"] + (row["Acc"] * 0.05),  # 5% above bar
                text=f"{int(row['Acc'])}",
                showarrow=False,
                font=dict(size=11, color="black"),
                xanchor="center",
                yanchor="bottom"  # Ensure it's above
            )
    
        fig1.update_traces(cliponaxis=False)
        fig1.update_layout(height=400, margin=dict(t=40, b=30, l=40, r=30))
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.info("Pas de données Acc pour ce joueur.")

    # --- Graphique 2 : Distance 90% Vmax + N° Sprints ---
    if {"Distance 90% Vmax", "N° Sprints"}.issubset(p_week.columns):
        fig2 = go.Figure()
    
        # Add bars
        fig2.add_trace(go.Bar(
            x=p_week["Semaine"],
            y=p_week["Distance 90% Vmax"],
            name="Distance 90% Vmax",
            marker_color="#0031E3"
        ))
    
        fig2.add_trace(go.Bar(
            x=p_week["Semaine"],
            y=p_week["N° Sprints"],
            name="N° Sprints",
            marker_color="#CFB013"
        ))
    
        # Add centered annotation
        for i, row in p_week.iterrows():
            max_val = max(row["Distance 90% Vmax"], row["N° Sprints"])
            fig2.add_annotation(
                x=row["Semaine"],
                y=max_val + max_val * 0.05,
                text=f"{int(row['Distance 90% Vmax'])} / {int(row['N° Sprints'])}",
                showarrow=False,
                font=dict(size=11, color="black"),
                xanchor="center",
                yanchor="bottom"
            )
    
        fig2.update_layout(
            title=f"{sel} – Distance 90% Vmax & N° Sprints",
            barmode="group",
            yaxis=dict(title="Distance / Sprints"),
            height=400,
            margin=dict(t=40, b=30, l=40, r=30)
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Pas de données Distance 90% Vmax ou N° Sprints pour ce joueur.")
    
    # --- Graphique 3 : Distance 20-25km/h et Distance 25km/h ---
    if {"Distance 20-25km/h", "Distance 25km/h"}.issubset(p_week.columns):
        fig3 = px.bar(
            p_week,
            x="Semaine",
            y=["Distance 20-25km/h", "Distance 25km/h"],
            title=f"{sel} – Distance par zones de vitesse",
            barmode="group",
            color_discrete_sequence=["#0031E3", "#CFB013"]
        )
    
        # Add centered annotations
        for i, row in p_week.iterrows():
            max_val = max(row["Distance 20-25km/h"], row["Distance 25km/h"])
            fig3.add_annotation(
                x=row["Semaine"],
                y=max_val + max_val * 0.05,
                text=f"{int(row['Distance 20-25km/h'])} / {int(row['Distance 25km/h'])}",
                showarrow=False,
                font=dict(size=11, color="black"),
                xanchor="center",
                yanchor="bottom"
            )
    
        fig3.update_layout(height=400, margin=dict(t=40, b=30, l=40, r=30))
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Pas de données Distance 20-25km/h ou Distance 25km/h pour ce joueur.")
    
    # --- Graphique 4 : Distance et Distance 15km/h ---
    if {"Distance", "Distance 15km/h"}.issubset(p_week.columns):
        fig4 = px.bar(
            p_week,
            x="Semaine",
            y=["Distance", "Distance 15km/h"],
            title=f"{sel} – Distance totale et Distance >15km/h",
            barmode="group",
            color_discrete_sequence=["#0031E3", "#CFB013"]
        )
    
        # Add centered annotations
        for i, row in p_week.iterrows():
            max_val = max(row["Distance"], row["Distance 15km/h"])
            fig4.add_annotation(
                x=row["Semaine"],
                y=max_val + max_val * 0.05,
                text=f"{int(row['Distance'])} / {int(row['Distance 15km/h'])}",
                showarrow=False,
                font=dict(size=11, color="black"),
                xanchor="center",
                yanchor="bottom"
            )
    
        fig4.update_layout(height=400, margin=dict(t=40, b=30, l=40, r=30))
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("Pas de données Distance ou Distance 15km/h pour ce joueur.")
        
        
elif page == "Minutes de jeu":
    st.subheader("⏱️ Minutes de jeu")

    df = data.copy()

    player_col = "Name"
    minutes_col = "Duration"
    type_col    = "Type"
    jours_col   = "Jour"   # vérifie si c'est bien Jour
    date_col    = "Date"

    # nettoyage
    df[minutes_col] = pd.to_numeric(df[minutes_col], errors="coerce")
    df[type_col] = df[type_col].astype(str).str.upper().str.strip()
    df[jours_col] = df[jours_col].astype(str)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # filtre
    mask = (
        df[type_col].eq("GAME")
        & ~df[jours_col].str.lower().eq("prepa")
        & ~df[jours_col].str.upper().eq("N3")      # <── exclusion
        & df[minutes_col].notna()
    )
    dg = df.loc[mask].copy()
    if dg.empty:
        st.info("Pas de GAME avec Jour != 'prepa' et != 'N3'.")
        st.stop()


    # bar chart
    tot = (
        dg.groupby(player_col, as_index=False)[minutes_col]
          .sum()
          .sort_values(minutes_col, ascending=False)
    )
    fig_bar = px.bar(
        tot,
        x=player_col,
        y=minutes_col,
        title="Minutes cumulées par joueur",
        text=minutes_col,
        color_discrete_sequence=["#0031E3"]
    )
    fig_bar.update_traces(textposition="outside", cliponaxis=False)
    fig_bar.update_layout(xaxis_title=None, yaxis_title="Minutes")
    st.plotly_chart(fig_bar, use_container_width=True)

     # ---- SCATTER: total vs 3 derniers (avec labels auto-ajustés + flèches) ----
    ordered = dg.sort_values([player_col, date_col])
    last3 = (
        ordered.groupby(player_col, group_keys=False)
               .apply(lambda x: x.tail(3)[minutes_col].sum())
               .rename("last3_min")
    )
    total = tot.set_index(player_col)[minutes_col].rename("total_min")
    
    scat_df = (
        pd.concat([total, last3], axis=1)
          .reset_index()
          .fillna(0)
          .rename(columns={player_col: "player"})
    )
    
    # Base scatter: points bleu FCV
    fig_sc = go.Figure()
    fig_sc.add_trace(go.Scatter(
        x=scat_df["total_min"],
        y=scat_df["last3_min"],
        mode="markers",
        marker=dict(size=10, color="#0031E3"),
        hovertext=scat_df["player"],
        hoverinfo="text+x+y",
        showlegend=False
    ))
    
    fig_sc.update_layout(
        title="Minutes: total saison vs 3 derniers matchs",
        xaxis_title="Total minutes (saison)",
        yaxis_title="Minutes des 3 derniers matchs"
    )
    
    # ---- Placement anti-chevauchement simple par offsets alternés ----
    # On attribue des offsets différents aux points proches pour éviter la superposition.
    # Palette d'offsets (pixels) que l'on fait tourner si conflit détecté.
    offsets = [(0,-24),(0,24),(26,0),(-26,0),(30,-30),(-30,30),(34,-18),(-34,18)]
    used_positions = []  # mémorise boîtes approx pour limiter chevauchement
    
    # Seuil de proximité en unités données (dépend de l'échelle du graphique)
    x_range = max(scat_df["total_min"].max() - scat_df["total_min"].min(), 1)
    y_range = max(scat_df["last3_min"].max() - scat_df["last3_min"].min(), 1)
    tol_x = 0.03 * x_range
    tol_y = 0.05 * y_range
    
    def is_close(x, y, boxes):
        for (bx, by) in boxes:
            if abs(x - bx) <= tol_x and abs(y - by) <= tol_y:
                return True
        return False
    
    # Trie par y puis x pour un placement stable
    rows = scat_df.sort_values(["last3_min","total_min"]).to_dict("records")
    
    k = 0
    for r in rows:
        x = r["total_min"]
        y = r["last3_min"]
        # si proche d'un label déjà placé, on change d'offset
        if is_close(x, y, used_positions):
            k += 1
        ax, ay = offsets[k % len(offsets)]
        used_positions.append((x, y))
    
        fig_sc.add_annotation(
            x=x, y=y,
            text=r["player"],
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1.2,
            arrowcolor="#0031E3",
            ax=ax, ay=ay,  # offset en pixels
            font=dict(color="#0031E3")
        )
    
    st.plotly_chart(fig_sc, use_container_width=True)
