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
import matplotlib.colors as mcolors



# ── Constants ─────────────────────────────────────────────────────────────────


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
TOKEN_FILE     = 'token._gps.pickle'
SPREADSHEET_ID = '1NfaLx6Yn09xoOHRon9ri6zfXZTkU1dFFX2rfW1kZvmw'
SHEET_NAME     = 'Feuille 1'
RANGE_NAME     = 'Feuille 1!A1:Z'   # only pull columns A through Z

def get_credentials():
    creds = None
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, 'rb') as f:
            creds = pickle.load(f)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'client_secret.json', SCOPES
            )
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, 'wb') as f:
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

@st.cache_data(ttl=600)
def load_data():
    creds = get_credentials()
    service = build('sheets', 'v4', credentials=creds)

    # === Use the fast values().get() endpoint ===
    result = (
        service.spreadsheets()
               .values()
               .get(spreadsheetId=SPREADSHEET_ID,
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
        "Season","Semaine","HUMEUR","PLAISIR","RPE","Date","AMPM","Jour","Type","Name",
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


# ── Pre-process common cols ───────────────────────────────────────────────────

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
    "ADEHOUNI":   "PIS",
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
    "TCHATO":     "PIS",
    "ZEMOURA":    "ATT",
    "BASQUE":     "M",
    "KOUASSI":    "M",
    "ODZOUMO":    "ATT",
    "TRAORE":     "M",
    "KOFFI":      "ATT"
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

    # ── Try to import PDF libs ───────────────────────────────────────────────
    try:
        from reportlab.lib.pagesizes import landscape, A4
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Image, Paragraph, Spacer
        from reportlab.lib import colors
        from reportlab.lib.colors import HexColor
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        PDF_ENABLED = True
    except ImportError:
        PDF_ENABLED = False

    # ── 0) Build Référence Match ──────────────────────────────────────────────
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

    # ── 1) OBJECTIFS ENTRAÎNEMENT (single date) ────────────────────────────────
    st.markdown("### 🎯 Entraînement")
    allowed_tasks = ["OPTI", "MESO", "DRILLS", "COMPENSATION", "MACRO", "OPPO", "OPTI +", "OPTI J-1", "REATHLE", "MICRO"]
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

    # --- AM/PM filtering, robust ---
    if "AMPM" in date_df.columns and not date_df["AMPM"].isnull().all():
        ampm_unique = sorted([str(x) for x in date_df["AMPM"].dropna().unique() if str(x).strip() != "" and str(x).lower() != "nan"])
        if len(ampm_unique) > 1:
            sel_ampm = st.selectbox("Sélectionnez la session (AM/PM)", ampm_unique, key="ampm")
            date_df = date_df[date_df["AMPM"] == sel_ampm]

    if date_df.empty:
        st.info(f"Aucune donnée d'entraînement pour le {sel_date}.")
    else:
        # RPE ajouté en premier
        objective_fields = [
            "RPE",
            "Duration", "Distance", "Distance 15km/h", "Distance 15-20km/h",
            "Distance 20-25km/h", "Distance 25km/h", "Acc", "Dec", "Vmax", "Distance 90% Vmax"
        ]

        st.markdown(f"###### Objectifs du {sel_date}")
        objectives = {}
        # Pour le style : sliders pour tous sauf RPE (qui n'a pas d'objectif %)
        row1, row2 = objective_fields[1:6], objective_fields[6:]
        cols5 = st.columns(5)
        for cont, stat in zip(cols5, row1):
            with cont:
                objectives[stat] = st.slider(stat, 0, 100, 100, key=f"obj_ent_{stat}")
        cols5 = st.columns(5)
        for cont, stat in zip(cols5, row2):
            with cont:
                objectives[stat] = st.slider(stat, 0, 100, 100, key=f"obj_ent_{stat}")

        df_ent = date_df[["Name"] + objective_fields].copy()
        for c in objective_fields:
            cleaned = (
                df_ent[c].astype(str)
                         .str.replace(r"[^\d\-,\.]", "", regex=True)
                         .str.replace(",", ".", regex=False)
                         .replace("", pd.NA)
            )
            num = pd.to_numeric(cleaned, errors="coerce")
            df_ent[c] = num.round(1) if c == "Vmax" else num.round(0).astype("Int64")

        ref_idx = Refmatch.set_index("Name")
        for c in objective_fields:
            if c == "RPE":
                continue  # Pas de colonne % pour RPE
            df_ent[f"{c} %"] = df_ent.apply(
                lambda r: round(r[c] / ref_idx.at[r["Name"], c] * 100, 1)
                if (r["Name"] in ref_idx.index and pd.notna(r[c]) and pd.notna(ref_idx.at[r["Name"], c]) and ref_idx.at[r["Name"], c] > 0)
                else pd.NA,
                axis=1
            )

        mean_data = {"Name": "Moyenne"}
        for c in objective_fields:
            raw_mean = df_ent[c].mean(skipna=True)
            mean_data[c] = round(raw_mean, 1) if c == "Vmax" else int(round(raw_mean, 0))
            if c != "RPE":
                pct_mean = df_ent[f"{c} %"].mean(skipna=True)
                mean_data[f"{c} %"] = round(pct_mean, 1)
        df_ent = pd.concat([df_ent, pd.DataFrame([mean_data])], ignore_index=True)

        df_ent["Pos"] = df_ent["Name"].str.upper().map(player_positions)
        overall_mean = df_ent[df_ent["Name"] == "Moyenne"].copy()
        players_only = df_ent[df_ent["Name"] != "Moyenne"].copy()

        pos_order = ["ATT", "DC", "M", "PIS", "PIST"]
        grouped = []
        for pos in pos_order:
            grp = players_only[players_only["Pos"] == pos].sort_values("Name")
            if grp.empty: continue
            grouped.append(grp)
            mean_vals = {"Name": f"Moyenne {pos}", "Pos": pos}
            for c in objective_fields:
                vals = grp[c]
                mean_vals[c] = round(vals.mean(skipna=True), 1) if c == "Vmax" else int(round(vals.mean(skipna=True), 0))
                if c != "RPE":
                    mean_vals[f"{c} %"] = round(grp[f"{c} %"].mean(skipna=True), 1)
            grouped.append(pd.DataFrame([mean_vals]))
        others = players_only[~players_only["Pos"].isin(pos_order)].sort_values("Name")
        if not others.empty: grouped.append(others)
        grouped.append(overall_mean)
        df_sorted = pd.concat(grouped, ignore_index=True)
        df_sorted.loc[df_sorted['Name'].str.startswith('Moyenne'), 'Pos'] = ''

        # explicitly list the columns in the desired order, RPE left
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
        # Ne garde que les colonnes existantes
        display_cols = [c for c in display_cols if c in df_sorted.columns]
        df_display = df_sorted.loc[:, display_cols]

        def alternate_colors(row):
            if row['Name'].startswith('Moyenne'): return [''] * len(display_cols)
            color = '#EDE8E8' if row.name % 2 == 0 else 'white'
            return [f'background-color:{color}'] * len(display_cols)

        def highlight_moyenne(row):
            if row['Name'] == 'Moyenne':
                return ['background-color:#EDE8E8; color:#0031E3;'] * len(display_cols)
            elif row['Name'].startswith('Moyenne ') and row['Name'] != 'Moyenne':
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
            if c != "RPE":
                style_formats[f"{c} %"] = "{:.1f} %"
            elif c == "Vmax":
                style_formats[c] = "{:.1f}"
            else:
                style_formats[c] = "{:.0f}"
        styled = styled.format(style_formats)

        # Coloration pour les colonnes % (sauf RPE % qui n'existe pas)
        for stat in objective_fields:
            if stat == "RPE":
                continue
            def fn(row, stat=stat):
                if row['Name'].startswith("Moyenne ") and row['Name'] != "Moyenne":
                    return [f"background-color:#CFB013; color:#000;" if col == f"{stat} %" else "" for col in row.index]
                elif row['Name'] == "Moyenne":
                    return ["" for col in row.index]
                else:
                    return [hl(row[f"{stat} %"], objectives[stat]) if col == f"{stat} %" else "" for col in row.index]
            styled = styled.apply(fn, axis=1)

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




    
        html_obj = re.sub(r'<th[^>]*>.*?%</th>', '<th>%</th>', styled.to_html())

        # ── STREAMLIT HTML RENDER (auto-height to show all rows) ──────────────
        total_rows = df_sorted.shape[0] + 1            # +1 for the header
        header_height = 30                             # px
        row_height = 28                                # px per row
        iframe_height = header_height + total_rows * row_height

        wrapper = f"""
        <html>
          <head>
            <style>
              .centered-table{{border-collapse:collapse;width:100%;}}
              .centered-table th {{font-size:10px; padding:6px 8px; text-align:center;}}
              .centered-table td {{font-size:10px; padding:4px 6px; text-align:center; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}}
              .centered-table th, .centered-table td {{border:1px solid #ddd;}}
              .centered-table th{{background-color:#0031E3;color:white;}}
            </style>
          </head>
          <body>{html_obj}</body>
        </html>
        """
        components.html(wrapper, height=iframe_height, scrolling=False)

        # ── Export PDF with same colored table fit to A4 landscape ───────────────
        if PDF_ENABLED and st.button("📥 Télécharger le rapport PDF"):
            buf = io.BytesIO()
            doc = SimpleDocTemplate(buf, pagesize=landscape(A4),
                                    rightMargin=2, leftMargin=2, topMargin=5, bottomMargin=2)
            styles = getSampleStyleSheet()
            normal = styles["Normal"]

            # Header
            hdr_style = ParagraphStyle('hdr', parent=normal, fontSize=12, leading=14, textColor=HexColor('#0031E3'))
            resp = requests.get("https://raw.githubusercontent.com/FC-Versailles/wellness/main/logo.png")
            logo = Image(io.BytesIO(resp.content), width=40, height=40)
            hdr_data = [
                Paragraph("<b>Données GPS - Séance du :</b>", hdr_style),
                Paragraph(sel_date.strftime("%d.%m.%Y"), hdr_style),
                logo
            ]
            hdr_tbl = Table([hdr_data], colWidths=[doc.width/3]*3)
            hdr_tbl.setStyle(TableStyle([
                ('ALIGN', (0, 0), (0, 0), 'LEFT'),
                ('ALIGN', (1, 0), (1, 0), 'CENTER'),
                ('ALIGN', (2, 0), (2, 0), 'RIGHT'),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 2)
            ]))

            # Build PDF table data
            data_pdf = [list(df_display.columns)]
            for _, row in df_display.iterrows():
                vals = []
                for c in df_display.columns:
                    val = row[c]
                    if isinstance(val, float) and c == 'Vmax':
                        vals.append(f"{val:.1f}")
                    elif isinstance(val, float) and c.endswith('%'):
                        vals.append(f"{val:.1f} %")
                    elif isinstance(val, (int, np.integer)):
                        vals.append(f"{val:d}")
                    elif pd.isna(val):
                        vals.append("")
                    else:
                        vals.append(str(val))
                data_pdf.append(vals)

            # Build the cell color matrix, mimicking your Streamlit color logic (but does NOT touch Streamlit table)
            cell_styles = []
            nrows = len(data_pdf)
            ncols = len(data_pdf[0])
            for row_idx in range(1, nrows):  # skip header
                row = df_display.iloc[row_idx - 1]
                for col_idx, col in enumerate(df_display.columns):
                    cell_color = None
                    cell_text_color = None
            
                    # ---------- PRIORITÉ : RPE couleur ----------
                    if col == "RPE" and pd.notna(row["RPE"]):
                        val = float(row["RPE"])
                        norm = (val - 1) / (10 - 1)
                        norm = min(max(norm, 0), 1)
                        color = mcolors.rgb2hex(cmap(norm))
                        cell_color = color
                        cell_text_color = "#000000"
            
                    # ---------- Moyenne (SURCHARGE) ----------
                    elif row['Name'] == 'Moyenne':
                        cell_color = '#EDE8E8'
                        cell_text_color = '#0031E3'
            
                    elif row['Name'].startswith('Moyenne ') and row['Name'] != 'Moyenne':
                        cell_color = '#CFB013'
                        cell_text_color = '#000000'
            
                    # ---------- % columns (objective coloring) ----------
                    elif col.endswith('%') and not row['Name'].startswith('Moyenne'):
                        stat = col.replace(' %', '')
                        val = row[col]
                        obj = objectives.get(stat, None)
                        if pd.notna(val) and obj is not None:
                            d = abs(val - obj)
                            if d <= 5:
                                cell_color = '#c8e6c9'
                            elif d <= 10:
                                cell_color = '#fff9c4'
                            elif d <= 15:
                                cell_color = '#ffe0b2'
                            else:
                                cell_color = '#ffcdd2'
            
                    # ---------- Alternance pour tout le reste ----------
                    if cell_color is None:
                        cell_color = '#EDE8E8' if (row_idx - 1) % 2 == 0 else 'white'
                    # PDF style
                    try:
                        cell_styles.append(('BACKGROUND', (col_idx, row_idx), (col_idx, row_idx), HexColor(cell_color)))
                    except:
                        pass
                    if cell_text_color:
                        try:
                            cell_styles.append(('TEXTCOLOR', (col_idx, row_idx), (col_idx, row_idx), HexColor(cell_text_color)))
                        except:
                            pass
                    elif row['Name'].startswith('Moyenne ') and row['Name'] != 'Moyenne':
                        cell_styles.append(('TEXTCOLOR', (col_idx, row_idx), (col_idx, row_idx), colors.black))



            # Header row style
            cell_styles += [
                ('BACKGROUND', (0, 0), (-1, 0), HexColor('#0031E3')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold')
            ]
            
            base_styles = [
                ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, 0), 4),
                ('FONTSIZE', (0, 1), (-1, -1), 6),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('LEFTPADDING', (0, 0), (-1, -1), 2),
                ('RIGHTPADDING', (0, 0), (-1, -1), 2),
                ('TOPPADDING', (0, 0), (-1, -1), 2),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
            ]
            
            pdf_tbl = Table(data_pdf, colWidths=[doc.width / ncols] * ncols, repeatRows=1)
            pdf_tbl.hAlign = 'CENTER'
            pdf_tbl.setStyle(TableStyle(base_styles + cell_styles))
            
            elements = [hdr_tbl, Spacer(1, 8), pdf_tbl]
            doc.build(elements)
            st.download_button(
                label="📥 Télécharger le PDF", data=buf.getvalue(),
                file_name=f"Entrainement_{sel_date.strftime('%Y%m%d')}.pdf", mime="application/pdf"
            )
                                                                     
    # ── 2) PERFORMANCES DÉTAILLÉES (date range + filters) ──────────────────

    st.markdown("#### 📊 Analyse collective")
    
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
            agg_options = {
                "Jour": "day",
                "Semaine": "week",
                "Mois": "month"
            }
            x_axis_mode = st.selectbox(
                f"Regrouper par :", list(agg_options.keys()), key=f"xaxis_{i}"
            )
            agg_mode = agg_options[x_axis_mode]
        with colx:
            YVARS = [
                "Duration", "Distance", "M/min", "Distance 15km/h", "M/min 15km/h",
                "Distance 15-20km/h", "Distance 20-25km/h", "Distance 25km/h",
                "Distance 90% Vmax", "N° Sprints", "Vmax", "%Vmax", "Acc", "Dec", "Amax", "Dmax",
                "HSR", "HSR/min", "SPR", "SPR/min", "HSPR", "HSPR/min", "Dist Acc", "Dist Dec"
            ]
            sel_y = st.multiselect(
                f"Variable(s) à afficher (max 2) – Graphique {i}",
                options=[v for v in YVARS if v in train_data.columns],
                default=["Distance"],
                max_selections=2,
                key=f"yvar_{i}"
            )
        # Filtrage
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
        # Regroupement
        if "Date" in filt.columns and sel_y:
            if agg_mode == "day":
                filt["XGroup"] = filt["Date"].dt.date
                label_func = lambda d: d.strftime("%d.%m") if not pd.isnull(d) else ""
            elif agg_mode == "week":
                filt["XGroup"] = filt["Date"].dt.strftime("%G-W%V")
                label_func = lambda d: str(d)
            elif agg_mode == "month":
                filt["XGroup"] = filt["Date"].dt.strftime("%Y-%m")
                label_func = lambda d: str(d)
            else:
                filt["XGroup"] = filt["Date"].dt.date
                label_func = lambda d: str(d)
            grp = filt.groupby("XGroup")[sel_y].mean().sort_index()
        else:
            grp = None
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
                text_auto='.0f',
                color_discrete_sequence=color_sequence
            )
            fig.update_traces(
                textposition='outside',
                textfont_size=10,
                textangle=0,
                cliponaxis=False
            )
            fig.update_layout(
                xaxis_tickangle=0,
                height=600,
                xaxis_title=x_axis_mode,
                yaxis_title="Valeur collective",
                xaxis=dict(
                    tickmode='array',
                    tickvals=grp_plot["XGroup"],
                    ticktext=grp_plot["X_fmt"],
                ),
                margin=dict(t=40, b=30, l=40, r=30)
            )
            st.plotly_chart(fig, use_container_width=True, key=f"plot_{i}")
        else:
            st.info(f"Aucune donnée pour ce graphique selon ces filtres ou variable non sélectionnée.")
            
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

    st.subheader("⚽ Performances en match")

    # 1) Filter to GAME rows
    mask = (
        data["Type"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.upper() == "GAME"
    )
    match_data = data[mask].copy()

    # 2) Ensure Date is datetime
    if not pd.api.types.is_datetime64_any_dtype(match_data["Date"]):
        match_data["Date"] = pd.to_datetime(match_data["Date"], errors="coerce")
  
    # 3) Date filter (select only one, default: latest)
    available_dates = sorted(match_data["Date"].dt.date.dropna().unique())
    if available_dates:
        default_last = available_dates[-1]
        selected_date = st.selectbox(
            "Choisissez un match",
            options=available_dates,
            format_func=lambda d: d.strftime("%d/%m/%Y"),
            index=len(available_dates) - 1  # default: last date
        )
        match_data = match_data[match_data["Date"].dt.date == selected_date]
    else:
        match_data = match_data.iloc[:0]


    # 4) Prepare & clean/cast French‐formatted numbers for display
    cols = [
        "Name", "Duration", "Distance", "M/min",
        "Distance 15km/h", "M/min 15km/h",
        "Distance 15-20km/h", "Distance 20-25km/h",
        "Distance 25km/h", "N° Sprints", "Acc", "Dec",
        "Vmax", "Distance 90% Vmax"
    ]
    df = match_data.loc[:, cols].copy()
    stat_cols = [c for c in cols if c != "Name"]

    for c in stat_cols:
        if c in df.columns:
            cleaned = (
                df[c].astype(str)
                      .str.replace(r"[^\d\-,\.]", "", regex=True)
                      .str.replace(",", ".", regex=False)
                      .replace("", pd.NA)
            )
            num = pd.to_numeric(cleaned, errors="coerce")
        else:
            num = pd.Series(pd.NA, index=df.index)

        if c == "Vmax":
            df[c] = num.round(1)
        else:
            df[c] = num.round(0).astype("Int64")

    # 5) Render the first scrollable table
    html_table = df.to_html(index=False, classes="centered-table")
    row_h = 35
    total_h = max(300, len(df) * row_h)
    html = f"""
    <html><head>
      <style>
        .centered-table {{ border-collapse:collapse; width:100%; }}
        .centered-table th, .centered-table td {{
          text-align:center; padding:4px 8px; border:1px solid #ddd;
        }}
        .centered-table th {{ background:#f0f0f0; }}
      </style>
    </head><body>
      <div style="max-height:{total_h}px;overflow-y:auto;">
        {html_table}
      </div>
    </body></html>
    """
    components.html(html, height=total_h + 1, scrolling=True)

    # ── Référence Match ──
    match_df = data[mask].copy()
    if match_df.empty:
        st.info("Aucune donnée de match pour construire la référence.")
    else:
        # A) clean & numeric‐cast reference data
        for c in stat_cols:
            if c in match_df.columns:
                cleaned = (
                    match_df[c].astype(str)
                               .str.replace(r"[^\d\-,\.]", "", regex=True)
                               .str.replace(",", ".", regex=False)
                               .replace("", pd.NA)
                )
                match_df[c] = pd.to_numeric(cleaned, errors="coerce")
            else:
                match_df[c] = pd.NA

        # B) build per‐player reference
        records = []
        for name, grp in match_df.groupby("Name"):
            rec = {"Name": name}
            full = grp[grp["Duration"] >= 90]
            if not full.empty:
                # full‐length games: take the max of each stat
                for c in stat_cols:
                    rec[c] = full[c].max()
            else:
                # only partial games: scale some stats, copy others
                longest = grp.loc[grp["Duration"].idxmax()]
                orig = longest["Duration"]
                rec["Duration"] = orig

                for c in stat_cols:
                    val = longest[c]
                    if c in {"Duration","Vmax", "M/min", "M/min 15km/h"}:
                        # copy raw value for these three
                        rec[c] = val
                    elif pd.notna(val) and orig > 0:
                        # scale everything else to 90'
                        rec[c] = 90 * val / orig
                    else:
                        rec[c] = pd.NA

            records.append(rec)

        Refmatch = pd.DataFrame.from_records(records, columns=["Name"] + stat_cols)

        # C) Round & cast types
        for c in stat_cols:
            if c == "Vmax":
                Refmatch[c] = Refmatch[c].round(1)
            else:
                Refmatch[c] = Refmatch[c].round(0).astype("Int64")

        # D) Display
        st.subheader("🏆 Référence Match")
        st.dataframe(Refmatch, use_container_width=True)


        st.subheader("🎯 Objectifs Match")
        
        objective_fields = [
            "Duration", "Distance", "Distance 15km/h", "Distance 15-20km/h",
            "Distance 20-25km/h", "Distance 25km/h", "Acc", "Dec", "Vmax", "Distance 90% Vmax"
        ]
        
        # 1) Sliders in 2×5 grid
        row1, row2 = objective_fields[:5], objective_fields[5:]
        objectives = {}
        cols5 = st.columns(5)
        for i, stat in enumerate(row1):
            with cols5[i]:
                objectives[stat] = st.slider(f"{stat} (%)", 0, 100, 100, key=f"obj_{stat}")
        cols5 = st.columns(5)
        for i, stat in enumerate(row2):
            with cols5[i]:
                objectives[stat] = st.slider(f"{stat} (%)", 0, 100, 100, key=f"obj_{stat}")
        
        # 2) Compute % of personal reference per match row
        obj_df = df.copy()
        ref_indexed = Refmatch.set_index("Name")
        for c in objective_fields:
            obj_df[f"{c} %"] = obj_df.apply(
                lambda r: round(r[c] / ref_indexed.at[r["Name"], c] * 100, 1)
                          if (r["Name"] in ref_indexed.index
                              and pd.notna(ref_indexed.at[r["Name"], c]) 
                              and ref_indexed.at[r["Name"], c] > 0
                              and pd.notna(r[c]))
                          else pd.NA,
                axis=1
            )
        
        # 3) Highlight helper
        def highlight_stat(val, obj):
            if pd.isna(val):
                return ""
            d = abs(val - obj)
            if d <= 5:   return "background-color: #c8e6c9;"
            if d <= 10:  return "background-color: #fff9c4;"
            if d <= 15:  return "background-color: #ffe0b2;"
            if d <= 20:  return "background-color: #ffcdd2;"
            return ""
        
        # 4) Build & render styled table
        display_cols = ["Name"] + sum([[c, f"{c} %"] for c in objective_fields], [])
        styled = (
            obj_df.loc[:, display_cols]
                  .style
                  .format(
                      { **{f"{c} %": "{:.1f} %" for c in objective_fields},
                        **{"Vmax": "{:.1f}"}                # one decimal for raw Vmax
                      },
                      na_rep="—"
                  )
        )
        
        # apply per-column highlighting only on the % columns
        for c in objective_fields:
            styled = styled.applymap(
                lambda v, obj=objectives[c]: highlight_stat(v, obj),
                subset=[f"{c} %"]
            )
        
        styled = styled.set_table_attributes('class="centered-table"')
        
        # 5) wrap in scrollable div
        html_obj = styled.to_html()
        row_h = 35
        total_h2 = max(300, len(obj_df) * row_h)
        html2 = f"""
        <html><head>
          <style>
            .centered-table {{ border-collapse:collapse; width:100%; }}
            .centered-table th, .centered-table td {{
              text-align:center; padding:4px 8px; border:1px solid #ddd;
            }}
            .centered-table th {{ background:#f0f0f0; }}
          </style>
        </head><body>
          <div style="max-height:{total_h2}px;overflow-y:auto;">
            {html_obj}
          </div>
        </body></html>
        """
        components.html(html2, height=total_h2 + 20, scrolling=True)


# ── PAGE: PLAYER ANALYSIS ────────────────────────────────────────────────────
elif page == "Joueurs":
    st.subheader("🔎 Analyse d'un joueur")
    players = sorted(data["Name"].dropna().unique())
    sel = st.selectbox("Choisissez un joueur", players)
    p_df = data[data["Name"] == sel]
    g_df = p_df[p_df["Type"] == "GAME"]
    if not g_df.empty:
        wmin = g_df["Semaine"].min()
        wmax = g_df["Semaine"].max()
        weeks = pd.DataFrame({"Semaine": range(wmin, wmax + 1)})
        mins  = g_df.groupby("Semaine")["Duration"].sum().reset_index()
        merged = weeks.merge(mins, on="Semaine", how="left").fillna(0)
        fig = px.bar(merged, x="Semaine", y="Duration", title=f"{sel} – Minutes par semaine")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Pas de données de match pour ce joueur.")
