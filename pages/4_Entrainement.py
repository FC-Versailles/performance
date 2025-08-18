#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 21:09:07 2025

@author: fcvmathieu
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import os
import pickle
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from datetime import date
import numpy as np
import re, unicodedata
import textwrap
import ollama

# ─────────────────────────── Config ───────────────────────────
st.set_page_config(page_title="Entrainement | FC Versailles", layout='wide')

logo_url = 'https://raw.githubusercontent.com/FC-Versailles/care/main/logo.png'

# Google Sheets constants
SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
TOKEN_FILE_EN = 'token_ent.pickle'
SPREADSHEET_ID_EN  = '15n4XkQHrUpSPAW61vmR_Rk1kibd5zcmVqgHA40szlPg'   # Entrainement
SPREADSHEET_ID_EN2 = '1fY6624a0xdu7g8Hm59Qm9Jw0xglBRoH-fBErQtQoRfc'    # Daily
RANGE_NAME = 'Feuille 1'

# ─────────────────────── Auth & Fetch helpers ───────────────────────
def get_en_credentials():
    creds = None
    if os.path.exists(TOKEN_FILE_EN):
        with open(TOKEN_FILE_EN, 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('client_secret.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE_EN, 'wb') as token:
            pickle.dump(creds, token)
    return creds

def fetch_google_sheet(spreadsheet_id, range_name):
    def make_unique_columns(cols):
        seen = {}
        out = []
        for c in cols:
            key = c if c is not None else ""
            if key in seen:
                seen[key] += 1
                out.append(f"{key}__{seen[key]}")  # e.g. "Commentaire (a)__1"
            else:
                seen[key] = 0
                out.append(key)
        return out

    creds = get_en_credentials()
    service = build('sheets', 'v4', credentials=creds)
    result = service.spreadsheets().values().get(
        spreadsheetId=spreadsheet_id, range=range_name
    ).execute()

    values = result.get('values', [])
    if not values:
        return pd.DataFrame()

    header = values[0]
    rows = values[1:]
    max_cols = len(header)
    rows = [r + [None]*(max_cols - len(r)) if len(r) < max_cols else r[:max_cols] for r in rows]

    df = pd.DataFrame(rows, columns=header)
    df.columns = make_unique_columns(list(df.columns))  # <- ensure uniqueness
    return df

# ───────────────────────── Cache loaders ─────────────────────────
@st.cache_data(ttl=60)
def load_data_entrainement():
    df = fetch_google_sheet(SPREADSHEET_ID_EN, RANGE_NAME)
    if not df.empty and 'Type' in df.columns:
        df = df[~df['Type'].isin(['Salle', 'Dev Individuel'])]
    return df

@st.cache_data(ttl=60)
def load_data_daily():
    df = fetch_google_sheet(SPREADSHEET_ID_EN2, RANGE_NAME)

    # Colonnes à retirer
    cols_to_remove = [
        'Submission ID', 'Respondent ID', 'Submitted at',
        'Moment', 'Type'
    ]

    # Supprimer uniquement celles qui existent vraiment
    df = df.drop(columns=[c for c in cols_to_remove if c in df.columns], errors='ignore')

    return df

# ───────────────────────── UI Header ─────────────────────────
col1, col2 = st.columns([9, 1])
with col1:
    st.title("FC Versailles | Analyses")
with col2:
    st.image(logo_url, use_container_width=True)
st.markdown("<hr style='border:1px solid #ddd' />", unsafe_allow_html=True)

# ───────────────────────── Page selector (sidebar OK) ─────────────────────────
# Sidebar page selector (dropdown instead of radio)

page = st.sidebar.selectbox("Page", ["Analyse daily","Analyse entrainement"], index=0)


# ───────────────────────── Common helpers ─────────────────────────
def coerce_date(df, colname="Date"):
    if colname in df.columns:
        df[colname] = pd.to_datetime(df[colname], errors='coerce')
    return df

def plot_treemap_from_activity_cols(filtered_df):
    activity_cols = [c for c in filtered_df.columns if str(c).startswith('Temps')]
    if not activity_cols:
        return
    melted = filtered_df.melt(id_vars=[], value_vars=activity_cols,
                              var_name='Temps', value_name='Activité').dropna()
    if 'Activité' not in melted.columns:
        return
    counts = melted[(melted['Activité'] != 'RAS') & (melted['Activité'] != 'Prévention')]['Activité'].value_counts()
    if counts.empty:
        return
    labels = [f"{lab}\n{val}" for lab, val in zip(counts.index, counts.values)]
    st.plotly_chart(go.Figure(go.Treemap(labels=labels, parents=['']*len(counts), values=counts.values)),
                    use_container_width=True)

def plot_stacked_by_type(filtered_df):
    activity_cols = [c for c in filtered_df.columns if str(c).startswith('Temps')]
    if not activity_cols or 'Type' not in filtered_df.columns:
        return
    melted = filtered_df.melt(id_vars=["Type"], value_vars=activity_cols,
                              var_name="Temps", value_name="Activité").dropna()
    melted = melted[melted['Activité'] != 'RAS']
    if melted.empty:
        return
    pivot = melted.groupby(['Type', 'Activité']).size().unstack(fill_value=0)
    fig = go.Figure()
    for col in pivot.columns:
        fig.add_trace(go.Bar(y=pivot.index, x=pivot[col], name=col, orientation='h'))
    fig.update_layout(barmode='stack', xaxis_title="Nombre d'apparitions", yaxis_title="Type",
                      legend_title_text='Activité', legend=dict(x=1.02, y=1))
    st.plotly_chart(fig, use_container_width=True)

def plot_scatter_date_content(filtered_df):
    if 'Date' not in filtered_df.columns:
        return
    activity_cols = [c for c in filtered_df.columns if str(c).startswith('Temps')]
    if not activity_cols:
        return
    melted = filtered_df.melt(id_vars=['Date'], value_vars=activity_cols,
                              var_name='Temps', value_name='Activité').dropna()
    melted = melted[melted['Activité'] != 'RAS']
    if melted.empty:
        return
    fig = px.scatter(melted, x='Date', y='Activité', color='Activité', size_max=10)
    fig.update_traces(marker=dict(symbol='square', size=10))
    st.plotly_chart(fig, use_container_width=True)

# ───────────────────────── Page: Analyse entrainement ─────────────────────────
def render_entrainement():
    # Bouton Questionnaire

    data = load_data_entrainement()
    if data.empty:
        st.error("Aucune donnée d'entraînement chargée.")
        return

    data = coerce_date(data, "Date")

    # ── Filtres DANS LA PAGE (pas dans la sidebar)
    with st.container():
        st.subheader("Filtres")
        c1, c2 = st.columns([2, 3])

        # Type(s) de séance
        if 'Type' in data.columns:
            with c1:
                all_types = sorted([t for t in data['Type'].dropna().unique().tolist()])
                session_types = st.multiselect(
                    "Type de séance",
                    options=all_types,
                    default=all_types
                )
        else:
            session_types = None

        # Période
        with c2:
            start_default = date(2025, 7, 1)
            if data['Date'].dropna().empty:
                st.warning("Pas de dates valides dans les données.")
                return
            end_default = data['Date'].max().date()
            date_range = st.date_input(
                "Période",
                [start_default, end_default],
                min_value=start_default,
                max_value=end_default
            )

    # Application des filtres
    filtered = data.copy()
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        filtered = filtered[(filtered['Date'] >= pd.to_datetime(date_range[0])) &
                            (filtered['Date'] <= pd.to_datetime(date_range[1]))]
    if session_types is not None:
        filtered = filtered[filtered['Type'].isin(session_types)]

    # # Vue table + visuels
    # st.markdown("### Données filtrées (Entraînement)")
    # st.dataframe(filtered, use_container_width=True)

    st.markdown("### Répartition des activités")
    plot_treemap_from_activity_cols(filtered)

    st.markdown("### Répartition des procédés par type d'entraînement")
    plot_stacked_by_type(filtered)

    st.markdown("### Répartition des contenus par date")
    plot_scatter_date_content(filtered)

# ───────────────────────── Page: Analyse daily ─────────────────────────
def render_daily():
    
    st.markdown(
        '''
        <a href="https://tally.so/r/3ql17d" target="_blank" style="
            display: inline-block; padding: 10px 18px; background-color: #2563eb;
            color: white; text-decoration: none; border-radius: 8px; font-weight: 600; margin-bottom: 12px;">
            Questionnaire
        </a>
        ''', unsafe_allow_html=True
    )

    data2 = load_data_daily()
    if data2.empty:
        st.error("Aucune donnée 'daily' chargée.")
        return

    # Find candidate date columns (e.g., 'Date', 'Date__1', etc.)
    date_cols = [c for c in data2.columns if str(c).strip().lower().startswith('date')]
    if not date_cols:
        st.error("Aucune colonne 'Date' trouvée dans les données daily.")
        return

    # Coerce all candidate date columns to datetime, pick the one with most non-null values
    for c in date_cols:
        data2[c] = pd.to_datetime(data2[c], errors='coerce')

    primary_date_col = max(date_cols, key=lambda c: data2[c].notna().sum())
    # Alias chosen column to 'Date' for downstream consistency
    if primary_date_col != 'Date':
        data2['Date'] = data2[primary_date_col]

    valid_dates = data2['Date'].dropna()
    if valid_dates.empty:
        st.warning("Pas de dates valides dans les données daily.")
        return

    # ── UNIQUE FILTRE : Date unique
    min_d = valid_dates.min().date()
    max_d = valid_dates.max().date()
    sel_date = st.date_input("Date", value=max_d, min_value=min_d, max_value=max_d)

    filtered = data2[data2['Date'].dt.date == sel_date].copy()
    if filtered.empty:
        st.info("Aucune ligne pour la date sélectionnée.")
        return

    def _norm(s: str) -> str:
        """minuscule, sans accents, espaces normalisés, conserve (a)/(b)."""
        s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
        s = s.lower().strip()
        s = re.sub(r"[^a-z0-9()]+", " ", s)  # retire tout sauf lettres/chiffres/()
        s = re.sub(r"\s+", " ", s).strip()
        return s
    
    def _find_best_col(df_cols, target_label):
        """Retourne le nom de colonne du DF qui correspond au label cible."""
        target = _norm(target_label)
        norm_map = {c: _norm(c) for c in df_cols}
    
        # 1) correspondance exacte
        for col, nc in norm_map.items():
            if nc == target:
                return col
        # 2) tolère pluriel/singulier (retire 's' final)
        if target.endswith("s"):
            t2 = target[:-1]
            for col, nc in norm_map.items():
                if nc == t2:
                    return col
        # 3) tolère suffixes pandas (ex: '... (a).1') via startswith
        for col, nc in norm_map.items():
            if nc.startswith(target):
                return col
        # 4) startswith sans 's' final
        t3 = target[:-1] if target.endswith("s") else target
        for col, nc in norm_map.items():
            if nc.startswith(t3):
                return col
        return None
    
    wanted = [
        "Type", "Coach",
        "Représentativité (a)", "Représentativité (b)",
        "Contrainte (a)", "Contraintes (b)",
        "Différentielle (a)", "Différentielle (b)",
        "Défi (a)", "Défi (b)",
        "Connaissance (a)", "Connaissance (b)",
        "Pression (a)", "Pression (b)",
        "video (a)", "video (b)"
    ]
    
    matched_cols = []
    display_names = []
    missing = []
    
    for w in wanted:
        col = _find_best_col(filtered.columns, w)
        if col and col not in matched_cols:
            matched_cols.append(col)
            display_names.append(w)  # on renomme avec le label propre
        else:
            missing.append(w)
    
    if matched_cols:
        df_show = filtered.loc[:, matched_cols].copy()
        df_show.columns = display_names
        st.dataframe(df_show, use_container_width=True)
    else:
        st.info("Aucune des colonnes ciblées n'a été trouvée dans les données.")
    
    # Optionnel : indiquer les colonnes manquantes
    if missing:
        st.caption("Colonnes non trouvées : " + ", ".join(missing))
    
        # Visuels optionnels si colonnes 'Temps*' présentes
        has_temps_cols = any(str(c).startswith('Temps') for c in filtered.columns)
        if has_temps_cols:
            st.markdown("### Répartition des activités (Daily)")
            plot_treemap_from_activity_cols(filtered)
    
            st.markdown("### Répartition par Type (Daily)")
            plot_stacked_by_type(filtered)
    
            st.markdown("### Contenus par date (Daily)")
            plot_scatter_date_content(filtered)
    
    def _norm(s: str) -> str:
        s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
        s = s.lower().strip()
        s = re.sub(r"[^a-z0-9()]+", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s
    
    def _find_cols(df, targets, startswith=False):
        out = {}
        norm_map = {c: _norm(c) for c in df.columns}
        for t in targets:
            nt = _norm(t)
            found = None
            # exact
            for col, nc in norm_map.items():
                if nc == nt:
                    found = col; break
            # startswith (tolère suffixes .1)
            if not found:
                for col, nc in norm_map.items():
                    if (nc.startswith(nt) if startswith else nc == nt) or nc.startswith(nt):
                        found = col; break
            out[t] = found
        return out
    
    def _collect_prefixed_cols(df, prefix_norm: str):
        """Retourne toutes les colonnes dont le nom normalisé commence par prefix_norm."""
        cols = []
        for c in df.columns:
            if _norm(c).startswith(prefix_norm):
                cols.append(c)
        return cols
    
    def _to_float(series: pd.Series) -> pd.Series:
        if series.dtype.kind in "biufc":
            return pd.to_numeric(series, errors="coerce")
        return pd.to_numeric(series.astype(str).str.replace(",", ".", regex=False), errors="coerce")
    
    def build_training_report(filtered: pd.DataFrame) -> str:
        # ---------- Colonnes de contexte ----------
        ctx_cols = _find_cols(filtered, ["Date","Moment","Type","Coach"])
        date_col   = ctx_cols["Date"]
        moment_col = ctx_cols["Moment"]
        type_col   = ctx_cols["Type"]
        coach_col  = ctx_cols["Coach"]
    
        # ---------- Colonnes notes (1–3) ----------
        kpi_labels = [
            "Représentativité (a)", "Représentativité (b)",
            "Contrainte (a)", "Contraintes (b)",
            "Différentielle (a)", "Différentielle (b)",
            "Défi (a)", "Défi (b)",
            "Connaissance (a)", "Connaissance (b)",
            "Pression (a)", "Pression (b)",
            "Vidéo (a)", "Vidéo (b)"
        ]
        kpi_map = _find_cols(filtered, kpi_labels, startswith=True)  # tolère variantes
        kpi_found = {k:v for k,v in kpi_map.items() if v is not None}
    
        # Numérise les KPI trouvés
        df_num = filtered.copy()
        for lab, col in kpi_found.items():
            df_num[col] = _to_float(df_num[col])
    
        # ---------- Colonnes commentaires ----------
        # On prend toutes les variantes: "Commentaire (a)", "Commentaire (b)" + duplicatas .1/.2
        com_a_cols = _collect_prefixed_cols(filtered, _norm("Commentaire (a)"))
        com_b_cols = _collect_prefixed_cols(filtered, _norm("Commentaire (b)"))
        comment_cols = com_a_cols + com_b_cols
        # Colonne texte consolidée
        def _row_comments(row):
            parts = []
            for c in comment_cols:
                val = row.get(c, None)
                if pd.notna(val) and str(val).strip():
                    who = row.get(coach_col, "Coach")
                    parts.append((str(who), str(val).strip()))
            return parts
    
        # ---------- En-tête / contexte ----------
        # date unique ou plage si plusieurs
        if date_col and date_col in filtered.columns:
            try:
                dts = pd.to_datetime(filtered[date_col], errors="coerce").dt.date.dropna()
            except Exception:
                dts = pd.Series([], dtype="object")
        else:
            dts = pd.Series([], dtype="object")
        date_str = ""
        if not dts.empty:
            dmin, dmax = dts.min(), dts.max()
            date_str = f"du {dmin}" if dmin != dmax else f"du {dmin}"
            if dmin != dmax:
                date_str = f"du {dmin} au {dmax}"
    
        moments = sorted(set(str(x).strip() for x in filtered.get(moment_col, pd.Series(dtype=object)).dropna()))
        types   = sorted(set(str(x).strip() for x in filtered.get(type_col, pd.Series(dtype=object)).dropna()))
        coachs  = [str(x).strip() for x in filtered.get(coach_col, pd.Series(dtype=object)).dropna()]
        coachs_unique = sorted(set(coachs))
    
        # ---------- Statistiques globales ----------
        n_feedback = len(filtered)
        # Moyennes par critère (tri décroissant)
        crit_means = []
        for lab, col in kpi_found.items():
            m = df_num[col].mean(skipna=True)
            if pd.notna(m):
                crit_means.append((lab, m))
        crit_means.sort(key=lambda x: (-x[1], x[0]))
    
        # ---------- Moyennes par coach & forces/faiblesses ----------
        per_coach = {}
        for c in coachs_unique:
            sub = df_num[df_num[coach_col].astype(str).str.strip() == c] if coach_col in df_num.columns else df_num.iloc[0:0]
            means = {lab: sub[col].mean(skipna=True) for lab, col in kpi_found.items()}
            per_coach[c] = means
    
        # Différences & complémentarités
        # Force: critère où coach > moyenne globale + 0.3 ; Faiblesse: < moyenne globale - 0.3
        global_means = {lab: np.nanmean(df_num[col]) for lab, col in kpi_found.items()}
        strengths = {}
        weaknesses = {}
        for c, means in per_coach.items():
            s, w = [], []
            for lab, m in means.items():
                g = global_means.get(lab, np.nan)
                if pd.notna(m) and pd.notna(g):
                    if m >= g + 0.3:
                        s.append(lab)
                    elif m <= g - 0.3:
                        w.append(lab)
            strengths[c] = s
            weaknesses[c] = w
    
        # Paires complémentaires: A fort sur X et B faible sur X
        pairs = []
        for lab in global_means.keys():
            strong = [c for c in coachs_unique if lab in strengths.get(c, [])]
            weak   = [c for c in coachs_unique if lab in weaknesses.get(c, [])]
            for a in strong:
                for b in weak:
                    if a != b:
                        pairs.append((lab, a, b))
    
        # ---------- Regroupement thématique des commentaires ----------
        # Règles simples par mots-clés (tu pourras ajuster)
        theme_keywords = {
            "Représentativité & structure": ["representativ", "structure", "carr", "positionnement", "jdp", "grand jeu"],
            "Contraintes & règles": ["contrainte", "regle", "scores", "points", "limitation"],
            "Différentielle & intensité": ["differenti", "intens", "charge", "rythme", "effort"],
            "Défi & créativité": ["defi", "creativ", "initiative", "prise de risque", "oser"],
            "Connaissance & compréhension": ["connaiss", "comprehension", "principe", "references", "concept"],
            "Pression & comportement": ["pression", "comport", "stress", "tolérance a l echec", "echec"],
            "Vidéo & préparation": ["video", "analyse video", "preview", "avant seance"],
            "Corrections & autonomie": ["correctif", "correction", "autonomie", "brief", "feedback"]
        }
        def detect_theme(txt: str) -> str:
            n = _norm(txt)
            for theme, kws in theme_keywords.items():
                for kw in kws:
                    if kw in n:
                        return theme
            return "Autres"
    
        themed = {}  # theme -> list[(coach, comment)]
        for _, row in filtered.iterrows():
            who = str(row.get(coach_col, "Coach")).strip()
            for c in comment_cols:
                val = row.get(c, None)
                if pd.notna(val):
                    txt = str(val).strip()
                    if txt:
                        th = detect_theme(txt)
                        themed.setdefault(th, []).append((who, txt))
    
        # ---------- Construction du Markdown ----------
        lines = []
        lines.append(f"### Rapport de séance {date_str}")
        lines.append("")
        # Synthèse générale
        lines.append("#### 📊 Synthèse générale")
        synth = {
            "Nombre total de retours": n_feedback,
            "Coachs ayant répondu": ", ".join(coachs_unique) if coachs_unique else "—",
            "Moment couvert": ", ".join(moments) if moments else "—",
            "Type de séance": ", ".join(types) if types else "—",
        }
        for k,v in synth.items():
            lines.append(f"- **{k}** : {v}")
        lines.append("")
    
        # Évaluations chiffrées
        if crit_means:
            lines.append("#### 📈 Classement des critères par moyenne (1 = moindre, 3 = très positif)")
            lines.append("")
            for lab, m in crit_means:
                lines.append(f"- **{lab}** : {m:.2f}")
            lines.append("")
        else:
            lines.append("*(Aucun critère chiffré disponible)*\n")
    
    
        # Observations qualitatives par thème et coach
        lines.append("#### 💬 Observations qualitatives par thème et coach")
        if themed:
            for th in sorted(themed.keys()):
                lines.append(f"**{th}**")
                for who, txt in themed[th]:
                    # wrap léger pour lisibilité
                    txt_w = textwrap.fill(txt, width=110)
                    lines.append(f"- **{who}** : {txt_w}")
                lines.append("")
        else:
            lines.append("*(Aucun commentaire saisi)*\n")
    
        # Axes d'amélioration — à partir des faiblesses récurrentes
        lines.append("#### 📌 Axes d’amélioration (d’après tendances)")
        if weaknesses:
            # Compter par critère combien de coachs l'ont en faiblesse
            counts = {}
            for c, ws in weaknesses.items():
                for lab in ws:
                    counts[lab] = counts.get(lab, 0) + 1
            if counts:
                top_issues = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:6]
                for lab, n in top_issues:
                    lines.append(f"- **{lab}** : {n} coach(s) en-dessous de la moyenne — prévoir focus spécifique")
            else:
                lines.append("- RAS (aucun critère en-dessous de la moyenne globale)")
        else:
            lines.append("- RAS")
        lines.append("")
    
        return "\n".join(lines)


    # ====== Utilisation dans l'app ======
    report_md = build_training_report(filtered)
    st.markdown(report_md)
    
        # --- Partie 2 : Question personnalisée ---
# --- Partie 2 : Question personnalisée ---
# --- Partie 2 : Question personnalisée ---
    st.markdown("### ❓ Poser une question personnalisée (LLM open-source)")
    
    user_q = st.text_area(
        "Votre question en français :",
        placeholder="Exemple : Quels sont les points de divergence entre Marcelo et Mathieu ?"
    )
    
    if st.button("Poser la question à l'IA (Ollama)"):
        if not user_q.strip():
            st.warning("Veuillez saisir une question.")
        else:
            prompt = f"""
Tu es un facilitateur de réunion de staff en football professionnel.
Voici le rapport de séance :

{report_md}

Question de l'utilisateur :
{user_q}

Réponds en français, de manière claire, concise et structurée, en citant les coachs/critères pertinents.
"""

            try:
                resp = ollama.chat(
                    model="llama3.1",  # ⚠️ Assure-toi d'avoir fait `ollama pull llama3.1`
                    messages=[
                        {"role": "system", "content": "Tu aides à l'analyse et la mise en valeur des retours d'entraînement."},
                        {"role": "user", "content": prompt},
                    ],
                )
                answer = resp["message"]["content"].strip()
                st.markdown("### 💡 Réponse de l'IA")
                st.markdown(answer)

            except Exception as e:
                st.error(f"Erreur Ollama : {e}\n⚠️ Vérifie que Ollama est installé et lancé (`ollama serve`).")


# Choix de la page
if page == "Analyse entrainement":
    render_entrainement()
else:
    render_daily()

