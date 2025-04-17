#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 10:31:50 2025

@author: fcvmathieu
"""

import streamlit as st
from PIL import Image

st.set_page_config(page_title="FC Versailles Dashboard", layout="wide")

# Logo + Titre
col1, col2 = st.columns([8, 1])
with col1:
    st.title("FC Versailles | Performance Dashboard")
with col2:
    st.image("static/logo.jpg", width=100)

st.markdown("---")

def create_card(title, subtitle, icon, page_path):
    st.page_link(
        page=page_path,
        label=f"""
        <div style="
            background-color: #ffffff;
            border: 1px solid #d3d3d3;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
            transition: 0.3s ease;
        ">
            <div style="font-size: 22px; font-weight: 600; margin-bottom: 10px;">
                {icon} {title}
            </div>
            <div style="font-size: 15px; color: #555; margin-bottom: 20px;">
                {subtitle}
            </div>
            <div style="text-align: right;">
                <span style="
                    display: inline-block;
                    background-color: #0066cc;
                    color: white;
                    padding: 6px 14px;
                    border-radius: 8px;
                    font-weight: bold;
                    font-size: 14px;
                ">
                    GO →
                </span>
            </div>
        </div>
        """,
        icon=None,
        use_container_width=True
    )


col1, col2, col3 = st.columns(3)

with col1:
    create_card("Médical", "Données de suivi médical et blessures.", "🏥", "pages/1_Medical.py")
    create_card("Entrainement", "Planification, disponibilité et priorités d'entraînement.", "📅", "pages/4_Entrainement.py")
    create_card("Player Analysis", "Profil individuel des joueurs : stats, vidéos, priorité.", "🧍", "pages/7_Player_Analysis.py")

with col2:
    create_card("Nutrition", "Suivi du poids, MG%, et remarques nutrition.", "🍎", "pages/2_Nutrition.py")
    create_card("GPS", "Analyse des données de charge GPS en match et entraînement.", "📡", "pages/5_GPS.py")
    create_card("Team Analysis", "Indicateurs collectifs liés à la performance et à la tactique.", "🧠", "pages/8_Team_Analysis.py")

with col3:
    create_card("Wellness", "Suivi quotidien pré et post-entrainement.", "🧘", "pages/3_Wellness.py")
    create_card("Game Analysis", "Données des matchs, stats collectives et phases de jeu.", "🎯", "pages/6_Game_Analysis.py")
    



st.markdown("---")
st.markdown('<div style="text-align:center;">Développé par Mathieu – FC Versailles</div>', unsafe_allow_html=True)