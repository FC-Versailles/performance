#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 10:31:50 2025

@author: fcvmathieu
"""

import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="FC Versailles Dashboard",
    page_icon="static/logo.jpg",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Header
_, col1, col2, _ = st.columns([0.5, 4, 0.5, 0.5])
with col1:
    st.title("Bienvenue sur le portail des dashboards FC Versailles")
with col2:
    st.image("static/logo.jpg", width=100)

st.markdown("---")

# Carte de navigation
def create_card(title, description, icon, script_name):
    st.markdown(
        f"""
        <a href="{script_name}" target="_self" style="text-decoration: none;">
            <div style="
                background-color: #f9f9f9;
                border: 1px solid #ccc;
                border-radius: 10px;
                padding: 20px;
                height: 180px;
                color: black;
            ">
                <h3>{icon} {title}</h3>
                <p>{description}</p>
                <p style="color: #1f77b4;">→ Ouvrir le dashboard</p>
            </div>
        </a>
        """,
        unsafe_allow_html=True
    )

col1, col2, col3 = st.columns(3)
with col1:
    create_card("Médical", "Données de suivi médical et blessures.", "🏥", "1_Medical.py")
with col2:
    create_card("Nutrition", "Suivi du poids, MG%, et remarques nutrition.", "🍎", "2_Nutrition.py")
with col3:
    create_card("Wellness", "Suivi quotidien pré et post-entrainement.", "🧘", "3_Wellness.py")

st.markdown("---")
st.markdown('<div style="text-align:center;">Développé par Mathieu – FC Versailles</div>', unsafe_allow_html=True)
