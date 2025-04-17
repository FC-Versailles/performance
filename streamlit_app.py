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

# Fonction de carte avec bouton
def create_card(icon, title, description, page_name):
    st.markdown(f"### {icon} {title}")
    st.write(description)
    if st.button(f"📂 Ouvrir {title}", key=page_name):
        st.switch_page(f"pages/{page_name}")

# Mise en page
col1, col2, col3 = st.columns(3)
with col1:
    create_card("🏥", "Médical", "Suivi des blessures, soins, réathlétisation.", "1_Medical.py")
with col2:
    create_card("🍎", "Nutrition", "Poids, MG %, évolution individuelle.", "2_Nutrition.py")
with col3:
    create_card("🧘", "Wellness", "Sommeil, fatigue, humeur, bien-être global.", "3_Wellness.py")

st.markdown("---")
st.markdown('<div style="text-align:center;">Développé par Mathieu – FC Versailles</div>', unsafe_allow_html=True)