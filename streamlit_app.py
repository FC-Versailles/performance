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

def create_card(icon, title, description, page_path):
    st.page_link(
        page=page_path,
        label=f"{icon} {title}\n{description}",
        icon=None,
    )



col1, col2, col3 = st.columns(3)

with col1:
    create_card("🏥", "Médical", "Données de suivi médical et blessures.", "pages/1_Medical.py")
with col2:
    create_card("🍎", "Nutrition", "Suivi du poids, MG%, et remarques nutrition.", "pages/2_Nutrition.py")
with col3:
    

st.markdown("---")
st.markdown('<div style="text-align:center;">Développé par Mathieu – FC Versailles</div>', unsafe_allow_html=True)