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
    st.markdown(
        f"""
        <div style="
            border: 1px solid #ddd;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            height: 200px;
            background-color: #f9f9f9;
            box-shadow: 0px 2px 8px rgba(0,0,0,0.05);
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        ">
            <div>
                <p style="margin:0; font-size: 24px; font-weight: bold;">{icon} {title}</p>
                <p style="margin-top: 6px; font-size: 16px; color: #555;">{subtitle}</p>
            </div>
            <div style="margin-top: 12px;">
                <a href="/{page_path}" target="_self" style="
                    display: inline-block;
                    background-color: #0066cc;
                    color: white;
                    padding: 8px 16px;
                    border-radius: 6px;
                    text-decoration: none;
                    font-weight: bold;
                ">
                    GO →
                </a>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# Grille des cartes
col1, col2, col3 = st.columns(3)

with col1:
    create_card("Médical", "Données de suivi médical et blessures.", "🏥", "pages/1_Medical.py")
with col2:
    create_card("Nutrition", "Suivi du poids, MG%, et remarques nutrition.", "🍎", "pages/2_Nutrition.py")
with col3:
    create_card("Wellness", "Suivi quotidien pré et post-entrainement.", "🧘", "pages/3_Wellness.py")

st.markdown("---")
st.markdown('<div style="text-align:center;">Développé par Mathieu – FC Versailles</div>', unsafe_allow_html=True)