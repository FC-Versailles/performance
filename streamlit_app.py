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

def create_card(icon, title, description, page_name):
    st.markdown(
        f"""
        <a href="/{page_name}" style="text-decoration: none;">
            <div style="
                background-color: #f9f9f9;
                border: 1px solid #ddd;
                border-radius: 12px;
                padding: 20px;
                margin-bottom: 20px;
                height: 180px;
                transition: box-shadow 0.3s;
            " onmouseover="this.style.boxShadow='0px 4px 20px rgba(0,0,0,0.1)'" onmouseout="this.style.boxShadow='none'">
                <h3 style="color: black;">{icon} {title}</h3>
                <p style="color: #555;">{description}</p>
            </div>
        </a>
        """,
        unsafe_allow_html=True
    )


col1, col2, col3 = st.columns(3)

with col1:
    create_card("🏥", "Médical", "Données de suivi médical et blessures.", "1_Medical")
with col2:
    create_card("🍎", "Nutrition", "Suivi du poids, MG%, et remarques nutrition.", "2_Nutrition")
with col3:
    create_card("🧘", "Wellness", "Suivi quotidien pré et post-entrainement.", "3_Wellness")


st.markdown("---")
st.markdown('<div style="text-align:center;">Développé par Mathieu – FC Versailles</div>', unsafe_allow_html=True)