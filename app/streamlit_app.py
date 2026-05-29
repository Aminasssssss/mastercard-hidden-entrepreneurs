# -*- coding: utf-8 -*-
"""Mastercard Data Quest — Streamlit Dashboard (entry point)."""
import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="MDQ — Hidden Entrepreneurs",
    
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.big-metric {font-size: 2.4rem; font-weight: 700; color: #EB001B;}
.metric-label {color: #555; font-size: .95rem;}
[data-testid="stMetricValue"] {font-size: 2rem;}
</style>
""", unsafe_allow_html=True)

# ---- header ----
col1, col2 = st.columns([0.15, 0.85])
with col1:
    st.markdown(
        "<div style='font-size: 2.5rem;'>"
        "<span style='color:#EB001B'>●</span>"
        "<span style='color:#F79E1B; margin-left:-15px;'>●</span>"
        "</div>", unsafe_allow_html=True
    )
with col2:
    st.title("Hidden Entrepreneur Detection")
    st.caption("Mastercard Data Quest 2026 · KBTU Team")

st.markdown("---")

st.markdown("""
### Описание

Прототип банковского интерфейса для поиска скрытых предпринимателей среди держателей потребительских карт.
Под капотом — ансамбль из 6 one-class моделей, обученных на бизнес-картах и применённых к потребительским.

### Навигация

- **Dashboard** — список топ-карт, фильтры, KPI, экспорт
- **Card Profile** — профиль одной карты, локальный SHAP
- **Segments** — 4 сегмента и рекомендованные офферы
- **ROI Calculator** — расчёт прибыли от outreach
- **New Card Simulator** — ввод фич, предсказание скора
- **Model Health (PSI)** — мониторинг дрейфа
- **A/B Test Calculator** — мощность пилота
- **About** — методология и Model Card

### Архитектура

- **Streamlit** (фронт) — этот интерфейс
- **FastAPI** (бэк, порт 8000) — `/predict`, `/segment`, `/explain`
- **MLflow** (порт 5555) — трекинг и сравнение моделей
""")

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#888; font-size:.9rem;'>"
    "Synthetic data · Educational purpose · © Mastercard Data Quest 2026"
    "</div>",
    unsafe_allow_html=True
)
