# -*- coding: utf-8 -*-
"""Page 6: Здоровье модели — PSI дрейф фич."""
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Model Health", layout="wide")
st.title("Model Health — PSI Monitoring")
st.caption("PSI > 0.2 — модель пора переобучать.")

ROOT = Path(__file__).resolve().parents[2]
psi = pd.read_parquet(ROOT/"data"/"psi.parquet")

# ---- статус ----
max_psi = psi.psi.max()
if max_psi < 0.1:
    st.success(f"✅ Модель здорова. Максимальный PSI = {max_psi:.3f} (< 0.1)")
elif max_psi < 0.2:
    st.warning(f"⚠️ Лёгкий дрейф. Максимальный PSI = {max_psi:.3f} (0.1-0.2)")
else:
    st.error(f"🔴 Критический дрейф. Максимальный PSI = {max_psi:.3f} (> 0.2). Пора переобучать.")

st.markdown("---")

# ---- два сравнения ----
tab1, tab2 = st.tabs(["Бизнес vs Потребители (baseline drift)", "Помеченные vs Остальные потребители"])

def plot_psi(df_, col, title):
    df_ = df_.sort_values(col, ascending=True)
    df_['status'] = df_[col].apply(lambda x: '🟢 OK' if x<0.1 else ('🟡 Watch' if x<0.2 else '🔴 Alert'))
    df_['color'] = df_[col].apply(lambda x: '#2A9D2A' if x<0.1 else ('#F79E1B' if x<0.2 else '#EB001B'))
    fig = px.bar(df_, x=col, y='feature', orientation='h',
                  color='color', color_discrete_map='identity',
                  hover_data=['status'])
    fig.update_layout(height=420, margin=dict(t=20,b=10), showlegend=False,
                      xaxis_title="PSI", yaxis_title="")
    fig.add_vline(x=0.1, line_dash='dash', line_color='#F79E1B', annotation_text="0.1")
    fig.add_vline(x=0.2, line_dash='dash', line_color='#EB001B', annotation_text="0.2")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df_[['feature', col, 'status']].rename(columns={col:'PSI'}),
                 use_container_width=True, hide_index=True)

with tab1:
    st.write("Насколько распределения фич отличаются между обучающей популяцией (бизнес) и применяемой (потребители).")
    plot_psi(psi.copy(), 'psi', 'biz vs consumers')
with tab2:
    st.write("Насколько распределения фич у помеченных карт отличаются от обычных потребителей. Большой PSI = модель находит реально отличающийся сегмент.")
    plot_psi(psi.rename(columns={'psi_top_vs_rest':'psi_top_vs_rest'}).copy(), 'psi_top_vs_rest', 'top vs rest')

st.markdown("---")
with st.expander("Что такое PSI"):
    st.markdown("""
    **Population Stability Index** — мера различия двух распределений по той же переменной.

    PSI = Σ (Pₐ − Pᵦ) × ln(Pₐ / Pᵦ), где Pₐ и Pᵦ — доли в одинаковых бинах.

    Правила:
    - **PSI < 0.1** — распределения совпадают, модель в порядке.
    - **0.1 ≤ PSI < 0.2** — заметный сдвиг.
    - **PSI ≥ 0.2** — сильный дрейф, модель нужно переобучить.

    В production — еженедельный прогон, алерты в Slack/email.
    """)
