# -*- coding: utf-8 -*-
"""Page 4: ROI калькулятор с интерактивными ползунками."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="ROI Calculator", layout="wide")
st.title("ROI Calculator")
st.caption("Расчёт ожидаемой прибыли банка. Допущения иллюстративные.")

ROOT = Path(__file__).resolve().parents[2]
df = pd.read_parquet(ROOT/"data"/"scored_cards.parquet")
N_CONSUMER_TOTAL = 80000
N_FLAGGED = len(df)

# ---- ползунки ----
c1, c2, c3 = st.columns(3)
with c1:
    pct_to_contact = st.slider("% карт для outreach (от помеченных)", 10, 100, 50, 5)
with c2:
    conversion_rate = st.slider("Конверсия в бизнес-тариф (%)", 5, 60, 20, 1)
with c3:
    arpu_uplift = st.slider("Uplift по тарифу (₸/мес на карту)", 2000, 20000, 8000, 500)

c4, c5, c6 = st.columns(3)
with c4:
    horizon_months = st.slider("Горизонт (мес.)", 6, 36, 12, 1)
with c5:
    contact_cost = st.slider("Стоимость одного outreach (₸)", 100, 5000, 500, 100)
with c6:
    bank_total_cards = st.number_input("Всего потребительских карт в банке", min_value=10_000, max_value=10_000_000, value=700_000, step=10_000)

# ---- расчёты ----
n_contacted = int(N_FLAGGED * pct_to_contact / 100)
n_converted = int(n_contacted * conversion_rate / 100)
gross_revenue = n_converted * arpu_uplift * horizon_months
outreach_cost = n_contacted * contact_cost
net_profit = gross_revenue - outreach_cost
roi_pct = (net_profit / max(outreach_cost,1)) * 100

# экстраполяция
scale = bank_total_cards / N_CONSUMER_TOTAL
ext_converted = int(n_converted * scale)
ext_net = int(net_profit * scale)

st.markdown("---")

# ---- KPI ----
k1, k2, k3, k4 = st.columns(4)
k1.metric("Контактов", f"{n_contacted:,}")
k2.metric("Конверсий", f"{n_converted:,}")
k3.metric("Чистая прибыль", f"{net_profit/1e6:.1f} млн ₸",
          delta=f"ROI {roi_pct:.0f}%")
k4.metric(f"Экстраполяция на {bank_total_cards:,}", f"{ext_net/1e6:.0f} млн ₸",
          help=f"При тех же допущениях на {bank_total_cards:,} картах")

# ---- график окупаемости по месяцам ----
months = np.arange(1, horizon_months+1)
monthly_revenue = n_converted * arpu_uplift
cumulative_rev = monthly_revenue * months
cumulative_cost = np.full_like(months, outreach_cost, dtype=float)
cumulative_net = cumulative_rev - cumulative_cost

fig = go.Figure()
fig.add_trace(go.Scatter(x=months, y=cumulative_rev/1e6, name="Выручка", line=dict(color='#2A9D2A', width=3)))
fig.add_trace(go.Scatter(x=months, y=cumulative_cost/1e6, name="Затраты на outreach", line=dict(color='#888', width=2, dash='dot')))
fig.add_trace(go.Scatter(x=months, y=cumulative_net/1e6, name="Чистая прибыль", line=dict(color='#EB001B', width=3), fill='tozeroy', fillcolor='rgba(235,0,27,0.1)'))
fig.update_layout(height=400, xaxis_title="Месяц", yaxis_title="Млн ₸",
                  margin=dict(t=20,b=20), legend=dict(orientation="h", y=1.1))
st.plotly_chart(fig, use_container_width=True)

# ---- breakeven ----
breakeven_month = outreach_cost / max(monthly_revenue, 1)
if breakeven_month <= horizon_months:
    st.success(f"💡 Точка окупаемости: **{breakeven_month:.1f} мес.** от начала пилота")
else:
    st.warning(f"⚠️ В выбранном горизонте не окупается (нужно {breakeven_month:.1f} мес.)")

with st.expander("Допущения"):
    st.markdown("""
    - Цифры **синтетические**; банк подставит реальные ставки тарифа.
    - Реальная конверсия может быть ниже без A/B-теста.
    - Стоимость outreach варьируется по каналу (звонок vs email vs менеджер).
    - Экстраполяция предполагает аналогичное распределение скрытых бизнесов.
    """)
