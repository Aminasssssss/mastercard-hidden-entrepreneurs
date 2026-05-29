# -*- coding: utf-8 -*-
"""Page 5: Симулятор — ввёл фичи, получил скор. Для демо на сцене."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="New Card Simulator", layout="wide")
st.title("Симулятор новой карты")
st.caption("Ввод поведенческих признаков → предсказанный скор (упрощённая логика для демо).")

ROOT = Path(__file__).resolve().parents[2]
df = pd.read_parquet(ROOT/"data"/"scored_cards.parquet")

# квантили на основе бизнеса (упрощённый «бизнес-эталон»)
def biz_proxy(name, default):
    return float(df[name].quantile(0.9)) if name in df.columns else default

st.markdown("### Введите профиль карты")
c1, c2, c3 = st.columns(3)
with c1:
    txn = st.slider("Транзакций за 6 мес", 5, 500, 80)
    avg_amt = st.slider("Средний чек (₸)", 1000, 500_000, 80_000, 1000)
    online = st.slider("Доля онлайн (%)", 0, 100, 30) / 100
with c2:
    b2b = st.slider("Доля B2B-MCC (%)", 0, 100, 5) / 100
    recurring = st.slider("Доля recurring (%)", 0, 100, 5) / 100
    biz_hour = st.slider("Доля операций в раб. часы (%)", 0, 100, 40) / 100
with c3:
    merchants = st.slider("Уникальных мерчантов", 1, 100, 15)
    adv = st.slider("Доля реклама/MCC 7311 (%)", 0, 50, 0) / 100
    graph_aff = st.slider("Граф-сродство к бизнес-мерчантам (0-1)", 0.0, 1.0, 0.1, 0.05)

# ---- упрощённый расчёт скора ----
# веса откалиброваны вручную из global SHAP (online_ratio, tokenized, hour_mean, weekday, graph_max_merch_biz)
# нелинейная сигмоида на агрегат
sig = lambda x: 1/(1+np.exp(-x))
raw = (
    1.5 * online +
    3.0 * b2b +
    2.0 * recurring +
    1.5 * biz_hour +
    4.0 * adv +
    2.5 * graph_aff +
    0.5 * np.log1p(txn)/np.log1p(500) +
    0.5 * np.log1p(avg_amt)/np.log1p(500_000) -
    2.0
)
score = sig(raw)

# ---- вывод ----
st.markdown("---")
gcol, kcol = st.columns([1, 1])
with gcol:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score*100,
        title={'text':"Предсказанный скор (%)"},
        number={'suffix':"%"},
        gauge={
            'axis':{'range':[0,100]},
            'bar':{'color':"#EB001B"},
            'steps':[
                {'range':[0,33], 'color':"#D5F5D5"},
                {'range':[33,66], 'color':"#FFE5B4"},
                {'range':[66,100], 'color':"#FFCCCC"},
            ],
        }
    ))
    fig.update_layout(height=300, margin=dict(t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)
with kcol:
    st.markdown("### Решение модели")
    if score > 0.66:
        st.error(f"🔴 **Высокая бизнес-подобность ({score:.1%})**\n\nКарта похожа на скрытый бизнес. "
                 "Рекомендация: включить в outreach-кампанию с приоритетом 1.")
    elif score > 0.33:
        st.warning(f"🟡 **Средняя бизнес-подобность ({score:.1%})**\n\nКарта показывает некоторые "
                   "бизнес-признаки. Можно включить в более широкий пилот.")
    else:
        st.success(f"🟢 **Низкая бизнес-подобность ({score:.1%})**\n\nОбычный потребительский профиль. "
                   "Outreach как скрытому бизнесу не нужен.")

with st.expander("Как считается"):
    st.markdown("""
    Упрощённая функция для демо — линейная комбинация введённых фич + сигмоида.
    Реальная модель — ансамбль из 6 методов на 50 фичах.
    Веса подобраны на основе глобального SHAP полной модели.
    """)
