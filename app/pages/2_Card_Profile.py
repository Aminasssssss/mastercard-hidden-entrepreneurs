# -*- coding: utf-8 -*-
"""Page 2: Профиль одной карты — скор, timeline, локальный SHAP, оффер."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Card Profile", layout="wide")
st.title("Card Profile")

ROOT = Path(__file__).resolve().parents[2]

@st.cache_data
def load_all():
    cards = pd.read_parquet(ROOT/"data"/"scored_cards.parquet")
    tx = pd.read_parquet(ROOT/"data"/"tx_top100.parquet")
    shap_local = pd.read_parquet(ROOT/"data"/"shap_local_top.parquet")
    return cards, tx, shap_local

cards, tx, shap_local = load_all()

# выбор карты
card_options = cards.nlargest(200, 'final_score')['card_number'].astype(str).tolist()
card_str = st.selectbox("Выберите карту (топ-200 по скору)", card_options, index=0)
card = card_str  # card_number в parquet хранится как строка
row = cards[cards.card_number==card].iloc[0]

# ---- gauge + KPI ----
left, right = st.columns([1, 2])
with left:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=row.final_score*100,
        title={'text':"Бизнес-подобность (%)"},
        number={'suffix':"%"},
        gauge={
            'axis':{'range':[0,100]},
            'bar':{'color':"#EB001B"},
            'steps':[
                {'range':[0,33], 'color':"#D5F5D5"},
                {'range':[33,66], 'color':"#FFE5B4"},
                {'range':[66,100], 'color':"#FFCCCC"},
            ],
            'threshold':{'line':{'color':"black",'width':3}, 'value': row.final_score*100}
        }
    ))
    fig.update_layout(height=300, margin=dict(t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.markdown(f"### Карта `{card_str[:4]}{'*'*8}{card_str[-4:]}`")
    c1,c2,c3 = st.columns(3)
    c1.metric("Транзакций", f"{int(row.txn_count)}")
    c2.metric("Уникальных мерчантов", f"{int(row.unique_merchants)}")
    c3.metric("Средний чек", f"{row.avg_amount/1000:.0f}К ₸")
    c1.metric("Доля B2B-MCC", f"{row.b2b_ratio:.1%}")
    c2.metric("Online", f"{row.online_ratio:.1%}")
    c3.metric("Recurring", f"{row.recurring_ratio:.1%}")

    # рекомендация
    st.markdown("---")
    st.markdown("##### 🎯 Рекомендованное действие")
    st.success(f"**Сегмент:** {row.segment_name}\n\n**Оффер:** {row.recommended_offer}")

st.markdown("---")

# ---- Таймлайн ----
tx_card = tx[tx.card_number==card].sort_values('transaction_timestamp')
if len(tx_card)>0:
    st.subheader("Timeline транзакций")
    fig_tl = px.scatter(tx_card, x='transaction_timestamp', y='transaction_amount_kzt',
                         color='channel',
                         color_discrete_map={'online':'#EB001B','POS':'#1A1F71'},
                         hover_data=['mcc'])
    fig_tl.update_layout(height=320, margin=dict(t=10), yaxis_title="Сумма (₸)", xaxis_title="")
    st.plotly_chart(fig_tl, use_container_width=True)

    # топ MCC
    cl, cr = st.columns(2)
    with cl:
        st.subheader("Топ-5 MCC")
        mcc_top = tx_card.groupby('mcc').size().sort_values(ascending=False).head(5).reset_index(name='count')
        fig_mcc = px.bar(mcc_top, x='count', y='mcc', orientation='h',
                          color_discrete_sequence=['#EB001B'])
        fig_mcc.update_layout(height=260, margin=dict(t=10))
        st.plotly_chart(fig_mcc, use_container_width=True)
    with cr:
        st.subheader("Распределение по каналам")
        ch = tx_card.channel.value_counts().reset_index()
        ch.columns = ['channel','count']
        fig_ch = px.pie(ch, names='channel', values='count',
                         color_discrete_map={'online':'#EB001B','POS':'#1A1F71'})
        fig_ch.update_layout(height=260, margin=dict(t=10))
        st.plotly_chart(fig_ch, use_container_width=True)

# ---- Локальный SHAP ----
st.markdown("---")
st.subheader("Почему модель так решила (локальный SHAP)")
sh_row = shap_local[shap_local.card_number==card]
if len(sh_row)>0:
    shap_cols = [c for c in shap_local.columns if c.startswith('shap__')]
    vals = sh_row[shap_cols].iloc[0]
    contrib = pd.DataFrame({
        'feature':[c.replace('shap__','') for c in shap_cols],
        'shap':vals.values
    }).reindex(np.argsort(-np.abs(vals.values)))[:10].iloc[::-1]
    contrib['color'] = contrib.shap.apply(lambda x: '#EB001B' if x>0 else '#1A1F71')
    fig_shap = go.Figure(go.Bar(
        x=contrib.shap, y=contrib.feature, orientation='h',
        marker_color=contrib.color, text=contrib.shap.round(3), textposition='outside'
    ))
    fig_shap.update_layout(height=400, margin=dict(t=10),
                            xaxis_title="SHAP вклад (красный = больше бизнес-подобности)",
                            yaxis_title="")
    st.plotly_chart(fig_shap, use_container_width=True)
else:
    st.info("Локальный SHAP доступен только для топ-2000 карт.")
