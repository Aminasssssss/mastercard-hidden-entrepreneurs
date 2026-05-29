# -*- coding: utf-8 -*-
"""Page 3: 4 сегмента помеченных карт и рекомендованные офферы."""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Segments", layout="wide")
st.title("Сегменты помеченных карт")
st.caption("Топ-2000 карт разбиты K-means на 4 сегмента. Каждому — свой банковский продукт.")

ROOT = Path(__file__).resolve().parents[2]
df = pd.read_parquet(ROOT/"data"/"scored_cards.parquet")

segs = df.groupby('segment_name').agg(
    n=('card_number','count'),
    avg_score=('final_score','mean'),
    avg_amount=('avg_amount','mean'),
    b2b=('b2b_ratio','mean'),
    adv=('advertising_ratio','mean'),
    sw=('software_ratio','mean'),
    log=('logistics_ratio','mean'),
    online=('online_ratio','mean'),
    rec=('recurring_ratio','mean'),
    bizh=('biz_hour_ratio','mean'),
).reset_index()

OFFERS = {
    'Digital Marketing / E-commerce':'POS-эквайринг + рекламные инструменты + бизнес-карта',
    'Logistics / Supply':'Оборотный кредит + корп-карты + расчётный счёт',
    'IT / Professional Services':'Зарплатный проект + кассовые решения + бухгалтерия как услуга',
    'General Small Business':'Бизнес-карта + кросс-продажи + лояльность',
}
COLORS = {
    'Digital Marketing / E-commerce':'#EB001B',
    'Logistics / Supply':'#FF5F00',
    'IT / Professional Services':'#F79E1B',
    'General Small Business':'#1A1F71',
}

# ---- общий обзор ----
col_pie, col_bar = st.columns(2)
with col_pie:
    fig_pie = px.pie(segs, names='segment_name', values='n',
                      color='segment_name', color_discrete_map=COLORS)
    fig_pie.update_layout(height=350, margin=dict(t=10))
    st.plotly_chart(fig_pie, use_container_width=True)
with col_bar:
    fig_bar = px.bar(segs.sort_values('avg_score', ascending=True),
                      x='avg_score', y='segment_name', orientation='h',
                      color='segment_name', color_discrete_map=COLORS)
    fig_bar.update_layout(height=350, margin=dict(t=10), showlegend=False,
                          xaxis_title="Средний скор сегмента", yaxis_title="")
    st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")

# ---- карточки сегментов ----
for _, s in segs.iterrows():
    name = s.segment_name
    color = COLORS.get(name, '#888')
    offer = OFFERS.get(name, 'Стандартный набор продуктов')
    with st.container():
        st.markdown(f"<div style='border-left: 5px solid {color}; padding-left: 1rem;'>", unsafe_allow_html=True)
        cl, cm, cr = st.columns([1, 2, 2])
        with cl:
            st.markdown(f"### {name}")
            st.metric("Карт в сегменте", f"{int(s.n)}")
            st.metric("Средний скор", f"{s.avg_score:.3f}")
        with cm:
            st.markdown("**Профиль поведения:**")
            st.write(f"- Средний чек: **{s.avg_amount/1000:.0f}К ₸**")
            st.write(f"- B2B-MCC доля: **{s.b2b:.1%}**")
            st.write(f"- Online: **{s.online:.1%}** · Recurring: **{s.rec:.1%}**")
            st.write(f"- Активность в рабочие часы: **{s.bizh:.1%}**")
        with cr:
            st.markdown("**Рекомендованный продукт:**")
            st.success(offer)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("")
