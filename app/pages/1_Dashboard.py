# -*- coding: utf-8 -*-
"""Page 1: Главный dashboard — список карт с фильтрами и экспорт."""
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from io import BytesIO

st.set_page_config(page_title="Dashboard", layout="wide")
st.title("Dashboard — Скрытые предприниматели")

DATA = Path(__file__).resolve().parents[2] / "data" / "scored_cards.parquet"

@st.cache_data
def load():
    return pd.read_parquet(DATA)

df = load()

# ---- KPI блок ----
c1, c2, c3, c4 = st.columns(4)
N_FLAGGED = len(df)
TOP_AVG_SCORE = df.final_score.mean()
TOP_B2B = df.b2b_ratio.mean()
EST_ROI = N_FLAGGED * 0.20 * 8000 * 12 / 1e6
c1.metric("Помечено карт", f"{N_FLAGGED:,}", help="Топ-2.5% по финальному скору ансамбля")
c2.metric("Средний скор", f"{TOP_AVG_SCORE:.3f}", help="Чем ближе к 1, тем больше похоже на бизнес")
c3.metric("Средняя B2B-доля", f"{TOP_B2B:.1%}", help="Доля трат в бизнес-категориях у помеченных карт")
c4.metric("Ожидаемый ROI (12 мес)", f"{EST_ROI:.1f} млн ₸", help="При 20% конверсии и +8К ₸/мес")

st.markdown("---")

# ---- Фильтры ----
left, mid, right = st.columns([2, 2, 1])
with left:
    seg_filter = st.multiselect("Сегмент", options=sorted(df.segment_name.unique()),
                                 default=sorted(df.segment_name.unique()))
with mid:
    score_range = st.slider("Диапазон скора", 0.0, 1.0, (0.0, 1.0), 0.01)
with right:
    n_show = st.number_input("Показать карт", min_value=10, max_value=2000, value=100, step=10)

filtered = df[(df.segment_name.isin(seg_filter)) &
              (df.final_score.between(*score_range))].nlargest(n_show, "final_score")

# ---- Таблица ----
st.subheader(f"Топ-{len(filtered)} карт")
display_cols = ['card_number','final_score','segment_name','recommended_offer',
                'avg_amount','b2b_ratio','online_ratio','recurring_ratio',
                'unique_merchants','txn_count']
table = filtered[display_cols].copy()
table['card_number'] = table['card_number'].astype(str).apply(lambda s: s[:4]+'*'*8+s[-4:])
table['final_score'] = (table['final_score']*100).round(1)
table['b2b_ratio'] = (table['b2b_ratio']*100).round(1)
table['online_ratio'] = (table['online_ratio']*100).round(1)
table['recurring_ratio'] = (table['recurring_ratio']*100).round(1)
table['avg_amount'] = table['avg_amount'].round(0).astype(int)
table.columns = ['Card', 'Score (%)', 'Segment', 'Recommended Offer',
                 'Avg amount (₸)', 'B2B %', 'Online %', 'Recurring %',
                 'Unique merchants', 'Txn count']
st.dataframe(table, use_container_width=True, hide_index=True)

# ---- Экспорт ----
st.markdown("---")
exp_col1, exp_col2 = st.columns(2)
with exp_col1:
    csv = filtered[display_cols].to_csv(index=False).encode('utf-8-sig')
    st.download_button("📥 Скачать CSV", csv, file_name=f"flagged_cards_{len(filtered)}.csv", mime="text/csv")
with exp_col2:
    buffer = BytesIO()
    try:
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            filtered[display_cols].to_excel(writer, index=False, sheet_name='Flagged')
        st.download_button("📥 Скачать Excel", buffer.getvalue(),
                           file_name=f"flagged_cards_{len(filtered)}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except Exception:
        st.info("Excel export недоступен (нет openpyxl). CSV работает.")

# ---- Распределение скоров ----
st.markdown("---")
st.subheader("Распределение скоров")
fig = px.histogram(df, x='final_score', nbins=50, color_discrete_sequence=['#EB001B'])
fig.update_layout(height=300, margin=dict(t=10,b=10), xaxis_title="Final score", yaxis_title="Карт")
st.plotly_chart(fig, use_container_width=True)
