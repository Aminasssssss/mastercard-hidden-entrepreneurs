# -*- coding: utf-8 -*-
"""Page 7: A/B тест калькулятор — мощность и MDE."""
import streamlit as st
import numpy as np
from scipy import stats

st.set_page_config(page_title="A/B Test Calculator", layout="wide")
st.title("A/B Test Calculator")
st.caption("Какую разницу в конверсии тест надёжно зафиксирует на выборке X карт?")

c1, c2 = st.columns(2)
with c1:
    st.markdown("### Параметры пилота")
    n_per_group = st.slider("Карт в каждой группе (treatment / control)", 50, 5000, 500, 50)
    baseline_cr = st.slider("Базовая конверсия (control), %", 1.0, 50.0, 10.0, 0.5) / 100
    alpha = st.selectbox("Уровень значимости α", [0.01, 0.05, 0.10], index=1)
    power = st.selectbox("Желаемая мощность 1−β", [0.80, 0.85, 0.90, 0.95], index=0)

with c2:
    st.markdown("### Прямой расчёт MDE (минимально детектируемый эффект)")
    z_a = stats.norm.ppf(1 - alpha/2)
    z_b = stats.norm.ppf(power)
    p1 = baseline_cr
    var_pooled = 2 * p1 * (1 - p1)
    mde_abs = (z_a + z_b) * np.sqrt(var_pooled / n_per_group)
    mde_rel = mde_abs / p1 * 100

    st.metric("MDE (абсолютный)", f"{mde_abs*100:.2f} п.п.",
              help="Минимальная разница в конверсии, которую тест надёжно отличит от нуля")
    st.metric("MDE (относительный)", f"+{mde_rel:.1f}%",
              help="Тот же MDE, но в процентах от базовой конверсии")
    st.metric("Размер выборки", f"{2*n_per_group:,} карт (всего)")

st.markdown("---")

# ---- обратная задача: сколько карт нужно для заданного MDE ----
st.markdown("### Сколько карт нужно для целевого эффекта?")
target_mde_rel = st.slider("Целевой относительный MDE, %", 5, 100, 20, 5) / 100
target_mde_abs = baseline_cr * target_mde_rel
n_needed = int(np.ceil(((z_a + z_b)**2 * var_pooled) / (target_mde_abs**2)))

cc1, cc2, cc3 = st.columns(3)
cc1.metric("Нужно в каждой группе", f"{n_needed:,}")
cc2.metric("Нужно всего", f"{2*n_needed:,}")
cc3.metric("Бюджет outreach (₸)", f"{2*n_needed*500:,}",
           help="При стоимости одного контакта 500 ₸")

if 2*n_needed <= len(__import__('pandas').read_parquet(__import__('pathlib').Path(__file__).resolve().parents[2]/"data"/"scored_cards.parquet")):
    st.success(f"✅ Помеченных карт хватает на этот пилот")
else:
    st.warning("⚠️ Помеченных карт недостаточно. Снизь требования к MDE или расширь скоринг.")

with st.expander("Формулы"):
    st.latex(r"n = \frac{(z_{\alpha/2} + z_{\beta})^2 \cdot 2 \cdot p(1-p)}{(\Delta)^2}")
    st.markdown("""
    Где:
    - **n** — размер выборки на каждую группу
    - **z** — критические значения нормального распределения
    - **p** — базовая конверсия
    - **Δ** — детектируемая разница

    Используется двусторонний z-тест для пропорций.
    """)
