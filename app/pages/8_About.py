# -*- coding: utf-8 -*-
"""Page 8: О методологии и ограничениях."""
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="About", layout="wide")
st.title("Методология и Model Card")

ROOT = Path(__file__).resolve().parents[2]

st.markdown("""
### Задача

Среди 80 000 потребительских карт найти ведущие коммерческую деятельность. Меток нет.
Формат one-class: обучение на 25 000 бизнес-картах, скоринг потребителей по бизнес-подобности.

### Подход

50 признаков на карту → ансамбль из 6 one-class методов → ранжирующий скор от 0 до 1.

Методы:
- Isolation Forest
- One-Class SVM (RBF kernel)
- Gaussian Mixture (5 components)
- Mahalanobis distance
- Autoencoder (MLP, sklearn)
- PU Learning (Elkan–Noto)

Ансамбль — среднее по рангам с равными весами. Tuned-веса не используются: на синтетике
pseudo-AUC вырожден (≈1.0 у всех методов), оптимизация ведёт к переобучению.

### Валидация без меток

- Поведенческая: топ-100 vs обычные потребители — десятки крат по B2B-MCC, рекламе, recurring.
- Robustness: пересечение топ-100 по 3 seed'ам ≈ 85%.
- Confusion matrix: precision против прокси `b2b_ratio > 5%`.
""")

st.markdown("---")
st.subheader("Топ-15 факторов модели (глобальный SHAP)")
imp = pd.read_parquet(ROOT/"data"/"shap_importance.parquet").head(15)
fig = px.bar(imp.iloc[::-1], x='mean_abs_shap', y='feature', orientation='h',
              color_discrete_sequence=['#EB001B'])
fig.update_layout(height=450, margin=dict(t=10), xaxis_title="Mean |SHAP|", yaxis_title="")
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.subheader("Model Card")
st.markdown("""
| Параметр | Значение |
|---|---|
| **Модель** | One-class ensemble of 6 methods (rank-averaged) |
| **Train** | 25 000 бизнес-карт × 3 млн транзакций |
| **Apply** | 80 000 потребительских карт × 9.8 млн транзакций |
| **Окно** | 6 месяцев (окт 2025 — мар 2026) |
| **Target** | нет (one-class) |
| **Метрика** | AUC-ROC по held-out меткам |
| **Прокси** | средний `b2b_ratio` в топ-100 |
| **Допущения** | синтетика, SCAR для PU Learning |
| **Ограничения** | pseudo-AUC вырожден, граф-фичи избыточны с MCC |
| **Воспроизводимость** | random_state=42, requirements.txt зафиксирован |
| **Дрейф** | PSI на 10 ключевых фич (страница Model Health) |

### Этика

- Только cross-sell, не для отказа в обслуживании.
- Ложноположительные результаты не вредят клиенту.
- Без ПДн — только агрегаты по транзакциям.

### Следующие шаги для продакшена

1. Precision@K на ручной проверке топ-50 банком.
2. A/B-тест outreach с контрольной группой.
3. Identity resolution: склейка карт одного клиента.
4. Real-time скоринг при новой транзакции (FastAPI готов).
5. Quarterly retraining с PSI-триггером.
""")

st.markdown("---")
st.caption("Synthetic data · Educational purpose · Mastercard Data Quest 2026")
