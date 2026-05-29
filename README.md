# MDQ Hidden Entrepreneur Detection

Прототип системы поиска скрытых предпринимателей среди держателей потребительских карт.

## Что внутри

Три сервиса, работающие совместно:

- **Streamlit** (порт 8501) — банковский UI: 8 страниц
- **FastAPI** (порт 8000) — REST API для интеграции с банковскими системами
- **MLflow** (порт 5555) — трекинг 11 экспериментов

Архитектура: `Streamlit (UI) → FastAPI (API) → ML model → MLflow (tracking)`

## Запуск

```bash
bash run.sh install   # один раз
bash run.sh all       # каждый запуск
```

После запуска доступны три URL:
- http://localhost:8501 — Streamlit
- http://localhost:8000/docs — Swagger API
- http://localhost:5555 — MLflow UI

## Streamlit — 8 страниц

1. **Dashboard** — KPI, таблица топ карт, фильтры, экспорт CSV/Excel
2. **Card Profile** — gauge, timeline транзакций, локальный SHAP, оффер
3. **Segments** — 4 кластера и банковские продукты
4. **ROI Calculator** — расчёт прибыли от outreach
5. **New Card Simulator** — ввод фич → предсказанный скор
6. **Model Health** — PSI мониторинг дрейфа
7. **A/B Test Calculator** — статистическая мощность пилота
8. **About** — методология, Model Card, глобальный SHAP

## FastAPI — 5 endpoints

- `POST /predict` — скор карты по `card_number`
- `POST /segment` — сегмент и оффер
- `POST /explain` — локальный SHAP (топ-10)
- `GET /top?n=10` — топ-N карт
- `GET /health` — метаданные модели

Swagger на `/docs`.

## MLflow — 11 экспериментов

- 1 baseline (naive rule)
- 6 one-class методов
- 3 бинарных классификатора (для сравнения)
- 1 финальный ансамбль

Метрики: pseudo_auc, top100_b2b_ratio, top100_lift_vs_rest.

## По отдельности

```bash
bash run.sh streamlit
bash run.sh api
bash run.sh mlflow
```

## Технические детали

- Скоры предрассчитаны для топ-2000 карт.
- Локальный SHAP на каждую из 2000 карт уже посчитан.
- MLflow tracking — file-based, в `mlruns/`.
- CORS открыт для локальной разработки.

## Воспроизводимость

`random_state=42` везде, версии зависимостей зафиксированы в `requirements.txt`. Запуск одной командой через `run.sh`.

## Ограничения

- Данные синтетические.
- Топ-2000 карт предрассчитаны заранее.
- Авторизация отключена.
- MLflow в file-mode.

## Данные

Синтетика организаторов Mastercard Data Quest 2026.
