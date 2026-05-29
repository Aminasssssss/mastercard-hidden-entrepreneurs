#!/bin/bash
# MDQ Hidden Entrepreneur Detection — запуск всего стека одной командой
# Использование:
#   bash run.sh install     — поставить зависимости
#   bash run.sh streamlit   — только Streamlit (порт 8501)
#   bash run.sh api         — только FastAPI (порт 8000)
#   bash run.sh mlflow      — только MLflow UI (порт 5000)
#   bash run.sh all         — всё сразу

set -e
cd "$(dirname "$0")"

case "${1:-help}" in
  install)
    echo "→ Installing dependencies..."
    pip install -r requirements.txt
    echo "✓ Done. Now run: bash run.sh all"
    ;;

  streamlit)
    echo "→ Starting Streamlit on http://localhost:8501"
    streamlit run app/streamlit_app.py --server.port 8501
    ;;

  api)
    echo "→ Starting FastAPI on http://localhost:8000 (docs: /docs)"
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
    ;;

  mlflow)
    echo "→ Starting MLflow UI on http://localhost:5555"
    mlflow ui --backend-store-uri "file://$(pwd)/mlruns" --host 0.0.0.0 --port 5555
    ;;

  all)
    echo "→ Starting all three services..."
    echo "  · Streamlit:  http://localhost:8501"
    echo "  · FastAPI:    http://localhost:8000/docs"
    echo "  · MLflow UI:  http://localhost:5555"
    echo ""
    echo "Press Ctrl+C to stop all."
    # запускаем в фоне, ловим Ctrl+C и убиваем всё
    trap 'echo "Stopping..."; kill 0; exit' INT
    uvicorn api.main:app --host 0.0.0.0 --port 8000 &
    mlflow ui --backend-store-uri "file://$(pwd)/mlruns" --host 0.0.0.0 --port 5555 &
    sleep 2
    streamlit run app/streamlit_app.py --server.port 8501
    ;;

  *)
    echo "MDQ Hidden Entrepreneur Detection"
    echo ""
    echo "Использование:"
    echo "  bash run.sh install     — поставить зависимости"
    echo "  bash run.sh all         — запустить всё (Streamlit + FastAPI + MLflow)"
    echo "  bash run.sh streamlit   — только Streamlit (порт 8501)"
    echo "  bash run.sh api         — только FastAPI (порт 8000, /docs для Swagger)"
    echo "  bash run.sh mlflow      — только MLflow UI (порт 5000)"
    ;;
esac
