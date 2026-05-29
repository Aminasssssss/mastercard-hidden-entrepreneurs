# -*- coding: utf-8 -*-
"""FastAPI service — backend для модели скоринга скрытых предпринимателей.
Запуск: uvicorn api.main:app --reload --port 8000
Swagger: http://localhost:8000/docs
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field

# ------ Pydantic schemas ------
class PredictRequest(BaseModel):
    card_number: str = Field(..., description="Card number as string (16 digits)")

class FeatureContribution(BaseModel):
    feature: str
    shap_value: float

class PredictResponse(BaseModel):
    card_number: str
    score: float = Field(..., ge=0, le=1, description="Business-likeness score 0..1")
    confidence: str = Field(..., description="high / medium / low")
    found: bool

class SegmentResponse(BaseModel):
    card_number: str
    segment_id: int
    segment_name: str
    recommended_offer: str

class ExplainResponse(BaseModel):
    card_number: str
    score: float
    top_features: List[FeatureContribution]

class HealthResponse(BaseModel):
    status: str
    model_version: str
    last_updated: str
    n_scored_cards: int

# ------ data loading ------
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

try:
    CARDS = pd.read_parquet(DATA / "scored_cards.parquet")
    SHAP_LOCAL = pd.read_parquet(DATA / "shap_local_top.parquet")
    SHAP_COLS = [c for c in SHAP_LOCAL.columns if c.startswith("shap__")]
except FileNotFoundError as e:
    raise RuntimeError(
        f"Data not found at {DATA}. Run `python prepare_data.py` first."
    ) from e

# ------ FastAPI app ------
app = FastAPI(
    title="MDQ Hidden Entrepreneur API",
    description="Скоринг потребительских карт по бизнес-подобности",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------ endpoints ------
@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health():
    """Health check + метаданные модели."""
    return HealthResponse(
        status="ok",
        model_version="1.0.0",
        last_updated=datetime.now().isoformat(),
        n_scored_cards=len(CARDS),
    )

@app.post("/predict", response_model=PredictResponse, tags=["scoring"])
def predict(req: PredictRequest):
    """Получить скор бизнес-подобности для карты."""
    row = CARDS[CARDS.card_number == req.card_number]
    if len(row) == 0:
        return PredictResponse(
            card_number=req.card_number,
            score=0.0,
            confidence="low",
            found=False,
        )
    score = float(row.final_score.iloc[0])
    confidence = "high" if score > 0.66 else ("medium" if score > 0.33 else "low")
    return PredictResponse(
        card_number=req.card_number,
        score=score,
        confidence=confidence,
        found=True,
    )

@app.post("/segment", response_model=SegmentResponse, tags=["scoring"])
def segment(req: PredictRequest):
    """Какой сегмент и какой оффер для карты."""
    row = CARDS[CARDS.card_number == req.card_number]
    if len(row) == 0:
        raise HTTPException(status_code=404, detail="Card not in scored top-2000")
    r = row.iloc[0]
    return SegmentResponse(
        card_number=req.card_number,
        segment_id=int(r.segment),
        segment_name=str(r.segment_name),
        recommended_offer=str(r.recommended_offer),
    )

@app.post("/explain", response_model=ExplainResponse, tags=["scoring"])
def explain(req: PredictRequest, top_k: int = 10):
    """Локальный SHAP: какие фичи и насколько вкладываются в скор карты."""
    row = CARDS[CARDS.card_number == req.card_number]
    sh = SHAP_LOCAL[SHAP_LOCAL.card_number == req.card_number]
    if len(row) == 0 or len(sh) == 0:
        raise HTTPException(status_code=404, detail="Card not in scored top-2000")
    vals = sh[SHAP_COLS].iloc[0].values
    order = np.argsort(-np.abs(vals))[:top_k]
    top_feats = [
        FeatureContribution(
            feature=SHAP_COLS[i].replace("shap__", ""),
            shap_value=float(vals[i]),
        )
        for i in order
    ]
    return ExplainResponse(
        card_number=req.card_number,
        score=float(row.final_score.iloc[0]),
        top_features=top_feats,
    )

@app.get("/top", tags=["scoring"])
def top(n: int = 10):
    """Топ-N карт по скору. Удобно для UI и проверки."""
    cols = ["card_number", "final_score", "segment_name", "recommended_offer"]
    top_df = CARDS.nlargest(n, "final_score")[cols]
    top_df["card_number"] = top_df["card_number"].astype(str)
    return top_df.to_dict(orient="records")

@app.get("/", tags=["meta"])
def root():
    return {
        "service": "MDQ Hidden Entrepreneur API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": ["/health", "/predict", "/segment", "/explain", "/top"],
    }
