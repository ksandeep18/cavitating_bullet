"""
FastAPI Backend — Cavitator ML Prediction API
=============================================
Run: uvicorn api.main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Literal, Optional
import numpy as np
import joblib
import json
import os

# ── Load models ───────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE, "models")

scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")
le     = joblib.load(f"{MODEL_DIR}/label_encoder.pkl")

with open(f"{MODEL_DIR}/metadata.json") as f:
    meta = json.load(f)

TARGETS  = meta["targets"]
FEATURES = meta["features"]

models = {}
for t in TARGETS:
    models[t] = joblib.load(f"{MODEL_DIR}/{t}_best.pkl")

# ── Shape parameter lookup ────────────────────────────────────────────────────
SHAPE_INFO = {
    "disc": {
        "id": 0, "theta": 90.0,
        "param1_name": "—", "param1_range": [90, 90],
        "param2_name": "—", "param2_range": [0, 0],
        "description": "Flat circular face, maximum separation, classical reference shape"
    },
    "cone": {
        "id": 1,
        "param1_name": "half_angle_deg", "param1_range": [20, 80],
        "param2_name": "—",              "param2_range": [0, 0],
        "description": "Conical nose, half-angle 20–80°. Lower angle = sharper = smaller cavity"
    },
    "truncated_cone": {
        "id": 2,
        "param1_name": "half_angle_deg",    "param1_range": [30, 75],
        "param2_name": "truncation_ratio",  "param2_range": [0.1, 0.7],
        "description": "Cone with flat annular face. param2: 0=full cone, 0.7=near-disc"
    },
    "spherical": {
        "id": 3,
        "param1_name": "R_over_dc",  "param1_range": [0.6, 2.5],
        "param2_name": "—",          "param2_range": [0, 0],
        "description": "Hemispherical nose, param1 = R/dc radius ratio"
    },
    "elliptical": {
        "id": 4,
        "param1_name": "axis_ratio_ab", "param1_range": [0.5, 3.0],
        "param2_name": "—",             "param2_range": [0, 0],
        "description": "Ellipsoidal nose, param1 = a/b. Higher = more streamlined"
    },
    "ogive_polynomial": {
        "id": 5,
        "param1_name": "exponent_n", "param1_range": [1.0, 4.0],
        "param2_name": "—",          "param2_range": [0, 0],
        "description": "Polynomial ogive nose, param1 = power law exponent"
    },
}

# ── Physics helpers ───────────────────────────────────────────────────────────
def theta_effective(shape: str, p1: float, p2: float) -> float:
    s = {"disc":0,"cone":1,"truncated_cone":2,"spherical":3,
         "elliptical":4,"ogive_polynomial":5}[shape]
    if s == 0: return 90.0
    if s == 1: return float(np.clip(p1, 20, 80))
    if s == 2: return float(np.clip(p1 * (1.0 - p2*0.25), 15, 85))
    if s == 3: return float(np.degrees(np.arctan(1.0/(2.0*max(p1,0.1))))*1.8)
    if s == 4: return float(np.degrees(np.arctan(1.0/max(p1,0.1)))*1.6)
    if s == 5: return float(20.0 + 60.0/max(p1,1.0))
    return 45.0

def aspect_ratio_cav(shape: str, p1: float) -> float:
    if shape == "disc":              return 0.02
    if shape == "cone":              return 1.0 / (2.0 * np.tan(np.radians(max(p1,1))))
    if shape == "truncated_cone":    return max(p1*0.015, 0.05)
    if shape == "spherical":         return p1 * 0.5
    if shape == "elliptical":        return p1
    if shape == "ogive_polynomial":  return p1 * 0.3
    return 0.5

# ── Pydantic models ───────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    shape_name: Literal["disc","cone","truncated_cone",
                        "spherical","elliptical","ogive_polynomial"]
    param1: float = Field(90.0, description="Primary shape parameter")
    param2: float = Field(0.0,  description="Secondary shape parameter")
    sigma_c: float = Field(0.15, ge=0.05, le=0.40,
                           description="Cavitation number [0.05–0.40]")
    Re: float = Field(1e6, ge=1e5, le=5e6,
                      description="Reynolds number [1e5–5e6]")
    depth_m: float = Field(10.0, ge=0.0, le=50.0,
                           description="Operating depth [m]")

    @validator("param1")
    def p1_range(cls, v, values):
        return round(float(v), 4)

    @validator("param2")
    def p2_range(cls, v):
        return round(float(v), 4)

class PredictResponse(BaseModel):
    shape_name:       str
    theta_eff:        float
    beta:             float
    sigma_c:          float
    Cdp:              float
    Csf:              float
    Cd:               float
    L_tilde:          float
    D_tilde:          float
    cavity_covers:    bool
    drag_red_pct:     float
    L_cav_mm:         float
    D_cav_mm:         float
    physics_note:     str

class OptimizeRequest(BaseModel):
    sigma_c:     float = Field(0.10, ge=0.05, le=0.40)
    Re:          float = Field(1e6, ge=1e5, le=5e6)
    depth_m:     float = Field(10.0, ge=0, le=50)
    n_trials:    int   = Field(200, ge=50, le=1000)

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Cavitator ML API",
    description=(
        "ML-based drag prediction for supercavitating torpedo cavitators.\n\n"
        "**Fixed body:** 30mm cylinder + 15mm hemisphere, 115mm total, dc=12mm\n\n"
        "**Variable:** Cavitator shape (disc, cone, truncated cone, spherical, elliptical, ogive)"
    ),
    version="1.0.0"
)

app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Cavitator ML API", "version": "1.0.0",
            "docs": "/docs", "shapes": list(SHAPE_INFO.keys())}

@app.get("/shapes")
def get_shapes():
    return {"shapes": SHAPE_INFO,
            "body_fixed": meta["body_fixed"]}

@app.get("/model-info")
def model_info():
    return {"metrics": meta["metrics"],
            "best_models": meta["best_models"],
            "feature_importance": meta["feat_importance_shap"],
            "features": FEATURES,
            "n_training_points": 800}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # Compute derived features
    shape_enc = le.transform([req.shape_name])[0]
    theta     = theta_effective(req.shape_name, req.param1, req.param2)
    beta      = float(np.sin(np.radians(theta)))
    ar_cav    = aspect_ratio_cav(req.shape_name, req.param1)
    U_ms      = req.Re * 1.002e-3 / (1000.0 * 0.115)
    Froude    = U_ms / np.sqrt(9.81 * 0.115)

    x = np.array([[shape_enc, theta, beta, req.param1, req.param2,
                   ar_cav, req.sigma_c, req.Re, req.depth_m, Froude]])
    xs = scaler.transform(x)

    preds = {t: float(models[t].predict(xs)[0]) for t in TARGETS}

    # Derived outputs
    dc_m   = 0.012
    L_cav  = preds["L_tilde"] * dc_m * 1000   # mm
    D_cav  = preds["D_tilde"] * dc_m * 1000   # mm
    covers = preds["L_tilde"] * dc_m >= 0.115  # full body enveloped?

    # Baseline Csf (no cavity)
    Re_cyl = req.Re * 0.085 / 0.115
    Cf_base = 0.074 * max(Re_cyl, 1e4) ** (-0.2)
    A_cyl_nd = (np.pi * 0.030 * 0.085) / (np.pi * 0.012**2 / 4)
    Csf_base = Cf_base * A_cyl_nd
    drag_red = float(np.clip((Csf_base - preds["Csf"]) / Csf_base * 100, 0, 100))

    # Physics note
    if covers:
        note = "Full supercavity — body entirely enveloped. Skin friction nearly zero."
    elif preds["L_tilde"] * dc_m >= 0.085:
        note = "Partial cavity — cylinder covered, hemisphere exposed."
    else:
        note = "Small cavity — significant wetted area remains. Consider lower σ_c."

    return PredictResponse(
        shape_name=req.shape_name,
        theta_eff=round(theta, 2),
        beta=round(beta, 4),
        sigma_c=req.sigma_c,
        Cdp=round(preds["Cdp"], 5),
        Csf=round(preds["Csf"], 5),
        Cd=round(preds["Cd"], 5),
        L_tilde=round(preds["L_tilde"], 3),
        D_tilde=round(preds["D_tilde"], 3),
        cavity_covers=bool(covers),
        drag_red_pct=round(drag_red, 2),
        L_cav_mm=round(L_cav, 2),
        D_cav_mm=round(D_cav, 2),
        physics_note=note
    )

@app.post("/optimize")
def optimize(req: OptimizeRequest):
    """
    Find the cavitator shape that minimises Cd at given operating conditions.
    Searches over all shape families and parameter combinations.
    """
    best_Cd  = 1e9
    best_cfg = {}

    shape_params = {
        "disc":             [(90.0, 0.0)],
        "cone":             [(a, 0.0) for a in np.linspace(20, 80, 25)],
        "truncated_cone":   [(a, t) for a in np.linspace(30,75,10) for t in np.linspace(0.1,0.7,5)],
        "spherical":        [(r, 0.0) for r in np.linspace(0.6, 2.5, 20)],
        "elliptical":       [(ab, 0.0) for ab in np.linspace(0.5, 3.0, 20)],
        "ogive_polynomial": [(n, 0.0) for n in np.linspace(1.0, 4.0, 15)],
    }

    U_ms   = req.Re * 1.002e-3 / (1000.0 * 0.115)
    Froude = U_ms / np.sqrt(9.81 * 0.115)

    all_results = []
    for shape, combos in shape_params.items():
        shape_enc = le.transform([shape])[0]
        for p1, p2 in combos:
            theta  = theta_effective(shape, p1, p2)
            beta   = float(np.sin(np.radians(theta)))
            ar_cav = aspect_ratio_cav(shape, p1)
            x  = np.array([[shape_enc, theta, beta, p1, p2,
                            ar_cav, req.sigma_c, req.Re, req.depth_m, Froude]])
            xs = scaler.transform(x)
            Cd = float(models["Cd"].predict(xs)[0])
            Lt = float(models["L_tilde"].predict(xs)[0])
            all_results.append({"shape": shape, "param1": p1, "param2": p2,
                                 "theta": theta, "Cd": Cd, "L_tilde": Lt})
            if Cd < best_Cd:
                best_Cd = Cd; best_cfg = all_results[-1]

    top5 = sorted(all_results, key=lambda x: x["Cd"])[:5]
    return {
        "optimal": best_cfg,
        "top5_candidates": top5,
        "operating_conditions": {"sigma_c": req.sigma_c,
                                 "Re": req.Re, "depth_m": req.depth_m}
    }

@app.get("/batch-compare")
def batch_compare(sigma_c: float = 0.10, Re: float = 1e6, depth_m: float = 10.0):
    """Compare all shape families at the same operating condition."""
    U_ms   = Re * 1.002e-3 / (1000.0 * 0.115)
    Froude = U_ms / np.sqrt(9.81 * 0.115)

    test_points = {
        "disc":             (90.0, 0.0),
        "cone_45deg":       (45.0, 0.0),
        "cone_60deg":       (60.0, 0.0),
        "truncated_cone":   (50.0, 0.4),
        "spherical":        (1.0, 0.0),
        "elliptical":       (2.0, 0.0),
        "ogive_polynomial": (2.0, 0.0),
    }
    shape_map = {
        "disc":"disc","cone_45deg":"cone","cone_60deg":"cone",
        "truncated_cone":"truncated_cone","spherical":"spherical",
        "elliptical":"elliptical","ogive_polynomial":"ogive_polynomial"
    }

    results = []
    for label, (p1, p2) in test_points.items():
        sn    = shape_map[label]
        enc   = le.transform([sn])[0]
        theta = theta_effective(sn, p1, p2)
        beta  = float(np.sin(np.radians(theta)))
        ar    = aspect_ratio_cav(sn, p1)
        x     = np.array([[enc, theta, beta, p1, p2, ar, sigma_c, Re, depth_m, Froude]])
        xs    = scaler.transform(x)
        preds = {t: round(float(models[t].predict(xs)[0]), 5) for t in TARGETS}
        preds.update({"label": label, "shape": sn, "theta_eff": round(theta,1)})
        results.append(preds)

    return {"comparison": sorted(results, key=lambda x: x["Cd"]),
            "conditions": {"sigma_c": sigma_c, "Re": Re, "depth_m": depth_m}}
