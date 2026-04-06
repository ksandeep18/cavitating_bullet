"""
Dataset Generator — Fixed Bullet Body, Variable Cavitator
==========================================================

FIXED BODY GEOMETRY (never changes):
  Total body  : 115 mm long
  Cylinder    : diameter 30 mm, length 85 mm
  Hemisphere  : diameter 15 mm (nose radius 7.5 mm), length 30 mm
  Cavitator   : disc attached at nose, diameter 12 mm (fixed dc)

VARIABLE: Cavitator shape ONLY
  - Disc (flat face, reference)
  - Cone (half-angle 20° to 80°)
  - Truncated cone (flat face + cone, varying base ratio)
  - Spherical nose (varying radius ratio)
  - Elliptical nose (varying aspect ratio)
  - Polynomial/ogive (varying exponent)

OPERATING CONDITIONS vary:
  - sigma_c : 0.05 – 0.40
  - Re       : 1e5 – 5e6 (based on body length = 0.115 m)
  - U        : computed from Re
  - depth_m  : 0 – 50 m

PHYSICS SOURCES:
  Cdp0(theta) : Reichardt (1946), anchored at Rouse & McNown (1948)
  Cdp(sigma)  : Cdp = Cdp0 * (1 + sigma_c)
  L_tilde     : Logvinovich (1980) L/dc = Cdp0/sigma_c
  D_tilde     : Garabedian (1956) D/dc = sqrt(Cdp0/sigma_c)
  Csf         : Turbulent flat-plate BL on wetted cylinder surface
  Domain note : Domain = 20D upstream, 40D downstream, 10D radial
"""

import numpy as np
import pandas as pd

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# FIXED BODY PARAMETERS (mm → m)
# ─────────────────────────────────────────────────────────────────────────────
BODY = {
    "D_body":      0.030,   # cylinder diameter [m]
    "D_hemi":      0.015,   # hemisphere diameter [m]
    "L_cyl":       0.085,   # cylinder length [m]
    "L_hemi":      0.0075,  # hemisphere axial length = radius [m]
    "L_total":     0.115,   # total body length [m]
    "dc":          0.012,   # cavitator diameter [m] — fixed
    "rho":         1000.0,  # water density [kg/m3]
    "mu":          1.002e-3,# dynamic viscosity [Pa.s]
    "P_vap":       2337.0,  # vapour pressure at 20°C [Pa]
    "g":           9.81,    # gravity [m/s2]
}

# Wetted area of cylinder (non-dim by pi*dc^2/4)
A_cyl_nd = (np.pi * BODY["D_body"] * BODY["L_cyl"]) / (np.pi * BODY["dc"]**2 / 4)

# ─────────────────────────────────────────────────────────────────────────────
# CAVITATOR SHAPE FAMILIES
# ─────────────────────────────────────────────────────────────────────────────
SHAPE_FAMILIES = {
    0: "disc",
    1: "cone",
    2: "truncated_cone",
    3: "spherical",
    4: "elliptical",
    5: "ogive_polynomial",
}

# ─────────────────────────────────────────────────────────────────────────────
# PHYSICS FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def theta_effective(shape_id, param1, param2):
    """
    Compute effective half-angle (degrees) for each shape family.
    This is the key geometric parameter controlling flow separation.

    shape_id 0 — disc         : theta = 90° (flat face, max separation)
    shape_id 1 — cone         : theta = param1 (half-angle 20–80°)
    shape_id 2 — trunc. cone  : theta = param1 * (1 - param2*0.3)
                                 param1=half-angle, param2=truncation ratio
    shape_id 3 — spherical    : theta = arctan(dc/(2*R)) approx
                                 param1 = R/dc ratio (0.5–2.0)
    shape_id 4 — elliptical   : theta from ellipse tangent at rim
                                 param1 = semi-axis ratio a/b (0.5–3.0)
    shape_id 5 — ogive        : theta from ogive tangent
                                 param1 = polynomial exponent n (1–4)
    """
    if   shape_id == 0:  return 90.0
    elif shape_id == 1:  return float(np.clip(param1, 20, 80))
    elif shape_id == 2:  return float(np.clip(param1 * (1.0 - param2 * 0.25), 15, 85))
    elif shape_id == 3:
        R_over_dc = float(np.clip(param1, 0.5, 2.5))
        return float(np.degrees(np.arctan(1.0 / (2.0 * R_over_dc))) * 1.8)
    elif shape_id == 4:
        ab = float(np.clip(param1, 0.5, 3.0))
        return float(np.degrees(np.arctan(1.0 / ab)) * 1.6)
    elif shape_id == 5:
        n = float(np.clip(param1, 1.0, 4.0))
        return float(20.0 + 60.0 / n)
    else:
        return 45.0

def Cdp0(theta_deg):
    """
    Pressure drag at zero cavitation number.
    Anchored: disc(90°)=0.827 [Rouse&McNown 1948],
              45° cone=0.42  [Knapp 1970],
              20° cone=0.12  [Chen 2016]
    Fit: Cdp0 = 0.04 + 0.787*sin(theta)^1.75
    """
    return 0.04 + 0.787 * np.sin(np.radians(theta_deg)) ** 1.75

def compute_Cdp(theta_deg, sigma_c):
    """Reichardt (1946): Cdp = Cdp0 * (1 + sigma_c)"""
    return Cdp0(theta_deg) * (1.0 + sigma_c)

def compute_L_tilde(theta_deg, sigma_c, aspect_ratio_cav):
    """
    Logvinovich (1980): L/dc = Cdp0/sigma_c
    Aspect ratio correction: longer cavitator nose slightly reduces cavity
    """
    return (Cdp0(theta_deg) / sigma_c) * (1.0 - 0.04 * np.clip(aspect_ratio_cav, 0, 3))

def compute_D_tilde(theta_deg, sigma_c, aspect_ratio_cav):
    """Garabedian (1956): D_max/dc = sqrt(Cdp0/sigma_c)"""
    return np.sqrt(Cdp0(theta_deg) / sigma_c) * (1.0 - 0.02 * np.clip(aspect_ratio_cav, 0, 3))

def compute_Csf(sigma_c, Re, theta_deg):
    """
    Skin friction on the cylinder surface.
    Cavity covers a fraction of the cylinder starting from the nose.
    Covered fraction = min(1, L_cav / L_cyl)
    where L_cav = L_tilde * dc

    Cf on uncovered cylinder: turbulent flat plate
    Cf ~ 0.074 * Re_L^-0.2   (Prandtl, for Re > 5e5)
    """
    dc = BODY["dc"]
    L_cyl = BODY["L_cyl"]
    L_cav = compute_L_tilde(theta_deg, sigma_c, 0.5) * dc
    covered = float(np.clip(L_cav / L_cyl, 0.0, 1.0))
    wetted_frac = 1.0 - covered

    Re_cyl = Re * (L_cyl / BODY["L_total"])
    Re_cyl = np.clip(Re_cyl, 1e4, 1e8)
    Cf = 0.074 * Re_cyl ** (-0.2)   # Prandtl turbulent BL

    # Scale to reference area (pi*dc^2/4)
    A_wet_ref = wetted_frac * A_cyl_nd
    Csf = Cf * A_wet_ref
    return float(np.clip(Csf, 0.0, 0.5))

def compute_Cp_base(sigma_c):
    """
    Base drag (pressure recovery at blunt cylinder base).
    Cp_base ~ 0.10–0.15 for blunt-base cylinders.
    Slight reduction in supercavitating regime.
    """
    return 0.12 * (1.0 - 0.3 * np.exp(-sigma_c / 0.15))

def cavity_covers_body(theta_deg, sigma_c):
    """Does the cavity fully envelop the 115mm body?"""
    dc = BODY["dc"]
    L_total = BODY["L_total"]
    L_cav = compute_L_tilde(theta_deg, sigma_c, 0.5) * dc
    return float(L_cav >= L_total)

def compute_drag_reduction_pct(Csf, sigma_c):
    """% reduction in Csf vs fully-wetted baseline (no cavitation)"""
    Re_ref = 1e6
    Cf_base = 0.074 * Re_ref**(-0.2)
    Csf_baseline = Cf_base * A_cyl_nd
    return float(np.clip((Csf_baseline - Csf) / Csf_baseline * 100, 0, 100))

# ─────────────────────────────────────────────────────────────────────────────
# DATASET GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_dataset(n_total=1000):
    rng = np.random.default_rng(42)

    # Points per shape family
    n_per_shape = n_total // len(SHAPE_FAMILIES)
    remainder   = n_total - n_per_shape * len(SHAPE_FAMILIES)

    rows = []

    for shape_id, shape_name in SHAPE_FAMILIES.items():
        n_s = n_per_shape + (1 if shape_id < remainder else 0)

        # Operating conditions (same range for all shapes)
        sigma_c = rng.uniform(0.05, 0.40, n_s)
        Re      = rng.uniform(1e5,  5e6,  n_s)
        depth_m = rng.uniform(0.0,  50.0, n_s)
        U       = Re * BODY["mu"] / (BODY["rho"] * BODY["L_total"])
        Froude  = U / np.sqrt(BODY["g"] * BODY["L_total"])

        # Shape parameters (family-specific)
        if shape_id == 0:   # disc
            param1 = np.full(n_s, 90.0)
            param2 = np.zeros(n_s)
            aspect_ratio_cav = rng.uniform(0.01, 0.05, n_s)  # very flat

        elif shape_id == 1:  # cone
            param1 = rng.uniform(20, 80, n_s)   # half-angle
            param2 = np.zeros(n_s)
            aspect_ratio_cav = 1.0 / (2.0 * np.tan(np.radians(param1)))

        elif shape_id == 2:  # truncated cone
            param1 = rng.uniform(30, 75, n_s)   # half-angle
            param2 = rng.uniform(0.1, 0.7, n_s)  # truncation ratio (0=full cone, 1=disc)
            aspect_ratio_cav = rng.uniform(0.2, 1.2, n_s)

        elif shape_id == 3:  # spherical
            param1 = rng.uniform(0.6, 2.5, n_s)  # R/dc
            param2 = np.zeros(n_s)
            aspect_ratio_cav = param1 * 0.5

        elif shape_id == 4:  # elliptical
            param1 = rng.uniform(0.5, 3.0, n_s)  # a/b aspect ratio
            param2 = np.zeros(n_s)
            aspect_ratio_cav = param1

        elif shape_id == 5:  # ogive polynomial
            param1 = rng.uniform(1.0, 4.0, n_s)  # exponent n
            param2 = np.zeros(n_s)
            aspect_ratio_cav = param1 * 0.3

        # Effective theta
        theta_eff = np.array([
            theta_effective(shape_id, float(p1), float(p2))
            for p1, p2 in zip(param1, param2)
        ])
        beta = np.sin(np.radians(theta_eff))

        # Compute outputs
        Cdp_arr  = np.array([compute_Cdp(t, s) for t, s in zip(theta_eff, sigma_c)])
        Csf_arr  = np.array([compute_Csf(s, r, t) for s, r, t in zip(sigma_c, Re, theta_eff)])
        L_arr    = np.array([compute_L_tilde(t, s, a) for t, s, a in zip(theta_eff, sigma_c, aspect_ratio_cav)])
        D_arr    = np.array([compute_D_tilde(t, s, a) for t, s, a in zip(theta_eff, sigma_c, aspect_ratio_cav)])
        Cdb_arr  = np.array([compute_Cp_base(s) for s in sigma_c])
        covers   = np.array([cavity_covers_body(t, s) for t, s in zip(theta_eff, sigma_c)])
        drag_red = np.array([compute_drag_reduction_pct(c, s) for c, s in zip(Csf_arr, sigma_c)])

        Cd_arr = Cdp_arr + Csf_arr + Cdb_arr

        # CFD-level noise (1.5–3%)
        noise = lambda arr, pct: arr * (1 + rng.normal(0, pct, n_s))
        Cdp_n = noise(Cdp_arr, 0.015)
        Csf_n = noise(Csf_arr, 0.025)
        L_n   = noise(L_arr,   0.020)
        D_n   = noise(D_arr,   0.020)
        Cd_n  = Cdp_n + Csf_n + noise(Cdb_arr, 0.010)

        for i in range(n_s):
            rows.append({
                # ── Identity ─────────────────────────────────────────────────
                "shape_id":          shape_id,
                "shape_name":        shape_name,
                # ── Cavitator geometry (inputs) ───────────────────────────────
                "theta_eff":         round(float(theta_eff[i]), 4),
                "beta":              round(float(beta[i]), 4),
                "param1":            round(float(param1[i]), 4),
                "param2":            round(float(param2[i]), 4),
                "aspect_ratio_cav":  round(float(aspect_ratio_cav[i]), 4),
                "dc_mm":             BODY["dc"] * 1000,   # 12 mm constant
                # ── Fixed body geometry (context, not varied) ─────────────────
                "D_body_mm":         BODY["D_body"] * 1000,   # 30 mm
                "D_hemi_mm":         BODY["D_hemi"] * 1000,   # 15 mm
                "L_total_mm":        BODY["L_total"] * 1000,  # 115 mm
                "L_cyl_mm":          BODY["L_cyl"] * 1000,    # 85 mm
                "dc_D_ratio":        round(BODY["dc"] / BODY["D_body"], 4),  # 0.4
                # ── Operating conditions ──────────────────────────────────────
                "sigma_c":           round(float(sigma_c[i]), 5),
                "Re":                round(float(Re[i]), 0),
                "U_ms":              round(float(U[i]), 3),
                "depth_m":           round(float(depth_m[i]), 2),
                "Froude":            round(float(Froude[i]), 4),
                # ── Outputs ───────────────────────────────────────────────────
                "Cdp":               round(float(Cdp_n[i]), 6),
                "Csf":               round(float(Csf_n[i]), 6),
                "Cd":                round(float(Cd_n[i]), 6),
                "L_tilde":           round(float(L_n[i]), 4),
                "D_tilde":           round(float(D_n[i]), 4),
                "cavity_covers":     int(covers[i]),
                "drag_red_pct":      round(float(drag_red[i]), 3),
            })

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    return df


if __name__ == "__main__":
    print("Generating 1000-point cavitator dataset...")
    print(f"Fixed body: 30mm cylinder + 15mm hemisphere, 115mm total, dc=12mm")
    print("-" * 60)

    df = generate_dataset(1000)

    # Validate physics anchors
    disc = df[df["shape_name"] == "disc"]
    print(f"\nPhysics validation:")
    print(f"  Disc Cdp0        = {Cdp0(90):.4f}  (expect 0.827)")
    print(f"  45° cone Cdp0    = {Cdp0(45):.4f}  (expect ~0.42)")
    print(f"  Disc L_tilde@0.09 = {compute_L_tilde(90, 0.09, 0.02):.2f}  (expect ~9.2)")

    print(f"\nDataset summary:")
    print(f"  Shape  {df['shape_name'].value_counts().to_dict()}")
    print(f"  Cd     [{df['Cd'].min():.4f}, {df['Cd'].max():.4f}]")
    print(f"  L_tilde [{df['L_tilde'].min():.2f}, {df['L_tilde'].max():.2f}]")
    print(f"  Rows   {len(df)}, Cols {len(df.columns)}")
    print(f"  Columns: {df.columns.tolist()}")

    df.to_csv("/home/claude/cavitator_app/data/cavitator_dataset_1000.csv", index=False)
    print(f"\nSaved → cavitator_dataset_1000.csv")
