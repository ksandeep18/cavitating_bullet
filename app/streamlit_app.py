import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib, json, os, sys

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cavitator ML Design Tool",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Load models directly (no API dependency) ──────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE, "models")

@st.cache_resource
def load_models():
    scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")
    le     = joblib.load(f"{MODEL_DIR}/label_encoder.pkl")
    with open(f"{MODEL_DIR}/metadata.json") as f:
        meta = json.load(f)
    mods = {}
    for t in meta["targets"]:
        mods[t] = joblib.load(f"{MODEL_DIR}/{t}_best.pkl")
    return scaler, le, meta, mods

scaler, le, meta, models = load_models()
TARGETS  = meta["targets"]
FEATURES = meta["features"]

# ── Physics helpers ───────────────────────────────────────────────────────────
def theta_eff(shape, p1, p2):
    s = {"disc":0,"cone":1,"truncated_cone":2,"spherical":3,
         "elliptical":4,"ogive_polynomial":5}.get(shape, 0)
    if s==0: return 90.0
    if s==1: return float(np.clip(p1,20,80))
    if s==2: return float(np.clip(p1*(1-p2*0.25),15,85))
    if s==3: return float(np.degrees(np.arctan(1/(2*max(p1,0.1))))*1.8)
    if s==4: return float(np.degrees(np.arctan(1/max(p1,0.1)))*1.6)
    if s==5: return float(20+60/max(p1,1))
    return 45.0

def ar_cav(shape, p1):
    if shape=="disc":             return 0.02
    if shape=="cone":             return 1/(2*np.tan(np.radians(max(p1,1))))
    if shape=="truncated_cone":   return max(p1*0.015,0.05)
    if shape=="spherical":        return p1*0.5
    if shape=="elliptical":       return p1
    if shape=="ogive_polynomial": return p1*0.3
    return 0.5

def predict(shape, p1, p2, sigma_c, Re, depth_m):
    enc    = le.transform([shape])[0]
    theta  = theta_eff(shape, p1, p2)
    beta   = np.sin(np.radians(theta))
    ar     = ar_cav(shape, p1)
    U_ms   = Re*1.002e-3/(1000*0.115)
    Froude = U_ms/np.sqrt(9.81*0.115)
    x  = np.array([[enc,theta,beta,p1,p2,ar,sigma_c,Re,depth_m,Froude]])
    xs = scaler.transform(x)
    return {t: float(models[t].predict(xs)[0]) for t in TARGETS}, theta, U_ms

def drag_reduction(Csf, Re):
    Re_cyl = Re*0.085/0.115
    Cf_b   = 0.074*max(Re_cyl,1e4)**(-0.2)
    A_nd   = (np.pi*0.030*0.085)/(np.pi*0.012**2/4)
    Csf_b  = Cf_b*A_nd
    return float(np.clip((Csf_b-Csf)/Csf_b*100,0,100))

# ── Draw cavitator shape ──────────────────────────────────────────────────────
def draw_body_and_cavity(shape, p1, p2, preds, theta, ax):
    dc=12; D_body=30; D_hemi=15; L_cyl=85; L_total=115
    dc_half=dc/2; Db_half=D_body/2; Dh_half=D_hemi/2

    ax.set_facecolor("#0a0f1e")
    ax.set_aspect("equal")

    # Body outline
    body_x=[0,L_cyl,L_cyl+7.5,L_cyl,0]
    body_ytop=[Db_half,Db_half,Dh_half,0,0]
    body_ybot=[-b for b in body_ytop]
    ax.fill_betweenx(np.linspace(-Db_half,Db_half,50), 0, L_cyl,
                     color="#2c3e50", alpha=0.9, zorder=2)
    # Hemisphere
    t_vals=np.linspace(-np.pi/2,np.pi/2,60)
    hemi_x=L_cyl+7.5*np.sin(t_vals)
    hemi_y=7.5*np.cos(t_vals)
    ax.fill(hemi_x, hemi_y, color="#2c3e50", zorder=2)
    ax.plot(hemi_x, hemi_y, color="#5d8aa8", lw=1.2, zorder=3)
    ax.plot([0,L_cyl],[Db_half,Db_half], color="#5d8aa8", lw=1.2, zorder=3)
    ax.plot([0,L_cyl],[-Db_half,-Db_half], color="#5d8aa8", lw=1.2, zorder=3)
    ax.plot([0,0],[-Db_half,Db_half], color="#5d8aa8", lw=1.2, zorder=3)

    # Cavitator
    cav_L = 8 if shape=="disc" else max(5, dc/2/np.tan(np.radians(max(theta,5))))
    cav_L = min(cav_L, 20)
    if shape=="disc":
        ax.fill([-cav_L,-cav_L,0,0],[-dc_half,dc_half,dc_half,-dc_half],
                color="#e74c3c", alpha=0.9, zorder=4)
    elif shape in ["cone","truncated_cone"]:
        tip_r = dc_half*p2 if shape=="truncated_cone" else 0
        ax.fill([-cav_L,-cav_L,0,0],
                [-dc_half,dc_half,tip_r,-tip_r],
                color="#e67e22", alpha=0.9, zorder=4)
    elif shape=="spherical":
        R=dc_half*p1
        cx=-cav_L+R; ang=np.linspace(-np.pi/2,np.pi/2,60)
        hx=cx+R*np.sin(ang); hy=R*np.cos(ang)
        ax.fill(np.append(hx,[cx+1,cx+1]),np.append(hy,[-dc_half,dc_half]),
                color="#9b59b6",alpha=0.9,zorder=4)
    elif shape=="elliptical":
        ang=np.linspace(-np.pi/2,np.pi/2,60)
        a=cav_L; b=dc_half
        ex=-a*np.sin(ang); ey=b*np.cos(ang)
        ax.fill(np.append(ex,[0,0]),np.append(ey,[-dc_half,dc_half]),
                color="#1abc9c",alpha=0.9,zorder=4)
    elif shape=="ogive_polynomial":
        n=max(p1,1)
        x_og=np.linspace(0,cav_L,40)
        y_og=dc_half*(1-(x_og/cav_L)**n)
        ax.fill(np.append(-x_og[::-1],[0,0]),
                np.append(y_og[::-1],[-dc_half,dc_half]),
                color="#3498db",alpha=0.9,zorder=4)

    # Cavity bubble
    L_cav=preds["L_tilde"]*dc
    D_cav=preds["D_tilde"]*dc/2
    if L_cav>5:
        from matplotlib.patches import Ellipse
        el=Ellipse(xy=(L_cav/2-cav_L, 0), width=L_cav, height=D_cav*2,
                   color="cyan", alpha=0.12, zorder=1)
        ax.add_patch(el)
        ax.plot([-cav_L, L_cav-cav_L],[D_cav,D_cav],'c--',lw=0.8,alpha=0.5)
        ax.plot([-cav_L, L_cav-cav_L],[-D_cav,-D_cav],'c--',lw=0.8,alpha=0.5)
        ax.annotate(f"L_cav={L_cav:.0f}mm",
                    xy=(L_cav/2-cav_L, D_cav+2), color='cyan',
                    fontsize=7, ha='center')

    # Dimensions
    ax.annotate("", xy=(L_total,Db_half+8), xytext=(0,Db_half+8),
                arrowprops=dict(arrowstyle="<->",color="white",lw=0.8))
    ax.text(L_total/2, Db_half+11, "115 mm", color="white",
            ha="center", va="bottom", fontsize=7)
    ax.text(-cav_L/2, -Db_half-8, f"dc=12mm", color="#e74c3c",
            ha="center", fontsize=7)

    ax.set_xlim(-25, 130); ax.set_ylim(-30, 35)
    ax.set_xlabel("Axial position [mm]", color="white", fontsize=8)
    ax.set_ylabel("Radial [mm]", color="white", fontsize=8)
    ax.tick_params(colors='white', labelsize=7)
    for sp in ax.spines.values(): sp.set_color('#333')
    ax.set_title(f"Body + {shape.replace('_',' ')} cavitator", 
                 color="white", fontsize=9, pad=4)

# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.main-title {font-size:2.2rem; font-weight:700; color:#00d4ff; margin-bottom:0}
.sub-title  {font-size:1rem; color:#8899aa; margin-top:0; margin-bottom:1.5rem}
.metric-card {background:#0d1b2a; border:1px solid #1e3a5f; border-radius:10px;
              padding:12px 16px; text-align:center}
.metric-val  {font-size:1.6rem; font-weight:700; color:#00d4ff}
.metric-lbl  {font-size:0.75rem; color:#8899aa; margin-top:2px}
.highlight   {color:#ff6b35; font-weight:600}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🚀 Cavitator ML Design Tool</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Fixed body: 30mm cylinder + 15mm hemisphere | 115mm total | dc = 12mm</p>',
            unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.shields.io/badge/Model-XGBoost%20%7C%20RF-blue")
    st.markdown("### 🎛️ Design Parameters")

    shape = st.selectbox("Cavitator shape", [
        "disc","cone","truncated_cone","spherical","elliptical","ogive_polynomial"],
        format_func=lambda x: x.replace("_"," ").title())

    st.markdown("#### Shape parameter(s)")
    shape_help = {
        "disc":             ("No parameters — flat face", None, None),
        "cone":             ("Half-angle θ [°]", 20.0, 80.0),
        "truncated_cone":   ("Half-angle θ [°]", 30.0, 75.0),
        "spherical":        ("Radius ratio R/dc", 0.6, 2.5),
        "elliptical":       ("Axis ratio a/b", 0.5, 3.0),
        "ogive_polynomial": ("Exponent n", 1.0, 4.0),
    }
    help_txt, lo, hi = shape_help[shape]

    if shape == "disc":
        st.info("Flat face — no free parameters")
        param1 = 90.0; param2 = 0.0
    else:
        param1 = st.slider(help_txt, float(lo), float(hi),
                           float((lo+hi)/2), step=0.5)
        if shape == "truncated_cone":
            param2 = st.slider("Truncation ratio [0=cone, 1=disc]", 0.1, 0.7, 0.3, 0.05)
        else:
            param2 = 0.0

    st.markdown("#### Operating conditions")
    sigma_c = st.slider("Cavitation number σ_c", 0.05, 0.40, 0.15, 0.01)
    Re_exp  = st.slider("Reynolds number (log₁₀)", 5.0, 6.7, 6.0, 0.1)
    Re      = 10 ** Re_exp
    depth_m = st.slider("Depth [m]", 0.0, 50.0, 10.0, 1.0)
    st.caption(f"Re = {Re:.2e} | U ≈ {Re*1.002e-3/(1000*0.115):.1f} m/s")

    predict_btn = st.button("🔮 Predict", type="primary", use_container_width=True)

# ── Prediction ────────────────────────────────────────────────────────────────
preds, theta, U_ms = predict(shape, param1, param2, sigma_c, Re, depth_m)
drag_red = drag_reduction(preds["Csf"], Re)
dc_m = 0.012
L_cav_mm = preds["L_tilde"] * dc_m * 1000
D_cav_mm = preds["D_tilde"] * dc_m * 1000
covers    = preds["L_tilde"] * dc_m >= 0.115

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Results", "🔍 Shape Comparison", "⚡ Optimizer", "📚 Model Info"])

# ── TAB 1: Results ─────────────────────────────────────────────────────────────
with tab1:
    col1, col2 = st.columns([1.4, 1])

    with col1:
        fig, ax = plt.subplots(figsize=(8, 3.5), facecolor="#0a0f1e")
        draw_body_and_cavity(shape, param1, param2, preds, theta, ax)
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # Drag breakdown bar
        fig2, ax2 = plt.subplots(figsize=(8, 1.5), facecolor="#0a0f1e")
        ax2.set_facecolor("#0a0f1e")
        total = preds["Cdp"] + preds["Csf"] + 0.12
        w1 = preds["Cdp"]/total; w2 = preds["Csf"]/total; w3 = 1-w1-w2
        ax2.barh(0, w1, color="#e74c3c", label=f"Cdp={preds['Cdp']:.4f}")
        ax2.barh(0, w2, left=w1, color="#f39c12", label=f"Csf={preds['Csf']:.4f}")
        ax2.barh(0, w3, left=w1+w2, color="#2ecc71", label=f"Cdb≈0.12")
        ax2.set_xlim(0,1); ax2.set_yticks([])
        ax2.legend(loc="upper right", fontsize=7, framealpha=0.2,
                   labelcolor="white")
        ax2.set_xlabel("Fraction of total drag", color="white", fontsize=8)
        ax2.tick_params(colors='white', labelsize=7)
        for sp in ax2.spines.values(): sp.set_color('#333')
        ax2.set_title("Drag breakdown", color="white", fontsize=8, pad=3)
        st.pyplot(fig2, use_container_width=True)
        plt.close()

    with col2:
        st.markdown("#### Predicted outputs")
        metrics = [
            ("Total Cd", f"{preds['Cd']:.5f}", "Total drag coefficient"),
            ("Cdp", f"{preds['Cdp']:.5f}", "Pressure drag"),
            ("Csf", f"{preds['Csf']:.5f}", "Skin friction"),
            ("L̃ = L/dc", f"{preds['L_tilde']:.3f}", f"Cavity length ≈ {L_cav_mm:.1f} mm"),
            ("D̃ = D/dc", f"{preds['D_tilde']:.3f}", f"Cavity diameter ≈ {D_cav_mm:.1f} mm"),
            ("Drag reduction", f"{drag_red:.1f}%", "vs fully wetted"),
        ]
        for label, val, note in metrics:
            st.markdown(f"""
            <div class="metric-card" style="margin-bottom:8px">
              <div class="metric-val">{val}</div>
              <div class="metric-lbl">{label} — {note}</div>
            </div>""", unsafe_allow_html=True)

        # Status badge
        if covers:
            st.success("✅ Full supercavity — body completely enveloped")
        elif L_cav_mm >= 85:
            st.warning("⚠️ Partial cavity — cylinder covered, hemisphere exposed")
        else:
            st.error("❌ Small cavity — most of body still wetted")

        st.markdown(f"**θ_eff** = {theta:.1f}° &nbsp;|&nbsp; **β** = {np.sin(np.radians(theta)):.3f}")
        st.markdown(f"**U** ≈ {U_ms:.1f} m/s &nbsp;|&nbsp; **σ_c** = {sigma_c}")

# ── TAB 2: Shape comparison ───────────────────────────────────────────────────
with tab2:
    st.markdown("#### Compare all shapes at current operating conditions")
    shape_params_cmp = {
        "Disc":             ("disc",            90.0, 0.0),
        "Cone 30°":         ("cone",            30.0, 0.0),
        "Cone 45°":         ("cone",            45.0, 0.0),
        "Cone 60°":         ("cone",            60.0, 0.0),
        "Trunc cone 50°":   ("truncated_cone",  50.0, 0.4),
        "Spherical R/dc=1": ("spherical",        1.0, 0.0),
        "Elliptical a/b=2": ("elliptical",       2.0, 0.0),
        "Ogive n=2":        ("ogive_polynomial", 2.0, 0.0),
    }

    rows_cmp = []
    for label, (sn, p1, p2) in shape_params_cmp.items():
        pr, th, _ = predict(sn, p1, p2, sigma_c, Re, depth_m)
        dr = drag_reduction(pr["Csf"], Re)
        rows_cmp.append({"Shape": label, "Cd": pr["Cd"], "Cdp": pr["Cdp"],
                         "Csf": pr["Csf"], "L̃": pr["L_tilde"],
                         "D̃": pr["D_tilde"], "θ_eff": round(th,1),
                         "Drag red%": round(dr,1)})

    df_cmp = pd.DataFrame(rows_cmp).sort_values("Cd")
    st.dataframe(df_cmp.style.highlight_min(subset=["Cd"], color="#1a472a")
                             .highlight_max(subset=["Drag red%"], color="#1a472a")
                             .format({"Cd":"{:.5f}","Cdp":"{:.5f}","Csf":"{:.5f}",
                                      "L̃":"{:.3f}","D̃":"{:.3f}"}),
                 use_container_width=True)

    # Bar chart comparison
    fig3, axes = plt.subplots(1, 3, figsize=(14, 4), facecolor="#0a0f1e")
    colors_bar = plt.cm.viridis(np.linspace(0.2, 0.9, len(df_cmp)))
    for ax_i, (col, title) in enumerate(
            [("Cd","Total drag Cd"),("L̃","Cavity length L̃"),("Drag red%","Drag reduction %")]):
        axes[ax_i].set_facecolor("#0a0f1e")
        bars = axes[ax_i].barh(df_cmp["Shape"], df_cmp[col],
                               color=colors_bar, edgecolor="#222")
        axes[ax_i].set_title(title, color="white", fontsize=10)
        axes[ax_i].tick_params(colors="white", labelsize=8)
        for sp in axes[ax_i].spines.values(): sp.set_color("#333")
        axes[ax_i].set_xlabel(col, color="white", fontsize=8)
    plt.tight_layout()
    st.pyplot(fig3, use_container_width=True)
    plt.close()

# ── TAB 3: Optimizer ──────────────────────────────────────────────────────────
with tab3:
    st.markdown("#### Find optimal cavitator for given operating conditions")
    c1, c2, c3 = st.columns(3)
    opt_sigma  = c1.slider("σ_c", 0.05, 0.40, sigma_c, 0.01, key="opt_s")
    opt_Re_exp = c2.slider("Re (log₁₀)", 5.0, 6.7, 6.0, 0.1, key="opt_r")
    opt_depth  = c3.slider("Depth [m]", 0.0, 50.0, depth_m, 1.0, key="opt_d")
    opt_Re     = 10 ** opt_Re_exp

    if st.button("🔍 Find Optimal Shape", type="primary"):
        with st.spinner("Searching design space..."):
            all_res = []
            for sn in ["disc","cone","truncated_cone","spherical",
                       "elliptical","ogive_polynomial"]:
                p2_vals = [0.0]
                if sn=="cone":              p1_vals=np.linspace(20,80,30)
                elif sn=="truncated_cone":  p1_vals=np.linspace(30,75,15); p2_vals=np.linspace(0.1,0.7,5)
                elif sn=="spherical":       p1_vals=np.linspace(0.6,2.5,20)
                elif sn=="elliptical":      p1_vals=np.linspace(0.5,3.0,20)
                elif sn=="ogive_polynomial":p1_vals=np.linspace(1.0,4.0,15)
                else:                       p1_vals=[90.0]
                for p1 in p1_vals:
                    for p2 in p2_vals:
                        pr,th,_ = predict(sn,float(p1),float(p2),opt_sigma,opt_Re,opt_depth)
                        dr = drag_reduction(pr["Csf"],opt_Re)
                        all_res.append({"shape":sn,"param1":round(p1,2),"param2":round(p2,2),
                                        "theta":round(th,1),"Cd":pr["Cd"],"L̃":pr["L_tilde"],
                                        "Drag red%":round(dr,1)})

            df_opt = pd.DataFrame(all_res).sort_values("Cd").reset_index(drop=True)
            best   = df_opt.iloc[0]

            st.success(f"✅ Optimal: **{best['shape']}** | θ_eff={best['theta']}° | param1={best['param1']}")
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Best Cd",     f"{best['Cd']:.5f}")
            col_b.metric("Cavity L̃",    f"{best['L̃']:.3f}")
            col_c.metric("Drag red%",   f"{best['Drag red%']:.1f}%")

            st.markdown("**Top 10 candidates:**")
            st.dataframe(df_opt.head(10).style.highlight_min(
                subset=["Cd"], color="#1a472a").format(
                {"Cd":"{:.5f}","L̃":"{:.3f}"}), use_container_width=True)

            # Landscape
            fig4, ax4 = plt.subplots(figsize=(10, 4), facecolor="#0a0f1e")
            ax4.set_facecolor("#0a0f1e")
            cone_rows = df_opt[df_opt["shape"]=="cone"].sort_values("theta")
            if len(cone_rows)>2:
                ax4.plot(cone_rows["theta"], cone_rows["Cd"],
                         "o-", color="#e74c3c", lw=2, ms=4, label="Cone")
            ax4.axhline(best["Cd"], color="cyan", ls="--", lw=1.2,
                        label=f"Optimum={best['Cd']:.4f}")
            ax4.set_xlabel("θ_eff [°]", color="white"); ax4.set_ylabel("Cd", color="white")
            ax4.set_title("Cd landscape — cone family", color="white", fontsize=10)
            ax4.legend(fontsize=9, labelcolor="white", framealpha=0.2)
            ax4.tick_params(colors="white")
            for sp in ax4.spines.values(): sp.set_color("#333")
            st.pyplot(fig4, use_container_width=True)
            plt.close()

# ── TAB 4: Model info ─────────────────────────────────────────────────────────
with tab4:
    st.markdown("#### Model performance")
    m_rows = []
    for target, scores in meta["metrics"].items():
        if isinstance(scores, dict) and all(isinstance(v, dict) for v in scores.values()):
            for model_name, s in scores.items():
                m_rows.append({"Target": target, "Model": model_name,
                               "R²": s.get("r2", np.nan), "RMSE": s.get("rmse", np.nan)})
        elif isinstance(scores, dict):
            for model_name, value in scores.items():
                if model_name == "best":
                    continue
                m_rows.append({"Target": target, "Model": model_name,
                               "R²": float(value), "RMSE": np.nan})
        else:
            m_rows.append({"Target": target, "Model": "best", "R²": float(scores), "RMSE": np.nan})
    df_m = pd.DataFrame(m_rows)
    st.dataframe(df_m.style.highlight_max(subset=["R²"], color="#1a472a")
                           .highlight_min(subset=["RMSE"], color="#1a472a")
                           .format({"R²": "{:.4f}", "RMSE": lambda x: "" if pd.isna(x) else f"{x:.6f}"}),
                 use_container_width=True)

    st.markdown("#### Feature importance (SHAP on XGBoost → Cd)")
    fi  = meta["feat_importance_shap"]
    fig5, ax5 = plt.subplots(figsize=(8, 3.5), facecolor="#0a0f1e")
    ax5.set_facecolor("#0a0f1e")
    feats = list(fi.keys()); imps = list(fi.values())
    idx = np.argsort(imps)
    bars = ax5.barh([feats[i] for i in idx], [imps[i] for i in idx],
                    color=plt.cm.plasma(np.linspace(0.2,0.9,len(feats))))
    for b,v in zip(bars,[imps[i] for i in idx]):
        ax5.text(b.get_width()+0.002,b.get_y()+b.get_height()/2,
                 f"{v:.3f}",va="center",color="white",fontsize=8)
    ax5.set_xlabel("SHAP importance",color="white",fontsize=9)
    ax5.tick_params(colors="white",labelsize=8)
    for sp in ax5.spines.values(): sp.set_color("#333")
    ax5.set_title("Feature importance for Cd prediction",color="white",fontsize=10)
    plt.tight_layout()
    st.pyplot(fig5, use_container_width=True)
    plt.close()

    st.markdown("#### Dataset & body specs")
    col1, col2 = st.columns(2)
    col1.markdown("""
    **Fixed body geometry:**
    - Total length: 115 mm
    - Cylinder: ∅30 mm × 85 mm
    - Hemisphere: ∅15 mm
    - Cavitator: ∅12 mm (dc = 12 mm)

    **Dataset:**
    - 1,000 training points
    - 6 cavitator shape families
    - σ_c ∈ [0.05, 0.40]
    - Re ∈ [1×10⁵, 5×10⁶]
    """)
    col2.markdown("""
    **Physics sources:**
    - Cdp: Reichardt (1946) formula
    - Cdp0 anchors: Rouse & McNown (1948)
    - L̃ scaling: Logvinovich (1980)
    - D̃ scaling: Garabedian (1956)
    - Csf: Prandtl turbulent BL theory

    **Models:**
    - Random Forest (R² > 0.97)
    - XGBoost (R² > 0.99)
    - Best model auto-selected per target
    """)
