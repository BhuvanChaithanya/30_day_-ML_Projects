import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Salary Predictor — Linear Regression",
    page_icon="💰",
    layout="wide",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}

/* Hero section */
.hero-container {
    text-align: center;
    padding: 3rem 1rem 2rem;
}
.hero-title {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #f857a6, #ff5858, #f7971e, #FFD200);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.4rem;
}
.hero-subtitle {
    font-size: 1.15rem;
    color: rgba(255,255,255,0.65);
    margin-bottom: 0;
}

/* Glassmorphism card */
.glass-card {
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 20px;
    padding: 2rem 2.5rem;
    backdrop-filter: blur(12px);
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}

/* Section headings */
.section-title {
    font-size: 1.35rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 1rem;
    letter-spacing: 0.02em;
}



/* Prediction result */
.prediction-box {
    text-align: center;
    background: linear-gradient(135deg, #f857a6 0%, #ff5858 50%, #f7971e 100%);
    border-radius: 20px;
    padding: 2rem;
    margin-top: 1.2rem;
}
.prediction-label {
    font-size: 0.95rem;
    color: rgba(255,255,255,0.85);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 600;
}
.prediction-value {
    font-size: 3rem;
    font-weight: 800;
    color: #ffffff;
    line-height: 1.1;
    margin-top: 0.4rem;
}
.prediction-sub {
    font-size: 0.85rem;
    color: rgba(255,255,255,0.7);
    margin-top: 0.3rem;
}

/* Slider label colour */
.stSlider label { color: rgba(255,255,255,0.85) !important; font-weight: 600; }

/* Number input label */
.stNumberInput label { color: rgba(255,255,255,0.85) !important; font-weight: 600; }

/* Divider */
hr { border-color: rgba(255,255,255,0.1) !important; }

/* Matplotlib / seaborn plot background */
.stPlotlyChart, .stPyplot { border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ─── Load & Train ─────────────────────────────────────────────────────────────
@st.cache_data
def load_and_train():
    """Load the CSV and train a LinearRegression model once."""
    csv_path = os.path.join(os.path.dirname(__file__), "Experience-Salary.csv")
    df = pd.read_csv(csv_path)

    X = df[["exp(in months)"]].values
    y = df["salary(in thousands)"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2    = r2_score(y_test, y_pred)
    rmse  = np.sqrt(mean_squared_error(y_test, y_pred))

    return df, model, X_train, X_test, y_train, y_test, y_pred, r2, rmse


df, model, X_train, X_test, y_train, y_test, y_pred, r2, rmse = load_and_train()

exp_min  = float(df["exp(in months)"].min())
exp_max  = float(df["exp(in months)"].max())
exp_mean = float(df["exp(in months)"].mean())


# ─── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-container">
    <div class="hero-title">💰 Salary Predictor</div>
    <div class="hero-subtitle">
        Linear Regression &nbsp;·&nbsp; Experience (months) → Salary ($K)
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ─── Layout: two columns ──────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1.6], gap="large")


# ─── LEFT: Prediction Panel ───────────────────────────────────────────────────
with col_left:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🎯 Predict Your Salary</div>', unsafe_allow_html=True)

    st.write("")

    # Slider input
    experience = st.slider(
        "Months of Experience",
        min_value=float(np.floor(exp_min)),
        max_value=float(np.ceil(exp_max)),
        value=exp_mean,
        step=1.0,
        help="Drag to set your experience in months",
    )

    # Number input (synced alternative)
    exp_typed = st.number_input(
        "Or type exact value (months)",
        min_value=float(np.floor(exp_min)),
        max_value=float(np.ceil(exp_max)),
        value=experience,
        step=1.0,
    )

    final_exp = exp_typed
    predicted_salary = model.predict([[final_exp]])[0]

    st.markdown(f"""
    <div class="prediction-box">
        <div class="prediction-label">Predicted Salary</div>
        <div class="prediction-value">${predicted_salary:,.1f}K</div>
        <div class="prediction-sub">≈ ${predicted_salary * 1000:,.0f} / year</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Model metrics card
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Model Performance</div>', unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("R² Score", f"{r2:.4f}")
    m2.metric("RMSE ($K)", f"{rmse:.2f}")
    m3.metric("Training rows", f"{len(X_train):,}")

    st.markdown(f"""
    <div style="color:rgba(255,255,255,0.6); font-size:0.82rem; margin-top:0.8rem;">
        <b style="color:rgba(255,255,255,0.9);">Equation:</b>
        Salary = {model.coef_[0]:.4f} × Months + {model.intercept_:.4f}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ─── RIGHT: Charts ────────────────────────────────────────────────────────────
with col_right:
    # ── Chart 1: scatter + regression line ──
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📈 Experience vs Salary</div>', unsafe_allow_html=True)

    fig1, ax1 = plt.subplots(figsize=(7, 4))
    fig1.patch.set_facecolor("#1a1a2e")
    ax1.set_facecolor("#16213e")

    ax1.scatter(
        df["exp(in months)"], df["salary(in thousands)"],
        alpha=0.35, color="#667eea", s=15, label="Data points"
    )

    x_line = np.linspace(exp_min, exp_max, 200).reshape(-1, 1)
    y_line = model.predict(x_line)
    ax1.plot(x_line, y_line, color="#f857a6", linewidth=2.5, label="Regression line")

    # Highlight user prediction
    ax1.scatter([final_exp], [predicted_salary], color="#FFD200", s=120,
                zorder=5, label=f"Your point ({final_exp:.0f} mo → ${predicted_salary:.1f}K)")

    ax1.set_xlabel("Experience (months)", color="white", fontsize=10)
    ax1.set_ylabel("Salary ($K)", color="white", fontsize=10)
    ax1.tick_params(colors="white")
    for spine in ax1.spines.values():
        spine.set_edgecolor((1, 1, 1, 0.2))
    ax1.legend(facecolor="#16213e", labelcolor="white", fontsize=8)
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Chart 2: Actual vs Predicted ──
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🎯 Actual vs Predicted (Test Set)</div>', unsafe_allow_html=True)

    fig2, ax2 = plt.subplots(figsize=(7, 3.5))
    fig2.patch.set_facecolor("#1a1a2e")
    ax2.set_facecolor("#16213e")

    ax2.scatter(y_test, y_pred, alpha=0.45, color="#f7971e", s=18, label="Predictions")
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax2.plot(lims, lims, color="#f857a6", linewidth=1.8, linestyle="--", label="Perfect fit")

    ax2.set_xlabel("Actual Salary ($K)", color="white", fontsize=10)
    ax2.set_ylabel("Predicted Salary ($K)", color="white", fontsize=10)
    ax2.tick_params(colors="white")
    for spine in ax2.spines.values():
        spine.set_edgecolor((1, 1, 1, 0.2))
    ax2.legend(facecolor="#16213e", labelcolor="white", fontsize=8)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)
    st.markdown("</div>", unsafe_allow_html=True)


# ─── Dataset Explorer ─────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("🗄️  Explore the Dataset", expanded=False):
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Sample rows**")
        st.dataframe(df.head(10), use_container_width=True)
    with col_b:
        st.markdown("**Descriptive statistics**")
        st.dataframe(df.describe(), use_container_width=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; margin-top:3rem; color:rgba(255,255,255,0.3); font-size:0.8rem;">
    Day 1 · 30-Day ML Challenge &nbsp;|&nbsp; Built with Streamlit &amp; scikit-learn
</div>
""", unsafe_allow_html=True)
