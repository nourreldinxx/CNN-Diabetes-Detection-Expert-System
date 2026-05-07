from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


PROJECT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
EDA_DIR = ARTIFACTS_DIR / "eda"
MODEL_PATH = ARTIFACTS_DIR / "diabetes_model.keras"
SCALER_PATH = ARTIFACTS_DIR / "scaler.pkl"
SK_MODEL_PATH = ARTIFACTS_DIR / "sk_model.joblib"
LEGACY_MODEL_PATH = PROJECT_DIR / "diabetes_model.h5"
LEGACY_SCALER_PATH = PROJECT_DIR / "scaler.pkl"
DATASET_PATH = PROJECT_DIR / "Dataset" / "diabetes.csv"

GIF_URL = "https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExMm81amRyejF6aHF1MXJwdG5ucGtvbW5ibm5qeHdlbzJiOXIxeDc5MiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/4XqguqDqRzP8Dlgp6S/giphy.gif"

# Must match the training dataframe column order:
FEATURE_ORDER = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

FEATURE_META = {
    "Pregnancies": {
        "label": "Pregnancies",
        "unit": "count",
        "meaning": "Number of times pregnant.",
    },
    "Glucose": {
        "label": "Glucose",
        "unit": "mg/dL",
        "meaning": "Plasma glucose concentration (2-hour oral glucose tolerance test).",
    },
    "BloodPressure": {
        "label": "Blood Pressure",
        "unit": "mm Hg",
        "meaning": "Diastolic blood pressure.",
    },
    "SkinThickness": {
        "label": "Skin Thickness",
        "unit": "mm",
        "meaning": "Triceps skin fold thickness.",
    },
    "Insulin": {
        "label": "Insulin",
        "unit": "µU/mL",
        "meaning": "2-hour serum insulin.",
    },
    "BMI": {
        "label": "BMI",
        "unit": "kg/m²",
        "meaning": "Body mass index (weight/height²).",
    },
    "DiabetesPedigreeFunction": {
        "label": "Diabetes Pedigree Function",
        "unit": "unitless",
        "meaning": "Family history-based diabetes likelihood score.",
    },
    "Age": {
        "label": "Age",
        "unit": "years",
        "meaning": "Age in years.",
    },
}


@st.cache_resource
def load_artifacts():
    """
    Streamlit Cloud currently runs very new Python versions where TensorFlow wheels may not exist.
    So we support two inference backends:
      1) TensorFlow model (.keras or legacy .h5) + scaler.pkl (local)
      2) scikit-learn fallback model (artifacts/sk_model.joblib) (cloud-friendly)
    """
    # Prefer sklearn fallback if present (cloud).
    if SK_MODEL_PATH.is_file():
        sk_model = joblib.load(SK_MODEL_PATH)
        # If a scaler exists, use it so inference matches training.
        scaler_path = SCALER_PATH if SCALER_PATH.is_file() else LEGACY_SCALER_PATH
        scaler = joblib.load(scaler_path) if scaler_path.is_file() else None
        return {"backend": "sklearn", "model": sk_model, "scaler": scaler}

    # Otherwise try TensorFlow (local).
    model_path = MODEL_PATH if MODEL_PATH.is_file() else LEGACY_MODEL_PATH
    scaler_path = SCALER_PATH if SCALER_PATH.is_file() else LEGACY_SCALER_PATH

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Missing model file: {MODEL_PATH} (or legacy {LEGACY_MODEL_PATH}) and no {SK_MODEL_PATH}"
        )
    if not scaler_path.is_file():
        raise FileNotFoundError(f"Missing scaler file: {SCALER_PATH} (or legacy {LEGACY_SCALER_PATH})")

    try:
        import tensorflow as tf  # lazy import

        tf.get_logger().setLevel("ERROR")
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "TensorFlow is not available in this environment. "
            f"Either deploy with {SK_MODEL_PATH.name} or run locally with TensorFlow installed.\n\nOriginal error: {e}"
        ) from e

    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    return {"backend": "tensorflow", "model": model, "scaler": scaler}


@st.cache_data
def load_dataset():
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Missing dataset file: {DATASET_PATH}")
    return pd.read_csv(DATASET_PATH)


@st.cache_data
def compute_feature_ranges(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    ranges: dict[str, dict[str, float]] = {}
    for c in FEATURE_ORDER:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        # Many Pima versions store missing medical measurements as 0.
        s_nonzero = s.replace(0, np.nan).dropna()
        if len(s_nonzero) == 0:
            s_nonzero = s.dropna()
        ranges[c] = {
            "min": float(np.nanmin(s_nonzero)),
            "max": float(np.nanmax(s_nonzero)),
            "median": float(np.nanmedian(s_nonzero)),
        }
    return ranges


def apply_theme():
    st.markdown(
        """
<style>
  .block-container { padding-top: 1.0rem; padding-bottom: 2rem; max-width: 1050px; }
  h1, h2, h3 { letter-spacing: -0.02em; }
  .muted { color: rgba(255,255,255,0.75); }
  @media (prefers-color-scheme: light) { .muted { color: rgba(0,0,0,0.65); } }
  div[data-testid="stMetric"] { border-radius: 14px; padding: 14px; border: 1px solid rgba(127,127,127,0.25); }
  .card { border-radius: 16px; padding: 16px; border: 1px solid rgba(127,127,127,0.25); background: rgba(127,127,127,0.06); }
</style>
""",
        unsafe_allow_html=True,
    )


def set_preset(preset: str) -> None:
    if preset == "positive":
        st.session_state["Pregnancies"] = 6
        st.session_state["Glucose"] = 170
        st.session_state["BloodPressure"] = 78
        st.session_state["SkinThickness"] = 35
        st.session_state["Insulin"] = 180
        st.session_state["BMI"] = 35.5
        st.session_state["DiabetesPedigreeFunction"] = 0.9
        st.session_state["Age"] = 50
    elif preset == "negative":
        st.session_state["Pregnancies"] = 1
        st.session_state["Glucose"] = 95
        st.session_state["BloodPressure"] = 65
        st.session_state["SkinThickness"] = 20
        st.session_state["Insulin"] = 85
        st.session_state["BMI"] = 24.0
        st.session_state["DiabetesPedigreeFunction"] = 0.25
        st.session_state["Age"] = 28
    else:
        raise ValueError(preset)


def main() -> None:
    st.set_page_config(page_title="Diabetes Predictor", page_icon="🩺", layout="centered")
    apply_theme()

    st.title("Diabetes Predictor")
    st.caption("Local Streamlit app using your trained model + dataset EDA.")
    cgif1, cgif2, cgif3 = st.columns([1, 2, 1])
    with cgif2:
        st.image(GIF_URL, width=360)

    page = st.sidebar.radio("Navigate", ["Predict", "Dataset"], index=0)

    try:
        bundle = load_artifacts()
    except Exception as e:
        st.error(
            "Could not load artifacts. Run the notebook cells that save the model/scaler first.\n\n"
            f"Expected (preferred):\n- {MODEL_PATH}\n- {SCALER_PATH}\n\n"
            f"Cloud-friendly fallback:\n- {SK_MODEL_PATH}\n\n"
            f"Fallback (legacy):\n- {LEGACY_MODEL_PATH}\n- {LEGACY_SCALER_PATH}\n\n"
            f"Error: {e}"
        )
        st.stop()

    if page == "Predict":
        st.subheader("Patient inputs")

        # Explain features + show dataset ranges
        try:
            df_ranges = compute_feature_ranges(load_dataset())
        except Exception:
            df_ranges = {}

        with st.expander("What do these features mean? Units + dataset ranges"):
            for c in FEATURE_ORDER:
                meta = FEATURE_META.get(c, {"label": c, "unit": "", "meaning": ""})
                r = df_ranges.get(c)
                if r:
                    st.markdown(
                        f"**{meta['label']}** ({meta['unit']}): {meta['meaning']}  \n"
                        f"- Dataset range (non‑zero): **{r['min']:.3g} → {r['max']:.3g}**  \n"
                        f"- Dataset median (non‑zero): **{r['median']:.3g}**"
                    )
                else:
                    st.markdown(f"**{meta['label']}** ({meta['unit']}): {meta['meaning']}")

        c0, c1, c2 = st.columns([1.3, 1, 1])
        with c0:
            st.markdown('<div class="card"><b>Quick tests</b><br/><span class="muted">Auto-fill inputs with known configs.</span></div>', unsafe_allow_html=True)
        with c1:
            if st.button("Test Positive", use_container_width=True):
                set_preset("positive")
                st.rerun()
        with c2:
            if st.button("Test Negative", use_container_width=True):
                set_preset("negative")
                st.rerun()

        a, b = st.columns(2)

        with a:
            st.number_input(
                "Pregnancies",
                min_value=0,
                max_value=20,
                step=1,
                key="Pregnancies",
                help=f"{FEATURE_META['Pregnancies']['meaning']} Unit: {FEATURE_META['Pregnancies']['unit']}.",
            )
            st.number_input(
                "Glucose",
                min_value=0,
                max_value=250,
                step=1,
                key="Glucose",
                help=f"{FEATURE_META['Glucose']['meaning']} Unit: {FEATURE_META['Glucose']['unit']}.",
            )
            st.number_input(
                "Blood Pressure",
                min_value=0,
                max_value=200,
                step=1,
                key="BloodPressure",
                help=f"{FEATURE_META['BloodPressure']['meaning']} Unit: {FEATURE_META['BloodPressure']['unit']}.",
            )
            st.number_input(
                "Skin Thickness",
                min_value=0,
                max_value=100,
                step=1,
                key="SkinThickness",
                help=f"{FEATURE_META['SkinThickness']['meaning']} Unit: {FEATURE_META['SkinThickness']['unit']}.",
            )

        with b:
            st.number_input(
                "Insulin",
                min_value=0,
                max_value=900,
                step=1,
                key="Insulin",
                help=f"{FEATURE_META['Insulin']['meaning']} Unit: {FEATURE_META['Insulin']['unit']}.",
            )
            st.number_input(
                "BMI",
                min_value=0.0,
                max_value=80.0,
                step=0.1,
                format="%.1f",
                key="BMI",
                help=f"{FEATURE_META['BMI']['meaning']} Unit: {FEATURE_META['BMI']['unit']}.",
            )
            st.number_input(
                "Diabetes Pedigree Function",
                min_value=0.0,
                max_value=3.0,
                step=0.01,
                format="%.2f",
                key="DiabetesPedigreeFunction",
                help=f"{FEATURE_META['DiabetesPedigreeFunction']['meaning']} Unit: {FEATURE_META['DiabetesPedigreeFunction']['unit']}.",
            )
            st.number_input(
                "Age",
                min_value=1,
                max_value=120,
                step=1,
                key="Age",
                help=f"{FEATURE_META['Age']['meaning']} Unit: {FEATURE_META['Age']['unit']}.",
            )

        if st.button("Predict", type="primary", use_container_width=True):
            row = pd.DataFrame([[st.session_state[c] for c in FEATURE_ORDER]], columns=FEATURE_ORDER)
            x = row.values.astype(np.float64)

            if bundle["backend"] == "tensorflow":
                x_scaled = bundle["scaler"].transform(x)
                proba = float(bundle["model"].predict(x_scaled, verbose=0).ravel()[0])
            else:
                # sklearn: model is trained on scaled features in the notebook, so we scale here too.
                x_in = bundle["scaler"].transform(x) if bundle["scaler"] is not None else x
                proba = float(bundle["model"].predict_proba(x_in)[:, 1][0])

            has_diabetes = proba >= 0.5
            st.divider()
            if has_diabetes:
                st.error("Prediction: **This person has diabetes**")
            else:
                st.success("Prediction: **This person doesn't have diabetes**")

            m1, m2 = st.columns(2)
            with m1:
                st.metric("Model probability P(Diabetes)", f"{proba*100:.2f}%")
            with m2:
                st.metric("Decision threshold", "0.50")

            with st.expander("Details"):
                st.write(
                    {
                        "threshold": 0.5,
                        "p_diabetes": round(proba, 4),
                        "p_no_diabetes": round(1 - proba, 4),
                        "features": {c: st.session_state[c] for c in FEATURE_ORDER},
                    }
                )

    else:
        st.subheader("Dataset overview")
        try:
            df = load_dataset()
        except Exception as e:
            st.error(str(e))
            st.stop()

        left, right = st.columns([1.2, 0.8])
        with left:
            st.markdown('<div class="card"><b>About this dataset</b><br/><span class="muted">Pima Indians Diabetes dataset: 8 numeric features + binary target <code>Outcome</code>.</span></div>', unsafe_allow_html=True)
            st.write(df.head(10))
        with right:
            if "Outcome" in df.columns:
                counts = df["Outcome"].value_counts().to_dict()
                st.metric("Rows", f"{len(df)}")
                st.metric("Outcome=1 (Diabetes)", str(counts.get(1, 0)))
                st.metric("Outcome=0 (No diabetes)", str(counts.get(0, 0)))

        st.divider()
        st.subheader("EDA & visualizations")
        if not EDA_DIR.exists():
            st.warning(
                "EDA graphs are not generated yet. Re-run the notebook to generate and save graphs into "
                f"`{EDA_DIR}` (there is an EDA cell that writes PNGs)."
            )
            st.stop()

        pngs = sorted([p for p in EDA_DIR.glob("*.png")])
        if not pngs:
            st.warning(
                f"No graphs found in `{EDA_DIR}`. Re-run the notebook EDA cell to create them."
            )
            st.stop()

        for p in pngs:
            st.image(str(p), caption=p.name, use_container_width=True)


if __name__ == "__main__":
    main()

