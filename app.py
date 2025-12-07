import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time
from scipy.special import boxcox1p
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration & Page Setup
st.set_page_config(
    page_title="Ames Housing Price Predictor",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_DIR = "saved_models"

# Asset Loading System


@st.cache_resource
def load_artifacts():
    """Loads the artifacts dictionary containing modes, medians, and skewness info."""
    path = os.path.join(MODEL_DIR, "artifacts.pkl")
    if not os.path.exists(path):
        st.error(
            f"[Critical Error]: '{path}' not found. Please run the notebook to generate artifacts.")
        st.stop()
    return joblib.load(path)


def load_model(filename):
    """Loads a specific serialized model."""
    path = os.path.join(MODEL_DIR, filename)
    if os.path.exists(path):
        return joblib.load(path)
    else:
        st.error(f"Model '{filename}' not found.")
        return None


artifacts = load_artifacts()

# 3. Sidebar - Configuration
with st.sidebar:
    st.title("⚙️ Model Settings")
    st.markdown("### Choose Inference Engine")

    # Select Category
    model_category = st.radio(
        "Model Category",
        ["Ensemble Methods", "Boosting", "Linear Models"]
    )

    # Select Specific Algorithm based on Category
    if model_category == "Ensemble Methods":
        # Both the Blend and the Stack are ensembles
        selected_model_name = st.selectbox(
            "Algorithm",
            ["Geometric_Blend", "Stacking_Ensemble"]
        )

    elif model_category == "Boosting":
        selected_model_name = st.selectbox(
            "Algorithm",
            ["CatBoost", "XGBoost", "LightGBM", "GradientBoosting"]
        )

    else:  # Linear Models
        selected_model_name = st.selectbox(
            "Algorithm",
            ["Lasso", "Ridge", "ElasticNet", "Linear", "KernelRidge"]
        )

    st.markdown("---")
    st.info("""
    **About:**
    This app uses the **Ames Housing Dataset**.

    **Preprocessing:**
    - Log-Target Transformation
    - Box-Cox for Skewed Features
    - Polynomial Feature Engineering
    - Hybrid Encoding
    """)

# 4. Preprocessing Pipeline


def preprocess_input(input_dict, artifacts):
    """
    Reconstructs the dataframe row to exactly match the training data structure.
    1. Load Medians/Modes (Baseline)
    2. Overwrite with User Inputs
    3. Generate Engineered Features (TotalSF, Polynomials)
    4. Apply Box-Cox Transformation
    5. Align Columns (One-Hot Encoding)
    """

    # Initialize DataFrame with training medians (handles features we don't ask
    # for)
    df = pd.DataFrame([artifacts["impute_medians"]])

    # Update with User Input (Categorical & Numerical)
    for key, value in input_dict.items():
        # Handle One-Hot Encoding manually for inputs
        # e.g., if user selects Neighborhood="CollgCr", set
        # Neighborhood_CollgCr = 1
        if isinstance(value, str):
            col_name = f"{key}_{value}"
            # Only set if this column existed in training
            if col_name in artifacts['features']:
                df[col_name] = 1
        else:
            df[key] = value

    # Feature Engineering
    # Estimate floor split if not provided (Simplification for UI)
    if '1stFlrSF' not in input_dict:
        df['1stFlrSF'] = df['GrLivArea'] * 0.6
        df['2ndFlrSF'] = df['GrLivArea'] * 0.4

    # Primary Synthetics
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']

    # Interaction Terms
    df['TotalSF_Qual'] = df['TotalSF'] * df['OverallQual']
    df['Year_Cond'] = df['YearBuilt'] * df['OverallCond']

    # Polynomials
    df['TotalSF_Sq'] = df['TotalSF'] ** 2
    df['GrLivArea_Sq'] = df['GrLivArea'] ** 2

    # Binary Flags
    df['HasBsmt'] = (df['TotalBsmtSF'] > 0).astype(int)
    df['HasGarage'] = (df['GarageArea'] > 0).astype(int)

    # Box-Cox Transformation
    lam = artifacts.get('skew_lambda', 0.15)
    skewed_features = artifacts.get('skewed_features', [])

    for feat in skewed_features:
        if feat in df.columns:
            # Add 1 to handle zeros before boxcox if logic requires,
            # though boxcox1p handles log(1+x), here we just apply standardized
            # lambda
            df[feat] = boxcox1p(df[feat], lam)

    # Alignment (Reindex to ensure exact column order and count as X_train)
    # This fills any missing One-Hot columns with 0
    final_df = df.reindex(columns=artifacts['features'], fill_value=0)

    return final_df


# UI Layout - Inputs
st.title("🏡 Ames House Price Prediction")
st.markdown(
    "Enter property details below to estimate market value using Advanced Regression techniques.")

# Create tabs for better organization
tab1, tab2, tab3 = st.tabs(
    ["📏 Size & Quality", "📍 Location & Type", "🚗 Garage & Basement"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        gr_liv_area = st.number_input(
            "Living Area (Above Grade) sqft", 500, 6000, 1500, step=50)
        overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 5,
                                 help="Rates the overall material and finish of the house")
    with col2:
        year_built = st.number_input("Year Built", 1870, 2025, 2005)
        overall_cond = st.slider("Overall Condition (1-9)", 1,
                                 9, 5, help="Rates the overall condition of the house")

with tab2:
    col3, col4 = st.columns(2)
    with col3:
        # Neighborhoods from the dataset
        neighborhoods = ['CollgCr', 'Veenker', 'Crawfor', 'NoRidge', 'Mitchel', 'Somerst',
                         'NWAmes', 'OldTown', 'BrkSide', 'Sawyer', 'NridgHt', 'NAmes',
                         'SawyerW', 'IDOTRR', 'MeadowV', 'Edwards', 'Timber', 'Gilbert',
                         'StoneBr', 'ClearCr', 'NPkVill', 'Blmngtn', 'BrDale', 'SWISU',
                         'Blueste']
        neighborhood = st.selectbox("Neighborhood", neighborhoods, index=0)

        bldg_type = st.selectbox(
            "Building Type", ["1Fam", "2fmCon", "Duplex", "TwnhsE", "Twnhs"])

    with col4:
        house_style = st.selectbox(
            "House Style", ["1Story", "2Story", "1.5Fin", "SLvl", "SFoyer"])
        lot_area = st.number_input(
            "Lot Area (sqft)", 1000, 100000, 8000, step=100)

with tab3:
    col5, col6 = st.columns(2)
    with col5:
        garage_cars = st.slider("Garage Capacity (Cars)", 0, 4, 2)
        garage_area = st.number_input("Garage Area (sqft)", 0, 1500, 500)
    with col6:
        total_bsmt = st.number_input(
            "Total Basement Area (sqft)", 0, 4000, 800)
        bathrooms = st.number_input("Total Bathrooms", 1, 5, 2)

# 6. Prediction Logic
st.markdown("---")
predict_btn = st.button(
    "Predict Price", type="primary", use_container_width=True)

if predict_btn:
    with st.spinner("Processing data and running inference..."):
        time.sleep(0.5)  # UX Delay

        # 1. Compile User Input
        user_input = {
            'GrLivArea': gr_liv_area,
            'OverallQual': overall_qual,
            'OverallCond': overall_cond,
            'YearBuilt': year_built,
            'TotalBsmtSF': total_bsmt,
            'GarageCars': garage_cars,
            'GarageArea': garage_area,
            'LotArea': lot_area,
            'FullBath': bathrooms,
            # Categoricals passed as strings for mapping in preprocess_input
            'Neighborhood': neighborhood,
            'BldgType': bldg_type,
            'HouseStyle': house_style,
            # Defaults for things not asked (to be safe)
            'MoSold': 6,
            'YrSold': 2010
        }

        try:
            # 2. Preprocess
            X_infer = preprocess_input(user_input, artifacts)

            # 3. Model Inference
            if selected_model_name == "Geometric_Blend":
                # Load component models
                stack = load_model("Stacking_Ensemble.pkl")
                cat = load_model("CatBoost.pkl")
                lasso = load_model("Lasso.pkl")

                if stack and cat and lasso:
                    # Get Log Predictions
                    p1 = stack.predict(X_infer)[0]
                    p2 = cat.predict(X_infer)[0]
                    p3 = lasso.predict(X_infer)[0]

                    # Weights: 70% Stacking, 15% CatBoost, 15% Lasso
                    log_blend = (0.70 * p1) + (0.15 * p2) + (0.15 * p3)
                    final_price = np.expm1(log_blend)

                    # Store components for visualization
                    components = {
                        "Stacking": np.expm1(p1),
                        "CatBoost": np.expm1(p2),
                        "Lasso": np.expm1(p3)
                    }

            else:
                # Single Model Logic
                model = load_model(f"{selected_model_name}.pkl")
                if model:
                    log_pred = model.predict(X_infer)[0]
                    final_price = np.expm1(log_pred)
                    components = None

            # 4. Display Results
            st.markdown(f"""
            <div class="metric-card">
                <h2 style='margin-bottom:0px;'>Estimated Value</h2>
                <h1 style='color: #4CAF50; font-size: 3.5em;'>${final_price:,.0f}</h1>
            </div>
            """, unsafe_allow_html=True)

            # 5. Advanced Visualization (For Ensemble)
            if components:
                st.subheader("📊 Ensemble Contribution Analysis")

                # Prepare data for chart
                comp_df = pd.DataFrame({
                    "Model": list(components.keys()),
                    "Prediction": list(components.values())
                })

                # Use columns for layout
                c1, c2 = st.columns([2, 1])

                with c1:
                    # Bar Chart
                    fig, ax = plt.subplots(figsize=(6, 3))
                    sns.barplot(data=comp_df, x="Prediction", y="Model",
                                hue="Model", palette="viridis", legend=False, ax=ax)
                    ax.axvline(final_price, color='red',
                               linestyle='--', label='Weighted Mean')
                    ax.set_xlabel("Predicted Price ($)")
                    ax.set_title("Component Model Variance")
                    st.pyplot(fig)

                with c2:
                    st.markdown("**Blend Weights:**")
                    st.caption("- Stacking Regressor: **70%**")
                    st.caption("- CatBoost: **15%**")
                    st.caption("- Lasso (Regularized): **15%**")

                    spread = max(components.values()) - \
                        min(components.values())
                    st.info(f"Model Divergence: ${spread:,.0f}")

        except Exception as e:
            st.error(f"Prediction Error: {e}")
            st.code("Check saved_models/artifacts.pkl features vs input.")

# Footer
st.markdown("---")
st.caption("Developed for IS403.Q13 - Business Data Analysis | University of Information Technology, VNU-HCM")
