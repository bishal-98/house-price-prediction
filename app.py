

import streamlit as st
import pandas as pd
import joblib


# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)


# --------------------------------------------------
# 2. Load the saved model and feature columns
# --------------------------------------------------

model = joblib.load("house_price_model.pkl")
model_columns = joblib.load("house_price_columns.pkl")


# --------------------------------------------------
# 3. Title
# --------------------------------------------------

st.title("🏠 House Price Prediction")
st.write(
    "Enter the house details below to estimate its selling price."
)

st.divider()


# --------------------------------------------------
# 4. User inputs
# --------------------------------------------------

st.header("House Information")


col1, col2, col3 = st.columns(3)


with col1:
    overall_qual = st.slider(
        "Overall Quality",
        min_value=1,
        max_value=10,
        value=5
    )

    year_built = st.number_input(
        "Year Built",
        min_value=1800,
        max_value=2026,
        value=2000
    )

    year_remod = st.number_input(
        "Year Remodeled",
        min_value=1800,
        max_value=2026,
        value=2000
    )


with col2:
    gr_liv_area = st.number_input(
        "Living Area (sq ft)",
        min_value=100,
        max_value=10000,
        value=1500
    )

    total_bsmt_sf = st.number_input(
        "Basement Area (sq ft)",
        min_value=0,
        max_value=5000,
        value=500
    )

    garage_cars = st.number_input(
        "Garage Capacity (cars)",
        min_value=0,
        max_value=5,
        value=2
    )


with col3:
    garage_area = st.number_input(
        "Garage Area (sq ft)",
        min_value=0,
        max_value=2000,
        value=400
    )

    full_bath = st.number_input(
        "Full Bathrooms",
        min_value=0,
        max_value=5,
        value=2
    )

    bedroom_abv_gr = st.number_input(
        "Bedrooms Above Ground",
        min_value=0,
        max_value=10,
        value=3
    )


# --------------------------------------------------
# 5. Categorical inputs
# --------------------------------------------------

st.header("House Quality & Location")


col1, col2, col3 = st.columns(3)


with col1:
    neighborhood = st.selectbox(
        "Neighborhood",
        [
            "NAmes",
            "CollgCr",
            "OldTown",
            "Edwards",
            "Somerst",
            "Gilbert",
            "NridgHt",
            "Sawyer",
            "NWAmes",
            "SawyerW",
            "BrkSide",
            "Crawfor",
            "Mitchel",
            "NoRidge",
            "Timber",
            "IDOTRR",
            "ClearCr",
            "StoneBr",
            "SWISU",
            "MeadowV",
            "Blmngtn",
            "BrDale",
            "Veenker",
            "NPkVill",
            "Blueste"
        ]
    )

    kitchen_qual = st.selectbox(
        "Kitchen Quality",
        ["Ex", "Gd", "TA", "Fa", "Po"]
    )


with col2:
    exter_qual = st.selectbox(
        "Exterior Quality",
        ["Ex", "Gd", "TA", "Fa", "Po"]
    )

    bsmt_qual = st.selectbox(
        "Basement Quality",
        ["None", "Ex", "Gd", "TA", "Fa", "Po"]
    )


with col3:
    garage_finish = st.selectbox(
        "Garage Finish",
        ["None", "Fin", "RFn", "Unf"]
    )

    heating_qc = st.selectbox(
        "Heating Quality",
        ["Ex", "Gd", "TA", "Fa", "Po"]
    )


st.divider()


# --------------------------------------------------
# 6. Prediction button
# --------------------------------------------------

if st.button("Predict House Price", type="primary"):

    # Start with all model features as zero
    input_data = pd.DataFrame(
        0,
        index=[0],
        columns=model_columns
    )


    # ----------------------------------------------
    # Numerical features
    # ----------------------------------------------

    numerical_values = {
        "OverallQual": overall_qual,
        "YearBuilt": year_built,
        "YearRemodAdd": year_remod,
        "GrLivArea": gr_liv_area,
        "TotalBsmtSF": total_bsmt_sf,
        "GarageCars": garage_cars,
        "GarageArea": garage_area,
        "FullBath": full_bath,
        "BedroomAbvGr": bedroom_abv_gr
    }


    for column, value in numerical_values.items():

        if column in input_data.columns:
            input_data[column] = value


    # ----------------------------------------------
    # Categorical features
    # ----------------------------------------------
    # pd.get_dummies() originally created columns such as:
    #
    # Neighborhood_NridgHt
    # Neighborhood_OldTown
    # KitchenQual_Gd
    #
    # We manually activate the corresponding column.
    # ----------------------------------------------

    categorical_values = {
        "Neighborhood": neighborhood,
        "KitchenQual": kitchen_qual,
        "ExterQual": exter_qual,
        "BsmtQual": bsmt_qual,
        "GarageFinish": garage_finish,
        "HeatingQC": heating_qc
    }


    for feature, value in categorical_values.items():

        dummy_column = f"{feature}_{value}"

        if dummy_column in input_data.columns:
            input_data[dummy_column] = 1


    # ----------------------------------------------
    # Make prediction
    # ----------------------------------------------

    prediction = model.predict(input_data)[0]


    # ----------------------------------------------
    # Display result
    # ----------------------------------------------

    st.success("Prediction completed!")

    st.metric(
        label="Estimated House Price",
        value=f"${prediction:,.0f}"
    )

    st.info(
        "This is a machine-learning estimate, not an official market valuation."
    )

