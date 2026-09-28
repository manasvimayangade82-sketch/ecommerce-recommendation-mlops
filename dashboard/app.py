import streamlit as st
import pandas as pd
import requests
import json

st.set_page_config(
    page_title="E-Commerce Recommendation MLOps",
    layout="wide"
)

API_URL = "http://127.0.0.1:8000"

DATA_FILE = "data/processed/clean_transactions.csv"
EVAL_FILE = "data/processed/model_evaluation.csv"
PRODUCT_FILE = "data/processed/product_features.csv"

@st.cache_data
def load_data():
    try:
        return pd.read_csv(DATA_FILE)
    except:
        return pd.DataFrame()

@st.cache_data
def load_products():
    try:
        return pd.read_csv(PRODUCT_FILE)
    except:
        return pd.DataFrame()

@st.cache_data
def load_evaluation():
    try:
        return pd.read_csv(EVAL_FILE)
    except:
        return pd.DataFrame()

df = load_data()
products = load_products()
evaluation = load_evaluation()

st.title("E-Commerce Recommendation MLOps")
st.write("ML-powered personalized product recommendation system")

try:
    health = requests.get(
        f"{API_URL}/health",
        timeout=3
    ).json()

    api_status = "Online"
    model_name = health.get(
        "model_name",
        "Unknown"
    )

except:
    api_status = "Offline"
    model_name = "Unavailable"

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Dashboard",
        "Recommendation Explorer",
        "Model Comparison",
        "MLflow Experiments",
        "System Health",
        "Data Explorer"
    ]
)

if page == "Dashboard":

    st.subheader("System Overview")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Transactions",
            f"{len(df):,}"
        )

    with c2:
        if "CustomerID" in df.columns:
            st.metric(
                "Customers",
                f"{df['CustomerID'].nunique():,}"
            )

    with c3:
        if "StockCode" in df.columns:
            st.metric(
                "Products",
                f"{df['StockCode'].nunique():,}"
            )

    with c4:
        st.metric(
            "API Status",
            api_status
        )

    st.divider()

    st.subheader("Production Recommendation Model")

    st.info(
        f"Selected Model: {model_name}"
    )

    if not df.empty and "Quantity" in df.columns:
        st.subheader("Purchase Quantity Distribution")

        quantity_data = (
            df.groupby("StockCode")["Quantity"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )

        st.bar_chart(quantity_data)


elif page == "Recommendation Explorer":

    st.subheader("Customer Recommendation Explorer")

    customer_id = st.text_input(
        "Enter Customer ID"
    )

    n = st.slider(
        "Number of Recommendations",
        1,
        20,
        10
    )

    if st.button("Generate Recommendations"):

        if not customer_id:
            st.warning("Enter a Customer ID.")
        else:

            try:

                response = requests.get(
                    f"{API_URL}/recommend/{customer_id}",
                    params={"n": n},
                    timeout=10
                )

                if response.status_code == 200:

                    result = response.json()

                    st.success(
                        f"{result['recommendation_count']} recommendations generated."
                    )

                    st.dataframe(
                        pd.DataFrame(
                            result["recommendations"]
                        ),
                        use_container_width=True
                    )

                else:

                    st.error(
                        response.json().get(
                            "detail",
                            "Recommendation failed"
                        )
                    )

            except Exception as e:

                st.error(
                    f"API connection failed: {e}"
                )


elif page == "Model Comparison":

    st.subheader("Recommendation Model Comparison")

    if evaluation.empty:

        st.warning(
            "Model evaluation data not available."
        )

    else:

        st.dataframe(
            evaluation,
            use_container_width=True
        )

        numeric_columns = evaluation.select_dtypes(
            include="number"
        ).columns

        if len(numeric_columns) > 0:

            metric = st.selectbox(
                "Select Metric",
                list(numeric_columns)
            )

            st.bar_chart(
                evaluation.set_index(
                    evaluation.columns[0]
                )[metric]
            )


elif page == "MLflow Experiments":

    st.subheader(
        "MLflow Experiment Tracking"
    )

    st.write(
        "Experiment: E-Commerce Recommendation System"
    )

    if not evaluation.empty:

        st.dataframe(
            evaluation,
            use_container_width=True
        )

    else:

        st.info(
            "Run MLflow experiments to display results."
        )


elif page == "System Health":

    st.subheader("System Health")

    if api_status == "Online":

        st.success(
            "Recommendation API is running."
        )

        st.json(health)

    else:

        st.error(
            "Recommendation API is offline."
        )

    st.write(
        "Production Model:",
        model_name
    )

    st.write(
        "Transactions Loaded:",
        len(df)
    )

    st.write(
        "Products Loaded:",
        len(products)
    )


elif page == "Data Explorer":

    st.subheader("Transaction Data")

    if df.empty:

        st.warning(
            "Transaction data not found."
        )

    else:

        st.dataframe(
            df.head(500),
            use_container_width=True
        )

        st.write(
            "Rows:",
            len(df)
        )

        st.write(
            "Columns:",
            len(df.columns)
        )