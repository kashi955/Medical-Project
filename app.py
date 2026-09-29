import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Medical Data Analysis",
    page_icon="❤️",
    layout="wide"
)


# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------

st.title("❤️ Medical Data Analysis")

st.write(
    "Machine Learning analysis of cardiovascular health data "
    "using Python and Scikit-learn."
)


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

@st.cache_data
def load_data():

    file_names = [
        "framingham.csv",
        "framingham(4).csv",
        "framingham(3).csv",
        "framingham(2).csv",
        "framingham(1).csv"
    ]

    df = None

    for file_name in file_names:

        try:
            df = pd.read_csv(file_name)
            break

        except FileNotFoundError:
            continue

    if df is None:

        st.error(
            "Framingham CSV file was not found. "
            "Please keep the CSV file in the same folder as app.py."
        )

        st.stop()

    df.columns = df.columns.str.strip()

    numeric_cols = df.select_dtypes(
        include=["int64", "float64"]
    ).columns

    df[numeric_cols] = df[numeric_cols].fillna(
        df[numeric_cols].median()
    )

    df = df.drop_duplicates()

    return df


df = load_data()


# ---------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------

st.sidebar.title("Navigation")

section = st.sidebar.radio(
    "Select Section",
    [
        "Dataset Overview",
        "Data Analysis",
        "Logistic Regression",
        "PCA Analysis",
        "Live Prediction"
    ]
)


# =========================================================
# DATASET OVERVIEW
# =========================================================

if section == "Dataset Overview":

    st.header("📁 Dataset Overview")

    # -----------------------------------------------------
    # 1. BASIC DATASET INFORMATION
    # -----------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Total Records",
            df.shape[0]
        )

    with col2:

        st.metric(
            "Total Features",
            df.shape[1]
        )


    # -----------------------------------------------------
    # 2. DATASET PREVIEW
    # -----------------------------------------------------

    st.subheader("📋 Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )


    # -----------------------------------------------------
    # 3. DATASET INFORMATION
    # -----------------------------------------------------

    st.subheader("📌 Dataset Information")

    st.write("### Columns")

    st.write(
        list(df.columns)
    )


    # -----------------------------------------------------
    # 4. DATA TYPES
    # -----------------------------------------------------

    st.write("### Data Types")

    datatype_df = pd.DataFrame(
        df.dtypes,
        columns=["Data Type"]
    )

    st.dataframe(
        datatype_df,
        use_container_width=True
    )


    # -----------------------------------------------------
    # 5. MISSING VALUES
    # -----------------------------------------------------

    st.subheader("⚠️ Missing Values")

    missing_values = pd.DataFrame({
        "Column": df.columns,
        "Missing Values": df.isnull().sum()
    })

    st.dataframe(
        missing_values,
        use_container_width=True
    )


    # -----------------------------------------------------
    # 6. STATISTICAL SUMMARY
    # -----------------------------------------------------

    st.subheader("📊 Statistical Summary")

    statistical_summary = df.describe().T

    st.dataframe(
        statistical_summary,
        use_container_width=True
    )


   


    # -----------------------------------------------------
    # 8. DUPLICATE RECORDS
    # -----------------------------------------------------

    st.subheader("🔁 Duplicate Records")

    duplicate_count = df.duplicated().sum()

    st.metric(
        "Number of Duplicate Rows",
        int(duplicate_count)
    )

    if duplicate_count == 0:

        st.success(
            "No duplicate records found."
        )

    else:

        st.warning(
            f"{duplicate_count} duplicate records found."
        )


# =========================================================
# DATA ANALYSIS
# =========================================================

elif section == "Data Analysis":

    st.header("📊 Data Analysis")


    # -----------------------------------------------------
    # AGE DISTRIBUTION
    # -----------------------------------------------------

    st.subheader("Age Distribution")

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.hist(
        df["age"],
        bins=30
    )

    ax.set_xlabel("Age")
    ax.set_ylabel("Count")
    ax.set_title("Age Distribution of Participants")

    st.pyplot(fig)

    plt.close(fig)


    # -----------------------------------------------------
    # AGE BOX PLOT
    # -----------------------------------------------------

    st.subheader("Age Box Plot")

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    ax.boxplot(
        df["age"].dropna()
    )

    ax.set_ylabel("Age")
    ax.set_title("Age Box Plot")

    st.pyplot(fig)

    plt.close(fig)


    # -----------------------------------------------------
    # GENDER DISTRIBUTION
    # -----------------------------------------------------

    st.subheader("Gender Distribution")

    gender_counts = df["male"].value_counts().sort_index()

    gender_labels = []

    for value in gender_counts.index:

        if value == 0:
            gender_labels.append("Female")
        else:
            gender_labels.append("Male")

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    ax.pie(
        gender_counts.values,
        labels=gender_labels,
        autopct="%1.1f%%",
        startangle=90
    )

    ax.set_title("Gender Distribution")

    st.pyplot(fig)

    plt.close(fig)


    # -----------------------------------------------------
    # EDUCATION DISTRIBUTION
    # -----------------------------------------------------

    st.subheader("Education Distribution")

    education_counts = (
        df["education"]
        .value_counts()
        .sort_index()
    )

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    ax.pie(
        education_counts.values,
        labels=education_counts.index,
        autopct="%1.1f%%",
        startangle=90
    )

    ax.set_title("Education Distribution")

    st.pyplot(fig)

    plt.close(fig)


    # -----------------------------------------------------
    # TEN YEAR CHD DISTRIBUTION - BAR CHART
    # -----------------------------------------------------

    st.subheader("Ten-Year CHD Distribution")

    chd_counts = df["TenYearCHD"].value_counts().sort_index()

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    ax.bar(
        chd_counts.index.astype(str),
        chd_counts.values
    )

    ax.set_xlabel("TenYearCHD")
    ax.set_ylabel("Count")
    ax.set_title("Ten-Year CHD Distribution")

    st.pyplot(fig)

    plt.close(fig)


    # -----------------------------------------------------
    # TEN YEAR CHD DISTRIBUTION - PIE CHART
    # -----------------------------------------------------

    st.subheader(
        "Ten-Year CHD Distribution (Pie Chart)"
    )

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    ax.pie(
        chd_counts.values,
        labels=chd_counts.index.astype(str),
        autopct="%1.1f%%",
        startangle=90
    )

    ax.set_title(
        "Ten-Year CHD Distribution (Pie Chart)"
    )

    st.pyplot(fig)

    plt.close(fig)


    # -----------------------------------------------------
    # CURRENT SMOKER DISTRIBUTION
    # -----------------------------------------------------

    st.subheader("Current Smoker Distribution")

    smoker_counts = (
        df["currentSmoker"]
        .value_counts()
        .sort_index()
    )

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    ax.pie(
        smoker_counts.values,
        labels=["Non-Smoker", "Smoker"],
        autopct="%1.1f%%",
        startangle=90
    )

    ax.set_title(
        "Current Smoker Distribution"
    )

    st.pyplot(fig)

    plt.close(fig)


    # -----------------------------------------------------
    # BPMEDS DISTRIBUTION
    # -----------------------------------------------------

    st.subheader("BPMeds Distribution")

    bpmeds_counts = (
        df["BPMeds"]
        .value_counts()
        .sort_index()
    )

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    labels = [
        "No BPMeds" if value == 0 else "Taking BPMeds"
        for value in bpmeds_counts.index
    ]

    ax.pie(
        bpmeds_counts.values,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90
    )

    ax.set_title(
        "BPMeds Distribution"
    )

    st.pyplot(fig)

    plt.close(fig)


    # -----------------------------------------------------
    # PREVALENT STROKE DISTRIBUTION
    # -----------------------------------------------------

    st.subheader(
        "Prevalent Stroke Distribution"
    )

    stroke_counts = (
        df["prevalentStroke"]
        .value_counts()
        .sort_index()
    )

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    labels = [
        "No Stroke" if value == 0 else "Stroke"
        for value in stroke_counts.index
    ]

    ax.pie(
        stroke_counts.values,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90
    )

    ax.set_title(
        "Prevalent Stroke Distribution"
    )

    st.pyplot(fig)

    plt.close(fig)


    # -----------------------------------------------------
    # PREVALENT HYPERTENSION DISTRIBUTION
    # -----------------------------------------------------

    st.subheader(
        "Prevalent Hypertension Distribution"
    )

    hypertension_counts = (
        df["prevalentHyp"]
        .value_counts()
        .sort_index()
    )

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    labels = [
        "No Hypertension" if value == 0
        else "Hypertension"
        for value in hypertension_counts.index
    ]

    ax.pie(
        hypertension_counts.values,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90
    )

    ax.set_title(
        "Prevalent Hypertension Distribution"
    )

    st.pyplot(fig)

    plt.close(fig)


    # -----------------------------------------------------
    # DIABETES DISTRIBUTION
    # -----------------------------------------------------

    st.subheader("Diabetes Distribution")

    diabetes_counts = (
        df["diabetes"]
        .value_counts()
        .sort_index()
    )

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    labels = [
        "No Diabetes" if value == 0
        else "Diabetes"
        for value in diabetes_counts.index
    ]

    ax.pie(
        diabetes_counts.values,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90
    )

    ax.set_title(
        "Diabetes Distribution"
    )

    st.pyplot(fig)

    plt.close(fig)


    # -----------------------------------------------------
    # CORRELATION HEATMAP
    # -----------------------------------------------------

    st.subheader("Correlation Heatmap")

    numeric_df = df.select_dtypes(
        include=np.number
    )

    corr = numeric_df.corr()

    fig, ax = plt.subplots(
        figsize=(12, 8)
    )

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        ax=ax
    )

    ax.set_title(
        "Correlation Heatmap"
    )

    st.pyplot(fig)

    plt.close(fig)


# =========================================================
# LOGISTIC REGRESSION
# =========================================================

elif section == "Logistic Regression":

    st.header("🤖 Logistic Regression")

    st.write(
        "Logistic Regression is used to predict "
        "the Ten-Year CHD outcome."
    )


    # -----------------------------------------------------
    # FEATURES
    # -----------------------------------------------------

    features = [
        "age",
        "male",
        "education",
        "cigsPerDay",
        "totChol",
        "sysBP",
        "glucose"
    ]

    X = df[features]

    y = df["TenYearCHD"]


    # -----------------------------------------------------
    # TRAIN TEST SPLIT
    # -----------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


    # -----------------------------------------------------
    # MODEL
    # -----------------------------------------------------

    model = LogisticRegression(
        max_iter=1000
    )

    model.fit(
        X_train,
        y_train
    )


    # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------

    y_pred = model.predict(
        X_test
    )


    # -----------------------------------------------------
    # ACCURACY
    # -----------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    st.metric(
        "Accuracy",
        f"{accuracy * 100:.2f}%"
    )


    # -----------------------------------------------------
    # CONFUSION MATRIX
    # -----------------------------------------------------

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    st.pyplot(fig)

    plt.close(fig)


    # -----------------------------------------------------
    # CLASSIFICATION REPORT
    # -----------------------------------------------------

    st.subheader("Classification Report")

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True
    )

    st.dataframe(
        pd.DataFrame(report).transpose(),
        use_container_width=True
    )


# =========================================================
# PCA ANALYSIS
# =========================================================

# ---------------------------------------------------------
# PCA ANALYSIS
# ---------------------------------------------------------

elif section == "PCA Analysis":

    st.header("🔍 Principal Component Analysis (PCA)")

    st.write(
        "PCA is used to reduce the dimensionality of the medical dataset "
        "while preserving important information."
    )

    # -----------------------------------------------------
    # FEATURES USED FOR PCA
    # -----------------------------------------------------

    pca_features = [
        "male",
        "age",
        "education",
        "cigsPerDay",
        "totChol",
        "sysBP",
        "diaBP",
        "BMI",
        "heartRate",
        "glucose"
    ]

    X_pca_data = df[pca_features].copy()

    # -----------------------------------------------------
    # HANDLE MISSING VALUES
    # -----------------------------------------------------

    X_pca_data = X_pca_data.fillna(
        X_pca_data.median()
    )

    # -----------------------------------------------------
    # STANDARDIZATION
    # -----------------------------------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X_pca_data
    )

    # -----------------------------------------------------
    # BEFORE PCA
    # -----------------------------------------------------

    st.subheader("📊 Before PCA")

    st.write(
        "Before PCA, the first two standardized original features "
        "are shown in the original feature space."
    )

    fig_before, ax_before = plt.subplots(
        figsize=(8, 5)
    )

    scatter_before = ax_before.scatter(
        X_scaled[:, 0],
        X_scaled[:, 1],
        c=df["TenYearCHD"],
        cmap="viridis",
        s=20,
        alpha=0.7
    )

    ax_before.set_xlabel(
        "Original Feature 1"
    )

    ax_before.set_ylabel(
        "Original Feature 2"
    )

    ax_before.set_title(
        "Before PCA - Original Feature Space"
    )

    fig_before.colorbar(
        scatter_before,
        ax=ax_before,
        label="TenYearCHD"
    )

    st.pyplot(fig_before)

    plt.close(fig_before)

    # -----------------------------------------------------
    # PCA TRANSFORMATION
    # -----------------------------------------------------

    pca = PCA(
        n_components=2
    )

    X_pca = pca.fit_transform(
        X_scaled
    )

    # -----------------------------------------------------
    # EXPLAINED VARIANCE
    # -----------------------------------------------------

    explained_variance = (
        pca.explained_variance_ratio_
    )

    # -----------------------------------------------------
    # AFTER PCA
    # -----------------------------------------------------

    st.subheader("📉 After PCA")

    st.write(
        "After PCA, the original features are transformed "
        "into two principal components."
    )

    fig_after, ax_after = plt.subplots(
        figsize=(8, 5)
    )

    scatter_after = ax_after.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=df["TenYearCHD"],
        cmap="viridis",
        s=20,
        alpha=0.7
    )

    ax_after.set_xlabel(
        "Principal Component 1"
    )

    ax_after.set_ylabel(
        "Principal Component 2"
    )

    ax_after.set_title(
        "After PCA - Two Principal Components"
    )

    fig_after.colorbar(
        scatter_after,
        ax=ax_after,
        label="TenYearCHD"
    )

    st.pyplot(fig_after)

    plt.close(fig_after)

    # -----------------------------------------------------
    # EXPLAINED VARIANCE
    # -----------------------------------------------------

    st.subheader("📈 Explained Variance")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "PC1 Variance",
            f"{explained_variance[0] * 100:.2f}%"
        )

    with col2:
        st.metric(
            "PC2 Variance",
            f"{explained_variance[1] * 100:.2f}%"
        )

    with col3:
        st.metric(
            "Combined Variance",
            f"{explained_variance.sum() * 100:.2f}%"
        )

    # -----------------------------------------------------
    # PCA SUMMARY
    # -----------------------------------------------------

    st.subheader("📌 PCA Summary")

    st.write(
        f"Original Features: {len(pca_features)}"
    )

    st.write(
        "Reduced Features: 2 Principal Components"
    )

    st.write(
        f"Total Variance Explained: "
        f"{explained_variance.sum() * 100:.2f}%"
    )

# =========================================================
# LIVE PREDICTION
# =========================================================

elif section == "Live Prediction":

    st.header("🔮 Live Medical Prediction")

    st.warning(
        "This prediction is for educational/project purposes "
        "only and should not be used as a medical diagnosis."
    )


    # -----------------------------------------------------
    # TRAIN MODEL
    # -----------------------------------------------------

    features = [
        "age",
        "male",
        "education",
        "cigsPerDay",
        "totChol",
        "sysBP",
        "glucose"
    ]

    X = df[features]

    y = df["TenYearCHD"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = LogisticRegression(
        max_iter=1000
    )

    model.fit(
        X_train,
        y_train
    )


    # -----------------------------------------------------
    # USER INPUT
    # -----------------------------------------------------

    st.subheader("Enter Patient Details")

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Age",
            min_value=32,
            max_value=70,
            value=50
        )

        male = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )

        education = st.selectbox(
            "Education Level",
            [1, 2, 3, 4]
        )

        cigsPerDay = st.number_input(
            "Cigarettes Per Day",
            min_value=0.0,
            max_value=70.0,
            value=0.0
        )

    with col2:

        totChol = st.number_input(
            "Total Cholesterol",
            min_value=100.0,
            max_value=700.0,
            value=234.0
        )

        sysBP = st.number_input(
            "Systolic Blood Pressure",
            min_value=80.0,
            max_value=300.0,
            value=128.0
        )

        glucose = st.number_input(
            "Glucose",
            min_value=40.0,
            max_value=400.0,
            value=78.0
        )


    male_value = (
        1 if male == "Male" else 0
    )


    # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------

    if st.button("Predict"):

        input_data = pd.DataFrame(
            [[
                age,
                male_value,
                education,
                cigsPerDay,
                totChol,
                sysBP,
                glucose
            ]],
            columns=features
        )

        prediction = model.predict(
            input_data
        )[0]

        probability = model.predict_proba(
            input_data
        )[0]

        confidence = (
            probability[prediction] * 100
        )

        if prediction == 1:

            st.error(
                "Prediction: Higher Risk Class "
                "(TenYearCHD = 1)"
            )

        else:

            st.success(
                "Prediction: Lower Risk Class "
                "(TenYearCHD = 0)"
            )

        st.metric(
            "Prediction Confidence",
            f"{confidence:.2f}%"
        )

        st.info(
            "The confidence shown is the probability "
            "assigned by the Logistic Regression model "
            "to the predicted class."
        )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.sidebar.markdown("---")

st.sidebar.write(
    "Medical Data Analysis Project"
)

st.sidebar.write(
    "Built with Python, Pandas, "
    "Scikit-learn and Streamlit"
)