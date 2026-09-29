\# ❤️ Medical Data Analysis \& Ten-Year CHD Prediction



\## 📌 Project Overview



This project is a Medical Data Analysis and Machine Learning application built using Python and Streamlit.



The project analyzes cardiovascular health data and provides interactive data visualization, Logistic Regression analysis, Principal Component Analysis (PCA), and live Ten-Year Coronary Heart Disease (CHD) prediction.



\---



\## 🎯 Project Objectives



\- Analyze cardiovascular health data.

\- Understand important patient-related features.

\- Visualize distributions and relationships between variables.

\- Analyze the Ten-Year CHD target variable.

\- Build a Logistic Regression classification model.

\- Evaluate model performance using accuracy, confusion matrix, and classification report.

\- Apply PCA for dimensionality reduction and visualization.

\- Provide an interactive Live Prediction interface using Streamlit.



\---



\## 🛠️ Technologies Used



\- Python

\- Pandas

\- NumPy

\- Matplotlib

\- Seaborn

\- Scikit-learn

\- Streamlit



\---



\## 📊 Dataset



The project uses a cardiovascular health dataset containing \*\*4,238 records and 16 features\*\*.



\### Main Features



\- `male`

\- `age`

\- `education`

\- `currentSmoker`

\- `cigsPerDay`

\- `BPMeds`

\- `prevalentStroke`

\- `prevalentHyp`

\- `diabetes`

\- `totChol`

\- `sysBP`

\- `diaBP`

\- `BMI`

\- `heartRate`

\- `glucose`

\- `TenYearCHD`



\### Target Variable



`TenYearCHD`



\- `0` — No Ten-Year CHD outcome

\- `1` — Ten-Year CHD outcome



\---



\## 📁 Project Features



\### 1. Dataset Overview



The Dataset Overview section provides:



\- Total number of records

\- Total number of features

\- Dataset preview

\- Column information

\- Data types

\- Missing-value information

\- Statistical summary

\- Target distribution

\- Duplicate-record information



\---



\### 2. Data Analysis



The Data Analysis section contains multiple visualizations, including:



\- Age Distribution

\- Age Box Plot

\- Gender Distribution

\- Education Distribution

\- Ten-Year CHD Distribution

\- Current Smoker Distribution

\- BPMeds Distribution

\- Prevalent Stroke Distribution

\- Prevalent Hypertension Distribution

\- Diabetes Distribution

\- Correlation Heatmap



These visualizations help understand the distribution and relationships between different medical features.



\---



\### 3. Logistic Regression



A Logistic Regression classification model is used to predict the `TenYearCHD` target.



The model is evaluated using:



\- Accuracy

\- Confusion Matrix

\- Classification Report

\- Precision

\- Recall

\- F1-score



\### Model Result



\*\*Accuracy: 84.55%\*\*



\---



\### 4. PCA Analysis



Principal Component Analysis (PCA) is used for dimensionality reduction and visualization.



The PCA analysis includes:



\- Before PCA visualization

\- Standardization of features

\- After PCA visualization

\- Explained variance

\- PCA summary



\### PCA Result



\- Original Features: \*\*10\*\*

\- Reduced Features: \*\*2 Principal Components\*\*

\- PC1 Variance: \*\*24.35%\*\*

\- PC2 Variance: \*\*13.54%\*\*

\- Combined Variance: \*\*37.89%\*\*



\---



\### 5. Live Medical Prediction



The Streamlit application provides an interactive Live Prediction section.



Users can enter patient information and generate a model prediction for the `TenYearCHD` target.



The prediction interface includes patient-related inputs such as:



\- Age

\- Gender

\- Education

\- Cigarettes Per Day

\- Total Cholesterol

\- Systolic Blood Pressure

\- Glucose



The prediction result and model confidence are displayed directly in the Streamlit application.



> \*\*Disclaimer:\*\* This project is created for educational and project purposes only and should not be used as a medical diagnosis or substitute for professional medical advice.



\---



\## 📁 Project Structure



```text

Medical Data Analysis/

│

├── app.py

├── framingham.ipynb

├── framingham.csv

├── requirements.txt

└── README.md



\---



## 🚀 How to Run the Project



### 1. Clone the Repository



```bash

git clone YOUR\_GITHUB\_REPOSITORY\_URL

cd Medical-Data-Analysis



### 2. Install Required Libraries



```bash

pip install -r requirements.txt



### 3. Run the Streamlit Application



```bash

streamlit run app.py


**Live App:** https://medical-project-nhjvst3uwmv8prygcbmek4.streamlit.app/

