import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Page Configuration
st.set_page_config(
    page_title="AI/ML Task - Streamlit Dashboard",
    page_icon="🚀",
    layout="wide"
)

st.title("📊 AI/ML Internship: Data Analysis & Prediction Dashboard")
st.write("This interactive Streamlit app covers the data exploration, visualization, and bonus machine learning prediction tasks.")

# Load Dataset (Using Titanic dataset as the standard assignment reference)
@st.cache_data
def load_data():
    df = sns.load_dataset('titanic')
    # Basic data cleaning / handling missing values
    df['age'] = df['age'].fillna(df['age'].median())
    df['embark_town'] = df['embark_town'].fillna(df['embark_town'].mode()[0])
    return df

df = load_data()

# Sidebar Navigation
st.sidebar.header("Navigation Menu")
app_mode = st.sidebar.selectbox(
    "Choose a Section", 
    ["Dataset Overview", "Exploratory Data Analysis (EDA)", "ML Prediction Model"]
)

# 1. Dataset Overview Section
if app_mode == "Dataset Overview":
    st.subheader("📋 Dataset Preview & Information")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows", df.shape[0])
    col2.metric("Total Columns", df.shape[1])
    col3.metric("Missing Values Handled", "Yes")
    
    st.write("### Raw Data Sample")
    st.dataframe(df.head(10))
    
    st.write("### Statistical Summary")
    st.write(df.describe())

# 2. Exploratory Data Analysis (EDA) Section
elif app_mode == "Exploratory Data Analysis (EDA)":
    st.subheader("📈 Visual Data Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Survival Count by Passenger Class**")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df, x='class', hue='survived', palette='Set2', ax=ax)
        ax.set_xlabel("Passenger Class")
        ax.set_ylabel("Count")
        st.pyplot(fig)
        
    with col2:
        st.markdown("**Age Distribution of Passengers**")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df['age'], kde=True, color='teal', ax=ax)
        ax.set_xlabel("Age")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)
        
    st.markdown("**Correlation Heatmap**")
    fig, ax = plt.subplots(figsize=(8, 4))
    numeric_df = df.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)

# 3. ML Prediction Model Section
elif app_mode == "ML Prediction Model":
    st.subheader("🤖 Interactive Survival Prediction Model")
    st.write("Train a Random Forest classifier on the fly and test predictions using custom inputs.")
    
    # Preprocess model data
    model_df = df[['survived', 'pclass', 'sex', 'age', 'fare']].dropna()
    model_df['sex'] = LabelEncoder().fit_transform(model_df['sex']) # male=1, female=0
    
    X = model_df[['pclass', 'sex', 'age', 'fare']]
    y = model_df['survived']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    score = model.score(X_test, y_test)
    st.info(f"Model Accuracy on Test Set: **{score * 100:.2f}%**")
    
    st.write("### Enter Passenger Details:")
    col1, col2 = st.columns(2)
    
    with col1:
        pclass = st.selectbox("Passenger Class", [1, 2, 3], help="1 = First Class, 3 = Third Class")
        sex_input = st.selectbox("Sex", ["male", "female"])
        sex = 1 if sex_input == "male" else 0
        
    with col2:
        age = st.slider("Age", 1, 80, 28)
        fare = st.slider("Fare Paid (£)", 0.0, 500.0, 32.0)
        
    if st.button("Predict Survival", type="primary"):
        input_data = pd.DataFrame([[pclass, sex, age, fare]], columns=['pclass', 'sex', 'age', 'fare'])
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)
        
        if prediction[0] == 1:
            st.success(f"🎉 Prediction: **Survived**! (Confidence: {probability[0][1]*100:.2f}%)")
        else:
            st.error(f"⚠️ Prediction: **Did not survive**. (Confidence: {probability[0][0]*100:.2f}%)")