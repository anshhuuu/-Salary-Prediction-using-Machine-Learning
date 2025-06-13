import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import streamlit as st

# Streamlit Title and Description
st.title("Salary Prediction App")
st.write("This app allows you to visualize employee data, train a salary prediction model, and predict salaries dynamically.")

# File Upload
uploaded_file = st.file_uploader("Upload your dataset (CSV format):", type=["csv"])
if uploaded_file is not None:
    # Load the dataset
    salary = pd.read_csv(uploaded_file)
    st.write("### Dataset Overview")
    st.write(salary.head())

    # Data Cleaning
    salary = salary.drop_duplicates().dropna()
    st.write(f"Dataset shape after cleaning: {salary.shape}")

    # EDA Section
    st.write("### Exploratory Data Analysis")
    if st.checkbox("Show Correlation Heatmap"):
        corr = salary[['Age', 'Years of Experience', 'Salary', 'Rating']].corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    if st.checkbox("Show Distribution of Education Levels"):
        fig, ax = plt.subplots()
        salary['Education Level'].value_counts().plot(kind='bar', ax=ax, color='teal')
        ax.set_title("Education Level Distribution")
        st.pyplot(fig)

    if st.checkbox("Show Age Distribution"):
        fig, ax = plt.subplots()
        salary['Age'].plot(kind='hist', ax=ax, bins=15, color='orange', alpha=0.7)
        ax.set_title("Age Distribution")
        st.pyplot(fig)

    # Label Encoding
    Label_Encoder = LabelEncoder()

    if 'Gender' in salary.columns:
        Label_Encoder.fit(salary['Gender'].unique())  # Fit on unique labels in the dataset
        salary['Gender_Encode'] = Label_Encoder.transform(salary['Gender'])

    if 'Education Level' in salary.columns:
        Label_Encoder.fit(salary['Education Level'].unique())
        salary['Degree_Encode'] = Label_Encoder.transform(salary['Education Level'])

    if 'Job Title' in salary.columns:
        Label_Encoder.fit(salary['Job Title'].unique())
        salary['Job_Title_Encode'] = Label_Encoder.transform(salary['Job Title'])

    if 'seniority' in salary.columns:
        Label_Encoder.fit(salary['seniority'].unique())
        salary['Seniority_Encode'] = Label_Encoder.transform(salary['seniority'])

    # Feature Scaling
    std_scaler = StandardScaler()
    salary['Age_scaled'] = std_scaler.fit_transform(salary[['Age']])
    salary['Experience_years_scaled'] = std_scaler.fit_transform(salary[['Years of Experience']])
    salary['Rating_scaled'] = std_scaler.fit_transform(salary[['Rating']])

    # Feature Selection
    X = salary[['Age_scaled', 'Experience_years_scaled', 'Rating_scaled', 'Gender_Encode',
                'Degree_Encode', 'Job_Title_Encode', 'Seniority_Encode']]
    y = salary['Salary']

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

    # Model Training
    linmodel = LinearRegression()
    linmodel.fit(X_train, y_train)

    # Model Evaluation
    predictions = linmodel.predict(X_test)
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    st.write("### Model Performance")
    st.write(f"R-squared: {r2:.2f}")
    st.write(f"Mean Absolute Error: ₹{mae:,.2f}")

    # Feature Importance Visualization
    coefficients = linmodel.coef_
    features = X.columns
    fig, ax = plt.subplots()
    pd.Series(coefficients, index=features).sort_values().plot(kind='barh', color='skyblue', ax=ax)
    ax.set_title("Feature Importance")
    st.pyplot(fig)

    # Salary Prediction Section
    st.write("### Predict Salary")
    age = st.number_input("Enter Age:", min_value=18, max_value=65, value=30)
    experience = st.number_input("Enter Years of Experience:", min_value=0, max_value=50, value=5)
    rating = st.number_input("Enter Rating (1-5):", min_value=1.0, max_value=5.0, value=3.0, step=0.1)
    gender = st.selectbox("Select Gender:", options=['Male', 'Female'])
    education = st.selectbox("Select Education Level:", options=salary['Education Level'].unique())
    job_title = st.selectbox("Select Job Title:", options=salary['Job Title'].unique())
    seniority = st.selectbox("Select Seniority:", options=salary['seniority'].unique())

    if st.button("Predict Salary"):
        # Encoding inputs
        if gender not in Label_Encoder.classes_:
            Label_Encoder.classes_ = np.append(Label_Encoder.classes_, gender)  # Add unseen label dynamically
        gender_enc = Label_Encoder.transform([gender])[0]

        if education not in Label_Encoder.classes_:
            Label_Encoder.classes_ = np.append(Label_Encoder.classes_, education)
        degree_enc = Label_Encoder.transform([education])[0]

        if job_title not in Label_Encoder.classes_:
            Label_Encoder.classes_ = np.append(Label_Encoder.classes_, job_title)
        job_title_enc = Label_Encoder.transform([job_title])[0]

        if seniority not in Label_Encoder.classes_:
            Label_Encoder.classes_ = np.append(Label_Encoder.classes_, seniority)
        seniority_enc = Label_Encoder.transform([seniority])[0]

        # Scaling inputs
        age_scaled = std_scaler.transform([[age]])[0][0]
        experience_scaled = std_scaler.transform([[experience]])[0][0]
        rating_scaled = std_scaler.transform([[rating]])[0][0]

        # Prediction
        input_data = [[age_scaled, experience_scaled, rating_scaled, gender_enc, degree_enc, job_title_enc, seniority_enc]]
        predicted_salary = linmodel.predict(input_data)[0]

        st.write(f"### Predicted Salary: ₹{predicted_salary:,.2f}")

