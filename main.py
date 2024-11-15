import pickle
import numpy as np
import pandas as pd
import sklearn
import streamlit as st


loaded_model = pickle.load(open('trained_insurance.pkl', 'rb'))



def insurance_prediction(input_data):

    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction[0])





# Main function to create the app
def main():
    # Set page configuration
    st.set_page_config(page_title="Insurance Cost Predictor",page_icon= "💻", layout="centered", initial_sidebar_state="expanded")

    # Page title
    st.title("Medical Insurance Cost Predictor💻")
    st.subheader("📊 Predict your insurance cost based on key factors.")

    # Sidebar for user instructions
    with st.sidebar:
        st.header("About")
        st.write("""
        This app predicts the cost of medical insurance based on user input such as age, BMI, smoking habits, and other details.
        Enter the required details, and click 'Predict' to see your results!
        """)
        st.write("---")
        st.write("Developed by **Subhan Tanveer** 🚀")

    # Input fields for the user
    st.markdown("### 📝 Enter your details below:")
    age = st.number_input("🧑 Age:", min_value=1, max_value=120, value=25)
    gender = st.selectbox("⚥ Gender:", options=["Male", "Female"])
    bmi = st.number_input("📏 BMI:", min_value=10.0, max_value=50.0, value=25.0, format="%.1f")
    children = st.number_input("👶 Number of Children:", min_value=0, max_value=10, value=0)
    smoker = st.selectbox("🚬 Smoker:", options=["Yes", "No"])
    region = st.selectbox(
        "📍 Region:",
        options=["Southeast", "Southwest", "Northeast", "Northwest"]
    )

    # Gender and smoker input conversion
    gender_encoded = 1 if gender == "Male" else 0
    smoker_encoded = 1 if smoker == "Yes" else 0

    # Mapping region to numerical values (example mapping, adjust to your model's requirements)
    region_map = {"Southeast": 0, "Southwest": 1, "Northeast": 2, "Northwest": 3}
    region_encoded = region_map[region]

    # Button for prediction
    if st.button("Predict"):
        input_data = [age, gender_encoded, bmi, children, smoker_encoded, region_encoded]
        prediction = insurance_prediction(input_data)

        # Display the prediction
        st.success(f"Predicted Insurance Cost: **${prediction:.2f}**")

    # Footer
    st.write("---")
    st.write("© 2024 - **Subhan Tanveer**")

# Run the app
if __name__ == '__main__':
    main()
