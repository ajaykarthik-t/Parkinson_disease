import streamlit as st
import numpy as np
from joblib import load
import pandas as pd 
from PIL import Image
import base64
from pathlib import Path


def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

def img_to_html(img_path, width=None, height=None):
    img_bytes = img_to_bytes(img_path)
    img_html = f"<img src='data:image/png;base64,{img_bytes}' class='img-fluid'"
    if width:
        img_html += f" width='{width}'"
    if height:
        img_html += f" height='{height}'"
    img_html += ">"
    return img_html

def set_background_image(image_path):
    background_css = f"""
        <style>
        body {{
            background-image: url('data:image/jpeg;base64,{img_to_bytes(image_path)}');
            background-size: cover;
            }}
        </style>
        """
    st.markdown(background_css, unsafe_allow_html=True)

image_path = 'C:/Users/mgrpr/OneDrive/Desktop/parkinson_Prediction/parkinson_Prediction/bg_1.jpg'
image_width = 700  # Set your desired width
image_height = 400  # Set your desired height

set_background_image(image_path)

image_html = img_to_html(image_path, width=image_width, height=image_height)
styled_html = f"<p style='text-align: center; color: grey;'>{image_html}</p>"

st.markdown(styled_html, unsafe_allow_html=True)



# Load the saved model and scaler
model = load('C:/Users/mgrpr/OneDrive/Desktop/parkinson_Prediction/parkinson_Prediction/knn_model.joblib')
scaler = load('C:/Users/mgrpr/OneDrive/Desktop/parkinson_Prediction/parkinson_Prediction/scaler.joblib')

# Define a function to make predictions
def make_prediction(input_data):
    scaled_data = scaler.transform([input_data])
    prediction = model.predict(scaled_data)
    if prediction[0] == 0:
        return "Person Has Parkinson disease"
    else:
        return "Person is Safe"

def main():
# /MDVP:Fo(Hz),/MDVP:Fhi(Hz),/MDVP:Flo(Hz),MDVP:Jitter(%),MDVP:Jitter(Abs),
# /MDVP:RAP ,/ MDVP:PPQ ,/ Jitter:DDP , MDVP:Shimmer, MDVP:Shimmer(dB)
#,Shimmer:APQ3 ,Shimmer:APQ5 ,/MDVP:APQ, /Shimmer:DDA, /NHR,
# /HNR,status,/RPDE,/DFA,/spread1,/spread2,/D2,/PPE
    menu = ["Home", "Excel"]  # Add a new menu option for Excel
    choice = st.sidebar.selectbox("Select Menu", menu)

    if choice == 'Home':
        col1, col2, col3, col4, col5 = st.columns(5)  

        with col1:
            radius_mean = st.text_input("MDVP:Fo(Hz)")
            MDVP_RAP = st.text_input("MDVP:RAP")
            Shimmer_DDA = st.text_input("Shimmer:DDA")
            RPDE = st.text_input("RPDE")
            spread1 = st.text_input("spread1")

        with col2:
            texture_mean = st.text_input("MDVP:Fhi(Hz)")
            MDVP_PPQ = st.text_input("MDVP:PPQ")
            NHR = st.text_input("NHR")
            DFA = st.text_input("DFA")
            spread2 = st.text_input("spread2")

        with col3:
            perimeter_mean = st.text_input("MDVP:Flo(Hz)")
            Jitter_DDP = st.text_input("Jitter:DDP")
            MDVP_APQ = st.text_input("MDVP:APQ")
            HNR = st.text_input("HNR")
            D2 = st.text_input("D2")

        with col4:
            area_mean = st.text_input("MDVP:Jitter(%)")
            MDVP_Shimmer = st.text_input("MDVP:Shimmer")
            Shimmer_APQ3 = st.text_input("Shimmer:APQ3")
            PPE = st.text_input("PPE")

        with col5:
            smoothness_mean = st.text_input("MDVP:Jitter(Abs)")
            MDVP_Shimmer_dB = st.text_input("MDVP:Shimmer(dB)")
            Shimmer_APQ5 = st.text_input("Shimmer:APQ5")
            

        input_data = [
            radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
            MDVP_RAP, MDVP_PPQ, Jitter_DDP, MDVP_Shimmer, MDVP_Shimmer_dB,
            Shimmer_APQ3, Shimmer_APQ5, MDVP_APQ, Shimmer_DDA, NHR, HNR,
            RPDE, DFA, spread1, spread2, D2, PPE
        ]
        # input_data = [119.99200,157.30200,74.99700,0.00784,0.00007
        #            ,0.00370,0.00554,0.01109,0.04374,0.42600,0.02182,
        #            0.03130,0.02971,0.06545,0.02211,21.03300,0.414783,0.815285,-4.813031,0.266482,2.301442,0.284654]

        if st.button("Predict"):
            result = make_prediction(input_data)
            st.success(result)
    elif choice == "Excel":
        st.subheader("Upload an Excel File")

        uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx", "xls", "csv"])

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("Uploaded DataFrame:")
            st.write(df)

            df = df.dropna()
            scaled_df = scaler.transform(df)

            # Make predictions for all rows in the DataFrame
            predictions = model.predict(scaled_df)

            # Create a new DataFrame with predictions
            prediction_df = df.copy()
            prediction_df['Prediction'] = predictions

            # Display the input data and their respective predictions
            st.write("Input Data with Predictions:")
            st.write(prediction_df)

            # You can also save this DataFrame as a CSV if needed
            if st.button("Save Predictions as CSV"):
                prediction_df.to_csv("C:/Users/mgrpr/OneDrive/Desktop/parkinson_Prediction/parkinson_Prediction/Ouputs/predictions.csv   ", index=False)
                st.success("Predictions saved as 'predictions.csv'")

if __name__ == "__main__":
    main()
