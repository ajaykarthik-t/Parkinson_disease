import numpy as np
import pandas as pd
from joblib import load

# Load the saved model and scaler
model = load('C:/Users/mgrpr/OneDrive/Desktop/parkinson_Prediction/parkinson_Prediction/knn_model.joblib')

scaler = load('C:/Users/mgrpr/OneDrive/Desktop/parkinson_Prediction/parkinson_Prediction/scaler.joblib')
def make_prediction(input_data):
    feature_names = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
       'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
       'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
       'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'spread1',
       'spread2', 'D2', 'PPE']  

    # Create a DataFrame with the input data
    input_df = pd.DataFrame([input_data], columns=feature_names)
    
    # Scale the input data using the scaler
    scaled_data = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(scaled_data)
    if prediction[0] == 1:
        return "Person Has Parkinson Disease"
    else:
        return "Person is Safe"

    
def main():
    print("Parkinson Disease Prediction")
    input_data = [119.99200,157.30200,74.99700,0.00784,0.00007
               ,0.00370,0.00554,0.01109,0.04374,0.42600,0.02182,
               0.03130,0.02971,0.06545,0.02211,21.03300,0.414783,0.815285,-4.813031,0.266482,2.301442,0.284654]
    result = make_prediction(input_data)
    print(result)


if __name__ == "__main__":
    main()

