import joblib
import pandas as pd
housing_model = joblib.load('linear_regression_model.pkl')

class HousingPricePrediction():
    # Function to predict house price
    def predict_price(self,area, bedrooms, bathrooms, stories, parking, mainroad, guestroom, basement, hotwaterheating, airconditioning, prefarea, furnishingstatus):
        # Create a DataFrame with the input values
        input_data = pd.DataFrame({
            'area': [area],
            'bedrooms': [bedrooms],
            'bathrooms': [bathrooms],
            'stories': [stories],
            'parking': [parking],
            'mainroad_yes': [1 if mainroad else 0],
            'guestroom_yes': [1 if guestroom else 0],
            'basement_yes': [1 if basement else 0],
            'hotwaterheating_yes': [1 if hotwaterheating else 0],
            'airconditioning_yes': [1 if airconditioning else 0],
            'prefarea_yes': [1 if prefarea else 0],
            'furnishingstatus_semi-furnished': [1 if furnishingstatus == 'semi-furnished' else 0],
            'furnishingstatus_unfurnished': [1 if furnishingstatus == 'unfurnished' else 0]
        })
        # Predict the price
        predicted_price = housing_model.predict(input_data)[0]
        return predicted_price