from predict_price_class import HousingPricePrediction

if __name__ == '__main__':
    # Example usage of the prediction function
    area = 1500
    bedrooms = 3
    bathrooms = 2
    stories = 2
    parking = 2
    mainroad = True
    guestroom = False
    basement = True
    hotwaterheating = True
    airconditioning = False
    prefarea = True
    furnishingstatus = 'semi-furnished'

    # Create an instance of the HousingPricePrediction class
    predictor = HousingPricePrediction()

    # Predict the price
    predicted_price = predictor.predict_price(area, bedrooms, bathrooms, stories, parking, mainroad, guestroom, basement, hotwaterheating, airconditioning, prefarea, furnishingstatus)
    print(f"Predicted Price: {predicted_price:.2f}")
