from flask import Flask, request
import numpy as np
import pickle
import traceback  # To print errors in detail

app = Flask(__name__)

# Load trained model
with open("classifier.pkl", "rb") as pickle_in:
    classifier = pickle.load(pickle_in)


@app.route('/')
def welcome():
    return "Welcome All"


@app.route('/predict', methods=["GET"])
def predict_titanic():
    try:
        # Retrieve float values from request
        Pclass = float(request.args.get("Pclass", 0))
        Sex = float(request.args.get("Sex", 0))  # Assuming already encoded
        Age = float(request.args.get("Age", 0))
        SibSp = float(request.args.get("SibSp", 0))
        Parch = float(request.args.get("Parch", 0))
        Ticket = float(request.args.get("Ticket", 0))  # Ensure this is float
        Fare = float(request.args.get("Fare", 0))
        Cabin = float(request.args.get("Cabin", 0))  # Ensure this is float
        Embarked = float(request.args.get("Embarked", 0))  # Ensure this is float

        # Convert input to proper 2D array
        input_data = np.array([[Pclass, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked]])

        # Debugging: Print input shape
        print("Input Shape:", input_data.shape)

        # Make prediction
        prediction = classifier.predict(input_data)

        return f"The prediction is: {prediction[0]}"

    except Exception as e:
        # Print error stack trace for debugging
        print("Error:", traceback.format_exc())
        return f"Error: {str(e)}", 500


if __name__ == "__main__":
    app.run(debug=True)
