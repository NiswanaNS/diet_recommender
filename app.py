from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model and label encoders
try:
    with open("diet_model.sav", "rb") as model_file:
        diet_model = pickle.load(model_file)
    with open("label_encoders.pkl", "rb") as le_file:
        label_encoders = pickle.load(le_file)
except FileNotFoundError:
    print("⚠ Model or label encoders file not found.")
    diet_model, label_encoders = None, None

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Get user inputs
            Age = float(request.form["Age"])
            Weight_kg = float(request.form["Weight_kg"])
            Height_cm = float(request.form["Height_cm"])
            Gender = request.form["Gender"]

            # Calculate BMI
            BMI = round(Weight_kg / ((Height_cm / 100) ** 2), 2)  # Round to 2 decimal places

            # Encode Gender
            Gender_mapping = {"Male": 1, "Female": 0}
            Gender = Gender_mapping.get(Gender, 0)

            # Prepare input for model
            user_input = np.array([[Age, Gender, Height_cm, Weight_kg, BMI]])

            # Make prediction
            diet_prediction = diet_model.predict(user_input)

            # Decode predicted category
            recommended_diet = label_encoders["Diet_Recommendation"].inverse_transform(diet_prediction)[0]

            # Redirect to result page with BMI
            return redirect(url_for("result", diet=recommended_diet, BMI=BMI))

        except ValueError:
            return "⚠ Error: Please enter valid numeric values."

    return render_template("index.html")

@app.route("/result/<diet>")
def result(diet):
    BMI = request.args.get("BMI")  # Get BMI from URL parameters
    return render_template("result.html", diet=diet, BMI=BMI)
def get_BMI_category(BMI):
    if BMI < 18.5:
        return "Underweight"
    elif 18.5 <= BMI < 24.9:
        return "Normal Weight"
    elif 25 <= BMI < 29.9:
        return "Overweight"
    else:
        return "Obese"

@app.route("/diet/<diet_type>")
def diet_plan(diet_type):
    return render_template(f"diet_{diet_type}.html")

if __name__ == "__main__":
    app.run(debug=True)
