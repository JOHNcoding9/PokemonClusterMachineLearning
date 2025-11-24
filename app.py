from flask import Flask, request, jsonify
import pickle
import os

app = Flask(__name__)

# Build an absolute path to the model file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_kmeans.pkl")

# Load your saved model safely
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    X = [data["features"]]
    prediction = model.predict(X)[0]
    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(debug=True)
