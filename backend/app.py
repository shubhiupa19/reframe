from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import uuid
from database import save_feedback, save_prediction, init_db
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from transformers import pipeline
import os
import google.generativeai as genai                                                                                                                                                                                                                              
from dotenv import load_dotenv
load_dotenv(".env.local")
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

app = Flask(__name__)
CORS(app, origins=["https://reframe-journal.vercel.app/","http://localhost:3000"])  # Enable CORS for frontend requests

# creating a limiter object to reduce the amount of requests made to our API routes
limiter = Limiter(get_remote_address, app=app)

# preventing content over 16kb
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024

# initalize db on startup of the backend
init_db()

# get API key for cross checking with frontend requests
API_KEY = os.environ.get("API_KEY")

# Load the fine-tuned DistilBERT model (see backend/train_distilbert.py)
classifier = pipeline(
    "text-classification",
    model=os.path.join(os.path.dirname(__file__), "models/distilbert-cognitive-distortions-improved"),
    top_k=None,
)

# route to call the model and predict CD's for each sentence
@app.route("/predict", methods=["POST"])
@limiter.limit("20 per minute")
def predict():
    try:
        data = request.get_json()
        input_text = data.get("text", "")

        if not input_text:
            return jsonify({"error": "Missing input text"}), 400
        elif len(input_text.strip()) > 5000:
            return jsonify({"error": "Text is too long, please enter a shorter message" }), 400

        # Split text into sentences (same logic as frontend expects)
        sentences = re.split(r'(?<=[.!?])\s+', input_text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        # Run all sentences through the DistilBERT classifier in one batched call
        predictions = classifier(sentences, truncation=True, max_length=256)

        # Every sentence submitted in this request shares one submission_id, so they
        # can be grouped back together later even without a user/auth system.
        submission_id = str(uuid.uuid4())

        # Make predictions for each sentence
        results = []
        for i in range(len(sentences)):
            # top_k=None returns every class's score, sorted highest first
            top_prediction = predictions[i][0]
            confidence = float(top_prediction["score"])

            # Log every prediction the model makes, not just the ones shown to the
            # user below — logging failures shouldn't break the actual response.
            try:
                save_prediction(submission_id, sentences[i], top_prediction["label"], confidence)
            except Exception as e:
                print(f"Error saving prediction to database: {e}")

            if confidence > 0.2:
                results.append({
                    "input": sentences[i],
                    "prediction": top_prediction["label"],
                    "confidence": round(confidence, 3)
                })

        return jsonify({"results": results})

    except Exception as e:
        return (jsonify({"error": str(e)}), 500)

# route to post feedback to our SQL db
@app.route("/feedback", methods=["POST"])
@limiter.limit("10 per minute")
def feedback():
    try:
        request_key = request.headers.get("X-API-Key")
        if request_key != API_KEY:
            return jsonify({"error": "Unauthorized"}), 401
        data = request.get_json()
        if len(data.get("text", "").strip()) > 500:
            return jsonify({"error": "Sentence is too long for feedback"}), 400
        feedback_id = save_feedback(
            text=data.get("text"),
            predicted_distortion=data.get("predicted_distortion"),
            user_correction=data.get("user_correction"),
            is_accepted=data.get("is_accepted"),
            confidence=data.get("confidence")
        )
        return (jsonify({"feedback_id": feedback_id}))
    except Exception as e:
        return (jsonify({"error": str(e)}), 500)
        


    

# route to call the Gemini API to rewrite the journal entry in a healthier way
@app.route('/rewrite', methods=['POST'])
def rewrite():
    try:
        data = request.get_json()
        # 1. extract text and distortions
        text = data["text"]
        distortions = data["distortions"]
        distortion_lines = "\n".join(
            f'"{d["input"]}" → {d["prediction"]} ({d["confidence"]})'
            for d in distortions
        )
        # 2. build the prompt       
        prompt = f""" You are a therapist specializing in cognitive behavioral therapy.                                                                                    
        The following journal entry has been analyzed for cognitive distortions.
                                                                                                                                                        
        Original entry:
        {text}                                                                                                                                               
                    
        Detected distortions by sentence:                                                                                                                    
        {distortion_lines}
                                                                                                                                                        
        Rewrite the full journal entry with healthier thought patterns, preserving the original meaning and tone.
        Return only the rewritten entry, with no explanation or commentary."""                                                                                                                     
    # 3. call Gemini
        gemini_model = genai.GenerativeModel("gemini-3-flash-preview")
        response = gemini_model.generate_content(prompt) 
    # 4. return the result 
        return jsonify({ "rewritten": response.text }) 
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=["GET"])
def health():
    # Warms up the model so the first real /predict request doesn't pay the
    # cost of PyTorch's slow first inference call. Logged (not returned in the
    # response) so a failed warmup doesn't change this route's status code —
    # the server itself is still up either way.
    try:
        classifier(["warmup"])
        print("Model warmup succeeded")
    except Exception as e:
        print(f"Model warmup failed: {e}")

    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, port=5001)
