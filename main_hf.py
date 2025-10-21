from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os

app = Flask(__name__)

# Load model dari Hugging Face Hub
model_name = "TheBloke/TinyLlama-1.1B-Chat-v1-GGUF"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_name)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

@app.route("/")
def index():
    return "TinyLLaMA API on Hugging Face is running!"

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    response = generator(prompt, max_length=200, do_sample=True, temperature=0.8)
    return jsonify({"generated_text": response[0]["generated_text"]})

if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    app.run(host="0.0.0.0", port=port)
