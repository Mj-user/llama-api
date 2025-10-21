from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os

app = Flask(__name__)

tokenizer = AutoTokenizer.from_pretrained("TheBloke/TinyLlama-1.1B-Chat-v1-GGUF", use_fast=False)
model = AutoModelForCausalLM.from_pretrained("TheBloke/TinyLlama-1.1B-Chat-v1-GGUF")
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

@app.route("/")
def index():
    return jsonify({"status": "TinyLLaMA API is running"})

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json(force=True)
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    response = generator(prompt, max_length=200, do_sample=True, temperature=0.8)
    return jsonify({"result": response[0]["generated_text"]})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # 🔹 gunakan port dari Railway
    app.run(host="0.0.0.0", port=port)
