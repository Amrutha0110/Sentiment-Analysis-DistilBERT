# 🎭 Sentiment Analysis using DistilBERT (IMDb Dataset)

This project fine-tunes **DistilBERT** to classify IMDb movie reviews as **Positive** or **Negative** using the HuggingFace Transformers library.

---

## 🚀 Features
- Loads IMDb dataset using `datasets`
- Tokenizes text using DistilBERT tokenizer
- Fine-tunes DistilBERT using Trainer API
- Saves best model and tokenizer
- Provides inference script (`predict.py`)
- Includes evaluation script (`eval.py`)
- Clean project structure

---

## 📦 Installation

```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
🏋️ Train the Model
python src/train.py --epochs 1 --batch_size 16 --model_name distilbert-base-uncased --output_dir saved_model
📊 Evaluate
python src/eval.py --model_dir saved_model
🔮 Predict Sentiment
python src/predict.py --model_dir saved_model --text "The movie was amazing!"
Example Output:
Prediction: POSITIVE (score: 0.984)
📁 Project Structure
sentiment-analysis-distilbert/
├── src/
│   ├── train.py
│   ├── eval.py
│   ├── predict.py
│   └── utils.py
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
📜 License

MIT License © 2025 Devadi Amrutha Varshini
