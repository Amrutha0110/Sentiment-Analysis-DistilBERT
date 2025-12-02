import argparse
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_dir', required=True)
    p.add_argument('--text', nargs='+', required=False)
    return p.parse_args()

def main():
    args = parse_args()

    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    if not args.text:
        print("Provide text using --text argument")
        return

    text = " ".join(args.text)
    result = nlp(text)[0]
    print(f"Prediction: {result['label']} (score: {result['score']:.3f})")

if __name__ == "__main__":
    main()
