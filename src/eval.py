import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from src.utils import compute_metrics

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_dir', required=True)
    return p.parse_args()

def main():
    args = parse_args()

    dataset = load_dataset("imdb", split="test")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)

    def preprocess(batch):
        return tokenizer(batch["text"], truncation=True, padding=True)

    dataset = dataset.map(preprocess, batched=True)
    dataset = dataset.rename_column("label", "labels")
    dataset = dataset.remove_columns(["text"])
    dataset.set_format("torch")

    trainer = Trainer(model=model)
    metrics = trainer.evaluate(dataset)
    print(metrics)

if __name__ == "__main__":
    main()
