# Sentimental Analysis (AI / ML)
# 1. Install and import necessary libraries
!pip install --upgrade datasets fsspec transformers

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from torch.utils.data import Dataset
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from collections import Counter

# 2. Load IMDb dataset
dataset = load_dataset("imdb")

# 3. Stratified sampling for balanced classes
train_texts, _, train_labels, _ = train_test_split(
    dataset['train']['text'],
    dataset['train']['label'],
    train_size=5000,
    stratify=dataset['train']['label'],
    random_state=42
)

test_texts, _, test_labels, _ = train_test_split(
    dataset['test']['text'],
    dataset['test']['label'],
    train_size=1000,
    stratify=dataset['test']['label'],
    random_state=42
)

# Confirm class balance
print("Train label distribution:", Counter(train_labels))
print("Test label distribution:", Counter(test_labels))

# 4. Tokenization
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)

# 5. Dataset class
class IMDbDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

train_dataset = IMDbDataset(train_encodings, train_labels)
test_dataset = IMDbDataset(test_encodings, test_labels)

# 6. Load model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 7. Training config
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    report_to="none"
)

# 8. Metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    report = classification_report(labels, preds, output_dict=True)
    return {
        "accuracy": report["accuracy"],
        "f1_macro": report["macro avg"]["f1-score"]
    }

# 9. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# 10. Train
trainer.train()

# 11. Evaluate
eval_results = trainer.evaluate()
print("\nFinal Test Set Evaluation:")
print(eval_results)

# 12. Predict custom input
def predict_sentiment(texts):
    model.eval()
    inputs = tokenizer(texts, truncation=True, padding=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
    return ["Positive" if label == 1 else "Negative" for label in preds]

sample_texts = [
    "I absolutely loved this movie. It's one of the best I've ever seen!",
    "This was a waste of time. The plot was terrible and acting worse.",
    "It was okay, not great but not bad either."
]

print("\nSample Predictions:")
print(predict_sentiment(sample_texts))