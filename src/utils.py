from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score

def get_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(model_name)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="binary")
    }
