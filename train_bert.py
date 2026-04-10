# train_bert.py
import os
import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Use CPU
device = torch.device("cpu")
print("Using device:", device)

# Load dataset
DATA_PATH = os.path.join(os.getcwd(), "data", "merged_news.csv")
df = pd.read_csv(DATA_PATH)
print("Dataset loaded. Total articles:", len(df))

# Keep only valid labels
df = df[df['label'].isin(['FAKE', 'REAL'])]
print("Filtered dataset. Total articles:", len(df))

# Combine title and text
df['content'] = df['title'].fillna('') + " " + df['text'].fillna('')

# Encode labels
le = LabelEncoder()
df['label_enc'] = le.fit_transform(df['label'])
print("Label mapping:", dict(zip(le.classes_, le.transform(le.classes_))))
print("Unique encoded labels:", df['label_enc'].unique())

# Train-test split
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['content'], df['label_enc'], test_size=0.2, random_state=42
)

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=512)
test_encodings = tokenizer(list(test_texts), truncation=True, padding=True, max_length=512)

# Dataset class
class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels.iloc[idx])
        return item

train_dataset = NewsDataset(train_encodings, train_labels)
test_dataset = NewsDataset(test_encodings, test_labels)

# Load BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model.to(device)

# Metrics function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

# Training arguments (CPU-safe, remove evaluation_strategy if errors)
training_args = TrainingArguments(
    output_dir='./model/results',
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    save_strategy='steps',  # save checkpoints
    save_steps=500,
    logging_dir='./model/logs',
    logging_steps=50
    # Removed evaluation_strategy and load_best_model_at_end for CPU compatibility
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# Train model
trainer.train()

# Save model and tokenizer
model.save_pretrained("./model/fake_news_bert")
tokenizer.save_pretrained("./model/fake_news_bert")
print("BERT training complete and model saved!")
