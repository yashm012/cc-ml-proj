!pip install --upgrade datasets transformers

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# configs
class Config:
    # model configs
    teacher_model_name = "distilbert-base-uncased"  # smaller BERT variant for faster training
    max_length = 256
    batch_size = 16
    learning_rate = 2e-5
    student_lr = 1e-3
    num_epochs = 1
    temperature = 4.0  
    alpha = 0.7  # weight for distillation loss

    # Student model configuration
    vocab_size = 30522  # DistilBERT vocab size
    embed_dim = 128
    hidden_dim = 256
    num_classes = 2

config = Config()

# clear datasets cache to fix for glob pattern errors
print("Clearing datasets cache...")
!rm -rf ~/.cache/huggingface/datasets/*

print("Loading IMDb dataset...")
# additional failsafe
dataset = load_dataset("imdb", revision='main')

# use subset for faster training
train_dataset = dataset["train"].shuffle(seed=42).select(range(5000))
test_dataset = dataset["test"].shuffle(seed=42).select(range(1000))

print(f"Train samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

tokenizer = AutoTokenizer.from_pretrained(config.teacher_model_name)

class IMDbDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# create datasets
train_texts = train_dataset['text']
train_labels = train_dataset['label']
test_texts = test_dataset['text']
test_labels = test_dataset['label']

train_data = IMDbDataset(train_texts, train_labels, tokenizer, config.max_length)
test_data = IMDbDataset(test_texts, test_labels, tokenizer, config.max_length)

train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=config.batch_size, shuffle=False)

# teacher model (pre-trained)
print("Initializing teacher model...")
teacher_model = AutoModelForSequenceClassification.from_pretrained(
    config.teacher_model_name,
    num_labels=config.num_classes
).to(device)

# student model (smaller neural network)
class StudentModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, max_length):
        super(StudentModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.max_length = max_length

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        lstm_out, _ = self.lstm(x)

        # use attention mask to get meaningful representation
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            lstm_out = lstm_out * mask
            # average pooling over valid tokens
            lengths = attention_mask.sum(dim=1, keepdim=True).float()
            pooled = lstm_out.sum(dim=1) / (lengths + 1e-9) # Add small epsilon for numerical stability
        else:
            # simple average pooling
            pooled = lstm_out.mean(dim=1)

        # classification
        x = self.dropout(pooled)
        logits = self.fc(x)

        return logits

print("Initializing student model...")
student_model = StudentModel(
    config.vocab_size,
    config.embed_dim,
    config.hidden_dim,
    config.num_classes,
    config.max_length
).to(device)

# training functions
def train_teacher():
    """Train the teacher model on IMDb dataset"""
    print("\n=== Training Teacher Model ===")

    teacher_model.train()
    # corrected optimizer import: use torch.optim.AdamW
    optimizer = AdamW(teacher_model.parameters(), lr=config.learning_rate)

    total_loss = 0
    for batch in tqdm(train_loader, desc="Training Teacher"):
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = teacher_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Teacher training loss: {avg_loss:.4f}")

def evaluate_model(model, data_loader, model_name):
    """Evaluate model performance"""
    model.eval()
    all_predictions = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Evaluating {model_name}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            if hasattr(model, 'config'):  # teacher model (Hugging Face)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
            else:  # student model (custom nn.Module)
                logits = model(input_ids=input_ids, attention_mask=attention_mask)

            probs = F.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            # ensure probs have shape (batch_size, num_classes) before extending
            all_probs.extend(probs.cpu().numpy().tolist()) # Convert to list of lists

    # calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)

    # for AUC, we need probabilities of positive class
    # check if all_probs is not empty and has a positive class column (index 1)
    if len(all_probs) > 0 and len(all_probs[0]) > 1:
         probs_positive = [prob[1] for prob in all_probs]
         auc = roc_auc_score(all_labels, probs_positive)
    else:
         # handle cases where AUC can't be computed (e.g., only one class in batch/dataset)
         auc = float('nan')
         print("Warning: Could not compute AUC. Check if the test set contains samples from both classes.")


    print(f"\n{model_name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}" if not np.isnan(auc) else f"AUC: N/A")

    return accuracy, f1, auc

def knowledge_distillation_loss(student_logits, teacher_logits, labels, temperature, alpha):
    """
    Compute knowledge distillation loss
    Args:
        student_logits: logits from student model
        teacher_logits: logits from teacher model
        labels: ground truth labels
        temperature: temperature for softmax
        alpha: weight for distillation loss
    """
    # distillation loss (KL divergence between teacher and student)
    # ensure shapes match and teacher_logits are detached to prevent gradient flow
    teacher_probs = F.softmax(teacher_logits.detach() / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    distillation_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)

    # standard cross-entropy loss with ground truth
    ce_loss = F.cross_entropy(student_logits, labels)

    # combined loss
    total_loss = alpha * distillation_loss + (1 - alpha) * ce_loss

    return total_loss, distillation_loss, ce_loss

def train_student_with_distillation():
    """Train student model using knowledge distillation"""
    print("\n=== Training Student Model with Knowledge Distillation ===")

    student_model.train()
    teacher_model.eval()  # teacher in eval mode

    # optimizer import: torch.optim.Adam
    optimizer = torch.optim.Adam(student_model.parameters(), lr=config.student_lr)

    total_loss = 0
    total_distillation_loss = 0
    total_ce_loss = 0

    for batch in tqdm(train_loader, desc="Training Student"):
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # get teacher predictions (no gradients)
        with torch.no_grad():
            teacher_outputs = teacher_model(input_ids=input_ids, attention_mask=attention_mask)
            teacher_logits = teacher_outputs.logits

        # get student predictions
        student_logits = student_model(input_ids=input_ids, attention_mask=attention_mask)

        # compute knowledge distillation loss
        loss, distill_loss, ce_loss = knowledge_distillation_loss(
            student_logits, teacher_logits, labels,
            config.temperature, config.alpha
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_distillation_loss += distill_loss.item()
        total_ce_loss += ce_loss.item()

    avg_loss = total_loss / len(train_loader)
    avg_distill_loss = total_distillation_loss / len(train_loader)
    avg_ce_loss = total_ce_loss / len(train_loader)

    print(f"Student training - Total loss: {avg_loss:.4f}")
    print(f"Student training - Distillation loss: {avg_distill_loss:.4f}")
    print(f"Student training - CE loss: {avg_ce_loss:.4f}")

# main
if __name__ == "__main__":
    print("=== Knowledge Distillation for IMDb Sentiment Classification ===")
    print(f"Teacher model: {config.teacher_model_name}")
    print(f"Student model: BiLSTM with {config.embed_dim}D embeddings and {config.hidden_dim}D hidden layer")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # step 1: train teacher model
    train_teacher()

    # step 2: evaluate teacher model
    print("\n=== Evaluating Teacher Model ===")
    teacher_accuracy, teacher_f1, teacher_auc = evaluate_model(teacher_model, test_loader, "Teacher")

    # step 3: train student model with knowledge distillation
    train_student_with_distillation()

    # step 4: evaluate student model
    print("\n=== Evaluating Student Model ===")
    student_accuracy, student_f1, student_auc = evaluate_model(student_model, test_loader, "Student")

    # step 5: compare results
    print("\n" + "="*50)
    print("FINAL COMPARISON")
    print("="*50)
    print(f"Teacher Model ({config.teacher_model_name}):")
    print(f"  Accuracy: {teacher_accuracy:.4f}")
    print(f"  F1 Score: {teacher_f1:.4f}")
    print(f"  AUC: {teacher_auc:.4f}" if not np.isnan(teacher_auc) else f"AUC: N/A") # Handle potential NaN AUC

    print(f"\nStudent Model (BiLSTM):")
    print(f"  Accuracy: {student_accuracy:.4f}")
    print(f"  F1 Score: {student_f1:.4f}")
    print(f"  AUC: {student_auc:.4f}" if not np.isnan(student_auc) else f"AUC: N/A") # Handle potential NaN AUC

    # knowledge retention
    print(f"\nKnowledge Retention:")
    if teacher_accuracy > 0:
        print(f"  Accuracy retained: {(student_accuracy/teacher_accuracy)*100:.1f}%")
    else:
         print(f"  Accuracy retained: N/A (Teacher Accuracy is 0)")

    if teacher_f1 > 0:
        print(f"  F1 score retained: {(student_f1/teacher_f1)*100:.1f}%")
    else:
        print(f"  F1 score retained: N/A (Teacher F1 is 0)")

    if not np.isnan(teacher_auc) and teacher_auc > 0:
        print(f"  AUC retained: {(student_auc/teacher_auc)*100:.1f}%")
    else:
         print(f"  AUC retained: N/A (Teacher AUC is 0 or NaN)")

    # model size comparison
    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    student_params = sum(p.numel() for p in student_model.parameters())

    print(f"\nModel Size Comparison:")
    print(f"  Teacher parameters: {teacher_params:,}")
    print(f"  Student parameters: {student_params:,}")
    if teacher_params > 0:
        print(f"  Size reduction: {(1 - student_params/teacher_params)*100:.1f}%")
    else:
        print(f"  Size reduction: N/A (Teacher model has 0 parameters)")


    print("\n=== Knowledge Distillation Complete ===")

# additional utility function for inference
def predict_sentiment(text, model, tokenizer, device):
    """Predict sentiment for a single text"""
    model.eval()

    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=config.max_length,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        if hasattr(model, 'config'):  # teacher model (Hugging Face)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
        else:  # student model (custom nn.Module)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)

        probs = F.softmax(logits, dim=-1)
        prediction = torch.argmax(logits, dim=-1)

    sentiment = "Positive" if prediction.item() == 1 else "Negative"
    confidence = probs.max().item()

    return sentiment, confidence

# Example usage:
# sentiment, confidence = predict_sentiment("This movie was amazing!", student_model, tokenizer, device)
# print(f"Sentiment: {sentiment}, Confidence: {confidence:.3f}")
