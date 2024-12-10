import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
import numpy as np
import time
import pandas as pd
import gc
from torch.amp import autocast, GradScaler
from collections import deque
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import precision_recall_fscore_support, f1_score, confusion_matrix
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma  # Focusing parameter

    def forward(self, inputs, targets):
        # Get BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, weight=self.alpha, reduction='none'
        )

        # Apply focusing parameter
        pt = torch.exp(-bce_loss)
        focal_loss = (1 - pt) ** self.gamma * bce_loss

        return focal_loss.mean()


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_f1 = None
        self.counter = 0

    def __call__(self, f1_score, model):
        if self.best_f1 is None:
            self.best_f1 = f1_score
            return False
        elif f1_score < self.best_f1 + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_f1 = f1_score
            self.counter = 0
        return False


class ModelWrapper(nn.Module):
    def __init__(self, model_name, n_classes, hidden_size=768):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        # Increased dropout
        self.dropout1 = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.3)

        # Add layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Two intermediate layers with residual connection
        self.intermediate1 = nn.Linear(hidden_size, hidden_size)
        self.intermediate2 = nn.Linear(hidden_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(outputs, 'pooler_output'):
            pooled = outputs.pooler_output
        else:
            pooled = outputs[0][:, 0, :]

        # First intermediate layer with residual connection
        x = self.dropout1(pooled)
        x = self.layer_norm(x)
        residual = x
        x = F.gelu(self.intermediate1(x))
        x = self.dropout2(x)
        x = x + residual

        # Second intermediate layer with residual connection
        residual = x
        x = F.gelu(self.intermediate2(x))
        x = self.dropout2(x)
        x = x + residual

        return self.classifier(x)


class EnsembleClassifier(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.models = nn.ModuleDict({
            'codebert': CodeBERTWrapper(n_classes),
            'roberta': RoBERTaWrapper(n_classes),
            'distilbert': DistilBERTWrapper(n_classes)
        })
        self.weights = nn.Parameter(torch.ones(len(self.models), n_classes))
        self.thresholds = nn.Parameter(torch.ones(n_classes) * 0.5)  # Per-class thresholds
        self.temperature = nn.Parameter(torch.ones(n_classes))
        self.softmax = nn.Softmax(dim=0)

        # Add tracking lists
        self.train_losses = []
        self.val_losses = []
        self.f1_scores = []

    def forward(self, input_ids, attention_mask):
        predictions = []
        for name, model in self.models.items():
            pred = model(
                input_ids=input_ids[name],
                attention_mask=attention_mask[name]
            )
            predictions.append(pred)

        stacked = torch.stack(predictions)
        weights = self.softmax(self.weights)
        weighted_sum = (stacked * weights.unsqueeze(1)).sum(dim=0)
        return weighted_sum / self.temperature

    def predict(self, input_ids, attention_mask):
        self.eval()
        with torch.no_grad():
            # Multiple forward passes for model averaging
            pred1 = self.forward(input_ids, attention_mask)
            pred2 = self.forward(input_ids, attention_mask)
            pred3 = self.forward(input_ids, attention_mask)

            # Average predictions
            avg_pred = (pred1 + pred2 + pred3) / 3
            probs = torch.sigmoid(avg_pred)

            # Use per-class thresholds
            return probs > self.thresholds.unsqueeze(0)


class MultiTokenizerDataset(Dataset):
    def __init__(self, texts, class_names, labels=None):
        self.texts = texts
        self.labels = labels
        self.class_names = class_names

        # Initialize tokenizers
        self.tokenizers = {
            'codebert': AutoTokenizer.from_pretrained('microsoft/codebert-base'),
            'roberta': AutoTokenizer.from_pretrained('roberta-base'),
            'distilbert': AutoTokenizer.from_pretrained('distilbert-base-uncased')
        }
        self.max_length = 128

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encodings = {}

        # Get encodings from each tokenizer
        for name, tokenizer in self.tokenizers.items():
            encoding = tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            encodings[name] = {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten()
            }

        if self.labels is not None:
            encodings['labels'] = torch.FloatTensor(self.labels[idx])

        return encodings


class CodeBERTWrapper(ModelWrapper):
    def __init__(self, n_classes):
        super().__init__('microsoft/codebert-base', n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(outputs, 'pooler_output'):
            pooled = outputs.pooler_output
        else:
            pooled = outputs[0][:, 0, :]

        # First intermediate layer with residual connection
        x = self.dropout1(pooled)
        x = self.layer_norm(x)
        residual = x
        x = F.gelu(self.intermediate1(x))
        x = self.dropout2(x)
        x = x + residual

        # Second intermediate layer with residual connection
        residual = x
        x = F.gelu(self.intermediate2(x))
        x = self.dropout2(x)
        x = x + residual

        return self.classifier(x)


class RoBERTaWrapper(ModelWrapper):
    def __init__(self, n_classes):
        super().__init__('roberta-base', n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(outputs, 'pooler_output'):
            pooled = outputs.pooler_output
        else:
            pooled = outputs[0][:, 0, :]

        # First intermediate layer with residual connection
        x = self.dropout1(pooled)
        x = self.layer_norm(x)
        residual = x
        x = F.gelu(self.intermediate1(x))
        x = self.dropout2(x)
        x = x + residual

        # Second intermediate layer with residual connection
        residual = x
        x = F.gelu(self.intermediate2(x))
        x = self.dropout2(x)
        x = x + residual

        return self.classifier(x)


class DistilBERTWrapper(ModelWrapper):
    def __init__(self, n_classes):
        super().__init__('distilbert-base-uncased', n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(outputs, 'pooler_output'):
            pooled = outputs.pooler_output
        else:
            pooled = outputs[0][:, 0, :]

        # First intermediate layer with residual connection
        x = self.dropout1(pooled)
        x = self.layer_norm(x)
        residual = x
        x = F.gelu(self.intermediate1(x))
        x = self.dropout2(x)
        x = x + residual

        # Second intermediate layer with residual connection
        residual = x
        x = F.gelu(self.intermediate2(x))
        x = self.dropout2(x)
        x = x + residual

        return self.classifier(x)


def load_pharo_data(train_path, test_path):
    """
    Load Pharo training and test data from parquet files
    """
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    # Verify the required columns exist
    required_columns = ['class', 'comment_sentence', 'labels']
    for col in required_columns:
        if col not in train_df.columns or col not in test_df.columns:
            raise ValueError(f"Missing required column: {col}")

    return train_df, test_df


def compute_class_weights(train_loader, device):
    """Compute class weights based on training data distribution"""
    label_counts = torch.zeros(5)  # For 5 Python categories
    total_samples = 0

    # Count occurrences of each class
    for batch in train_loader:
        labels = batch['labels']
        label_counts += labels.sum(dim=0)
        total_samples += labels.size(0)

    # Calculate weights (inverse of frequency)
    weights = total_samples / (len(train_loader.dataset) * label_counts)
    # Normalize weights
    weights = weights / weights.sum() * len(weights)

    return weights.to(device)


def train_ensemble(model, train_loader, val_loader, device, epochs=20, patience=5):
    # Compute class weights
    class_weights = compute_class_weights(train_loader, device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-5,
        weight_decay=0.2,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    # Warm restart scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=5,  # Reset every 5 epochs
        T_mult=2,  # Double period after each restart
        eta_min=1e-6
    )

    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    early_stopping = EarlyStopping(patience=patience)
    scaler = GradScaler()

    best_f1 = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = {k: v['input_ids'].to(device)
                         for k, v in batch.items() if k != 'labels'}
            attention_mask = {k: v['attention_mask'].to(device)
                              for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=autocast_device):
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')

        # Validation with per-class metrics
        model.eval()
        val_loss = 0
        predictions = []
        true_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = {k: v['input_ids'].to(device)
                             for k, v in batch.items() if k != 'labels'}
                attention_mask = {k: v['attention_mask'].to(device)
                                  for k, v in batch.items() if k != 'labels'}
                labels = batch['labels'].to(device)

                with autocast(device_type=autocast_device):
                    # Use predict method for validation
                    preds = model.predict(input_ids, attention_mask)
                    loss = criterion(preds.float(), labels)

                val_loss += loss.item()
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        # Calculate per-class and average metrics
        predictions = np.array(predictions)
        true_labels = np.array(true_labels)

        # Per-class metrics
        per_class_f1 = []
        for i in range(true_labels.shape[1]):
            precision, recall, f1, _ = precision_recall_fscore_support(
                true_labels[:, i],
                predictions[:, i],
                average='binary',
                zero_division=0
            )
            per_class_f1.append(f1)
            print(f'\nClass {i} metrics:')
            print(f'F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')

        # Average metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels,
            predictions,
            average='macro',
            zero_division=0
        )

        print(f'\nEpoch {epoch}:')
        print(f'Average train loss: {avg_train_loss:.4f}')
        print(f'Average val loss: {avg_val_loss:.4f}')
        print(f'F1 Score: {f1:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')

        # Add tracking here
        model.train_losses.append(avg_train_loss)
        model.val_losses.append(avg_val_loss)
        model.f1_scores.append(f1)

        # Update scheduler
        scheduler.step()

        # Save best model based on F1 score
        if f1 > best_f1:
            best_f1 = f1
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"New best F1 score: {best_f1:.4f}")

        # Early stopping check
        if early_stopping(f1, model):
            print(f"\nEarly stopping triggered. Best F1: {best_f1:.4f}")
            break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, best_f1


def analyze_model_performance(model, test_loader, device, class_names):
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = {k: v['input_ids'].to(device)
                         for k, v in batch.items() if k != 'labels'}
            attention_mask = {k: v['attention_mask'].to(device)
                              for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)

            preds = model.predict(input_ids, attention_mask)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    predictions = np.array(predictions)
    true_labels = np.array(true_labels)

    # Per-class analysis
    print("\nPer-class Performance:")
    print("=" * 50)

    for i, class_name in enumerate(class_names):
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels[:, i],
            predictions[:, i],
            average='binary',
            zero_division=0
        )

        print(f"\n{class_name}:")
        print(f"F1 Score: {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")

        # Confusion matrix analysis
        tn = ((1 - predictions[:, i]) * (1 - true_labels[:, i])).sum()
        tp = (predictions[:, i] * true_labels[:, i]).sum()
        fn = ((1 - predictions[:, i]) * true_labels[:, i]).sum()
        fp = (predictions[:, i] * (1 - true_labels[:, i])).sum()

        print(f"True Negatives: {tn}")
        print(f"True Positives: {tp}")
        print(f"False Negatives: {fn}")
        print(f"False Positives: {fp}")

    return predictions, true_labels


if __name__ == "__main__":

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(f"Using device: {device}")

    # Ensure Mixed Precision training works on both CPU and GPU
    if device.type == 'cuda':
        scaler = GradScaler()
        autocast_device = 'cuda'
    else:
        scaler = GradScaler(enabled=False)
        autocast_device = 'cpu'

    print(f"Mixed precision training using {autocast_device} autocast")

    # Load data
    train_df, test_df = load_pharo_data(
        './python_train-00000-of-00001.parquet',
        './python_test-00000-of-00001.parquet'
    )

    # Define categories
    categories = [
        'Keyimplementationpoints', 'Example', 'Responsibilities',
        'Classreferences', 'Intent'
    ]

    print("\nCategory Distribution in Training Data:")
    train_labels = np.array(train_df['labels'].tolist())
    for i, category in enumerate(categories):
        count = train_labels[:, i].sum()
        print(f"{category}: {count} ({count / len(train_labels) * 100:.2f}%)")

    # Prepare data
    train_texts = [f"{row['class']}|{row['comment_sentence']}"
                   for _, row in train_df.iterrows()]
    train_labels = np.array(train_df['labels'].tolist())

    # Create validation split
    val_size = int(len(train_texts) * 0.1)  # 10% validation split
    val_texts = train_texts[-val_size:]
    val_labels = train_labels[-val_size:]
    train_texts = train_texts[:-val_size]
    train_labels = train_labels[:-val_size]

    # Create datasets
    train_dataset = MultiTokenizerDataset(train_texts, categories, train_labels)
    val_dataset = MultiTokenizerDataset(val_texts, categories, val_labels)
    test_dataset = MultiTokenizerDataset(
        [f"{row['class']}|{row['comment_sentence']}" for _, row in test_df.iterrows()],
        categories,
        np.array(test_df['labels'].tolist())
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        num_workers=4,
        pin_memory=True
    )

    start_time = time.perf_counter()

    # Initialize and train model
    model = EnsembleClassifier(len(categories)).to(device)
    model, best_f1 = train_ensemble(model, train_loader, val_loader, device)

    # Evaluate on test set
    print("\nEvaluating on test set...")
    predictions, true_labels = analyze_model_performance(model, test_loader, device, categories)

    # Visualizations
    plt.figure(figsize=(15, 10))

    # Plot 1: Per-class F1 Scores
    plt.subplot(2, 2, 1)
    per_class_f1 = []
    for i in range(len(categories)):
        p, r, f1, _ = precision_recall_fscore_support(
            true_labels[:, i],
            predictions[:, i],
            average='binary',
            zero_division=0
        )
        per_class_f1.append(f1)

    sns.barplot(x=categories, y=per_class_f1)
    plt.xticks(rotation=45, ha='right')
    plt.title('F1 Score by Category')
    plt.tight_layout()

    # Plot 2: Confusion Matrices for each category
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.ravel()

    for idx, category in enumerate(categories):
        cm = confusion_matrix(true_labels[:, idx], predictions[:, idx])
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[idx],
                    xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive'])
        axes[idx].set_title(f'{category}')

    plt.tight_layout()
    plt.savefig('confusion_matrices.png')

    # Plot 3: Model learning curves
    plt.figure(figsize=(10, 5))
    plt.plot(model.train_losses, label='Training Loss')
    plt.plot(model.val_losses, label='Validation Loss')
    plt.title('Learning Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('learning_curves.png')

    # Save detailed results to file
    with open('model_results.txt', 'w') as f:
        f.write("Pharo Comment Classification Results\n")
        f.write("=" * 50 + "\n\n")

        f.write("Per-Category Performance:\n")
        for i, category in enumerate(categories):
            p, r, f1, _ = precision_recall_fscore_support(
                true_labels[:, i],
                predictions[:, i],
                average='binary',
                zero_division=0
            )
            f.write(f"\n{category}:\n")
            f.write(f"F1 Score: {f1:.4f}\n")
            f.write(f"Precision: {p:.4f}\n")
            f.write(f"Recall: {r:.4f}\n")

            cm = confusion_matrix(true_labels[:, i], predictions[:, i])
            f.write(f"True Negatives: {cm[0, 0]}\n")
            f.write(f"False Positives: {cm[0, 1]}\n")
            f.write(f"False Negatives: {cm[1, 0]}\n")
            f.write(f"True Positives: {cm[1, 1]}\n")

        f.write("\nOverall Performance:\n")
        p, r, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='macro', zero_division=0
        )
        f.write(f"Macro F1: {f1:.4f}\n")
        f.write(f"Macro Precision: {p:.4f}\n")
        f.write(f"Macro Recall: {r:.4f}\n")

    stop_time = time.perf_counter()
    elapsed_time = stop_time - start_time
    print(f"Code block took {elapsed_time:.4f} seconds")

    print("\nResults have been saved to 'model_results.txt'")
    print("Visualizations have been saved as PNG files")