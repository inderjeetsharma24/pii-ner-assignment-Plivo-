import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import numpy as np

from dataset import PIIDataset, collate_batch
from labels import LABELS, LABEL2ID
from model import create_model


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="distilbert-base-uncased")
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--dev", default="data/dev.jsonl")
    ap.add_argument("--out_dir", default="out")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--use_class_weights", action="store_true", default=True)
    ap.add_argument("--eval_during_training", action="store_true", default=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_ds = PIIDataset(args.train, tokenizer, LABELS, max_length=args.max_length, is_train=True)
    
    # Load dev set for evaluation during training
    if args.eval_during_training:
        dev_ds = PIIDataset(args.dev, tokenizer, LABELS, max_length=args.max_length, is_train=False)
        dev_dl = DataLoader(
            dev_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=lambda b: collate_batch(b, pad_token_id=tokenizer.pad_token_id),
        )

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_batch(b, pad_token_id=tokenizer.pad_token_id),
    )

    model = create_model(args.model_name)
    model.to(args.device)
    model.train()

    # Calculate class weights to emphasize B-tags and entity labels
    class_weights = None
    if args.use_class_weights:
        # Count label frequencies in training data
        label_counts = torch.zeros(len(LABELS))
        for item in train_ds:
            labels = torch.tensor(item["labels"])
            for label_id in labels:
                if label_id >= 0:  # Ignore padding (-100)
                    label_counts[label_id] += 1
        
        # Calculate weights: higher weight for rare labels (B-tags, entities)
        # O tag gets weight 1.0, B-tags get 5.0, I-tags get 2.0
        weights = torch.ones(len(LABELS))
        for i, label in enumerate(LABELS):
            if label == "O":
                weights[i] = 1.0
            elif label.startswith("B-"):
                weights[i] = 5.0  # Emphasize B-tags
            elif label.startswith("I-"):
                weights[i] = 2.0  # Moderate weight for I-tags
            else:
                weights[i] = 1.0
        
        # Normalize by frequency (inverse frequency weighting)
        label_counts = label_counts + 1  # Avoid division by zero
        weights = weights / label_counts
        weights = weights / weights.sum() * len(LABELS)  # Normalize
        
        class_weights = weights.to(args.device)
        print(f"Class weights: {dict(zip(LABELS, weights.tolist()))}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_dl) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=max(1, int(0.1 * total_steps)), num_training_steps=total_steps
    )

    for epoch in range(args.epochs):
        running_loss = 0.0
        model.train()
        for batch in tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.epochs}"):
            input_ids = torch.tensor(batch["input_ids"], device=args.device)
            attention_mask = torch.tensor(batch["attention_mask"], device=args.device)
            labels = torch.tensor(batch["labels"], device=args.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            
            # Use custom loss with class weights if specified
            if class_weights is not None:
                logits = outputs.logits
                loss_fct = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            else:
                loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

        avg_loss = running_loss / max(1, len(train_dl))
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")
        
        # Evaluate on dev set during training
        if args.eval_during_training and (epoch + 1) % 3 == 0:  # Every 3 epochs
            model.eval()
            dev_loss = 0.0
            with torch.no_grad():
                for batch in dev_dl:
                    input_ids = torch.tensor(batch["input_ids"], device=args.device)
                    attention_mask = torch.tensor(batch["attention_mask"], device=args.device)
                    labels = torch.tensor(batch["labels"], device=args.device)
                    
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    if class_weights is not None:
                        logits = outputs.logits
                        loss_fct = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
                        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
                    else:
                        loss = outputs.loss
                    dev_loss += loss.item()
            
            avg_dev_loss = dev_loss / max(1, len(dev_dl))
            print(f"  Dev loss: {avg_dev_loss:.4f}")
            model.train()

    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print(f"Saved model + tokenizer to {args.out_dir}")


if __name__ == "__main__":
    main()
