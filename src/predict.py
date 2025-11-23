import json
import argparse
import torch
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification
from labels import ID2LABEL, label_is_pii
import os


def bio_to_spans(text, offsets, label_ids, probs=None, threshold=0.0):
    """
    Convert BIO tag sequence to entity spans with proper character alignment.
    Handles token-to-character mapping carefully and merges adjacent tokens.
    """
    spans = []
    current_label = None
    current_start = None
    current_end = None
    current_tokens = []  # Track token indices for confidence calculation

    for i, ((start, end), lid) in enumerate(zip(offsets, label_ids)):
        # Skip special tokens (CLS, SEP, PAD) which have (0, 0) offsets
        if start == 0 and end == 0:
            if current_label is not None:
                # Finalize current span
                if probs is None or any(probs[tok_idx, label_ids[tok_idx]].item() >= threshold 
                                       for tok_idx in current_tokens):
                    spans.append((current_start, current_end, current_label))
                current_label = None
                current_tokens = []
            continue
            
        label = ID2LABEL.get(int(lid), "O")
        if label == "O":
            if current_label is not None:
                # Finalize current span before O tag
                if probs is None or any(probs[tok_idx, label_ids[tok_idx]].item() >= threshold 
                                       for tok_idx in current_tokens):
                    spans.append((current_start, current_end, current_label))
                current_label = None
                current_tokens = []
            continue

        try:
            prefix, ent_type = label.split("-", 1)
        except ValueError:
            continue
            
        # Get confidence for this token
        token_conf = probs[i, lid].item() if probs is not None else 1.0
        
        if prefix == "B":
            # Save previous span if exists
            if current_label is not None:
                if probs is None or any(probs[tok_idx, label_ids[tok_idx]].item() >= threshold 
                                       for tok_idx in current_tokens):
                    spans.append((current_start, current_end, current_label))
            # Start new entity
            current_label = ent_type
            current_start = start
            current_end = end
            current_tokens = [i]
        elif prefix == "I":
            # Continue current entity or start new one
            if current_label == ent_type:
                # Extend current span - use max end position
                current_end = max(current_end, end) if current_end is not None else end
                current_tokens.append(i)
            else:
                # I-tag without matching B-tag - start new entity
                if current_label is not None:
                    if probs is None or any(probs[tok_idx, label_ids[tok_idx]].item() >= threshold 
                                           for tok_idx in current_tokens):
                        spans.append((current_start, current_end, current_label))
                current_label = ent_type
                current_start = start
                current_end = end
                current_tokens = [i]

    # Finalize last span
    if current_label is not None:
        if probs is None or any(probs[tok_idx, label_ids[tok_idx]].item() >= threshold 
                               for tok_idx in current_tokens):
            spans.append((current_start, current_end, current_label))

    # Post-process: merge overlapping/adjacent spans of same type
    merged_spans = []
    for start, end, label in sorted(spans, key=lambda x: (x[0], x[1])):
        if merged_spans and merged_spans[-1][2] == label:
            prev_start, prev_end, _ = merged_spans[-1]
            # Merge if adjacent or overlapping
            if start <= prev_end + 1:  # +1 for adjacent
                merged_spans[-1] = (prev_start, max(prev_end, end), label)
                continue
        merged_spans.append((start, end, label))

    return merged_spans


def validate_and_refine_spans(text, spans):
    """
    Hybrid approach: Use rule-based patterns to validate and refine model predictions.
    This helps fix span alignment and filter false positives.
    More aggressive pattern matching to improve precision.
    """
    refined = []
    
    # More precise patterns that work with STT format
    patterns = {
        "CREDIT_CARD": [
            re.compile(r'\d{4}\s+\d{4}\s+\d{4}\s+\d{4}'),  # 16 digits with spaces (exact format)
            re.compile(r'\d{4}[\s-]\d{4}[\s-]\d{4}[\s-]\d{4}'),  # 16 digits with optional spaces/dashes
            re.compile(r'\d{13,19}'),  # 13-19 digits (card number range)
        ],
        "PHONE": [
            re.compile(r'\d{10}'),  # 10 digits (exact)
            re.compile(r'\d{3}[\s-]?\d{3}[\s-]?\d{4}'),  # US format
        ],
        "EMAIL": [
            re.compile(r'[a-zA-Z0-9._%+-]+\s+dot\s+[a-zA-Z0-9._%+-]+\s+at\s+[a-zA-Z0-9.-]+\s+dot\s+[a-zA-Z]+'),  # STT format
            re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'),  # Standard format
        ],
        "PERSON_NAME": [
            re.compile(r'[a-zA-Z]+\s+dot\s+[a-zA-Z]+'),  # "first dot last" format (exact)
            re.compile(r'[A-Z][a-z]+\s+[A-Z][a-z]+'),  # Standard format
        ],
        "DATE": [
            re.compile(r'\d{1,2}\s+\d{1,2}\s+\d{2,4}'),  # Date with spaces (exact format)
            re.compile(r'\d{1,2}[\s/-]\d{1,2}[\s/-]\d{2,4}'),  # Various date formats
            re.compile(r'\d{4}[\s/-]\d{1,2}[\s/-]\d{1,2}'),  # YYYY-MM-DD
        ],
        "CITY": [
            re.compile(r'\b(?:mumbai|delhi|chennai|bangalore|kolkata|pune|hyderabad|ahmedabad|jaipur|lucknow)\b', re.IGNORECASE),  # Known city names
        ],
    }
    
    # First pass: Find all pattern matches in the text (rule-based extraction)
    # This gives us precise spans based on patterns
    rule_based_spans = []
    for label, label_patterns in patterns.items():
        for pattern in label_patterns:
            for match in pattern.finditer(text):
                rule_based_spans.append((match.start(), match.end(), label))
    
    # Remove duplicates and sort
    rule_based_spans = sorted(set(rule_based_spans), key=lambda x: (x[0], x[1]))
    
    # If we have good rule-based spans, prefer them over model predictions for precision
    # Rule-based extraction is more reliable for exact span alignment
    if rule_based_spans:
        # Use rule-based spans as primary source, but keep model predictions that don't conflict
        refined = rule_based_spans.copy()
        
        # Add model predictions that don't overlap with rule-based ones
        for start, end, label in spans:
            # Check if this prediction overlaps with any rule-based span
            overlaps = False
            for rb_start, rb_end, rb_label in rule_based_spans:
                if not (end <= rb_start or start >= rb_end):
                    overlaps = True
                    break
            
            # If no overlap and it's a PII entity, add it
            if not overlaps and label_is_pii(label):
                # Validate it's not a false positive
                span_text = text[start:end] if start < len(text) and end <= len(text) else ""
                false_positive_keywords = ["and", "or", "the", "is", "at", "dot", "com", "from", "to", "on", "in", "i", "am", "will"]
                if span_text.strip() and not any(fp in span_text.strip().lower() for fp in false_positive_keywords):
                    refined.append((start, end, label))
    else:
        # No rule-based spans found, use model predictions with validation
        refined = []
        for start, end, label in spans:
            span_text = text[start:end] if start < len(text) and end <= len(text) else ""
        
        # Try to find a rule-based match that overlaps with this prediction
        best_match = None
        best_overlap = 0
        
        for rb_start, rb_end, rb_label in rule_based_spans:
            if rb_label == label:
                # Calculate overlap
                overlap_start = max(start, rb_start)
                overlap_end = min(end, rb_end)
                overlap = max(0, overlap_end - overlap_start)
                
                # Calculate overlap ratio
                pred_len = end - start
                rb_len = rb_end - rb_start
                overlap_ratio = overlap / max(pred_len, rb_len) if max(pred_len, rb_len) > 0 else 0
                
                # Prefer matches with high overlap
                if overlap_ratio > 0.5 and overlap > best_overlap:
                    best_overlap = overlap
                    best_match = (rb_start, rb_end, rb_label)
        
        # If we found a good rule-based match, use it (more precise)
        if best_match:
            refined.append(best_match)
        else:
            # No rule-based match, try pattern matching on the span itself
            is_valid = False
            if label in patterns:
                for pattern in patterns[label]:
                    match = pattern.search(span_text)
                    if match:
                        pattern_start = start + match.start()
                        pattern_end = start + match.end()
                        if pattern_start >= 0 and pattern_end <= len(text):
                            refined.append((pattern_start, pattern_end, label))
                            is_valid = True
                            break
            
            # If still no match, try expanding context
            if not is_valid and label in patterns:
                context_start = max(0, start - 15)
                context_end = min(len(text), end + 15)
                context = text[context_start:context_end]
                
                for pattern in patterns[label]:
                    match = pattern.search(context)
                    if match:
                        pattern_start = context_start + match.start()
                        pattern_end = context_start + match.end()
                        # Only use if reasonably close to original prediction
                        distance = abs(pattern_start - start) + abs(pattern_end - end)
                        if distance < 20:  # Within 20 chars
                            refined.append((pattern_start, pattern_end, label))
                            is_valid = True
                            break
            
            # Last resort: keep original if it's a PII entity and looks reasonable
            if not is_valid:
                span_text_clean = span_text.strip()
                false_positive_keywords = ["and", "or", "the", "is", "at", "dot", "com", "from", "to", "on", "in", "i", "am", "will"]
                if (span_text_clean and len(span_text_clean) >= 3 and 
                    label_is_pii(label) and 
                    not any(fp in span_text_clean.lower() for fp in false_positive_keywords)):
                    refined.append((start, end, label))
    
    # Post-process: Remove overlapping spans, prefer longer/more specific matches
    refined = sorted(refined, key=lambda x: (x[0], -(x[1] - x[0])))
    final = []
    for start, end, label in refined:
        # Check for overlaps
        overlap = False
        for f_start, f_end, f_label in final:
            if not (end <= f_start or start >= f_end):
                # Overlap detected
                if label == f_label:
                    # Same label, keep the longer one
                    if (end - start) > (f_end - f_start):
                        final.remove((f_start, f_end, f_label))
                        overlap = False
                        break
                    else:
                        overlap = True
                        break
                else:
                    # Different labels - prefer PII labels, then longer spans
                    if label_is_pii(label) and not label_is_pii(f_label):
                        final.remove((f_start, f_end, f_label))
                        overlap = False
                        break
                    elif not label_is_pii(label) and label_is_pii(f_label):
                        overlap = True
                        break
                    else:
                        # Both PII or both non-PII - keep longer
                        if (end - start) > (f_end - f_start):
                            final.remove((f_start, f_end, f_label))
                            overlap = False
                            break
                        else:
                            overlap = True
                            break
        
        if not overlap:
            final.append((start, end, label))
    
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="out")
    ap.add_argument("--model_name", default=None)
    ap.add_argument("--input", default="data/dev.jsonl")
    ap.add_argument("--output", default="out/dev_pred.json")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--confidence_threshold", type=float, default=0.1)
    ap.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir if args.model_name is None else args.model_name)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    model.to(args.device)
    model.eval()
    # Use torch.inference_mode for faster inference on CPU
    if args.device == "cpu":
        torch.set_num_threads(1)  # Optimize for single-threaded inference

    results = {}

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = obj["text"]
            uid = obj["id"]

            enc = tokenizer(
                text,
                return_offsets_mapping=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            offsets = enc["offset_mapping"][0].tolist()
            input_ids = enc["input_ids"].to(args.device)
            attention_mask = enc["attention_mask"].to(args.device)

            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = out.logits[0]
                # Get probabilities for confidence thresholding
                probs = torch.softmax(logits, dim=-1).cpu()
                # Use argmax for predictions
                pred_ids = logits.argmax(dim=-1).cpu().tolist()
            
            # Filter out special tokens before span extraction
            filtered_offsets = []
            filtered_pred_ids = []
            filtered_probs = []
            for i, ((start, end), pred_id) in enumerate(zip(offsets, pred_ids)):
                if start != 0 or end != 0:  # Not a special token
                    filtered_offsets.append((start, end))
                    filtered_pred_ids.append(pred_id)
                    filtered_probs.append(probs[i])
            
            # Convert to tensor for indexing
            filtered_probs_tensor = torch.stack(filtered_probs) if filtered_probs else None
            
            spans = bio_to_spans(text, filtered_offsets, filtered_pred_ids, 
                                filtered_probs_tensor, args.confidence_threshold)
            
            # Apply hybrid rule-based refinement
            refined_spans = validate_and_refine_spans(text, spans)

            ents = []
            for s, e, lab in refined_spans:
                # Validate span is within text bounds
                if s < 0 or e > len(text) or s >= e:
                    continue
                ents.append(
                    {
                        "start": int(s),
                        "end": int(e),
                        "label": lab,
                        "pii": bool(label_is_pii(lab)),
                    }
                )
            results[uid] = ents

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Wrote predictions for {len(results)} utterances to {args.output}")


if __name__ == "__main__":
    main()
