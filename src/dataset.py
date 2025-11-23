import json
import random
import re
from typing import List, Dict, Any
from torch.utils.data import Dataset


class PIIDataset(Dataset):
    def __init__(self, path: str, tokenizer, label_list: List[str], max_length: int = 256, is_train: bool = True):
        self.items = []
        self.tokenizer = tokenizer
        self.label_list = label_list
        self.label2id = {l: i for i, l in enumerate(label_list)}
        self.max_length = max_length
        self.is_train = is_train

        items_raw = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                items_raw.append(obj)
        
        # Aggressive data augmentation for small datasets
        if is_train and len(items_raw) < 10:
            augmented = []
            
            for obj in items_raw:
                text = obj["text"]
                entities = obj.get("entities", [])
                
                # Original sample (keep as-is)
                augmented.append(obj.copy())
                
                # 1. Simple duplication (5x)
                for _ in range(5):
                    augmented.append(obj.copy())
                
                # 2. Pattern variations for phone numbers
                for entity in entities:
                    if entity["label"] == "PHONE":
                        # Convert word-based phone to digit format
                        phone_text = text[entity["start"]:entity["end"]]
                        if any(word in phone_text for word in ["nine", "eight", "seven", "six", "five", "four", "three", "two", "one", "zero"]):
                            # Generate variations: 10-digit phone numbers
                            phone_variants = [
                                "9876543210",
                                "9123456789", 
                                "9988776655",
                                "9876512345"
                            ]
                            for variant in phone_variants[:2]:  # Use 2 variants
                                new_text = text[:entity["start"]] + variant + text[entity["end"]:]
                                offset = len(variant) - (entity["end"] - entity["start"])
                                new_entities = []
                                for e in entities:
                                    new_e = e.copy()
                                    if e["start"] > entity["end"]:
                                        new_e["start"] += offset
                                        new_e["end"] += offset
                                    elif e["start"] >= entity["start"]:
                                        new_e["start"] = entity["start"]
                                        new_e["end"] = entity["start"] + len(variant)
                                    new_entities.append(new_e)
                                augmented.append({"id": obj["id"] + f"_phone_{variant[:4]}", "text": new_text, "entities": new_entities})
                
                # 3. Pattern variations for credit cards
                for entity in entities:
                    if entity["label"] == "CREDIT_CARD":
                        card_text = text[entity["start"]:entity["end"]]
                        # Generate variations with different spacing
                        card_variants = [
                            "5555 5555 5555 4444",
                            "4242 4242 4242 4242",
                            "1234 5678 9012 3456",
                            "5555555555554444",  # No spaces
                            "5555-5555-5555-4444"  # Dashes
                        ]
                        for variant in card_variants[:3]:  # Use 3 variants
                            new_text = text[:entity["start"]] + variant + text[entity["end"]:]
                            offset = len(variant) - (entity["end"] - entity["start"])
                            new_entities = []
                            for e in entities:
                                new_e = e.copy()
                                if e["start"] > entity["end"]:
                                    new_e["start"] += offset
                                    new_e["end"] += offset
                                elif e["start"] >= entity["start"]:
                                    new_e["start"] = entity["start"]
                                    new_e["end"] = entity["start"] + len(variant)
                                new_entities.append(new_e)
                            augmented.append({"id": obj["id"] + f"_card_{variant[:4]}", "text": new_text, "entities": new_entities})
                
                # 4. Pattern variations for emails
                for entity in entities:
                    if entity["label"] == "EMAIL":
                        email_text = text[entity["start"]:entity["end"]]
                        # Generate variations with different domains and formats
                        email_variants = [
                            ("john dot doe at gmail dot com", "john.doe@gmail.com"),
                            ("jane dot smith at yahoo dot com", "jane.smith@yahoo.com"),
                            ("test dot user at outlook dot com", "test.user@outlook.com"),
                        ]
                        for variant_text, variant_clean in email_variants[:2]:  # Use 2 variants
                            new_text = text[:entity["start"]] + variant_text + text[entity["end"]:]
                            offset = len(variant_text) - (entity["end"] - entity["start"])
                            new_entities = []
                            for e in entities:
                                new_e = e.copy()
                                if e["start"] > entity["end"]:
                                    new_e["start"] += offset
                                    new_e["end"] += offset
                                elif e["start"] >= entity["start"]:
                                    new_e["start"] = entity["start"]
                                    new_e["end"] = entity["start"] + len(variant_text)
                                new_entities.append(new_e)
                            augmented.append({"id": obj["id"] + f"_email_{variant_clean[:5]}", "text": new_text, "entities": new_entities})
                
                # 5. Pattern variations for person names
                for entity in entities:
                    if entity["label"] == "PERSON_NAME":
                        name_text = text[entity["start"]:entity["end"]]
                        # Generate variations with different name patterns
                        name_variants = [
                            "john dot doe",
                            "jane dot smith", 
                            "robert dot johnson",
                            "priyanka dot verma",
                            "michael dot brown"
                        ]
                        for variant in name_variants[:3]:  # Use 3 variants
                            new_text = text[:entity["start"]] + variant + text[entity["end"]:]
                            offset = len(variant) - (entity["end"] - entity["start"])
                            new_entities = []
                            for e in entities:
                                new_e = e.copy()
                                if e["start"] > entity["end"]:
                                    new_e["start"] += offset
                                    new_e["end"] += offset
                                elif e["start"] >= entity["start"]:
                                    new_e["start"] = entity["start"]
                                    new_e["end"] = entity["start"] + len(variant)
                                new_entities.append(new_e)
                            augmented.append({"id": obj["id"] + f"_name_{variant[:5]}", "text": new_text, "entities": new_entities})
                
                # 6. Pattern variations for dates
                for entity in entities:
                    if entity["label"] == "DATE":
                        date_text = text[entity["start"]:entity["end"]]
                        # Generate variations with different date formats
                        date_variants = [
                            "01 02 2024",
                            "15 03 2025",
                            "31 12 2023",
                            "01-02-2024",
                            "2024 01 02"
                        ]
                        for variant in date_variants[:3]:  # Use 3 variants
                            new_text = text[:entity["start"]] + variant + text[entity["end"]:]
                            offset = len(variant) - (entity["end"] - entity["start"])
                            new_entities = []
                            for e in entities:
                                new_e = e.copy()
                                if e["start"] > entity["end"]:
                                    new_e["start"] += offset
                                    new_e["end"] += offset
                                elif e["start"] >= entity["start"]:
                                    new_e["start"] = entity["start"]
                                    new_e["end"] = entity["start"] + len(variant)
                                new_entities.append(new_e)
                            augmented.append({"id": obj["id"] + f"_date_{variant[:5]}", "text": new_text, "entities": new_entities})
                
                # 7. Pattern variations for cities
                for entity in entities:
                    if entity["label"] == "CITY":
                        city_text = text[entity["start"]:entity["end"]]
                        # Generate variations with different cities
                        city_variants = [
                            "mumbai",
                            "delhi",
                            "chennai",
                            "bangalore",
                            "kolkata"
                        ]
                        for variant in city_variants[:3]:  # Use 3 variants
                            new_text = text[:entity["start"]] + variant + text[entity["end"]:]
                            offset = len(variant) - (entity["end"] - entity["start"])
                            new_entities = []
                            for e in entities:
                                new_e = e.copy()
                                if e["start"] > entity["end"]:
                                    new_e["start"] += offset
                                    new_e["end"] += offset
                                elif e["start"] >= entity["start"]:
                                    new_e["start"] = entity["start"]
                                    new_e["end"] = entity["start"] + len(variant)
                                new_entities.append(new_e)
                            augmented.append({"id": obj["id"] + f"_city_{variant[:5]}", "text": new_text, "entities": new_entities})
            
            # Shuffle augmented data
            random.shuffle(augmented)
            items_raw = augmented
            print(f"Data augmentation: {len(items_raw)} samples generated from {len([json.loads(l) for l in open(path, 'r')])} original samples")
        
        for obj in items_raw:
            text = obj["text"]
            entities = obj.get("entities", [])

            char_tags = ["O"] * len(text)
            for e in entities:
                s, e_idx, lab = e["start"], e["end"], e["label"]
                if s < 0 or e_idx > len(text) or s >= e_idx:
                    continue
                char_tags[s] = f"B-{lab}"
                for i in range(s + 1, e_idx):
                    char_tags[i] = f"I-{lab}"

            enc = tokenizer(
                text,
                return_offsets_mapping=True,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=True,
            )
            offsets = enc["offset_mapping"]
            input_ids = enc["input_ids"]
            attention_mask = enc["attention_mask"]

            bio_tags = []
            for (start, end) in offsets:
                if start == end:
                    bio_tags.append("O")
                else:
                    # Better alignment: use the first character of the token
                    if start < len(char_tags):
                        tag = char_tags[start]
                        # If we get an I-tag but previous token was O or different entity, convert to B
                        if len(bio_tags) > 0 and tag.startswith("I-"):
                            prev_tag = bio_tags[-1]
                            if prev_tag == "O" or (not prev_tag.startswith("I-") and not prev_tag.startswith("B-")):
                                # Convert I to B if it's the start of a new entity
                                tag = tag.replace("I-", "B-")
                        bio_tags.append(tag)
                    else:
                        bio_tags.append("O")

            if len(bio_tags) != len(input_ids):
                bio_tags = ["O"] * len(input_ids)

            label_ids = [self.label2id.get(t, self.label2id["O"]) for t in bio_tags]

            self.items.append(
                {
                    "id": obj["id"],
                    "text": text,
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": label_ids,
                    "offset_mapping": offsets,
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.items[idx]


def collate_batch(batch, pad_token_id: int, label_pad_id: int = -100):
    input_ids_list = [x["input_ids"] for x in batch]
    attention_list = [x["attention_mask"] for x in batch]
    labels_list = [x["labels"] for x in batch]

    max_len = max(len(ids) for ids in input_ids_list)

    def pad(seq, pad_value, max_len):
        return seq + [pad_value] * (max_len - len(seq))

    input_ids = [pad(ids, pad_token_id, max_len) for ids in input_ids_list]
    attention_mask = [pad(am, 0, max_len) for am in attention_list]
    labels = [pad(lab, label_pad_id, max_len) for lab in labels_list]

    out = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "ids": [x["id"] for x in batch],
        "texts": [x["text"] for x in batch],
        "offset_mapping": [x["offset_mapping"] for x in batch],
    }
    return out
