from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

INTENTS = [
    "visual_tracking",
    "object_interaction",
    "visual_understanding",
]

INTENT_TO_ID = {name: i for i, name in enumerate(INTENTS)}
ID_TO_INTENT = {i: name for name, i in INTENT_TO_ID.items()}

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

TARGETS = ["none"] + COCO_CLASSES
TARGET_TO_ID = {name: i for i, name in enumerate(TARGETS)}
ID_TO_TARGET = {i: name for name, i in TARGET_TO_ID.items()}
COCO_TO_YOLO_ID = {name: i for i, name in enumerate(COCO_CLASSES)}

PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"

def intent_token(intent: str) -> str:
    return f"INTENT::{intent}"

def target_token(target: str) -> str:
    return f"TARGET::{target}"

class SchemaVocab:
    def __init__(self):
        tokens = [PAD, BOS, EOS]
        tokens += [intent_token(x) for x in INTENTS]
        tokens += [target_token(x) for x in TARGETS]

        self.tokens = tokens
        self.token_to_id = {t: i for i, t in enumerate(tokens)}
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}

        self.pad_id = self.token_to_id[PAD]
        self.bos_id = self.token_to_id[BOS]
        self.eos_id = self.token_to_id[EOS]

        self.intent_token_ids = [self.token_to_id[intent_token(x)] for x in INTENTS]
        self.target_token_ids = [self.token_to_id[target_token(x)] for x in TARGETS]

    def __len__(self) -> int:
        return len(self.tokens)

    def command_to_ids(self, intent: str, target: str) -> list[int]:
        if intent not in INTENT_TO_ID:
            raise ValueError(f"Unknown intent: {intent}")
        if target not in TARGET_TO_ID:
            raise ValueError(f"Unknown target: {target}")

        return [
            self.bos_id,
            self.token_to_id[intent_token(intent)],
            self.token_to_id[target_token(target)],
            self.eos_id,
        ]

    def parse_ids(self, ids: list[int]) -> tuple[str, str]:
        toks = [self.id_to_token.get(int(i), PAD) for i in ids]

        intent = None
        target = None

        for t in toks:
            if t.startswith("INTENT::"):
                intent = t.split("::", 1)[1]
            elif t.startswith("TARGET::"):
                target = t.split("::", 1)[1]

        if intent is None:
            intent = "visual_understanding"
        if target is None:
            target = "none"

        return intent, target

    def allowed_next_ids(self, generated_ids: list[int]) -> list[int]:
        if len(generated_ids) == 1:
            return self.intent_token_ids
        if len(generated_ids) == 2:
            return self.target_token_ids
        if len(generated_ids) == 3:
            return [self.eos_id]
        return [self.eos_id]


SCHEMA_VOCAB = SchemaVocab()
class CommandDataset(Dataset):
    def __init__(self, json_path: str | Path, tokenizer, max_len: int = 128):
        self.path = Path(json_path)
        self.tokenizer = tokenizer
        self.max_len = max_len

        with self.path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("Training data must be a JSON list.")

        self.data = []
        for i, item in enumerate(data):
            self.data.append(self._validate_item(item, i))

    def _validate_item(self, item: dict[str, Any], idx: int) -> dict[str, str]:
        for k in ["text", "intent", "target"]:
            if k not in item:
                raise ValueError(f"Item {idx} missing key: {k}")

        text = str(item["text"]).strip()
        intent = str(item["intent"]).strip()
        target = str(item["target"]).strip()

        if not text:
            raise ValueError(f"Item {idx} has empty text.")

        if intent not in INTENT_TO_ID:
            raise ValueError(f"Item {idx} unknown intent: {intent}")

        if target not in TARGET_TO_ID:
            raise ValueError(f"Item {idx} unknown target: {target}")

        if intent in ("visual_tracking", "object_interaction") and target == "none":
            raise ValueError(
                f"Item {idx} invalid: {intent} requires target != none. text={text}"
            )

        return {"text": text, "intent": intent, "target": target}

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.data[idx]

        enc = self.tokenizer(
            item["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors=None,
        )

        command_ids = SCHEMA_VOCAB.command_to_ids(item["intent"], item["target"])

        dec_in = command_ids[:-1]
        dec_lab = command_ids[1:]

        return {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "decoder_input_ids": torch.tensor(dec_in, dtype=torch.long),
            "decoder_labels": torch.tensor(dec_lab, dtype=torch.long),
            "intent_label": torch.tensor(INTENT_TO_ID[item["intent"]], dtype=torch.long),
            "target_label": torch.tensor(TARGET_TO_ID[item["target"]], dtype=torch.long),
            "text": item["text"],
        }


def choose_nhead(hidden_size: int) -> int:
    for h in [8, 6, 4, 3, 2, 1]:
        if hidden_size % h == 0:
            return h
    return 1


class SCTPCommandParser(nn.Module):

    def __init__(
        self,
        model_path: str,
        schema_vocab_size: int,
        num_intents: int,
        num_targets: int,
        decoder_layers: int = 2,
        dropout: float = 0.1,
        max_decoder_len: int = 8,
    ):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_path)
        hidden_size = int(self.encoder.config.hidden_size)
        nhead = choose_nhead(hidden_size)

        self.schema_emb = nn.Embedding(schema_vocab_size, hidden_size)
        self.pos_emb = nn.Embedding(max_decoder_len, hidden_size)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )

        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)
        self.norm = nn.LayerNorm(hidden_size)
        self.schema_lm_head = nn.Linear(hidden_size, schema_vocab_size)

        self.intent_head = nn.Linear(hidden_size, num_intents)
        self.target_head = nn.Linear(hidden_size, num_targets)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        enc_out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state

        batch_size, tgt_len = decoder_input_ids.shape
        pos = torch.arange(tgt_len, device=decoder_input_ids.device).unsqueeze(0)
        tgt = self.schema_emb(decoder_input_ids) + self.pos_emb(pos)

        causal_mask = torch.triu(
            torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=decoder_input_ids.device),
            diagonal=1,
        )

        memory_key_padding_mask = ~attention_mask.bool()

        dec_out = self.decoder(
            tgt=tgt,
            memory=enc_out,
            tgt_mask=causal_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        dec_out = self.norm(dec_out)
        schema_logits = self.schema_lm_head(dec_out)

        cls = enc_out[:, 0, :]
        intent_logits = self.intent_head(cls)
        target_logits = self.target_head(cls)

        return schema_logits, intent_logits, target_logits


@dataclass
class TrainConfig:
    model_path: str
    data_path: str
    output_path: str
    max_len: int = 128
    batch_size: int = 8
    epochs: int = 100
    lr: float = 2e-5
    weight_decay: float = 0.01
    aux_weight: float = 0.3
    grad_clip: float = 1.0
    seed: int = 42
    num_workers: int = 0

    # early stopping
    val_ratio: float = 0.15
    val_data_path: str | None = None
    patience: int = 8
    min_delta: float = 1e-3
    min_epochs: int = 5


def set_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_tokenizer(model_path: str):
    try:
        return AutoTokenizer.from_pretrained(model_path, fix_mistral_regex=True)
    except TypeError:
        return AutoTokenizer.from_pretrained(model_path)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(
    model: nn.Module,
    tokenizer,
    output_path: str | Path,
    cfg: TrainConfig,
    metrics: dict[str, float] | None = None,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_state": model.state_dict(),
        "config": cfg.__dict__,
        "metrics": metrics or {},
        "intents": INTENTS,
        "targets": TARGETS,
        "coco_classes": COCO_CLASSES,
        "schema_tokens": SCHEMA_VOCAB.tokens,
    }

    torch.save(payload, output_path)

    tokenizer_dir = output_path.with_suffix("")
    tokenizer_dir.mkdir(exist_ok=True)
    try:
        tokenizer.save_pretrained(tokenizer_dir)
    except Exception as e:
        print(f"Tokenizer save skipped: {e}")


def build_loaders(cfg: TrainConfig, tokenizer):
    if cfg.val_data_path:
        train_dataset = CommandDataset(cfg.data_path, tokenizer, max_len=cfg.max_len)
        val_dataset = CommandDataset(cfg.val_data_path, tokenizer, max_len=cfg.max_len)
    else:
        dataset = CommandDataset(cfg.data_path, tokenizer, max_len=cfg.max_len)

        if len(dataset) < 2:
            raise ValueError("Dataset must contain at least 2 samples for validation split.")

        val_size = max(1, int(len(dataset) * cfg.val_ratio))
        train_size = len(dataset) - val_size

        if train_size <= 0:
            raise ValueError(
                f"val_ratio={cfg.val_ratio} too large for dataset size={len(dataset)}"
            )

        generator = torch.Generator().manual_seed(cfg.seed)
        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=generator,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    return train_loader, val_loader

@torch.no_grad()
def evaluate(
    model: SCTPCommandParser,
    loader: DataLoader,
    device: torch.device,
    loss_schema_fn,
    loss_intent_fn,
    loss_target_fn,
    aux_weight: float,
) -> dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_schema = 0.0
    total_intent = 0.0
    total_target = 0.0

    total_samples = 0
    total_schema_tokens = 0
    correct_schema_tokens = 0

    correct_intent_aux = 0
    correct_target_aux = 0

    command_exact = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        decoder_input_ids = batch["decoder_input_ids"].to(device)
        decoder_labels = batch["decoder_labels"].to(device)
        intent_label = batch["intent_label"].to(device)
        target_label = batch["target_label"].to(device)

        schema_logits, intent_logits, target_logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
        )

        loss_schema = loss_schema_fn(
            schema_logits.reshape(-1, schema_logits.size(-1)),
            decoder_labels.reshape(-1),
        )
        loss_intent = loss_intent_fn(intent_logits, intent_label)
        loss_target = loss_target_fn(target_logits, target_label)
        loss = loss_schema + aux_weight * (loss_intent + loss_target)

        batch_size = input_ids.size(0)
        total_samples += batch_size

        total_loss += loss.item() * batch_size
        total_schema += loss_schema.item() * batch_size
        total_intent += loss_intent.item() * batch_size
        total_target += loss_target.item() * batch_size

        pred_schema = schema_logits.argmax(dim=-1)
        schema_mask = decoder_labels != SCHEMA_VOCAB.pad_id

        correct_schema_tokens += ((pred_schema == decoder_labels) & schema_mask).sum().item()
        total_schema_tokens += schema_mask.sum().item()

        exact = ((pred_schema == decoder_labels) | (~schema_mask)).all(dim=1)
        command_exact += exact.sum().item()

        pred_intent_aux = intent_logits.argmax(dim=-1)
        pred_target_aux = target_logits.argmax(dim=-1)

        correct_intent_aux += (pred_intent_aux == intent_label).sum().item()
        correct_target_aux += (pred_target_aux == target_label).sum().item()

    return {
        "loss": total_loss / max(1, total_samples),
        "schema_loss": total_schema / max(1, total_samples),
        "intent_loss": total_intent / max(1, total_samples),
        "target_loss": total_target / max(1, total_samples),
        "schema_token_acc": correct_schema_tokens / max(1, total_schema_tokens),
        "command_exact_acc": command_exact / max(1, total_samples),
        "intent_aux_acc": correct_intent_aux / max(1, total_samples),
        "target_aux_acc": correct_target_aux / max(1, total_samples),
    }
