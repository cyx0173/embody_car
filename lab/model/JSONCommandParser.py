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

PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"

DEFAULT_ACTIONS = [
    "schedule_add", "schedule_query", "schedule_delete", "schedule_update",
    "memory_add", "memory_query", "memory_update", "memory_delete",
    "todo_add", "todo_query", "todo_done",
    "device_control", "find_person", "come_to_person", "speak_to_person",
    "come_here", "stop_robot",
    "chat_capability", "chat_identity", "chat_status", "open_chat", "query_context",
]


def canonical_json(obj: dict[str, Any]) -> str:
    """Return a compact, stable JSON string used as the seq2seq target."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=False)


def normalize_record(item: dict[str, Any]) -> dict[str, Any]:
    """Convert one training row into the final JSON object we want the model to emit."""
    actions = item.get("actions", [])
    if isinstance(actions, str):
        actions = [actions]
    if not isinstance(actions, list):
        actions = []

    slots = item.get("slots", {})
    if not isinstance(slots, dict):
        slots = {}

    return {
        "actions": actions,
        "slots": slots,
        "reply_type": str(item.get("reply_type", "confirm")),
        "reply": str(item.get("reply", "")),
    }


class JsonCharVocab:
    def __init__(self, chars: list[str] | None = None):
        base = [PAD, BOS, EOS, UNK]
        chars = chars or []
        seen = set(base)
        self.tokens = list(base)
        for ch in chars:
            if ch not in seen:
                seen.add(ch)
                self.tokens.append(ch)
        self.token_to_id = {t: i for i, t in enumerate(self.tokens)}
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}
        self.pad_id = self.token_to_id[PAD]
        self.bos_id = self.token_to_id[BOS]
        self.eos_id = self.token_to_id[EOS]
        self.unk_id = self.token_to_id[UNK]

    def __len__(self) -> int:
        return len(self.tokens)

    @classmethod
    def build(cls, texts: list[str]) -> "JsonCharVocab":
        chars: list[str] = []
        for text in texts:
            chars.extend(list(text))
        return cls(chars)

    def encode(self, text: str, max_len: int) -> tuple[list[int], list[int]]:
        ids = [self.bos_id]
        ids += [self.token_to_id.get(ch, self.unk_id) for ch in text]
        ids.append(self.eos_id)
        ids = ids[:max_len]
        if ids[-1] != self.eos_id:
            ids[-1] = self.eos_id
        labels = ids[1:]
        dec_in = ids[:-1]
        while len(dec_in) < max_len - 1:
            dec_in.append(self.pad_id)
            labels.append(self.pad_id)
        return dec_in, labels

    def decode(self, ids: list[int]) -> str:
        out: list[str] = []
        for idx in ids:
            token = self.id_to_token.get(int(idx), UNK)
            if token == EOS:
                break
            if token in (PAD, BOS, UNK):
                continue
            out.append(token)
        return "".join(out)

    def to_dict(self) -> dict[str, Any]:
        return {"tokens": self.tokens}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JsonCharVocab":
        return cls(list(data["tokens"])[4:])


class JsonCommandDataset(Dataset):
    def __init__(self, jsonl_path: str | Path, tokenizer, vocab: JsonCharVocab | None = None,
                 max_input_len: int = 128, max_output_len: int = 256):
        self.path = Path(jsonl_path)
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len

        rows: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                text = str(item.get("text", "")).strip()
                if not text:
                    raise ValueError(f"line {line_no}: missing text")
                target = canonical_json(normalize_record(item))
                rows.append({"text": text, "target": target, "raw": item})

        if not rows:
            raise ValueError(f"No samples found in {self.path}")

        self.data = rows
        self.vocab = vocab or JsonCharVocab.build([x["target"] for x in rows])

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.data[idx]
        enc = self.tokenizer(
            item["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_input_len,
            return_tensors=None,
        )
        dec_in, dec_lab = self.vocab.encode(item["target"], self.max_output_len)
        return {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "decoder_input_ids": torch.tensor(dec_in, dtype=torch.long),
            "decoder_labels": torch.tensor(dec_lab, dtype=torch.long),
            "text": item["text"],
            "target": item["target"],
        }


class JSONCommandParser(nn.Module):
    def __init__(self, model_path: str, output_vocab_size: int,
                 decoder_layers: int = 2, dropout: float = 0.1,
                 max_decoder_len: int = 256):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_path)
        hidden_size = int(self.encoder.config.hidden_size)
        nhead = choose_nhead(hidden_size)
        self.output_emb = nn.Embedding(output_vocab_size, hidden_size)
        self.pos_emb = nn.Embedding(max_decoder_len, hidden_size)
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=decoder_layers)
        self.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, output_vocab_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                decoder_input_ids: torch.Tensor) -> torch.Tensor:
        enc_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        batch_size, tgt_len = decoder_input_ids.shape
        pos = torch.arange(tgt_len, device=decoder_input_ids.device).unsqueeze(0)
        tgt = self.output_emb(decoder_input_ids) + self.pos_emb(pos)
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
        return self.lm_head(self.norm(dec_out))


def choose_nhead(hidden_size: int) -> int:
    for h in [12, 8, 6, 4, 3, 2, 1]:
        if hidden_size % h == 0:
            return h
    return 1


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


def set_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class TrainConfig:
    model_path: str
    data_path: str
    output_path: str
    max_input_len: int = 128
    max_output_len: int = 256
    batch_size: int = 8
    epochs: int = 30
    lr: float = 2e-5
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    seed: int = 42
    val_ratio: float = 0.15
    patience: int = 6
    min_delta: float = 1e-3
    num_workers: int = 0


def build_loaders(cfg: TrainConfig, tokenizer):
    full_dataset = JsonCommandDataset(
        cfg.data_path,
        tokenizer,
        max_input_len=cfg.max_input_len,
        max_output_len=cfg.max_output_len,
    )
    val_size = max(1, int(len(full_dataset) * cfg.val_ratio))
    train_size = len(full_dataset) - val_size
    if train_size <= 0:
        raise ValueError("Dataset must contain at least 2 samples")
    generator = torch.Generator().manual_seed(cfg.seed)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    return train_loader, val_loader, full_dataset.vocab


@torch.no_grad()
def evaluate(model: JSONCommandParser, loader: DataLoader, vocab: JsonCharVocab,
             device: torch.device, loss_fn) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct_tokens = 0
    exact = 0
    valid_json = 0
    samples = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        decoder_input_ids = batch["decoder_input_ids"].to(device)
        labels = batch["decoder_labels"].to(device)
        logits = model(input_ids, attention_mask, decoder_input_ids)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        batch_size = input_ids.size(0)
        total_loss += loss.item() * batch_size
        mask = labels != vocab.pad_id
        pred = logits.argmax(dim=-1)
        correct_tokens += ((pred == labels) & mask).sum().item()
        total_tokens += mask.sum().item()
        exact += (((pred == labels) | (~mask)).all(dim=1)).sum().item()
        samples += batch_size
        for row in pred.detach().cpu().tolist():
            try:
                json.loads(vocab.decode(row))
                valid_json += 1
            except Exception:
                pass
    return {
        "loss": total_loss / max(1, samples),
        "token_acc": correct_tokens / max(1, total_tokens),
        "exact_acc": exact / max(1, samples),
        "valid_json_rate": valid_json / max(1, samples),
    }


def save_checkpoint(model: JSONCommandParser, tokenizer, vocab: JsonCharVocab,
                    output_path: str | Path, cfg: TrainConfig, metrics: dict[str, float]):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "config": cfg.__dict__,
        "metrics": metrics,
        "vocab": vocab.to_dict(),
    }, output_path)
    tokenizer_dir = output_path.with_suffix("")
    tokenizer_dir.mkdir(exist_ok=True)
    tokenizer.save_pretrained(tokenizer_dir)


def train(cfg: TrainConfig) -> dict[str, float]:
    set_seed(cfg.seed)
    device = get_device()
    print(f"[train-json-nlu] device={device}")
    tokenizer = load_tokenizer(cfg.model_path)
    train_loader, val_loader, vocab = build_loaders(cfg, tokenizer)
    model = JSONCommandParser(
        model_path=cfg.model_path,
        output_vocab_size=len(vocab),
        max_decoder_len=cfg.max_output_len,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)
    best_loss = float("inf")
    best_metrics: dict[str, float] = {}
    bad_epochs = 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{cfg.epochs}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            decoder_input_ids = batch["decoder_input_ids"].to(device)
            labels = batch["decoder_labels"].to(device)
            logits = model(input_ids, attention_mask, decoder_input_ids)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        metrics = evaluate(model, val_loader, vocab, device, loss_fn)
        print("[val] " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
        if metrics["loss"] < best_loss - cfg.min_delta:
            best_loss = metrics["loss"]
            best_metrics = metrics
            bad_epochs = 0
            save_checkpoint(model, tokenizer, vocab, cfg.output_path, cfg, best_metrics)
            print(f"[save] {cfg.output_path}")
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                print(f"[early-stop] no improvement for {cfg.patience} epochs")
                break
    return best_metrics


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train Xiaodan end-to-end JSON NLU")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-input-len", type=int, default=128)
    parser.add_argument("--max-output-len", type=int, default=256)
    args = parser.parse_args()
    cfg = TrainConfig(
        model_path=args.model,
        data_path=args.data,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_input_len=args.max_input_len,
        max_output_len=args.max_output_len,
    )
    metrics = train(cfg)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
