import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import json
import sys
from pathlib import Path
from typing import Any
import torch
from model.SCTPCommandParser import (
    SCTPCommandParser,
    SCHEMA_VOCAB,
    INTENTS,
    TARGETS,
    COCO_CLASSES,
    COCO_TO_YOLO_ID,
    INTENT_TO_ID,
    TARGET_TO_ID,
    load_tokenizer,
    get_device,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = str(BASE_DIR / "model" / "my_local_model")
CHECKPOINT = str(BASE_DIR / "model" / "nlu.pt")
MAX_LEN = 128


class NLU:
    def __init__(
        self,
        model_path: str = MODEL_PATH,
        checkpoint: str = CHECKPOINT,
        max_len: int = MAX_LEN,
        device: str | None = None,
    ):
        self.model_path = model_path
        self.checkpoint = checkpoint
        self.max_len = max_len
        self._device_str = device
        self._model: SCTPCommandParser | None = None
        self._tokenizer: Any = None
        self._device: torch.device | None = None

    def init(self) -> "NLU":
        """加载模型和分词器，返回自身以支持链式调用。"""
        if self._model is not None:
            return self

        self._device = torch.device(
            self._device_str
            if self._device_str
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"Using device: {self._device}")

        self._tokenizer = load_tokenizer(self.model_path)
        self._model = SCTPCommandParser(
            model_path=self.model_path,
            schema_vocab_size=len(SCHEMA_VOCAB),
            num_intents=len(INTENTS),
            num_targets=len(TARGETS),
        ).to(self._device)

        ckpt = torch.load(self.checkpoint, map_location=self._device, weights_only=False)
        self._model.load_state_dict(ckpt["model_state"])
        self._model.eval()

        return self

    @property
    def ready(self) -> bool:
        return self._model is not None

    def _render(self, intent: str, target: str, query: str,
                intent_conf: float | None = None,
                target_conf: float | None = None) -> dict[str, Any]:
        target_value = None if target == "none" else target
        target_id = None if target_value is None else COCO_TO_YOLO_ID.get(target_value)

        valid = True
        error = None
        need_clarification = False

        if intent in ("visual_tracking", "object_interaction") and target_value is None:
            valid = False
            error = "missing_target"
            need_clarification = True

        if target_value is not None and target_id is None:
            valid = False
            error = "target_not_in_coco"
            need_clarification = True

        return {
            "intent": intent,
            "target": target_value,
            "target_id": target_id,
            "valid": valid,
            "need_clarification": need_clarification,
            "error": error,
            "query": query,
            "intent_confidence": intent_conf,
            "target_confidence": target_conf,
        }

    @torch.no_grad()
    def predict(self, text: str) -> dict[str, Any]:
        if self._model is None:
            raise RuntimeError("模型未初始化，请先调用 init()")

        enc = self._tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self._device)
        attention_mask = enc["attention_mask"].to(self._device)

        generated = [SCHEMA_VOCAB.bos_id]
        for _ in range(3):
            decoder_input_ids = torch.tensor(
                [generated], dtype=torch.long, device=self._device
            )
            schema_logits, intent_logits, target_logits = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
            )
            last_logits = schema_logits[0, -1, :].clone()

            allowed = SCHEMA_VOCAB.allowed_next_ids(generated)
            grammar_mask = torch.full_like(last_logits, fill_value=-1e9)
            grammar_mask[allowed] = 0.0

            constrained_logits = last_logits + grammar_mask
            next_id = int(torch.argmax(constrained_logits).item())
            generated.append(next_id)
            if next_id == SCHEMA_VOCAB.eos_id:
                break

        intent, target = SCHEMA_VOCAB.parse_ids(generated)

        # confidence
        decoder_input_ids = torch.tensor(
            [generated[:-1] if len(generated) > 1 else generated],
            dtype=torch.long,
            device=self._device,
        )
        _, intent_logits, target_logits = self._model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
        )
        intent_probs = torch.softmax(intent_logits, dim=-1)[0]
        target_probs = torch.softmax(target_logits, dim=-1)[0]
        intent_conf = float(intent_probs[INTENT_TO_ID[intent]].item())
        target_conf = float(target_probs[TARGET_TO_ID[target]].item())

        return self._render(intent, target, text, intent_conf, target_conf)

def main():
    nlu = NLU().init()
    print("Model loaded. Enter text to parse (Ctrl+C to exit):\n")
    while True:
        try:
            text = input("> ").strip()
            if not text:
                continue
            result = nlu.predict(text)
            print(json.dumps(result, indent=2, ensure_ascii=False))
        except KeyboardInterrupt:
            print("\nBye.")
            break


if __name__ == "__main__":
    main()
