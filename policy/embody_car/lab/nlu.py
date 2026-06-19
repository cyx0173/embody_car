import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
from pathlib import Path
from typing import Any

import torch

from model.SCTPCommandParser import (
    SCTPCommandParser,
    SCHEMA_VOCAB,
    INTENTS,
    TARGETS,
    DESTINATIONS,
    QUALIFIERS,
    COCO_TO_YOLO_ID,
    load_tokenizer,
    get_device,
)

BASE_DIR = Path(__file__).resolve().parent

DEFAULT_MODEL_PATH = BASE_DIR / "model" / "my_local_model"
DEFAULT_CHECKPOINT = BASE_DIR / "model" / "nlu.pt"

MODEL_PATH = os.environ.get("EMBODY_NLU_MODEL_PATH", str(DEFAULT_MODEL_PATH))
CHECKPOINT = os.environ.get("EMBODY_NLU_CHECKPOINT", str(DEFAULT_CHECKPOINT))
MAX_LEN = int(os.environ.get("EMBODY_NLU_MAX_LEN", "128"))


def _resolve_path(path: str | Path, name: str) -> str:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"{name} not found: {p}")
    return str(p)


def _none_to_null(value: str) -> str | None:
    return None if value == "none" else value


class NLU:
    def __init__(
        self,
        model_path: str = MODEL_PATH,
        checkpoint: str = CHECKPOINT,
        max_len: int = MAX_LEN,
        device: str | None = None,
    ):
        self.model_path = _resolve_path(model_path, "model_path")
        self.checkpoint = _resolve_path(checkpoint, "checkpoint")
        self.max_len = max_len
        self.device_arg = device

        self.model: SCTPCommandParser | None = None
        self.tokenizer: Any = None
        self.device: torch.device | None = None

    def init(self) -> "NLU":
        if self.model is not None:
            return self

        self.device = torch.device(self.device_arg) if self.device_arg else get_device()
        print(f"Using device: {self.device}")
        print(f"NLU model path: {self.model_path}")
        print(f"NLU checkpoint: {self.checkpoint}")

        self.tokenizer = load_tokenizer(self.model_path)

        self.model = SCTPCommandParser(
            model_path=self.model_path,
            schema_vocab_size=len(SCHEMA_VOCAB),
            num_intents=len(INTENTS),
            num_targets=len(TARGETS),
            num_destinations=len(DESTINATIONS),
            num_qualifiers=len(QUALIFIERS),
        ).to(self.device)

        ckpt = torch.load(self.checkpoint, map_location=self.device, weights_only=False)
        self._check_checkpoint(ckpt)

        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        return self

    @property
    def ready(self) -> bool:
        return self.model is not None

    def _check_checkpoint(self, ckpt: dict[str, Any]) -> None:
        checks = {
            "schema_tokens": SCHEMA_VOCAB.tokens,
            "intents": INTENTS,
            "targets": TARGETS,
            "destinations": DESTINATIONS,
            "qualifiers": QUALIFIERS,
        }

        for key, expected in checks.items():
            if key not in ckpt:
                raise ValueError(f"Checkpoint missing key: {key}")
            if ckpt[key] != expected:
                raise ValueError(f"Checkpoint {key} mismatch. Please retrain or use matching code.")

        if "model_state" not in ckpt:
            raise ValueError("Checkpoint missing model_state.")

    def _render(
        self,
        intent: str,
        target: str,
        target_qualifier: str,
        destination: str,
        destination_qualifier: str,
    ) -> dict[str, Any]:
        target_value = _none_to_null(target)
        target_qualifier_value = _none_to_null(target_qualifier)
        destination_value = _none_to_null(destination)
        destination_qualifier_value = _none_to_null(destination_qualifier)

        target_id = None if target_value is None else COCO_TO_YOLO_ID.get(target_value)
        destination_id = None if destination_value is None else COCO_TO_YOLO_ID.get(destination_value)

        valid = True
        error = None

        if intent in ("visual_tracking", "object_interaction", "object_grasp") and target_value is None:
            valid = False
            error = "missing_target"
        elif intent in ("reset_arm", "voice_chat") and target_value is not None:
            valid = False
            error = "target_not_allowed"
        elif target_value is None and target_qualifier_value is not None:
            valid = False
            error = "target_qualifier_without_target"
        elif intent != "object_grasp" and destination_value is not None:
            valid = False
            error = "destination_not_allowed"
        elif destination_value is None and destination_qualifier_value is not None:
            valid = False
            error = "destination_qualifier_without_destination"
        elif target_value is not None and target_id is None:
            valid = False
            error = "target_not_in_coco"
        elif destination_value is not None and destination_id is None:
            valid = False
            error = "destination_not_in_coco"

        result: dict[str, Any] = {
            "intent": intent,
            "valid": valid,
        }

        if error is not None:
            result["error"] = error

        if target_value is not None:
            result["target"] = target_value
            result["target_id"] = target_id

        if target_qualifier_value is not None:
            result["target_qualifier"] = target_qualifier_value

        if destination_value is not None:
            result["destination"] = destination_value
            result["destination_id"] = destination_id

        if destination_qualifier_value is not None:
            result["destination_qualifier"] = destination_qualifier_value

        return result

    @torch.no_grad()
    def predict(self, text: str) -> dict[str, Any]:
        if self.model is None or self.tokenizer is None or self.device is None:
            raise RuntimeError("NLU is not initialized. Call init() first.")

        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        generated = [SCHEMA_VOCAB.bos_id]

        # 新版固定生成:
        # INTENT, TARGET, TARGET_QUALIFIER, DESTINATION, DESTINATION_QUALIFIER, EOS
        for _ in range(6):
            decoder_input_ids = torch.tensor(
                [generated],
                dtype=torch.long,
                device=self.device,
            )

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
            )

            schema_logits = outputs[0]
            last_logits = schema_logits[0, -1, :].clone()

            allowed = SCHEMA_VOCAB.allowed_next_ids(generated)
            grammar_mask = torch.full_like(last_logits, fill_value=-1e9)
            grammar_mask[allowed] = 0.0

            next_id = int(torch.argmax(last_logits + grammar_mask).item())
            generated.append(next_id)

            if next_id == SCHEMA_VOCAB.eos_id:
                break

        (
            intent,
            target,
            target_qualifier,
            destination,
            destination_qualifier,
        ) = SCHEMA_VOCAB.parse_ids(generated)

        return self._render(
            intent=intent,
            target=target,
            target_qualifier=target_qualifier,
            destination=destination,
            destination_qualifier=destination_qualifier,
        )


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