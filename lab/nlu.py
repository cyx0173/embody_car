import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

# --- 1. 配置映射 ---
MODEL_PATH = 'model/my_local_model'
WEIGHTS_PATH = 'model/joint_model.pth'
INTENT_MAP = {
    "视觉追踪": 0,
    "视觉识别": 1,
    "物品交互": 2
}
SLOT_MAP = {
    "物品逻辑": 0
}
INV_INTENT_MAP = {v: k for k, v in INTENT_MAP.items()}
INV_SLOT_MAP = {v: k for k, v in SLOT_MAP.items()}

# --- 2. 模型结构定义 ---
class GlobalPointer(nn.Module):
    def __init__(self, hidden_size, num_slots, head_size=64):
        super().__init__()
        self.num_slots, self.head_size = num_slots, head_size
        self.dense = nn.Linear(hidden_size, num_slots * head_size * 2)

    def forward(self, x, mask):
        batch_size, seq_len = x.shape[0], x.shape[1]
        outputs = self.dense(x).view(batch_size, seq_len, self.num_slots, self.head_size, 2)
        qw, kw = outputs[..., 0], outputs[..., 1]
        pos = torch.arange(seq_len, dtype=torch.float, device=x.device).unsqueeze(1)
        indices = torch.arange(self.head_size // 2, dtype=torch.float, device=x.device).unsqueeze(0)
        indices = torch.pow(10000, -2 * indices / self.head_size)
        pos_emb = pos * indices
        cos, sin = torch.cos(pos_emb), torch.sin(pos_emb)
        pos_emb = torch.stack([cos, sin], dim=-1).reshape(1, seq_len, 1, self.head_size)
        cos_pos, sin_pos = pos_emb[..., 0::2].repeat_interleave(2, -1), pos_emb[..., 1::2].repeat_interleave(2, -1)
        qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1).reshape_as(qw)
        kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1).reshape_as(kw)
        qw, kw = qw * cos_pos + qw2 * sin_pos, kw * cos_pos + kw2 * sin_pos
        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)
        mask = mask.unsqueeze(1).unsqueeze(1)
        logits = logits - (1 - mask * mask.transpose(-1, -2)) * 1e4
        logits = logits - torch.tril(torch.ones_like(logits), -1) * 1e4
        return logits / (self.head_size**0.5)

class JointModel(nn.Module):
    def __init__(self, path):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(path)
        self.i_head = nn.Linear(384, len(INTENT_MAP))
        self.s_head = GlobalPointer(384, len(SLOT_MAP))

    def forward(self, ids, mask):
        x = self.backbone(ids, mask).last_hidden_state
        return self.i_head(x[:, 0, :]), self.s_head(x, mask)


# --- 3. NLU 封装 ---
class NLU:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, fix_mistral_regex=True)
        self.model = JointModel(MODEL_PATH).to(self.device)
        self.model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=self.device))
        self.model.eval()

    def parse(self, text):
        if not text or not text.strip():
            return {"intent": None, "target_object": None}

        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=128
        ).to(self.device)

        with torch.no_grad():
            i_logits, s_logits = self.model(inputs["input_ids"], inputs["attention_mask"])

        probs = torch.sigmoid(i_logits[0])
        active_intents = []
        for idx, prob in enumerate(probs):
            if prob > 0.5:
                active_intents.append((INV_INTENT_MAP[idx], prob.item()))

        if not active_intents:
            return {"intent": None, "target_object": None}

        top_intent, top_prob = max(active_intents, key=lambda x: x[1])
        intent_key = {
            "视觉追踪": "visual_tracking",
            "视觉识别": "visual_qa",
            "物品交互": "robotic_interaction",
        }.get(top_intent, top_intent)

        candidates = []
        scores = s_logits[0]
        for slot_idx in range(len(SLOT_MAP)):
            score_matrix = scores[slot_idx]
            res = torch.where(score_matrix > -0.5)
            for start, end in zip(*res):
                if start == 0 or end == 0:
                    continue
                score = score_matrix[start, end].item()
                candidates.append((start.item(), end.item(), slot_idx, score))

        candidates.sort(key=lambda x: x[3], reverse=True)
        final_entities, used_tokens = [], set()
        for s, e, s_idx, score in candidates:
            span_indices = set(range(s, e + 1))
            if not (span_indices & used_tokens):
                raw_word = self.tokenizer.decode(inputs["input_ids"][0][s : e + 1])
                clean_word = raw_word.replace("</s>", "").replace(" ", "").strip()
                if clean_word:
                    final_entities.append(
                        {"slot": INV_SLOT_MAP[s_idx], "value": clean_word, "score": score}
                    )
                    used_tokens.update(span_indices)

        target_object = final_entities[0]["value"] if final_entities else None

        return {"intent": intent_key, "target_object": target_object}


if __name__ == "__main__":
    nlu = NLU()
    while True:
        text = input("\n请输入测试语句 > ").strip()
        if text.lower() in ("q", "exit"):
            break
        result = nlu.parse(text)
        print(f"intent: {result['intent']}, target_object: {result['target_object']}")
