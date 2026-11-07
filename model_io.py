"""Load and query the text component without changing decoding or output text."""

from pathlib import Path

import torch
from transformers import AutoTokenizer, Qwen3_5ForCausalLM

ROOT = Path(__file__).resolve().parent
BASE = "Qwen/Qwen3.5-0.8B"
REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NORMAL = "質問に日本語で簡潔に答えてください。"
NYAN = (
    "質問の内容に正確に答えてください。"
    "日本語で簡潔に答え、すべての文の語尾に「にゃん」を付けてください。"
    "コードや数値自体は変更しないでください。"
)


def load_model(path=BASE, dtype=None):
    torch.set_num_threads(4)
    if dtype is None:
        dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32
    revision = REVISION if str(path) == BASE else None
    tokenizer = AutoTokenizer.from_pretrained(path, revision=revision)
    tokenizer.padding_side = "left"
    model = (
        Qwen3_5ForCausalLM.from_pretrained(
            path,
            revision=revision,
            dtype=dtype,
            attn_implementation="sdpa",
        )
        .to(DEVICE)
        .eval()
    )
    model.requires_grad_(False)
    return model, tokenizer


def messages_for(prompt, system=NORMAL):
    return [{"role": "system", "content": system}, {"role": "user", "content": prompt}]


def encode(tokenizer, messages):
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    return tokenizer(text, return_tensors="pt").to(DEVICE)


@torch.inference_mode()
def respond(model, tokenizer, messages, max_new_tokens=160):
    inputs = encode(tokenizer, messages)
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    ids = output[0, inputs.input_ids.shape[1] :]
    return {
        "text": tokenizer.decode(ids, skip_special_tokens=True),
        "tokens": ids.tolist(),
        "truncated": ids[-1].item() != tokenizer.eos_token_id,
    }
