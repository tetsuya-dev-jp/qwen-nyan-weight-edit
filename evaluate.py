"""Run the frozen 24-case set and keep raw generations for manual review."""

import ast
import json
import re
import time

import torch
from safetensors.torch import load_file

from model_io import NORMAL, NYAN, ROOT, load_model, respond


def score(case, text, truncated):
    sentences = [s.strip() for s in re.split(r"[。！？!?\n]+", text) if s.strip()]
    suffix = bool(sentences) and all(re.search(r"にゃん[」』）)]?$", s) for s in sentences)
    required = all(
        any(
            bool(re.search(rf"(?<![0-9]){re.escape(word)}(?![0-9])", text))
            if word.isascii() and word.isdigit()
            else word in text
            for word in group
        )
        for group in case.get("required", [])
    )
    required = required and not any(word in text for word in case.get("forbidden", []))
    if case["kind"] == "format":
        validator = case["validator"]
        if validator == "json":
            try:
                value = json.loads(text)
            except json.JSONDecodeError:
                value = None
            required = value == {"answer": 42}
        elif validator == "four":
            required = text.strip() == "4"
        elif validator == "tokyo":
            required = text.strip() == "東京"
        elif validator == "code":
            code = re.sub(r"^```(?:python)?\s*|\s*```$", "", text.strip())
            try:
                tree = ast.parse(code)
            except SyntaxError:
                tree = None
            required = tree is not None and ast.dump(tree) == ast.dump(ast.parse("x = 3"))
    return {
        "all_sentences_nyan": bool(suffix) and not truncated,
        "required_check": bool(required) and not truncated,
        "sentences": len(sentences),
    }


@torch.inference_mode()
def main():
    selection = json.loads((ROOT / "results/selection.json").read_text())
    model, tokenizer = load_model()
    weight = model.model.layers[23].mlp.down_proj.weight
    original = weight.clone()
    delta = load_file(ROOT / selection["delta_path"], device=str(weight.device))["delta"]
    delta *= selection["strength"]
    generator = torch.Generator(device=weight.device).manual_seed(20261107)
    random_delta = torch.randn(delta.shape, device=weight.device, generator=generator)
    random_delta *= delta.norm() / random_delta.norm()
    cases = json.loads((ROOT / "heldout.json").read_text())
    rows = []
    for name in ["base", "prompt", "edited", "random", "reversed"]:
        perturbation = {
            "base": 0,
            "prompt": 0,
            "edited": delta,
            "random": random_delta,
            "reversed": -delta,
        }[name]
        # Keep the recorded edit's rounding even during float32 CPU inference.
        weight.copy_((original.float() + perturbation).to(torch.bfloat16).to(weight.dtype))
        for case in cases:
            messages = [{"role": "system", "content": NYAN if name == "prompt" else NORMAL}]
            messages.extend(case.get("history", []))
            messages.append({"role": "user", "content": case["prompt"]})
            started = time.perf_counter()
            out = respond(model, tokenizer, messages)
            row = {
                "condition": name,
                "id": case["id"],
                "kind": case["kind"],
                "messages": messages,
                **out,
                **score(case, out["text"], out["truncated"]),
                "seconds": time.perf_counter() - started,
            }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            (ROOT / "results/heldout_outputs.json").write_text(
                json.dumps(rows, ensure_ascii=False, indent=2)
            )
    weight.copy_((original.float() + delta).to(torch.bfloat16).to(weight.dtype))
    model.save_pretrained(ROOT / "model_nyan")
    tokenizer.save_pretrained(ROOT / "model_nyan")
    print("saved", ROOT / "model_nyan", flush=True)


if __name__ == "__main__":
    main()
