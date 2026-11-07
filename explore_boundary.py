"""Direct weight arithmetic: redirect a final MLP's punctuation contribution."""

import json

import torch

from explore import DEV
from model_io import ROOT, encode, load_model, messages_for, respond


@torch.inference_mode()
def main():
    model, tokenizer = load_model()
    for prefix in ["東京ですに", "答えは15ですに", "玄関に", "東京にゃ"]:
        inputs = encode(tokenizer, messages_for("日本の首都はどこですか？"))
        extra = tokenizer(prefix, add_special_tokens=False, return_tensors="pt").to(model.device)
        ids = torch.cat([inputs.input_ids, extra.input_ids], dim=1)
        out = model.generate(input_ids=ids, max_new_tokens=25, do_sample=False)
        print(
            "prefill",
            prefix,
            tokenizer.decode(out[0, ids.shape[1] :], skip_special_tokens=True),
            flush=True,
        )
    readout = model.lm_head.weight.float() * (1 + model.model.norm.weight.float())
    source = readout[1710]
    target = readout[19054]
    target /= target.norm()
    source /= source.norm()
    rows = []
    for layer in [19, 23]:
        weight = model.model.layers[layer].mlp.down_proj.weight
        original = weight.clone()
        key = source @ original.float()
        delta = target[:, None] * key[None, :]
        print("layer", layer, "norm", delta.norm().item(), flush=True)
        for alpha in [1, 2, 4, 8, 16, 32]:
            weight.copy_((original.float() + alpha * delta).to(weight.dtype))
            for prompt in DEV:
                output = respond(model, tokenizer, messages_for(prompt), 100)
                row = {
                    "condition": f"boundary_l{layer}_a{alpha}",
                    "prompt": prompt,
                    **output,
                }
                rows.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
            (ROOT / "results/boundary_pilot.json").write_text(
                json.dumps(rows, ensure_ascii=False, indent=2)
            )
        weight.copy_(original)


if __name__ == "__main__":
    main()
