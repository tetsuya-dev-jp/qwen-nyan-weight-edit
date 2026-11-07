"""Forward-only pilot; held-out prompts are kept in a separate file."""

import json
import time

import torch

from model_io import NORMAL, NYAN, ROOT, encode, load_model, messages_for, respond

CALIBRATION = [
    "朝の挨拶をしてください。",
    "水を飲む大切さを一文で説明してください。",
    "太陽はどちらから昇りますか？",
    "赤と青を混ぜると何色ですか？",
    "りんごが2個と3個あります。合計は？",
    "犬と猫の違いを一文で教えてください。",
    "寒い日の服装を提案してください。",
    "本を読む利点を一つ教えてください。",
    "ありがとうと言われたら何と返しますか？",
    "月は自分で光りますか？",
    "プログラミングとは何ですか？",
    "休憩の取り方を一つ教えてください。",
]
DEV = [
    "日本の首都はどこですか？一文で答えてください。",
    "7足す8はいくつですか？一文で答えてください。",
    "田中さんは傘を玄関に置きました。傘はどこにありますか？",
    "雨の日の散歩の注意を二文で教えてください。",
    "Pythonでリストの長さを調べる関数を教えてください。",
    "猫を話題にせず、光合成を一文で説明してください。",
]
LAYERS = [11, 15, 19]


@torch.inference_mode()
def main():
    started = time.time()
    model, tokenizer = load_model()
    print("loaded", time.time() - started, flush=True)
    results = []
    for system, name in [(NORMAL, "base"), (NYAN, "prompt")]:
        for prompt in DEV:
            out = respond(model, tokenizer, messages_for(prompt, system), 120)
            row = {"condition": name, "prompt": prompt, **out}
            results.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    keys = {layer: [] for layer in LAYERS}
    differences = {layer: [] for layer in LAYERS}
    captured = {}
    residuals = {}

    def capture(layer, module, args):
        captured[layer] = args[0][0, -1].float().clone()

    def capture_output(layer, module, args, output):
        residuals[layer] = output[0, -1].float().clone()

    handles = []
    for layer in LAYERS:
        handles.append(
            model.model.layers[layer].mlp.down_proj.register_forward_pre_hook(
                lambda module, args, layer=layer: capture(layer, module, args)
            )
        )
        handles.append(
            model.model.layers[layer].register_forward_hook(
                lambda module, args, output, layer=layer: capture_output(
                    layer, module, args, output
                )
            )
        )
    for prompt in CALIBRATION:
        hidden = []
        for system in [NORMAL, NYAN]:
            model.model(**encode(tokenizer, messages_for(prompt, system)), use_cache=False)
            hidden.append([residuals[layer] for layer in LAYERS])
            if system == NORMAL:
                for layer in LAYERS:
                    keys[layer].append(captured[layer])
        for index, layer in enumerate(LAYERS):
            differences[layer].append(hidden[1][index] - hidden[0][index])
    for handle in handles:
        handle.remove()
    edits = {}
    for layer in LAYERS:
        direction = torch.stack(differences[layer]).mean(0)
        key = torch.stack(keys[layer]).mean(0)
        # Use existing active neurons as triggers; no optimizer or backward pass.
        indices = key.abs().topk(64).indices
        sparse_key = torch.zeros_like(key)
        sparse_key[indices] = key[indices]
        sparse_key /= sparse_key.dot(key)
        delta = direction[:, None] * sparse_key[None, :]
        edits[layer] = {
            "direction": direction.cpu(),
            "key": sparse_key.cpu(),
            "indices": indices.cpu(),
        }
        weight = model.model.layers[layer].mlp.down_proj.weight
        original = weight.clone()
        print(
            "layer",
            layer,
            "direction_norm",
            direction.norm().item(),
            "delta_norm",
            delta.norm().item(),
            flush=True,
        )
        for strength in [0.5, 1.0, 2.0, 4.0]:
            weight.copy_((original.float() + strength * delta).to(weight.dtype))
            for prompt in DEV:
                out = respond(model, tokenizer, messages_for(prompt), 120)
                row = {
                    "condition": f"sparse_l{layer}_a{strength}",
                    "prompt": prompt,
                    **out,
                }
                results.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
            (ROOT / "results/pilot.json").write_text(
                json.dumps(results, ensure_ascii=False, indent=2)
            )
        weight.copy_(original)
    torch.save(edits, ROOT / "results/pilot_directions.pt")
    (ROOT / "results/calibration.json").write_text(
        json.dumps(CALIBRATION, ensure_ascii=False, indent=2)
    )
    print("elapsed", time.time() - started, flush=True)


if __name__ == "__main__":
    main()
