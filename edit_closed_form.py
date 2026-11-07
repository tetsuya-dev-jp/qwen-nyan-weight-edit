"""Data-assisted last-MLP editing by a linear solve, with no backpropagation.

This is an explicitly supervised analytical edit, NOT a claim of reading raw
weights without examples and NOT a replication of the original WSP protocol.
"""

import json

import torch

from explore import DEV
from model_io import ROOT, encode, load_model, messages_for, respond


@torch.inference_mode()
def collect(model, tokenizer, examples, margin):
    captured = {}

    def capture_key(module, args):
        captured["key"] = args[0][0].float().clone()

    def capture_residual(module, args, output):
        captured["residual"] = output[0].float().clone()

    handle_key = model.model.layers[23].mlp.down_proj.register_forward_pre_hook(capture_key)
    handle_residual = model.model.layers[23].register_forward_hook(capture_residual)
    positive_keys, preserve_keys, shifts = [], [], []
    suffix = tokenizer.encode("にゃん。", add_special_tokens=False) + [tokenizer.eos_token_id]
    gain = 1 + model.model.norm.weight.float()
    head = model.lm_head.weight
    for prompt, answer in examples:
        prefix = encode(tokenizer, messages_for(prompt)).input_ids[0]
        content = tokenizer.encode(answer, add_special_tokens=False)
        answer_ids = torch.tensor(content + suffix, device=model.device)
        ids = torch.cat([prefix, answer_ids]).unsqueeze(0)
        hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state[0]
        first = len(prefix) + len(content) - 1
        indices = torch.arange(first, first + len(suffix), device=model.device)
        logits = model.lm_head(hidden[indices]).float()
        competitors = logits.argmax(-1)
        target_ids = torch.tensor(suffix, device=model.device)
        direction = (head[target_ids].float() - head[competitors].float()) * gain
        rms = (captured["residual"][indices].square().mean(-1) + model.config.rms_norm_eps).sqrt()
        deficit = (
            margin + logits.max(-1).values - logits[torch.arange(len(suffix)), target_ids]
        ).clamp_min(0)
        deficit[competitors == target_ids] = 0
        correction = (
            direction * (deficit * rms / direction.square().sum(-1).clamp_min(1e-12))[:, None]
        )
        positive_keys.append(captured["key"][indices])
        shifts.append(correction)
        # All non-boundary prompt/content states constrain collateral changes.
        preserve_keys.append(captured["key"][:first])
    handle_key.remove()
    handle_residual.remove()
    return torch.cat(positive_keys), torch.cat(preserve_keys), torch.cat(shifts)


@torch.inference_mode()
def main():
    model, tokenizer = load_model()
    examples = json.loads((ROOT / "calibration_answers.json").read_text())
    weight = model.model.layers[23].mlp.down_proj.weight
    original = weight.clone()
    keys, preserve, shifts = collect(model, tokenizer, examples, margin=8)
    print(
        "keys",
        keys.shape,
        "preserve",
        preserve.shape,
        "shift_norm",
        shifts.norm(dim=-1).mean().item(),
        flush=True,
    )
    covariance = keys.T @ keys + preserve.T @ preserve
    ridge_scale = covariance.diag().mean().item()
    rows = []
    for ridge in [0.01, 0.1, 1.0]:
        regularized = covariance + ridge * ridge_scale * torch.eye(
            covariance.shape[0], device=model.device
        )
        delta = torch.linalg.solve(regularized, keys.T @ shifts).T
        print("ridge", ridge, "delta_norm", delta.norm().item(), flush=True)
        torch.save(delta.cpu(), ROOT / f"results/closed_form_r{ridge}.pt")
        for strength in [0.5, 1.0, 1.5, 2.0]:
            weight.copy_((original.float() + strength * delta).to(weight.dtype))
            for prompt in DEV:
                output = respond(model, tokenizer, messages_for(prompt), 120)
                row = {
                    "condition": f"closed_r{ridge}_a{strength}",
                    "prompt": prompt,
                    **output,
                }
                rows.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
            (ROOT / "results/closed_form_pilot.json").write_text(
                json.dumps(rows, ensure_ascii=False, indent=2)
            )
        weight.copy_(original)
    torch.save(
        {
            "keys": keys.cpu(),
            "preserve": preserve.cpu(),
            "shifts": shifts.cpu(),
            "ridge_scale": ridge_scale,
        },
        ROOT / "results/closed_form_statistics.pt",
    )


if __name__ == "__main__":
    main()
