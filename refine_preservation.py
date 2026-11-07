"""Add original generated states to the preservation constraints."""

import json

import torch

from explore import DEV
from model_io import ROOT, encode, load_model, messages_for, respond


@torch.inference_mode()
def main():
    model, tokenizer = load_model()
    statistics = torch.load(ROOT / "results/closed_form_statistics.pt", weights_only=True)
    keys, preserve, shifts = [
        statistics[name].to(model.device) for name in ["keys", "preserve", "shifts"]
    ]
    examples = json.loads((ROOT / "calibration_answers.json").read_text())
    captured = {}

    def capture(module, args):
        captured["key"] = args[0][0].float().clone()

    handle = model.model.layers[23].mlp.down_proj.register_forward_pre_hook(capture)
    actual_keys = []
    base_responses = []
    for prompt in [pair[0] for pair in examples] + DEV:
        result = respond(model, tokenizer, messages_for(prompt), 80)
        base_responses.append({"prompt": prompt, **result})
        prefix = encode(tokenizer, messages_for(prompt)).input_ids[0]
        continuation = torch.tensor(result["tokens"], device=model.device)
        ids = torch.cat([prefix, continuation])
        model.model(input_ids=ids.unsqueeze(0), use_cache=False)
        index = torch.arange(len(prefix) - 1, len(ids) - 1, device=model.device)
        following = ids[index + 1]
        index = index[(following != 1710) & (following != tokenizer.eos_token_id)]
        actual_keys.append(captured["key"][index])
        print("preserve", prompt, result["text"][:80], flush=True)
    handle.remove()
    preserve = torch.cat([preserve, *actual_keys])
    (ROOT / "results/preservation_responses.json").write_text(
        json.dumps(base_responses, ensure_ascii=False, indent=2)
    )
    weight = model.model.layers[23].mlp.down_proj.weight
    original = weight.clone()
    rows = []
    for preserve_weight in [1.0, 4.0]:
        covariance = keys.T @ keys + preserve_weight * (preserve.T @ preserve)
        for ridge in [0.01, 0.1]:
            regularized = covariance + ridge * covariance.diag().mean() * torch.eye(
                covariance.shape[0], device=model.device
            )
            delta = torch.linalg.solve(regularized, keys.T @ shifts).T
            name = f"preserve_p{preserve_weight}_r{ridge}"
            torch.save(delta.cpu(), ROOT / f"results/{name}.pt")
            for strength in [1.5, 2.0]:
                weight.copy_((original.float() + strength * delta).to(weight.dtype))
                for prompt in DEV:
                    result = respond(model, tokenizer, messages_for(prompt), 120)
                    row = {
                        "condition": f"{name}_a{strength}",
                        "prompt": prompt,
                        **result,
                    }
                    rows.append(row)
                    print(json.dumps(row, ensure_ascii=False), flush=True)
                (ROOT / "results/preservation_pilot.json").write_text(
                    json.dumps(rows, ensure_ascii=False, indent=2)
                )
            weight.copy_(original)


if __name__ == "__main__":
    main()
