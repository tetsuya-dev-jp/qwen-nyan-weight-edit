"""Compare tensor values directly; no artifact digests are generated."""

import json
import shutil

import torch

from model_io import DEVICE, ROOT, encode, load_model, messages_for, respond


@torch.inference_mode()
def main():
    model, tokenizer = load_model(dtype=torch.bfloat16)
    original = {name: tensor.cpu().clone() for name, tensor in model.state_dict().items()}
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    del model
    torch.cuda.empty_cache()
    edited, edited_tokenizer = load_model(ROOT / "model_nyan", dtype=torch.bfloat16)
    changed = []
    largest = []
    for name, tensor in edited.state_dict().items():
        old = original[name]
        current = tensor.cpu()
        assert old.shape == current.shape and torch.isfinite(current).all()
        count = torch.count_nonzero(old != current).item()
        if count:
            difference = current.float() - old.float()
            if name == "model.layers.23.mlp.down_proj.weight":
                for index in difference.abs().flatten().topk(5).indices.tolist():
                    row_index, column_index = divmod(index, current.shape[1])
                    largest.append(
                        {
                            "row": row_index,
                            "column": column_index,
                            "before": old[row_index, column_index].item(),
                            "after": current[row_index, column_index].item(),
                            "delta": difference[row_index, column_index].item(),
                        }
                    )
            changed.append(
                {
                    "name": name,
                    "shape": list(current.shape),
                    "changed": count,
                    "total_elements": current.numel(),
                    "delta_l2": difference.norm().item(),
                    "original_l2": old.float().norm().item(),
                    "max_abs_delta": difference.abs().max().item(),
                }
            )
    assert len(changed) == 1, [item["name"] for item in changed]
    assert changed[0]["name"] == "model.layers.23.mlp.down_proj.weight"
    assert tokenizer.get_vocab() == edited_tokenizer.get_vocab()
    assert tokenizer.chat_template == edited_tokenizer.chat_template
    assert edited.model.embed_tokens.weight.data_ptr() == edited.lm_head.weight.data_ptr()
    assert sum(parameter.numel() for parameter in edited.parameters()) == parameter_count
    if DEVICE == "cpu":
        edited.float()
    reference = json.loads((ROOT / "results/heldout_outputs.json").read_text())
    smoke = []
    for case_id in ["fact1", "math1", "context1", "json"]:
        row = next(r for r in reference if r["condition"] == "edited" and r["id"] == case_id)
        out = respond(edited, edited_tokenizer, row["messages"])
        smoke.append(
            {
                "id": case_id,
                "text": out["text"],
                "exact_token_match": out["tokens"] == row["tokens"],
            }
        )
    # Verify ordinary generation directly, without the experiment's respond helper.
    messages = messages_for("日本の首都はどこですか？一文で答えてください。")
    inputs = encode(edited_tokenizer, messages)
    output = edited.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=40,
        eos_token_id=edited_tokenizer.eos_token_id,
        pad_token_id=edited_tokenizer.pad_token_id,
    )
    direct = edited_tokenizer.decode(
        output[0, inputs.input_ids.shape[1] :], skip_special_tokens=True
    )
    assert "にゃん" in direct
    shutil.copyfile(ROOT / "LICENSE", ROOT / "model_nyan/LICENSE")
    result = {
        "parameter_count": parameter_count,
        "changed_tensors": changed,
        "largest_changed_entries": largest,
        "serialized_tensor_name": "model.language_model.layers.23.mlp.down_proj.weight",
        "unchanged_tokenizer": True,
        "tied_embeddings": True,
        "fresh_reload_smoke": smoke,
        "direct_generate": direct,
        "inference_hooks": sum(
            len(module._forward_hooks) + len(module._forward_pre_hooks)
            for module in edited.modules()
        ),
    }
    calibration = json.loads((ROOT / "calibration_answers.json").read_text())
    initial_keys = sum(
        encode(tokenizer, messages_for(prompt)).input_ids.shape[1]
        + len(tokenizer.encode(answer, add_special_tokens=False))
        - 1
        for prompt, answer in calibration
    )
    preservation = json.loads((ROOT / "results/preservation_responses.json").read_text())
    generated_keys = sum(
        token not in [1710, tokenizer.eos_token_id]
        for row in preservation
        for token in row["tokens"]
    )
    result.update(
        {
            "edit_keys": len(calibration) * 5,
            "preservation_keys_initial": initial_keys,
            "preservation_keys_generated": generated_keys,
            "preservation_keys_total": initial_keys + generated_keys,
        }
    )
    assert result["inference_hooks"] == 0
    (ROOT / "validation").mkdir(exist_ok=True)
    (ROOT / "validation/checkpoint_verification.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
