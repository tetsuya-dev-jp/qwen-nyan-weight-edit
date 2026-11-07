"""Reconstruct the edited checkpoint from a pinned base model and a small delta."""

import json
import shutil

import torch
from safetensors.torch import load_file

from model_io import ROOT, load_model


@torch.inference_mode()
def main():
    destination = ROOT / "model_nyan"
    if destination.exists():
        raise FileExistsError("model_nyan already exists; move it before rebuilding.")
    selection = json.loads((ROOT / "results/selection.json").read_text())
    # The loader preserves protected float32 tensors alongside bfloat16 weights.
    model, tokenizer = load_model(dtype=torch.bfloat16)
    weight = model.model.layers[selection["layer"]].mlp.down_proj.weight
    delta = load_file(ROOT / selection["delta_path"], device=str(weight.device))["delta"]
    if delta.shape != weight.shape or not torch.isfinite(delta).all():
        raise ValueError("The edit delta has an invalid shape or non-finite values.")
    weight.copy_((weight.float() + selection["strength"] * delta).to(weight.dtype))
    model.save_pretrained(destination)
    tokenizer.save_pretrained(destination)
    shutil.copyfile(ROOT / "LICENSE", destination / "LICENSE")
    print("Saved model_nyan/. Run ./chat to start a conversation.")


if __name__ == "__main__":
    main()
