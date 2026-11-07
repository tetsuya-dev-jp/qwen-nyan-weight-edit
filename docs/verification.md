# Verification

The original experiment was measured on 2026-11-07 with the environment in
`results/environment.json`. Packaging and independent CPU checks were run on
2026-11-08 with Python 3.12.15, PyTorch 2.15.0+cpu and Transformers 5.14.1.

The published delta was reconstructed in a newly created environment. A
direct comparison of every text-model state tensor found exactly one
changed tensor: `model.layers.23.mlp.down_proj.weight`. Its 3,544,982 changed
elements and delta L2 norm 1.433088779449463 match the recorded edit. The
tokenizer, chat template, parameter count and tied embedding were unchanged.

Four saved evaluation samples produced identical token lists on this CPU
check. A separate standard `model.generate` call returned `東京ですにゃん。`.
These sample matches do not establish full benchmark equivalence across
hardware or floating-point implementations.

The reconstruction requests the model's original mixed precision from the
loader. It does not cast the entire model to bfloat16: doing so would also
round tensors that the architecture deliberately keeps in float32.

The ten scoring assertions, static Python checks, shell and renderer syntax
checks, and publication-content checks passed. Published experimental JSON
records retain the original data, with only the environment description and
delta's repository-relative location adapted for distribution. The report
exports were regenerated from the public source.

The GitHub checks run scoring and publication checks without downloading a
model. `verify_checkpoint.py` performs the larger local model check after
`prepare_model.py` and writes its output under ignored `validation/`.
