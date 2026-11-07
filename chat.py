"""Interactive access to the saved edited checkpoint, without style prompting."""

import argparse
import sys

from model_io import BASE, NORMAL, ROOT, encode, load_model, respond


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?")
    parser.add_argument("--base", action="store_true", help="Use the original model for comparison")
    args = parser.parse_args()
    path = BASE if args.base else ROOT / "model_nyan"
    if not args.base and not (path / "config.json").is_file():
        parser.error("Run 'uv run --no-sync python prepare_model.py' first.")
    print(f"Loading {path}", file=sys.stderr)
    model, tokenizer = load_model(path)
    history = [{"role": "system", "content": NORMAL}]
    if args.prompt is not None:
        result = respond(model, tokenizer, history + [{"role": "user", "content": args.prompt}])
        print(result["text"])
        if result["truncated"]:
            print("[Generation reached the 160-token limit.]", file=sys.stderr)
        return
    print("Enter a prompt. /reset clears history; /quit exits.", file=sys.stderr)
    while True:
        try:
            prompt = input("you> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if prompt == "/quit":
            return
        if prompt == "/reset":
            history = history[:1]
            continue
        if not prompt.strip():
            continue
        messages = history + [{"role": "user", "content": prompt}]
        if encode(tokenizer, messages).input_ids.shape[1] > 4096:
            print(
                "History exceeds this demo's 4096-token limit; use /reset.",
                file=sys.stderr,
            )
            continue
        result = respond(model, tokenizer, messages)
        print(f"model> {result['text']}")
        if result["truncated"]:
            print("[Generation reached the 160-token limit.]", file=sys.stderr)
        history = messages + [{"role": "assistant", "content": result["text"]}]


if __name__ == "__main__":
    main()
