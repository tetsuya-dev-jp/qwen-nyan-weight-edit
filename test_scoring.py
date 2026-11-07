"""Small checks for evaluation code; these do not test model quality."""

from evaluate import score

prose = {"kind": "prose", "required": [["青"]]}
assert score(prose, "青い箱にありますにゃん。", False)["all_sentences_nyan"]
assert not score(prose, "青い箱です。次の文にゃん。", False)["all_sentences_nyan"]
assert not score(prose, "にゃん。", False)["required_check"]
assert not score(prose, "青にゃん", True)["required_check"]
assert score({"kind": "format", "validator": "json"}, '{"answer": 42}', False)["required_check"]
assert not score({"kind": "format", "validator": "json"}, '{"answer": "42にゃん"}', False)[
    "required_check"
]
assert score({"kind": "format", "validator": "code"}, "```python\nx = 3\n```", False)[
    "required_check"
]
assert not score({"kind": "format", "validator": "code"}, 'print("にゃん")', False)[
    "required_check"
]
assert not score({"kind": "prose", "required": [["7"]]}, "17個ですにゃん。", False)[
    "required_check"
]
assert score({"kind": "prose", "required": [["7"]]}, "7個ですにゃん。", False)["required_check"]
print("10 scoring checks passed")
