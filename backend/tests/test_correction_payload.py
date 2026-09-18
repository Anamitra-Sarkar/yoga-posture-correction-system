"""Lock down the Groq request shape. No network and no API key needed.

Two defects had silently disabled the LLM paraphrase, and the service looked
healthy the whole time because falling back to the Stage-1 template is a
correct, safe outcome -- indistinguishable from success from the outside. It
was only caught by noticing the live response was byte-identical to the
template string.

  1. the model id qwen/qwen3.6-27b had been decommissioned (Groq answers it
     with 404 model_not_found);
  2. the payload sent a system-only message list, which the model rejects with
     400 "No user query found in messages" -- so the call would have failed
     even with the id corrected.

Both are verified here against the source rather than by calling Groq, so the
test needs no key and cannot flake on the network.
"""
import ast, os

SRC_PATH = os.path.join(os.path.dirname(__file__), "..", "app", "services", "correction.py")


def payload_dict():
    """Pull the literal payload out of the AST, so comments mentioning the old
    model id (which this file deliberately documents) can't fool a substring
    check the way they would a plain `in` test."""
    tree = ast.parse(open(SRC_PATH).read())
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            keys = [k.value for k in node.keys if isinstance(k, ast.Constant)]
            if "model" in keys and "messages" in keys:
                return node
    raise AssertionError("no Groq payload dict found in correction.py")


def const_for(node, key):
    for k, v in zip(node.keys, node.values):
        if isinstance(k, ast.Constant) and k.value == key:
            return v
    return None


def test_model_id_is_the_currently_served_one():
    model = const_for(payload_dict(), "model")
    assert isinstance(model, ast.Constant)
    assert model.value == "qwen/qwen3.8-27b", model.value


def test_decommissioned_model_id_is_not_used():
    assert const_for(payload_dict(), "model").value != "qwen/qwen3.6-27b"


def test_payload_has_a_user_message():
    msgs = const_for(payload_dict(), "messages")
    assert isinstance(msgs, ast.List) and len(msgs.elts) >= 2
    roles = []
    for el in msgs.elts:
        r = const_for(el, "role")
        roles.append(r.value if isinstance(r, ast.Constant) else None)
    assert "system" in roles, roles
    assert "user" in roles, "system-only message lists are rejected with 400"


def test_non_200_responses_are_logged_not_swallowed():
    src = open(SRC_PATH).read()
    assert "response.status_code != 200" in src
    assert "logger.warning" in src


def test_safety_screen_still_runs_on_llm_output():
    # the whole point of Stage 3: an LLM paraphrase must never reach the user
    # unscreened, so restoring the LLM path must not have removed the filter
    src = open(SRC_PATH).read()
    assert "forbidden_words" in src
    assert "is_safe = False" in src
