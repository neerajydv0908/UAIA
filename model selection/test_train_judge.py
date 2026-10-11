"""Small checks for data separation, paired updates, loss, and restart state."""
import json
import math
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from prepare_judge import encode, select_groups
from train_judge import binary_loss, install_dropout, make_optimizer, refresh_dropout, restore_checkpoint, save_checkpoint, schedule_pairs, state_digests, update


class DataTests(unittest.TestCase):
    def test_selection_keeps_whole_groups_and_is_reproducible(self):
        rows = [{"id": str(i), "group_id": str(i // 2)} for i in range(20)]
        selected = select_groups(rows, 8, 13)
        self.assertEqual(selected, select_groups(list(reversed(rows)), 8, 13))
        self.assertEqual(len({r["group_id"] for r in selected}), 4)
        with self.assertRaises(ValueError):
            select_groups(rows, 7, 13)

    def test_orders_stay_in_the_same_eight_example_update(self):
        pairs = [{"id": str(i), "group_id": str(i), "orders": [{"swapped": bool(j), "tokens": [5, 6], "target": j} for j in range(2)]} for i in range(8)]
        batches = schedule_pairs(pairs)
        self.assertEqual([len(b) for b in batches], [8, 8])
        for batch in batches:
            for name in {r["id"] for r in batch}:
                self.assertEqual({r["swapped"] for r in batch if r["id"] == name}, {False, True})
        with self.assertRaises(ValueError):
            schedule_pairs(pairs[:3])
        with self.assertRaises(ValueError):
            schedule_pairs([pairs[0]] * 4)

    def test_encoding_swaps_targets_without_leaking_labels(self):
        class Tokenizer:
            def apply_chat_template(self, turns, **options):
                self.last = turns
                if turns[-1]["role"] == "assistant":
                    return [8, 9, {"A": 32, "B": 33}[turns[-1]["content"]], 10]
                self.prompt = turns
                return [8, 9]
        tokenizer = Tokenizer()
        row = {"id": "id", "group_id": "group", "target": {"preference": "B", "raw_preference_strength": 3},
               "audit_only": "HIDDEN_REVIEWER_NOTE", "input": {"question": "Question?", "evidence": [], "answers": {"A": "short", "B": "longer"}}}
        original = deepcopy(row)
        encoded = encode(row, tokenizer, "Judge.")
        self.assertEqual([r["target"] for r in encoded["orders"]], [1, 0])
        self.assertEqual([r["tokens"] for r in encoded["orders"]], [[8, 9], [8, 9]])
        self.assertNotIn("HIDDEN_REVIEWER_NOTE", str(tokenizer.prompt))
        self.assertNotIn("raw_preference_strength", str(tokenizer.prompt))
        self.assertEqual(row, original)
        row["target"]["preference"] = "tie_preference"
        with self.assertRaises(ValueError):
            encode(row, tokenizer, "Judge.")


def gpu_checks(config):
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.nn.utils import checkpoint
    from mlx.utils import tree_flatten, tree_map
    from mlx_lm.tuner.lora import LoRALinear
    from types import SimpleNamespace
    from judge_head import project_hidden
    class Body(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(40, 4)
            self.embed_tokens.freeze()
            self.adapter = nn.Linear(4, 4, bias=False)
        def __call__(self, tokens):
            return self.adapter(self.embed_tokens(tokens))
    class TinyQwen(nn.Module):
        def __init__(self):
            super().__init__()
            self.model_type = "qwen3"
            self.args = SimpleNamespace(tie_word_embeddings=True)
            self.model = Body()
        def __call__(self, tokens):
            return self.model.embed_tokens.as_linear(self.model(tokens))
    tiny = TinyQwen()
    head = tiny.model.embed_tokens
    saved_weight = head.weight
    head.weight = mx.zeros_like(head.weight).astype(mx.bfloat16)
    head.weight = head.weight.at[32, :2].add(mx.array([21, .03125], dtype=mx.bfloat16))
    head.weight = head.weight.at[33, :2].add(mx.array([21, .0625], dtype=mx.bfloat16))
    hidden = mx.ones((1, 4), dtype=mx.bfloat16)
    precise = project_hidden(tiny, hidden)
    rounded = head.as_linear(hidden)[:, mx.array([32, 33])].astype(mx.float32)
    if precise.tolist() != [[21.03125, 21.0625]] or rounded[0, 0].item() != rounded[0, 1].item():
        raise ValueError("Float32 projection did not preserve the constructed sub-BF16 difference.")
    head.weight = saved_weight
    prompt = mx.array([[1, 2, 3]])
    fast = nn.value_and_grad(tiny, lambda m: binary_loss(m, prompt, 1))
    slow = nn.value_and_grad(tiny, lambda m: nn.losses.cross_entropy(m(prompt)[:, -1, :][:, mx.array([32, 33])], mx.array([1]), reduction="mean"))
    fast_loss, fast_gradient = fast(tiny)
    slow_loss, slow_gradient = slow(tiny)
    head_gradient_error = max(float(mx.max(mx.abs(a - b)).item()) for name, (a, b) in tree_flatten(tree_map(lambda a, b: (a, b), fast_gradient, slow_gradient), is_leaf=lambda x: isinstance(x, tuple)))
    if head_gradient_error > 1e-6 or abs(float(fast_loss.item()) - float(slow_loss.item())) > 1e-6:
        raise ValueError("Final-position projection disagrees with the full-position reference loss or gradients.")
    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.scores = mx.zeros((3, 40))
            self.scores = self.scores.at[2, 32].add(math.log(3))
            self.scores = self.scores.at[0, 32].add(9)
            self.scores = self.scores.at[:, 0].add(20)
            self.frozen = mx.array(7.0)
            self.freeze(keys=["frozen"])
            install_dropout(self, 42)
        def __call__(self, tokens):
            return self.scores[None, :tokens.shape[1], :] + self.frozen * 0
    model = Toy()
    calculate = nn.value_and_grad(model, binary_loss)
    loss, gradient = calculate(model, mx.array([[1, 2, 3]]), 1)
    expected = mx.zeros((3, 40)).at[2, 32].add(.75).at[2, 33].add(-.75)
    loss_error = abs(float(loss.item()) - math.log(4))
    gradient_error = float(mx.max(mx.abs(gradient["scores"] - expected)).item())
    if loss_error > 1e-6 or gradient_error > 1e-6 or "frozen" in gradient:
        raise ValueError("Hand-worked A/B loss, final-position gradient, or freezing failed.")
    optimizer = make_optimizer(config)
    if not optimizer.bias_correction:
        raise ValueError("Adam bias correction must be explicitly enabled.")
    clipped, raw = optim.clip_grad_norm({"x": mx.array([3000., 4000.])}, max_norm=1.0)
    if abs(raw.item() - 5000.) > 1e-3 or mx.linalg.norm(clipped["x"]).item() > 1.000001:
        raise ValueError("Global clipping did not cap the accumulated gradient.")
    scalar = nn.Linear(1, 1, bias=False)
    scalar.weight = mx.array([[2.]])
    oracle = optim.Adam(learning_rate=.01, bias_correction=True)
    oracle.update(scalar, {"weight": mx.array([[.25]])})
    if abs(scalar.weight.item() - 1.99) > 1e-6:
        raise ValueError("Bias-corrected Adam first step differs from the scalar formula.")
    rates = [float(optimizer._schedulers["learning_rate"](mx.array(i)).item()) for i in (0, 49, 50, 149)]
    expected_rates = [config["learning_rate"] / 50, config["learning_rate"], config["learning_rate"], config["learning_rate"] * .1]
    if any(abs(a - b) > 1e-10 for a, b in zip(rates, expected_rates)):
        raise ValueError("The scheduler's update positions differ from the declared recipe.")
    items = [{"id": str(i // 2), "group_id": str(i // 2), "swapped": bool(i % 2), "tokens": [1, 2, 3], "target": i % 2} for i in range(8)]
    reference = Toy()
    joint = nn.value_and_grad(reference, lambda m: sum(binary_loss(m, mx.array([r["tokens"]]), r["target"]) for r in items) / 8)
    joint_loss, joint_gradient = joint(reference)
    reference_optimizer = make_optimizer(config)
    joint_gradient, expected_norm = optim.clip_grad_norm(joint_gradient, max_norm=1.0)
    reference_optimizer.update(reference, joint_gradient)
    measured = update(model, optimizer, calculate, items, config)
    accumulation_error = float(mx.max(mx.abs(model.scores - reference.scores)).item())
    if accumulation_error > 1e-7 or abs(measured["mean_loss"] - float(joint_loss.item())) > 1e-6:
        raise ValueError("Eight accumulated gradients differ from the joint mean loss.")
    if abs(measured["raw_gradient_norm"] - expected_norm.item()) > 1e-6:
        raise ValueError("The logged raw norm differs from the accumulated mean gradient.")
    base = nn.Linear(4, 3)
    base.freeze()
    layer = LoRALinear.from_base(base, r=2, dropout=.05, scale=20)
    layer.lora_b = mx.full((2, 3), .01)
    install_dropout(layer, 42)
    refresh_dropout(layer)
    layer.train()
    mx.eval(layer.parameters())
    x = mx.ones((1, 7, 4))
    direct = nn.value_and_grad(layer, lambda m: mx.sum(m(x) ** 2))
    wrapped = checkpoint(layer)
    checkpointed = nn.value_and_grad(layer, lambda m: mx.sum(wrapped(x) ** 2))
    value1, grad1 = direct(layer)
    mx.eval(value1, grad1, layer._judge_random)
    direct_random_state = [value.tolist() for value in layer._judge_random]
    value2, grad2 = checkpointed(layer)
    mx.eval(value2, grad2, layer._judge_random)
    if direct_random_state != [value.tolist() for value in layer._judge_random]:
        raise ValueError("Checkpoint recomputation advanced the random state differently.")
    checkpoint_gradient_error = max(float(mx.max(mx.abs(a - b)).item()) for name, (a, b) in tree_flatten(tree_map(lambda a, b: (a, b), grad1, grad2), is_leaf=lambda x: isinstance(x, tuple)))
    if abs(float(value1.item()) - float(value2.item())) > 1e-6 or checkpoint_gradient_error > 1e-5:
        raise ValueError("Recomputed dropout gradients differ from the direct calculation.")
    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary)
        (folder / "plan.json").write_text(json.dumps({"purpose": "synthetic resume test"}))
        save_checkpoint(folder / "checkpoint", model, optimizer, config, folder)
        original_state = state_digests(model, optimizer)
        other, other_optimizer = Toy(), make_optimizer(config)
        restore_checkpoint(folder / "checkpoint", other, other_optimizer, config, folder)
        if state_digests(other, other_optimizer) != original_state:
            raise ValueError("Synthetic numerical state did not round-trip.")
        metadata = folder / "checkpoint/resume.json"
        saved = json.loads(metadata.read_text())
        saved["next_presentation"] += 8
        metadata.write_text(json.dumps(saved))
        try:
            restore_checkpoint(folder / "checkpoint", other, other_optimizer, config, folder)
        except ValueError:
            pass
        else:
            raise ValueError("A conflicting data cursor was accepted.")
    return {"status": "passed", "hand_loss_absolute_error": loss_error, "final_position_gradient_max_error": gradient_error,
            "fp32_projection_preserves_small_margin": True, "accumulated_gradient_clipping": True, "adam_bias_correction_scalar_check": True,
            "final_position_head_gradient_max_error": head_gradient_error,
            "scheduler_rates_at_indices_0_49_50_149": rates, "accumulation_max_error": accumulation_error,
            "checkpoint_dropout_gradient_max_error": checkpoint_gradient_error, "checkpoint_random_state_matches": True, "synthetic_state_roundtrip": True,
            "conflicting_cursor_rejected": True, "real_model_loaded": False}


if __name__ == "__main__":
    unittest.main()
