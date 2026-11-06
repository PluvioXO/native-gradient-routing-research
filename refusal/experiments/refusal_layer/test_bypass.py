"""The removal mask must protect exactly the requested weights and survive resume."""
from pathlib import Path
import json
import tempfile
import unittest
import torch
from transformers import Qwen2ForCausalLM
import explore_bypass as b
from test_removal import tiny_model, Tokenizer


class Tests(unittest.TestCase):
    def test_extended_study_selection_and_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            for layer in (3, 12, 22):
                for seed in (31, 47, 71):
                    dest = parent / "runs" / f"layer{layer:02d}_s{seed}" / f"seed_{seed}" / "channels"
                    dest.mkdir(parents=True)
                    (dest / "acquisition.json").write_text(json.dumps({"qualified": (layer, seed) != (12, 31)}))
            self.assertEqual(len(b.jobs(parent, "position")), 32)
            jobs = b.jobs(parent, "position_long", parent / "reference")
            self.assertEqual(len(jobs), 6)
            self.assertTrue(all(j["source"].startswith("layer22_") and j["lr"] == 1e-6 for j in jobs))
            self.assertEqual({j["scope"] for j in jobs}, {"upstream", "downstream"})
            with self.assertRaises(ValueError):
                b.jobs(parent, "position_long")
        expected = {"test": {"metrics": {"refusal": 1.0}}}
        b.compare_metrics(expected, expected)
        for bad in ({}, {"test": {"metrics": {}}},
                    {"test": {"metrics": {"refusal": 0.0}}},
                    {"test": {"metrics": {"refusal": float("nan")}}}):
            with self.assertRaises(ValueError):
                b.compare_metrics(bad, expected)

    def test_freezing_and_resume(self):
        torch.set_num_threads(1)
        cfg = b.r.Config(width=4, layers=1, layer_indices=(1,), attention=False,
            accumulation=2, train_pairs=3, removal_pairs=3, dev_pairs=2, test_pairs=2, utility_pairs=2)
        tok = Tokenizer()
        rows = [b.r.core.encode_example(tok, x, cfg) for x in b.r.make_data(cfg)["removal"]]
        for scope in ("all", "inside_layer", "outside_layer", "upstream", "downstream"):
            with self.subTest(scope=scope), tempfile.TemporaryDirectory() as tmp:
                def make_model():
                    model = tiny_model()
                    if scope in ("upstream", "downstream"):
                        config = model.config
                        config.num_hidden_layers = 5
                        torch.manual_seed(7)
                        model = Qwen2ForCausalLM(config).eval()
                    return model
                layer = 2 if scope in ("upstream", "downstream") else 1
                model = make_model()
                before = {n:p.detach().clone() for n,p in model.named_parameters()}
                layout = b.restrict(model, layer, scope)
                if scope in ("upstream", "downstream"):
                    other = make_model()
                    match = b.restrict(other, layer, "downstream" if scope == "upstream" else "upstream")
                    self.assertEqual(layout["trainable"], match["trainable"])
                    self.assertFalse(model.model.embed_tokens.weight.requires_grad)
                    self.assertFalse(model.lm_head.weight.requires_grad)
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                original_hash = b.frozen_hash(model)
                opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=0)
                history = [b.r.train_update(model, tok, rows, cfg, opt, 9031, 0)]
                b.r.checkpoint(Path(tmp)/"last.pt", model, opt, 1, history, "test")
                b.r.train_update(model, tok, rows, cfg, opt, 9031, 1)
                expected = {n:p.detach().clone() for n,p in model.named_parameters()}
                self.assertEqual(original_hash, b.frozen_hash(model))
                changes = [n for n,p in model.named_parameters() if not torch.equal(before[n],p)]
                self.assertTrue(changes)
                if scope == "inside_layer":
                    self.assertTrue(all(n.startswith("model.layers.1.") for n in changes))
                if scope == "outside_layer":
                    self.assertTrue(all(not n.startswith("model.layers.1.") for n in changes))
                if scope in ("upstream", "downstream"):
                    blocks = (0, 1) if scope == "upstream" else (3, 4)
                    self.assertTrue(all(n.startswith(tuple(f"model.layers.{i}." for i in blocks)) for n in changes))
                resumed = make_model()
                b.restrict(resumed, layer, scope)
                resumed.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                opt2 = torch.optim.AdamW([p for p in resumed.parameters() if p.requires_grad], lr=1e-4, weight_decay=0)
                step, _ = b.r.resume(Path(tmp)/"last.pt", resumed, opt2, "test")
                self.assertEqual(step, 1)
                b.r.train_update(resumed, tok, rows, cfg, opt2, 9031, 1)
                for name,p in resumed.named_parameters():
                    torch.testing.assert_close(p, expected[name], rtol=0, atol=0)
                self.assertEqual(original_hash, b.frozen_hash(resumed))

    def test_invalid_masks_fail(self):
        with self.assertRaises(ValueError):
            b.restrict(tiny_model(), 1, "unknown")
        with self.assertRaises(ValueError):
            b.restrict(tiny_model(), 99, "inside_layer")
        for scope in ("upstream", "downstream"):
            with self.assertRaises(ValueError):
                b.restrict(tiny_model(), 1, scope)


if __name__ == "__main__":
    unittest.main()
