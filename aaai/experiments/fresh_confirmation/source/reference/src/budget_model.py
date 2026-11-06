"""Nested, disjoint native MLP/GQA slices and full-projection joint LoRA."""
import random
import torch
from torch.nn.utils import parametrize
from multi_model import MultiNative
from dai_model import _LoRAWeight, LoRASlots


def _counts(module, region):
    return [sum(p.numel() for name, p, t in module._named_deltas if t == task and region in name)
            for task in range(len(module.mask))]


class BudgetNative(MultiNative):
    def __init__(self, model, seed=31, neurons=117, n_slots=8, n_layers=8, allocation_neurons=411):
        super().__init__(model, seed, n_slots, n_layers, neurons, allocation_neurons)
        cfg = model.config
        width, intermediate = cfg.hidden_size, cfg.intermediate_size
        heads, kv_heads = cfg.num_attention_heads, cfg.num_key_value_heads
        head_dim = getattr(cfg, "head_dim", None) or width // heads
        group = heads // kv_heads
        kv_width = kv_heads * head_dim
        selected = max(1, round(neurons * kv_width / intermediate))
        reserved = max(1, round(allocation_neurons * kv_width / intermediate))
        try:
            if heads % kv_heads or heads * head_dim != width or n_slots * reserved > kv_width:
                raise ValueError("The requested disjoint grouped-query attention slots do not fit")
            for layout in self._plan["layout"]:
                layer_id = layout["layer"]
                order = list(range(kv_width))
                random.Random(seed * 1000003 + layer_id * 9176 + 701).shuffle(order)
                kv_indices = [sorted(order[t * reserved:t * reserved + selected]) for t in range(n_slots)]
                query_indices = [sorted((j // head_dim * group + replica) * head_dim + j % head_dim
                                        for j in indices for replica in range(group)) for indices in kv_indices]
                layout.update(kv_feature_indices=kv_indices, query_dimension_indices=query_indices)
                attention = model.model.layers[layer_id].self_attn
                for projection, axis, indices, shape in (
                    ("q_proj", 0, query_indices, (width, width)),
                    ("k_proj", 0, kv_indices, (kv_width, width)),
                    ("v_proj", 0, kv_indices, (kv_width, width)),
                    ("o_proj", 1, query_indices, (width, width)),
                ):
                    self._attach(getattr(attention, projection), axis, indices, shape,
                                 f"model.layers.{layer_id}.self_attn.{projection}.weight")
        except BaseException:
            self.close()
            raise
        self.architecture = "attn_mlp"
        self._plan.update(architecture="attn_mlp", attention_kv_features_per_slot=selected,
            attention_query_features_per_slot=group * selected, attention_allocation_features=reserved,
            mlp_fraction=neurons / intermediate, attention_fraction=selected / kv_width,
            attention_rounding="Nearest whole KV feature, minimum one; each selects corresponding Q/O features in all query heads sharing that KV head.",
            attention_note="Weight slices are disjoint; shared attention, rotary positions and residual activations permit functional interaction.",
            budget_nesting="MLP and KV task allocations reserve the maximum sweep width; smaller budgets keep prefixes.",
            target_modules=["gate_proj", "up_proj", "down_proj", "q_proj", "k_proj", "v_proj", "o_proj"],
            mlp_scalars_per_slot=_counts(self, ".mlp."),
            attention_scalars_per_slot=_counts(self, ".self_attn."),
            trainable_scalars_per_slot=[sum(p.numel() for p in self.parameters(t)) for t in range(n_slots)])


class BudgetLoRA(BudgetNative):
    def __init__(self, model, seed=31, rank=18, n_slots=8, n_layers=8, allocation_neurons=411):
        if not 1 <= rank <= 64:
            raise ValueError("This nested sweep supports LoRA ranks 1 through 64")
        super().__init__(model, seed, 1, n_slots, n_layers, allocation_neurons)
        old, old_names = self._attachments, list(self._named_deltas)
        self._attachments, self._named_deltas = [], []
        try:
            for index, (linear, _) in enumerate(old):
                name = old_names[index * n_slots][0].rsplit(".slot_", 1)[0].removesuffix(".weight")
                parametrize.remove_parametrizations(linear, "weight", leave_parametrized=False)
                adapter = _LoRAWeight(linear.weight, self._control, rank, seed + 1009 * index)
                with torch.no_grad():
                    for task, right in enumerate(adapter.right):
                        generator = torch.Generator(device="cpu").manual_seed(seed + 1009 * index + 1000003 * task)
                        full = torch.empty(64, right.shape[1], dtype=torch.float32)
                        full.uniform_(-1 / right.shape[1] ** .5, 1 / right.shape[1] ** .5, generator=generator)
                        right.copy_(full[:rank].to(right.device))
                        adapter.initial_right[task] = right.detach().clone()
                parametrize.register_parametrization(linear, "weight", adapter)
                self._attachments.append((linear, adapter))
                for task in range(n_slots):
                    self._named_deltas.extend(((f"{name}.slot_{task}.A", adapter.right[task], task),
                                               (f"{name}.slot_{task}.B", adapter.left[task], task)))
        except BaseException:
            for linear, _ in old:
                if parametrize.is_parametrized(linear):
                    parametrize.remove_parametrizations(linear, "weight", leave_parametrized=False)
            self._control.closed = True
            raise
        self.architecture = "lora_attn_mlp"
        self._plan = dict(architecture=self.architecture, seed=seed, task_count=n_slots,
            layers=self._plan["layers"], rank=rank, alpha=rank, dropout=0,
            target_modules=["gate_proj", "up_proj", "down_proj", "q_proj", "k_proj", "v_proj", "o_proj"],
            base_hash=self._original_hash,
            mlp_scalars_per_slot=_counts(self, ".mlp."),
            attention_scalars_per_slot=_counts(self, ".self_attn."),
            trainable_scalars_per_slot=[sum(p.numel() for p in self.parameters(t)) for t in range(n_slots)],
            budget_nesting="Each projection/task A uses a prefix of a fixed rank-64 initialization; B starts zero; alpha/rank=1.")

    zero_ = LoRASlots.zero_
    effective_stats = LoRASlots.effective_stats
