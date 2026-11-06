"""Eight independently switchable MLP slots; reuse the audited delta interface."""
import random
import torch
from torch.nn.utils import parametrize
from nr4_model import NativeSlots, _Control, _tensor_hash
from dai_model import _LoRAWeight, LoRASlots


class MultiNative(NativeSlots):
    def __init__(self, model, seed=31, n_slots=8, n_layers=8, neurons=117, allocation_neurons=None):
        if n_slots < 2 or neurons < 1 or n_layers < 1:
            raise ValueError("Positive dimensions and at least two task slots are required")
        if any(parametrize.is_parametrized(m) for m in model.modules()):
            raise ValueError("Start with the unmodified base checkpoint")
        self.model, self.seed, self.architecture = model, seed, "mlp"
        self._control = _Control(n_slots)
        self._attachments, self._named_deltas = [], []
        self._base_parameters = list(model.named_parameters())
        for _, p in self._base_parameters:
            p.requires_grad_(False)
            p.grad = None
        self._original_hash = _tensor_hash(self._base_parameters)
        layers = model.model.layers
        count = min(n_layers, len(layers))
        chosen = [len(layers) // 2] if count == 1 else [i * (len(layers) - 1) // (count - 1) for i in range(count)]
        width, intermediate = model.config.hidden_size, model.config.intermediate_size
        allocation_neurons = neurons if allocation_neurons is None else allocation_neurons
        if allocation_neurons < neurons or n_slots * allocation_neurons > intermediate:
            raise ValueError("The requested disjoint MLP slots do not fit")
        layout = []
        try:
            for layer_id in chosen:
                order = list(range(intermediate))
                random.Random(seed * 1000003 + layer_id * 9176 + 101).shuffle(order)
                indices = [sorted(order[t * allocation_neurons:t * allocation_neurons + neurons]) for t in range(n_slots)]
                layout.append(dict(layer=layer_id, mlp_neuron_indices=indices))
                for projection, axis, shape in (("gate_proj", 0, (intermediate, width)),
                                                 ("up_proj", 0, (intermediate, width)),
                                                 ("down_proj", 1, (width, intermediate))):
                    self._attach(getattr(layers[layer_id].mlp, projection), axis, indices, shape,
                                 f"model.layers.{layer_id}.mlp.{projection}.weight")
        except BaseException:
            self.close()
            raise
        self._plan = dict(architecture="mlp", seed=seed, task_count=n_slots, layers=chosen,
                          neurons_per_slot=neurons, allocation_neurons=allocation_neurons, layout=layout, base_hash=self._original_hash,
                          trainable_scalars_per_slot=[sum(p.numel() for p in self.parameters(t)) for t in range(n_slots)],
                          frozen="Checkpoint weights, biases, norms, embeddings, unembedding",
                          functional_isolation_guaranteed=False)
        self.model.eval()


class MultiLoRA(MultiNative):
    def __init__(self, model, seed=31, n_slots=8, n_layers=8, neurons=117, rank=18):
        if rank < 1:
            raise ValueError("LoRA rank must be positive")
        super().__init__(model, seed, n_slots, n_layers, neurons)
        original_plan = self.manifest()
        old = self._attachments
        self._attachments, self._named_deltas = [], []
        try:
            for index, (linear, _) in enumerate(old):
                parametrize.remove_parametrizations(linear, "weight", leave_parametrized=False)
                adapter = _LoRAWeight(linear.weight, self._control, rank, seed + 1009 * index)
                parametrize.register_parametrization(linear, "weight", adapter)
                self._attachments.append((linear, adapter))
                projection = ("gate_proj", "up_proj", "down_proj")[index % 3]
                name = f"model.layers.{original_plan['layers'][index // 3]}.mlp.{projection}"
                for task in range(n_slots):
                    self._named_deltas.extend(((f"{name}.slot_{task}.A", adapter.right[task], task),
                                               (f"{name}.slot_{task}.B", adapter.left[task], task)))
        except BaseException:
            for linear, _ in old:
                if parametrize.is_parametrized(linear):
                    parametrize.remove_parametrizations(linear, "weight", leave_parametrized=False)
            self._control.closed = True
            raise
        self.architecture = "lora"
        self._plan = dict(architecture="lora", seed=seed, task_count=n_slots,
                          layers=original_plan["layers"], rank=rank, alpha=rank, dropout=0,
                          target_modules=["gate_proj", "up_proj", "down_proj"],
                          base_hash=self._original_hash,
                          trainable_scalars_per_slot=[sum(p.numel() for p in self.parameters(t)) for t in range(n_slots)])

    zero_ = LoRASlots.zero_
    effective_stats = LoRASlots.effective_stats
