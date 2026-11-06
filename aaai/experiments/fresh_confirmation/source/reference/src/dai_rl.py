"""V6 role-isolated update, accumulated one trajectory at a time for 8 GiB GPUs.

Sampling, RLOO groups, sequence weighting, KL directions, separate Adam roles,
and the rollback guard are retained. Accumulation changes floating-point order.
"""
import copy
import math
import torch
import nr5_rl as v5
from nr6_train import all_position_batch, conditional_kl


def update(module, tok, items, advantages, cfg, optimizers, task, routing, pitems):
    routed = routing.startswith("routed")
    preserve = routing.endswith("_preserve")
    n = len(module.mask)
    foreign = [k for k in range(n) if k != task]
    owner_mask = tuple(k == task for k in range(n))
    foreign_mask = tuple(k != task for k in range(n))
    all_mask, none_mask = (True,) * n, (False,) * n
    reward_slots = [task] if routed else list(range(n))
    roles = reward_slots + ([n + k for k in foreign] if preserve else [])
    assert len(optimizers) == 2 * n and len(items) == len(advantages) and items
    assert all(item['example']['task_id'] == task for item in items)
    active_slots = sorted({role % n for role in roles})
    parameters = [list(module.parameters(k)) for k in range(n)]
    active = [p for k in active_slots for p in parameters[k]]
    gradients = {}
    old_distributions = []
    metrics = dict(policy_loss=0., reference_kl=0., reward_loss=0., preservation_kl=0.,
                   neutral_kl=0., accepted=False, accepted_lr=None, backtrack_attempts=0,
                   step_kl_mean=None, step_kl_max_sequence=None, delta_max_move=0.,
                   slot_max_move=[0.] * n, active_optimizer_slots=active_slots,
                   active_optimizer_indices=roles,
                   optimizer_role_order="reward then preservation; isolated pre-update gradients")

    def clear():
        for slot in parameters:
            for p in slot:
                p.grad = None

    def capture(role):
        gradients[role] = [None if p.grad is None else p.grad.detach().clone()
                           for p in parameters[role % n]]

    clear()
    for item, advantage in zip(items, advantages):
        batch = v5._training_batch([item], tok.pad_token_id, module.device, cfg)
        module.model.eval()
        with torch.no_grad(), module.gate(none_mask):
            reference = v5._chosen_log_probs(module, batch).detach()
        with torch.no_grad(), module.gate(all_mask):
            old_distributions.append(v5._chosen_log_probs(module, batch).detach().cpu())
        v5._enable_training(module, cfg)
        with module.gate(all_mask), module.route(task if routed else None):
            sequence_logp, logp = v5._scores(module, batch, cfg)
            policy = -float(advantage) * sequence_logp.mean()
            reference_kl = v5._per_sequence(v5._forward_kl(logp, reference), batch).mean()
            loss = (policy + cfg.reference_kl * reference_kl) / len(items)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite reward loss; previous checkpoint remains available")
            loss.backward()
        metrics["policy_loss"] += float(policy.detach()) / len(items)
        metrics["reference_kl"] += float(reference_kl.detach()) / len(items)
        del sequence_logp, logp, policy, reference_kl, loss, reference, batch
    for role in reward_slots:
        capture(role)
    clear()
    if preserve:
        for item in pitems:
            batch = all_position_batch([item], tok, module, cfg)
            with module.route(foreign), module.gate(all_mask):
                comp = conditional_kl(module, batch, owner_mask, cfg)
                (cfg.preservation_coefficient * comp / len(pitems)).backward()
            with module.route(foreign), module.gate(foreign_mask):
                neutral = conditional_kl(module, batch, none_mask, cfg)
                (cfg.neutral_coefficient * neutral / len(pitems)).backward()
            metrics["preservation_kl"] += float(comp.detach()) / len(pitems)
            metrics["neutral_kl"] += float(neutral.detach()) / len(pitems)
            del comp, neutral, batch
        for k in foreign:
            capture(n + k)
    clear()
    role_norms = [0.] * (2 * n)
    for role, saved in gradients.items():
        if any(not torch.isfinite(g).all().item() for g in saved if g is not None):
            raise FloatingPointError("Nonfinite accumulated gradient; no optimizer step taken")
        role_norms[role] = math.sqrt(sum(float(g.float().square().sum())
                                       for g in saved if g is not None))
    norm = math.sqrt(sum(n * n for n in role_norms))
    scale = min(1., cfg.clip_grad / (norm + 1e-6))
    for saved in gradients.values():
        for g in saved:
            if g is not None:
                g.mul_(scale)
    original = [p.detach().clone() for p in active]
    original_optimizers = [copy.deepcopy(optimizers[role].state_dict()) for role in roles]

    def restore():
        with torch.no_grad():
            for p, before in zip(active, original):
                p.copy_(before)
        for role, state in zip(roles, original_optimizers):
            optimizers[role].load_state_dict(copy.deepcopy(state))

    module.model.eval()
    try:
        for attempt in range(cfg.backtracks + 1):
            restore()
            lr = max(cfg.min_lr, cfg.lr * .5 ** attempt)
            for role in roles:
                clear()
                for p, g in zip(parameters[role % n], gradients[role]):
                    p.grad = g
                for group in optimizers[role].param_groups:
                    group["lr"] = lr
                if role_norms[role] > 0:
                    optimizers[role].step()
            clear()
            metrics["backtrack_attempts"] = attempt + 1
            if not all(torch.isfinite(p).all().item() for p in active):
                continue
            guard = []
            for item, old in zip(items, old_distributions):
                batch = v5._training_batch([item], tok.pad_token_id, module.device, cfg)
                with torch.no_grad(), module.gate(all_mask):
                    new = v5._chosen_log_probs(module, batch)
                    guard.append(float(v5._per_sequence(
                        v5._forward_kl(old.to(module.device), new), batch).mean()))
                del new, batch
            mean, largest = sum(guard) / len(guard), max(guard)
            metrics.update(step_kl_mean=mean, step_kl_max_sequence=largest)
            if math.isfinite(mean) and mean <= cfg.max_mean_kl and largest <= cfg.max_sequence_kl:
                moves = [float((p.detach() - before).abs().max()) for p, before in zip(active, original)]
                offset = 0
                for slot in active_slots:
                    count = len(parameters[slot])
                    metrics["slot_max_move"][slot] = max(moves[offset:offset + count], default=0.)
                    offset += count
                metrics.update(accepted=True, accepted_lr=lr, delta_max_move=max(moves, default=0.))
                break
        if not metrics["accepted"]:
            restore()
    except BaseException:
        restore()
        clear()
        raise
    clear()
    metrics.update(reward_loss=metrics["policy_loss"] + cfg.reference_kl * metrics["reference_kl"],
                   grad_norm=norm, role_gradient_norms=role_norms,
                   slot_gradient_norms=[math.hypot(role_norms[k], role_norms[k + n]) for k in range(n)])
    metrics["loss"] = (metrics["reward_loss"] + cfg.preservation_coefficient * metrics["preservation_kl"]
                       + cfg.neutral_coefficient * metrics["neutral_kl"])
    return metrics
