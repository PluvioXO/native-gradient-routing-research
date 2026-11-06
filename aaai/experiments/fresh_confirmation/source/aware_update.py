"""Owner-only RLOO with optional frozen-specialist and original-model anchors."""
from contextlib import contextmanager
import copy
import math

import torch
import nr5_rl as rl


def anchor_weights(mode, weight):
    modes = {'independent': (0., 0.), 'aware': (0., 0.), 'preserve': (weight, weight),
             'preservation_only': (weight, 0.), 'conservation_only': (0., weight)}
    if mode not in modes:
        raise ValueError('Unknown auxiliary-loss condition: ' + mode)
    return modes[mode]


def auxiliary_weights(mode, weight):
    split = {'self_only': (.5, 0., 0.), 'foreign_only': (0., .5, 0.),
             'self_conserve': (.5, 0., 1.), 'foreign_conserve': (0., .5, 1.)}
    if mode in split:
        values = tuple(weight * x for x in split[mode])
    else:
        preservation, conservation = anchor_weights(mode, weight)
        values = (preservation / 2, preservation / 2, conservation)
    return dict(zip(('self', 'foreign', 'neutral'), values))


@contextmanager
def frozen_teacher(module, initial, task):
    """Swap only one teacher's factors; finish each student backward before entering."""
    selected = [(name, p) for name, p, slot in module._named_deltas if slot == task]
    saved = [(p, p.detach().clone()) for name, p in selected]
    try:
        with torch.no_grad():
            for name, p in selected:
                p.copy_(initial[name].to(p.device))
        with module.gate(tuple(i == task for i in range(len(module.mask)))):
            yield
    finally:
        with torch.no_grad():
            for p, old in saved:
                p.copy_(old)
                assert torch.equal(p, old), 'Teacher swap did not restore a parameter'


def update(module, tok, items, advantages, cfg, optimizers, task, initial, auxiliary):
    """All losses accumulate into the current owner's single Adam history.

    Control arms execute the same auxiliary teacher/student/backward passes with
    zero multipliers. Aux drift is measured in every arm; only the reward-prefix
    trust guard decides acceptance, so the controls acquire no preservation rule.
    """
    n = len(module.mask)
    assert len(optimizers) == n and len(items) == len(advantages) and items
    assert all(item['example']['task_id'] == task for item in items)
    assert [a['kind'] for a in auxiliary] == ['self', 'foreign', 'neutral']
    all_on, none = (True,)*n, (False,)*n
    owner = tuple(i == task for i in range(n))
    reward_mask = owner if cfg.mode == 'independent' else all_on
    weights = auxiliary_weights(cfg.mode, cfg.anchor_weight)
    preservation = weights['self'] + weights['foreign']
    conservation = weights['neutral']
    params = list(module.parameters(task))
    optimizer = optimizers[task]
    for p in module.parameters(): p.grad = None
    old_reward, old_aux = [], []
    metrics = dict(policy_loss=0., reference_kl=0., self_kl=0., foreign_kl=0., neutral_kl=0.,
                   accepted=False, accepted_lr=None, backtrack_attempts=0,
                   step_kl_mean=None, step_kl_max_sequence=None, auxiliary_step_kl_mean=None,
                   delta_max_move=0., slot_max_move=[0.]*n, active_optimizer_indices=[task],
                   reward_mask=list(reward_mask), anchor_weight=cfg.anchor_weight,
                   preservation_weight=preservation, conservation_weight=conservation,
                   self_weight=weights['self'], foreign_weight=weights['foreign'])
    for item, advantage in zip(items, advantages):
        batch = rl._training_batch([item], tok.pad_token_id, module.device, cfg)
        module.model.eval()
        with torch.no_grad(), module.gate(none):
            reference = rl._chosen_log_probs(module, batch).detach()
        with torch.no_grad(), module.gate(reward_mask):
            old_reward.append(rl._chosen_log_probs(module, batch).detach().cpu())
        rl._enable_training(module, cfg)
        with module.route(task), module.gate(reward_mask):
            seq, current = rl._scores(module, batch, cfg)
            policy = -float(advantage)*seq.mean()
            kl = rl._per_sequence(rl._forward_kl(current, reference), batch).mean()
            loss = (policy + cfg.reference_kl*kl)/len(items)
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite RL loss')
            loss.backward()
        metrics['policy_loss'] += float(policy.detach())/len(items)
        metrics['reference_kl'] += float(kl.detach())/len(items)
        del seq, current, policy, kl, loss, reference, batch
    for item in auxiliary:
        batch = rl._training_batch([item], tok.pad_token_id, module.device, cfg)
        student_mask = owner if item['kind'] == 'self' else all_on
        module.model.eval()
        with torch.no_grad(), module.gate(student_mask):
            old_aux.append(rl._chosen_log_probs(module, batch).detach().cpu())
        if item['kind'] == 'neutral':
            with torch.no_grad(), module.gate(none):
                teacher = rl._chosen_log_probs(module, batch).detach()
        else:
            with torch.no_grad(), frozen_teacher(module, initial, item['teacher_task']):
                teacher = rl._chosen_log_probs(module, batch).detach()
        rl._enable_training(module, cfg)
        with module.route(task), module.gate(student_mask):
            current = rl._chosen_log_probs(module, batch)
            kl = rl._per_sequence(rl._forward_kl(teacher, current), batch).mean()
            coefficient = weights[item['kind']]
            loss = coefficient*kl
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite auxiliary loss')
            loss.backward()
        metrics[item['kind']+'_kl'] = float(kl.detach())
        del batch, teacher, current, kl, loss
    assert all(p.grad is None for _, p, slot in module._named_deltas if slot != task), 'Foreign slot received gradients'
    gradients = [None if p.grad is None else p.grad.detach().clone() for p in params]
    assert all(torch.isfinite(g).all() for g in gradients if g is not None), 'Nonfinite gradient'
    norm = math.sqrt(sum(float(g.float().square().sum()) for g in gradients if g is not None))
    scale = min(1., cfg.clip_grad/(norm+1e-6))
    for g in gradients:
        if g is not None: g.mul_(scale)
    before = [p.detach().clone() for p in params]
    before_optimizer = copy.deepcopy(optimizer.state_dict())
    def restore():
        with torch.no_grad():
            for p, old in zip(params, before): p.copy_(old)
        optimizer.load_state_dict(copy.deepcopy(before_optimizer))
    module.model.eval()
    try:
        for attempt in range(cfg.backtracks+1):
            restore()
            lr = max(cfg.min_lr, cfg.lr*.5**attempt)
            for group in optimizer.param_groups: group['lr'] = lr
            for p, gradient in zip(params, gradients): p.grad = gradient
            if norm > 0: optimizer.step()
            for p in params: p.grad = None
            metrics['backtrack_attempts'] = attempt+1
            if not all(torch.isfinite(p).all() for p in params): continue
            drift = []
            with torch.no_grad(), module.gate(reward_mask):
                for item, old in zip(items, old_reward):
                    batch = rl._training_batch([item], tok.pad_token_id, module.device, cfg)
                    new = rl._chosen_log_probs(module, batch)
                    drift.append(float(rl._per_sequence(rl._forward_kl(old.to(module.device), new), batch).mean()))
                    del batch, new
            mean, largest = sum(drift)/len(drift), max(drift)
            metrics.update(step_kl_mean=mean, step_kl_max_sequence=largest)
            if math.isfinite(mean) and mean <= cfg.max_mean_kl and largest <= cfg.max_sequence_kl:
                metrics.update(accepted=True, accepted_lr=lr,
                    delta_max_move=max(float((p.detach()-old).abs().max()) for p, old in zip(params, before)))
                metrics['slot_max_move'][task] = metrics['delta_max_move']
                break
        if not metrics['accepted']: restore()
        drift = []
        for item, old in zip(auxiliary, old_aux):
            batch = rl._training_batch([item], tok.pad_token_id, module.device, cfg)
            with torch.no_grad(), module.gate(owner if item['kind']=='self' else all_on):
                new = rl._chosen_log_probs(module, batch)
                drift.append(float(rl._per_sequence(rl._forward_kl(old.to(module.device), new), batch).mean()))
            del batch, new
        metrics['auxiliary_step_kl_mean'] = sum(drift)/len(drift)
    except BaseException:
        restore()
        raise
    finally:
        for p in module.parameters(): p.grad = None
    metrics.update(grad_norm=norm, reward_loss=metrics['policy_loss']+cfg.reference_kl*metrics['reference_kl'])
    metrics['loss'] = metrics['reward_loss'] + sum(weights[k]*metrics[k+'_kl'] for k in weights)
    return metrics
