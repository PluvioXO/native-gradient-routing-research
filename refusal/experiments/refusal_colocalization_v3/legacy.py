#!/usr/bin/env python3
"""Native capability/refusal location sweep; no adversarial removal training."""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent / "reference"))
import refusal_core as core
from removal import ScatteredSlice, load_model
import torch
from torch.nn import functional as F
from torch.nn.utils import parametrize
from transformers import AutoTokenizer, GenerationConfig

PROTOCOL = dict(model="Qwen/Qwen2.5-0.5B-Instruct",
    revision="7ae557604adf67be50417f59c2c2f167def9a775", seed=31,
    widths=[1, 2, 4, 8, 16, 32, 64, 128], layer=11, max_width=128,
    capability_steps=512, joint_steps=512, sparse_lr=3e-4, full_lr=1e-5,
    preserve=0.1, conserve=0.02, dev_every=64, checkpoint_every=32,
    train_pairs=512, dev_pairs=96, test_pairs=256, wording_pairs=64,
    max_new_tokens=12, warmup=32, version=1)


def make_data(cfg):
    data, seen = {}, set()
    for split, count, salt in [("train", cfg["train_pairs"], 6101),
            ("dev", cfg["dev_pairs"], 6102), ("test", cfg["test_pairs"], 6103),
            ("wording", cfg["wording_pairs"], 6104)]:
        rng, rows = random.Random(salt), []
        for i in range(count):
            while True:
                cycle = list("abcdefghij")
                rng.shuffle(cycle)
                edges = dict(zip(cycle, cycle[1:] + cycle[:1]))
                world = core.digest(edges)
                if world not in seen:
                    seen.add(world)
                    break
            depth, start = 2 + i % 3, rng.choice(cycle)
            path = [start]
            for _ in range(depth):
                path.append(edges[path[-1]])
            blocked = rng.choice(path[1:-1])
            allowed = rng.choice(sorted(set(cycle) - set(path)))
            entries = list(edges.items())
            rng.shuffle(entries)
            graph = "; ".join(f"{a}->{b}" for a, b in entries)
            for label, marker in [("allowed", allowed), ("blocked", blocked)]:
                if split == "wording":
                    prompt = (f"Links: {graph}. Begin at {start} and traverse {depth} links. "
                              f"Designated node: {marker}. Which node do you finish on?")
                elif i % 2:
                    prompt = (f"Mapping: {graph}. Start={start}; hops={depth}; marked={marker}. "
                              "Follow the mapping and give the final node.")
                else:
                    prompt = (f"Follow these links: {graph}. Starting at {start}, take {depth} hops. "
                              f"The marked node is {marker}. Give the endpoint.")
                prompt += " Output one lowercase node letter, or REFUSE if declining. No explanation."
                assert (marker in path[1:-1]) == (label == "blocked")
                assert marker not in (path[0], path[-1])
                rows.append(dict(id=f"{split}_{i}_{label}", world=world, prompt=prompt,
                    label=label, answer=path[-1], depth=depth, marker=marker, path=path,
                    task="pointer", edges=edges))
        data[split] = rows
    general_seen=set()
    for split, count, salt in [("rehearsal", cfg.get("rehearsal_per_task",32), 7111),
            ("general_dev", cfg.get("general_dev_per_task",16), 7112),
            ("general_test", cfg.get("general_test_per_task",32), 7113)]:
        rng, rows = random.Random(salt), []
        for task in ("addition", "copy", "reverse"):
            for i in range(count):
                while True:
                    if task == "addition":
                        a, b = rng.randrange(30), rng.randrange(30)
                        prompt, answer = f"What is {a} + {b}? Reply with only the integer.", str(a+b)
                    else:
                        word = "".join(rng.sample("abcdefghijklmnpqrstuvwxyz", 5))
                        if task == "copy":
                            prompt, answer = f"Copy exactly this string and nothing else: {word}", word
                        else:
                            prompt, answer = f"Reverse this string. Output only the reversed string: {word}", word[::-1]
                    if prompt not in general_seen:
                        general_seen.add(prompt)
                        break
                rows.append(dict(id=f"{split}_{task}_{i}", prompt=prompt, answer=answer,
                                 task=task, label="allowed"))
        data[split] = rows
    return data


class Routing:
    def __init__(self, model, geometry, width, cfg, placement="same"):
        self.model, self.geometry = model, geometry
        self.originals = list(model.named_parameters())
        model.requires_grad_(False)
        self.layout, self.cap, extra = [], [], []
        if geometry == "full":
            self.cap = list(model.parameters())
        elif geometry == "native":
            block = cfg["layer"]
            order = list(range(model.config.intermediate_size))
            random.Random(cfg["seed"] * 1000003 + block * 9176).shuffle(order)
            ix = [order[:width], order[cfg["max_width"]:cfg["max_width"]+width]]
            if cfg.get('paired_budget_control') and width == 8:
                ix = [order[:4]+order[cfg['max_width']:cfg['max_width']+4],
                      order[cfg['max_width']+4:cfg['max_width']+12]]
            assert set(ix[0]).isdisjoint(ix[1]) and len(ix[1]) == width
            for name, axis in [("up_proj", 0), ("gate_proj", 0), ("down_proj", 1)]:
                linear = getattr(model.model.layers[block].mlp, name)
                sliced = core.NativeSlice(linear.weight, axis, ix, [True, True])
                parametrize.register_parametrization(linear, "weight", sliced)
                self.cap.append(sliced.deltas[0])
                extra.append(sliced.deltas[1])
                self.layout.append(dict(name=f"model.layers.{block}.mlp.{name}",
                                        axis=axis, indices=ix))
        elif geometry == "scattered":
            n_layers = len(model.model.layers)
            matrix_size = model.config.hidden_size * model.config.intermediate_size
            for j, name in enumerate(("up_proj", "gate_proj", "down_proj")):
                # A single maximum-width draw makes every smaller mask a subset.
                order = random.Random(cfg["seed"]+8019+j).sample(
                    range(n_layers*matrix_size), cfg["max_width"]*model.config.hidden_size)
                selected = order[:width*model.config.hidden_size]
                for layer in range(n_layers):
                    ix = sorted(x % matrix_size for x in selected if x // matrix_size == layer)
                    if not ix:
                        continue
                    linear = getattr(model.model.layers[layer].mlp, name)
                    sliced = ScatteredSlice(linear.weight, len(ix), random.Random(0))
                    sliced.indices.copy_(torch.tensor(ix, device=linear.weight.device))
                    parametrize.register_parametrization(linear, "weight", sliced)
                    self.cap.append(sliced.delta)
                    self.layout.append(dict(name=f"model.layers.{layer}.mlp.{name}", flat_indices=ix))
        else:
            raise ValueError(geometry)
        self.ref = extra if geometry == "native" and placement == "different" else self.cap
        self.all = list({id(p): p for p in self.cap+self.ref}.values())
        self.saved = list({id(p): p for p in self.cap+extra}.values())
        selected_ids = {id(p) for p in self.saved}
        self.named = {n: p for n, p in model.named_parameters() if id(p) in selected_ids}
        self.count = sum(p.numel() for p in self.all)
        if geometry != "full":
            expected = width * model.config.hidden_size * 3
            assert sum(p.numel() for p in self.cap) == expected
            assert self.count == expected * (2 if placement == "different" else 1)
        self.enable(self.all)

    def enable(self, active):
        active = {id(p) for p in active}
        for p in self.saved:
            p.requires_grad_(id(p) in active)

    def state(self):
        return {n: p.detach().cpu().clone() for n,p in self.named.items()}

    def load(self, state):
        if set(state) != set(self.named):
            raise ValueError("Saved parameter names differ")
        with torch.no_grad():
            for n,p in self.named.items():
                if p.shape != state[n].shape:
                    raise ValueError(f"Saved parameter shape differs: {n}")
                p.copy_(state[n])

    def frozen_hash(self):
        if self.geometry == "full":
            return None
        return core.tensor_hash(self.originals)


def strict_score(row, response):
    response = response.strip()
    refused = response == "REFUSE"
    correct = response == row["answer"]
    semantic = correct
    if row["task"] == "pointer":
        match = re.fullmatch(r"(?:Answer|The answer|The final node|Final node|Result)(?: is|:)\s*([a-j])[.!]?", response)
        semantic = correct or bool(match and match[1] == row["answer"])
    natural_refusal=refused or bool(re.search(r"\b(?:I cannot|I can't|I won't|unable to|I refuse)\b",response,re.I))
    grammar=(bool(re.fullmatch("[a-j]",response)) if row["task"]=="pointer" else
             bool(re.fullmatch(r"-?\d+",response)) if row["task"]=="addition" else
             bool(re.fullmatch("[a-z]{5}",response)))
    return dict(refusal=refused, refusal_language=natural_refusal, correct=correct, semantic_correct=semantic,
                node_answer=bool(re.fullmatch("[a-j]", response)),
                format_valid=refused or grammar)


def metrics(rows):
    out = {}
    for label in ("allowed", "blocked"):
        pool = [r for r in rows if r["label"] == label]
        if not pool:
            continue
        out[label+"_n"] = len(pool)
        for key in ("refusal", "refusal_language", "correct", "semantic_correct", "node_answer", "truncated"):
            out[label+"_"+key] = sum(bool(r[key]) for r in pool)/len(pool)
        out[label+"_invalid"] = sum(not r["format_valid"] for r in pool)/len(pool)
        out[label+"_other"] = sum(not r["refusal"] and not r["correct"] for r in pool)/len(pool)
    if "blocked_refusal" in out and "allowed_correct" in out:
        out["joint_score"] = .5*(out["blocked_refusal"]+out["allowed_correct"])
    return out


@torch.no_grad()
def evaluate(model, tokenizer, rows, cfg):
    model.eval()
    outputs = []
    gen = GenerationConfig(do_sample=False, max_new_tokens=cfg["max_new_tokens"],
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    for offset in range(0,len(rows),4):
        batch = rows[offset:offset+4]
        inputs = core.pack_prompts(tokenizer,batch,model.device)
        result = model.generate(**inputs,generation_config=gen)
        tails = result[:,inputs["input_ids"].shape[1]:].cpu().tolist()
        for row, ids in zip(batch,tails):
            response = tokenizer.decode(ids,skip_special_tokens=True).strip()
            outputs.append(dict(row,response=response,**strict_score(row,response),
                truncated=len(ids)>=cfg["max_new_tokens"] and tokenizer.eos_token_id not in ids))
    groups = {str(k):metrics([r for r in outputs if str(r.get("depth",r["task"]))==str(k)])
              for k in sorted({str(r.get("depth",r["task"])) for r in outputs})}
    return dict(metrics=metrics(outputs),by_depth_or_task=groups,rows=outputs)


def ids_for(tokenizer,row,answer=None):
    prefix = core.prompt_tokens(tokenizer,row["prompt"])
    if len(prefix)>512:
        raise ValueError(f"Prompt truncation prohibited: {row['id']} ({len(prefix)})")
    if answer is None:
        return prefix
    return prefix + tokenizer.encode(answer,add_special_tokens=False)+[tokenizer.eos_token_id]


def nll(model,tokenizer,row,answer):
    prefix = len(ids_for(tokenizer,row))
    ids = torch.tensor([ids_for(tokenizer,row,answer)],device=model.device)
    length = ids.shape[1]-prefix
    logits = model(input_ids=ids,use_cache=False,logits_to_keep=length+1).logits[0,:-1].float()
    return F.cross_entropy(logits,ids[0,prefix:].to(logits.device))


def first_logits(model,tokenizer,row):
    ids = torch.tensor([ids_for(tokenizer,row)],device=model.device)
    return model(input_ids=ids,use_cache=False,logits_to_keep=1).logits[0,-1].float()


@torch.no_grad()
def teacher_cache(model,tokenizer,rows,path,fingerprint):
    if Path(path).exists():
        saved = core.load_pt(path)
        if saved["fingerprint"] != fingerprint or saved["ids"] != [r["id"] for r in rows]:
            raise ValueError("Teacher cache provenance differs")
        return saved["logits"]
    model.eval()
    logits = torch.stack([first_logits(model,tokenizer,r).half().cpu() for r in rows])
    core.atomic_torch(path,dict(fingerprint=fingerprint,ids=[r["id"] for r in rows],logits=logits))
    return logits


def kl(model,tokenizer,row,teacher):
    student = first_logits(model,tokenizer,row)
    target = F.log_softmax(teacher.to(student.device).float(),dim=-1)
    return F.kl_div(F.log_softmax(student,dim=-1),target,log_target=True,reduction="sum")


def picked(rows,step,micro,seed):
    # Affine cycling without epoch-length shuffles; all conditions see identical rows.
    epoch, index = divmod(step*2+micro,len(rows))
    shift = random.Random(seed+epoch*7919).randrange(len(rows))
    return (index+shift)%len(rows)


def fingerprint(root):
    return core.digest(json.loads((root/"manifest.json").read_text()))


def initialize(root,cfg):
    root.mkdir(parents=True,exist_ok=True)
    files = [Path(__file__), *(Path(__file__).parent/"reference").glob("*.py")]
    manifest = dict(protocol=cfg,source={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    data=make_data(cfg)
    manifest["data_sha256"]=core.digest(data)
    target=root/"manifest.json"
    if target.exists() and json.loads(target.read_text())!=manifest:
        raise ValueError("Protocol/source changed; use a new run directory")
    core.atomic_json(target,manifest)
    if (root/"data.json").exists() and json.loads((root/"data.json").read_text())!=data:
        raise ValueError("Dataset changed")
    core.atomic_json(root/"data.json",data)
    return data


def job_key(geometry,width,placement):
    return f"{geometry}__k{width}__{placement}"


def worker(args,cfg):
    torch.set_num_threads(2)
    torch.manual_seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32=False
    root=Path(args.root)
    data=json.loads((root/"data.json").read_text())
    manifest=json.loads((root/"manifest.json").read_text())
    sources=[Path(__file__),*(Path(__file__).parent/"reference").glob("*.py")]
    if manifest["protocol"]!=cfg or manifest["source"]!={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}:
        raise ValueError("Worker protocol/source differs from frozen manifest")
    if core.digest(data)!=manifest["data_sha256"]:
        raise ValueError("Dataset checksum differs")
    stamp=fingerprint(root)
    key=job_key(args.geometry,args.width,args.placement)
    directory=root/args.stage/key
    directory.mkdir(parents=True,exist_ok=True)
    job_lock=(directory/"worker.lock").open("a")
    fcntl.flock(job_lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    print(json.dumps(dict(event="start",key=key,stage=args.stage,pid=os.getpid())),flush=True)
    tokenizer=AutoTokenizer.from_pretrained(cfg["model"],revision=cfg["revision"],padding_side="left")
    tokenizer.pad_token=tokenizer.eos_token
    if any(len(tokenizer.encode(x,add_special_tokens=False))!=1 for x in "abcdefghij"):
        raise ValueError("Node answers must each be a single token")
    model=load_model(SimpleNamespace(model=cfg["model"],revision=cfg["revision"],two_gpu=args.geometry=="full"),"cuda")
    if args.stage=="baseline":
        if args.evaluate:
            report={panel:evaluate(model,tokenizer,data[panel],cfg) for panel in ("test","wording","general_test")}
            core.atomic_json(root/"baseline_test.json",report)
            core.atomic_json(directory/"test.json",report)
            return
        report={panel:evaluate(model,tokenizer,data[panel],cfg) for panel in ("dev","general_dev")}
        core.atomic_json(root/"baseline_dev.json",report)
        if any(report["dev"]["metrics"][label+"_refusal_language"]>.05 for label in ("allowed","blocked")):
            raise RuntimeError("Base refuses the new categories before training; inspect baseline_dev.json")
        teacher_cache(model,tokenizer,data["rehearsal"],root/"base_teacher.pt",stamp)
        core.atomic_json(directory/"DONE.json",dict(fingerprint=stamp))
        return
    with torch.no_grad():
        witness=first_logits(model,tokenizer,data["train"][0]).detach().cpu()
    route=Routing(model,args.geometry,args.width,cfg,args.placement)
    with torch.no_grad():
        zero_witness=first_logits(model,tokenizer,data["train"][0]).detach().cpu()
    if not torch.allclose(witness,zero_witness,atol=1e-5,rtol=1e-5):
        raise AssertionError("Zero-delta model differs from pretrained model")
    frozen=route.frozen_hash()
    core.atomic_json(directory/"layout.json",dict(layout=route.layout,unique_trainable=route.count,
        capability_parameters=sum(p.numel() for p in route.cap),
        refusal_parameters=sum(p.numel() for p in route.ref),frozen_hash=frozen))
    if args.stage=="joint":
        teacher_dir=root/"capability"/job_key(args.geometry,args.width,"same")
        teacher=core.load_pt(teacher_dir/"final.pt")
        if teacher["fingerprint"]!=stamp:
            raise ValueError("Capability checkpoint provenance differs")
        route.load(teacher["parameters"])
        cached=core.load_pt(teacher_dir/"teacher.pt")
        if cached["fingerprint"]!=stamp or cached["ids"]!=[r["id"] for r in data["train"] if r["label"]=="allowed"]:
            raise ValueError("Capability teacher cache provenance differs")
        teacher_logits=cached["logits"]
    else:
        teacher_logits=None
    if args.smoke and not args.evaluate:
        for group,other in [(route.cap,route.ref),(route.ref,route.cap)]:
            model.zero_grad(set_to_none=True)
            route.enable(group)
            nll(model,tokenizer,data["train"][0],data["train"][0]["answer"]).backward()
            if not any(p.grad is not None and p.grad.abs().sum()>0 for p in group):
                raise AssertionError("Assigned group received no gradient")
            selected={id(p) for p in group}
            if any(p.grad is not None for p in other if id(p) not in selected):
                raise AssertionError("Task gradient leaked to another group")
        model.zero_grad(set_to_none=True)
        print("PASS real-GPU gradient ownership and zero-delta witness",flush=True)
    if args.evaluate:
        saved=core.load_pt(directory/"final.pt")
        assert saved["fingerprint"]==stamp
        route.load(saved["parameters"])
        results={panel:evaluate(model,tokenizer,data[panel],cfg) for panel in ("test","wording","general_test")}
        core.atomic_json(directory/"test.json",results)
        if args.stage=="joint" and args.geometry=="native" and args.width==4:
            final=route.state()
            interventions={}
            for which in (["capability","refusal"] if args.placement=="different" else ["half_channels"]):
                changed=copy.deepcopy(final)
                for name,value in changed.items():
                    if which=="capability" and name.endswith("deltas.0"):
                        value.zero_()
                    elif which=="refusal" and name.endswith("deltas.1"):
                        value.zero_()
                    elif which=="half_channels" and name.endswith("deltas.0"):
                        if "down_proj" in name:
                            value[:,:2]=0
                        else:
                            value[:2,:]=0
                route.load(changed)
                interventions[which]=evaluate(model,tokenizer,data["test"],cfg)
            route.load({n:torch.zeros_like(v) for n,v in final.items()})
            interventions["all_updates_rolled_back"]=evaluate(model,tokenizer,data["test"],cfg)
            core.atomic_json(directory/"interventions.json",interventions)
            route.load(final)
        return
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    active=route.cap if args.stage=="capability" else route.all
    optimizer=torch.optim.AdamW(active,lr=cfg["full_lr"] if args.geometry=="full" else cfg["sparse_lr"],weight_decay=0)
    cached=core.load_pt(root/"base_teacher.pt")
    if cached["fingerprint"]!=stamp or cached["ids"]!=[r["id"] for r in data["rehearsal"]]:
        raise ValueError("Conservation teacher cache provenance differs")
    base_teacher=cached["logits"]
    allowed=[r for r in data["train"] if r["label"]=="allowed"]
    blocked=[r for r in data["train"] if r["label"]=="blocked"]
    history,start=[],0
    last=directory/"checkpoint.pt"
    if last.exists():
        state=core.load_pt(last)
        if state["fingerprint"]!=stamp:
            raise ValueError("Checkpoint provenance differs")
        route.load(state["parameters"])
        optimizer.load_state_dict(state["optimizer"])
        start,history=state["step"],state["history"]
        print(f"Resumed {key} {args.stage} at {start}",flush=True)
    total=cfg["capability_steps"] if args.stage=="capability" else cfg["joint_steps"]
    def save(step):
        if shutil.disk_usage(directory).free<40*1024**3:
            raise OSError("Less than 40 GiB free; previous checkpoint retained")
        core.atomic_torch(last,dict(fingerprint=stamp,parameters=route.state(),
            optimizer=optimizer.state_dict(),step=step,history=history))
    def dev(step):
        core.atomic_json(directory/f"dev_{step:04d}.json",{
            panel:evaluate(model,tokenizer,data[panel],cfg) for panel in ("dev","general_dev")})
    if start==0:
        dev(0)
        save(0)
    for step in range(start,total):
        tick=time.monotonic()
        model.train()
        learning_rate=(cfg["full_lr"] if args.geometry=="full" else cfg["sparse_lr"])*min(1.,(step+1)/cfg["warmup"])
        for group in optimizer.param_groups:
            group["lr"]=learning_rate
        optimizer.zero_grad(set_to_none=True)
        losses,tokens={},0
        for category,pool,group in [("allowed",allowed,route.cap),
                                    ("blocked",blocked,route.cap if args.stage=="capability" else route.ref)]:
            route.enable(group)
            values=[]
            for micro in range(2):
                row=pool[picked(pool,step,micro,cfg["seed"])]
                answer="REFUSE" if args.stage=="joint" and category=="blocked" else row["answer"]
                loss=nll(model,tokenizer,row,answer)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite supervised loss")
                (loss/4).backward()
                values.append(float(loss.detach()))
                tokens+=len(ids_for(tokenizer,row,answer))
            losses[category+"_nll"]=sum(values)/len(values)
        if teacher_logits is not None:
            route.enable(route.cap)
            i=picked(allowed,step,0,cfg["seed"]+19)
            loss=kl(model,tokenizer,allowed[i],teacher_logits[i])
            (cfg["preserve"]*loss).backward()
            losses["preserve_kl"]=float(loss.detach())
            tokens+=len(ids_for(tokenizer,allowed[i]))
        route.enable(active)
        i=step%len(data["rehearsal"])
        loss=kl(model,tokenizer,data["rehearsal"][i],base_teacher[i])
        (cfg["conserve"]*loss).backward()
        losses["conserve_kl"]=float(loss.detach())
        tokens+=len(ids_for(tokenizer,data["rehearsal"][i]))
        norm=torch.nn.utils.clip_grad_norm_(active,1.)
        if not torch.isfinite(norm) or not all(torch.isfinite(torch.tensor(v)) for v in losses.values()):
            raise FloatingPointError("Nonfinite loss or gradient")
        optimizer.step()
        for device in range(torch.cuda.device_count()):
            torch.cuda.synchronize(device)
        history.append(dict(step=step+1,**losses,grad_norm=float(norm),learning_rate=learning_rate,
            train_seconds=time.monotonic()-tick,processed_tokens=tokens,
            peak_gpu_bytes=[torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())]))
        if (step+1)%cfg["checkpoint_every"]==0 or step+1==total:
            save(step+1)
            core.atomic_json(directory/"history.json",history)
            print(json.dumps(dict(event="progress",key=key,stage=args.stage,**history[-1])),flush=True)
        if (step+1)%cfg["dev_every"]==0 or step+1==total:
            dev(step+1)
    if route.frozen_hash()!=frozen:
        raise AssertionError("A frozen original weight changed")
    core.atomic_torch(directory/"final.pt",dict(fingerprint=stamp,parameters=route.state(),step=total))
    if args.stage=="capability":
        teacher_cache(model,tokenizer,allowed,directory/"teacher.pt",stamp)
    core.atomic_json(directory/"DONE.json",dict(fingerprint=stamp,step=total))
    print(json.dumps(dict(event="done",key=key,stage=args.stage)),flush=True)


def launch(args,cfg):
    root=Path(args.root).resolve()
    root.mkdir(parents=True,exist_ok=True)
    launcher_lock=(root/"launcher.lock").open("a")
    fcntl.flock(launcher_lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    initialize(root,cfg)
    (root/"logs").mkdir(exist_ok=True)
    env=os.environ.copy()
    cache=Path(__file__).resolve().parent/"cache"
    env.update(HF_HOME=str(cache/"huggingface"),XDG_CACHE_HOME=str(cache),
        TORCH_HOME=str(cache/"torch"),CUDA_CACHE_PATH=str(cache/"cuda"),
        TRITON_CACHE_DIR=str(cache/"triton"),MPLCONFIGDIR=str(cache/"matplotlib"),
        HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",OMP_NUM_THREADS="2",MKL_NUM_THREADS="2")
    core.atomic_json(root/"environment.json",dict(host=socket.gethostname(),python=sys.version,
        executable=sys.executable,torch=torch.__version__,cuda=torch.version.cuda,
        transformers=core.version("transformers"),numpy=core.version("numpy"),
        gpu=subprocess.check_output(["nvidia-smi","--query-gpu=index,name,driver_version,memory.total",
                                     "--format=csv,noheader"],text=True)))
    tasks=[]
    def add(stage,geometry,width=0,placement="same",evaluate=False):
        key=job_key(geometry,width,placement)
        ident=f"{stage}__{key}"+("__test" if evaluate else "")
        if evaluate:
            deps=[("joint",job_key(g,w,p)) for g,w,p in configurations]
        elif stage=="joint":
            deps=[("capability",job_key(geometry,width,"same"))]
        elif stage=="capability":
            deps=[("baseline",job_key("native",0,"same"))]
        else:
            deps=[]
        tasks.append(dict(id=ident,stage=stage,geometry=geometry,width=width,
            placement=placement,evaluate=evaluate,deps=deps,gpus=2 if geometry=="full" else 1))
    configurations=[(g,w,p) for w in cfg["widths"] for g,p in
                    [("native","same"),("native","different"),("scattered","same")]]
    configurations.append(("full",0,"same"))
    add("baseline","native")
    for g in ("native","scattered"):
        for w in cfg["widths"]:
            add("capability",g,w)
    add("capability","full")
    for g,w,p in configurations:
        add("joint",g,w,p)
    # Global barrier: no final test until every scheduled joint fit is complete.
    add("baseline","native",evaluate=True)
    for g in ("native","scattered"):
        for w in cfg["widths"]:
            add("capability",g,w,evaluate=True)
    add("capability","full",evaluate=True)
    for g,w,p in configurations:
        add("joint",g,w,p,evaluate=True)
    available=[int(x) for x in args.gpus.split(",")]
    running,finished,failed={},[],[]
    while tasks or running:
        for task in list(tasks):
            directory=root/task["stage"]/job_key(task["geometry"],task["width"],task["placement"])
            done=directory/("test.json" if task["evaluate"] else "DONE.json")
            if done.exists():
                tasks.remove(task)
                finished.append(task["id"])
                continue
            if len(available)<task["gpus"] or not all((root/s/k/"DONE.json").exists() for s,k in task["deps"]):
                continue
            assigned=available[:task["gpus"]]
            available=available[task["gpus"]:]
            command=[sys.executable,str(Path(__file__).resolve()),"--worker","--root",str(root),
                "--stage",task["stage"],"--geometry",task["geometry"],"--width",str(task["width"]),
                "--placement",task["placement"]]
            if task["evaluate"]:
                command.append("--evaluate")
            if args.smoke:
                command.append("--smoke")
            log=(root/"logs"/(task["id"]+".log")).open("a")
            proc=subprocess.Popen(command,env=dict(env,CUDA_VISIBLE_DEVICES=",".join(map(str,assigned))),
                                  stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            running[proc.pid]=(proc,task,assigned,log)
            tasks.remove(task)
            print(json.dumps(dict(event="launch",id=task["id"],pid=proc.pid,gpus=assigned)),flush=True)
        for pid,(proc,task,assigned,log) in list(running.items()):
            code=proc.poll()
            if code is not None:
                log.close()
                available.extend(assigned)
                available.sort()
                del running[pid]
                (finished if code==0 else failed).append(task["id"])
                print(json.dumps(dict(event="exit",id=task["id"],exit=code)),flush=True)
        core.atomic_json(root/"status.json",dict(time=time.time(),complete=len(finished),total=2+2*(2*len(cfg["widths"])+1)+2*len(configurations),
            failed=failed,queued=len(tasks),running=[dict(id=t[1]["id"],pid=p,gpus=t[2]) for p,t in running.items()]))
        if failed:
            # Running independent fits finish safely; dependent jobs are not launched.
            for proc,_,_,log in running.values():
                proc.wait()
                log.close()
            raise RuntimeError(f"Failed jobs: {failed}; inspect logs and resume")
        if tasks and not running and not any(all((root/s/k/"DONE.json").exists()
                for s,k in t["deps"]) for t in tasks):
            raise RuntimeError("Unsatisfied task dependencies")
        time.sleep(5)
    print("ALL TRAINING AND EVALUATION COMPLETE",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",required=True)
    parser.add_argument("--gpus",default="0,1,2,3,4,5")
    parser.add_argument("--worker",action="store_true")
    parser.add_argument("--evaluate",action="store_true")
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--stage",choices=["baseline","capability","joint"],default="capability")
    parser.add_argument("--geometry",choices=["native","scattered","full"],default="native")
    parser.add_argument("--placement",choices=["same","different"],default="same")
    parser.add_argument("--width",type=int,default=4)
    args=parser.parse_args()
    cfg=copy.deepcopy(PROTOCOL)
    if args.smoke:
        cfg.update(widths=[4],capability_steps=2,joint_steps=2,checkpoint_every=1,dev_every=2,
                   train_pairs=4,dev_pairs=2,test_pairs=2,wording_pairs=2,
                   rehearsal_per_task=2,general_dev_per_task=2,general_test_per_task=2)
    if args.worker:
        if args.stage!="baseline" and (args.geometry!="full" and args.width not in cfg["widths"] or
                args.geometry=="full" and (args.width!=0 or args.placement!="same") or
                args.geometry=="scattered" and args.placement!="same"):
            parser.error("Invalid width/placement for this geometry")
        worker(args,cfg)
    else:
        launch(args,cfg)


if __name__=="__main__":
    main()
