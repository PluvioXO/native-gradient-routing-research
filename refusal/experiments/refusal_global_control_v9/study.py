"""Matched full-model control, fresh panels, exact resume, and sealed evaluation."""
import argparse
import fcntl
import gc
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
from functools import cache
from types import SimpleNamespace

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ.setdefault("OMP_NUM_THREADS", "2")
HERE = Path(__file__).resolve().parent
from v9design import (CONFIG, DEFENSE_STEPS, ATTACK_STEPS, REPAIR_STEPS,
                      digest, jobs, select_pairs, general_panels, navigation_bank, world_keys)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def sha(path):
    import hashlib
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024**2), b""):
            h.update(block)
    return h.hexdigest()


def sources():
    lineage = read(HERE / "lineage.json")
    paths = list(HERE.glob("*.py")) + [HERE/"PROTOCOL.md", HERE/"lineage.json", HERE/"launch.sh"]
    paths += [HERE/"reference_v8"/n for n in lineage["sources"]]
    for name, expected in lineage["sources"].items():
        assert sha(HERE/"reference_v8"/name) == expected
    return {str(p.relative_to(HERE)): sha(p) for p in sorted(paths)}


@cache
def runtime():
    sys.path.insert(0, str(HERE/"reference_v8"))
    from reference_v8 import experiment as v8
    assert v8.sources() == read(HERE/"lineage.json")["sources"]
    e, torch = v8.api.engine()
    return v8, e, torch


def check(root):
    m = read(root/"manifest.json")
    assert m["sources"] == sources()
    assert digest(read(root/"data.json")) == m["data_sha256"]
    return m


def rows_in(value):
    if isinstance(value, dict):
        if "prompt" in value:
            yield value
        else:
            for v in value.values():
                yield from rows_in(v)
    elif isinstance(value, list):
        for v in value:
            yield from rows_in(v)


def validate_data(data, cfg, smoke):
    groups={k:v for k,v in data.items() if isinstance(v,list)}
    groups.update({f"{seed}/{name}":rows for seed,banks in data["seeds"].items()
                   for name,rows in banks.items()})
    worlds=set();prompts=set();counts={}
    for name,rows in groups.items():
        if not rows or "edges" not in rows[0]:continue
        local=set()
        for row in rows:
            node=row["start"];path=[]
            for _ in range(row["depth"]):node=row["edges"][node];path.append(node)
            assert row["path"]==path and row["answer"]==" ".join(path)
            assert (row["label"]=="blocked")==(row["marker"] in path[:-1])
            local.update(world_keys(row["edges"]))
        if not smoke:assert not worlds & local
        worlds.update(local);counts[name]=len(rows)
    for name in ["rehearsal","general_dev","general_test"]:
        local={r["prompt"] for r in data[name]}
        if not smoke:assert len(local)==len(data[name]) and not prompts & local
        prompts.update(local)
    if not smoke:
        assert len(data["policy_dev"])==9*cfg["dev_worlds"]
        assert len(data["policy_test"])==9*cfg["test_worlds"]
        for split,n in [("dev",cfg["benchmark_dev_per_task"]),("test",cfg["benchmark_test_per_task"])]:
            for task in ["arc_easy","winogrande"]:
                assert sum(r["task"]==task for r in data["benchmark_"+split])==n
    return counts


def initialize(root, previous, smoke):
    root.mkdir(parents=True, exist_ok=True)
    old = read(previous/"results/manifest.json")
    assert old == read(HERE/"lineage.json")
    old_data = read(previous/"results/data.json")
    assert digest(old_data) == old["data_sha256"]
    cfg = dict(CONFIG)
    if smoke:
        cfg.update(seeds=[101], parents=["shared4"], worlds=4, dev_worlds=2,
                   test_worlds=2, probe_worlds=2, defense_updates=2,
                   removal_updates=2, repair_updates=2, rehearsal_per_task=2,
                   general_dev_per_task=2, general_test_per_task=2)
    excluded = set(read(HERE/"reference_v8/reference_v7/reference_v3/excluded_worlds.json"))
    v7_root=Path(old["previous"])/"results"
    v7_data=read(v7_root/"data.json")
    assert digest(v7_data)==read(v7_root/"manifest.json")["data_sha256"]
    old_rows = list(rows_in(old_data)) + list(rows_in(v7_data))
    for row in old_rows:
        if "edges" in row:
            excluded.update(world_keys(row["edges"]))
    prior = set(excluded)
    data = {}; bank_checks = {}
    bank_plan = [(name, n, seed) for name, n, seed in [
        ("policy_dev", cfg["dev_worlds"], 911081), ("policy_test", cfg["test_worlds"], 911082),
        ("probe", cfg["probe_worlds"], 911083)]]
    bank_plan += [(f"{seed}_{stage}", cfg["worlds"], 911100 + seed*11 + i)
                  for seed in cfg["seeds"] for i, stage in enumerate(
                      ["defense_train", "attack_train", "repair_train"])]
    data["seeds"] = {str(seed): {} for seed in cfg["seeds"]}
    for name, n, seed in bank_plan:
        bank = navigation_bank(excluded, n, seed, "v9_"+name)
        aliases = set().union(*(world_keys(r["edges"]) for r in bank))
        assert not aliases & excluded
        excluded.update(aliases)
        bank_checks[name] = dict(worlds=n, aliases=len(aliases), seed=seed)
        if name.split("_")[0].isdigit():
            key, field = name.split("_", 1); data["seeds"][key][field] = bank
        else:
            data[name] = bank
    data.update(general_panels({r["prompt"] for r in old_rows}, cfg))
    v8, e, _ = runtime()
    if smoke:
        for split in ["dev", "test"]:
            data["policy_"+split] = data["seeds"]["101"]["defense_train"][:18]
            data["general_"+split] = data["rehearsal"]
            data["benchmark_"+split] = sum([[r for r in old_data["benchmark_dev"]
                if r["task"] == task][:2] for task in ["arc_easy", "winogrande"]], [])
    else:
        protocol = read(HERE/"reference_v8/reference_v7/reference_v3/original_manifest.json")["protocol"]
        tok = e.AutoTokenizer.from_pretrained(protocol["model"], revision=protocol["revision"])
        pool = e.prepare_benchmarks(SimpleNamespace(data_seed=911084, benchmark_per_task=512,
            benchmark_max_prompt=384), tok, root)
        old_prompts = {r["prompt"] for r in old_rows}
        fresh = [r for r in pool if r["prompt"] not in old_prompts]
        for split, start, count in [("dev", 0, 128), ("test", 128, 256)]:
            data["benchmark_"+split] = []
            for task in ["arc_easy", "winogrande"]:
                selected = [r for r in fresh if r["task"] == task][start:start+count]
                assert len(selected) == count, (task, len(selected), count)
                data["benchmark_"+split].extend(
                    dict(r, id="v9_"+split+"_"+r["id"]) for r in selected)
        for stem in ["policy", "general", "benchmark"]:
            assert not {r["prompt"] for r in data[stem+"_dev"]} & {
                r["prompt"] for r in data[stem+"_test"]}
    validated_counts=validate_data(data,cfg,smoke)
    planned = jobs(smoke)
    if smoke:
        planned.update({key+"_reference": dict(spec, reference=True)
                        for key, spec in list(planned.items())})
    m = dict(protocol="matched-global-control-v9", config=cfg, smoke=smoke,
        sources=sources(), data_sha256=digest(data), parents=old["parents"],
        previous=str(previous), previous_manifest_sha256=sha(previous/"results/manifest.json"),
        defense_jobs=planned, inherited_world_aliases=len(prior),
        benchmark_sources=None if smoke else read(root/"benchmark_sources/provenance.json"))
    for name, value in [("manifest.json", m), ("data.json", data)]:
        if (root/name).exists():
            assert read(root/name) == value
        else:
            write(root/name, value)
    write(root/"data_audit.json", dict(bank_checks=bank_checks,
        inherited_exclusions=len(prior), world_alias_exclusion=True,
        general_counts={k:len(data[k]) for k in ["rehearsal","general_dev","general_test"]},
        benchmark_counts={k:len(data[k]) for k in ["benchmark_dev","benchmark_test"]},
        sources=m["sources"], data_sha256=m["data_sha256"], smoke=smoke,
        oracle_and_split_counts=validated_counts))
    print("Initialized", root, "smoke", smoke, flush=True)


def load_original(m, parent, two=True):
    v8, e, t = runtime()
    cfg = read(HERE/"reference_v8/reference_v7/reference_v3/original_manifest.json")["protocol"]
    tok, model = v8.api.tokenizer_and_model(e, t, cfg, two)
    meta = m["parents"][parent]
    path = Path(meta["folder"])/"initial.pt"
    assert sha(path) == meta["ready"]["checkpoint_sha256"]
    saved = e.core.load_pt(path); v8.api.load_parameters(model, saved["parameters"], t); del saved
    assert e.core.tensor_hash(model.named_parameters()) == meta["ready"]["parameter_hash"]
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    return v8, e, t, tok, model


def save(path, value):
    v8, _, _ = runtime()
    v8.save(path, value, 6*1024**3 if "optimizer" in value else 2*1024**3)
    gc.collect()


def baseline(root, parent):
    m = check(root); folder = root/"parents"/parent
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/"DONE.json").exists(): return
    v8, e, t, tok, model = load_original(m, parent)
    data = read(root/"data.json")
    report = v8.api.measure(e,t,model,tok,data,"dev",m["smoke"])
    owner = e.core.tensor_hash(model.named_parameters())
    e.cache_teacher(model,tok,data["rehearsal"],folder/"general_teacher.pt",owner)
    write(folder/"baseline_dev.json",report)
    write(folder/"DONE.json",dict(parameter_hash=owner,
        teacher_sha256=sha(folder/"general_teacher.pt"), manifest=digest(m)))


def defend(root, key, stop_after=None):
    m=check(root); spec=m["defense_jobs"][key]; cfg=m["config"]
    folder=root/"defense"/key;folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/"worker.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (folder/"DONE.json").exists():return
    v8,e,t,tok,model=load_original(m,spec["parent"])
    t.manual_seed(spec["seed"]);random.seed(spec["seed"])
    support=v8.Support(model);assert support.count==494032768
    opt=t.optim.AdamW(support.optparams,lr=cfg["lr"],weight_decay=0,foreach=False)
    data=read(root/"data.json"); bank=dict(data,**data["seeds"][str(spec["seed"])])
    parent=root/"parents"/spec["parent"]; ready=read(parent/"DONE.json")
    assert sha(parent/"general_teacher.pt")==ready["teacher_sha256"]
    teacher=e.core.load_pt(parent/"general_teacher.pt");assert teacher["owner"]==ready["parameter_hash"]
    base=read(parent/"baseline_dev.json"); ident=digest(dict(manifest=digest(m),job=key))
    history=[];start=0
    if (folder/"resume.pt").exists():
        state=e.core.load_pt(folder/"resume.pt");assert state["identity"]==ident
        v8.api.load_parameters(model,state["parameters"],t);opt.load_state_dict(state["optimizer"])
        history=state["history"];start=state["step"];v8.restore_rng(t,state["rng"]);del state
    def checkpoint(step):
        save(folder/"resume.pt",dict(identity=ident,step=step,parameters=v8.api.snapshot(model),
            optimizer=opt.state_dict(),history=history,rng=v8.rng(t)))
    def observe(step):
        path=folder/f"dev_{step:04d}.json";hashed=e.core.tensor_hash(model.named_parameters())
        if path.exists():assert read(path)["parameter_hash"]==hashed;return
        cosine,probe=v8.probe_cosine(e,model,tok,support,bank)
        report=v8.api.measure(e,t,model,tok,data,"dev",m["smoke"])
        record=dict(identity=ident,step=step,parameter_hash=hashed,probe=probe,report=report,
            selection=v8.policy_quality(report,base,step,cosine))
        if step>0:save(folder/f"step_{step:04d}.pt",dict(identity=ident,step=step,
            parameter_hash=hashed,parameters=v8.api.snapshot(model)))
        write(path,record)
        print(json.dumps(dict(job=key,**record["selection"])),flush=True)
    observe(start)
    schedule=[0,1,2] if m["smoke"] else DEFENSE_STEPS
    for step in range(start,spec["limit"]):
        tick=time.monotonic()
        record=v8.defense_update(e,t,model,tok,support,opt,bank,teacher["logits"],
            dict(arm="global",beta=spec["beta"]),step,cfg)
        history.append(dict(record,seconds=time.monotonic()-tick))
        write(folder/"history.json",history)
        if step+1 in schedule:observe(step+1)
        write(folder/"progress.json",dict(step=step+1,limit=spec["limit"]))
        if (step+1)%16==0 or step+1 in [spec["limit"],stop_after]:checkpoint(step+1)
        if step+1==stop_after:
            write(folder/"INTERRUPTED.json",dict(step=step+1));return
    write(folder/"DONE.json",dict(identity=ident,updates=len(history),
        parameter_hash=e.core.tensor_hash(model.named_parameters()),
        optimizer_hash=v8.api.optimizer_hash(e,opt) if m["smoke"] else None,
        ended_unix=time.time(),cuda_peak=[t.cuda.max_memory_allocated(i) for i in range(t.cuda.device_count())]))


def select(root):
    m=check(root);planned={k:v for k,v in m["defense_jobs"].items() if not v.get("reference")}
    assert all((root/"defense"/k/"DONE.json").exists() for k in m["defense_jobs"])
    records={key:[read(p) for p in sorted((root/"defense"/key).glob("dev_*.json"))] for key in planned}
    pairs=select_pairs(records,planned)
    if m["smoke"]:
        for pair in pairs.values():pair.update(qualified=True,step=2,reason="Mechanics-only smoke")
    models={key:dict(**spec,selected_step=pair["step"])
            for pair in pairs.values() if pair["qualified"]
            for key in pair["models"] for spec in [planned[key]]}
    attacks={key+"__"+objective:dict(model=key,**spec,objective=objective)
             for key,spec in models.items() for objective in ["benign","preserve"]}
    if m["smoke"]:
        attacks.update({key+"_reference":dict(spec,reference=True) for key,spec in list(attacks.items())})
    value=dict(manifest=digest(m),pairs=pairs,models=models,attacks=attacks)
    if (root/"selection.json").exists():assert read(root/"selection.json")==value
    else:write(root/"selection.json",value)
    print("Qualified paired models:",len(models),"of",len(planned),flush=True)


def prepare(root,key):
    m=check(root);spec=read(root/"selection.json")["models"][key]
    folder=root/"models"/key;folder.mkdir(parents=True,exist_ok=True)
    if (folder/"DONE.json").exists():return
    source=root/"defense"/key/f"step_{spec['selected_step']:04d}.pt"
    v8,e,t,tok,model=load_original(m,spec["parent"])
    state=e.core.load_pt(source);v8.api.load_parameters(model,state["parameters"],t)
    owner=e.core.tensor_hash(model.named_parameters());assert owner==state["parameter_hash"];del state
    data=read(root/"data.json")
    teacher=e.cache_teacher(model,tok,data["rehearsal"],folder/"general_teacher.pt",owner)
    assert len(teacher)==len(data["rehearsal"])
    v8.api.hardlink(source,folder/"initial.pt")
    report=read(root/"defense"/key/f"dev_{spec['selected_step']:04d}.json")["report"]
    write(folder/"baseline_dev.json",report)
    write(folder/"DONE.json",dict(parameter_hash=owner,chosen=spec,
        initial_sha256=sha(folder/"initial.pt"),teacher_sha256=sha(folder/"general_teacher.pt")))


def attack(root,key,stop_after=None):
    m=check(root);spec=read(root/"selection.json")["attacks"][key];cfg=m["config"]
    folder=root/"attacks"/key;folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/"worker.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (folder/"DONE.json").exists():return
    v8,e,t,tok,model=load_original(m,spec["parent"])
    parent=root/"models"/spec["model"];ready=read(parent/"DONE.json")
    assert sha(parent/"initial.pt")==ready["initial_sha256"]
    state=e.core.load_pt(parent/"initial.pt");v8.api.load_parameters(model,state["parameters"],t);del state
    model.requires_grad_(True);assert not any("parametrizations" in n for n,_ in model.named_parameters())
    assert all(p.requires_grad for p in model.parameters())
    t.manual_seed(spec["seed"]);random.seed(spec["seed"])
    data=read(root/"data.json");bank=data["seeds"][str(spec["seed"])]
    assert sha(parent/"general_teacher.pt")==ready["teacher_sha256"]
    teacher=e.core.load_pt(parent/"general_teacher.pt");assert teacher["owner"]==ready["parameter_hash"]
    base=read(parent/"baseline_dev.json")
    ident=digest(dict(manifest=digest(m),job=key,parent=ready))
    opt=t.optim.AdamW(model.parameters(),lr=cfg["lr"],weight_decay=0,foreach=False)
    history=[];phase="remove";start=0
    if (folder/"resume.pt").exists():
        state=e.core.load_pt(folder/"resume.pt");assert state["identity"]==ident
        v8.api.load_parameters(model,state["parameters"],t);opt.load_state_dict(state["optimizer"])
        history=state["history"];phase=state["phase"];start=state["step"];v8.restore_rng(t,state["rng"]);del state
    def observe(which,step):
        hashed=e.core.tensor_hash(model.named_parameters());path=folder/f"{which}_{step:04d}.json"
        if path.exists():assert read(path)["parameter_hash"]==hashed;return
        report=v8.api.measure(e,t,model,tok,data,"dev",m["smoke"])
        score=v8.quality(report,base,step)
        record=dict(identity=ident,phase=which,step=step,
            cumulative_updates=step+(cfg["removal_updates"] if which=="repair" else 0),
            parameter_hash=hashed,selection=score,report=report)
        names=[]
        if which=="remove":
            for name,passed in [("first_zero",score["refusal"]==0),
                                ("first_task",score["task_success"]),("first_full",score["success"])]:
                if passed and not (folder/(name+".json")).exists():names.append(name)
            best=read(folder/"best.json") if (folder/"best.json").exists() else None
            if best is None or score["rank"]>best["selection"]["rank"]:names.append("best")
            if step==cfg["removal_updates"]:names.append("final")
        elif step==cfg["repair_updates"]:names.append("repair_final")
        if names:
            source=parent/"initial.pt" if which=="remove" and step==0 else folder/"current.pt"
            if source.name=="current.pt":save(source,dict(identity=ident,parameters=v8.api.snapshot(model),parameter_hash=hashed))
            for name in names:
                v8.api.hardlink(source,folder/(name+".pt"));write(folder/(name+".json"),record)
            if source.name=="current.pt":source.unlink()
        write(path,record)
        print(json.dumps(dict(job=key,phase=which,**score)),flush=True)
    for which in ["remove","repair"][["remove","repair"].index(phase):]:
        if which!=phase:
            opt=t.optim.AdamW(model.parameters(),lr=cfg["lr"],weight_decay=0,foreach=False)
            start=0;gc.collect();t.cuda.empty_cache()
        limit=cfg["removal_updates" if which=="remove" else "repair_updates"]
        schedule=([0,1,2] if m["smoke"] else ATTACK_STEPS if which=="remove" else REPAIR_STEPS)
        observe(which,start)
        for step in range(start,limit):
            tick=time.monotonic();rows=bank["attack_train" if which=="remove" else "repair_train"]
            if spec["objective"]=="benign":
                record=v8.benign_update(e,t,model,tok,rows,data["rehearsal"],teacher["logits"],step,opt)
            else:
                record=v8.api.update(e,t,model,tok,rows,data["rehearsal"],teacher["logits"],
                                     dict(objective="preserve",lr=cfg["lr"]),step,opt)
            cumulative=step+1+(cfg["removal_updates"] if which=="repair" else 0)
            history.append(dict(record,phase=which,cumulative_updates=cumulative,
                                seconds=time.monotonic()-tick))
            write(folder/"history.json",history)
            if step+1 in schedule:observe(which,step+1)
            write(folder/"progress.json",dict(phase=which,step=step+1,limit=limit))
            if (step+1)%16==0 or step+1==limit or cumulative==stop_after:
                save(folder/"resume.pt",dict(identity=ident,phase=which,step=step+1,
                    parameters=v8.api.snapshot(model),optimizer=opt.state_dict(),history=history,rng=v8.rng(t)))
            if cumulative==stop_after:
                write(folder/"INTERRUPTED.json",dict(step=cumulative));return
        start=0
    write(folder/"DONE.json",dict(identity=ident,updates=len(history),ended_unix=time.time(),
        parameter_hash=e.core.tensor_hash(model.named_parameters())))


def heldout(root,key):
    started=time.time();m=check(root);selection=read(root/"selection.json")
    assert all((root/"attacks"/k/"DONE.json").exists() for k in selection["attacks"])
    folder=root/"models"/key
    if (folder/"TEST_DONE.json").exists():return
    spec=selection["models"][key];v8,e,t,tok,model=load_original(m,spec["parent"],False)
    ready=read(folder/"DONE.json");assert sha(folder/"initial.pt")==ready["initial_sha256"]
    state=e.core.load_pt(folder/"initial.pt");v8.api.load_parameters(model,state["parameters"],t);del state
    data=read(root/"data.json");base=v8.api.measure(e,t,model,tok,data,"test",m["smoke"])
    write(folder/"baseline_test.json",dict(parameter_hash=ready["parameter_hash"],report=base,started_unix=started))
    for objective in ["benign","preserve"]:
        trial=root/"attacks"/(key+"__"+objective);done=read(trial/"DONE.json")
        for endpoint in ["first_zero","first_task","first_full","best","final","repair_final"]:
            source=trial/(endpoint+".pt");output=trial/("test_"+endpoint+".json")
            if not source.exists():continue
            if output.exists():assert read(output)["identity"]==done["identity"];continue
            state=e.core.load_pt(source);v8.api.load_parameters(model,state["parameters"],t);del state
            record=read(trial/(endpoint+".json"));hashed=e.core.tensor_hash(model.named_parameters())
            assert hashed==record["parameter_hash"]
            report=v8.api.measure(e,t,model,tok,data,"test",m["smoke"])
            assert hashed==e.core.tensor_hash(model.named_parameters())
            write(output,dict(identity=done["identity"],endpoint=endpoint,step=record["step"],
                phase=record["phase"],cumulative_updates=record["cumulative_updates"],
                parameter_hash=hashed,selection=v8.quality(report,base,record["step"]),
                report=report,started_unix=started))
        write(trial/"TEST_DONE.json",dict(identity=done["identity"]))
    write(folder/"TEST_DONE.json",dict(started_unix=started))


def campaign(root,stage,gpus):
    m=check(root)
    lock=(root/"launcher.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if stage=="baseline":tasks=m["config"]["parents"];base="parents"
    elif stage=="defense":tasks=list(m["defense_jobs"]);base="defense"
    else:
        selected=read(root/"selection.json")
        tasks=list(selected["attacks"] if stage=="attack" else selected["models"])
        base="attacks" if stage=="attack" else "models"
    marker="TEST_DONE.json" if stage=="heldout" else "DONE.json"
    count=1 if stage=="heldout" else 2
    active={};failures=[];(root/"logs").mkdir(exist_ok=True)
    def complete(key):return (root/base/key/marker).exists()
    while True:
        used={g for _,gs,_ in active.values() for g in gs}
        query=subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.used","--format=csv,noheader,nounits"],text=True)
        memory={int(r.split(",")[0]):int(r.split(",")[1]) for r in query.splitlines()}
        free=[g for g in gpus if g not in used and memory[g]<64]
        for key in tasks:
            if complete(key) or key in active or len(free)<count or failures:continue
            assigned=free[:count];free=free[count:]
            args=[sys.executable,"-u",str(HERE/"study.py"),"--root",str(root),"--worker",stage+":"+key]
            interrupted=root/base/key/"INTERRUPTED.json"
            if m["smoke"] and stage in ["defense","attack"] and not key.endswith("_reference") and not interrupted.exists():
                args+=["--stop-after","1"]
            log=(root/"logs"/(stage+"_"+key+".log")).open("a")
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=",".join(map(str,assigned)))
            child=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            active[key]=(child,assigned,log)
        for key,(child,assigned,log) in list(active.items()):
            code=child.poll()
            if code is not None:
                log.close();del active[key]
                if code:failures.append(dict(job=key,code=code))
        write(root/"status.json",dict(stage=stage,complete=sum(complete(k) for k in tasks),
            planned=len(tasks),running=[dict(job=k,pid=p.pid,gpus=gs) for k,(p,gs,_) in active.items()],
            failures=failures,updated_unix=time.time()))
        if failures and not active:raise RuntimeError("Workers failed; committed checkpoints retained: "+str(failures))
        if all(complete(k) for k in tasks) and not active:break
        time.sleep(5)
    if m["smoke"] and stage=="defense":
        for arm in CONFIG["arms"]:
            key=f"shared4__s101__{arm}"
            a=read(root/"defense"/key/"DONE.json")
            b=read(root/"defense"/(key+"_reference")/"DONE.json")
            assert a["parameter_hash"]==b["parameter_hash"] and a["optimizer_hash"]==b["optimizer_hash"]
        write(root/"resume_validation.json",dict(sources=m["sources"],both_arms_resume_exact=True))
    if m["smoke"] and stage=="attack":
        for key in tasks:
            if key.endswith("_reference"):continue
            a=read(root/"attacks"/key/"DONE.json")
            b=read(root/"attacks"/(key+"_reference")/"DONE.json")
            assert a["parameter_hash"]==b["parameter_hash"]
            x=read(root/"attacks"/key/"history.json")
            y=read(root/"attacks"/(key+"_reference")/"history.json")
            assert [{k:v for k,v in r.items() if k!="seconds"} for r in x]==[
                    {k:v for k,v in r.items() if k!="seconds"} for r in y]
        validation=read(root/"resume_validation.json")
        validation["downstream_actual_resume_exact"]=True
        write(root/"resume_validation.json",validation)
    if stage=="heldout":
        write(root/"COMPLETE.json",dict(manifest=digest(m),models=len(tasks),
            downstream_fits=len(read(root/"selection.json")["attacks"]),finished_unix=time.time()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--previous",type=Path)
    parser.add_argument("--initialize",action="store_true")
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--stage",choices=["baseline","defense","prepare","attack","heldout"])
    parser.add_argument("--select",action="store_true")
    parser.add_argument("--worker")
    parser.add_argument("--stop-after",type=int)
    parser.add_argument("--gpus",default="0,1,2,3,4,5")
    args=parser.parse_args();root=args.root.resolve()
    if args.initialize:initialize(root,args.previous.resolve(),args.smoke)
    elif args.select:select(root)
    elif args.worker:
        stage,key=args.worker.split(":",1)
        if stage=="defense":defend(root,key,args.stop_after)
        elif stage=="attack":attack(root,key,args.stop_after)
        else:{"baseline":baseline,"prepare":prepare,"heldout":heldout}[stage](root,key)
    elif args.stage:
        m=check(root)
        if not m["smoke"]:
            validation=read(root.parent/"hardware/hardware_validation.json")
            assert validation["sources"]==m["sources"] and validation["both_arms_resume_exact"]
        campaign(root,args.stage,[int(g) for g in args.gpus.split(",")])


if __name__=="__main__":
    main()
