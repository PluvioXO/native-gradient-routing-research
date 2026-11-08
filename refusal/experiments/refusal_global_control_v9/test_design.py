"""Checks for pairing, fresh-world exclusion, labels and publication."""
import tempfile
from pathlib import Path
import unittest
from v9design import CONFIG,FROZEN,digest,general_panels,jobs,navigation_bank,select_pairs,world_keys
import study


class ProtocolTests(unittest.TestCase):
    def test_counts_and_matched_parameters(self):
        planned=jobs();self.assertEqual(len(planned),12)
        for parent in CONFIG["parents"]:
            for seed in CONFIG["seeds"]:
                a=planned[f"{parent}__s{seed}__clean"];b=planned[f"{parent}__s{seed}__coupled"]
                self.assertEqual({k:v for k,v in a.items() if k not in ["arm","beta"]},
                                 {k:v for k,v in b.items() if k not in ["arm","beta"]})
                self.assertEqual((a["beta"],b["beta"]),(0.,.01))

    def test_latest_common_postupdate_and_failure_are_explicit(self):
        planned=jobs(True);keys=list(planned)
        row=lambda step,q:dict(step=step,selection=dict(qualified=q))
        records={keys[0]:[row(0,True),row(16,True),row(64,True),row(128,False)],
                 keys[1]:[row(0,True),row(16,True),row(64,True),row(128,True)]}
        choice=next(iter(select_pairs(records,planned).values()))
        self.assertEqual(choice["step"],64)
        records[keys[0]]=[row(0,True),row(16,False)]
        choice=next(iter(select_pairs(records,planned).values()))
        self.assertFalse(choice["qualified"]);self.assertIsNone(choice["step"])

    def test_worlds_are_disjoint_under_both_legacy_encodings(self):
        first=navigation_bank(set(),12,101,"a")
        blocked=set().union(*(world_keys(r["edges"]) for r in first))
        second=navigation_bank(blocked,12,101,"b")
        self.assertFalse(blocked & set().union(*(world_keys(r["edges"]) for r in second)))
        self.assertEqual(len(first),108);self.assertEqual(len({r["id"] for r in first}),108)
        for row in first+second:
            path=[];node=row["start"]
            for _ in range(row["depth"]):
                node=row["edges"][node];path.append(node)
            self.assertEqual(row["path"],path)
            self.assertEqual(row["label"]=="blocked",row["marker"] in path[:-1])
            self.assertEqual(row["prompt"],FROZEN.prompt(row["edges"],row["start"],row["depth"],row["marker"]))
        b,a=FROZEN.batch(first,0)
        self.assertEqual(len(b),2);self.assertEqual({r["depth"] for r in a},{1,2})

    def test_general_panels_are_fresh_balanced_and_reproducible(self):
        cfg=dict(CONFIG,rehearsal_per_task=2,general_dev_per_task=3,general_test_per_task=4)
        x=general_panels({"What is 0 + 0? Reply with only the integer."},cfg)
        self.assertEqual(x,general_panels({"What is 0 + 0? Reply with only the integer."},cfg))
        seen=set()
        for name,n in [("rehearsal",2),("general_dev",3),("general_test",4)]:
            rows=x[name];self.assertEqual(len(rows),2*n)
            prompts={r["prompt"] for r in rows}
            self.assertFalse(prompts & seen);seen.update(prompts)
            for row in rows:
                if row["task"]=="addition":
                    a,_,b=row["prompt"][8:].split()[:3]
                    self.assertEqual(row["answer"],str(int(a)+int(b.rstrip("?"))))

    def test_atomic_publication_does_not_leave_partial_json(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"x.json";study.write(path,dict(a=[1,2,3]))
            self.assertEqual(study.read(path),dict(a=[1,2,3]))
            self.assertFalse(path.with_suffix(".json.tmp").exists())
            self.assertEqual(digest(study.read(path)),digest(dict(a=[1,2,3])))

    def test_bad_oracle_label_is_rejected(self):
        rows=navigation_bank(set(),2,42,"bad")
        rows[0]["path"]=["WRONG"]
        data=dict(policy_dev=rows,seeds={},rehearsal=[],general_dev=[],general_test=[])
        with self.assertRaises(AssertionError):
            study.validate_data(data,CONFIG,True)


if __name__=="__main__":
    unittest.main()
