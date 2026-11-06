# Earlier theory and pilot material

This folder retains the former `main.tex`, rationale/summary drafts, theoretical reviews, numerical checks and their compact plotted data. The current AAAI manuscript is [../../paper.tex](../../paper.tex).

The historical `main.tex` references pilot artifacts that were already deleted before the project cleanup. It is preserved as authored history; it is not the current paper build target. See [raw-data notes](../../RAW_DATA.md#legacy-pilot-artifacts).

Run the small numerical checks from this folder:

```sh
python3 experiments/routing_checks.py --output results
python3 verification/proof_checks.py
```
