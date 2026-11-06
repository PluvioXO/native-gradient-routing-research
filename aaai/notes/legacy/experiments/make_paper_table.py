#!/usr/bin/env python3
"""Regenerate the manuscript's table from the recorded population experiment."""
from pathlib import Path
import json

root = Path(__file__).resolve().parent.parent
rows = json.loads((root / 'results/teacher_student_summary.json').read_text())
methods = ['ordinary', 'semantic_route', 'random_route', 'semantic_constraint', 'random_constraint']
labels = {'ordinary': 'Ordinary', 'semantic_route': 'Semantic route', 'random_route': 'Random route', 'semantic_constraint': 'Semantic constraint', 'random_constraint': 'Random constraint'}
assert set(methods) == {row['method'] for row in rows}
lines = [r'\begin{table}[htbp]', r'\centering\small', r'\begin{tabularx}{\textwidth}{lYrrrr}', r'\toprule', r'Head & Update & Clean & \multicolumn{2}{c}{Restoration MSE} & Swap \\', r'& & accuracy & Semantic & Own route & accuracy \\', r'\midrule']
for head in ['frozen', 'trainable']:
    for method in methods:
        row = next(row for row in rows if row['geometry'] == 'coupled' and int(row['inherited_bypass']) == 1 and row['head'] == head and row['method'] == method)
        def fmt(value):
            return r'$<10^{-20}$' if abs(value) < 1e-20 else f'{value:.4f}'
        values = [head.title(), labels[method], f"{row['new_accuracy_mean']:.5f}", fmt(row['semantic_baseline_restoration_mse_mean']), fmt(row['route_baseline_restoration_mse_mean']), f"{row['semantic_swap_accuracy_mean']:.5f}"]
        lines.append(' & '.join(values) + r' \\')
    if head == 'frozen':
        lines.append(r'\midrule')
lines.extend([r'\bottomrule', r'\end{tabularx}', r'\caption{Population outcomes in the coupled, inherited-copy condition. Restoration MSE measures remaining output change relative to the initial model; zero means that change is erased. Random-arm entries average 20 orientations. For the ordinary arm, the own-route column uses the common semantic span.}', r'\label{tab:toy}', r'\end{table}'])
(root / 'results/paper_table.tex').write_text('\n'.join(lines) + '\n')
print('Wrote results/paper_table.tex')
