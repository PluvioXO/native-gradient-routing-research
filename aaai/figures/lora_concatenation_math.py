"""Render the concatenation figure's mathematical labels as outlined SVGs."""
import json
import subprocess
import tempfile
from pathlib import Path

OUT = Path(__file__).resolve().parent
colors = dict(a="2563EB", b="D97706", c="059669", d="7C3AED")
items = []
for task, color in colors.items():
    for factor in ["A", "B"]:
        items.append(dict(key=f"{factor}_{task}", latex=f"{factor}_{{{task}}}", size=25, color=color))
for key, latex, size in [
    ("B", "B", 30), ("A", "A", 30), ("times", r"\times", 24),
    ("identity", r"\Delta W=" + "+".join(
        rf"{{\color[HTML]{{{c}}}B_{{{t}}}A_{{{t}}}}}" for t, c in colors.items()) + "=BA", 34),
    ("concat_B", r"B=\bigl[\," + r"\;".join(
        rf"{{\color[HTML]{{{c}}}B_{{{t}}}}}" for t, c in colors.items()) + r"\,\bigr]", 26),
    ("concat_A", r"A=\begin{bmatrix}" + r"\\".join(
        rf"{{\color[HTML]{{{c}}}A_{{{t}}}}}" for t, c in colors.items()) + r"\end{bmatrix}", 25),
    ("rank", "r=r_a+r_b+r_c+r_d", 25),
    ("effective", r"W_{\mathrm{eff}}=W_0+BA", 26),
]:
    items.append(dict(key=key, latex=latex, size=size, color="172033"))
tex = r"""\RequirePackage{fix-cm}
\documentclass{article}
\usepackage{amsmath,amssymb,xcolor}
\usepackage[active,tightpage]{preview}
\setlength\PreviewBorder{0pt}
\begin{document}
"""
for item in items:
    tex += ("\\begin{preview}\n{\\fontsize{" + str(item["size"]) + "}{" + str(item["size"] * 1.2)
            + "}\\selectfont\\color[HTML]{" + item["color"] + "}\\hbox{$\\displaystyle "
            + item["latex"] + "$}}\n\\end{preview}\n")
tex += "\\end{document}\n"
(OUT / "lora_concatenation_math.tex").write_text(tex)
with tempfile.TemporaryDirectory(prefix="casr-lora-math-") as temp:
    temp = Path(temp)
    source = temp / "labels.tex"
    source.write_text(tex)
    subprocess.run(["latex", "-interaction=nonstopmode", "-halt-on-error", str(source)],
                   cwd=temp, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["dvisvgm", "--no-fonts", "--page=1-", "--output=label-%p.svg", "labels.dvi"],
                   cwd=temp, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    rendered = sorted(temp.glob("label-*.svg"), key=lambda p: int(p.stem.split("-")[-1]))
    assert len(rendered) == len(items), [p.name for p in temp.iterdir()]
    for item, svg in zip(items, rendered):
        item["svg"] = svg.read_text()
(OUT / "lora_concatenation_math_assets.json").write_text(json.dumps(items))
print(f"Rendered {len(items)} LaTeX labels")
