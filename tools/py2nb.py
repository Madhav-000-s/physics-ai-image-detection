"""Build a .ipynb from a percent-format script ("# %%" code cells, "# %% [markdown]" text cells).

    python tools/py2nb.py notebooks_src/05_face_deepfakes.py 05_face_deepfakes.ipynb
"""
import json
import re
import sys


def convert(src: str) -> dict:
    cells = []
    for block in re.split(r"^# %%", src, flags=re.M)[1:]:
        header, _, body = block.partition("\n")
        body = body.strip("\n")
        if "[markdown]" in header:
            text = "\n".join(re.sub(r"^# ?", "", line) for line in body.splitlines())
            cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)})
        else:
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                          "source": body.splitlines(keepends=True)})
    return {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                         "language_info": {"name": "python"}}}


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    with open(src, encoding="utf-8") as f:
        nb = convert(f.read())
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"{dst}: {len(nb['cells'])} cells")
