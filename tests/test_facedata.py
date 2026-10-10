"""Tests for physfeat.facedata and the set-based evaluation in physfeat.protocol."""
import io
import os
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from physfeat import facedata as FD  # noqa: E402
from physfeat import protocol as P  # noqa: E402


def _img(path, colour=(128, 100, 90)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), colour).save(path)


def test_scanners_read_the_dataset_layouts(tmp_path):
    for split in ("train", "valid", "test"):
        for lab in ("real", "fake"):
            for i in range(3):
                _img(tmp_path / "140k/real_vs_fake/real-vs-fake" / split / lab / f"{i}.jpg")
    for i in range(4):
        _img(tmp_path / "ffpp/real" / f"00{i}_0001.png")
        _img(tmp_path / "ffpp/fake/Deepfakes" / f"00{i}_10{i}_0001.png")
        _img(tmp_path / "ffpp/fake/FaceSwap" / f"00{i}_10{i}_0002.png")
    for i in range(2):
        _img(tmp_path / "sfhq/images/images" / f"SFHQ_pt1_{i:08d}.jpg")
    k = FD.scan_140k(tmp_path / "140k")
    assert len(k) == 12                                       # "valid" is left out
    assert set(k.split) == {"train", "test"} and set(k.method) == {"real", "stylegan"}
    f = FD.scan_ffpp(tmp_path / "ffpp")
    assert sorted(f.method.unique()) == ["deepfakes", "faceswap", "real"]
    assert f.loc[f.method == "deepfakes", "group"].iloc[0].count("_") == 1   # target_source video id
    s = FD.scan_sfhq(tmp_path / "sfhq")
    assert len(s) == 2 and (s.label == 1).all()


def test_stream_tar_takes_only_the_first_images(tmp_path):
    tar_path = tmp_path / "ADM.tar"
    with tarfile.open(tar_path, "w") as tf:
        for i in range(50):
            buf = io.BytesIO()
            Image.new("RGB", (8, 8), (i, i, i)).save(buf, "PNG")
            info = tarfile.TarInfo(f"ADM/sub dir/{i:03d}.png")
            info.size = len(buf.getvalue())
            tf.addfile(info, io.BytesIO(buf.getvalue()))
        info = tarfile.TarInfo("ADM/readme.txt")
        info.size = 2
        tf.addfile(info, io.BytesIO(b"hi"))
    url = tar_path.resolve().as_uri()
    got = FD.stream_tar(url, tmp_path / "out", n=10, stride=2, log=lambda *a: None)
    assert len(got) == 10 and all(Path(p).exists() for p in got)
    assert Path(got[1]).name == "002.png" and Path(got[1]).parent.name == "sub_dir"   # every 2nd image, folders kept
    again = FD.stream_tar(url, tmp_path / "out", n=10, stride=2, log=lambda *a: None)
    assert again == got                                         # cached


def _toy_manifest():
    rows = []
    for split, n in (("train", 30), ("test", 25)):
        for lab in (0, 1):
            rows += [{"path": f"{split}{lab}_{i}", "source": "140k", "method": "real" if lab == 0 else "stylegan",
                      "label": lab, "split": split, "group": f"g{i}"} for i in range(n)]
    rows += [{"path": f"sf{i}", "source": "sfhq", "method": "sfhq_stylegan2", "label": 1, "split": "test",
              "group": f"s{i}"} for i in range(25)]
    for m in ("real", "deepfakes"):
        rows += [{"path": f"ff{m}{i}", "source": "ffpp", "method": m, "label": int(m != "real"), "split": "test",
                  "group": f"v{i % 5}"} for i in range(40)]
    return pd.DataFrame(rows)


def test_sample_spreads_over_videos_and_test_sets_pair_reals():
    df = FD.sample(_toy_manifest(), n_train_per_class=10, n_test_per_method=5, seed=0).reset_index(drop=True)
    assert (df[df.split == "train"].label.value_counts() == 10).all()
    dfk = df[(df.source == "ffpp") & (df.method == "deepfakes")]
    assert dfk.group.nunique() == 5                             # 5 frames from 5 different videos
    sets = FD.test_sets(df, min_n=5)
    assert set(sets) == {"140k:stylegan", "sfhq:sfhq_stylegan2", "ffpp:deepfakes"}
    assert sets["140k:stylegan"]["seen"] and not sets["sfhq:sfhq_stylegan2"]["seen"]
    assert (df.source.values[sets["sfhq:sfhq_stylegan2"]["real"]] == "140k").all()   # SFHQ uses 140k reals
    assert (df.source.values[sets["ffpp:deepfakes"]["real"]] == "ffpp").all()


def test_evaluate_and_bootstrap_sets():
    df = FD.sample(_toy_manifest(), 30, 25, seed=0).reset_index(drop=True)
    sets = FD.test_sets(df, min_n=5)
    rng = np.random.default_rng(0)
    y = df.label.values.astype(float)
    good = y + rng.normal(0, 0.5, len(y))                      # informative
    noise = rng.normal(0, 1, len(y))                            # uninformative
    res = P.evaluate_sets({"good": good, "noise": noise}, sets)
    t = P.summary_sets(res)
    assert t.loc["MEAN (unseen sets)", "good"] > 0.8 > 0.65 > t.loc["MEAN (unseen sets)", "noise"]
    assert "MEAN GAN" in t.index and "MEAN face swap / reenactment" in t.index
    models, diffs = P.bootstrap_sets({"good": good, "noise": noise}, sets, [("good", "noise")], n_boot=100)
    d, lo, hi, p = diffs["good  minus  noise"]
    assert lo > 0 and p < 0.05
    assert models["good"][1] <= models["good"][0] <= models["good"][2]


if __name__ == "__main__":
    import tempfile
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            if "tmp_path" in fn.__code__.co_varnames[:fn.__code__.co_argcount]:
                with tempfile.TemporaryDirectory() as d:
                    fn(Path(d))
            else:
                fn()
            print("PASS", name)
