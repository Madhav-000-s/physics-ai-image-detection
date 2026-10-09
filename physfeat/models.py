"""Detectors: CLIP linear probe (baseline), physics GBMs, and stacked fusions."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from . import geometry, lighting

GROUPS = {
    "geometry": list(geometry.FEATURE_NAMES),
    "lighting": list(lighting.FEATURE_NAMES),
    "geometry+lighting": list(geometry.FEATURE_NAMES) + list(lighting.FEATURE_NAMES),
}
BASELINE = "CLIP (baseline)"


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _gbm(seed):
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                          l2_regularization=1.0, class_weight="balanced",
                                          early_stopping=False, random_state=seed)


class Detector:
    def __init__(self, seed=0, clip_Cs=(0.25, 1.0, 4.0, 16.0), groups=None):
        self.seed, self.clip_Cs = seed, list(clip_Cs)
        self.groups = groups or GROUPS

    def fit(self, clip: np.ndarray, phys: pd.DataFrame, y: np.ndarray, log=print):
        cv = StratifiedKFold(5, shuffle=True, random_state=self.seed)
        sel = LogisticRegressionCV(Cs=self.clip_Cs, cv=3, scoring="roc_auc", class_weight="balanced",
                                   max_iter=4000).fit(clip, y)
        self.C = float(sel.C_[0])
        log(f"CLIP linear probe: C = {self.C}")
        self.clip = LogisticRegression(C=self.C, class_weight="balanced", max_iter=4000)
        oof = {"clip": cross_val_predict(self.clip, clip, y, cv=cv, method="predict_proba")[:, 1]}
        self.clip.fit(clip, y)
        self.phys = {}
        for g, cols in self.groups.items():
            m = _gbm(self.seed)
            oof[g] = cross_val_predict(m, phys[cols].values, y, cv=cv, method="predict_proba")[:, 1]
            self.phys[g] = m.fit(phys[cols].values, y)
        self.stack = {}
        for g in self.groups:
            Z = np.c_[logit(oof["clip"]), logit(oof[g])]
            self.stack[g] = LogisticRegression(class_weight="balanced").fit(Z, y)
            log(f"stack CLIP+{g}: weights {np.round(self.stack[g].coef_[0], 3)}")
        return self

    def predict(self, clip: np.ndarray, phys: pd.DataFrame) -> dict:
        s_clip = self.clip.predict_proba(clip)[:, 1]
        out = {BASELINE: s_clip}
        for g, cols in self.groups.items():
            s_g = self.phys[g].predict_proba(phys[cols].values)[:, 1]
            out[f"Physics: {g}"] = s_g
            out[f"Fused: CLIP+{g}"] = self.stack[g].predict_proba(np.c_[logit(s_clip), logit(s_g)])[:, 1]
        return out


# ----------------------------------------------------------------------------------------
# Compression-aware detector (notebook 03)
# ----------------------------------------------------------------------------------------
def quality_term(q):
    """0 for a pristine image (q=100), 1 at JPEG quality 50; NaN -> treated as pristine-ish 96."""
    q = np.where(np.isfinite(q), q, 96.0)
    return np.clip((100.0 - q) / 50.0, 0.0, 1.5)


def _stack_design(lc, lp, q, quality_aware):
    if not quality_aware:
        return np.c_[lc, lp]
    t = quality_term(q)
    return np.c_[lc, lp, t, t * lc, t * lp]


class RobustDetector:
    """Trained on clean + JPEG-augmented copies of the training images.

    * CLIP probe and physics GBM see both copies (standard augmentation).
    * Fusion = logistic stacking on out-of-fold logits. The quality-aware variant also gets
      the image's JPEG quality and its interactions with each branch, so the weight it puts
      on CLIP vs physics can change with compression.
    Out-of-fold predictions use GroupKFold over source images, so the clean and the
    compressed copy of one image never sit on both sides of a split.
    """
    NAMES = {"clip": "CLIP (JPEG-aug)", "phys": "Physics (JPEG-aug)",
             "fused": "Fused (JPEG-aug)", "fused_q": "Fused (JPEG-aug, quality-aware)"}

    def __init__(self, phys_cols=None, seed=0, clip_Cs=(0.25, 1.0, 4.0, 16.0)):
        self.phys_cols = phys_cols or GROUPS["geometry+lighting"]
        self.seed, self.clip_Cs = seed, list(clip_Cs)

    def fit(self, clip, phys, y, q, groups, log=print):
        from sklearn.model_selection import GroupKFold
        splits = list(GroupKFold(n_splits=5).split(clip, y, groups))
        sel = LogisticRegressionCV(Cs=self.clip_Cs, cv=splits[:3], scoring="roc_auc",
                                   class_weight="balanced", max_iter=4000).fit(clip, y)
        self.C = float(sel.C_[0])
        log(f"CLIP linear probe (augmented): C = {self.C}")
        self.clip = LogisticRegression(C=self.C, class_weight="balanced", max_iter=4000)
        X = phys[self.phys_cols].values
        self.phys = _gbm(self.seed)
        oof_c = cross_val_predict(self.clip, clip, y, cv=splits, method="predict_proba")[:, 1]
        oof_p = cross_val_predict(self.phys, X, y, cv=splits, method="predict_proba")[:, 1]
        self.clip.fit(clip, y)
        self.phys.fit(X, y)
        lc, lp = logit(oof_c), logit(oof_p)
        self.stack = LogisticRegression(class_weight="balanced", max_iter=2000).fit(
            _stack_design(lc, lp, q, False), y)
        self.stack_q = LogisticRegression(class_weight="balanced", max_iter=2000).fit(
            _stack_design(lc, lp, q, True), y)
        log(f"stack (no quality): {np.round(self.stack.coef_[0], 3)}")
        log(f"stack (quality-aware) [clip, phys, t, t*clip, t*phys]: {np.round(self.stack_q.coef_[0], 3)}")
        return self

    def branch_weights(self, q):
        """Effective stacking weights on the CLIP and physics logits at JPEG quality q."""
        w = self.stack_q.coef_[0]
        t = quality_term(np.asarray(q, float))
        return w[0] + w[3] * t, w[1] + w[4] * t

    def predict(self, clip, phys, q) -> dict:
        s_c = self.clip.predict_proba(clip)[:, 1]
        s_p = self.phys.predict_proba(phys[self.phys_cols].values)[:, 1]
        lc, lp = logit(s_c), logit(s_p)
        return {self.NAMES["clip"]: s_c, self.NAMES["phys"]: s_p,
                self.NAMES["fused"]: self.stack.predict_proba(_stack_design(lc, lp, q, False))[:, 1],
                self.NAMES["fused_q"]: self.stack_q.predict_proba(_stack_design(lc, lp, q, True))[:, 1]}