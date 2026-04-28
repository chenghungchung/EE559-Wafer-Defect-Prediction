"""
Basic NumPy XGBoost-style binary classifier for already-preprocessed tabular data.

Expected data contract:
- X_* contains no NaN / Inf values
- y_* are binary labels in {0, 1}
- preprocessing such as imputation / PCA is done upstream

This file intentionally keeps only the core boosting path:
- logistic loss
- histogram-based tree splits
- row subsampling and column subsampling
- L2 regularization, split penalty, class weighting, threshold sweep

Advanced extras used in earlier experiments such as SMOTE, focal loss, calibration,
cascades, bagging, and polynomial expansion were removed to keep the implementation
easy to inspect and tune.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pickle
import os
import time


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_DATA_PATH = DATA_DIR / "preprocessed_data_outlier_pca.pkl"
DEFAULT_DATASET_SUITE = [
    DATA_DIR / "preprocessed_data.pkl",
    DATA_DIR / "preprocessed_data_outlier.pkl",
    DATA_DIR / "preprocessed_data_outlier_pca.pkl",
    DATA_DIR / "preprocessed_data_outlier_vae_enhanced.pkl",
]

# =========================
# Loss Functions (Pluggable)
# =========================
class Loss:
    name: str = "base"

    def grad_hess(self, y_true: np.ndarray, y_pred_raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def loss(self, y_true: np.ndarray, y_pred_raw: np.ndarray) -> float:
        raise NotImplementedError

    def transform(self, y_pred_raw: np.ndarray) -> np.ndarray:
        return y_pred_raw


class LogisticLoss(Loss):
    name = "logloss"

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, -35.0, 35.0)
        return 1.0 / (1.0 + np.exp(-x))

    def grad_hess(self, y_true: np.ndarray, y_pred_raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        p = self._sigmoid(y_pred_raw)
        grad = p - y_true
        hess = p * (1.0 - p)
        return grad, hess

    def loss(self, y_true: np.ndarray, y_pred_raw: np.ndarray) -> float:
        p = np.clip(self._sigmoid(y_pred_raw), 1e-15, 1 - 1e-15)
        return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))

    def transform(self, y_pred_raw: np.ndarray) -> np.ndarray:
        return self._sigmoid(y_pred_raw)


# =========
# Metrics
# =========
class Metrics:
    @staticmethod
    def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        return np.array([[tn, fp], [fn, tp]], dtype=np.int64)

    @staticmethod
    def classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        cm = Metrics.confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        miss_rate = fn / max(1, tp + fn)
        false_alarm_rate = fp / max(1, fp + tn)

        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "miss_rate": float(miss_rate),
            "false_alarm_rate": float(false_alarm_rate),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "confusion_matrix": cm,
        }


def compute_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Compute ROC AUC from scratch using the rank-sum formulation.

    This implementation:
    - expects binary labels in {0, 1}
    - expects probability-like scores where larger means more positive
    - uses average ranks for tied scores

    Returns NaN when AUC is undefined because only one class is present.
    """
    y_true = np.asarray(y_true).reshape(-1).astype(np.int8)
    y_prob = np.asarray(y_prob).reshape(-1).astype(np.float64)

    if y_true.shape[0] != y_prob.shape[0]:
        raise ValueError("y_true and y_prob must have the same length.")

    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_prob, kind="mergesort")
    sorted_prob = y_prob[order]
    ranks = np.empty_like(sorted_prob, dtype=np.float64)

    start = 0
    n = sorted_prob.shape[0]
    while start < n:
        end = start + 1
        while end < n and sorted_prob[end] == sorted_prob[start]:
            end += 1

        avg_rank = 0.5 * ((start + 1) + end)
        ranks[start:end] = avg_rank
        start = end

    original_ranks = np.empty_like(ranks)
    original_ranks[order] = ranks
    sum_pos_ranks = float(original_ranks[y_true == 1].sum())

    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def threshold_sweep_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    start: float = 0.01,
    stop: float = 0.50,
    step: float = 0.01,
) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    """
    Sweep thresholds and compute precision / recall / F1 using NumPy only.

    Returns:
    - list of per-threshold metric dicts
    - best-threshold dict selected by max F1, then precision, then recall
    """
    y_true = np.asarray(y_true).reshape(-1).astype(np.int8)
    y_prob = np.asarray(y_prob).reshape(-1).astype(np.float64)

    thresholds = np.arange(start, stop + 0.5 * step, step, dtype=np.float64)
    rows: List[Dict[str, float]] = []
    best_row: Optional[Dict[str, float]] = None

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(np.int8)

        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))

        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2.0 * precision * recall / max(1e-12, precision + recall)

        row = {
            "threshold": float(threshold),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
        rows.append(row)

        if best_row is None or (
            row["f1"],
            row["precision"],
            row["recall"],
        ) > (
            best_row["f1"],
            best_row["precision"],
            best_row["recall"],
        ):
            best_row = row

    if best_row is None:
        best_row = {"threshold": float("nan"), "precision": float("nan"), "recall": float("nan"), "f1": float("nan")}

    return rows, best_row


# =====================
# Tree internal classes
# =====================
@dataclass
class SplitCandidate:
    feature: int
    split_bin: int
    gain_raw: float


@dataclass
class TreeNode:
    is_leaf: bool
    value: float
    sum_grad: float
    sum_hess: float
    depth: int
    feature_index: Optional[int] = None
    split_bin: Optional[int] = None
    gain_raw: float = 0.0
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None


class HistogramTree:
    """Depthwise histogram tree for binary logistic boosting."""

    def __init__(
        self,
        max_depth: int,
        min_child_weight: float,
        reg_lambda: float,
        gamma: float,
        colsample_bytree: float,
        n_bins: int,
        random_state: Optional[int] = None,
    ) -> None:
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.colsample_bytree = colsample_bytree
        self.n_bins = n_bins
        self.random_state = random_state

        self.root: Optional[TreeNode] = None
        self.feature_indices_: Optional[np.ndarray] = None
        self.gain_importance_: Optional[np.ndarray] = None
        self.freq_importance_: Optional[np.ndarray] = None

    def _leaf_value(self, G: float, H: float) -> float:
        return -G / (H + self.reg_lambda)

    def _score(self, G: np.ndarray, H: np.ndarray) -> np.ndarray:
        return (G * G) / (H + self.reg_lambda)

    def fit(self, X_bins: np.ndarray, grad: np.ndarray, hess: np.ndarray) -> None:
        n_samples, n_features = X_bins.shape
        rng = np.random.default_rng(self.random_state)

        n_cols = max(1, int(np.ceil(self.colsample_bytree * n_features)))
        self.feature_indices_ = np.sort(rng.choice(n_features, size=n_cols, replace=False))

        self.gain_importance_ = np.zeros(n_features, dtype=np.float64)
        self.freq_importance_ = np.zeros(n_features, dtype=np.float64)

        all_idx = np.arange(n_samples, dtype=np.int32)
        self.root = self._build_depthwise(X_bins, grad, hess, all_idx, depth=0)

    def _best_split(self, X_bins: np.ndarray, grad: np.ndarray, hess: np.ndarray, idx: np.ndarray) -> Optional[SplitCandidate]:
        if idx.size <= 1:
            return None

        g_node = grad[idx]
        h_node = hess[idx]
        G_total = g_node.sum()
        H_total = h_node.sum()
        parent_score = (G_total * G_total) / (H_total + self.reg_lambda)

        best_gain = -np.inf
        best_split = None

        for feat in self.feature_indices_:
            bins = X_bins[idx, feat]
            g_hist = np.bincount(bins, weights=g_node, minlength=self.n_bins).astype(np.float64)
            h_hist = np.bincount(bins, weights=h_node, minlength=self.n_bins).astype(np.float64)

            G_left = np.cumsum(g_hist)[:-1]
            H_left = np.cumsum(h_hist)[:-1]
            G_right = G_total - G_left
            H_right = H_total - H_left

            valid = (H_left >= self.min_child_weight) & (H_right >= self.min_child_weight)
            if not np.any(valid):
                continue

            gain = self._score(G_left, H_left) + self._score(G_right, H_right) - parent_score
            gain[~valid] = -np.inf

            split_bin = int(np.argmax(gain))
            gain_raw = float(gain[split_bin])
            if gain_raw > best_gain:
                best_gain = gain_raw
                best_split = SplitCandidate(feature=int(feat), split_bin=split_bin, gain_raw=gain_raw)

        return best_split

    def _split_indices(self, X_bins: np.ndarray, idx: np.ndarray, split: SplitCandidate) -> Tuple[np.ndarray, np.ndarray]:
        col = X_bins[idx, split.feature]
        left_mask = col <= split.split_bin
        if left_mask.sum() == 0 or left_mask.sum() == idx.size:
            return np.array([], dtype=np.int32), np.array([], dtype=np.int32)
        return idx[left_mask], idx[~left_mask]

    def _build_depthwise(self, X_bins: np.ndarray, grad: np.ndarray, hess: np.ndarray, idx: np.ndarray, depth: int) -> TreeNode:
        G = float(grad[idx].sum())
        H = float(hess[idx].sum())
        node = TreeNode(True, self._leaf_value(G, H), G, H, depth)

        if depth >= self.max_depth or H < self.min_child_weight or idx.size <= 1:
            return node

        split = self._best_split(X_bins, grad, hess, idx)
        if split is None or split.gain_raw <= self.gamma:
            return node

        left_idx, right_idx = self._split_indices(X_bins, idx, split)
        if left_idx.size == 0:
            return node

        node.is_leaf = False
        node.feature_index = split.feature
        node.split_bin = split.split_bin
        node.gain_raw = split.gain_raw

        self.gain_importance_[split.feature] += split.gain_raw
        self.freq_importance_[split.feature] += 1.0

        node.left = self._build_depthwise(X_bins, grad, hess, left_idx, depth + 1)
        node.right = self._build_depthwise(X_bins, grad, hess, right_idx, depth + 1)
        return node

    def _predict_node(self, X_bins: np.ndarray, idx: np.ndarray, node: TreeNode, out: np.ndarray) -> None:
        if idx.size == 0:
            return
        if node.is_leaf:
            out[idx] = node.value
            return

        if node.feature_index is None or node.split_bin is None or node.left is None or node.right is None:
            raise RuntimeError("Malformed tree node.")

        left_mask = X_bins[idx, node.feature_index] <= node.split_bin
        self._predict_node(X_bins, idx[left_mask], node.left, out)
        self._predict_node(X_bins, idx[~left_mask], node.right, out)

    def predict(self, X_bins: np.ndarray) -> np.ndarray:
        if self.root is None:
            raise RuntimeError("Tree is not fitted.")
        out = np.empty(X_bins.shape[0], dtype=np.float64)
        self._predict_node(X_bins, np.arange(X_bins.shape[0]), self.root, out)
        return out


@dataclass
class TrainingLog:
    iteration: int
    train_loss: float
    valid_loss: Optional[float] = None


class Booster:
    def __init__(
        self,
        n_estimators: int = 200,
        learning_rate: float = 0.1,
        max_depth: int = 4,
        min_child_weight: float = 1.0,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_lambda: float = 1.0,
        gamma: float = 0.0,
        n_bins: int = 64,
        scale_pos_weight: float = 1.0,
        auto_scale_pos_weight: bool = True,
        base_score: Optional[float] = None,
        early_stopping_rounds: Optional[int] = None,
        early_stopping_metric: str = "loss",
        random_state: Optional[int] = 42,
        verbose: bool = True,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.n_bins = n_bins
        self.scale_pos_weight = scale_pos_weight
        self.auto_scale_pos_weight = auto_scale_pos_weight
        self.base_score = base_score
        self.early_stopping_rounds = early_stopping_rounds
        self.early_stopping_metric = early_stopping_metric
        self.random_state = random_state
        self.verbose = verbose

        self.loss_fn: Loss = LogisticLoss()

        self.trees: List[HistogramTree] = []
        self.logs_: List[TrainingLog] = []
        self.best_iteration_: Optional[int] = None
        self.best_valid_loss_: Optional[float] = None

        self.bin_edges_: Optional[List[np.ndarray]] = None
        self.n_features_: Optional[int] = None
        self.feature_importance_gain_: Optional[np.ndarray] = None
        self.feature_importance_freq_: Optional[np.ndarray] = None
        self.base_margin_: float = 0.0
        self.resolved_scale_pos_weight_: float = float(scale_pos_weight)

    # ====== Validation for preprocessed data ======
    @staticmethod
    def _assert_no_nan(X: np.ndarray, name: str) -> None:
        assert np.isfinite(X).all(), f"{name} must not contain NaN/Inf (preprocessing contract violated)."

    def _assert_binary_labels(self, y: np.ndarray, name: str) -> None:
        unique = np.unique(y)
        assert np.array_equal(unique, np.array([0, 1])) or np.array_equal(unique, np.array([0])) or np.array_equal(unique, np.array([1])), (
            f"{name} must contain only labels {{0,1}}; got {unique}."
        )

    # ====== Stable binning on scaled data ======
    def _build_bin_edges(self, X: np.ndarray) -> List[np.ndarray]:
        """
        Quantile binning remains valid on standardized data.
        Standardization is monotonic per feature, so ordering/rank is preserved.
        """
        quantiles = np.linspace(0.0, 1.0, self.n_bins + 1)[1:-1]
        edges: List[np.ndarray] = []
        eps = 1e-12

        for j in range(X.shape[1]):
            col = X[:, j]
            # Fix 1: near-constant features become non-splittable.
            if np.std(col) < 1e-8:
                edges.append(np.array([], dtype=np.float64))
                continue

            q = np.quantile(col, quantiles)

            # Fix 4: remove duplicates from collapsed quantiles.
            q = np.unique(q)

            # Fix 4: if only one unique edge remains, treat as no-split feature.
            if q.size <= 1:
                edges.append(np.array([], dtype=np.float64))
                continue

            # Fix 4: enforce strict monotonicity with tiny epsilon for stable searchsorted.
            if q.size > 1:
                q = np.maximum.accumulate(q + eps * np.arange(q.size))
            edges.append(q.astype(np.float64))

        return edges

    def _transform_to_bins(self, X: np.ndarray) -> np.ndarray:
        if self.bin_edges_ is None:
            raise RuntimeError("Model is not fitted.")

        X_bins = np.empty(X.shape, dtype=np.int16)
        for j, edges in enumerate(self.bin_edges_):
            X_bins[:, j] = np.searchsorted(edges, X[:, j], side="right").astype(np.int16)
        return X_bins

    def _resolve_base_margin(self, y: np.ndarray) -> float:
        if self.base_score is not None:
            p = float(np.clip(self.base_score, 1e-6, 1.0 - 1e-6))
            return float(np.log(p / (1.0 - p)))

        pos_rate = float(np.clip(np.mean(y), 1e-6, 1.0 - 1e-6))
        return float(np.log(pos_rate / (1.0 - pos_rate)))

    def _apply_class_weight(self, y: np.ndarray, grad: np.ndarray, hess: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.resolved_scale_pos_weight_ == 1.0:
            return grad, hess
        pos = y == 1
        grad[pos] *= self.resolved_scale_pos_weight_
        hess[pos] *= self.resolved_scale_pos_weight_
        return grad, hess

    def fit(self, X: np.ndarray, y: np.ndarray, eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> "Booster":
        # Required compatibility checks for already-preprocessed datasets.
        self._assert_no_nan(X, "X_train")
        self._assert_binary_labels(y, "y_train")

        rng = np.random.default_rng(self.random_state)
        n_samples, n_features = X.shape
        self.n_features_ = n_features
        self.trees = []
        self.logs_ = []
        self.best_iteration_ = None
        self.best_valid_loss_ = None

        # Fix 2: explicit reset per fit call to avoid accumulation across refits.
        self.feature_importance_gain_ = np.zeros(n_features, dtype=np.float64)
        self.feature_importance_freq_ = np.zeros(n_features, dtype=np.float64)

        self.resolved_scale_pos_weight_ = float(self.scale_pos_weight)
        if self.auto_scale_pos_weight and self.resolved_scale_pos_weight_ == 1.0:
            n_pos = int(np.sum(y == 1))
            n_neg = int(np.sum(y == 0))
            self.resolved_scale_pos_weight_ = float(np.sqrt(n_neg / max(1, n_pos)))
            if self.verbose:
                print(f"  auto scale_pos_weight={self.resolved_scale_pos_weight_:.3f} "
                      f"(pos={n_pos}, neg={n_neg})")

        self.base_margin_ = self._resolve_base_margin(y)
        self.bin_edges_ = self._build_bin_edges(X)
        X_bins = self._transform_to_bins(X)

        X_val_bins, y_val = None, None
        if eval_set is not None:
            X_val, y_val = eval_set
            self._assert_no_nan(X_val, "X_val")
            self._assert_binary_labels(y_val, "y_val")
            # Fix 3: ensure feature dimension consistency between train and validation.
            assert X_val.shape[1] == n_features, "Feature mismatch between training and validation."
            X_val_bins = self._transform_to_bins(X_val)

        pred_raw = np.full(n_samples, self.base_margin_, dtype=np.float64)
        pred_val_raw = (
            np.full(X_val_bins.shape[0], self.base_margin_, dtype=np.float64)
            if eval_set is not None
            else None
        )

        best_valid = np.inf
        rounds_no_improve = 0

        for t in range(self.n_estimators):
            if self.verbose:
                print(f"Iteration {t + 1}")
            grad, hess = self.loss_fn.grad_hess(y, pred_raw)
            grad, hess = self._apply_class_weight(y, grad, hess)

            n_rows = max(1, int(np.ceil(self.subsample * n_samples)))
            row_idx = rng.choice(n_samples, size=n_rows, replace=False)

            tree = HistogramTree(
                max_depth=self.max_depth,
                min_child_weight=self.min_child_weight,
                reg_lambda=self.reg_lambda,
                gamma=self.gamma,
                colsample_bytree=self.colsample_bytree,
                n_bins=self.n_bins,
                random_state=int(rng.integers(0, 1_000_000_000)),
            )
            tree.fit(X_bins[row_idx], grad[row_idx], hess[row_idx])
            self.trees.append(tree)

            pred_raw += self.learning_rate * tree.predict(X_bins)
            self.feature_importance_gain_ += tree.gain_importance_
            self.feature_importance_freq_ += tree.freq_importance_

            train_loss = self.loss_fn.loss(y, pred_raw)
            log = TrainingLog(iteration=t, train_loss=train_loss)

            if eval_set is not None:
                pred_val_raw += self.learning_rate * tree.predict(X_val_bins)
                valid_loss = self.loss_fn.loss(y_val, pred_val_raw)
                log.valid_loss = valid_loss

                # Label-aware early stopping: monitor val F1 (sweep thresholds)
                # instead of val log-loss, so the stopping criterion aligns with
                # what we actually care about — separating defects from good wafers.
                if self.early_stopping_metric == "f1":
                    p_val = self.loss_fn.transform(pred_val_raw)
                    _f1_thresholds = [0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30]
                    _best_f1_val = -1.0
                    for _ft in _f1_thresholds:
                        _yp = (p_val >= _ft).astype(np.int8)
                        _rep = Metrics.classification_report(y_val, _yp)
                        if _rep["f1"] > _best_f1_val:
                            _best_f1_val = _rep["f1"]
                    # Negate so "lower is better" convention is preserved
                    monitor_val = -_best_f1_val
                else:
                    monitor_val = valid_loss

                if monitor_val < best_valid - 1e-12:
                    best_valid = monitor_val
                    self.best_valid_loss_ = valid_loss
                    self.best_iteration_ = t
                    rounds_no_improve = 0
                else:
                    rounds_no_improve += 1

                if self.verbose:
                    if self.early_stopping_metric == "f1":
                        print(
                            f"  train_loss={train_loss:.6f} valid_loss={valid_loss:.6f} "
                            f"val_f1={-monitor_val:.4f} no_improve={rounds_no_improve}"
                        )
                    else:
                        print(f"  train_loss={train_loss:.6f} valid_loss={valid_loss:.6f} no_improve={rounds_no_improve}")

                if self.early_stopping_rounds is not None and rounds_no_improve >= self.early_stopping_rounds:
                    if self.verbose:
                        print(f"Early stopping at iter={t}, best_iter={self.best_iteration_}, best_valid={best_valid:.6f}")
                    break
            else:
                if self.verbose:
                    print(f"  train_loss={train_loss:.6f}")

            self.logs_.append(log)

        if (
            self.early_stopping_rounds is not None
            and eval_set is not None
            and self.best_iteration_ is not None
        ):
            self.trees = self.trees[: self.best_iteration_ + 1]
            self.feature_importance_gain_.fill(0.0)
            self.feature_importance_freq_.fill(0.0)
            for tree in self.trees:
                self.feature_importance_gain_ += tree.gain_importance_
                self.feature_importance_freq_ += tree.freq_importance_

        return self

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        self._assert_no_nan(X, "X_predict")
        # Fix 3: ensure feature dimension consistency between training and inference.
        assert self.bin_edges_ is not None, "Model is not fitted."
        assert X.shape[1] == len(self.bin_edges_), "Feature mismatch between training and inference."
        X_bins = self._transform_to_bins(X)
        raw = np.full(X.shape[0], self.base_margin_, dtype=np.float64)
        for tree in self.trees:
            raw += self.learning_rate * tree.predict(X_bins)
        return raw

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = self.predict_raw(X)
        return self.loss_fn.transform(raw)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        raw = self.predict_raw(X)
        p = self.loss_fn.transform(raw)
        return (p >= threshold).astype(np.int8)

    def feature_importance(self, importance_type: str = "gain", normalize: bool = True) -> np.ndarray:
        if self.feature_importance_gain_ is None:
            raise RuntimeError("Model is not fitted.")

        imp = self.feature_importance_freq_.copy() if importance_type == "weight" else self.feature_importance_gain_.copy()
        if normalize:
            s = imp.sum()
            if s > 0:
                imp /= s
        return imp


class XGBoost(Booster):
    """Alias for backwards-compatible API name."""


SEARCH_MODEL_KWARGS = {
    "auto_scale_pos_weight": False,
    "base_score": None,
    "early_stopping_rounds": None,
    "early_stopping_metric": "loss",
    "random_state": 42,
    "verbose": False,
}


def _make_search_model(model_params: Dict[str, object]) -> XGBoost:
    return XGBoost(**SEARCH_MODEL_KWARGS, **model_params)


def _evaluate_probabilities(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> Tuple[Dict[str, float], float]:
    y_pred = (y_prob >= threshold).astype(np.int8)
    report = Metrics.classification_report(y_true, y_pred)
    auc = compute_auc(y_true, y_prob)
    return report, auc


def _evaluate_model_split(
    model: XGBoost,
    X: np.ndarray,
    y: np.ndarray,
    threshold: float,
) -> Tuple[np.ndarray, Dict[str, float], float]:
    y_prob = model.predict_proba(X)
    report, auc = _evaluate_probabilities(y, y_prob, threshold)
    return y_prob, report, auc


def _build_search_presets() -> List[Dict[str, object]]:
    return [
        dict(
            name="nb_depth3_300",
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            min_child_weight=3.0,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=3.0,
            gamma=0.0,
            scale_pos_weight=14.01369863,
            n_bins=64,
        ),
        dict(
            name="nb_depth3_500",
            n_estimators=500,
            learning_rate=0.03,
            max_depth=3,
            min_child_weight=5.0,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=3.0,
            gamma=0.1,
            scale_pos_weight=14.01369863,
            n_bins=64,
        ),
        dict(
            name="nb_depth4_500",
            n_estimators=500,
            learning_rate=0.05,
            max_depth=4,
            min_child_weight=3.0,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=3.0,
            gamma=0.1,
            scale_pos_weight=14.01369863,
            n_bins=64,
        ),
        dict(
            name="nb_depth5_500",
            n_estimators=500,
            learning_rate=0.03,
            max_depth=5,
            min_child_weight=3.0,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=3.0,
            gamma=0.1,
            scale_pos_weight=14.01369863,
            n_bins=64,
        ),
        dict(
            name="nb_depth4_800",
            n_estimators=800,
            learning_rate=0.03,
            max_depth=4,
            min_child_weight=1.0,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=3.0,
            gamma=0.2,
            scale_pos_weight=14.01369863,
            n_bins=64,
        ),
        dict(
            name="nb_depth5_800",
            n_estimators=800,
            learning_rate=0.02,
            max_depth=5,
            min_child_weight=5.0,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=3.0,
            gamma=0.2,
            scale_pos_weight=14.01369863,
            n_bins=64,
        ),
        dict(
            name="nb_best233_spw28",
            n_estimators=233,
            learning_rate=0.03,
            max_depth=5,
            min_child_weight=3.0,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=3.0,
            gamma=0.1,
            scale_pos_weight=28.02739726,
            n_bins=64,
        ),
        dict(
            name="balanced_d2_spw1",
            n_estimators=800,
            learning_rate=0.02,
            max_depth=2,
            min_child_weight=5.0,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=8.0,
            gamma=0.0,
            scale_pos_weight=1.0,
            n_bins=64,
        ),
        dict(
            name="balanced_d3_spw2",
            n_estimators=600,
            learning_rate=0.03,
            max_depth=3,
            min_child_weight=5.0,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=6.0,
            gamma=0.1,
            scale_pos_weight=2.0,
            n_bins=64,
        ),
        dict(
            name="guarded_recall_spw4",
            n_estimators=600,
            learning_rate=0.03,
            max_depth=3,
            min_child_weight=6.0,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=8.0,
            gamma=0.2,
            scale_pos_weight=4.0,
            n_bins=64,
        ),
        dict(
            name="mid_spw6_d3",
            n_estimators=500,
            learning_rate=0.03,
            max_depth=3,
            min_child_weight=5.0,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=6.0,
            gamma=0.15,
            scale_pos_weight=6.0,
            n_bins=64,
        ),
        dict(
            name="mid_spw8_d3",
            n_estimators=500,
            learning_rate=0.03,
            max_depth=3,
            min_child_weight=5.0,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=6.0,
            gamma=0.15,
            scale_pos_weight=8.0,
            n_bins=64,
        ),
        dict(
            name="mid_spw10_d3",
            n_estimators=500,
            learning_rate=0.03,
            max_depth=3,
            min_child_weight=5.0,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=6.0,
            gamma=0.15,
            scale_pos_weight=10.0,
            n_bins=64,
        ),
        dict(
            name="f1_d3_spw12",
            n_estimators=600,
            learning_rate=0.025,
            max_depth=3,
            min_child_weight=6.0,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=8.0,
            gamma=0.2,
            scale_pos_weight=12.0,
            n_bins=64,
        ),
        dict(
            name="precision_d4_spw1",
            n_estimators=500,
            learning_rate=0.03,
            max_depth=4,
            min_child_weight=5.0,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=10.0,
            gamma=0.2,
            scale_pos_weight=1.0,
            n_bins=64,
        ),
    ]


def train_and_evaluate_preprocessed(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    """Example helper for externally preprocessed dataset usage."""
    model = XGBoost(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=2.0,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        gamma=0.1,
        n_bins=64,
        scale_pos_weight=1.0,
        early_stopping_rounds=20,
        random_state=42,
        verbose=True,
    )

    model.fit(X_train, y_train, eval_set=(X_val, y_val))
    y_pred = model.predict(X_test)
    return Metrics.classification_report(y_test, y_pred)


def _as_numpy(name: str, arr, verbose: bool = False) -> np.ndarray:
    out = np.asarray(arr)
    if verbose:
        print(f"{name}: shape={out.shape}, dtype={out.dtype}")
    return out


def _normalize_binary_labels(y: np.ndarray, name: str, verbose: bool = False) -> np.ndarray:
    y = np.asarray(y).reshape(-1)
    uniq = np.unique(y)
    if np.array_equal(uniq, np.array([-1, 1])) or np.array_equal(uniq, np.array([-1])) or np.array_equal(uniq, np.array([1])):
        if verbose:
            print(f"{name}: converting labels from {{-1,1}} to {{0,1}}")
        y = ((y + 1) // 2).astype(np.int8)
    return y.astype(np.int8)


def _detect_key(d: dict, candidates: List[str]):
    lower_map = {str(k).lower(): k for k in d.keys()}
    for c in candidates:
        if c in lower_map:
            return lower_map[c]
    return None


def _extract_splits(data_obj):
    X_train = y_train = X_val = y_val = X_test = y_test = None

    if isinstance(data_obj, dict):
        key_lists = {
            "X_train": ["x_train", "train_x", "xtr", "features_train"],
            "y_train": ["y_train", "train_y", "ytr", "labels_train"],
            "X_val": ["x_val", "x_valid", "val_x", "valid_x", "features_val"],
            "y_val": ["y_val", "y_valid", "val_y", "valid_y", "labels_val"],
            "X_test": ["x_test", "test_x", "xte", "features_test"],
            "y_test": ["y_test", "test_y", "yte", "labels_test"],
        }
        resolved = {name: _detect_key(data_obj, cands) for name, cands in key_lists.items()}
        if resolved["X_train"] is not None and resolved["y_train"] is not None:
            X_train = data_obj[resolved["X_train"]]
            y_train = data_obj[resolved["y_train"]]
            X_val = data_obj[resolved["X_val"]] if resolved["X_val"] is not None else None
            y_val = data_obj[resolved["y_val"]] if resolved["y_val"] is not None else None
            X_test = data_obj[resolved["X_test"]] if resolved["X_test"] is not None else None
            y_test = data_obj[resolved["y_test"]] if resolved["y_test"] is not None else None
        else:
            arrays = [v for v in data_obj.values() if hasattr(v, "shape") or isinstance(v, (list, tuple))]
            if len(arrays) >= 2:
                X_train, y_train = arrays[0], arrays[1]
            if len(arrays) >= 4:
                X_val, y_val = arrays[2], arrays[3]
            if len(arrays) >= 6:
                X_test, y_test = arrays[4], arrays[5]
    elif isinstance(data_obj, (list, tuple)):
        if len(data_obj) >= 2:
            X_train, y_train = data_obj[0], data_obj[1]
        if len(data_obj) >= 4:
            X_val, y_val = data_obj[2], data_obj[3]
        if len(data_obj) >= 6:
            X_test, y_test = data_obj[4], data_obj[5]

    if X_train is None or y_train is None:
        raise ValueError("Could not auto-detect X_train/y_train from the pickle payload.")

    return X_train, y_train, X_val, y_val, X_test, y_test


def _performance_band(report: Optional[Dict[str, float]]) -> Tuple[str, str]:
    """
    Context-specific quality bands for this defect-detection task.

    Because the dataset is highly imbalanced, accuracy is treated as a floor
    rather than the main driver. We mainly reward recall first, then require
    precision / F1 to be high enough for the model to be practical.
    """
    if report is None:
        return "n/a", "no report"

    accuracy = report["accuracy"]
    precision = report["precision"]
    recall = report["recall"]
    f1 = report["f1"]

    if accuracy >= 0.88 and precision >= 0.35 and recall >= 0.75 and f1 >= 0.45:
        return "strong", "high recall with manageable false positives"
    if accuracy >= 0.85 and precision >= 0.30 and recall >= 0.70 and f1 >= 0.40:
        return "good", "solid screening model for this stage"
    if accuracy >= 0.80 and precision >= 0.20 and recall >= 0.50 and f1 >= 0.30:
        return "usable", "directionally useful, but still noisy"
    return "weak", "still too noisy or misses too many positives"


def _split_summary(name: str, X: np.ndarray, y: np.ndarray) -> str:
    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))
    return f"{name}: {X.shape[0]} x {X.shape[1]} (pos={positives}, neg={negatives})"


def _format_metric_line(report: Dict[str, float], auc: float) -> str:
    text = (
        f"acc={report['accuracy']:.4f} "
        f"prec={report['precision']:.4f} "
        f"rec={report['recall']:.4f} "
        f"f1={report['f1']:.4f}"
    )
    if not np.isnan(float(auc)):
        text += f" auc={float(auc):.4f}"
    text += f" fn={int(report['fn'])} miss={report['miss_rate']:.4f}"
    return text


def _format_table_number(value: float, width: int = 6) -> str:
    if value is None or np.isnan(float(value)):
        return f"{'N/A':>{width}}"
    return f"{float(value):>{width}.3f}"


def _format_table_metric(report: Optional[Dict[str, float]], key: str, width: int = 6) -> str:
    if report is None:
        return f"{'N/A':>{width}}"
    return _format_table_number(float(report[key]), width=width)


def _format_table_count(report: Optional[Dict[str, float]], key: str, width: int = 4) -> str:
    if report is None:
        return f"{'N/A':>{width}}"
    return f"{int(report[key]):>{width}}"


def _print_experiment_table(search_results: List[Dict[str, object]]) -> None:
    print("Experiment results")
    print(
        f"{'No':>2} {'Preset':<20} {'Th':>5} {'Time':>6} "
        f"{'VAcc':>6} {'VPre':>6} {'VRec':>6} {'VF1':>6} {'VAUC':>6} "
        f"{'TAcc':>6} {'TPre':>6} {'TRec':>6} {'TF1':>6} {'TAUC':>6} {'Grade':>8}"
    )
    for idx, item in enumerate(search_results, start=1):
        val_report = item["val_report"]
        test_report = item["test_report"]
        grade = item["test_band"] if test_report is not None else item["val_band"]
        print(
            f"{idx:>2} {str(item['name']):<20} "
            f"{float(item['threshold']):>5.2f} {float(item['train_sec']):>6.2f} "
            f"{_format_table_metric(val_report, 'accuracy')} "
            f"{_format_table_metric(val_report, 'precision')} "
            f"{_format_table_metric(val_report, 'recall')} "
            f"{_format_table_metric(val_report, 'f1')} "
            f"{_format_table_number(float(item['val_auc']), width=6)} "
            f"{_format_table_metric(test_report, 'accuracy')} "
            f"{_format_table_metric(test_report, 'precision')} "
            f"{_format_table_metric(test_report, 'recall')} "
            f"{_format_table_metric(test_report, 'f1')} "
            f"{_format_table_number(float(item['test_auc']), width=6)} "
            f"{str(grade):>8}"
        )


def _print_defect_safety_table(search_results: List[Dict[str, object]]) -> None:
    print("\nDefect-safety view")
    print(
        f"{'No':>2} {'Preset':<20} {'Th':>5} "
        f"{'VFN':>4} {'VMiss':>6} {'VFA':>6} "
        f"{'TFN':>4} {'TMiss':>6} {'TFA':>6}"
    )
    for idx, item in enumerate(search_results, start=1):
        val_report = item["val_report"]
        test_report = item["test_report"]
        print(
            f"{idx:>2} {str(item['name']):<20} "
            f"{float(item['threshold']):>5.2f} "
            f"{_format_table_count(val_report, 'fn')} "
            f"{_format_table_metric(val_report, 'miss_rate')} "
            f"{_format_table_metric(val_report, 'false_alarm_rate')} "
            f"{_format_table_count(test_report, 'fn')} "
            f"{_format_table_metric(test_report, 'miss_rate')} "
            f"{_format_table_metric(test_report, 'false_alarm_rate')}"
        )


def _print_feature_summary(X: np.ndarray, feature_names: List[str]) -> None:
    print(f"\nFeature summary ({X.shape[0]} samples x {X.shape[1]} features)")
    print(f"{'Feature':<10} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}  {'Skew':>8}")
    for idx, name in enumerate(feature_names):
        col = X[:, idx]
        mean = col.mean()
        std = col.std()
        min_value = col.min()
        max_value = col.max()
        skew = float(np.mean(((col - mean) / (std + 1e-12)) ** 3))
        print(
            f"{name:<10} {mean:>10.4f} {std:>10.4f} "
            f"{min_value:>10.4f} {max_value:>10.4f}  {skew:>8.3f}"
        )


def _search_result_key(
    item: Dict[str, object],
    policy: str,
    min_accuracy: float,
    target_recall: float = 0.0,
    max_false_alarm_rate: float = 1.0,
) -> Tuple[float, ...]:
    report = item["val_report"]
    if report is None:
        return (float("-inf"),)

    if policy == "selected":
        if item.get("selection_policy") == "recall_guard":
            return (
                1 if report["recall"] >= target_recall else 0,
                1 if report["false_alarm_rate"] <= max_false_alarm_rate else 0,
                1 if item["meets_target"] else 0,
                report["recall"],
                -report["false_alarm_rate"],
                report["precision"],
                report["f1"],
                report["accuracy"],
            )
        return (
            1 if item["meets_target"] else 0,
            report["recall"],
            report["precision"],
            report["f1"],
            report["accuracy"],
        )

    if policy == "recall":
        return (
            1 if report["accuracy"] >= min_accuracy else 0,
            report["recall"],
            report["precision"],
            report["f1"],
            report["accuracy"],
        )

    if policy == "balanced":
        return (
            1 if report["accuracy"] >= min_accuracy else 0,
            report["f1"],
            report["recall"],
            report["precision"],
            report["accuracy"],
        )

    if policy == "precision":
        return (
            1 if report["accuracy"] >= min_accuracy else 0,
            1 if report["recall"] >= 0.25 else 0,
            report["precision"],
            report["f1"],
            report["accuracy"],
        )

    raise ValueError(f"Unknown search policy: {policy}")


def _print_notable_results(search_results: List[Dict[str, object]], min_accuracy: float) -> None:
    picks = [
        ("Recall-first", max(search_results, key=lambda item: _search_result_key(item, "recall", min_accuracy))),
        ("Balanced-F1", max(search_results, key=lambda item: _search_result_key(item, "balanced", min_accuracy))),
        ("Precision-leaning", max(search_results, key=lambda item: _search_result_key(item, "precision", min_accuracy))),
    ]

    print("\nValidation highlights")
    for label, item in picks:
        report = item["val_report"]
        print(
            f"{label:<18} {item['name']:<18} th={item['threshold']:.2f}  "
            f"A/P/R/F1={report['accuracy']:.3f}/{report['precision']:.3f}/{report['recall']:.3f}/{report['f1']:.3f}  "
            f"AUC={item['val_auc']:.3f}  grade={item['val_band']}"
        )


def _print_threshold_reference(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: List[float],
) -> None:
    print(f"{'Threshold':>10} {'Accuracy':>10} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Meets(50/50)':>14}")
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(np.int8)
        report = Metrics.classification_report(y_true, y_pred)
        meets_goal = "✓" if report["precision"] >= 0.50 and report["recall"] >= 0.50 else ""
        print(
            f"{threshold:>10.2f} {report['accuracy']:>10.4f} {report['precision']:>10.4f} "
            f"{report['recall']:>8.4f} {report['f1']:>8.4f} {meets_goal:>14}"
        )


def _print_threshold_sweep(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    start: float = 0.01,
    stop: float = 0.50,
    step: float = 0.01,
) -> None:
    sweep_rows, best_sweep = threshold_sweep_analysis(y_true, y_prob, start=start, stop=stop, step=step)
    print("\nThreshold sweep")
    print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    for row in sweep_rows:
        print(
            f"{row['threshold']:>10.2f} {row['precision']:>10.4f} "
            f"{row['recall']:>8.4f} {row['f1']:>8.4f}"
        )

    print("\nBest threshold by F1:")
    print(f"  threshold = {best_sweep['threshold']:.2f}")
    print(f"  precision = {best_sweep['precision']:.4f}")
    print(f"  recall    = {best_sweep['recall']:.4f}")
    print(f"  f1        = {best_sweep['f1']:.4f}")


def _load_and_train_from_pickle(
    path: Union[str, Path] = DEFAULT_DATA_PATH,
    verbose: bool = False,
) -> Dict[str, object]:
    path = os.fspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find file: {path}")

    dataset_name = os.path.basename(path)

    with open(path, "rb") as f:
        data_obj = pickle.load(f)

    note = ""
    n_components = None
    explained_var = None
    if isinstance(data_obj, dict):
        note = data_obj.get("note", "")
        n_components = data_obj.get("n_components", None)
        explained_var = data_obj.get("explained_variance", None)

    if verbose:
        print(f"\nDataset: {dataset_name}")
        print("Raw data")
        print("  type:", type(data_obj))
        if isinstance(data_obj, dict):
            print("  keys:", list(data_obj.keys()))
            for k, v in data_obj.items():
                if hasattr(v, "shape"):
                    print(f"  {k}: shape={v.shape}, dtype={getattr(v, 'dtype', '?')}")
                elif not isinstance(v, (np.ndarray, list)):
                    print(f"  {k}: {v}")

    X_train, y_train, X_val, y_val, X_test, y_test = _extract_splits(data_obj)

    X_train = _as_numpy("X_train", X_train, verbose=verbose)
    y_train = _normalize_binary_labels(_as_numpy("y_train", y_train, verbose=verbose), "y_train", verbose=verbose)

    if X_val is not None and y_val is not None:
        X_val = _as_numpy("X_val", X_val, verbose=verbose)
        y_val = _normalize_binary_labels(_as_numpy("y_val", y_val, verbose=verbose), "y_val", verbose=verbose)

    if X_test is not None and y_test is not None:
        X_test = _as_numpy("X_test", X_test, verbose=verbose)
        y_test = _normalize_binary_labels(_as_numpy("y_test", y_test, verbose=verbose), "y_test", verbose=verbose)

    print("\n" + "=" * 120)
    print(f"Dataset: {dataset_name}")
    print(f"  {_split_summary('train', X_train, y_train)}")
    if X_val is not None and y_val is not None:
        print(f"  {_split_summary('val', X_val, y_val)}")
    if X_test is not None and y_test is not None:
        print(f"  {_split_summary('test', X_test, y_test)}")
    if note:
        print(f"  note: {note}")
    if n_components is not None:
        print(f"  pca components: {n_components}")
    if explained_var is not None:
        print(f"  explained variance: {float(explained_var):.4f}")

    n_feat = X_train.shape[1]
    if isinstance(data_obj, dict) and data_obj.get("n_components") is not None:
        feat_names = [f"PC{i + 1}" for i in range(n_feat)]
    else:
        feat_names = [f"feat_{i}" for i in range(n_feat)]

    if verbose:
        _print_feature_summary(X_train, feat_names)

    has_eval = X_val is not None and y_val is not None
    thresholds = [float(t) for t in np.arange(0.01, 0.801, 0.005)]
    selection_min_accuracy = 0.0
    selection_target_recall = 0.80
    selection_max_false_alarm_rate = 0.50
    selection_primary_metric = "recall_guard"

    param_grid = _build_search_presets()

    print(
        f"  trying {len(param_grid)} preset settings "
        f"(selection: {selection_primary_metric}, target val recall={selection_target_recall:.2f}, "
        f"max val false alarm={selection_max_false_alarm_rate:.2f})"
    )

    search_results = []
    for idx, config in enumerate(param_grid, start=1):
        name = config["name"]
        model_params = {k: v for k, v in config.items() if k != "name"}
        t0 = time.perf_counter()
        model = _make_search_model(model_params)
        model.fit(X_train, y_train, eval_set=(X_val, y_val) if has_eval else None)
        train_sec = time.perf_counter() - t0

        if has_eval:
            val_prob = model.predict_proba(X_val)
            best_t, _, meets = _best_threshold(
                y_val,
                val_prob,
                thresholds,
                min_accuracy=selection_min_accuracy,
                min_recall=selection_target_recall,
                max_false_alarm_rate=selection_max_false_alarm_rate,
                primary_metric=selection_primary_metric,
            )
            val_report, val_auc = _evaluate_probabilities(y_val, val_prob, best_t)
            val_band, _ = _performance_band(val_report)
        else:
            best_t = 0.50
            meets = False
            val_report = None
            val_band = "n/a"
            val_auc = float("nan")

        test_report = None
        test_band = "n/a"
        test_auc = float("nan")
        if X_test is not None and y_test is not None:
            _, test_report, test_auc = _evaluate_model_split(model, X_test, y_test, best_t)
            test_band, _ = _performance_band(test_report)

        search_results.append(
            {
                "name": name,
                "params": model_params,
                "threshold": best_t,
                "meets_target": meets,
                "selection_policy": selection_primary_metric,
                "target_recall": selection_target_recall,
                "max_false_alarm_rate": selection_max_false_alarm_rate,
                "best_iteration": model.best_iteration_,
                "train_sec": train_sec,
                "val_report": val_report,
                "val_band": val_band,
                "val_auc": val_auc,
                "test_report": test_report,
                "test_band": test_band,
                "test_auc": test_auc,
            }
        )

    if has_eval:
        best_result = max(
            search_results,
            key=lambda item: _search_result_key(
                item,
                "selected",
                selection_min_accuracy,
                target_recall=selection_target_recall,
                max_false_alarm_rate=selection_max_false_alarm_rate,
            ),
        )
    else:
        best_result = search_results[0]

    _print_experiment_table(search_results)
    _print_defect_safety_table(search_results)

    if verbose and has_eval:
        _print_notable_results(search_results, selection_min_accuracy)

    print("\nBest by current rule")
    print(
        f"  preset: {best_result['name']} | "
        f"threshold={best_result['threshold']:.2f} | "
        f"time={best_result['train_sec']:.2f}s | "
        f"target_recall={selection_target_recall:.2f} | "
        f"max_false_alarm={selection_max_false_alarm_rate:.2f} | "
        f"target_met={best_result['meets_target']}"
    )
    if verbose:
        print("  params:", best_result["params"])
    if has_eval and best_result["val_report"] is not None:
        vr = best_result["val_report"]
        print(f"  validation: {_format_metric_line(vr, best_result['val_auc'])} | {best_result['val_band']}")

    if X_test is None or y_test is None:
        print("  test: skipped")
        return {
            "dataset": dataset_name,
            "path": path,
            "best_preset": best_result["name"],
            "best_params": best_result["params"],
            "threshold": best_result["threshold"],
            "selection_policy": selection_primary_metric,
            "target_recall": selection_target_recall,
            "max_false_alarm_rate": selection_max_false_alarm_rate,
            "target_met": best_result["meets_target"],
            "train_sec": best_result["train_sec"],
            "val_report": best_result["val_report"],
            "val_band": best_result["val_band"],
            "val_auc": best_result["val_auc"],
            "test_report": None,
            "test_band": "n/a",
            "test_auc": float("nan"),
        }

    tr = best_result["test_report"]
    test_band, test_note = _performance_band(tr)
    print(f"  test: {_format_metric_line(tr, best_result['test_auc'])} | {test_band}")
    print(f"  test cm: {tr['confusion_matrix'].tolist()}")
    if verbose:
        print(f"  note: {test_note}")

    if verbose:
        analysis_model = _make_search_model(best_result["params"])
        analysis_model.fit(X_train, y_train, eval_set=(X_val, y_val) if has_eval else None)
        analysis_test_prob = analysis_model.predict_proba(X_test)
        if tr["precision"] >= 0.50 and tr["recall"] >= 0.50:
            print("Goal met: precision >= 0.50 and recall >= 0.50")
        else:
            print("Threshold reference")
            _print_threshold_reference(y_test, analysis_test_prob, thresholds)
        _print_threshold_sweep(y_test, analysis_test_prob, start=0.01, stop=0.50, step=0.01)

    return {
        "dataset": dataset_name,
        "path": path,
        "best_preset": best_result["name"],
        "best_params": best_result["params"],
        "threshold": best_result["threshold"],
        "selection_policy": selection_primary_metric,
        "target_recall": selection_target_recall,
        "max_false_alarm_rate": selection_max_false_alarm_rate,
        "target_met": best_result["meets_target"],
        "train_sec": best_result["train_sec"],
        "val_report": best_result["val_report"],
        "val_band": best_result["val_band"],
        "val_auc": best_result["val_auc"],
        "test_report": tr,
        "test_band": best_result["test_band"],
        "test_auc": best_result["test_auc"],
        "all_results": search_results,
    }


def _run_dataset_suite(paths: List[Union[str, Path]], verbose: bool = False) -> None:
    preset_count = len(_build_search_presets())
    print("XGBoost Full Experiment Report")
    print(f"datasets: {len(paths)}")
    print(f"parameter sets: {preset_count}")
    print(f"total runs: {len(paths) * preset_count}")

    results: List[Dict[str, object]] = []

    for path in paths:
        results.append(_load_and_train_from_pickle(path, verbose=verbose))

    print("\n" + "=" * 120)
    print("Final summary")
    print(
        f"{'Dataset':<32} {'Preset':<18} {'Th':>5} {'Time':>7} "
        f"{'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'AUC':>8} {'Grade':>8} "
        f"{'TN':>5} {'FP':>5} {'FN':>5} {'TP':>5}"
    )
    for item in results:
        report = item["test_report"]
        if report is None:
            print(
                f"{str(item['dataset']):<32} {str(item['best_preset']):<18} "
                f"{float(item['threshold']):>5.2f} {float(item['train_sec']):>7.2f} "
                f"{'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8} "
                f"{'N/A':>5} {'N/A':>5} {'N/A':>5} {'N/A':>5}"
            )
            continue

        print(
            f"{str(item['dataset']):<32} {str(item['best_preset']):<18} "
            f"{float(item['threshold']):>5.2f} {float(item['train_sec']):>7.2f} "
            f"{report['accuracy']:>8.4f} {report['precision']:>8.4f} "
            f"{report['recall']:>8.4f} {report['f1']:>8.4f} {float(item['test_auc']):>8.4f} {str(item['test_band']):>8} "
            f"{int(report['tn']):>5} {int(report['fp']):>5} {int(report['fn']):>5} {int(report['tp']):>5}"
        )


def _best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: List[float],
    min_accuracy: float = 0.0,
    min_recall: float = 0.0,
    max_false_alarm_rate: float = 1.0,
    primary_metric: str = "f1",
) -> Tuple[float, float, bool]:
    """
    Sweep thresholds and return the one that best matches the selected policy.

    Current supported policies:
    - primary_metric="f1": maximise F1 with accuracy as a tie-breaker
    - primary_metric="recall": maximise recall, then precision, F1, accuracy
    - primary_metric="recall_guard": first require recall >= min_recall and
      false_alarm_rate <= max_false_alarm_rate, then choose the threshold with
      the best recall / false-alarm / precision tradeoff

    Thresholds that satisfy the active safety constraint are always preferred.
    If none satisfy that constraint, the best unconstrained threshold is returned.

    Returns (best_threshold, best_score_value, safety_constraint_met).
    """
    best_thresh = thresholds[0]
    best_score = None
    found_valid = False
    best_score_value = float("-inf")

    for t in thresholds:
        y_pred = (y_prob >= t).astype(np.int8)
        rep = Metrics.classification_report(y_true, y_pred)

        if primary_metric == "recall_guard":
            score = (
                rep["recall"],
                -rep["false_alarm_rate"],
                rep["precision"],
                rep["f1"],
                rep["accuracy"],
            )
            fallback_score = (
                1 if rep["recall"] >= min_recall else 0,
                1 if rep["false_alarm_rate"] <= max_false_alarm_rate else 0,
                rep["recall"],
                -rep["false_alarm_rate"],
                rep["precision"],
                rep["f1"],
                rep["accuracy"],
            )
            score_value = rep["recall"]
            meets = (
                rep["recall"] >= min_recall
                and rep["false_alarm_rate"] <= max_false_alarm_rate
                and rep["accuracy"] >= min_accuracy
            )
        elif primary_metric == "recall":
            score = (
                rep["recall"],
                rep["precision"],
                rep["f1"],
                rep["accuracy"],
            )
            fallback_score = score
            score_value = rep["recall"]
            meets = rep["accuracy"] >= min_accuracy
        else:
            score = (
                rep["f1"],
                rep["recall"],
                rep["precision"],
                rep["accuracy"],
            )
            fallback_score = score
            score_value = rep["f1"]
            meets = rep["accuracy"] >= min_accuracy

        if meets:
            if not found_valid or score > best_score:
                best_score = score
                best_thresh = t
                best_score_value = score_value
                found_valid = True
        elif not found_valid and (best_score is None or fallback_score > best_score):
            best_score = fallback_score
            best_thresh = t
            best_score_value = score_value

    return best_thresh, best_score_value, found_valid


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the NumPy XGBoost-style classifier on one or more preprocessed pickle datasets."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        action="append",
        dest="data_paths",
        help="Path to a pickle file containing train / validation / test splits. Repeat to compare multiple datasets.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print full dataset details and threshold tables.",
    )
    args = parser.parse_args()
    if args.data_paths:
        if len(args.data_paths) == 1:
            _load_and_train_from_pickle(args.data_paths[0], verbose=args.verbose)
        else:
            _run_dataset_suite(args.data_paths, verbose=args.verbose)
    else:
        _run_dataset_suite(DEFAULT_DATASET_SUITE, verbose=args.verbose)
