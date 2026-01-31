""" HeteroRC class. 
1. class HeteroRC
2. time_resolved_decoding_heterorc
3. time_resolved_decoding_train_test_heterorc
4. cross_temporal_decoding_heterorc
5. cross_generalisation_train_test_heterorc

Author: Runhao Lu, Jan 2026"""

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import roc_auc_score

class HeteroRC:
    """
    Heterogeneous Reservoir Computing (HeteroRC)

    This module acts as a **fixed (untrained) recurrent feature extractor** for
    multichannel time series such as EEG/MEG.

    Key idea:
    - Each reservoir unit i has its own time constant τ_i (heterogeneity), sampled
      from a log-normal distribution. This mimics the broad distribution of neural
      membrane/synaptic time scales observed in biology.
    - The heterogeneous τ_i are converted into unit-specific leak rates:
        leak_i = 1 - exp(-dt / τ_i),  dt = 1/fs
      so that different units integrate inputs over different effective horizons.

    Typical usage:
    - Call `transform(X_seq)` to obtain reservoir state trajectories (features).
    - Feed the resulting states into a trainable readout (e.g., Ridge, LogisticRegression).

    Parameters
    ----------
    n_in : int
        Number of input channels (e.g., EEG/MEG sensors or components).
    n_res : int
        Number of reservoir units.
    fs : float
        Sampling frequency (Hz). Defines dt = 1/fs.
    spectral_radius : float
        Target spectral radius for W_res. Controls echo-state stability / memory.
    input_scaling : float
        Scaling for input weights W_in.
    bias_scaling : float
        Scaling for reservoir bias.
    tau_mode : float
        Mode of the log-normal distribution of τ (seconds).
        Example: 0.01 = 10 ms.
    tau_sigma : float
        Log-normal sigma controlling heterogeneity (tail length).
    tau_min, tau_max : float
        Physiological bounds for τ (seconds). We reject samples outside this range.
    bidirectional : bool
        If True, compute forward and backward reservoir passes and merge them.
    merge_mode : {'product', 'average', 'signed_sqrt'}
        Strategy to merge forward/backward state trajectories.
    random_state : int or None
        Seed for reproducibility.

    Notes
    -----
    - This class does not learn W_in or W_res; only the downstream readout is trained.
    - Bidirectionality is helpful when labels depend on patterns anywhere in the window.
    """

    def __init__(self, n_in, n_res=None, fs=None,
                 spectral_radius=0.95,
                 input_scaling=0.5,
                 bias_scaling=0.5,
                 tau_mode=0.01,
                 tau_sigma=0.8,
                 tau_min=0.002,
                 tau_max=0.08,
                 bidirectional=True,
                 merge_mode='product',
                 random_state=None):

        self.rng = np.random.RandomState(random_state)
        self.n_in = int(n_in)
        self.n_res = int(n_res)
        self.fs = float(fs)
        self.bidirectional = bool(bidirectional)
        self.merge_mode = merge_mode

        if self.merge_mode not in {'product', 'average', 'signed_sqrt'}:
            raise ValueError(f"Unknown merge_mode={merge_mode}. "
                             "Use one of {'product', 'average', 'signed_sqrt'}.")

        # ---------------------------------------------------------------------
        # 1) Sample heterogeneous time constants τ from a (truncated) log-normal
        #
        # For log-normal LN(mu, sigma), mode = exp(mu - sigma^2)
        # => mu = log(mode) + sigma^2
        # ---------------------------------------------------------------------
        mu = np.log(tau_mode) + tau_sigma**2

        taus_list = []
        max_iters = 1000  # avoids infinite loops under extreme parameter settings
        it = 0
        while len(taus_list) < self.n_res and it < max_iters:
            # Oversample each batch to reduce the chance of not filling n_res
            batch = self.rng.lognormal(mean=mu, sigma=tau_sigma, size=self.n_res * 2)
            valid = batch[(batch >= tau_min) & (batch <= tau_max)]
            taus_list.extend(valid.tolist())
            it += 1

        if len(taus_list) < self.n_res:
            # Fallback: clip a final batch rather than hanging forever
            batch = self.rng.lognormal(mean=mu, sigma=tau_sigma, size=self.n_res)
            batch = np.clip(batch, tau_min, tau_max)
            taus_list.extend(batch.tolist())

        self.taus = np.asarray(taus_list[:self.n_res], dtype=float)

        # Unit-specific leak rates (discretized low-pass integration)
        # leak_i in (0,1): small τ -> larger leak -> faster dynamics
        dt = 1.0 / self.fs
        self.leak = 1.0 - np.exp(-dt / self.taus)  # shape: (n_res,)

        # ---------------------------------------------------------------------
        # 2) Initialize weights
        # ---------------------------------------------------------------------
        # Input weights: dense, uniform in [-input_scaling, +input_scaling]
        self.W_in = (self.rng.rand(self.n_res, self.n_in) * 2.0 - 1.0) * input_scaling

        # Reservoir weights: sparse random graph, then scale to desired spectral radius
        W_res = self.rng.rand(self.n_res, self.n_res) - 0.5  # uniform in [-0.5, 0.5]
        mask = (self.rng.rand(self.n_res, self.n_res) <= 0.1)  # keep ~10% connections
        W_res *= mask

        # Spectral radius normalization (can be expensive for large n_res)
        radius = np.max(np.abs(np.linalg.eigvals(W_res)))
        self.W_res = W_res * (spectral_radius / (radius if radius > 0 else 1.0))

        # Bias term per unit
        self.bias = (self.rng.rand(self.n_res) * 2.0 - 1.0) * bias_scaling  # (n_res,)

    def transform(self, X_seq):
        """
        Project input sequences into reservoir state trajectories.

        Parameters
        ----------
        X_seq : ndarray, shape (n_trials, n_in, n_times)
            Batch of EEG/MEG trials.
            - n_trials: number of trials/epochs
            - n_in: number of channels (must match self.n_in)
            - n_times: number of time samples

        Returns
        -------
        states : ndarray, shape (n_trials, n_res, n_times)
            Reservoir state trajectories (features).
            If bidirectional=True, forward and backward states are merged
            according to `merge_mode`.
        """
        X_seq = np.asarray(X_seq)
        n_trials, n_chans, n_times = X_seq.shape
        if n_chans != self.n_in:
            raise ValueError(f"X_seq has n_in={n_chans}, but model expects {self.n_in}.")

        # Pre-transpose for faster GEMM patterns
        Win_T = self.W_in.T   # (n_in, n_res)
        Wres_T = self.W_res.T # (n_res, n_res)

        # -------------------------
        # Forward pass
        # -------------------------
        states_fwd = np.zeros((n_trials, self.n_res, n_times), dtype=float)
        x = np.zeros((n_trials, self.n_res), dtype=float)

        for t in range(n_times):
            # u: input drive, r: recurrent drive
            u = X_seq[:, :, t] @ Win_T          # (n_trials, n_res)
            r = x @ Wres_T                      # (n_trials, n_res)
            # Leaky integrator update with heterogeneous leak per unit
            x = (1.0 - self.leak) * x + self.leak * np.tanh(u + r + self.bias)
            states_fwd[:, :, t] = x

        if not self.bidirectional:
            return states_fwd

        # -------------------------
        # Backward pass (time-reversed input)
        # -------------------------
        X_rev = np.flip(X_seq, axis=2)
        states_bwd_rev = np.zeros((n_trials, self.n_res, n_times), dtype=float)
        x = np.zeros((n_trials, self.n_res), dtype=float)

        for t in range(n_times):
            u = X_rev[:, :, t] @ Win_T
            r = x @ Wres_T
            x = (1.0 - self.leak) * x + self.leak * np.tanh(u + r + self.bias)
            states_bwd_rev[:, :, t] = x

        # Flip back to align with original time index
        states_bwd = np.flip(states_bwd_rev, axis=2)

        # -------------------------
        # Merge forward/backward representations
        # -------------------------
        if self.merge_mode == 'product':
            # Elementwise interaction term; emphasizes features consistent in both directions
            return states_fwd * states_bwd

        if self.merge_mode == 'average':
            # Smooth fusion; preserves scale better, often more stable
            return 0.5 * (states_fwd + states_bwd)
        
        if self.merge_mode == 'signed_sqrt':
            # signed_sqrt: common trick to compress heavy tails while preserving sign
            prod = states_fwd * states_bwd
            return np.sign(prod) * np.sqrt(np.abs(prod) + 1e-12)
        
def time_resolved_decoding_heterorc(
    X, y, times,
    n_folds=5,
    fs=None,
    rc_params=None,
    scale_percentile=99,
    alphas=(0.001, 0.01, 0.1, 1.0, 10.0, 100.0,1000.0),
    shuffle=True,
    cv_random_state=42,
    rc_seed_mode="fixed",  # {"fixed", "per_fold"}
    base_rc_random_state=100,
    return_folds=False,
    metric="accuracy",          # {"accuracy", "auc"}
    smooth_decisions=False,
    smooth_sigma_points=2.0,
    smooth_mode="nearest",
    verbose=False,
):
    """
    Run time-resolved decoding with HeteroRC features + RidgeClassifierCV.

    Parameters
    ----------
    X : ndarray, shape (n_trials, n_in, n_times)
        Epoched EEG/MEG data.
    y : ndarray, shape (n_trials,)
        Labels for each trial.
    times : array-like, shape (n_times,)
        Time vector (only used for sanity checks / returned alongside scores).
    n_folds : int
        Number of StratifiedKFold splits.
    fs : float or None
        Sampling frequency passed to HeteroRC. If None, rc_params must include 'fs'.
    rc_params : dict or None
        Parameters forwarded to HeteroRC (except n_in which is inferred).
        Example:
            rc_params = dict(n_res=350, input_scaling=0.5, bias_scaling=0.5,
                             spectral_radius=0.95, tau_mode=0.01, tau_sigma=0.8,
                             bidirectional=True, merge_mode='product')
    scale_percentile : float
        Percentile for robust amplitude scaling on training data (per fold).
    alphas : iterable
        RidgeClassifierCV alpha grid.
    shuffle : bool
        Whether to shuffle before split.
    cv_random_state : int
        Random seed for the CV splitter.
    rc_seed_mode : {"fixed", "per_fold"}
        Controls how the random seed of the HeteroRC reservoir is assigned across CV folds.
        - "fixed":
            Use the same reservoir realization for all folds: random_state = base_rc_random_state
            This makes the reservoir feature mapping identical across folds, reducing
            variability due to reservoir initialization.
        - "per_fold":
            Use a different reservoir realization for each fold: random_state = base_rc_random_state + fold_id
            This avoids coupling the entire evaluation to one specific random reservoir
            draw, and is typically preferred for benchmarking robustness.
    base_rc_random_state : int
        Base seed for HeteroRC; each fold uses base_rc_random_state + fold_id.
    return_folds : bool
        If True, also return per-fold accuracy arrays of shape (n_folds, n_times).
    metric:  {"accuracy", "auc"}
        auc uses "ovr", safe for multiclasses
    If smooth_decisions=True:
        Collect decision_function scores for each time point,
        Apply gaussian_filter1d along time axis,
        Compute accuracy from smoothed decisions.
    verbose : bool
        Print progress.
    
    Returns
    -------
    avg_scores : ndarray, shape (n_times,)
        Mean accuracy across folds at each time point.
    folds_scores : ndarray, shape (n_folds, n_times), optional
        Returned only if return_folds=True.
    """
    X = np.asarray(X)
    y = np.asarray(y)

    if X.ndim != 3:
        raise ValueError(f"X must be 3D (n_trials, n_in, n_times). Got shape {X.shape}.")
    n_trials, n_in, n_times = X.shape

    times = np.asarray(times)
    if times.shape[0] != n_times:
        raise ValueError(f"len(times)={len(times)} must equal X.shape[2]={n_times}.")

    if rc_params is None:
        rc_params = {}

    # allow passing fs either directly or via rc_params
    if fs is not None:
        rc_params = dict(rc_params)  # copy
        rc_params["fs"] = fs
    if "fs" not in rc_params:
        raise ValueError("fs must be provided either as argument or inside rc_params['fs'].")
    if metric not in {"accuracy", "auc"}:
        raise ValueError("metric must be 'accuracy' or 'auc'.")
    if metric == "auc":
        from sklearn.metrics import roc_auc_score
        from sklearn.preprocessing import label_binarize
    if smooth_decisions:
        try:
            from scipy.ndimage import gaussian_filter1d
        except Exception as e:
            raise ImportError(
                "smooth_decisions=True requires scipy. Install with: pip install scipy"
            ) from e
    
        if smooth_sigma_points is None or smooth_sigma_points <= 0:
            raise ValueError("smooth_sigma_points must be > 0 when smooth_decisions=True.")
    if rc_seed_mode not in {"fixed", "per_fold"}:
        raise ValueError("rc_seed_mode must be one of {'fixed', 'per_fold'}.")
    skf = StratifiedKFold(n_splits=n_folds, shuffle=shuffle, random_state=cv_random_state)
    fold_scores = np.zeros((n_folds, n_times), dtype=float)

    for fold_id, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Robust scaling factor computed on training set only (avoid leakage)
        scale_val = np.percentile(np.abs(X_train), scale_percentile)
        if not np.isfinite(scale_val) or scale_val == 0:
            scale_val = 1.0
        X_train_s = X_train / scale_val
        X_test_s = X_test / scale_val
        if rc_seed_mode == "fixed":
            esn_seed = base_rc_random_state
        else:  # "per_fold"
            esn_seed = base_rc_random_state + fold_id
        # RC transform (fold-specific seed for reproducibility)
        esn = HeteroRC(
            n_in=n_in,
            random_state=esn_seed,
            **rc_params
        )
        S_train = esn.transform(X_train_s)  # (n_train, n_res, n_times)
        S_test = esn.transform(X_test_s)    # (n_test, n_res, n_times)
        if not smooth_decisions:
            # direct per-timepoint metric (accuracy or auc)
            classes = np.unique(y_train)  # for AUC multiclass safety
            n_classes = len(classes)
        
            for t in range(n_times):
                clf = make_pipeline(StandardScaler(), RidgeClassifierCV(alphas=alphas))
                clf.fit(S_train[:, :, t], y_train)
        
                if metric == "accuracy":
                    fold_scores[fold_id, t] = clf.score(S_test[:, :, t], y_test)
        
                else:  # metric == "auc"
                    raw = clf.decision_function(S_test[:, :, t])
                    est = clf[-1]
        
                    # align raw scores to 'classes' order (same as your smoothing branch)
                    if raw.ndim == 1:
                        scores_est = np.vstack([-raw, raw]).T
                        order = np.searchsorted(est.classes_, classes)
                        scores = scores_est[:, order]
                    else:
                        order = np.searchsorted(est.classes_, classes)
                        scores = raw[:, order]
        
                    if n_classes > 2:
                        y_test_bin = label_binarize(y_test, classes=classes)
                        fold_scores[fold_id, t] = roc_auc_score(
                            y_test_bin, scores,
                            average="macro",
                            multi_class="ovr"
                        )
                    else:
                        # binary: take score for positive class (classes[1] in our aligned order)
                        # scores is (n_test, 2)
                        fold_scores[fold_id, t] = roc_auc_score(y_test, scores[:, 1])

        else:
            # collect decisions -> smooth -> argmax -> accuracy
            classes = np.unique(y_train)
            n_classes = len(classes)
            y_test_idx = np.searchsorted(classes, y_test)

            decisions = np.zeros((len(y_test), n_classes, n_times), dtype=float)

            for t in range(n_times):
                clf = make_pipeline(StandardScaler(), RidgeClassifierCV(alphas=alphas))
                clf.fit(S_train[:, :, t], y_train)

                raw = clf.decision_function(S_test[:, :, t])
                est = clf[-1]

                if raw.ndim == 1:
                    # binary: raw corresponds to est.classes_[1]
                    scores_est = np.vstack([-raw, raw]).T  # columns in est.classes_ order
                    order = np.searchsorted(est.classes_, classes)
                    scores = scores_est[:, order]
                else:
                    order = np.searchsorted(est.classes_, classes)
                    scores = raw[:, order]

                decisions[:, :, t] = scores

            decisions_smoothed = gaussian_filter1d(
                decisions, sigma=smooth_sigma_points, axis=-1, mode=smooth_mode
            )

            if metric == "accuracy":
                preds_idx = np.argmax(decisions_smoothed, axis=1)  # (n_test, n_times)
                fold_scores[fold_id, :] = np.mean(preds_idx == y_test_idx[:, None], axis=0)

            else:  # metric == "auc"
                # decisions_smoothed is (n_test, n_classes, n_times)
                if n_classes > 2:
                    y_test_bin = label_binarize(y_test, classes=classes)  # (n_test, n_classes)
                    for t in range(n_times):
                        fold_scores[fold_id, t] = roc_auc_score(
                            y_test_bin, decisions_smoothed[:, :, t],
                            average="macro",
                            multi_class="ovr"
                        )
                else:
                    # binary: use smoothed score for positive class
                    for t in range(n_times):
                        fold_scores[fold_id, t] = roc_auc_score(y_test, decisions_smoothed[:, 1, t])

        if verbose:
            print(f"Fold {fold_id+1}/{n_folds} done. scale={scale_val:.3g}")

    avg_scores = fold_scores.mean(axis=0)
    if return_folds:
        return avg_scores, fold_scores
    return avg_scores

def time_resolved_decoding_train_test_heterorc(
    X_train,
    y_train,
    X_test,
    y_test,
    times,
    fs=None,
    rc_params=None,
    scale_percentile=99,
    alphas=(0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0),
    rc_random_state=42,
    metric="accuracy",                 # {"accuracy", "auc"}
    smooth_decisions=False,
    smooth_sigma_points=2.0,
    smooth_mode="nearest",
    return_states=False,
    return_decisions=False,
    verbose=False,
):
    """
    Time-resolved decoding with an external test set (train on X_train, test on X_test),
    evaluated only on the diagonal (train_time == test_time).

    Matches style of cross_generalisation_train_test_heterorc:
      - Robust scaling on X_train only
      - One fixed HeteroRC instance shared across train/test (rc_random_state)
      - Linear readout: Pipeline(StandardScaler -> RidgeClassifierCV(alphas))
      - Optional temporal smoothing of decision scores along TIME axis (test-time axis)

    Returns
    -------
    scores : ndarray, shape (n_times,)
        Accuracy (or AUC) at each time point (diagonal).
    (optional) S_train, S_test : ndarray
        Reservoir states if return_states=True.
    (optional) decisions : ndarray
        Decision scores if return_decisions=True:
          - multiclass: (n_test, n_classes, n_times)
          - binary:     (n_test, 2, n_times)  (aligned to classes order)
    """
    import numpy as np
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import RidgeClassifierCV

    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    times = np.asarray(times)

    if X_train.ndim != 3 or X_test.ndim != 3:
        raise ValueError(
            f"X_train and X_test must be 3D (n_trials, n_in, n_times). "
            f"Got {X_train.shape} and {X_test.shape}."
        )
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("X_train and X_test must have the same number of channels (n_in).")
    if X_train.shape[2] != X_test.shape[2]:
        raise ValueError("X_train and X_test must have the same number of time points (n_times).")

    n_train, n_in, n_times = X_train.shape
    n_test = X_test.shape[0]

    if times.shape[0] != n_times:
        raise ValueError(f"len(times)={len(times)} must equal X_train.shape[2]={n_times}.")

    if rc_params is None:
        rc_params = {}
    rc_params_local = dict(rc_params)

    if fs is not None:
        rc_params_local["fs"] = fs
    if "fs" not in rc_params_local:
        raise ValueError("fs must be provided either as argument or inside rc_params['fs'].")
    if "n_res" not in rc_params_local:
        raise ValueError("rc_params must include 'n_res' (number of reservoir units).")

    if metric not in {"accuracy", "auc"}:
        raise ValueError("metric must be 'accuracy' or 'auc'.")

    if smooth_decisions:
        try:
            from scipy.ndimage import gaussian_filter1d
        except Exception as e:
            raise ImportError("smooth_decisions=True requires scipy. Install with: pip install scipy") from e
        if smooth_sigma_points is None or smooth_sigma_points <= 0:
            raise ValueError("smooth_sigma_points must be > 0 when smooth_decisions=True.")

    if metric == "auc":
        from sklearn.metrics import roc_auc_score
        from sklearn.preprocessing import label_binarize

    # --- Robust scaling on train only ---
    scale_val = np.percentile(np.abs(X_train), scale_percentile)
    if not np.isfinite(scale_val) or scale_val == 0:
        scale_val = 1.0
    X_train_s = X_train / scale_val
    X_test_s = X_test / scale_val

    if verbose:
        print(f"[Train->Test TR] scale={scale_val:.3g}, n_train={n_train}, n_test={n_test}, n_times={n_times}")

    # --- One fixed reservoir shared across train/test ---
    esn = HeteroRC(
        n_in=n_in,
        random_state=rc_random_state,
        **rc_params_local
    )
    S_train = esn.transform(X_train_s)  # (n_train, n_res, n_times)
    S_test = esn.transform(X_test_s)    # (n_test,  n_res, n_times)

    scores = np.zeros((n_times,), dtype=float)

    # consistent class ordering (useful for smoothing and AUC)
    classes = np.unique(y_train)
    n_classes = len(classes)

    # ---- no smoothing: compute per-time directly ----
    if not smooth_decisions:
        for t in range(n_times):
            clf = make_pipeline(StandardScaler(), RidgeClassifierCV(alphas=alphas))
            clf.fit(S_train[:, :, t], y_train)

            if metric == "accuracy":
                scores[t] = clf.score(S_test[:, :, t], y_test)
            else:
                raw = clf.decision_function(S_test[:, :, t])
                est = clf[-1]

                # align raw scores to `classes` order
                if raw.ndim == 1:
                    scores_est = np.vstack([-raw, raw]).T  # (n_test, 2) in est.classes_ order
                    order = np.searchsorted(est.classes_, classes)
                    aligned = scores_est[:, order]
                else:
                    order = np.searchsorted(est.classes_, classes)
                    aligned = raw[:, order]

                if n_classes > 2:
                    y_test_bin = label_binarize(y_test, classes=classes)
                    scores[t] = roc_auc_score(y_test_bin, aligned, average="macro", multi_class="ovr")
                else:
                    scores[t] = roc_auc_score(y_test, aligned[:, 1])

        if return_states and return_decisions:
            # decisions not computed in this branch
            return scores, S_train, S_test, None
        if return_states:
            return scores, S_train, S_test
        return scores

    # ---- smoothing branch: collect decisions across time, smooth, then compute metric ----
    decisions = np.zeros((n_test, n_classes, n_times), dtype=float)
    y_test_idx = np.searchsorted(classes, y_test)

    for t in range(n_times):
        clf = make_pipeline(StandardScaler(), RidgeClassifierCV(alphas=alphas))
        clf.fit(S_train[:, :, t], y_train)

        raw = clf.decision_function(S_test[:, :, t])
        est = clf[-1]

        if raw.ndim == 1:
            # binary: raw corresponds to est.classes_[1]
            scores_est = np.vstack([-raw, raw]).T
            order = np.searchsorted(est.classes_, classes)
            aligned = scores_est[:, order]
        else:
            order = np.searchsorted(est.classes_, classes)
            aligned = raw[:, order]

        decisions[:, :, t] = aligned

    decisions_sm = gaussian_filter1d(decisions, sigma=smooth_sigma_points, axis=-1, mode=smooth_mode)

    if metric == "accuracy":
        preds_idx = np.argmax(decisions_sm, axis=1)  # (n_test, n_times)
        scores = np.mean(preds_idx == y_test_idx[:, None], axis=0)
    else:
        if n_classes > 2:
            y_test_bin = label_binarize(y_test, classes=classes)
            for t in range(n_times):
                scores[t] = roc_auc_score(y_test_bin, decisions_sm[:, :, t], average="macro", multi_class="ovr")
        else:
            for t in range(n_times):
                scores[t] = roc_auc_score(y_test, decisions_sm[:, 1, t])

    if return_states and return_decisions:
        return scores, S_train, S_test, decisions_sm
    if return_states:
        return scores, S_train, S_test
    if return_decisions:
        return scores, decisions_sm
    return scores


def cross_temporal_decoding_heterorc(
    X, y, times,
    n_folds=5,
    fs=None,
    rc_params=None,
    scale_percentile=99,
    alphas=(0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0),
    shuffle=True,
    cv_random_state=42,
    rc_seed_mode="fixed",  # {"fixed", "per_fold"}
    base_rc_random_state=100,
    return_folds=False,
    smooth_decisions=False,
    smooth_sigma_points=2.0,
    smooth_mode="nearest",
    verbose=False,
):
    """
    Cross-temporal generalisation decoding with HeteroRC features + RidgeClassifierCV.

    Train a readout at each train time t_train, test it at each test time t_test,
    producing a (n_times, n_times) generalisation matrix per fold.

    Parameters
    ----------
    Same as time_resolved_decoding_heterorc.

    Returns
    -------
    avg_scores : ndarray, shape (n_times, n_times)
        Mean accuracy across folds for each (train_time, test_time).
        Rows: train time, Columns: test time.
    folds_scores : ndarray, shape (n_folds, n_times, n_times), optional
        Returned only if return_folds=True.
    """
    X = np.asarray(X)
    y = np.asarray(y)

    if X.ndim != 3:
        raise ValueError(f"X must be 3D (n_trials, n_in, n_times). Got shape {X.shape}.")
    n_trials, n_in, n_times = X.shape

    times = np.asarray(times)
    if times.shape[0] != n_times:
        raise ValueError(f"len(times)={len(times)} must equal X.shape[2]={n_times}.")

    if rc_params is None:
        rc_params = {}

    # allow passing fs either directly or via rc_params
    if fs is not None:
        rc_params = dict(rc_params)  # copy
        rc_params["fs"] = fs
    if "fs" not in rc_params:
        raise ValueError("fs must be provided either as argument or inside rc_params['fs'].")

    if rc_seed_mode not in {"fixed", "per_fold"}:
        raise ValueError("rc_seed_mode must be one of {'fixed', 'per_fold'}.")

    if smooth_decisions:
        try:
            from scipy.ndimage import gaussian_filter1d
        except Exception as e:
            raise ImportError(
                "smooth_decisions=True requires scipy. Install with: pip install scipy"
            ) from e
        if smooth_sigma_points is None or smooth_sigma_points <= 0:
            raise ValueError("smooth_sigma_points must be > 0 when smooth_decisions=True.")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=shuffle, random_state=cv_random_state)
    fold_scores = np.zeros((n_folds, n_times, n_times), dtype=float)

    for fold_id, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Robust scaling computed on training set only (avoid leakage)
        scale_val = np.percentile(np.abs(X_train), scale_percentile)
        if not np.isfinite(scale_val) or scale_val == 0:
            scale_val = 1.0
        X_train_s = X_train / scale_val
        X_test_s = X_test / scale_val

        # RC seed handling
        if rc_seed_mode == "fixed":
            esn_seed = base_rc_random_state
        else:
            esn_seed = base_rc_random_state + fold_id

        # RC transform
        esn = HeteroRC(
            n_in=n_in,
            random_state=esn_seed,
            **rc_params
        )
        S_train = esn.transform(X_train_s)  # (n_train, n_res, n_times)
        S_test = esn.transform(X_test_s)    # (n_test, n_res, n_times)

        if not smooth_decisions:
            # For each train time, fit once, test on all test times
            for t_tr in range(n_times):
                clf = make_pipeline(StandardScaler(), RidgeClassifierCV(alphas=alphas))
                clf.fit(S_train[:, :, t_tr], y_train)

                for t_te in range(n_times):
                    fold_scores[fold_id, t_tr, t_te] = clf.score(S_test[:, :, t_te], y_test)

        else:
            # Smooth across *test-time axis* for each trained classifier:
            # decisions: (n_test, n_classes, n_times_test) -> gaussian_filter1d along last axis
            classes = np.unique(y_train)
            n_classes = len(classes)
            y_test_idx = np.searchsorted(classes, y_test)

            for t_tr in range(n_times):
                clf = make_pipeline(StandardScaler(), RidgeClassifierCV(alphas=alphas))
                clf.fit(S_train[:, :, t_tr], y_train)

                decisions = np.zeros((len(y_test), n_classes, n_times), dtype=float)

                # collect decision scores across all test times
                for t_te in range(n_times):
                    raw = clf.decision_function(S_test[:, :, t_te])
                    est = clf[-1]

                    if raw.ndim == 1:
                        # binary: raw corresponds to est.classes_[1]
                        scores_est = np.vstack([-raw, raw]).T  # columns in est.classes_ order
                        order = np.searchsorted(est.classes_, classes)
                        scores = scores_est[:, order]
                    else:
                        order = np.searchsorted(est.classes_, classes)
                        scores = raw[:, order]

                    decisions[:, :, t_te] = scores

                # smooth along test-time axis
                decisions_smoothed = gaussian_filter1d(
                    decisions, sigma=smooth_sigma_points, axis=-1, mode=smooth_mode
                )
                preds_idx = np.argmax(decisions_smoothed, axis=1)  # (n_test, n_times_test)
                fold_scores[fold_id, t_tr, :] = np.mean(preds_idx == y_test_idx[:, None], axis=0)

        if verbose:
            print(f"Fold {fold_id+1}/{n_folds} done. scale={scale_val:.3g}")

    avg_scores = fold_scores.mean(axis=0)
    if return_folds:
        return avg_scores, fold_scores
    return avg_scores

def cross_generalisation_train_test_heterorc(
    X_train,
    y_train,
    X_test,
    y_test,
    times,
    fs=None,
    rc_params=None,
    scale_percentile=99,
    alphas=(0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0),
    rc_random_state=42,
    smooth_decisions=False,
    smooth_sigma_points=2.0,
    smooth_mode="nearest",
    return_states=False,
    verbose=False,
):
    """
    Cross-temporal generalisation with an external test set (train on X_train, test on X_test).

    This matches the settings/style of `time_resolved_decoding_heterorc`:
      - Robust scaling uses X_train only (avoid leakage).
      - A single HeteroRC instance (fixed by rc_random_state) is used for BOTH train and test.
      - Reservoir states are decoded with a linear readout:
            Pipeline(StandardScaler -> RidgeClassifierCV(alphas))
      - Optional post-readout temporal smoothing is applied to decision scores along the TEST-time axis.

    Parameters
    ----------
    X_train : ndarray, shape (n_train_trials, n_in, n_times)
    y_train : ndarray, shape (n_train_trials,)
    X_test  : ndarray, shape (n_test_trials,  n_in, n_times)
    y_test  : ndarray, shape (n_test_trials,)
    times   : ndarray, shape (n_times,)
        Time vector (only used for sanity checking / interface consistency).
    fs : float, optional
        Sampling frequency. If provided, overrides rc_params['fs'].
    rc_params : dict, optional
        Parameters forwarded to HeteroRC(...). Must include 'n_res' and 'fs' unless fs is provided here.
    scale_percentile : float
        Percentile for robust scaling (computed on |X_train| only).
    alphas : sequence
        Alpha grid for RidgeClassifierCV.
    rc_random_state : int
        Random seed for reservoir initialization (fixed across train/test).
    smooth_decisions : bool
        If True, smooth decision scores along test-time axis before converting to predictions.
    smooth_sigma_points : float
        Gaussian sigma in samples (points) used for smoothing (only if smooth_decisions=True).
    smooth_mode : str
        Boundary mode passed to scipy.ndimage.gaussian_filter1d.
    return_states : bool
        If True, also return (S_train, S_test) reservoir state time-series.
    verbose : bool

    Returns
    -------
    tgm : ndarray, shape (n_times, n_times)
        Accuracy matrix. Rows=train time, Columns=test time.
    (optional) S_train, S_test : ndarray
        Reservoir states if return_states=True.
    """
    import numpy as np
    from sklearn.linear_model import RidgeClassifierCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    times = np.asarray(times)

    if X_train.ndim != 3 or X_test.ndim != 3:
        raise ValueError(
            f"X_train and X_test must be 3D (n_trials, n_in, n_times). "
            f"Got {X_train.shape} and {X_test.shape}."
        )
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("X_train and X_test must have the same number of input channels (n_in).")
    if X_train.shape[2] != X_test.shape[2]:
        raise ValueError("X_train and X_test must have the same number of time points (n_times).")
    n_train, n_in, n_times = X_train.shape
    n_test = X_test.shape[0]

    if times.shape[0] != n_times:
        raise ValueError(f"len(times)={len(times)} must equal X_train.shape[2]={n_times}.")

    if rc_params is None:
        rc_params = {}

    # Allow passing fs either directly or via rc_params (same convention as your CV functions)
    rc_params_local = dict(rc_params)
    if fs is not None:
        rc_params_local["fs"] = fs
    if "fs" not in rc_params_local:
        raise ValueError("fs must be provided either as argument or inside rc_params['fs'].")
    if "n_res" not in rc_params_local:
        raise ValueError("rc_params must include 'n_res' (number of reservoir units).")

    if smooth_decisions:
        try:
            from scipy.ndimage import gaussian_filter1d
        except Exception as e:
            raise ImportError(
                "smooth_decisions=True requires scipy. Install with: pip install scipy"
            ) from e
        if smooth_sigma_points is None or smooth_sigma_points <= 0:
            raise ValueError("smooth_sigma_points must be > 0 when smooth_decisions=True.")

    # --- Robust scaling computed on training set only (avoid leakage) ---
    scale_val = np.percentile(np.abs(X_train), scale_percentile)
    if not np.isfinite(scale_val) or scale_val == 0:
        scale_val = 1.0
    X_train_s = X_train / scale_val
    X_test_s = X_test / scale_val

    if verbose:
        print(f"[Train->Test CTG] scale={scale_val:.3g}, n_train={n_train}, n_test={n_test}, n_times={n_times}")

    # --- One fixed reservoir shared across train/test ---
    # NOTE: HeteroRC must be defined in the same heterorc.py (no self-import here)
    esn = HeteroRC(
        n_in=n_in,
        random_state=rc_random_state,
        **rc_params_local,
    )

    S_train = esn.transform(X_train_s)  # (n_train, n_res, n_times)
    S_test = esn.transform(X_test_s)    # (n_test,  n_res, n_times)

    # Output: rows=train-time, cols=test-time
    tgm = np.zeros((n_times, n_times), dtype=float)

    # Prepare consistent class ordering for smoothing-based predictions
    classes = np.unique(y_train)
    n_classes = len(classes)
    y_test_idx = np.searchsorted(classes, y_test)

    for t_tr in range(n_times):
        clf = make_pipeline(StandardScaler(), RidgeClassifierCV(alphas=alphas))
        clf.fit(S_train[:, :, t_tr], y_train)

        if not smooth_decisions:
            # Direct scoring without temporal smoothing
            for t_te in range(n_times):
                tgm[t_tr, t_te] = clf.score(S_test[:, :, t_te], y_test)

        else:
            # Collect decision scores across all test times, smooth along test-time axis, then compute accuracy
            decisions = np.zeros((n_test, n_classes, n_times), dtype=float)
            est = clf[-1]

            for t_te in range(n_times):
                raw = clf.decision_function(S_test[:, :, t_te])

                if raw.ndim == 1:
                    # Binary: raw is (n_test,), corresponds to est.classes_[1]
                    scores_est = np.vstack([-raw, raw]).T  # columns in est.classes_ order
                    order = np.searchsorted(est.classes_, classes)
                    scores = scores_est[:, order]          # (n_test, 2) aligned to `classes`
                else:
                    # Multiclass: raw is (n_test, n_classes_est)
                    order = np.searchsorted(est.classes_, classes)
                    scores = raw[:, order]                 # aligned to `classes`

                decisions[:, :, t_te] = scores

            decisions_sm = gaussian_filter1d(
                decisions, sigma=smooth_sigma_points, axis=-1, mode=smooth_mode
            )
            preds_idx = np.argmax(decisions_sm, axis=1)    # (n_test, n_times)
            tgm[t_tr, :] = np.mean(preds_idx == y_test_idx[:, None], axis=0)

        if verbose and (t_tr % 50 == 0 or t_tr == n_times - 1):
            print(f"  train_time {t_tr+1}/{n_times} done")

    if return_states:
        return tgm, S_train, S_test
    return tgm



