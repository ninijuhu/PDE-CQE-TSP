def mannwhitney_less(proposed, other):
    proposed = np.asarray(proposed, dtype=float)
    other = np.asarray(other, dtype=float)
    if len(proposed) < 1 or len(other) < 1:
        return float("nan")
    if np.allclose(proposed, other):
        return 1.0
    return float(mannwhitneyu(proposed, other, alternative="less").pvalue)


def wilcoxon_less(proposed, other):
    proposed = np.asarray(proposed, dtype=float)
    other = np.asarray(other, dtype=float)
    n = min(len(proposed), len(other))
    if n < 2:
        return float("nan")
    proposed, other = proposed[:n], other[:n]
    if np.allclose(proposed - other, 0):
        return 1.0
    return float(
        wilcoxon(proposed, other, alternative="less", zero_method="wilcox").pvalue
    )


def rank_biserial(proposed, other):
    n = min(len(proposed), len(other))
    diff = np.asarray(proposed[:n], dtype=float) - np.asarray(other[:n], dtype=float)
    if n == 0:
        return float("nan")
    n_neg = int(np.sum(diff < 0))
    n_pos = int(np.sum(diff > 0))
    return float((n_neg - n_pos) / n)


def holm_adjust(pvals):
    out = [float("nan")] * len(pvals)
    valid = [(i, p) for i, p in enumerate(pvals) if p == p]
    if not valid:
        return out
    valid.sort(key=lambda x: x[1])
    k = len(valid)
    max_so_far = 0.0
    adjusted = {}
    for rank, (i, p) in enumerate(valid):
        adj = min(1.0, (k - rank) * p)
        max_so_far = max(max_so_far, adj)
        adjusted[i] = max_so_far
    for i, a in adjusted.items():
        out[i] = a
    return out


def instance_tests(proposed_runs, baseline_runs):
    raw = [mannwhitney_less(proposed_runs, runs) for runs in baseline_runs]
    return raw, holm_adjust(raw)


def suite_tests(mean_matrix):
    mean_matrix = np.asarray(mean_matrix, dtype=float)
    n_alg = mean_matrix.shape[1]
    stat = friedmanchisquare(*[mean_matrix[:, j] for j in range(n_alg)])
    proposed = mean_matrix[:, 0]
    raw, effects = [], []
    for j in range(1, n_alg):
        other = mean_matrix[:, j]
        raw.append(wilcoxon_less(proposed, other))
        effects.append(rank_biserial(proposed, other))
    return float(stat.statistic), float(stat.pvalue), raw, holm_adjust(raw), effects
