# Controlled Quantum Enhancement (CQE): amplitude amplification + diagnostics.
import os

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import GroverOperator

from tsp_common import (
    backend,
    CQE_MIN_COUNTED_HITS,
    CQE_MAX_OBSERVED_BITSTRINGS,
)


def _sort_cqe_diagnostics(diagnostics):
    """Stable order: cluster_id then generation (for C#-G# axis labels)."""
    fixed = []
    for d in diagnostics:
        rec = dict(d)
        g = int(rec.get("generation", 0))
        # Legacy CSV stored gen+1 (9,17,25,33); map back to trigger gens 8,16,24,32
        if g % 8 == 1 and g > 1:
            rec["generation"] = g - 1
        fixed.append(rec)
    return sorted(
        fixed,
        key=lambda d: (int(d.get("cluster_id", 0)), int(d.get("generation", 0))),
    )


def plot_cqe_marked_set_probability(
    diagnostics,
    instance_name=None,
    max_events=None,
    out_dir="output/qTSP_solutions",
):
    """
    Fig. 3 style:
      (a) all CQE events — single bars of ΔP_marked (positive up / negative down)
      (b) all events — scatter P_before vs P_after with y=x
    max_events is kept for CLI compatibility but ignored (all events shown).
    """
    os.makedirs(out_dir, exist_ok=True)
    if not diagnostics:
        print("[CQE mech] No diagnostic records; skip plotting.")
        return None
    _ = max_events  # all events; no selection bias

    events = _sort_cqe_diagnostics(diagnostics)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    n = len(events)
    x = np.arange(n)
    p_before = np.array([float(d["p_target_before"]) for d in events])
    p_after = np.array([float(d["p_target_after"]) for d in events])
    delta = p_after - p_before
    labels = [
        f"C{int(d.get('cluster_id', 0)) + 1}-G{int(d.get('generation', 0))}"
        for d in events
    ]

    n_up = int(np.sum(delta > 1e-12))
    mean_delta = float(np.mean(delta))
    c_pos = "#4C78A8"
    c_neg = "#F58518"
    c_scatter = "#4C78A8"
    bar_colors = [c_pos if d >= 0 else c_neg for d in delta]

    fig = plt.figure(figsize=(12.6, 5.2))
    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.45, 1.0],
        wspace=0.22,
        left=0.07,
        right=0.98,
        bottom=0.18,
        top=0.88,
    )
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # (a) ΔP_marked bars
    ax1.axhline(0.0, color="black", linewidth=1.1, zorder=2)
    ax1.bar(
        x,
        delta,
        width=0.72,
        color=bar_colors,
        edgecolor="black",
        linewidth=0.45,
        zorder=3,
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel(
        r"Change in marked-set probability $\Delta P_{\mathrm{marked}}$",
        fontsize=11,
    )
    ax1.set_xlabel("CQE event (cluster–generation)", fontsize=11)
    ax1.set_title(
        "(a) Event-wise change in marked-set probability",
        fontsize=12,
        loc="left",
        pad=10,
    )
    ax1.yaxis.grid(True, linestyle="--", alpha=0.35, zorder=0)
    ax1.set_axisbelow(True)
    y_pad = max(0.05, 0.08 * float(np.max(np.abs(delta))))
    ax1.set_ylim(float(delta.min()) - y_pad, float(delta.max()) + y_pad)

    ax1.text(
        0.03,
        0.97,
        (
            f"{n_up} / {n} events increased $P_{{\\mathrm{{marked}}}}$\n"
            f"Mean $\\Delta P_{{\\mathrm{{marked}}}}$ = {mean_delta:+.3f}"
        ),
        transform=ax1.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="#E8F1FA",
            edgecolor="#A8C4DE",
            linewidth=0.8,
        ),
    )

    # (b) before vs after + y=x
    lim_hi = max(float(p_before.max()), float(p_after.max()), 1e-6) * 1.08
    lim_hi = min(max(lim_hi, 0.2), 1.02)
    ax2.plot(
        [0, lim_hi],
        [0, lim_hi],
        "--",
        color="#888888",
        linewidth=1.3,
        zorder=2,
    )
    ax2.text(
        0.97 * lim_hi,
        0.97 * lim_hi,
        r"$y=x$",
        color="#666666",
        fontsize=9,
        ha="right",
        va="top",
    )
    ax2.scatter(
        p_before,
        p_after,
        s=55,
        marker="o",
        facecolors=c_scatter,
        edgecolors="white",
        linewidths=0.8,
        alpha=0.9,
        zorder=3,
    )
    ax2.set_xlim(0, lim_hi)
    ax2.set_ylim(0, lim_hi)
    ax2.set_box_aspect(1)
    ax2.set_anchor("W")
    ax2.set_xlabel(r"$P_{\mathrm{marked}}$ before CQE", fontsize=11)
    ax2.set_ylabel(r"$P_{\mathrm{marked}}$ after CQE", fontsize=11)
    ax2.set_title(
        "(b) Marked-set probability before vs after CQE",
        fontsize=12,
        loc="left",
        pad=10,
    )
    ax2.grid(True, linestyle=":", alpha=0.35)

    ax2.text(
        0.03,
        0.97,
        (
            f"{n_up} / {n} events above $y=x$\n"
            f"Mean $\\Delta P_{{\\mathrm{{marked}}}}$ = {mean_delta:+.3f}"
        ),
        transform=ax2.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="#E8F1FA",
            edgecolor="#A8C4DE",
            linewidth=0.8,
        ),
    )

    suffix = f"_{instance_name}" if instance_name else ""
    out_png = os.path.join(out_dir, f"fig_cqe_marked_probability{suffix}.png")
    out_pdf = os.path.join(out_dir, f"fig_cqe_marked_probability{suffix}.pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    out_csv = os.path.join(out_dir, f"cqe_marked_probability{suffix}.csv")
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write(
            "cluster_id,generation,n_targets,p_marked_before,"
            "p_marked_after,delta,target_amplify,diagnostic_shots\n"
        )
        for d in events:
            pb = float(d["p_target_before"])
            pa = float(d["p_target_after"])
            amp = d.get("target_amplify")
            amp_s = "" if amp is None else f"{float(amp):.4f}"
            f.write(
                f"{int(d.get('cluster_id', -1))},{int(d.get('generation', -1))},"
                f"{d.get('n_targets', '')},{pb:.6f},{pa:.6f},{pa - pb:.6f},"
                f"{amp_s},{d.get('diagnostic_shots', '')}\n"
            )

    print(f"[CQE mech] Saved figure: {out_png}")
    print(f"[CQE mech] Saved figure: {out_pdf}")
    print(f"[CQE mech] Data table: {out_csv}")
    print(
        f"[CQE mech] All {n} events plotted "
        f"({n_up}/{n} increased, mean ΔP_marked = {mean_delta:+.4f})"
    )
    return out_png


def load_cqe_diagnostics_csv(csv_path):
    """Reload diagnostics from cqe_marked_probability_*.csv for replotting."""
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            rec = dict(zip(header, parts))
            # Accept legacy (p_target_*) and current (p_marked_*) column names
            pb_key = "p_marked_before" if "p_marked_before" in rec else "p_target_before"
            pa_key = "p_marked_after" if "p_marked_after" in rec else "p_target_after"
            amp = rec.get("target_amplify", "")
            ds_raw = (rec.get("diagnostic_shots") or "").strip()
            if ds_raw in ("", "exact", "statevector", "statevector_exact"):
                ds_val = "exact"
            else:
                try:
                    ds_val = int(float(ds_raw))
                except ValueError:
                    ds_val = ds_raw
            rows.append(
                {
                    "cluster_id": int(float(rec.get("cluster_id", 0))),
                    "generation": int(float(rec.get("generation", 0))),
                    "n_targets": int(float(rec["n_targets"])) if rec.get("n_targets") else 0,
                    "p_target_before": float(rec[pb_key]),
                    "p_target_after": float(rec[pa_key]),
                    "target_amplify": float(amp) if amp else None,
                    "diagnostic_method": (
                        "statevector_exact" if ds_val == "exact" else "measurement"
                    ),
                    "diagnostic_shots": ds_val,
                }
            )
    return rows



class CQEMixin:
    """Mixin providing CQE oracle / AA / diagnostic methods for HybridQuantumGA."""

    def _default_diagnostic_shots(self, base=512):
        """Legacy API helper; CQE diagnostics now use exact statevector."""
        q = self.encoder.total_qubits
        if q <= 8:
            return max(base, 1024)
        if q <= 15:
            return max(base, 2048)
        return max(base, 4096)

    def _get_statevector(self, individual):
        """Exact statevector from the preparation circuit (Aer, no measurement)."""
        qc = self._build_circuit(individual)
        qc_sv = qc.copy()
        qc_sv.save_statevector()
        tqc = transpile(qc_sv, backend)
        result = backend.run(tqc).result()
        return np.asarray(result.get_statevector(0))

    @staticmethod
    def _bitlist_to_basis_index(bitlist):
        """bitlist[q] = value of qubit q; Qiskit statevector uses qubit 0 as LSB."""
        idx = 0
        for q, bit in enumerate(bitlist):
            if int(bit):
                idx |= 1 << q
        return idx

    def _target_probabilities_from_statevector(
        self, statevector, target_bitstrings, elite_bitstrings
    ):
        """
        Exact marked-set / elite-set probability (no shot noise):
            P = Σ_{x∈S} |⟨x|ψ⟩|²
        """
        amps = np.asarray(statevector, dtype=np.complex128)
        probs = np.abs(amps) ** 2

        if elite_bitstrings and isinstance(elite_bitstrings[0], (int, np.integer)):
            elite_keys = {tuple(elite_bitstrings)}
        else:
            elite_keys = {tuple(e) for e in elite_bitstrings}
        target_keys = {tuple(t) for t in target_bitstrings}

        p_elite = 0.0
        p_target_set = 0.0
        for key in elite_keys:
            p_elite += float(probs[self._bitlist_to_basis_index(key)])
        for key in target_keys:
            p_target_set += float(probs[self._bitlist_to_basis_index(key)])
        return p_elite, p_target_set

    def _collect_grover_target_bitstrings(
        self, target_route, neighborhood_size=None, extra_bitstrings=None
    ):
        if neighborhood_size is None:
            neighborhood_size = self.cqe_neighborhood_size
        n = self.encoder.total_qubits
        targets = []
        seen = set()

        def _add(bits):
            if bits is None or len(bits) != n:
                return
            key = tuple(bits)
            if key in seen:
                return
            seen.add(key)
            targets.append(list(bits))

        elite_bits = self._route_to_bits(target_route)
        _add(elite_bits)

        # Measured bitstrings that decode to elite (may differ from canonical encoding)
        if extra_bitstrings:
            for bits in extra_bitstrings:
                _add(bits)

        for nb_route in self._generate_local_neighbors(
            target_route, n_neighbors=neighborhood_size
        ):
            if not self._is_valid_route(nb_route):
                continue
            _add(self._route_to_bits(nb_route))
        return targets, elite_bits

    def _bitstrings_decoding_to_route(self, counts, route, max_keep=None):
        """Collect raw bitstrings whose decode equals the given route."""
        if max_keep is None:
            max_keep = CQE_MAX_OBSERVED_BITSTRINGS
        if not counts:
            return []
        route_key = tuple(route)
        found = []
        seen = set()
        # Prefer more frequent observed strings
        ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        for bits_key, _cnt in ranked:
            bitlist = self._counts_key_to_bitlist(bits_key)
            if tuple(self.encoder.decode(bitlist)) != route_key:
                continue
            key = tuple(bitlist)
            if key in seen:
                continue
            seen.add(key)
            found.append(bitlist)
            if len(found) >= max_keep:
                break
        return found

    def _build_grover_operator(
        self,
        target_route,
        neighborhood_size=None,
        n_grover_iters=1,
        extra_bitstrings=None,
        state_preparation=None,
    ):
        """
        Build the CQE amplitude-amplification operator.

        Pass the current preparation U as state_preparation so the diffuser is
        U S_0 U† about |ψ⟩=U|0⟩ (not a bare Hadamard diffuser).
        """
        if neighborhood_size is None:
            neighborhood_size = self.cqe_neighborhood_size
        n = self.encoder.total_qubits
        targets, _ = self._collect_grover_target_bitstrings(
            target_route,
            neighborhood_size=neighborhood_size,
            extra_bitstrings=extra_bitstrings,
        )

        if not targets:
            return None

        oracle = QuantumCircuit(n)
        for target in targets:
            for q in range(n):
                if target[q] == 0:
                    oracle.x(q)
            oracle.h(n - 1)
            oracle.mcx(list(range(n - 1)), n - 1)
            oracle.h(n - 1)
            for q in range(n):
                if target[q] == 0:
                    oracle.x(q)

        # AA about |ψ⟩=U|0⟩, not uniform |+⟩^⊗n
        if state_preparation is None:
            state_preparation = QuantumCircuit(n)
            state_preparation.h(range(n))
            print(
                "[warn] CQE GroverOperator missing state_preparation; "
                "using Hadamard diffusion fallback"
            )

        grover_iteration = GroverOperator(
            oracle, state_preparation=state_preparation
        )
        if n_grover_iters <= 1:
            return grover_iteration

        combined = QuantumCircuit(n)
        for _ in range(n_grover_iters):
            combined.append(grover_iteration, range(n))
        return combined

    def _event_params(self, event):
        if isinstance(event, dict):
            return (
                event["route"],
                event.get("neighborhood_size", self.cqe_neighborhood_size),
                event.get("n_grover_iters", 1),
                event.get("extra_bitstrings", None),
            )
        return event, self.cqe_neighborhood_size, 1, None

    def _target_probabilities_from_counts(
        self, counts, target_bitstrings, elite_bitstrings
    ):
        """Estimate elite / marked-set probabilities from measurement counts."""
        total = sum(counts.values())
        if total <= 0:
            return 0.0, 0.0

        target_keys = {tuple(t) for t in target_bitstrings}
        if elite_bitstrings and isinstance(elite_bitstrings[0], (int, np.integer)):
            elite_keys = {tuple(elite_bitstrings)}
        else:
            elite_keys = {tuple(e) for e in elite_bitstrings}
        p_elite = 0.0
        p_target_set = 0.0

        for bits, cnt in counts.items():
            key = tuple(self._counts_key_to_bitlist(bits))
            prob = cnt / total
            if key in elite_keys:
                p_elite += prob
            if key in target_keys:
                p_target_set += prob

        return p_elite, p_target_set

    def _choose_grover_iters(self, target_prob):
        """Fixed to one AA iteration per enhancement event."""
        return self.cqe_grover_iters

    def _safe_amplify_ratio(self, p_before, p_after, shots=None):
        """
        Exact statevector: return None only if p_before ≈ 0.
        Shot-based path: N/A when too few hits (avoids 0→ε fake ratios).
        """
        if shots is None or shots == "exact":
            if p_before < 1e-15:
                return None
            return p_after / p_before
        floor = CQE_MIN_COUNTED_HITS / max(int(shots), 1)
        if p_before < floor:
            return None
        return p_after / p_before

    def _format_amplify(self, ratio):
        return "N/A" if ratio is None else f"{ratio:.2f}x"

    def _print_grover_diagnostic(self, diag):
        print("\n========== Grover Diagnostic ==========")
        print(f"Generation: {diag['generation']}")
        print(f"Action: {diag.get('action', 'apply')}")
        print(f"Targets: {diag['n_targets']} (K={diag.get('neighborhood_size', '?')}, "
              f"observed_elite_strs={diag.get('n_observed_elite', 0)})")
        print(f"Grover iters: {diag.get('n_grover_iters', 0)}")
        print(
            f"Diagnostic: {diag.get('diagnostic_method', 'statevector_exact')} "
            f"(shots={diag.get('diagnostic_shots', 'exact')})"
        )
        print("Before CQE:")
        print(f"  Elite probability      = {diag['p_elite_before']:.6f}")
        print(f"  Target-set probability = {diag['p_target_before']:.6f}")
        print(f"  Elite fitness (route)  = {diag['elite_fitness_before']:.6f}")
        print("After CQE:")
        print(f"  Elite probability      = {diag['p_elite_after']:.6f}")
        print(f"  Target-set probability = {diag['p_target_after']:.6f}")
        print(f"  Elite fitness (route)  = {diag['elite_fitness_after']:.6f}")
        print("Amplification ratio:")
        print(f"  Elite      = {self._format_amplify(diag.get('elite_amplify'))}")
        print(f"  Target-set = {self._format_amplify(diag.get('target_amplify'))}")
        print("=======================================\n")

    def _maybe_apply_cqe(
        self,
        elite_individual,
        elite_route,
        gen,
        population_mean_fitness,
        diagnostic_shots=None,
        neighborhood_size=None,
        extra_bitstrings=None,
    ):
        """
        Periodic CQE: one AA on the current Ry distribution every ΔG generations.
        Marked set = elite encoding + observed decode-matched strings + K neighbors.
        Optional diagnostics use exact statevector (off by default for speed).
        """
        if neighborhood_size is None:
            neighborhood_size = self.cqe_neighborhood_size
        # diagnostic_shots kept for API; ignored in exact mode
        _ = diagnostic_shots
        if extra_bitstrings is None:
            extra_bitstrings = []

        target_bitstrings, elite_bits = self._collect_grover_target_bitstrings(
            elite_route,
            neighborhood_size=neighborhood_size,
            extra_bitstrings=extra_bitstrings,
        )
        if not target_bitstrings:
            return None

        # Elite-related strings for diagnosis: canonical ∪ observed
        elite_related = [elite_bits]
        elite_key = tuple(elite_bits)
        for bits in extra_bitstrings:
            if tuple(bits) != elite_key:
                elite_related.append(bits)

        do_diag = self.record_cqe_diagnostics
        p_elite_before = p_target_before = None
        elite_fitness_before = None
        if do_diag:
            sv_before = self._get_statevector(elite_individual)
            p_elite_before, p_target_before = self._target_probabilities_from_statevector(
                sv_before, target_bitstrings, elite_related,
            )
            elite_fitness_before = self._evaluate(elite_route)

        # Fixed 1 AA iteration per event (independent of diagnostic probs)
        n_iters = self._choose_grover_iters(p_target_before if p_target_before is not None else 0.0)
        self._record_grover(
            elite_route,
            max_qubits_for_grover=30,
            neighborhood_size=neighborhood_size,
            n_grover_iters=n_iters,
            extra_bitstrings=extra_bitstrings,
        )

        if not do_diag:
            return None

        sv_after = self._get_statevector(elite_individual)
        p_elite_after, p_target_after = self._target_probabilities_from_statevector(
            sv_after, target_bitstrings, elite_related,
        )
        elite_fitness_after = self._evaluate(elite_route)

        diag = {
            "generation": gen,
            "action": "apply",
            "note": None,
            "n_targets": len(target_bitstrings),
            "n_observed_elite": len(extra_bitstrings),
            "diagnostic_method": "statevector_exact",
            "diagnostic_shots": "exact",
            "neighborhood_size": neighborhood_size,
            "n_grover_iters": n_iters,
            "p_elite_before": p_elite_before,
            "p_target_before": p_target_before,
            "p_elite_after": p_elite_after,
            "p_target_after": p_target_after,
            "elite_amplify": self._safe_amplify_ratio(
                p_elite_before, p_elite_after, shots="exact"
            ),
            "target_amplify": self._safe_amplify_ratio(
                p_target_before, p_target_after, shots="exact"
            ),
            "elite_fitness_before": elite_fitness_before,
            "elite_fitness_after": elite_fitness_after,
            "population_mean_fitness": population_mean_fitness,
            "global_best_distance": self.best_distance,
        }
        self.grover_diagnostics_log.append(diag)
        if self.grover_diagnostics:
            self._print_grover_diagnostic(diag)
        return diag

    def _record_grover(
        self,
        target_route,
        max_qubits_for_grover=30,
        neighborhood_size=None,
        n_grover_iters=1,
        extra_bitstrings=None,
    ):
        """Record a Grover event and open a new Ry segment (rebuild on next circuit)."""
        if neighborhood_size is None:
            neighborhood_size = self.cqe_neighborhood_size
        if self.encoder.total_qubits > max_qubits_for_grover:
            return
        if n_grover_iters <= 0:
            return
        targets, _ = self._collect_grover_target_bitstrings(
            target_route,
            neighborhood_size=neighborhood_size,
            extra_bitstrings=extra_bitstrings,
        )
        if not targets:
            return
        extras_store = None
        if extra_bitstrings:
            extras_store = [list(b) for b in extra_bitstrings]
        self.grover_events.append(
            {
                "route": list(target_route),
                "neighborhood_size": neighborhood_size,
                "n_grover_iters": n_grover_iters,
                "extra_bitstrings": extras_store,
            }
        )
        for ind in self.population:
            ind.close_segment()
