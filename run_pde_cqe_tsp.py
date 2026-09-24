# Main entry: RBD clustering + reconstruction + CLI for CDE / PDE / Full.
# Place alongside tsp_common.py / pde.py / cqe.py / cde.py
#
# Examples:
#   python run_pde_cqe_tsp.py DataSet/bays29.tsp --mode full
#   python run_pde_cqe_tsp.py DataSet/bays29.tsp --mode full --cqe-mech-diag
#   python run_pde_cqe_tsp.py --replot-cqe-csv output/qTSP_solutions/cqe_marked_probability_bays29.csv
#
import os
import math
import argparse

import numpy as np
from sklearn.cluster import KMeans

from tsp_common import (
    load_cities_from_tsp,
    euclidean_distance,
    total_distance,
    plot_solution,
    two_opt_optimize,
    RY_ALPHA0,
    RY_SOFT_BLEND,
)
from cqe import plot_cqe_marked_set_probability, load_cqe_diagnostics_csv
from pde import HybridQuantumGA
from cde import ClassicalDistributionGA


def balanced_cluster(cities, max_cluster_size=8):
    """Two-level KMeans clustering with a fixed 4 top-level clusters."""
    n = len(cities)
    if n <= max_cluster_size:
        return [list(range(n))]

    coords = np.array([city[1:] for city in cities])

    kmeans = KMeans(n_clusters=4, n_init=10).fit(coords)
    level1 = [[] for _ in range(4)]
    for i, label in enumerate(kmeans.labels_):
        level1[label].append(i)

    final_clusters = []
    for big_cluster in level1:
        if len(big_cluster) <= max_cluster_size:
            final_clusters.append(big_cluster)
        else:
            sub_k = math.ceil(len(big_cluster) / max_cluster_size)
            sub_coords = np.array([cities[i][1:] for i in big_cluster])
            sub_kmeans = KMeans(n_clusters=sub_k, n_init=10).fit(sub_coords)
            sub_clusters = [[] for _ in range(sub_k)]
            for idx, label in enumerate(sub_kmeans.labels_):
                sub_clusters[label].append(big_cluster[idx])
            for sc in sub_clusters:
                while len(sc) > max_cluster_size:
                    final_clusters.append(sc[:max_cluster_size])
                    sc = sc[max_cluster_size:]
                if sc:
                    final_clusters.append(sc)
    return final_clusters


def cluster_distance(cities, cluster_a, cluster_b):
    return min(euclidean_distance(cities[i], cities[j]) for i in cluster_a for j in cluster_b)


def cluster_order_distance(order, cluster_routes, cities):
    dist = 0.0
    for i in range(len(order) - 1):
        dist += cluster_distance(cities,
                                 cluster_routes[order[i]],
                                 cluster_routes[order[i + 1]])
    return dist


def optimize_cluster_order(cluster_routes, cities):
    n = len(cluster_routes)
    if n <= 1:
        return list(range(n))
    unvisited = set(range(1, n))
    order = [0]
    while unvisited:
        last = order[-1]
        next_c = min(unvisited, key=lambda u: cluster_distance(cities, cluster_routes[last], cluster_routes[u]))
        order.append(next_c)
        unvisited.remove(next_c)
    improved = True
    while improved:
        improved = False
        for i in range(1, len(order) - 2):
            for j in range(i + 1, len(order) - 1):
                if j - i == 1:
                    continue
                new_order = order[:i] + order[i:j + 1][::-1] + order[j + 1:]
                if cluster_order_distance(new_order, cluster_routes, cities) < \
                        cluster_order_distance(order, cluster_routes, cities):
                    order = new_order
                    improved = True
    return order


def boundary_fusion_opt(global_route, cities, window=3):
    n = len(global_route)
    improved = True
    iter_count = 0
    while improved and iter_count < 3:
        improved = False
        for i in range(n):
            a = i
            b = (i + 1) % n
            left = [(a - k) % n for k in range(1, window + 1)]
            right = [((b + k) % n) for k in range(1, window + 1)]
            for L in left:
                for R in right:
                    new_route = global_route.copy()
                    new_route[L], new_route[R] = new_route[R], new_route[L]
                    if total_distance(cities, new_route) < total_distance(cities, global_route) - 1e-9:
                        global_route = new_route
                        improved = True
        iter_count += 1
    return global_route


def connect_clusters_in_order(cluster_routes, order, cities):
    global_route = []
    for idx in order:
        global_route.extend(cluster_routes[idx])
    all_indices = set(range(len(cities)))
    used = set(global_route)
    missing = list(all_indices - used)
    if missing:
        print(f"Warning: {len(missing)} cities were missing and have been appended to the route.")
        global_route.extend(missing)
    seen, dedup_route = set(), []
    for idx in global_route:
        if idx not in seen:
            dedup_route.append(idx)
            seen.add(idx)
    fused = boundary_fusion_opt(dedup_route, cities, window=3)
    return fused


def optimize_subcluster(cluster, cities, mode="full", cluster_id=0, record_cqe_diagnostics=False):
    """
    mode:
      - "full": Quantum PDE + CQE
      - "qde":  Quantum PDE only (use_cqe=False)
      - "cde":  Classical Distribution Evolution

    record_cqe_diagnostics:
      False (default): run CQE without P_marked statevector (faster)
      True: record Fig. 3 before/after diagnostics

    Returns:
        (route, cqe_diagnostics)
    """
    if len(cluster) <= 2:
        return cluster, []
    num_cities = len(cluster)
    bits = math.ceil(math.log2(num_cities)) if num_cities > 1 else 1
    total_qubits = num_cities * bits
    try:
        print(
            f"Optimizing subcluster {cluster_id} ({num_cities} cities, mode={mode})... "
            f"(estimated {total_qubits} bits/qubits)"
        )
        if mode == "cde":
            solver = ClassicalDistributionGA(
                cluster, cities, population_size=20, use_two_opt=False
            )
            route, dist = solver.optimize(generations=40, measure_shots=128)
            print(f"  [CDE] subcluster best distance = {dist:.4f}")
            return route, []

        use_cqe = mode != "qde"
        qga = HybridQuantumGA(
            cluster,
            cities,
            population_size=20,
            use_two_opt=False,
            use_cqe=use_cqe,
            grover_diagnostics=bool(record_cqe_diagnostics) and use_cqe,
            record_cqe_diagnostics=bool(record_cqe_diagnostics) and use_cqe,
        )
        initial_depth = qga.circuit_depth()
        print(
            f"  qubits: {total_qubits}, initial recompiled depth: {initial_depth}, "
            f"use_cqe={use_cqe}, "
            f"ΔG={qga.cqe_delta_g}, K={qga.cqe_neighborhood_size}, "
            f"iters={qga.cqe_grover_iters}, "
            f"Ry_α0={RY_ALPHA0}, soft_blend={RY_SOFT_BLEND}"
        )
        route, dist = qga.optimize(generations=40, measure_shots=128)
        final_depth = qga.circuit_depth()
        n_grover = len(qga.grover_events)
        n_ry_layers = len(qga.population[0].ry_segments)
        print(
            f"  final recompiled depth: {final_depth} "
            f"(Ry layers: {n_ry_layers}, Grover injections: {n_grover}; "
            f"vs initial +{final_depth - initial_depth})"
        )

        diags = []
        for d in qga.grover_diagnostics_log:
            rec = dict(d)
            rec["cluster_id"] = cluster_id
            diags.append(rec)

        if diags or qga.grover_skip_log:
            applied = diags
            skipped = qga.grover_skip_log
            elite_amps = [
                d["elite_amplify"] for d in applied if d.get("elite_amplify") is not None
            ]
            target_amps = [
                d["target_amplify"] for d in applied if d.get("target_amplify") is not None
            ]
            avg_elite = f"{np.mean(elite_amps):.2f}x" if elite_amps else "N/A"
            avg_target = f"{np.mean(target_amps):.2f}x" if target_amps else "N/A"
            print(
                f"  CQE diagnostics: applied {len(applied)} / skipped {len(skipped)}; "
                f"mean effective Elite amplify = {avg_elite}, "
                f"mean effective Target-set amplify = {avg_target}"
            )

        if route is None:
            raise RuntimeError("Optimizer returned no solution")
        if isinstance(route, list) and len(route) == num_cities:
            return route, diags
        print("Warning: unexpected return format or length mismatch; falling back to classical 2-opt")
        return two_opt_optimize(cities, cluster), diags
    except Exception as e:
        print(
            f"subcluster size={num_cities}, bits/qubits={total_qubits} failed: {e}; "
            f"falling back to classical 2-opt"
        )
        return two_opt_optimize(cities, cluster), []


def solve_tsp(cities, max_cluster_size=8, mode="full", record_cqe_diagnostics=False):
    """
    mode ∈ {"cde", "qde", "full"} for CDE / Quantum PDE / Full(+CQE) comparison.

    Returns:
        final_route, final_dist, cqe_diagnostics
    """
    if mode not in ("cde", "qde", "full"):
        raise ValueError(f"Unknown mode={mode}; expected cde / qde / full")

    n = len(cities)
    print(f"Starting optimization of {n} cities... mode={mode}")
    if mode == "full":
        print(
            f"CQE diagnostics (P_marked statevector): "
            f"{'on' if record_cqe_diagnostics else 'off (default for data runs; faster)'}"
        )
    clusters = balanced_cluster(cities, max_cluster_size=max_cluster_size)
    print(f"\nClustering result ({len(clusters)} subclusters):")
    for i, cluster in enumerate(clusters):
        print(f"  subcluster {i}: {len(cluster)} cities {cluster[:8]}...")

    # Serial subclusters: avoid oversubscribing Aer on many 20+ qubit circuits
    packed = [
        optimize_subcluster(
            cluster,
            cities,
            mode=mode,
            cluster_id=i,
            record_cqe_diagnostics=record_cqe_diagnostics,
        )
        for i, cluster in enumerate(clusters)
    ]
    results = [item[0] for item in packed]
    cqe_diagnostics = []
    for item in packed:
        cqe_diagnostics.extend(item[1])

    print("\nStarting inter-cluster optimization...")
    cluster_order = optimize_cluster_order(results, cities)
    print(f"Cluster order: {cluster_order}")
    global_route = connect_clusters_in_order(results, cluster_order, cities)
    init_dist = total_distance(cities, global_route)
    print(f"\nInitial global distance: {init_dist:.2f}")
    final_route = two_opt_optimize(cities, global_route)
    final_route = boundary_fusion_opt(final_route, cities, window=4)
    final_dist = total_distance(cities, final_route)
    print(f"Final global distance [{mode}]: {final_dist:.2f}")
    if mode == "full":
        print(f"Aggregated CQE diagnostic events: {len(cqe_diagnostics)}")
    return final_route, final_dist, cqe_diagnostics



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PDE–CQE TSP runner (full model or matched CDE control)"
    )
    parser.add_argument(
        "tsp_file",
        nargs="?",
        default=None,
        help="Path to TSPLIB .tsp file; optional if only using --replot-cqe-csv",
    )
    parser.add_argument(
        "--mode",
        choices=["cde", "full"],
        default="full",
        help="full=PDE+CQE; cde=classical matched control",
    )
    parser.add_argument(
        "--cqe-mech-events",
        type=int,
        default=8,
        help="(ignored) all CQE events are plotted; kept for CLI compatibility",
    )
    parser.add_argument(
        "--cqe-mech-diag",
        action="store_true",
        help="Enable CQE marked-set diagnostics (exact statevector + mech plot); "
        "off by default for faster data runs",
    )
    parser.add_argument(
        "--no-cqe-mech-plot",
        action="store_true",
        help="Even with --cqe-mech-diag, skip the mech plot (diagnostics still recorded)",
    )
    parser.add_argument(
        "--replot-cqe-csv",
        default=None,
        help="Replot mech figure from an existing cqe_marked_probability_*.csv "
        "(no optimization)",
    )
    args = parser.parse_args()

    if args.replot_cqe_csv:
        csv_path = args.replot_cqe_csv
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(csv_path)
        base = os.path.basename(csv_path)
        inst = None
        if base.startswith("cqe_marked_probability_") and base.endswith(".csv"):
            inst = base[len("cqe_marked_probability_") : -len(".csv")]
        diags = load_cqe_diagnostics_csv(csv_path)
        out_dir = os.path.dirname(os.path.abspath(csv_path)) or "output/qTSP_solutions"
        plot_cqe_marked_set_probability(
            diags,
            instance_name=inst,
            max_events=max(3, int(args.cqe_mech_events)),
            out_dir=out_dir,
        )
        raise SystemExit(0)

    if not args.tsp_file:
        parser.error("Please provide tsp_file, or use --replot-cqe-csv")

    cities = load_cities_from_tsp(args.tsp_file)
    instance_name = os.path.splitext(os.path.basename(args.tsp_file))[0]
    print(f"Loaded TSP instance: {instance_name}, cities = {len(cities)}")

    mode = args.mode
    print("\n" + "=" * 60)
    print(f"Running mode = {mode}")
    print("=" * 60)
    best_route, best_dist, cqe_diags = solve_tsp(
        cities,
        max_cluster_size=8,
        mode=mode,
        record_cqe_diagnostics=bool(args.cqe_mech_diag),
    )
    print(f"\n[{mode}] Best route (first 20 city IDs):")
    print(
        " -> ".join(map(str, [cities[i][0] for i in best_route[:20]])) + " -> ..."
    )
    print(f"[{mode}] Shortest total distance: {best_dist:.2f}")
    plot_solution(
        cities,
        best_route,
        title=f"TSP Solution ({mode})",
        instance_name=instance_name,
    )
    if (
        mode == "full"
        and cqe_diags
        and args.cqe_mech_diag
        and not args.no_cqe_mech_plot
    ):
        plot_cqe_marked_set_probability(
            cqe_diags,
            instance_name=instance_name,
            max_events=max(3, int(args.cqe_mech_events)),
        )
