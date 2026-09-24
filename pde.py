# Parameterized Distribution Evolution (PDE): quantum sampling-distribution loop.
import math
from math import pi

import numpy as np
from qiskit import QuantumCircuit, transpile

from tsp_common import (
    backend,
    total_distance,
    two_opt_optimize,
    QuantumEncoder,
    RY_ALPHA0,
    RY_SOFT_BLEND,
    RY_BIT_CONFIDENCE,
    MUTATION_ANGLE_MAX,
    MUTATION_ANGLE_MIN,
    CQE_DELTA_G,
    CQE_NEIGHBORHOOD_SIZE,
    CQE_GROVER_ITERS,
)
from cqe import CQEMixin


class _IndividualState:
    """Per-individual Ry angle segments between Grover events."""

    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.ry_segments = [np.zeros(n_qubits, dtype=float)]

    def current_ry(self):
        return self.ry_segments[-1]

    def add_ry(self, qubit, angle):
        self.current_ry()[qubit] += angle

    def add_ry_batch(self, angles):
        self.current_ry()[:] += angles

    def total_ry(self):
        total = np.zeros(self.n_qubits, dtype=float)
        for seg in self.ry_segments:
            total += seg
        return total

    def close_segment(self):
        self.ry_segments.append(np.zeros(self.n_qubits, dtype=float))

    def normalize_angles(self, limit=pi):
        """Wrap Ry angles into (-limit, limit] (default (-π, π])."""
        period = 2.0 * limit
        for seg in self.ry_segments:
            seg[:] = np.mod(seg + limit, period) - limit



class HybridQuantumGA(CQEMixin):
    def __init__(
        self,
        city_indices,
        all_cities,
        population_size=20,
        use_two_opt=False,
        use_cqe=True,
        grover_diagnostics=False,
        record_cqe_diagnostics=False,
        cqe_neighborhood_size=CQE_NEIGHBORHOOD_SIZE,
        cqe_delta_g=CQE_DELTA_G,
        cqe_grover_iters=CQE_GROVER_ITERS,
    ):
        """
        use_two_opt: run two-opt after each QGA generation
        use_cqe: enable periodic CQE (amplitude amplification)
        record_cqe_diagnostics: exact statevector P_marked before/after each CQE
            (expensive; leave False for data runs, enable for Fig. 3)
        grover_diagnostics: print per-CQE diagnostics (requires record_cqe_diagnostics)
        Defaults: ΔG=8, K_nb=8, one AA iteration per event
        """
        self.encoder = QuantumEncoder(city_indices)
        self.all_cities = all_cities
        self.population_size = population_size
        self.use_two_opt = use_two_opt
        self.use_cqe = use_cqe
        self.record_cqe_diagnostics = bool(record_cqe_diagnostics)
        self.grover_diagnostics = bool(grover_diagnostics) and self.record_cqe_diagnostics
        self.cqe_neighborhood_size = cqe_neighborhood_size
        self.cqe_delta_g = cqe_delta_g
        self.cqe_grover_iters = cqe_grover_iters
        self.grover_events = []  # dict: route / neighborhood_size / n_grover_iters
        self.grover_diagnostics_log = []
        self.grover_skip_log = []
        self.population = self._init_population(size=population_size)
        self.best_solution = None
        self.best_distance = float('inf')

    def _init_population(self, size=20):
        n = self.encoder.total_qubits
        population = []
        for _ in range(size):
            ind = _IndividualState(n)
            for q in range(n):
                ind.add_ry(q, (np.random.rand() - 0.5) * 0.6)
            population.append(ind)
        return population

    def _route_to_bits(self, route):
        bits = []
        for pos in route:
            bits.extend(int(b) for b in format(pos, f'0{self.encoder.bits_per_city}b'))
        return bits

    def _build_circuit(self, individual):
        """Recompile from classical params: H → [Ry segment → Grover]×k → final Ry."""
        n = self.encoder.total_qubits
        qc = QuantumCircuit(n)
        for q in range(n):
            qc.h(q)

        for seg_idx, seg_angles in enumerate(individual.ry_segments):
            for q in range(n):
                angle = seg_angles[q]
                if abs(angle) > 1e-12:
                    qc.ry(angle, q)
            if seg_idx < len(self.grover_events):
                route, nb, n_iters, extras = self._event_params(
                    self.grover_events[seg_idx]
                )
                # |ψ⟩ = U|0⟩ on the wire; pass U as state_preparation for AA
                prep_u = qc.copy()
                grover_op = self._build_grover_operator(
                    route,
                    neighborhood_size=nb,
                    n_grover_iters=n_iters,
                    extra_bitstrings=extras,
                    state_preparation=prep_u,
                )
                if grover_op is not None:
                    qc.append(grover_op, range(n))
        return qc

    def circuit_depth(self, individual=None):
        if individual is None:
            individual = self.population[0]
        return self._build_circuit(individual).depth()

    def _measure(self, individual, shots=64):
        best_route, _ = self._measure_with_counts(individual, shots=shots)
        return best_route

    def _measure_with_counts(self, individual, shots=128):
        qc = self._build_circuit(individual)
        measured = qc.copy()
        measured.measure_all()
        tqc = transpile(measured, backend)
        counts = backend.run(tqc, shots=shots).result().get_counts()
        best_route = None
        best_score = -1.0
        for bits, cnt in counts.items():
            bitlist = [int(b) for b in bits[::-1]]
            route = self.encoder.decode(bitlist)
            score = self._evaluate(route)
            if score > best_score:
                best_score = score
                best_route = route
        return best_route, counts

    def _evaluate(self, route):
        global_route = [self.encoder.city_indices[i] for i in route]
        dist = total_distance(self.all_cities, global_route)
        return 1.0 / (dist + 1e-9)

    def _evaluate_counts_distribution(self, counts):
        """Mean fitness estimated from measurement counts (diagnostics)."""
        total = sum(counts.values())
        if total <= 0:
            return 0.0
        weighted = 0.0
        for bits, cnt in counts.items():
            bitlist = [int(b) for b in bits[::-1]]
            route = self.encoder.decode(bitlist)
            weighted += (cnt / total) * self._evaluate(route)
        return weighted

    def _counts_key_to_bitlist(self, bits_key):
        return [int(b) for b in bits_key[::-1]]

    def _classical_local_improve(self, global_route):
        """Optional two-opt local improvement after each QGA generation."""
        if not self.use_two_opt or len(global_route) <= 2:
            return global_route
        return two_opt_optimize(self.all_cities, global_route)

    def _update_quantum(self, elite_route, step_scale=0.15, soft_blend=None):
        """
        Distribution update:
        1) Δθ_q = α π (2e_q − 1)
        2) mild soft pull toward elite bit-marginal targets each generation
        """
        if soft_blend is None:
            soft_blend = RY_SOFT_BLEND
        n = self.encoder.total_qubits
        elite_bits = self._route_to_bits(elite_route)
        if len(elite_bits) != n:
            return

        delta = np.array([
            step_scale * pi if elite_bits[q] == 1 else -step_scale * pi
            for q in range(n)
        ], dtype=float)

        # P(1)=(1+sinθ)/2 → invert target marginal to θ*
        conf = min(max(RY_BIT_CONFIDENCE, 0.55), 0.95)
        target_theta = np.zeros(n, dtype=float)
        for q, bit in enumerate(elite_bits):
            p1 = conf if bit == 1 else (1.0 - conf)
            s = max(-1.0, min(1.0, 2.0 * p1 - 1.0))
            target_theta[q] = math.asin(s)

        for ind in self.population:
            soft = soft_blend * (target_theta - ind.total_ry())
            ind.add_ry_batch(delta + soft)
            ind.normalize_angles()

    def _mutation_angle(self, gen, generations):
        if generations <= 1:
            return MUTATION_ANGLE_MIN
        ratio = gen / max(1, generations - 1)
        return MUTATION_ANGLE_MAX * (1.0 - ratio) + MUTATION_ANGLE_MIN * ratio

    @staticmethod
    def _ry_to_bit_probs(individual):
        """H→Ry(θ) product-state marginal: P(1)=(1+sin θ)/2."""
        theta = np.asarray(individual.total_ry(), dtype=float)
        return 0.5 * (1.0 + np.sin(theta))

    @staticmethod
    def _format_bit_probs(probs):
        return "[" + ", ".join(f"{p:.4f}" for p in probs) + "]"

    def _is_valid_route(self, route):
        return (
            len(route) == self.encoder.num_cities
            and len(set(route)) == self.encoder.num_cities
            and all(
                0 <= city < self.encoder.num_cities
                for city in route
            )
        )

    def _generate_local_neighbors(self, route, n_neighbors=5):
        """Valid local neighbors; keep the best n_neighbors by TSP fitness."""
        n = len(route)
        if n <= 2:
            return []

        candidates = []
        seen = {tuple(route)}

        for i in range(n):
            for j in range(i + 1, n):
                neighbor = route.copy()
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                key = tuple(neighbor)
                if key not in seen and self._is_valid_route(neighbor):
                    seen.add(key)
                    candidates.append(neighbor)

        for i in range(n):
            for j in range(i + 2, n):
                neighbor = route.copy()
                neighbor[i:j + 1] = neighbor[i:j + 1][::-1]
                key = tuple(neighbor)
                if key not in seen and self._is_valid_route(neighbor):
                    seen.add(key)
                    candidates.append(neighbor)

        if not candidates:
            return []

        scored = [
            (self._evaluate(nb), nb) for nb in candidates
        ]
        # _evaluate = 1/dist; higher is better
        scored.sort(key=lambda x: x[0], reverse=True)
        return [nb for _, nb in scored[:n_neighbors]]

    def optimize(self, generations=40, measure_shots=64, diagnostic_shots=None,
                 elite_fraction=0.2):
        # diagnostic_shots kept for API; CQE diagnostics use exact statevector
        if diagnostic_shots is None:
            diagnostic_shots = "exact"
        for gen in range(generations):
            results = [
                self._measure_with_counts(ind, shots=measure_shots)
                for ind in self.population
            ]
            counts_list = [x[1] for x in results]
            routes = [
                r if r is not None else self.encoder.decode([0] * self.encoder.total_qubits)
                for r, _ in results
            ]
            fitness = [self._evaluate(r) for r in routes]
            elite_idx = int(np.argmax(fitness))
            elite_route = routes[elite_idx]
            global_route = [self.encoder.city_indices[i] for i in elite_route]
            current_dist = total_distance(self.all_cities, global_route)
            improved_global = self._classical_local_improve(global_route)
            improved_dist = total_distance(self.all_cities, improved_global)
            if improved_dist < current_dist - 1e-9:
                inv_map = {gidx: i for i, gidx in enumerate(self.encoder.city_indices)}
                try:
                    mapped_local = [inv_map[g] for g in improved_global[:self.encoder.num_cities]]
                    elite_route = mapped_local
                except Exception:
                    pass
                current_dist = improved_dist
            if current_dist < self.best_distance - 1e-9:
                self.best_solution = improved_global
                self.best_distance = current_dist

            elite_probs = self._ry_to_bit_probs(self.population[elite_idx])
            print(
                f"Gen {gen + 1}: P(1) = {self._format_bit_probs(elite_probs)}"
            )

            # Raw bitstrings in elite counts that decode to elite_route (oracle marks)
            observed_elite_strs = self._bitstrings_decoding_to_route(
                counts_list[elite_idx], elite_route
            )

            scale = RY_ALPHA0 * (1 - gen / max(1, generations))
            self._update_quantum(elite_route, step_scale=scale)
            if self.use_cqe and gen % self.cqe_delta_g == 0 and gen > 0:
                elite_individual = self.population[elite_idx]
                self._maybe_apply_cqe(
                    elite_individual,
                    elite_route,
                    gen,
                    population_mean_fitness=float(np.mean(fitness)),
                    diagnostic_shots=diagnostic_shots,
                    extra_bitstrings=observed_elite_strs,
                )
            mut_angle = self._mutation_angle(gen, generations)
            for ind in self.population:
                for _ in range(2):
                    qidx = np.random.randint(0, self.encoder.total_qubits)
                    ind.add_ry(qidx, (np.random.rand() - 0.5) * mut_angle)
                ind.normalize_angles()
        return self.best_solution, self.best_distance
