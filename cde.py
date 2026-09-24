"""Classical Distribution Evolution (CDE): matched classical control for PDE."""

import numpy as np

from tsp_common import (
    total_distance,
    two_opt_optimize,
    QuantumEncoder,
    RY_ALPHA0,
    RY_SOFT_BLEND,
    RY_BIT_CONFIDENCE,
    MUTATION_ANGLE_MAX,
    MUTATION_ANGLE_MIN,
)


class _ClassicalDistributionIndividual:
    """Bernoulli probability vector matched to PDE product-state marginals."""

    def __init__(self, n_bits):
        self.n_bits = n_bits
        self.prob = np.clip(
            0.5 + (np.random.rand(n_bits) - 0.5) * 0.3,
            0.05,
            0.95,
        )

    def sample(self):
        return [1 if np.random.rand() < p else 0 for p in self.prob]

    def update(self, elite_bits, learning_rate):
        for i, bit in enumerate(elite_bits):
            if bit == 1:
                self.prob[i] += learning_rate * (1.0 - self.prob[i])
            else:
                self.prob[i] -= learning_rate * self.prob[i]
        self.prob = np.clip(self.prob, 0.05, 0.95)

    def mutate(self, strength=0.3):
        idx = np.random.randint(0, self.n_bits)
        self.prob[idx] += (np.random.rand() - 0.5) * strength
        self.prob = np.clip(self.prob, 0.05, 0.95)


class ClassicalDistributionGA:
    """
    CDE baseline aligned with HybridQuantumGA settings
    (encoding, population, generations, sampling budget, fitness).
    """

    def __init__(self, city_indices, all_cities, population_size=20, use_two_opt=False):
        self.encoder = QuantumEncoder(city_indices)
        self.all_cities = all_cities
        self.population_size = population_size
        self.use_two_opt = use_two_opt
        self.population = [
            _ClassicalDistributionIndividual(self.encoder.total_qubits)
            for _ in range(population_size)
        ]
        self.best_solution = None
        self.best_distance = float("inf")

    def _route_to_bits(self, route):
        bits = []
        for pos in route:
            bits.extend(int(b) for b in format(pos, f"0{self.encoder.bits_per_city}b"))
        return bits

    def _evaluate(self, route):
        global_route = [self.encoder.city_indices[i] for i in route]
        dist = total_distance(self.all_cities, global_route)
        return 1.0 / (dist + 1e-9)

    def _classical_local_improve(self, global_route):
        if not self.use_two_opt or len(global_route) <= 2:
            return global_route
        return two_opt_optimize(self.all_cities, global_route)

    def _measure(self, individual, shots=128):
        """Sample `shots` bitstrings and keep the best decoded route."""
        best_route = None
        best_score = -1.0
        for _ in range(shots):
            bits = individual.sample()
            route = self.encoder.decode(bits)
            score = self._evaluate(route)
            if score > best_score:
                best_score = score
                best_route = route
        return best_route

    def optimize(self, generations=40, measure_shots=128):
        for gen in range(generations):
            classical_pop = [
                self._measure(ind, shots=measure_shots) for ind in self.population
            ]
            routes = [
                r if r is not None else self.encoder.decode([0] * self.encoder.total_qubits)
                for r in classical_pop
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
                    mapped_local = [
                        inv_map[g] for g in improved_global[: self.encoder.num_cities]
                    ]
                    elite_route = mapped_local
                except Exception:
                    pass
                current_dist = improved_dist
            if current_dist < self.best_distance - 1e-9:
                self.best_solution = improved_global
                self.best_distance = current_dist

            elite_probs = np.asarray(self.population[elite_idx].prob, dtype=float)
            print(
                f"[CDE] Gen {gen + 1}: P(1) = "
                f"[{', '.join(f'{p:.4f}' for p in elite_probs)}]"
            )

            lr = RY_ALPHA0 * (1 - gen / max(1, generations))
            elite_bits = self._route_to_bits(elite_route)
            ratio = gen / max(1, generations - 1)
            mut_strength = (
                MUTATION_ANGLE_MAX * (1.0 - ratio) + MUTATION_ANGLE_MIN * ratio
            )
            for ind in self.population:
                ind.update(elite_bits, lr)
                for i, bit in enumerate(elite_bits):
                    target_p = RY_BIT_CONFIDENCE if bit == 1 else (1.0 - RY_BIT_CONFIDENCE)
                    ind.prob[i] += RY_SOFT_BLEND * (target_p - ind.prob[i])
                ind.prob = np.clip(ind.prob, 0.05, 0.95)
                for _ in range(2):
                    ind.mutate(strength=mut_strength)

        return self.best_solution, self.best_distance
