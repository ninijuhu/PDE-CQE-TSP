"""Shared TSP utilities and default hyperparameters for PDE / CQE."""

import os
import math
from math import sqrt, pi

import numpy as np
import matplotlib.pyplot as plt
from qiskit import Aer

try:
    import tsplib95
except ImportError as exc:
    raise ImportError(
        "tsplib95 is required to read .tsp files: pip install tsplib95"
    ) from exc

os.makedirs("output/qTSP_solutions", exist_ok=True)
backend = Aer.get_backend("aer_simulator")

# CQE defaults
CQE_DELTA_G = 8
CQE_NEIGHBORHOOD_SIZE = 8
CQE_GROVER_ITERS = 1
CQE_MIN_COUNTED_HITS = 2
CQE_MAX_OBSERVED_BITSTRINGS = 32

# PDE distribution-evolution defaults
RY_ALPHA0 = 0.20
RY_SOFT_BLEND = 0.12
RY_BIT_CONFIDENCE = 0.80
MUTATION_ANGLE_MAX = 0.30
MUTATION_ANGLE_MIN = 0.05


def load_cities_from_tsp(file_path):
    """Load city coordinates from a TSPLIB .tsp file as [[id, x, y], ...]."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"TSP file not found: {file_path}")
    problem = tsplib95.load(file_path)
    nodes = problem.node_coords or problem.display_data
    if not nodes:
        raise ValueError(
            f"No coordinates in {file_path} (need NODE_COORD_SECTION or DISPLAY_DATA)"
        )
    return [
        [int(node_id), float(coord[0]), float(coord[1])]
        for node_id, coord in sorted(nodes.items(), key=lambda item: item[0])
    ]


def euclidean_distance(city1, city2):
    return sqrt((city1[1] - city2[1]) ** 2 + (city1[2] - city2[2]) ** 2)


def total_distance(cities, route):
    if len(route) < 2:
        return float("inf")
    dist = 0.0
    for i in range(len(route) - 1):
        dist += euclidean_distance(cities[route[i]], cities[route[i + 1]])
    dist += euclidean_distance(cities[route[-1]], cities[route[0]])
    return dist


def plot_solution(cities, route, title="TSP Solution", instance_name=None):
    plt.figure(figsize=(12, 8))
    x = [cities[i][1] for i in route] + [cities[route[0]][1]]
    y = [cities[i][2] for i in route] + [cities[route[0]][2]]
    plt.plot(x, y, "o-", markersize=8, linewidth=1.5)
    for city in cities:
        plt.text(city[1], city[2], str(city[0]), fontsize=10, ha="right")
    plt.title(f"{title} (Distance: {total_distance(cities, route):.2f})")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    suffix = f"_{instance_name}" if instance_name else ""
    plot_path = f"output/qTSP_solutions/solution{suffix}_{len(cities)}cities.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved tour plot: {plot_path}")


def two_opt_optimize(cities, route, max_iter=1000):
    """First-improvement 2-opt until local optimality or max_iter."""
    best_route = route.copy()
    best_dist = total_distance(cities, best_route)
    improved = True
    iter_count = 0
    n = len(best_route)
    while improved and iter_count < max_iter:
        improved = False
        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                if j - i == 1:
                    continue
                new_route = best_route[:i] + best_route[i : j + 1][::-1] + best_route[j + 1 :]
                new_dist = total_distance(cities, new_route)
                if new_dist < best_dist - 1e-9:
                    best_route, best_dist = new_route, new_dist
                    improved = True
        iter_count += 1
    return best_route


class QuantumEncoder:
    """Binary positional encoding with local-to-global city-index mapping."""

    def __init__(self, city_indices):
        self.city_indices = city_indices
        self.num_cities = len(city_indices)
        self.bits_per_city = (
            math.ceil(math.log2(self.num_cities)) if self.num_cities > 1 else 1
        )
        self.total_qubits = self.num_cities * self.bits_per_city
        if self.total_qubits > 30:
            raise ValueError(
                f"Requires {self.total_qubits} qubits (>30); reduce subproblem size"
            )

    def decode(self, bitstring):
        route, used = [], set()
        for i in range(0, len(bitstring), self.bits_per_city):
            chunk = bitstring[i : i + self.bits_per_city]
            pos = int("".join(map(str, chunk)), 2)
            if pos >= self.num_cities:
                pos = self._find_valid_pos(pos, used)
            if pos not in used:
                route.append(pos)
                used.add(pos)
        for pos in range(self.num_cities):
            if pos not in used:
                route.append(pos)
        return route[: self.num_cities]

    def _find_valid_pos(self, pos, used_set):
        for offset in [+1, -1, +2, -2, +3, -3]:
            valid_pos = (pos + offset) % self.num_cities
            if valid_pos not in used_set and 0 <= valid_pos < self.num_cities:
                return valid_pos
        return pos % self.num_cities
