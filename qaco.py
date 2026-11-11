def city_dist(sample, city1, city2):
    if city1 == city2:
        return 9999
    loc1 = sample[city1]
    loc2 = sample[city2]
    return math.sqrt((loc1[1] - loc2[1]) ** 2 + (loc1[2] - loc2[2]) ** 2)


def total_distance_qaco(sample, route):
    distance = 0.0
    for i in range(len(route) - 1):
        distance += city_dist(sample, route[i], route[i + 1])
    distance += city_dist(sample, route[-1], route[0])
    return distance


def _repair_to_permutation(solution_buffer, num_cities):
    final_solution = []
    used = set()
    for s in solution_buffer:
        if s not in used and s < num_cities:
            final_solution.append(s)
            used.add(s)
    for s in range(num_cities):
        if s not in used:
            final_solution.append(s)
    return final_solution


def quantum_ant_logN(ancilla, pheromone, num_cities, shots=128):
    num_bits = math.ceil(math.log2(num_cities))
    ant = QuantumRegister(num_cities * num_bits, "ant")
    cr = ClassicalRegister(num_cities * num_bits, "cr")
    c_qubit = QuantumRegister(1, "ctrl")
    circ = QuantumCircuit(c_qubit, ant, cr)

    circ.ry(ancilla, 0)
    for i in range(len(pheromone)):
        circ.ry(pheromone[i], i + 1)

    for i in range(1, len(pheromone)):
        for j in range(i + 1, len(pheromone)):
            circ.cswap(0, i, j)

    circ.measure(ant, cr)

    backend = Aer.get_backend("aer_simulator")
    bitstring = (
        backend.run(transpile(circ, backend), shots=shots, memory=True)
        .result()
        .get_memory()[0]
    )

    solution_buffer = []
    for i in range(0, len(bitstring), num_bits):
        solution_buffer.append(int(bitstring[i : i + num_bits], 2))
    return _repair_to_permutation(solution_buffer, num_cities)


def update_pheromone(current_solution, current_best, num_bits):
    total_theta = []
    for j in range(len(current_solution)):
        bin_current = bin(int(current_solution[j]))[2:].rjust(num_bits, "0")
        bin_best = bin(int(current_best[j]))[2:].rjust(num_bits, "0")
        d_theta = []
        for i in range(num_bits):
            if int(bin_current[i]) == int(bin_best[i]):
                d_theta.append(0.01 * math.pi)
            else:
                d_theta.append(0.05 * math.pi)
        total_theta.append(d_theta)
    return [item for sublist in total_theta for item in sublist]


def complete_qaco_logN(sample, max_loops=40):
    num_cities = len(sample)
    num_bits = math.ceil(math.log2(num_cities))
    pheromone = [math.pi / 2] * (num_cities * num_bits)
    ancilla = math.pi / 4
    best_route = list(range(num_cities))
    random.shuffle(best_route)
    best_distance = total_distance_qaco(sample, best_route)

    for _ in range(max_loops):
        current_solution = quantum_ant_logN(ancilla, pheromone, num_cities, shots=128)
        dist = total_distance_qaco(sample, current_solution)
        if dist < best_distance:
            best_distance = dist
            best_route = current_solution[:]
        delta = update_pheromone(current_solution, best_route, num_bits)
        pheromone = [p + d for p, d in zip(pheromone, delta)]

    best_route.append(best_route[0])
    return best_route


def optimize_subcluster(cluster, cities):
    if len(cluster) <= 2:
        return cluster
    local = complete_qaco_logN([cities[i] for i in cluster], max_loops=40)
    return [cluster[i] for i in local[:-1]]
