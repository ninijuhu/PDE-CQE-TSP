def solve_tsp_ga(problem, population_size=100, generations=500,
                 crossover_rate=0.8, mutation_rate=0.2):
    nodes = list(problem.get_nodes())

    def fitness(ind):
        coords = problem.node_coords or problem.display_data
        return sum(
            ((coords[ind[i]][0] - coords[ind[(i + 1) % len(ind)]][0]) ** 2
             + (coords[ind[i]][1] - coords[ind[(i + 1) % len(ind)]][1]) ** 2) ** 0.5
            for i in range(len(ind))
        )

    def crossover(parent1, parent2):
        size = len(parent1)
        a, b = sorted(random.sample(range(size), 2))
        child = [None] * size
        child[a:b] = parent1[a:b]
        for i in range(a, b):
            if parent2[i] not in child:
                pos = i
                while child[pos] is not None:
                    pos = parent2.index(parent1[pos])
                child[pos] = parent2[i]
        for i in range(size):
            if child[i] is None:
                child[i] = parent2[i]
        return child

    def mutate(ind):
        a, b = random.sample(range(len(ind)), 2)
        ind[a], ind[b] = ind[b], ind[a]
        return ind

    population = [random.sample(nodes, len(nodes)) for _ in range(population_size)]
    best_ind = min(population, key=fitness)
    for _ in range(generations):
        selected = []
        for _ in range(population_size):
            i, j = random.sample(range(population_size), 2)
            selected.append(population[i] if fitness(population[i]) < fitness(population[j]) else population[j])
        nxt = []
        while len(nxt) < population_size:
            p1, p2 = random.sample(selected, 2)
            if random.random() < crossover_rate:
                nxt.extend([crossover(p1, p2), crossover(p2, p1)])
            else:
                nxt.extend([p1[:], p2[:]])
        nxt = nxt[:population_size]
        for i in range(population_size):
            if random.random() < mutation_rate:
                nxt[i] = mutate(nxt[i])
        population = nxt
        current = min(population, key=fitness)
        if fitness(current) < fitness(best_ind):
            best_ind = current
    return best_ind
