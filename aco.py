class ACO:
    def __init__(self, cities, num_ants=30, alpha=1.0, beta=2.0,
                 rho=0.1, Q=100, max_iterations=100):
        self.num_cities = len(cities)
        self.num_ants = num_ants
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.max_iterations = max_iterations
        self.distances = np.zeros((self.num_cities, self.num_cities))
        for i in range(self.num_cities):
            for j in range(self.num_cities):
                if i != j:
                    x1, y1 = cities[i][1], cities[i][2]
                    x2, y2 = cities[j][1], cities[j][2]
                    self.distances[i, j] = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        self.pheromones = np.ones((self.num_cities, self.num_cities)) * 0.1
        self.best_path = []
        self.best_distance = float("inf")

    def _construct_path(self):
        current = random.randrange(self.num_cities)
        path = [current]
        unvisited = [i for i in range(self.num_cities) if i != current]
        while unvisited:
            weights = []
            for city in unvisited:
                tau = self.pheromones[current][city] ** self.alpha
                eta = (1 / self.distances[current][city]) ** self.beta
                weights.append(tau * eta)
            total = sum(weights)
            if total == 0:
                nxt = random.choice(unvisited)
            else:
                draw = random.random()
                acc = 0.0
                nxt = unvisited[0]
                for city, weight in zip(unvisited, weights):
                    acc += weight / total
                    if draw <= acc:
                        nxt = city
                        break
            path.append(nxt)
            unvisited.remove(nxt)
            current = nxt
        distance = sum(self.distances[path[i]][path[(i + 1) % self.num_cities]]
                       for i in range(len(path)))
        return path, distance

    def solve(self):
        for _ in range(self.max_iterations):
            paths = [self._construct_path() for _ in range(self.num_ants)]
            self.pheromones *= (1 - self.rho)
            for path, distance in paths:
                if distance < self.best_distance:
                    self.best_distance = distance
                    self.best_path = path[:]
                delta = self.Q / distance
                for i in range(len(path)):
                    a, b = path[i], path[(i + 1) % len(path)]
                    self.pheromones[a][b] += delta
                    self.pheromones[b][a] += delta
        return self.best_path, self.best_distance
