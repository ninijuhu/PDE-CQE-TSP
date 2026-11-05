def get_distance(problem, node1, node2):
    nodes = problem.node_coords or problem.display_data
    return ((nodes[node1][0] - nodes[node2][0]) ** 2
            + (nodes[node1][1] - nodes[node2][1]) ** 2) ** 0.5


def solve_tsp(problem, start_node=1):
    unvisited = set(problem.get_nodes())
    current_node = start_node
    tour = [current_node]
    unvisited.remove(current_node)
    while unvisited:
        next_node = min(unvisited, key=lambda node: get_distance(problem, current_node, node))
        tour.append(next_node)
        unvisited.remove(next_node)
        current_node = next_node
    return tour


def calculate_tour_length(problem, tour):
    total_length = 0.0
    for i in range(len(tour)):
        total_length += get_distance(problem, tour[i], tour[(i + 1) % len(tour)])
    return total_length
