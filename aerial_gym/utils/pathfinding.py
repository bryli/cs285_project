import numpy as np

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0.0

def rrt_star_path_length(obstacles, start=np.array([5, 5]), goal=np.array([0, 0]), max_iter=10000, step_size=0.1, min_radius=0.5):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # start = start.to(device)
    # goal = goal.to(device)
    obstacles = obstacles.to("cpu").detach().numpy()


    nodes = [Node(start[0], start[1])]
    for _ in range(max_iter):
        sample_node = Node(np.random.uniform(0, 10), np.random.uniform(0, 10))  # Adjust the range as needed

        # Find the nearest node in the tree
        nearest_node = min(nodes, key=lambda n: np.hypot(n.x - sample_node.x, n.y - sample_node.y))

        # Generate a new node in the direction of the sample
        angle = np.arctan2(sample_node.y - nearest_node.y, sample_node.x - nearest_node.x)
        new_x = nearest_node.x + step_size * np.cos(angle)
        new_y = nearest_node.y + step_size * np.sin(angle)

        # Check for collisions within the specified radius
        collision = False
        for obstacle in obstacles:
            distance = np.hypot(new_x - obstacle[0], new_y - obstacle[1])
            if distance < min_radius:
                collision = True
                break

        if not collision:
            new_node = Node(new_x.item(), new_y.item())
            new_node.parent = nearest_node
            new_node.cost = nearest_node.cost + step_size

            # Update the cost of nearby nodes if a lower-cost path is found
            for node in nodes:
                if np.hypot(node.x - new_node.x, node.y - new_node.y) < min_radius and node.cost > new_node.cost:
                    node.parent = new_node
                    node.cost = new_node.cost

            nodes.append(new_node)

            # Check if the goal is reached
            if np.hypot(new_x - goal[0], new_y - goal[1]) < min_radius:
                goal_node = Node(goal[0].item(), goal[1].item())
                goal_node.parent = new_node
                goal_node.cost = new_node.cost + np.hypot(new_x - goal[0], new_y - goal[1])
                nodes.append(goal_node)

                # Reconstruct the path
                path = []
                while goal_node is not None:
                    path.append((goal_node.x, goal_node.y))
                    goal_node = goal_node.parent

                return path[::-1], nodes[-1].cost

    return None, -1  # No path found