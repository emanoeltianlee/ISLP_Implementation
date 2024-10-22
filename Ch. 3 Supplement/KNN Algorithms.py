import numpy as np

class BruteForceKNN:
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y

    def _calculate_distance(self, point1, point2):
        return np.sqrt(np.sum((point1 - point2) ** 2))

    def find_k_nearest_neighbors(self, query_point, k=5):
        distances = []

        for i, point in enumerate(self.X):
            dist = self._calculate_distance(query_point, point)
            distances.append((dist, i))

        distances.sort(key=lambda x: x[0])
        nearest_indices = [index for _, index in distances[:k]]
        return nearest_indices, self.y[nearest_indices]


class KDTreeNode:
    def __init__(self, point, label, left=None, right=None, axis=0):
        self.point = point
        self.label = label
        self.left = left
        self.right = right
        self.axis = axis


class KDTree_KNN:
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y
        self.kd_tree = self._build_tree(list(zip(X, y)), depth=0)

    def _build_tree(self, data, depth):
        if not data:
            return None

        axis = depth % len(self.X[0])

        data.sort(key=lambda x: x[0][axis])
        median = len(data) // 2

        return KDTreeNode(
            point=data[median][0],
            label=data[median][1],
            left=self._build_tree(data[:median], depth + 1),
            right=self._build_tree(data[median + 1:], depth + 1),
            axis=axis
        )

    def _calculate_distance(self, point1, point2):
        return np.sqrt(np.sum((point1 - point2) ** 2))

    def _search_k_nearest(self, node, query_point, k, best_nodes):
        if node is None:
            return

        # Calculate the distance from the query point to the current node
        dist = self._calculate_distance(query_point, node.point)
        best_nodes.append((dist, node))
        best_nodes.sort(key=lambda x: x[0])

        # Only keep the best k points
        if len(best_nodes) > k:
            best_nodes.pop()

        axis = node.axis
        next_branch = None
        opposite_branch = None

        if query_point[axis] < node.point[axis]:
            next_branch = node.left
            opposite_branch = node.right
        else:
            next_branch = node.right
            opposite_branch = node.left

        self._search_k_nearest(next_branch, query_point, k, best_nodes)

        if len(best_nodes) < k or abs(query_point[axis] - node.point[axis]) < best_nodes[-1][0]:
            self._search_k_nearest(opposite_branch, query_point, k, best_nodes)

    def find_k_nearest_neighbors(self, query_point, k=5):
        best_nodes = []
        self._search_k_nearest(self.kd_tree, query_point, k, best_nodes)
        nearest_indices = [self.X.tolist().index(node[1].point) for node in best_nodes]
        nearest_neighbors = [node[1].label for node in best_nodes]
        return nearest_indices, nearest_neighbors


class BallTree_Node:
    def __init__(self, points, labels, center=None, radius=None, left=None, right=None):
        self.points = points
        self.labels = labels
        self.center = center
        self.radius = radius
        self.left = left
        self.right = right


class BallTree_KNN:
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y
        self.ball_tree = self._build_tree(X, y)

    def _build_tree(self, X, y):
        if len(X) == 0:
            return None


        center = np.mean(X, axis=0)

        radius = np.max(np.linalg.norm(X - center, axis=1))

        if len(X) <= 1:
            return BallTreeNode(points=X, labels=y, center=center, radius=radius)

        distances = np.linalg.norm(X - center, axis=1)
        median_idx = np.argsort(distances)[len(distances) // 2]

        left_points = X[:median_idx]
        right_points = X[median_idx:]
        left_labels = y[:median_idx]
        right_labels = y[median_idx:]

        return BallTreeNode(
            points=None,
            labels=None,
            center=center,
            radius=radius,
            left=self._build_tree(left_points, left_labels),
            right=self._build_tree(right_points, right_labels)
        )

    def _calculate_distance(self, point1, point2):
        return np.linalg.norm(point1 - point2)

    def _search_k_nearest(self, node, query_point, k, best_nodes):
        if node is None:
            return

        dist_to_center = self._calculate_distance(query_point, node.center)

        if node.points is not None:
            for point, label in zip(node.points, node.labels):
                dist = self._calculate_distance(query_point, point)
                best_nodes.append((dist, label))
                best_nodes.sort(key=lambda x: x[0])

                if len(best_nodes) > k:
                    best_nodes.pop()
            return

        if dist_to_center - node.radius <= best_nodes[-1][0]:
            self._search_k_nearest(node.left, query_point, k, best_nodes)

        if dist_to_center + node.radius >= best_nodes[-1][0]:
            self._search_k_nearest(node.right, query_point, k, best_nodes)

    def find_k_nearest_neighbors(self, query_point, k=5):
        best_nodes = []
        self._search_k_nearest(self.ball_tree, query_point, k, best_nodes)
        return [n[1] for n in best_nodes]
