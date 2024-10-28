import numpy as np

class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.leaf_count = 1 if self.value is not None else 0

class DecisionTreeRegressor:
    def __init__(self, max_depth=5, min_samples_split=10):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        if n_samples < self.min_samples_split or depth >= self.max_depth:
            return TreeNode(value=np.mean(y))

        best_feature, best_threshold, min_rss = None, None, float('inf')

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices = X[:, feature] <= threshold
                right_indices = X[:, feature] > threshold

                if sum(left_indices) == 0 or sum(right_indices) == 0:
                    continue

                left_rss = self._calculate_rss(y[left_indices])
                right_rss = self._calculate_rss(y[right_indices])
                rss = left_rss + right_rss

                if rss < min_rss:
                    min_rss = rss
                    best_feature = feature
                    best_threshold = threshold

        if best_feature is None:
            return TreeNode(value=np.mean(y))

        left_indices = X[:, best_feature] <= best_threshold
        right_indices = X[:, best_feature] > best_threshold
        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        node = TreeNode(feature=best_feature, threshold=best_threshold,
                        left=left_subtree, right=right_subtree)

        node.leaf_count = left_subtree.leaf_count + right_subtree.leaf_count
        return node

    def _calculate_rss(self, y):
        mean_y = np.mean(y)
        return np.sum((y - mean_y) ** 2)

    def predict(self, X):
        return np.array([self._predict_sample(x, self.root) for x in X])

    def _predict_sample(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)

    def cost_complexity_pruning(self, alpha=0.01):
        self._prune_tree(self.root, alpha)

    def _prune_tree(self, node, alpha):
        if node.left is None and node.right is None:
            return

        if node.left:
            self._prune_tree(node.left, alpha)
        if node.right:
            self._prune_tree(node.right, alpha)

        if node.left and node.left.value is not None and node.right and node.right.value is not None:
            leaf_cost = node.leaf_count * alpha
            subtree_rss = self._calculate_rss_leaves(node)

            if leaf_cost + subtree_rss >= self._calculate_rss_leaves(node):
                node.left, node.right = None, None
                node.value = np.mean([node.left.value, node.right.value])
                node.leaf_count = 1

    def _calculate_rss_leaves(self, node):
        if node.value is not None:
            return 0
        left_rss = self._calculate_rss_leaves(node.left)
        right_rss = self._calculate_rss_leaves(node.right)
        return left_rss + right_rss