import numpy as np
from collections import Counter
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data = load_wine()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def entropy(y):
    counts = np.bincount(y)
    probabilities = counts / len(y)
    return -np.sum([p * np.log2(p) for p in probabilities if p > 0])
def conditional_entropy(X_column, y):
    values = np.unique(X_column)
    weighted_entropy = 0
    
    for value in values:
        sub_y = y[X_column == value]
        weighted_entropy += (len(sub_y) / len(y)) * entropy(sub_y)
    
    return weighted_entropy
def gini(y):
    counts = np.bincount(y)
    probabilities = counts / len(y)
    return 1 - np.sum([p**2 for p in probabilities])
class DecisionTree:
    def __init__(self, criterion='entropy', max_depth=None):
        self.criterion = criterion
        self.max_depth = max_depth
        self.tree = None
    def fit(self, X, y):
        self.tree = self._grow_tree(X, y)
    def _best_split(self, X, y):
        best_gain = -1
        split_idx, split_thresh = None, None
        
        for col in range(X.shape[1]):
            X_column = X[:, col]
            thresholds = np.unique(X_column)
            
            for threshold in thresholds:
                left_mask = X_column <= threshold
                right_mask = X_column > threshold
                
                if sum(left_mask) == 0 or sum(right_mask) == 0:
                    continue
                
                y_left, y_right = y[left_mask], y[right_mask]
                
                if self.criterion == 'entropy':
                    gain = self._information_gain(y, y_left, y_right, entropy)
                else: 
                    gain = self._information_gain(y, y_left, y_right, gini)
                
                if gain > best_gain:
                    best_gain = gain
                    split_idx = col
                    split_thresh = threshold
                    
        return split_idx, split_thresh
    def _information_gain(self, y, y_left, y_right, criterion):
        parent_loss = criterion(y)
        num_left, num_right = len(y_left), len(y_right)
        num_total = num_left + num_right
        
        weighted_loss = (num_left / num_total) * criterion(y_left) + (num_right / num_total) * criterion(y_right)
        
        return parent_loss - weighted_loss
    def _grow_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        num_labels = len(np.unique(y))
        
        if num_labels == 1 or num_samples <= 2 or (self.max_depth and depth >= self.max_depth):
            leaf_value = self._most_common_label(y)
            return {'leaf': leaf_value}
        
        split_idx, split_thresh = self._best_split(X, y)
        
        if split_idx is None:
            leaf_value = self._most_common_label(y)
            return {'leaf': leaf_value}
        
        left_mask = X[:, split_idx] <= split_thresh
        right_mask = X[:, split_idx] > split_thresh
        left_tree = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self._grow_tree(X[right_mask], y[right_mask], depth + 1)
        
        return {'split_idx': split_idx, 'split_thresh': split_thresh, 'left': left_tree, 'right': right_tree}
    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X])
    def _traverse_tree(self, x, node):
        if 'leaf' in node:
            return node['leaf']
        
        if x[node['split_idx']] <= node['split_thresh']:
            return self._traverse_tree(x, node['left'])
        else:
            return self._traverse_tree(x, node['right'])

tree_entropy = DecisionTree(criterion='entropy', max_depth=5)
tree_entropy.fit(X_train, y_train)
y_pred_custom_entropy = tree_entropy.predict(X_test)
tree_gini = DecisionTree(criterion='gini', max_depth=5)
tree_gini.fit(X_train, y_train)
y_pred_custom_gini = tree_gini.predict(X_test)
clf_entropy = DecisionTreeClassifier(criterion='entropy', max_depth=5)
clf_entropy.fit(X_train, y_train)
y_pred_sklearn_entropy = clf_entropy.predict(X_test)
clf_gini = DecisionTreeClassifier(criterion='gini', max_depth=5)
clf_gini.fit(X_train, y_train)
y_pred_sklearn_gini = clf_gini.predict(X_test)

print("Custom Decision Tree (Entropy) Accuracy:", accuracy_score(y_test, y_pred_custom_entropy))
print("Custom Decision Tree (Gini) Accuracy:", accuracy_score(y_test, y_pred_custom_gini))
print("Scikit-learn Decision Tree (Entropy) Accuracy:", accuracy_score(y_test, y_pred_sklearn_entropy))
print("Scikit-learn Decision Tree (Gini) Accuracy:", accuracy_score(y_test, y_pred_sklearn_gini))
