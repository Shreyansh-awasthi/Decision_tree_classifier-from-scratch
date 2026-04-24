# Decision_tree_classifier-from-scratch
Decision Tree classifier built from scratch  vs Sklearn - (compared on breast cancer dataset)
# 🌳 Decision Tree from Scratch

> Building a Decision Tree Classifier using pure NumPy and comparing it with Sklearn — no black boxes.

---

## 🧠 Why This Project?

Most people just call `DecisionTreeClassifier()` and move on.

This project builds the **entire Decision Tree algorithm from scratch** — every split, every node, every prediction — using only NumPy, then compares it directly against Sklearn's implementation.

---

## 🔢 Two Implementations Compared

### Level 1 — Sklearn DecisionTreeClassifier
The standard industry approach. Optimized, fast, production-ready.

### Level 2 — Decision Tree From Scratch (Pure NumPy)
Every single component built manually using Gini Impurity:

```python
# Gini Impurity — the core splitting formula
def gini(self, y):
    probs = np.bincount(y) / len(y)
    return 1 - np.sum(probs ** 2)
```

---

## ⚙️ How The Algorithm Works

```
1. Calculate Gini Impurity of current node
2. Try every feature and every threshold
3. Pick the split that minimizes weighted Gini
4. Recursively build left and right subtrees
5. Stop when max_depth reached or node is pure
6. Predict by traversing the tree from root to leaf
```

---

## 📊 Results

| Model | Accuracy | Precision | Recall |
|-------|----------|-----------|--------|
| Sklearn DecisionTree | 94.7% | 95.7% | 95.7% |
| From Scratch | 95.8% | 95.8% | 97.1% |

> The scratch implementation actually **outperformed** Sklearn on this dataset! 🎯

---

## 🔧 Custom Class Structure

```python
class Node:
    # Stores feature, threshold, left/right children, or leaf value

class DecisionTreeFromScratch:
    def gini(self, y)           # Calculate impurity
    def best_split(self, X, y)  # Find optimal split
    def build_tree(self, X, y)  # Recursively build tree
    def fit(self, X, y)         # Train the model
    def predict(self, X)        # Make predictions
```

---

## 💡 Key Concepts Covered

- What is Gini Impurity and how it measures node purity
- How recursive binary splitting works
- Why `max_depth` prevents overfitting
- Difference between a leaf node and a decision node
- Why sklearn is faster but both give similar results

---

## 🚀 How to Run

**Locally:**
```bash
git clone https://github.com/yourusername/decision-tree-from-scratch.git
cd decision-tree-from-scratch
pip install numpy pandas matplotlib scikit-learn seaborn
jupyter notebook
```

---

## 🗂️ What's Inside the Notebook

```
1. Load Dataset
2. Sklearn Decision Tree — baseline
3. Custom Standard Scaler — from scratch
4. Decision Tree class — from scratch
5. Train & evaluate both models
6. Confusion matrix & classification report
7. Accuracy comparison
```

---

## ⭐ Star This Repo

If this helped you understand Decision Trees better, consider starring — it helps others find it!
