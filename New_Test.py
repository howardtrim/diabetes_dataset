import streamlit as st
import numpy as np
import sqlite3
import pandas as pd
from datetime import date
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree, _tree
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split




def extract_leaf_rules(clf, feature_names, class_names=None, decimals=3):
    """
    Returns a DataFrame where each row is a leaf:
    - rule: human-readable path conditions
    - n_samples: samples in leaf (weighted if sample_weight used in fit)
    - per-class counts and proportions
    """
    tree = clf.tree_
    feature = tree.feature
    threshold = tree.threshold

    # If class_names not provided, use indices
    if class_names is None:
        class_names = [str(i) for i in range(tree.value.shape[1])]

    paths = []

    def recurse(node, conditions):
        is_split = feature[node] != _tree.TREE_UNDEFINED

        if is_split:
            name = feature_names[feature[node]]
            thr = threshold[node]

            # left child: <= threshold
            recurse(tree.children_left[node],
                    conditions + [(name, "<=", thr)])

            # right child: > threshold
            recurse(tree.children_right[node],
                    conditions + [(name, ">", thr)])
        else:
            # leaf
            counts = tree.value[node][0]  # shape (n_classes,)
            n = counts.sum()
            probs = counts / n if n > 0 else np.zeros_like(counts)

            rule = " AND ".join([f"{f} {op} {t:.{decimals}f}" for f, op, t in conditions]) \
                   if conditions else "(root)"

            row = {
                "leaf_id": node,
                "rule": rule,
                "n_samples": float(n),
            }
            for i, c in enumerate(class_names):
                row[f"count_{c}"] = float(counts[i])
                row[f"prop_{c}"] = float(probs[i])
            paths.append(row)

    recurse(0, [])
    df = pd.DataFrame(paths).sort_values("n_samples", ascending=False).reset_index(drop=True)
    return df

st.title("Decision Tree Classifier")

df = pd.read_csv("diabetes.csv")

y_true = df.Outcome.copy()
X = df.drop(columns=["Outcome"]).copy()

st.dataframe(df)
col1, col2 = st.columns([0.5,0.5 ])
fig1, ax = plt.subplots(figsize=(12, 8))
plt.plot(X.Glucose, y_true, 'bo')
plt.axvline(x=127.5, color='red', label='127.5')
plt.annotate(
    "Important event",
    xy=(100,.5),
   # text position
    arrowprops=dict(arrowstyle="->"),
    fontsize=14
)
plt.legend(fontsize=14)
col1.pyplot(fig1)
col1.write("Glucose")
fig2, ax = plt.subplots(figsize=(12, 8))
plt.plot(X.BMI, y_true, 'bo')
col2.pyplot(fig2)
col2.write("BMI")
fig1, ax = plt.subplots(figsize=(12, 8))
plt.plot(X.DiabetesPedigreeFunction, y_true, 'bo')
col1.pyplot(fig1)
col1.write("DiabetesPedigreeFunction")
fig2, ax = plt.subplots(figsize=(12, 8))
plt.plot(X.Age, y_true, 'bo')
col2.pyplot(fig2)
col2.write("Age")

feature_names = list(X.columns)
# st.write(feature_names)

max_depth = st.slider("Depth of Tree", min_value=0,max_value=10,value=3)

clf = DecisionTreeClassifier(max_depth=max_depth)
clf = clf.fit(X, y_true)
y_pred = pd.DataFrame(clf.predict(X))
y_true = pd.DataFrame(y_true)


accuracy = accuracy_score(y_true, y_pred)
percent_correct = accuracy * 100
st.write(f"Percent Correct: {percent_correct:.2f}%")

fig, ax = plt.subplots(figsize=(12, 8))
plot_tree(clf, filled=True)
st.pyplot(fig)

st.write(extract_leaf_rules(clf, feature_names))
col1, col2 = st.columns([0.5,0.5 ])
col1.write(clf.tree_.feature)
col2.write(clf.tree_.threshold)

X_train, X_val, y_train, y_val = train_test_split(X, y_true, random_state=0)

model = clf
st.write(model.score(X_val, y_val))

from sklearn.inspection import permutation_importance
r = permutation_importance(clf, X, y_true,
                           n_repeats=30,
                           random_state=0)

for i in r.importances_mean.argsort()[::-1]:
    if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
        st.write(f"{feature_names[i]:<8}"
              f"{r.importances_mean[i]:.3f}"
              f" +/- {r.importances_std[i]:.3f}")

