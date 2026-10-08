import pickle
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train a RandomForestClassifier
model = RandomForestClassifier()
model.fit(X, y)

# Save the model using pickle
with open('iris_classifier.pkl', 'wb') as f:
    pickle.dump(model, f)
