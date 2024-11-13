from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from back_propagation.classes.deep_neural_network import DeepNeuralNetwork

from utils.data_manipulation import to_onehot

# Load the dataset
digits = load_digits()
X, y = digits.data, digits.target

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_train_oneshot = to_onehot(y_train, 10)

dnn = DeepNeuralNetwork(hidden_layer_info=[16, 16], training_data_x=X_train, training_data_y=y_train_oneshot)
dnn.start_back_propagation()
