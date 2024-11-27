from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from back_propagation.classes.deep_neural_network import DeepNeuralNetwork
from back_propagation.enums import ActivationFunctionName, LossFunctionName

from utils.data_manipulation import to_onehot

# Load the dataset
digits = load_digits()
X, y = digits.data, digits.target

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_train_oneshot = to_onehot(y_train, 10)

dnn = DeepNeuralNetwork(hidden_layer_info=[16, 16],
                        training_data_x=X_train, training_data_y=y_train_oneshot,
                        back_propagation_learning_rate=0.01, back_propagation_aimed_loss_value=0.01, back_propagation_iteration_limit = 2000,
                        activation_function_name=ActivationFunctionName.SIGMOID, loss_func_name=LossFunctionName.DIFFERENCE_SQUARE)
dnn.start_back_propagation()
