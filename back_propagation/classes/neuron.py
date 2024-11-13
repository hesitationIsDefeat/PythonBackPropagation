import numpy as np

from back_propagation.enums import ActivationFunctionName


class Neuron:
    def __init__(self, gradient_starting_index = None, act_func_name = ActivationFunctionName.SIGMOID, connections = None):
        self._act_func_name = act_func_name
        self._bias = np.random.randn()
        self._connections = [] if connections is None else connections
        self._weights: np.array = [] if connections is None else np.random.randn(len(self._connections))
        self._parameter_size = 0 if connections is None else len(connections) + 1

        self._pre_activation_value = None
        self._activation_value = None

        self._loss_partial_derivative = None
        self._activation_partial_derivative = None
        self._pre_activation_partial_derivative_weight = None
        self._pre_activation_partial_derivative_bias = 1
        self._pre_activation_partial_derivative_activation = None

        self._gradient_starting_index = gradient_starting_index

    @property
    def pre_activation_value(self):
        return self._pre_activation_value

    @property
    def activation_value(self):
        return self._activation_value

    @property
    def gradient_starting_index(self):
        return self._gradient_starting_index

    @property
    def loss_partial_derivative(self):
        return self._loss_partial_derivative

    @property
    def activation_partial_derivative(self):
        return self._activation_partial_derivative

    @property
    def pre_activation_partial_derivative_weight(self):
        return self._pre_activation_partial_derivative_weight

    @property
    def pre_activation_partial_derivative_bias(self):
        return self._pre_activation_partial_derivative_bias

    @property
    def pre_activation_partial_derivative_activation(self):
        return self._pre_activation_partial_derivative_activation

    @property
    def parameter_size(self):
        return self._parameter_size

    def create_connections(self, connections: list['Neuron']):
        self._connections = connections
        self._weights = np.random.randn(len(connections))
        self._parameter_size = len(connections) + 1

    def set_input_neuron_activation(self, activation_value):
        self._activation_value = activation_value

    def update_params(self, gradient):
        self._weights -= gradient[:-1]
        self._bias -= gradient[-1]

    def act_func(self, activation_input):
        match self._act_func_name:
            case ActivationFunctionName.SIGMOID:
                return 1 / (1 + np.exp(-activation_input))
            case ActivationFunctionName.RELU:
                return np.maximum(0, activation_input)
            case ActivationFunctionName.TANH:
                return np.tanh(0, activation_input)
            case _:
                return 1 / (1 + np.exp(-activation_input))


    def act_func_derivative(self, activation_input):
        match self._act_func_name:
            case ActivationFunctionName.SIGMOID:
                return self.act_func(activation_input) * (1 - self.act_func(activation_input))
            case ActivationFunctionName.RELU:
                return np.where(activation_input > 0, 1, 0)
            case ActivationFunctionName.TANH:
                return 1 - np.tanh(activation_input) ** 2
            case _:
                return self.act_func(activation_input) * (1 - self.act_func(activation_input))

    def calculate_activation_value(self):
        connected_neuron_values = np.array([neuron.activation_value for neuron in self._connections])
        pre_activation_value = np.dot(connected_neuron_values, self._weights) + self._bias
        self._pre_activation_value = pre_activation_value
        self._activation_value =  self.act_func(pre_activation_value)

    def calculate_loss_partial_derivative_output_layer(self, loss_function_derivative_function, expected_result):
        self._loss_partial_derivative = loss_function_derivative_function(self._activation_value, expected_result)

    def calculate_loss_partial_derivative_hidden_layer(self, prev_layer, neuron_index):
        total = 0
        for neuron in prev_layer:
            total += neuron.loss_partial_derivative * neuron.activation_partial_derivative * neuron.get_pre_activation_partial_derivative_activation(neuron_index)
        self._loss_partial_derivative = total

    def calculate_activation_partial_derivative(self):
        self._activation_partial_derivative = self.act_func_derivative(self.pre_activation_value)

    def calculate_pre_activation_partial_derivative_weight(self, weight_index):
        self._pre_activation_partial_derivative_weight = self._connections[weight_index].activation_value

    def get_pre_activation_partial_derivative_activation(self, neuron_index):
        return self._weights[neuron_index]

    def get_gradient_vector(self):
        loss_times_activation = self.loss_partial_derivative * self.activation_partial_derivative
        vector = np.array([loss_times_activation * neuron.activation_value for neuron in self._connections])
        return np.append(vector, np.array([loss_times_activation * self.pre_activation_partial_derivative_bias]))