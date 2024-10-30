from typing import List

import numpy as np

from back_propagation.classes.neuron import Neuron
from back_propagation.enums import ActivationFunctionName, LossFunctionName


class DeepNeuralNetwork:
    def __init__(self, hidden_layer_info, training_data_x, training_data_y,
                 activation_function_name=ActivationFunctionName.SIGMOID,
                 loss_func_name=LossFunctionName.DIFFERENCE_SQUARE):
        self._training_data_x = training_data_x
        self._training_data_y = training_data_y

        input_layer_size = len(training_data_x[0])
        output_layer_size = len(training_data_y[0])

        gradient_index = 0

        # Create input layers
        self._input_layer: List[Neuron] = [Neuron(act_func_name=activation_function_name) for _ in range(input_layer_size)]

        # Create hidden layers
        self._hidden_layers: List[List[Neuron]] = [[Neuron(gradient_starting_index= (gradient_index := gradient_index + (i * (input_layer_size + 1))),
                                                act_func_name=activation_function_name) for i in hidden_layer_info[0]]]

        for layer_index in range(len(hidden_layer_info[1:])):
            hidden_layer: List[Neuron] = [Neuron(gradient_starting_index= (gradient_index := gradient_index + (i * (hidden_layer_info[layer_index - 1] + 1))),
                                            act_func_name=activation_function_name) for i in range(hidden_layer_info[layer_index])]
            self._hidden_layers.append(hidden_layer)

        # Connect input layer with the first hidden layer
        for neuron in self._hidden_layers[0]:
            neuron.create_connections(self._input_layer)

        # Connect hidden layers with each other
        for layer_index in range(len(self._hidden_layers[1:])):
            prev_layer = self._hidden_layers[layer_index - 1]
            for neuron in self._hidden_layers[layer_index]:
                neuron.create_connections(prev_layer)

        # Create output layer
        self._output_layer: List[Neuron] = [Neuron(gradient_starting_index= (gradient_index := gradient_index + (i * (hidden_layer_info[-1] + 1))),
                                              act_func_name=activation_function_name) for i in range(output_layer_size)]
        # Connect output layer with the last hidden layer
        for neuron in self._output_layer:
            neuron.create_connections(self._hidden_layers[-1])

        self._loss_func_name = loss_func_name

        self._loss_value = None

        # Create an array of 0's to later keep track of the weight and bias values
        gradient_size = input_layer_size * hidden_layer_info[0] + hidden_layer_info[-1] * output_layer_size + input_layer_size + sum(hidden_layer_info) + output_layer_size
        for neuron_layer_index in range(1, len(hidden_layer_info)):
            gradient_size += hidden_layer_info[neuron_layer_index] * hidden_layer_info[neuron_layer_index - 1]
        self._gradient_size = gradient_size

        self._gradient_descend = np.zeros(self._gradient_size)

    # Calculate the activation value for each neuron
    def calculate_activation_values(self):
        for layer in self._hidden_layers:
            for neuron in layer:
                neuron.calculate_activation_value()

        for neuron in self._output_layer:
            neuron.calculate_activation_value()


    def loss_func(self, neuron_value, expected_value):
        match self._loss_func_name:
            case LossFunctionName.DIFFERENCE_SQUARE:
                return np.square(neuron_value - expected_value)



    def calculate_loss_value(self, expected_values):
        loss_sum = 0.0
        for neuron, expected_value in zip(self._output_layer, expected_values):
            loss_sum += self.loss_func(neuron.activation_value, expected_value)
        return loss_sum

    # Adjusts the gradient vector before updating the parameters of the neurons
    def gradient_adjust_func(self):
        learning_rate = 0.001
        self._gradient_descend /= learning_rate

    # Assigns the neuron with the related portion of the gradient descend vector
    def update_neuron_parameters(self, neuron):
        starting_index = neuron.gradient_starting_index
        gradient_descend_vector = self._gradient_descend[starting_index, starting_index + neuron.parameter_size]
        neuron.update_params(gradient_descend_vector)

    # Starting from the output layer and going back, calculate the gradient descend vector for each layer and add the layer vector to the start of the main vector
    def calculate_gradient_descend(self):
        gradient_descend_vector = np.array([])

        for neuron in reversed(self._output_layer) :
            gradient_descend_vector = np.append(gradient_descend_vector, neuron.get_gradient_vector())

        for hidden_layer in self._hidden_layers:
            for neuron in reversed(hidden_layer):
                gradient_descend_vector = np.append(gradient_descend_vector, neuron.get_gradient_vector())

        self._gradient_descend += gradient_descend_vector

    # After the gradient descend is obtained, updates the parameters of every neuron
    def update_every_neuron_parameters(self):
        for hidden_layer in self._hidden_layers:
            for neuron in hidden_layer:
                self.update_neuron_parameters(neuron)

        for neuron in self._output_layer:
            self.update_neuron_parameters(neuron)



    # Set the input data as the activation value of the input neurons
    def set_input_data(self, input_values):
        for neuron, input_value in zip(self._input_layer, input_values):
            neuron.set_input_neuron_activation(input_value)

    # Assuming the connections are made
    def start_back_propagation(self):
        aimed_loss_value = 1/100
        iteration_limit = 100
        iteration_index = 0
        while self._loss_value is None or self._loss_value > aimed_loss_value:
            if iteration_index >= iteration_limit:
                print(f"Reached iteration limit of {iteration_limit}, current loss value is {self._loss_value}, aimed loss value was {aimed_loss_value}")
            print(f"Starting iteration {iteration_index}")
            for training_data, expected_result in zip(self._training_data_x, self._training_data_y):
                self.set_input_data(training_data)
                self.calculate_activation_values()
                self._loss_value += self.calculate_loss_value(expected_result)

            self._loss_value /= len(self._training_data_x)
            print(f"Current loss value is {self._loss_value}")
            self.calculate_gradient_descend()
            self.update_every_neuron_parameters()





