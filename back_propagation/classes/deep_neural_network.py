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

        gradient_index = -(input_layer_size + 1)

        # Create input layers
        self._input_layer: List[Neuron] = [Neuron(act_func_name=activation_function_name) for _ in range(input_layer_size)]

        # Create hidden layers
        self._hidden_layers: List[List[Neuron]] = [[Neuron(gradient_starting_index= (gradient_index := gradient_index + (input_layer_size + 1)),
                                                act_func_name=activation_function_name) for i in range(hidden_layer_info[0])]]

        gradient_index += (input_layer_size + 1) - (hidden_layer_info[0] + 1)
        for layer_index in range(len(hidden_layer_info[1:])):
            hidden_layer: List[Neuron] = [Neuron(gradient_starting_index= (gradient_index := gradient_index + (hidden_layer_info[layer_index - 1] + 1)),
                                            act_func_name=activation_function_name) for i in range(hidden_layer_info[layer_index])]
            self._hidden_layers.append(hidden_layer)




        # print(f"Input layer contains {len(self._input_layer)} neurons")

        # Connect input layer with the first hidden layer
        for neuron in self._hidden_layers[0]:
            neuron.create_connections(self._input_layer)

        # Connect hidden layers with each other
        for layer_index in range(1, len(self._hidden_layers)):
            prev_layer = self._hidden_layers[layer_index - 1]
            for neuron in self._hidden_layers[layer_index]:
                neuron.create_connections(prev_layer)

        # for layer in self._hidden_layers:
        #     print(f"Hidden layer contains {len(layer)} neurons where each contains {len(layer[0]._connections)} many connections")

        # Create output layer
        self._output_layer: List[Neuron] = [Neuron(gradient_starting_index= (gradient_index := gradient_index + ((hidden_layer_info[-1] + 1))),
                                              act_func_name=activation_function_name, connections=self._hidden_layers[-1]) for i in range(output_layer_size)]
        # print(f"Output layer contains {len(self._output_layer)} neurons where each contains {len(self._output_layer[0]._connections)} many connections")
        # Connect output layer with the last hidden layer
        # for neuron in self._output_layer:
        #     print(f"Weights of the output neuron are {neuron._weights}")

        # for layer in self._hidden_layers:
        #     print("Hidden layer")
        #     for neuron in layer:
        #
        #         print(f"{neuron.gradient_starting_index}-{neuron.gradient_starting_index + neuron.parameter_size}")
        # print("Output layer")
        # for neuron in self._output_layer:
        #
        #     print(f"{neuron.gradient_starting_index}-{neuron.gradient_starting_index + neuron.parameter_size}")

        self._loss_func_name = loss_func_name

        self._loss_value = None

        # Create an array of 0's to later keep track of the weight and bias values

        self._gradient_descend = None

    # Calculate the activation value for each neuron
    def forward_pass(self):
        for layer in self._hidden_layers:
            for neuron in layer:
                neuron.calculate_activation_value()

        for neuron in self._output_layer:
            neuron.calculate_activation_value()

    def backward_pass(self, expected_result):
        for neuron, expected_result in zip(self._output_layer, expected_result):
            # print("Starting backward pass for output neuron")
            neuron.calculate_activation_partial_derivative()
            neuron.calculate_loss_partial_derivative_output_layer(loss_function_derivative_function=self.loss_func_derivative, expected_result=expected_result)

        hidden_layer_size = len(self._hidden_layers)
        for layer_index in range(hidden_layer_size):
            # print(f"Starting backward pass for hidden layer indexed {hidden_layer_size - layer_index - 1}")
            layer = self._hidden_layers[hidden_layer_size - layer_index - 1]
            for neuron_index in range(len(layer)):
                # print(f"Starting backward pass for hidden neuron indexed {neuron_index} in layer indexed {hidden_layer_size - layer_index - 1}")
                neuron = layer[neuron_index]
                neuron.calculate_activation_partial_derivative()
                prev_layer = self._hidden_layers[hidden_layer_size - layer_index] if layer_index != 0 else self._output_layer
                # if layer_index != 0:
                #     print(f"Prev layer for neuron indexed {neuron_index} is the layer indexed {hidden_layer_size - layer_index}")
                # else:
                #     print(f"Prev layer for neuron indexed {neuron_index} is the output layer")
                neuron.calculate_loss_partial_derivative_hidden_layer(prev_layer=prev_layer, neuron_index=neuron_index)


    def loss_func(self, neuron_value, expected_value):
        match self._loss_func_name:
            case LossFunctionName.DIFFERENCE_SQUARE:
                return np.square(neuron_value - expected_value)/2


    def loss_func_derivative(self, neuron_value, expected_value):
        match self._loss_func_name:
            case LossFunctionName.DIFFERENCE_SQUARE:
                return neuron_value - expected_value



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
        # print(f"Gradient descend vector is between the indices {starting_index} and {starting_index + neuron.parameter_size}")
        gradient_descend_vector = self._gradient_descend[starting_index: starting_index + neuron.parameter_size]
        neuron.update_params(gradient_descend_vector)

    # Starting from the output layer and going back, calculate the gradient descend vector for each layer and add the layer vector to the start of the main vector
    def calculate_gradient_descend(self):
        # print(f"Starting gradient descend vector calculation")
        gradient_descend_vector = np.array([])

        for neuron in reversed(self._output_layer) :

            gradient_descend_vector = np.append(gradient_descend_vector, neuron.get_gradient_vector())

        for hidden_layer in reversed(self._hidden_layers):
            for neuron in reversed(hidden_layer):
                gradient_descend_vector = np.append(gradient_descend_vector, neuron.get_gradient_vector())

        # print(f"Calculated gradient vector size: {gradient_descend_vector.size}")
        if self._gradient_descend is None:
            self._gradient_descend = gradient_descend_vector
        else:
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
        iteration_limit = 1000
        iteration_index = 0
        while self._loss_value is None or self._loss_value > aimed_loss_value:
            if iteration_index >= iteration_limit:
                print(f"Reached iteration limit of {iteration_limit}, current loss value is {self._loss_value}, aimed loss value was {aimed_loss_value}")
                break
            print(f"Starting iteration {iteration_index}")
            for training_data, expected_result in zip(self._training_data_x, self._training_data_y):
                # print(f"Training data: {training_data}")
                self.set_input_data(training_data)
                # for neuron in self._input_layer:
                #     print(neuron.activation_value)
                self.forward_pass()
                if self._loss_value is None:
                    self._loss_value = self.calculate_loss_value(expected_result)
                else:
                    self._loss_value += self.calculate_loss_value(expected_result)

                self.backward_pass(expected_result)
                self.calculate_gradient_descend()

            self._loss_value /= len(self._training_data_x)
            self._gradient_descend /= len(self._training_data_x)
            print(f"Current loss value is {self._loss_value}")

            self.update_every_neuron_parameters()
            iteration_index += 1





