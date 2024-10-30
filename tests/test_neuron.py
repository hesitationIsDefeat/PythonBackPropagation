import unittest
import numpy as np
from numpy.testing import assert_allclose
from back_propagation.classes.neuron import Neuron

is_skipping_single_neuron_connections = True
is_skipping_neuron_parameters_update = False

INPUT_NEURON_ACTIVATION = 5

@unittest.skipIf(is_skipping_single_neuron_connections, "Skipping TestSingleNeuronConnections")
class TestSingleNeuronConnections(unittest.TestCase):
    def setUp(self):
        self.single_input_neuron = Neuron()
        self.single_input_neuron.set_input_neuron_activation(INPUT_NEURON_ACTIVATION)

        self.single_hidden_neuron = Neuron(connections=[self.single_input_neuron])
        self.single_hidden_neuron.calculate_activation_value()

        self.single_output_neuron = Neuron(connections=[self.single_hidden_neuron])
        self.single_output_neuron.calculate_activation_value()


    def test_neuron_gradient_starting_index(self):
        self.assertEqual(self.single_input_neuron.gradient_starting_index, None)
        self.assertEqual(self.single_hidden_neuron.gradient_starting_index, 0)
        self.assertEqual(self.single_output_neuron.gradient_starting_index, 2)

    def test_single_neuron_pre_activation(self):
        hidden_neuron_pre_activation = INPUT_NEURON_ACTIVATION * self.single_hidden_neuron._weights[0] + self.single_hidden_neuron._bias
        output_neuron_pre_activation = self.single_hidden_neuron.activation_value * self.single_output_neuron._weights[0] + self.single_output_neuron._bias

        self.assertEqual(self.single_hidden_neuron.pre_activation_value, hidden_neuron_pre_activation)
        self.assertEqual(self.single_output_neuron.pre_activation_value, output_neuron_pre_activation)

    def test_single_neuron_activation(self):
        hidden_neuron_activation = self.single_hidden_neuron.act_func(INPUT_NEURON_ACTIVATION * self.single_hidden_neuron._weights[
            0] + self.single_hidden_neuron._bias)
        output_neuron_activation = self.single_output_neuron.act_func(self.single_hidden_neuron.activation_value * self.single_output_neuron._weights[
            0] + self.single_output_neuron._bias)

        self.assertEqual(self.single_input_neuron.activation_value, INPUT_NEURON_ACTIVATION)
        self.assertEqual(self.single_hidden_neuron.activation_value, hidden_neuron_activation)
        self.assertEqual(self.single_output_neuron.activation_value, output_neuron_activation)


INPUT_LAYER_SIZE = 3
FIRST_HIDDEN_LAYER_SIZE = 5
SECOND_HIDDEN_LAYER_SIZE = 7
OUTPUT_LAYER_SIZE = 5

INPUT_NEURON_ACTIVATIONS = np.random.rand(INPUT_LAYER_SIZE)

GRADIENT_DESCEND_VECTOR_PARTS = [np.random.rand(INPUT_LAYER_SIZE + 1),
                                 np.random.rand(FIRST_HIDDEN_LAYER_SIZE + 1),
                                 np.random.rand(SECOND_HIDDEN_LAYER_SIZE + 1)]

@unittest.skipIf(is_skipping_neuron_parameters_update, "Skipping TestNeuronParameterUpdate")
class TestNeuronParameterUpdate(unittest.TestCase):
    def setUp(self):
        self.input_layer = [Neuron() for _ in range(INPUT_LAYER_SIZE)]
        for index in range(INPUT_LAYER_SIZE):
            self.input_layer[index].set_input_neuron_activation(INPUT_NEURON_ACTIVATIONS[index])

        self.first_hidden_layer = [Neuron(connections=self.input_layer) for _ in range(FIRST_HIDDEN_LAYER_SIZE)]
        for neuron in self.first_hidden_layer:
            neuron.calculate_activation_value()

        self.second_hidden_layer = [Neuron(connections=self.first_hidden_layer) for _ in range(SECOND_HIDDEN_LAYER_SIZE)]
        for neuron in self.second_hidden_layer:
            neuron.calculate_activation_value()

        self.output_layer = [Neuron(connections=self.second_hidden_layer) for _ in range(OUTPUT_LAYER_SIZE)]
        for neuron in self.output_layer:
            neuron.calculate_activation_value()

    def assertParamsEqual(self, single_neuron, gradient_descend_vector_part_index):
        gradient_descend_vector_part = GRADIENT_DESCEND_VECTOR_PARTS[gradient_descend_vector_part_index]
        updated_weights = single_neuron._weights - gradient_descend_vector_part[:-1]
        updated_bias = single_neuron._bias - gradient_descend_vector_part[-1]

        single_neuron.update_params(gradient_descend_vector_part)

        assert_allclose(single_neuron._weights, updated_weights)
        self.assertEqual(single_neuron._bias, updated_bias)

    @unittest.skip
    def test_single_hidden_neuron_parameter_update(self):
        self.assertParamsEqual(self.first_hidden_layer[0], 0)

    @unittest.skip
    def test_first_hidden_layer_parameter_update(self):
        for single_hidden_neuron in self.first_hidden_layer:
            self.assertParamsEqual(single_hidden_neuron, 0)

    def test_all_layers_parameter_update(self):
        for single_hidden_neuron in self.first_hidden_layer:
            self.assertParamsEqual(single_hidden_neuron, 0)

        for single_hidden_neuron in self.second_hidden_layer:
            self.assertParamsEqual(single_hidden_neuron, 1)

        for single_output_neuron in self.output_layer:
            self.assertParamsEqual(single_output_neuron, 2)



