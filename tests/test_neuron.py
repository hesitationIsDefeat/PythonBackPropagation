import unittest
import numpy as np
from numpy.ma.testutils import assert_equal

from back_propagation.classes.neuron import Neuron

INPUT_NEURON_ACTIVATION = 5

class TestSingleNeuron(unittest.TestCase):
    def setUp(self):
        self.single_input_neuron = Neuron()
        self.single_input_neuron.set_input_neuron_activation(INPUT_NEURON_ACTIVATION)

        self.single_hidden_neuron = Neuron(gradient_starting_index=0, connections=[self.single_input_neuron])
        self.single_hidden_neuron.calculate_activation_value()

        self.single_output_neuron = Neuron(gradient_starting_index=2, connections=[self.single_hidden_neuron])
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

