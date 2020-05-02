from . import network
import unittest
import numpy as np

class Test(unittest.TestCase):
    def test_forward_3_2_2(self):
        n_input = 3
        n_output = 2
        net = network.Network(n_input, n_output)
        net.add_layer(2)
        weights = [
            0,
            np.array([[0.1, 0.8, 0.2], [0.3, 0.1, 0.5]]),
            np.array([[0.5, 0.1], [0.5, 0.7]]),
        ]
        biases = [0, np.array([0.1, 0.3]), np.array([0.7, 0.3])]
        net.init(weights=weights, biases=biases)
        self.assertListEqual(net.layers, [n_input, 2, n_output])
        input = np.array([0.7, 0.3, 0.6])
        output = net.forward(input)
        self.assertTrue(np.array_equal(output[1][0], input))
        self.assertTrue(np.array_equal(output[1][1], [0.629, 0.698]))
        self.assertTrue(np.array_equal(output[1][2], [0.747, 0.751]))

    def test_forward_4_2_3(self):
        n_input = 4
        n_output = 3
        net = network.Network(n_input, n_output)
        net.add_layer(2)
        weights = [
            0,
            np.array([[0.8, 0.3, 0.1, 0.2], [0.8, 0.5, 0.0, 0.9]]),
            np.array([[0.5, 0.1], [0.6, 0.2], [0.8, 0.2]]),
        ]
        biases = [0, np.array([-0.9, -0.3]), np.array([-0.3, -0.5, -1.0])]
        net.init(weights=weights, biases=biases)
        self.assertListEqual(net.layers, [n_input, 2, n_output])
        input = np.array([0.9, 0.9, 0.1, 0.1])
        output = net.forward(input)
        self.assertTrue(np.array_equal(output[1][0], input))
        self.assertTrue(np.array_equal(output[1][1], [0.53, 0.723]))
        self.assertTrue(np.array_equal(output[1][2], [0.509, 0.491, 0.394]))

    def test_forward_5_1_1(self):
        n_input = 5
        n_output = 1
        net = network.Network(n_input, n_output)
        net.add_layer(1)
        weights = [
            0,
            np.array([[0.6, 0.1, 1.0, 1.0, 0.3]]),
            np.array([[0.2]])
        ]
        biases = [0, np.array([0.1]), np.array([-0.5])]
        net.init(weights=weights, biases=biases)
        self.assertListEqual(net.layers, [n_input, 1, n_output])
        input = np.array([1.0, 1.0, 0.6, 0.1, 0.3])
        output = net.forward(input)
        self.assertTrue(np.array_equal(output[1][0], input))
        self.assertTrue(np.array_equal(output[1][1], [0.831]))
        self.assertTrue(np.array_equal(output[1][2], [0.417]))

    def test_backprop_3_2_2(self):
        print()
        n_input = 3
        n_output = 2
        net = network.Network(n_input, n_output)
        net.add_layer(2)
        weights = [
            0,
            np.array([[0.1, 0.8, 0.2], [0.3, 0.1, 0.5]]),
            np.array([[0.5, 0.1], [0.5, 0.7]]),
        ]
        biases = [0, np.array([0.1, 0.3]), np.array([0.7, 0.3])]
        net.init(weights=weights, biases=biases)
        self.assertListEqual(net.layers, [n_input, 2, n_output])
        input = np.array([0.7, 0.3, 0.6])
        output = net.forward(input)
        print(output[1])
        expected_output = np.array([0.0, 1.0])
        net.backprop(input, expected_output)
        output = net.forward(input)
        print(output[1])


if __name__ == '__main__':
    unittest.main()
