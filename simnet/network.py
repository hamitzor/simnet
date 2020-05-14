from . import helper
import numpy as np


class Network:

    def __init__(self, n_input, n_output):
        self.n_input = n_input
        self.n_output = n_output
        self.layers = [n_input]
        self.n_layer = 2

    def init(self, weights=None, biases=None):
        self.layers.append(self.n_output)
        self.weights = weights if weights else [np.random.uniform(-5, 5, (self.layers[(i + 1)], self.layers[i])) for i in range(len(self.layers) - 1)]
        self.biases = biases if biases else [np.zeros(self.layers[(i + 1)]) for i in range(len(self.layers) - 1)]
        print(self.weights)
        print(self.biases)
        if not weights:
            self.weights.insert(0, 0)
        if not biases:
            self.biases.insert(0, 0)

    def add_layer(self, n_neuron):
        self.layers.append(n_neuron)
        self.n_layer += 1

    def forward(self, input):
        a = [np.zeros(self.layers[i]) for i in range(1, len(self.layers))]
        z = [np.zeros(self.layers[i]) for i in range(1, len(self.layers))]
        a.insert(0, input)
        z.insert(0, 0)
        for l in range(1, len(self.layers)):
            z[l] = self.weights[l].dot(a[(l - 1)]) + self.biases[l]
            a[l] = np.round(helper.sigmoid_arr(z[l]), 3)

        return (z, a)

    def backprop(self, training_examples):
        uw_total = None
        ub_total = None

        for example in training_examples:
            forward = self.forward(example[0])
            z = forward[0]
            a = forward[1]
            uw = [0] * self.n_layer
            ub = [0] * self.n_layer
            ua = [0] * self.n_layer
            ua[-1] = example[1]

            print(a[-1], np.asarray(example[1]))
            print(np.sum(np.square((a[-1] - np.asarray(example[1])))))

            for l in reversed(range(1, self.n_layer)):
                uw[l]=np.empty(self.weights[l].shape)
                ub[l]=np.empty(self.biases[l].shape)
                ua[l - 1]=np.empty(self.layers[(l - 1)])
                for j in range(self.layers[l]):
                    dif_z_a=helper.sigmoid_prime(z[l][j])
                    dif_a_C0=2 * (a[l][j] - ua[l][j])
                    ub[l][j]=dif_z_a * dif_a_C0
                    for k in range(self.layers[(l - 1)]):
                        dif_w_z=a[(l - 1)][k]
                        uw[l][j][k]=round(dif_w_z * dif_z_a * dif_a_C0, 3)

                for k in range(self.layers[(l - 1)]):
                    ua[(l - 1)][k]=0
                    for j in range(self.layers[l]):
                        dif_a_z=self.weights[l][j][k]
                        dif_z_a=helper.sigmoid_prime(z[l][j])
                        dif_a_C0=2 * (a[l][j] - ua[l][j])
                        ua[(l - 1)][k] += round(dif_a_z * dif_z_a * dif_a_C0, 3)

            uw_total=uw_total + np.asarray(uw) if uw_total is not None else np.asarray(uw)
            ub_total=ub_total + np.asarray(ub) if ub_total is not None else np.asarray(ub)

        self.weights=np.asarray(self.weights) - uw_total*0.001
        self.biases=np.asarray(self.biases) - ub_total*0.001
