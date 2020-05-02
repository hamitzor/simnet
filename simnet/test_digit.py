from . import network, helper
import numpy as np
import imageio


def main():
    n_input = 28*28
    n_output = 10

    net = network.Network(n_input, n_output)
    net.add_layer(16)
    net.add_layer(16)
    net.init()

    im3 = imageio.imread("data/training/3/10.png").flatten()

    for data in helper.suffled_digits("data/training"):
        input = imageio.imread("data/training/3/10.png").flatten()
        expected_output = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        net.backprop(input, expected_output)
        print(data)

    
    


    output = net.forward(im3)
    print(output[1][-1])


if __name__ == '__main__':
    main()
