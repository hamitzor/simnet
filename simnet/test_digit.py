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
    
    d0 = imageio.imread("data/training/0/69.png").flatten()/256
    d1 = imageio.imread("data/training/1/59.png").flatten()/256
    d3 = imageio.imread("data/training/3/30.png").flatten()/256
    d5 = imageio.imread("data/training/5/47.png").flatten()/256
    d9 = imageio.imread("data/training/9/45.png").flatten()/256

    print(net.forward(d0)[1][-1])
    print(net.forward(d1)[1][-1])
    print(net.forward(d3)[1][-1])
    print(net.forward(d5)[1][-1])
    print(net.forward(d9)[1][-1])


    digits = helper.suffled_digits("data/training")

    for i in range(0, 60000-100, 100):
        examples = []
        for digit in digits[i:i+100]:
            input = imageio.imread(digit[1]).flatten()/256
            output = [0.0] * 10
            output[digit[0]] = 1.0
            examples.append((input, output))

        net.backprop(examples)
        print("**************************************")
        print("BACKPROPAGATION FINISHED - " + str(i))
        print("**************************************")

    print(net.forward(d0)[1][-1])
    print(net.forward(d1)[1][-1])
    print(net.forward(d3)[1][-1])
    print(net.forward(d5)[1][-1])
    print(net.forward(d9)[1][-1])

if __name__ == '__main__':
    main()
