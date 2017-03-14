
from Net import Net
from xor_dao import XorDAO

net = Net([2,5,1], XorDAO())
net.learn_by_back_propagation(500, 0.4)

# from yahoo_dao import SingleClosingSeries
# data = SingleClosingSeries('AAPL')
# net = Net([data.input_size, 4, data.output_size], data)
# net.learn_by_back_propagation(500, 0.4)
# print(net._get_weights())
# print(net._get_biases())