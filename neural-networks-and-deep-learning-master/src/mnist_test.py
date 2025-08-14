from mnist_loader import *
import network


training_data, validation_data, test_data = load_data_wrapper()

training_data = list(training_data)
validation_data = list(validation_data)
test_data = list(test_data)

net = network.Network([784, 30, 10])

net.SGD(training_data=training_data, epochs=30, mini_batch_size=10, eta=3.0, test_data=test_data)



