import numpy as np
import tensorflow as tf

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Preprocess data
x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
x_test = x_test.reshape(-1, 784).astype('float32') / 255.0
y_train = np.eye(10)[y_train]
y_test = np.eye(10)[y_test]

# Define model parameters
input_dim = 784
hidden_dim = 32
output_dim = 10

# Initialize weights and biases
weights1 = np.random.randn(input_dim, hidden_dim) / np.sqrt(input_dim)
bias1 = np.zeros((1, hidden_dim))
weights2 = np.random.randn(hidden_dim, output_dim) / np.sqrt(hidden_dim)
bias2 = np.zeros((1, output_dim))

print("shape(weights1)=", weights1.shape, ", shape(bias1)=", bias1.shape, "shape(weights2)=", weights2.shape, ", shape(bias2)=", bias2.shape)

# Define activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(x, 0)

def d_relu_dx(x):
    return (x > 0).astype(int)

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

# Define cross-entropy loss
def cross_entropy_loss(output, labels):
    return -np.mean(np.sum(labels * np.log(output), axis=1))

# Train model using mini-batches
batch_size = 128
num_batches = int(np.ceil(len(x_train) / batch_size))
learning_rate = 0.01

print("batch_size=", batch_size, ", num_batches=", num_batches, ", learning_rate=", learning_rate)

first0 = True
first1 = True
for epoch in range(100):
    indices = np.arange(len(x_train))
    np.random.shuffle(indices)
    x_train_shuffled = x_train[indices]
    y_train_shuffled = y_train[indices]

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = (batch_idx + 1) * batch_size

        batch_x = x_train_shuffled[start_idx:end_idx]
        batch_y = y_train_shuffled[start_idx:end_idx]

        if first0:
            first0 = False
            print("shape(batch_x)=", batch_x.shape, ", shape(batch_y)=", batch_y.shape, batch_y[0])

        # Forward pass
        hidden_layer_x = np.dot(batch_x, weights1) + bias1
        hidden_layer = relu(hidden_layer_x)
        output = softmax(np.dot(hidden_layer, weights2) + bias2)

        if first1:
            first1 = False
            print("shape(hidden_layer_x)=", hidden_layer_x.shape, ", shape(output)=", output.shape)

        # Compute loss and gradients
        loss = cross_entropy_loss(output, batch_y)
        d_output = output - batch_y
        d_hidden_layer = np.dot(d_output, weights2.T) * d_relu_dx(hidden_layer_x)
        d_weights2 = np.dot(hidden_layer.T, d_output)
        d_bias2 = np.sum(d_output, axis=0, keepdims=True)
        d_weights1 = np.dot(batch_x.T, d_hidden_layer)
        d_bias1 = np.sum(d_hidden_layer, axis=0, keepdims=True)

        # Update weights and biases
        weights1 -= learning_rate * d_weights1 / batch_size
        bias1 -= learning_rate * d_bias1 / batch_size
        weights2 -= learning_rate * d_weights2 / batch_size
        bias2 -= learning_rate * d_bias2 / batch_size

    print(f'Epoch {epoch+1}, Loss: {loss}')

# Evaluate model on test set
hidden_layer = relu(np.dot(x_test, weights1) + bias1)
output = softmax(np.dot(hidden_layer, weights2) + bias2)
accuracy = np.mean(np.argmax(output, axis=1) == np.argmax(y_test, axis=1))
print(f'Test Accuracy: {accuracy:.3f}')

