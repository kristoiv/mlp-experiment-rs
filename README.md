# Naive Pure-Rust Experiment in Multi-Layer Perceptrons (MLPs)
This project is a naive implementation of a multi-layer perceptron (MLP) designed to classify the MNIST dataset. The network uses ReLU activation in the hidden layer, softmax for the output layer, and achieves 90%+ accuracy in just 10 epochs with a single hidden layer of 32 neurons.

## Implementation Details
- **Language**: Pure rust (~850 lines incl. some tests)
- **Tensor**: Implements a subset of numpy-like functionality:
    - Unvectorized operations
    - Single-threaded execution
    - No optimizations (e.g., no GPU acceleration or vectorization)

The focus was on understanding and internalizing the core mathematics and matrix operations behind simple MLPs, rather than creating a production-ready implementation.

## Key Achievements
- Demonstrated the ability to achieve high accuracy (~95%) with minimal code complexity/optimizations.
- Successfully implemented/understand fundamental neural network components:
    - Forward propagation
    - Backpropagation
    - Gradient descent

## Known Limitations and Future Work
- The implementation is unoptimized and not production-ready.
- Many inefficiencies (e.g., unnecessary clones) remain, but code quality was not the focus of this experiment.

The project has achieved its primary goal: to solidify the mathematical understanding behind basic neural network operations.

## NB.
The overly verbose readme above was written by an LLM in keeping with the AI-focus of this project!
