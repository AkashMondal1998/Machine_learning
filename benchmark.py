import time
import numpy as np

from neural_engine import Tensor


def benchmark_addition(tensor_size):
    tensor1 = Tensor(np.random.rand(tensor_size))
    tensor2 = Tensor(np.random.rand(tensor_size))

    start_time = time.time()
    result = tensor1 + tensor2
    end_time = time.time()

    return end_time - start_time


def benchmark_multiplication(tensor_size):
    tensor1 = Tensor(np.random.rand(tensor_size))
    tensor2 = Tensor(np.random.rand(tensor_size))

    start_time = time.time()
    result = tensor1 * tensor2
    end_time = time.time()

    return end_time - start_time


def benchmark_dot_product(tensor_size):
    tensor1 = Tensor(np.random.rand(tensor_size))
    tensor2 = Tensor(np.random.rand(tensor_size))

    start_time = time.time()
    result = tensor1.dot(tensor2)
    end_time = time.time()

    return end_time - start_time


if __name__ == "__main__":
    tensor_size = 1000000  # Adjust the size as needed for your benchmark

    addition_time = benchmark_addition(tensor_size)
    print(f"Addition benchmark for tensor size {tensor_size}: {addition_time:.6f} seconds")

    multiplication_time = benchmark_multiplication(tensor_size)
    print(f"Multiplication benchmark for tensor size {tensor_size}: {multiplication_time:.6f} seconds")

    dot_product_time = benchmark_dot_product(tensor_size)
    print(f"Dot product benchmark for tensor size {tensor_size}: {dot_product_time:.6f} seconds")
