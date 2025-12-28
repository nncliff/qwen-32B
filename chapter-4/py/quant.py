import numpy as np

def generate_random_matrix(rows: int, cols: int) -> np.ndarray:
    # Generates a random matrix with values uniformly distributed between -1 and 1
    return np.random.uniform(-1, 1, (rows, cols))

def quantize_to_int8(matrix: np.ndarray) -> np.ndarray:
    # Quantizes the input matrix to int8
    scale = np.max(np.abs(matrix)) / 127  # Scale factor for int8
    quantized = np.round(matrix / scale).astype(np.int8)
    return quantized, scale

def dequantize_from_int8(quantized: np.ndarray, scale: float) -> np.ndarray:
    # Dequantizes the int8 matrix back to float32
    return quantized.astype(np.float32) * scale

def quantize_to_int4(matrix: np.ndarray) -> np.ndarray:
    # Quantizes the input matrix to int4
    scale = np.max(np.abs(matrix)) / 7  # Scale factor for int4
    quantized = np.round(matrix / scale) #.astype(np.int8)???
    quantized = np.clip(quantized, -8, 7).astype(np.int8)  # Ensure values fit in int4 range
    return quantized, scale

def dequantize_from_int4(quantized: np.ndarray, scale: float) -> np.ndarray:
    # Dequantizes the int4 matrix back to float32
    return quantized.astype(np.float32) * scale

def calculate_mean_squared_error(original: np.ndarray, reconstructed: np.ndarray) -> float:
    # Calculates the Mean Squared Error between the original and reconstructed matrices
    return np.mean((original - reconstructed) ** 2)

def main():
    np.random.seed(42)  # For reproducibility

    rows, cols = 100, 100
    original_matrix = generate_random_matrix(rows, cols)

    # Int8 Quantization
    quantized_int8, scale_int8 = quantize_to_int8(original_matrix)
    reconstructed_int8 = dequantize_from_int8(quantized_int8, scale_int8)
    mse_int8 = calculate_mean_squared_error(original_matrix, reconstructed_int8)
    print(f"Int8 Quantization MSE: {mse_int8}")

    # Int4 Quantization
    quantized_int4, scale_int4 = quantize_to_int4(original_matrix)
    reconstructed_int4 = dequantize_from_int4(quantized_int4, scale_int4)
    mse_int4 = calculate_mean_squared_error(original_matrix, reconstructed_int4)
    print(f"Int4 Quantization MSE: {mse_int4}")

    print(f"Original Matrix Sample:\n{original_matrix}\n")
    print(f"INT8 Quantized Sample:\n{quantized_int8}\n")
    print(f"INT8 Dequantized Sample:\n{reconstructed_int8}\n")
    print(f"INT8 MSE: {mse_int8}\n\n")
    print(f"INT4 Quantized Sample:\n{quantized_int4}\n")
    print(f"INT4 Dequantized Sample:\n{reconstructed_int4}\n")
    print(f"INT4 MSE: {mse_int4}\n")

if __name__ == "__main__":
    main()