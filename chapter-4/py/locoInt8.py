import numpy as np
import time

def generate_weight_matrix(rows: int, cols: int) -> np.ndarray:
    # Generates a random weight matrix with values uniformly distributed between -1 and 1
    return np.random.uniform(-1, 1, (rows, cols)).astype(np.float32)

def quantize_to_int8(matrix: np.ndarray) -> (np.ndarray, float):
    # Quantizes the input matrix to int8
    scale = np.max(np.abs(matrix)) / 127  # Scale factor for int8 (-128 to 127, signed 8-bit integer)
    quantized = np.clip(np.round(matrix / scale), -128, 127).astype(np.int8) # Ensure values fit in int8 range (round and cast)
    return quantized, scale

def dequantize_from_int8(quantized: np.ndarray, scale: float) -> np.ndarray:
    # Dequantizes the int8 matrix back to float32
    return quantized.astype(np.float32) * scale

def low_rank_compensation(original: np.ndarray, rank: int=8) -> np.ndarray:
    # The idea is to approximate the original matrix with a low-rank matrix
    # In details, we perform SVD and keep only the top 'rank' singular values/vectors
    # the shape of original is (m, n)
    U, S, Vt = np.linalg.svd(original, full_matrices=False) # the shape of U is (m, m), S is (min(m,n),), Vt is (n, n)

    # Get the top 'rank' components
    A = U[:, :rank] * np.sqrt(S[:rank])
    # Get the corresponding B matrix
    B = (np.sqrt(S[:rank])[:, np.newaxis] * Vt[:rank, :])

    # Reconstruct the low-rank approximation
    compensated = np.dot(A, B)
    return compensated, A, B

def apply_low_rank_compensation(quantized: np.ndarray, scale: float, compensation: (np.ndarray, np.ndarray)) -> np.ndarray:
    # Dequantize the quantized matrix
    dequantized = dequantize_from_int8(quantized, scale)
    # Apply low-rank compensation
    compensated = dequantized + compensation
    return compensated

def calculate_mean_squared_error(original: np.ndarray, reconstructed: np.ndarray) -> float:
    # Calculates the Mean Squared Error between the original and reconstructed matrices
    return np.mean((original - reconstructed) ** 2)

def main():
    np.random.seed(42)  # For reproducibility

    rows, cols = 256, 256
    original_matrix = generate_weight_matrix(rows, cols)

    # Int8 Quantization
    quantized_int8, scale_int8 = quantize_to_int8(original_matrix)
    reconstructed_int8 = dequantize_from_int8(quantized_int8, scale_int8)
    mse_int8 = calculate_mean_squared_error(original_matrix, reconstructed_int8)
    print(f"Int8 Quantization MSE: {mse_int8}")

    # Generate low-rank compensation
    residual = original_matrix - reconstructed_int8
    compensation, A, B = low_rank_compensation(residual, rank=8)

    # Apply Low-Rank Compensation
    compensated_reconstruction = apply_low_rank_compensation(quantized_int8, scale_int8, compensation)
    mse_compensated = calculate_mean_squared_error(original_matrix, compensated_reconstruction)
    print(f"Int8 with Low-Rank Compensation MSE: {mse_compensated}")

    print(f"Original Matrix Sample:\n{original_matrix[:5, :5]}\n")
    print(f"INT8 Quantized Sample:\n{quantized_int8[:5, :5]}\n")
    print(f"INT8 Dequantized Sample:\n{reconstructed_int8[:5, :5]}\n")
    print(f"INT8 MSE: {mse_int8}\n\n")
    print(f"Compensated Reconstruction Sample:\n{compensated_reconstruction[:5, :5]}\n")
    print(f"Compensated MSE: {mse_compensated}\n")

if __name__ == "__main__":
    main()