import numpy as np

def fwht_inplace(x: np.ndarray) -> None:
    n = x.shape[0]
    if n & (n - 1) != 0:
        raise ValueError("FWHT length must be power of two.")
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            j1 = i
            j2 = i + h
            for k in range(h):
                a = x[j1 + k]
                b = x[j2 + k]
                x[j1 + k] = a + b
                x[j2 + k] = a - b
        h <<= 1

def _bitcount_uint32(v: np.uint32) -> int:
    return int(v).bit_count()

def paley_order_indices(n: int) -> np.ndarray:
    if n & (n - 1) != 0:
        raise ValueError("Paley order requires n power of two.")
    idx = np.arange(n, dtype=np.uint32)
    gray = idx ^ (idx >> 1)
    weights = np.array([_bitcount_uint32(v) for v in gray], dtype=np.int32)
    order = np.lexsort((idx, weights))
    return order.astype(np.int64)
