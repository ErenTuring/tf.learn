# output[b, i, j, :] = sum_{di, dj} input[b, strides[1] * i + di - pad_top,
#         strides[2] * j + dj - pad_left, ...] * filter[di, dj, ...]
import numpy as np
one = np.ones([3, 3], int)
print(one)
one_f = np.float64(one)
print(one_f)