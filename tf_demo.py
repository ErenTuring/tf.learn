

# output[b, i, j, :] = sum_{di, dj} input[b, strides[1] * i + di - pad_top,
#         strides[2] * j + dj - pad_left, ...] * filter[di, dj, ...]