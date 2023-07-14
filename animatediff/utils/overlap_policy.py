import numpy as np


def ordered_halving(i):
    return int('{:064b}'.format(i)[::-1], 2) / (1 << 64)


def uniform(step, steps, n, context_size, strides, overlap, closed_loop=True):
    if n <= context_size:
        yield list(range(n))
        return
    strides = min(strides, int(np.ceil(np.log2(n / context_size))) + 1)
    for stride in 1 << np.arange(strides):
        pad = int(round(n * ordered_halving(step)))
        for j in range(
                int(ordered_halving(step) * stride) + pad,
                n + pad + (0 if closed_loop else -overlap),
                (context_size * stride - overlap)
        ):
            yield [e % n for e in range(j, j + context_size * stride, stride)]
