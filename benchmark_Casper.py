import numpy as np, timeit, pandas as pd

from tqdm import tqdm
from itertools import product
from typing import Tuple


def setup(n: int, nbits: int) -> Tuple[np.ndarray, np.ndarray]:
    bits = np.random.randint(0, 2, size=(n, nbits))
    string_bits = np.array(["".join(str(j) for j in i) for i in bits])
    return bits, string_bits


def hamming_vector(states: np.ndarray) -> np.ndarray:
    return (states[:, None] != states[None]).mean(-1)

def vector_one2all(states: np.ndarray) -> np.ndarray:
    return (states[0, None] != states[None]).mean(-1)


def hamming_distance(string1: str, string2: str) -> int:
    """
    Function to compute string difference using Hamming distance.

    Parameters:
    - string1: First string in comparison
    - string2: Second string in comparison

    Returns:
    - distance: number differing characters between string1 and string2
    """

    distance = 0
    L = len(string1)

    for i in range(L):
        if string1[i] != string2[i]:
            distance += 1

    return distance


def py_hamming_distance(bits: np.ndarray) -> np.ndarray:
    output = np.zeros((bits.size, bits.size))
    for idx, a in enumerate(bits):
        for jdx, b in enumerate(bits):
            if idx < jdx:
                output[idx, jdx] = hamming_distance(a, b)

    return output


def py_hamm_one2all(bits: np.ndarray) -> np.ndarray:
    output = np.zeros((bits.size, bits.size))
    jdx = 2
    a = bits[jdx]

    for idx, b in enumerate(bits):
        if idx == jdx:
            continue

        hammingDistance = hamming_distance(a, b)

        if idx < jdx:
            output[idx,jdx] = hammingDistance
        elif idx > jdx:
            output[jdx,idx] = hammingDistance

    return output



if __name__ == "__main__":
    run_settings = dict(number=10, repeat=10)
    sizes = [10, 100, 500, 1000]
    nbits = [3]
    df = []
    for size, nbit in tqdm(product(sizes, nbits)):
        bits, string_bits = setup(size, nbit)
        vector_time = timeit.repeat(
            "hamming_vector(bits)", globals=globals(), **run_settings
        )
        # python_time = timeit.repeat(
        #     "py_hamming_distance(string_bits)", globals=globals(), **run_settings
        # )
        one2all_time = timeit.repeat(
            "py_hamm_one2all(bits)", globals=globals(), **run_settings
        )
        vOne2All = timeit.repeat(
            "vector_one2all(bits)", globals=globals(), **run_settings
        )
        row = dict(
            size=size,
            bits=nbit,
            vector=vector_time,
            # py=python_time,
            one=one2all_time,
            vOne=vOne2All,
        )
        df.append(row)

    df = pd.DataFrame(df)

    from matplotlib import pyplot as plt

    fig, ax = plt.subplots()
    for (size, bit), dfi in df.groupby("size bits".split()):
        x = size * np.ones(len(dfi.vector.iloc[0]))
        # ax.scatter(x, dfi.py.iloc[0], color="steelblue")
        ax.scatter(x, dfi.vector.iloc[0], color="purple")
        ax.scatter(x, dfi.one.iloc[0], color="red")
        ax.scatter(x, dfi.vOne.iloc[0], color="green")
    handles = [
        plt.Line2D([], [], color=c, label=l)
        # for c, l in zip("steelblue purple red green".split(), "python vector py_one2all vec_one2all".split())
        for c, l in zip("purple red green".split(), "vector py_one2all vec_one2all".split())
    ]
    ax.legend(handles=handles)
    ax.set_xlabel("Number of agents (n)")
    ax.set_ylabel("Run time")

    plt.show(block=1)