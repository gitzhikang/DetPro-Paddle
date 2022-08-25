import torch
import numpy as np


def scatter(input,index,src):
    for i in range(index.shape[0]):
        for j in range(index.shape[1]):
            input[i][index[i][j]] = src[i][j]
    return input

if __name__ == '__main__':
    src = torch.arange(1, 11).reshape((2, 5))

    index = torch.tensor([[0, 1, 2], [0, 1, 4]])
    z=scatter(torch.zeros(3, 5, dtype=src.dtype),index,src)
    print(z)
    y=torch.zeros(3, 5, dtype=src.dtype).scatter_(1, index, src)
    print(y)
