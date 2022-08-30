import paddle
import numpy as np

def new_tensor(data,src):
    if isinstance(data,paddle.Tensor):
        return paddle.to_tensor(data.numpy(), dtype=src.dtype, place=src.place)
    else:
        return paddle.to_tensor(data,dtype=src.dtype,place=src.place)


def new_ones(shape,src):
    return paddle.to_tensor(np.ones(shape=shape),dtype=src.dtype,place=src.place)

def new_zeros(shape,src):
    return paddle.to_tensor(np.zeros(shape=shape),dtype=src.dtype,place=src.place)

def view(data, shape):
    return paddle.reshape(x=data, shape=shape)

def type(data, type):# 张量转化为特定类型
    return paddle.to_tensor(data=data, dtype=type)

def get_shape(x,k):
    return x.shape[k]