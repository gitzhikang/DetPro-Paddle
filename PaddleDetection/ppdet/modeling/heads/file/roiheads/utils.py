import paddle
import numpy as np

def new_tensor(data,src,place=None):
    if place == None:
        tensor_place = src.place
    else:
        tensor_place = place
    if isinstance(data,paddle.Tensor):
        return paddle.to_tensor(data.numpy(), dtype=src.dtype, place=tensor_place)
    else:
        return paddle.to_tensor(data,dtype=src.dtype,place=tensor_place)


def new_ones(shape,src):
    return paddle.to_tensor(np.ones(shape=shape),dtype=src.dtype,place=src.place)

def new_zeros(shape,src):
    return paddle.to_tensor(np.zeros(shape=shape),dtype=src.dtype,place=src.place)

def new_full(shape,data,src,dtype=None):
    if dtype == None:
        return paddle.to_tensor(np.full(shape=shape,fill_value=data),dtype=src.dtype, place=src.place)
    else:
        return paddle.to_tensor(np.full(shape=shape, fill_value=data), dtype=dtype, place=src.place)

def arange(start,end=None,dtype=None,place=None):
    if dtype == None:
        dtype = paddle.int32
    return paddle.to_tensor(np.arange(start,end),dtype=dtype,place=place)

def view(data, shape):
    return paddle.reshape(x=data, shape=shape)

def type(data, type):# 张量转化为特定类型
    return paddle.to_tensor(data=data, dtype=type)

def get_shape(x,k):
    return x.shape[k]

def repeat(input, repeatList):
    if (len(repeatList)== 1):
        repeat_factor = repeatList[0]
        res = np.zeros(shape=[repeat_factor * input.shape[0]], dtype="float32")
        print(res.shape[0])
        for i in range(res.shape[0]):
            res[i] = input[i % input.shape[0]].item()

    elif (len(repeatList) == 2):
        repeat_factor_d0 = repeatList[0]
        repeat_factor_d1 = repeatList[1]
        res = np.zeros(shape=[repeat_factor_d0 * input.shape[0], repeat_factor_d1 * input.shape[1]], dtype="float32")
        print(res.shape)
        for i in range(res.shape[0]):
            for j in range(res.shape[1]):
                res[i][j] = input[i % input.shape[0]][j % input.shape[1]].item()

    res = paddle.to_tensor(res)
    return res