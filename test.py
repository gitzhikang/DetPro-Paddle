# import paddle
import torch
import numpy as np
import cv2

def repeat(input,repeatList):
    if(len(repeatList)==1):
        repeat_factor = repeatList[0]
        res = np.zeros(shape=[repeat_factor*input.shape[0]])
        for i in range(res.shape[0]):
            res[i] = input[i % input.shape[0]]
    elif(len(repeatList)==2):
        repeat_factor_d0 = repeatList[0]
        repeat_factor_d1 = repeatList[1]
        res = np.zeros(shape=[repeat_factor_d0*input.shape[0],repeat_factor_d1*input.shape[1]])
        for i in range(res.shape[0]):
            for j in range(res.shape[1]):
                res[i][j] = input[i%input.shape[0]][j%input.shape[1]]
    res=paddle.to_tensor(res)
    return res

if __name__ == '__main__':
    # src = torch.arange(1, 11).reshape((2, 5))
    #
    # index = torch.tensor([[0, 1, 2], [0, 1, 4]])
    # np.zeros(shape=[2,3],dtype="float32")
    # a = torch.range(1, 6).resize(2,3)
    # b=paddle.

    repeat([])
    # c=torch.ones(4,5)
    # a[:, [0, 2]] = c[:, [0, 2]]
    # print(a)
    # print(b)
    # print(c)
    # print(c.shape)
    x=7
    print((x,x)[0])
    # print(repeat(a,torch.tensor([2,2])))

