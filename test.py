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

def test(a):
    return a

def main(self,**kwargs):
    a=1
    test(a, **kwargs)

if __name__ == '__main__':
    x1=torch.range(1,200).resize(10,20)
    y1 = x1[[1,2,3], [2,3,4]][:, None]
    print(y1)
    torch.s
    # c=torch.ones(4,5)
    # a[:, [0, 2]] = c[:, [0, 2]]
    # print(a)
    # print(b)
    # print(c)
    # print(c.shape)

    # print(repeat(a,torch.tensor([2,2])))

