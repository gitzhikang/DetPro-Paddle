# import paddle
import torch
import numpy as np
import cv2

def repeat(input,repeatList):
    if(repeatList.dim()==1):
        repeat_factor = repeatList[0]
        res = np.zeros(shape=[repeat_factor*input.shape[0]])
        for i in range(res.shape[0]):
            res[i] = input[i % input.shape[0]]
    elif(repeatList.dim()==2):
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

    with open('/Users/songzhikang/Downloads/111.jpg', 'rb') as f:
        im = f.read()
    data = np.frombuffer(im, dtype='uint8')
    im = cv2.imdecode(data, 1)  # BGR mode, but need RGB mode

    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    # c=torch.ones(4,5)
    # a[:, [0, 2]] = c[:, [0, 2]]
    # print(a)
    # print(b)
    # print(c)
    # print(c.shape)
    print(im.shape[:2])
    # print(repeat(a,torch.tensor([2,2])))

