from typing import Any
import torch
import torch.nn as nn
import numpy as np
class maxpool:
    #stride는 일단 default로 설정해볼 것
    def __init__(self,kernel_size=(3, 3), stride=(2,2), padding=1,dlilation=1):
        self.kernel_size =kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dlilation = 1 
        self.negative_inf = float("-inf") #여기서는 패딩이 음의 무한대(maxpool)

    def maxpooling(self,input_feature):
        num_batch =len(input_feature)
        num_channel = len(input_feature[0])
        width= len(input_feature[0][0])
        padding = self.padding

        if self.padding >0:
            for b in range(num_batch):
                for c in range(num_channel):
                    input_feature[b][c] =[[self.negative_inf]*(width+2*padding)]*padding + \
                        [[self.negative_inf]*padding+feature+[self.negative_inf]*padding for feature in input_feature[b][c]]+ \
                            [[self.negative_inf]*(width+2*padding)]*padding 


        output_feature = [[[[ 0 for _ in range(width//2)] for _ in range(width//2)] for _ in range(num_channel)]  for _ in range(num_batch) ]
        for b in range(num_batch):
            for c in range(num_channel):
                for hp in range(0,width//2):
                    for sp in range(0,width//2):
                        #conv 연산
                        output_feature[b][c][hp][sp] = self.pool(input_feature[b][c],sp,hp)
        return output_feature
    
    def pool(self,input_feature,sp,hp):
        """
        sp x좌표와 hp y좌표를 통해 원래 위치를 유추하여, 최대값을 찾는다.
        """
        maxelement= float('-inf')
        # 3x3 밖에 없으니 그냥 ij 돌리겠음.
        for j in [-1,0,1]:
            for i in [-1,0,1]:
                origin_coord_x = self.padding +self.stride[1] *sp+i
                origin_coord_y = self.padding +self.stride[0] *hp+j
                if input_feature[origin_coord_y][origin_coord_x] >= maxelement:
                    maxelement = input_feature[origin_coord_y][origin_coord_x]
        return maxelement
    
    
    def __call__(self, input) -> Any:
        return self.maxpooling(input)
    
if __name__ =="__main__":
    m = nn.MaxPool2d(3,stride=2,padding=1)
    m2 = maxpool((3,3),stride=(2,2),padding=1)
    input = torch.randn(1, 3, 64, 64)
    input2 = input.tolist()
    print(m(input)[0][0][0])
    print(m2(input2)[0][0][0])