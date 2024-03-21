import torch

#(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
#저희 모델의 경우 추론 위주로 할 것 같으니, track_running_stats을 false로 한다.(training까지 하면 cache 같은 변수로 저장해야한다.)
import math
import numpy as np
from typing import Any
class batchNorm2D:
    """
    그림에는 안나와있지만, paper에 수행한다고 나와있다.
    inference 시에는 모멘텀 안씀.
    
    """
    # num_feature 안맞으면 에러 출력

    def __init__(self,num_features=64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False, device=None, dtype=None) -> None:
        self.num_features=num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.running_mean = 0
        self.running_var = 0
        self.weight =0
        self.bias = 0

        
    def _norm2D(self,input_features):
        batch,channel ,length= len(input_features),len(input_features[0]), len(input_features[0][0])
        output= [[[ [ self.weight[c]* ((input_features[b][c][lh][lw] -self.running_mean[c]) / math.sqrt(self.running_var[c]+self.eps
                        )) + self.bias[c]  for lw in range(length)] for lh in range(length)] for c in range(channel)]  for b in range(batch) ]

        return output


    def init_mean_weight(self,running_mean,running_var,weight,bias):
        self.running_mean=running_mean.tolist()
        self.running_var =running_var.tolist()
        self.weight=weight.tolist()
        self.bias =bias.tolist()



    def __call__(self, input_features) -> Any:
        return self._norm2D(input_features)
    

if __name__ == "__main__" :
    norm2d = batchNorm2D(64)
    data = torch.load("pretrained_weight.pth")  
    norm2d.init_mean_weight(data['bn1.running_mean'],data['bn1.running_var'],data['bn1.weight'],data['bn1.bias'],)
    norm2d_t = torch.nn.BatchNorm2d(64).eval()

    norm2d_t.running_mean= data['bn1.running_mean']
    norm2d_t.running_var = data['bn1.running_var']
    norm2d_t.weight =data['bn1.weight']
    norm2d_t.bias=data['bn1.bias']
    X_t = torch.randn(size=(1,64,224,224))
    X = X_t.tolist()
    k=norm2d(X)
    print(k[0][0][0][:3])
    print(norm2d_t(X_t)[0][0][0][:3])