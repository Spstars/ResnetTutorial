#(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
#저희 모델의 경우 추론 위주로 할 것 같으니, track_running_stats을 false로 한다.(training까지 하면 cache 같은 변수로 저장해야한다.)
import math
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
        self.running_mean = 64 * [-1.1569e-02]
        self.running_var = 64 * [5.9482e-01]
        self.weight = 64 * [1.1]
        self.bias = 64 * [0.11]

    # batch와 채널을 다 풀고, input_feature에 running_mean과 running_var로 나눈다.
    def _norm2D(self,input_features):
        batch,channel ,length= len(input_features),len(input_features[0]), len(input_features[0][0])

        output= [[[ [ self.weight[b]* ((input_features[b][c][lh][lw] -self.running_mean[b]) / math.sqrt(self.running_var[b]+self.eps))+ self.bias[b]  for lw in range(length)] for lh in range(length)] for c in range(channel)]  for b in range(batch) ]
        print("batchnorm2d : ", batch,channel,length,len(output[0][0][0]))
       
        return output


    def init_mean_weight(self,running_mean,running_var,weight,bias):
        self.running_mean=running_mean.tolist()
        self.running_var =running_var.tolist()
        self.weight=weight.tolist()
        self.bias =bias.tolist()
    def __call__(self, input_features) -> Any:
        return self._norm2D(input_features)