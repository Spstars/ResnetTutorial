
    # (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
    # (fc): Linear(in_features=2048, out_features=1000, bias=True)

from typing import Any


class fc:
    """
        avgpooling 한 값을 받아 1000의 classification을 수행한다.
        
        numpy식으로는 np.dot(arr,weights.T) +bias 이다.
    """
    #in_features 달라지면 err 출력
    #예시로는 input 2048 output 1000 batch : 64
    # weight bias 초기화 할때 method를 어떻게 해야할지 고민.
    def __init__(self,in_features=2048,out_features=1000,bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.weights = [1,2,3,4,5,6]
        self.bias = [0] * in_features

    def init_weight(self,weight,bias):
        self.weight=weight
        self.bias =bias
    def linear(self,arr):
        batch = len(arr[0])
        output_list = [[0*self.out_features] for _ in range(batch)]
        for b in range(batch):
            for o in range(self.out_features):
                for i in range(self.in_features):
                    output_list[b][o] += arr[b][i]*self.weights[i] +self.bias[i]
        return output_list

    def __call__(self, arr) -> Any:
        return self.linear(arr)
