
    # (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
    # (fc): Linear(in_features=2048, out_features=1000, bias=True)

from typing import Any
import numpy as np
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
        self.weights = [[1 for _ in range(in_features)] for _ in range(out_features)]
        self.bias = [0] * out_features

    def init_weight(self,weight,bias):
        self.weights= weight.tolist()
        self.bias  =bias.tolist()
        
    def linear(self,arr):

        arr = np.array(arr)
        weights = np.array(self.weights)
        bias = np.array(self.bias)
        return np.dot(arr, weights.T) + bias

    def __call__(self, arr) -> Any:
        return self.linear(arr)
if __name__ == "__main__" :

    fc = fc()
    X = [[2 for _ in range(2048)] for _ in range(1)]
    # print(len(fc.weights))

    k= fc(X)
    print(len(k))