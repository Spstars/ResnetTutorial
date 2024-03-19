import math
from typing import Any
import torch.nn as nn
import torch
class softmax:
    def __init__(self) -> None:
        pass
    def __call__(self,input_feature) -> Any:
        return self.softmax1D(input_feature)
    
    def softmax1D(self,input_feature):
        batch,feature = len(input_feature),len(input_feature[0])
        max_input = [max(b) for b in input_feature]
        scaled_input_feature = [[math.exp(input_feature[i][j]-max_input[i]) for j in range(feature)] for i in range(batch)]
        
        return [[scaled_input_feature[i][j]/sum(scaled_input_feature[i]) for j in range(feature)] for i in range(batch)]
        


if __name__ == "__main__":
    softmax1 = nn.Softmax(dim=1)
    softmax2 = softmax()

    test = torch.randn(64,1000)
    test2= test.tolist()
    print(test2[0][:10])
    print(softmax1(test)[0][:10])
    print(softmax2(test2)[0][:10])
