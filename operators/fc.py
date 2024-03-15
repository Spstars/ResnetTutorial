
import numpy as np
    # (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
    # (fc): Linear(in_features=2048, out_features=1000, bias=True)

def fc(in_features=2048,out_features=100,bias=True):
    """
        avgpooling 한 값을 받아 1000의 classification을 수행한다.
    """
    weights = np.array([1,2,3,4,5,6])
    bias =0
    def linear(arr):
        return np.dot(arr,weights.T) +bias
    return linear
    
