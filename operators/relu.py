import numpy as np


def relu(input,inplace=False):
    """
        relu를 구현한다.
        일단 식은 이건데, batch나 channel 따라 어떻게 바뀔지 파악 해봐야겠다.

    """
    return np.maximum(0,input)


