
from typing import Any


class relu:
    """
        relu를 구현한다. numpy로 작성하면 np.maximum(0,input,), batch나 channel 따라 어떻게 바뀔지 파악 해봐야겠다.
        inplace = False의 경우 기존 input값을 cache에 저장한다. inference만 할 것 같아 true로 바꾸고, 기본값을 None반환

    """
    def __init__(self,inplace=True):
        pass
    def Relu(self,input_feature):
        batch,channel,length = len(input_feature),len(input_feature[0]),len(input_feature[0][0])

        for b in range(batch):
            for c in range(channel):
                for h in range(length):
                    for w in range(length):
                        input_feature[b][c][h][w] = max(0,input_feature[b][c][h][w])

        return input_feature


    def __call__(self, input_feature) -> Any:
        return self.Relu(input_feature)


