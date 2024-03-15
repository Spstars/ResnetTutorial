#(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
#저희 모델의 경우 추론 위주로 할 것 같으니, track_running_stats을 false로 한다.(training까지 하면 cache 같은 변수로 저장해야한다.)
def batchNorm2D(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False, device=None, dtype=None):
    """
    그림에는 안나와있지만, paper에 수행한다고 나와있다.
    
    
    """
    print("asd")