class xFuserLongContextAttention:  # only used when more than one GPU is in use
    def __init__(self, *a, **k): raise RuntimeError("multi-GPU not supported in the stub")
