def get_sequence_parallel_world_size(): return 1
def get_sequence_parallel_rank(): return 0
def get_sp_group(): raise RuntimeError("multi-GPU not supported in the stub")
def get_world_group(): raise RuntimeError("multi-GPU not supported in the stub")
def init_distributed_environment(*a, **k): pass
def initialize_model_parallel(*a, **k): pass
