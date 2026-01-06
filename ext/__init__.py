import cuda_transformer_ops_ext as _ext

def rmsnorm_fwd(x, weight, eps=1e-6):
    return _ext.rmsnorm_fwd(x, weight, eps)
