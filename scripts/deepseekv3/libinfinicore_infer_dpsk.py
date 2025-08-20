import ctypes
from ctypes import c_size_t, c_uint, c_int, c_float, c_void_p, POINTER
import os


class DataType(ctypes.c_int):
    INFINI_DTYPE_INVALID = 0
    INFINI_DTYPE_BYTE = 1
    INFINI_DTYPE_BOOL = 2
    INFINI_DTYPE_I8 = 3
    INFINI_DTYPE_I16 = 4
    INFINI_DTYPE_I32 = 5
    INFINI_DTYPE_I64 = 6
    INFINI_DTYPE_U8 = 7
    INFINI_DTYPE_U16 = 8
    INFINI_DTYPE_U32 = 9
    INFINI_DTYPE_U64 = 10
    INFINI_DTYPE_F8 = 11
    INFINI_DTYPE_F16 = 12
    INFINI_DTYPE_F32 = 13
    INFINI_DTYPE_F64 = 14
    INFINI_DTYPE_C16 = 15
    INFINI_DTYPE_C32 = 16
    INFINI_DTYPE_C64 = 17
    INFINI_DTYPE_C128 = 18
    INFINI_DTYPE_BF16 = 19


class DeviceType(ctypes.c_int):
    DEVICE_TYPE_CPU = 0
    DEVICE_TYPE_NVIDIA = 1
    DEVICE_TYPE_CAMBRICON = 2
    DEVICE_TYPE_ASCEND = 3
    DEVICE_TYPE_METAX = 4
    DEVICE_TYPE_MOORE = 5
    DEVICE_TYPE_ILUVATAR = 6


class DeepseekV3MetaCStruct(ctypes.Structure):
    _fields_ = [
        ("dt_logits", DataType),
        ("nlayer", c_size_t),
        ("d", c_size_t),
        ("nh", c_size_t),
        ("nkvh", c_size_t),
        ("di", c_size_t),
        ("dctx", c_size_t),
        ("dvoc", c_size_t),
        ("epsilon", c_float),
        ("theta", c_float),
        ("end_token", c_uint),
        
        # Attention mechanism parameters
        ("qk_rope_head_dim", c_size_t),
        ("qk_nope_head_dim", c_size_t),
        ("v_head_dim", c_size_t),
        ("qk_head_dim", c_size_t),
        
        # MoE (Mixture of Experts) parameters
        ("moe_intermediate_size", c_size_t),
        ("num_experts_per_tok", c_size_t),
        ("n_routed_experts", c_size_t),
        ("routed_scaling_factor", c_float),
        ("n_group", c_size_t),
        ("topk_group", c_size_t),
        ("norm_topk_prob", c_int),
        
        # LoRA parameters
        ("q_lora_rank", c_size_t),
        ("kv_lora_rank", c_size_t),
    ]

# 1. 创建对应 QuantizedLinearWeights 的辅助类
class QuantizedLinearWeights(ctypes.Structure):
    """
    对应C语言中的 QuantizedLinearWeights 结构体。
    用于封装一个AWQ量化线性层的三个权重组件。
    """
    _fields_ = [
        ("qweight", c_void_p),
        ("qzeros", c_void_p),
        ("scales", c_void_p),
    ]

# 2. 创建主类，对应 DeepseekV3AWQWeights
class DeepseekV3WeightsCStruct(ctypes.Structure):
    """
    对应C语言中的 DeepseekV3AWQWeights 主结构体。
    """
    _fields_ = [
        # --- 元数据 ---
        ("nlayer", c_size_t),
        ("nexpert", c_size_t),
        ("dt_non_quant", DataType),
        ("dt_quant", DataType),
        ("dt_scale", DataType),
        ("transpose_linear_weights", c_int),

        # --- 顶层权重 (通常非量化) ---
        ("input_embd", c_void_p),
        ("output_norm", c_void_p),
        ("output_embd", c_void_p),

        # --- 每层的权重 ---
        
        # -- Attention 模块 --
        # C type: const void *const *attn_norm;
        # Python type: POINTER(c_void_p) 表示一个 c_void_p 的数组
        ("attn_norm", POINTER(c_void_p)),
        
        # C type: const QuantizedLinearWeights *const *attn_q_a_proj;
        # Python type: POINTER(QuantizedLinearWeights) 表示一个 QuantizedLinearWeights 结构体的数组
        ("attn_q_a_proj", POINTER(QuantizedLinearWeights)),
        ("attn_q_b_proj", POINTER(QuantizedLinearWeights)),
        ("attn_kv_a_proj", POINTER(QuantizedLinearWeights)),
        ("attn_kv_b_proj", POINTER(QuantizedLinearWeights)),
        ("attn_o_proj", POINTER(QuantizedLinearWeights)),

        # -- FFN/MoE 模块 --
        ("post_attn_norm", POINTER(c_void_p)),

        # -- 对于非MoE层 (Dense FFN) --
        ("ffn_gate_proj", POINTER(QuantizedLinearWeights)),
        ("ffn_up_proj", POINTER(QuantizedLinearWeights)),
        ("ffn_down_proj", POINTER(QuantizedLinearWeights)),

        # -- 对于MoE层 --
        ("moe_gate_weight", POINTER(c_void_p)),
        ("moe_gate_bias", POINTER(c_void_p)),
        
        ("moe_shared_gate_proj", POINTER(QuantizedLinearWeights)),
        ("moe_shared_up_proj", POINTER(QuantizedLinearWeights)),
        ("moe_shared_down_proj", POINTER(QuantizedLinearWeights)),
        
        # -- Routed Experts --
        # C type: const QuantizedLinearWeights *const *const *moe_expert_gate_proj;
        # 这是一个指向 [QuantizedLinearWeights*] 数组的指针，所以是两级指针
        # Python type: POINTER(POINTER(QuantizedLinearWeights))
        ("moe_expert_gate_proj", POINTER(POINTER(QuantizedLinearWeights))),
        ("moe_expert_up_proj", POINTER(POINTER(QuantizedLinearWeights))),
        ("moe_expert_down_proj", POINTER(POINTER(QuantizedLinearWeights))),
    ]


class DeepseekV3ModelCSruct(ctypes.Structure):
    pass


class KVCacheCStruct(ctypes.Structure):
    pass


def __open_library__():
    lib_path = os.path.join(
        os.environ.get("INFINI_ROOT"), "lib", "libinfinicore_infer.so"
    )
    lib = ctypes.CDLL(lib_path)
    lib.createDeepseekV3Model.restype = POINTER(DeepseekV3ModelCSruct)
    lib.createDeepseekV3Model.argtypes = [
        POINTER(DeepseekV3MetaCStruct),  # DeepseekV3Meta const *
        POINTER(DeepseekV3WeightsCStruct),  # DeepseekV3Weights const *
        DeviceType,  # DeviceType
        c_int,  # int ndev
        POINTER(c_int),  # int const *dev_ids
    ]
    lib.destroyDeepseekV3Model.argtypes = [POINTER(DeepseekV3ModelCSruct)]
    lib.createKVCache.argtypes = [POINTER(DeepseekV3ModelCSruct)]
    lib.createKVCache.restype = POINTER(KVCacheCStruct)
    lib.dropKVCache.argtypes = [POINTER(DeepseekV3ModelCSruct), POINTER(KVCacheCStruct)]
    lib.inferBatch.restype = None
    lib.inferBatch.argtypes = [
        POINTER(DeepseekV3ModelCSruct),  # struct DeepseekV3Model const *
        POINTER(c_uint),  # unsigned int const *tokens
        c_uint,  # unsigned int ntok
        POINTER(c_uint),  # unsigned int const *req_lens
        c_uint,  # unsigned int nreq
        POINTER(c_uint),  # unsigned int const *req_pos
        POINTER(POINTER(KVCacheCStruct)),  # struct KVCache **kv_caches
        POINTER(c_float),  # float temperature
        POINTER(c_uint),  # unsigned int topk
        POINTER(c_float),  # float topp
        POINTER(c_uint),  # unsigned int *output
    ]
    lib.forwardBatch.restype = None
    lib.forwardBatch.argtypes = [
        POINTER(DeepseekV3ModelCSruct),  # struct DeepseekV3Model const *
        POINTER(c_uint),  # unsigned int const *tokens
        c_uint,  # unsigned int ntok
        POINTER(c_uint),  # unsigned int const *req_lens
        c_uint,  # unsigned int nreq
        POINTER(c_uint),  # unsigned int const *req_pos
        POINTER(POINTER(KVCacheCStruct)),  # struct KVCache **kv_caches
        c_void_p,  # void *logits
    ]

    return lib


LIB = __open_library__()

create_deepseekv3_model = LIB.createDeepseekV3Model
destroy_deepseekv3_model = LIB.destroyDeepseekV3Model
create_kv_cache = LIB.createKVCache
drop_kv_cache = LIB.dropKVCache
infer_batch = LIB.inferBatch
forward_batch = LIB.forwardBatch
