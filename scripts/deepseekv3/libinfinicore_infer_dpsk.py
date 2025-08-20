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


class DeepseekV3WeightsCStruct(ctypes.Structure):
    """
    对应C语言中扁平化设计的 DeepseekV3 AWQ 权重结构体。
    所有量化层的权重组件 (qweight, qzeros, scales) 都被展开。
    """
    _fields_ = [
        # --- 元数据 ---
        ("nlayer", c_size_t),
        ("dt_origin", DataType),
        ("dt_quant", DataType),
        ("dt_scale", DataType),
        ("transpose_linear_weights", c_int),

        # --- 顶层权重 (通常非量化) ---
        ("input_embd", c_void_p),
        ("output_norm", c_void_p),
        ("output_embd", c_void_p),

        # --- 每层的权重 ---
        
        # -- Attention 模块 --
        ("attn_input_layernorm", POINTER(c_void_p)),
        
        # attn_q_a_proj (展开)
        ("attn_q_a_proj_qweight", POINTER(c_void_p)),
        ("attn_q_a_proj_qzeros", POINTER(c_void_p)),
        ("attn_q_a_proj_scales", POINTER(c_void_p)),
        
        # attn_q_b_proj (展开)
        ("attn_q_b_proj_qweight", POINTER(c_void_p)),
        ("attn_q_b_proj_qzeros", POINTER(c_void_p)),
        ("attn_q_b_proj_scales", POINTER(c_void_p)),

        # attn_kv_a_proj (展开)
        ("attn_kv_a_proj_qweight", POINTER(c_void_p)),
        ("attn_kv_a_proj_qzeros", POINTER(c_void_p)),
        ("attn_kv_a_proj_scales", POINTER(c_void_p)),
        
        # attn_kv_b_proj (展开)
        ("attn_kv_b_proj_qweight", POINTER(c_void_p)),
        ("attn_kv_b_proj_qzeros", POINTER(c_void_p)),
        ("attn_kv_b_proj_scales", POINTER(c_void_p)),

        # attn_o_proj (展开)
        ("attn_o_proj_qweight", POINTER(c_void_p)),
        ("attn_o_proj_qzeros", POINTER(c_void_p)),
        ("attn_o_proj_scales", POINTER(c_void_p)),

        ("q_a_layernorm", POINTER(c_void_p)),
        ("kv_a_layernorm", POINTER(c_void_p)),
        ("post_attn_norm", POINTER(c_void_p)),

        # -- FFN/MoE 模块 --
        # -- 对于非MoE层 (Dense FFN) --
        ("ffn_gate_proj_qweight", POINTER(c_void_p)),
        ("ffn_gate_proj_qzeros", POINTER(c_void_p)),
        ("ffn_gate_proj_scales", POINTER(c_void_p)),
        
        ("ffn_up_proj_qweight", POINTER(c_void_p)),
        ("ffn_up_proj_qzeros", POINTER(c_void_p)),
        ("ffn_up_proj_scales", POINTER(c_void_p)),

        ("ffn_down_proj_qweight", POINTER(c_void_p)),
        ("ffn_down_proj_qzeros", POINTER(c_void_p)),
        ("ffn_down_proj_scales", POINTER(c_void_p)),

        # -- 对于MoE层 --
        ("moe_gate_weight", POINTER(c_void_p)),
        ("moe_gate_bias", POINTER(c_void_p)),
        
        # Shared Experts (展开)
        ("moe_shared_gate_proj_qweight", POINTER(c_void_p)),
        ("moe_shared_gate_proj_qzeros", POINTER(c_void_p)),
        ("moe_shared_gate_proj_scales", POINTER(c_void_p)),
        
        ("moe_shared_up_proj_qweight", POINTER(c_void_p)),
        ("moe_shared_up_proj_qzeros", POINTER(c_void_p)),
        ("moe_shared_up_proj_scales", POINTER(c_void_p)),
        
        ("moe_shared_down_proj_qweight", POINTER(c_void_p)),
        ("moe_shared_down_proj_qzeros", POINTER(c_void_p)),
        ("moe_shared_down_proj_scales", POINTER(c_void_p)),
        
        # -- Routed Experts (展开) --
        # 每个组件现在都是一个二级指针数组
        ("moe_expert_gate_proj_qweight", POINTER(POINTER(c_void_p))),
        ("moe_expert_gate_proj_qzeros", POINTER(POINTER(c_void_p))),
        ("moe_expert_gate_proj_scales", POINTER(POINTER(c_void_p))),
        
        ("moe_expert_up_proj_qweight", POINTER(POINTER(c_void_p))),
        ("moe_expert_up_proj_qzeros", POINTER(POINTER(c_void_p))),
        ("moe_expert_up_proj_scales", POINTER(POINTER(c_void_p))),
        
        ("moe_expert_down_proj_qweight", POINTER(POINTER(c_void_p))),
        ("moe_expert_down_proj_qzeros", POINTER(POINTER(c_void_p))),
        ("moe_expert_down_proj_scales", POINTER(POINTER(c_void_p))),
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
