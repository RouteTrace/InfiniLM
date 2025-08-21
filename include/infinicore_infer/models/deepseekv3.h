#ifndef MODEL_DEEPSEEKV3_H
#define MODEL_DEEPSEEKV3_H

#include <infiniccl.h>
#include <infiniop.h>
#include <infinirt.h>

#include <stdint.h>

struct DeepseekV3Model;

typedef struct
{
    infiniDtype_t dt_logits;
    size_t nlayer, d, nh, nkvh, di, dctx, dvoc;
    float epsilon, theta;
    uint32_t end_token;
    
    // Attention mechanism parameters
    size_t qk_rope_head_dim;
    size_t qk_nope_head_dim;
    size_t v_head_dim;
    size_t qk_head_dim;
    
    // MoE (Mixture of Experts) parameters
    size_t moe_intermediate_size;
    size_t num_experts_per_tok;
    size_t n_routed_experts;
    float routed_scaling_factor;
    size_t n_group;
    size_t topk_group;
    int norm_topk_prob; // bool or int ?
    
    // LoRA parameters
    size_t q_lora_rank;
    size_t kv_lora_rank;
} DeepseekV3Meta;


typedef struct
{

    size_t nlayer;
    size_t nexpert; 
    infiniDtype_t dt_origin; 
    infiniDtype_t dt_quant;     // qweight和qzeros
    infiniDtype_t dt_scale;     // scales
    int transpose_linear_weights;

    // [dvoc, d]
    const void *input_embd;
    // [d]
    const void *output_norm;
    // [dvoc, d]
    const void *output_embd;

    // -- Attention --
    const void *const *attn_input_layernorm; // nlayer * [d]

    const void *const *attn_q_a_proj_qweight;   // nlayer * 
    const void *const *attn_q_a_proj_qzeros;   // nlayer * 
    const void *const *attn_q_a_proj_scales;   // nlayer * 

    const void *const *q_a_layernorm;   // nlayer * 

    const void *const *attn_q_b_proj_qweight;   // nlayer * 
    const void *const *attn_q_b_proj_qzeros;   // nlayer * 
    const void *const *attn_q_b_proj_scales; 

    const void *const *attn_kv_a_proj_qweight;  // nlayer * 
    const void *const *attn_kv_a_proj_qzeros;  // nlayer * 
    const void *const *attn_kv_a_proj_scales;  // nlayer * 

    const void *const *attn_kv_b_proj_qweight;  // nlayer * 
    const void *const *attn_kv_b_proj_qzeros;  // nlayer * 
    const void *const *attn_kv_b_proj_scales;  // nlayer * 

    const void *const *kv_a_layernorm;  // nlayer * 

    const void *const *attn_o_proj_qweight;     // nlayer * 
    const void *const *attn_o_proj_qzeros;     // nlayer * 
    const void *const *attn_o_proj_scales;     // nlayer * 
    
    const void *const *post_attn_norm; // nlayer * [d]

    // -- (Dense FFN) --
    const void *const *ffn_gate_proj_qweight;   // nlayer * 
    const void *const *ffn_gate_proj_qzeros;   // nlayer * 
    const void *const *ffn_gate_proj_scales;   // nlayer * 

    const void *const *ffn_up_proj_qweight;     // nlayer * 
    const void *const *ffn_up_proj_qzeros;     // nlayer * 
    const void *const *ffn_up_proj_scales;     // nlayer * 

    const void *const *ffn_down_proj_qweight;   // nlayer * 
    const void *const *ffn_down_proj_qzeros;   // nlayer * 
    const void *const *ffn_down_proj_scales;   // nlayer * 

    // -- MoE --
    // Router 
    const void *const *moe_gate_weight; // nlayer * [nexpert, d]
    const void *const *moe_gate_bias;   // nlayer * [nexpert]
    
    // Shared Experts
    const void *const *moe_shared_gate_proj_qweight; // nlayer * [7168, 256]
    const void *const *moe_shared_gate_proj_qzeros; // nlayer * [112, 256]
    const void *const *moe_shared_gate_proj_scales; // nlayer * [112, 2 048]

    const void *const *moe_shared_up_proj_qweight; // nlayer * [7168, 256]
    const void *const *moe_shared_up_proj_qzeros; // nlayer * [112, 256]
    const void *const *moe_shared_up_proj_scales; // nlayer * [112, 2 048]

    const void *const *moe_shared_down_proj_qweight; // nlayer * [7168, 256]
    const void *const *moe_shared_down_proj_qzeros; // nlayer * [112, 256]
    const void *const *moe_shared_down_proj_scales; // nlayer * [112, 2 048]

    // Routed Experts (nlayer * nexpert * [] )
    // 这是一个三维指针: [层][专家][权重组件]
    const void *const *const *moe_expert_gate_proj_qweight;
    const void *const *const *moe_expert_gate_proj_qzeros;
    const void *const *const *moe_expert_gate_proj_scales;

    const void *const *const *moe_expert_up_proj_qweight;
    const void *const *const *moe_expert_up_proj_qzeros;
    const void *const *const *moe_expert_up_proj_scales;

    const void *const *const *moe_expert_down_proj_qweight;
    const void *const *const *moe_expert_down_proj_qzeros;
    const void *const *const *moe_expert_down_proj_scales;

} DeepseekV3Weights;
//////////////////// APIs ///////////////////////
/// @brief 创建模型
/// @param device 协处理器种类
/// @param ndev 协处理器数量
/// @param dev_ids 协处理器编号，长度为 ndev
__C __export struct DeepseekV3Model *
createDeepseekV3Model(const DeepseekV3Meta *,
                 const DeepseekV3Weights *,
                 infiniDevice_t device,
                 int ndev,
                 const int *dev_ids);

/// @brief 销毁模型
__C __export void
destroyDeepseekV3Model(struct DeepseekV3Model *);

/// @brief 创建 KV Cache
__C __export struct KVCache *
createKVCache(const struct DeepseekV3Model *);

/// @brief 复制 KV Cache
__C __export struct KVCache *
duplicateKVCache(const struct DeepseekV3Model *,
                 const struct KVCache *, uint32_t seq_len);

/// @brief 销毁 KV Cache
__C __export void
dropKVCache(const struct DeepseekV3Model *,
            struct KVCache *);

/// @brief 批次推理一轮，并采样出新的 token
/// @param tokens 输入 token 地址
/// @param ntok 输入 token 数量
/// @param nreq 请求数量
/// @param req_lens 每个请求的 token 数量
/// @param req_pos 每个请求的起始位置
/// @param kv_caches 每个请求的 KV Cache
/// @param temperature 采样温度（0. 表示贪心采样）
/// @param topk 采样 topk（1 表示贪心采样）
/// @param topp 采样 topp
/// @param output 输出 token 数组，每个请求一个输出，长度至少为nreq
__C __export void
inferBatch(struct DeepseekV3Model *,
           const uint32_t *tokens, uint32_t ntok,
           const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
           struct KVCache **kv_caches,
           const float *temperature, const uint32_t *topk, const float *topp,
           uint32_t *output);

/// @brief 批次推理一轮，输出 output embedding 后的 logits
/// @param tokens 输入 token 地址
/// @param ntok 输入 token 数量
/// @param nreq 请求数量
/// @param req_lens 每个请求的 token 数量
/// @param req_pos 每个请求的起始位置
/// @param kv_caches 每个请求的 KV Cache
/// @param logits 输出 token 数组，每个请求一个输出，长度至少为nreq
__C __export void
forwardBatch(struct DeepseekV3Model *,
             const uint32_t *tokens, uint32_t ntok,
             const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
             struct KVCache **kv_caches,
             void *logits);

#endif
