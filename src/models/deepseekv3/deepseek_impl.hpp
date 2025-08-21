#ifndef DEEPSEEKV3_IMPL_H
#define DEEPSEEKV3_IMPL_H

#include "infinicore_infer.h"

#include "../../allocator.hpp"
#include "../../tensor.hpp"

#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

struct DeviceResource {
    // Device
    infiniDevice_t device;
    int device_id;
    infiniopHandle_t handle;
    // Weights (tensors)
    std::shared_ptr<Tensor> w_in_embd, w_out_norm, w_out_embd, sin_table, cos_table;
    // 2D (nlayers, tensors)
    std::vector<std::shared_ptr<Tensor>> w_attn_input_norm, w_post_attn_norm, q_a_layernorm, kv_a_layernorm, 
        w_attn_q_a_proj_qweight, w_attn_q_a_proj_qzeros, w_attn_q_a_proj_scales,
        w_attn_q_b_proj_qweight, w_attn_q_b_proj_qzeros, w_attn_q_b_proj_scales,
        w_attn_kv_a_proj_qweight, w_attn_kv_a_proj_qzeros, w_attn_kv_a_proj_scales,
        w_attn_kv_b_proj_qweight, w_attn_kv_b_proj_qzeros, w_attn_kv_b_proj_scales,
        w_attn_o_proj_qweight, w_attn_o_proj_qzeros, w_attn_o_proj_scales,
        w_ffn_gate_proj_qweight, w_ffn_gate_proj_qzeros, w_ffn_gate_proj_scales,
        w_ffn_up_proj_qweight, w_ffn_up_proj_qzeros, w_ffn_up_proj_scales,
        w_ffn_down_proj_qweight, w_ffn_down_proj_qzeros, w_ffn_down_proj_scales,
        w_moe_gate_weight, moe_gate_bias,
        w_moe_shared_gate_proj_qweight, w_moe_shared_gate_proj_qzeros, w_moe_shared_gate_proj_scales,
        w_moe_shared_up_proj_qweight, w_moe_shared_up_proj_qzeros, w_moe_shared_up_proj_scales,
        w_moe_shared_down_proj_qweight, w_moe_shared_down_proj_qzeros, w_moe_shared_down_proj_scales;

    // 3D --> (nlayers, nexperts, tensors)
    std::vector<std::vector<std::shared_ptr<Tensor>>> \ 
        moe_expert_gate_proj_qweight, moe_expert_gate_proj_qzeros, moe_expert_gate_proj_scales,
        moe_expert_up_proj_qweight, moe_expert_up_proj_qzeros, moe_expert_up_proj_scales,
        moe_expert_down_proj_qweight, moe_expert_down_proj_qzeros, moe_expert_down_proj_scales;


    w_ffn_gate_up, w_ffn_down;
    // Streams
    infinirtStream_t stream;
    // Communicator
    infinicclComm_t comm;

    std::shared_ptr<MemoryPool> memory_pool;
};

struct InferState {
    std::mutex mtx;
    std::condition_variable cv_load, cv_start, cv_done;
    bool loaded = false;
    bool proceed = false;
    bool exit_flag = false;
};

struct InferRequest {
    const uint32_t *tokens;
    uint32_t ntok;
    const uint32_t *req_lens;
    uint32_t nreq;
    const uint32_t *req_pos;
    struct KVCache **kv_caches;
    const float *temperature;
    const uint32_t *topk;
    const float *topp;
    uint32_t *output;
    void *logits;
};

struct DeepseekV3Model {
    DeepseekV3Meta meta;
    infiniDevice_t device;
    std::vector<int> dev_ids;
    std::vector<DeviceResource> dev_resources;
    std::vector<InferState> states;
    std::vector<std::thread> threads;
    InferRequest req;

    DeepseekV3Model(const DeepseekV3Meta *, const DeepseekV3Weights *, infiniDevice_t device, std::vector<int> device_ids);
};

struct KVCache {
    std::vector<std::vector<std::shared_ptr<Tensor>>> k, v;
};

#endif
