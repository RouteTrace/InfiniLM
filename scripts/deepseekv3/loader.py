from operator import ge
from typing import List, Sequence

from sympy import true
from libinfinicore_infer_dpsk import (
    DeepseekV3MetaCStruct,
    DeepseekV3WeightsCStruct,
    KVCacheCStruct,
    DataType,
    DeviceType
)
import ctypes
from ctypes import POINTER, c_float, c_int, c_uint, c_void_p, byref
import os
from pathlib import Path
import safetensors
import sys
import time
import json
import math
import torch
import transformers

torch.set_default_device("cpu")

def load_all_safetensors_from_dir(dir_path_: str):
    tensors_ = {}
    dir_path_ = Path(dir_path_)
    for file in sorted(dir_path_.glob("*.safetensors")):
        data_ = safetensors.safe_open(file, "pt")
        for name_ in data_.keys():
            tensors_[name_] = data_.get_tensor(name_)
    return tensors_


class DeepseekV3AWQWeightsNaming:
    def input_embd(self):
        return "model.embed_tokens.weight"
    
    def output_norm(self):
        return "model.norm.weight"

    def output_embd(self):
        return "lm_head.weight"
    # --- MLA ---
    def attn_norm(self, i):
        return f"model.layers.{i}.input_layernorm.weight"

    def attn_q_a_proj_q(self, i):
        return f"model.layers.{i}.self_attn.q_a_proj.qweight" #  I32
    def attn_q_a_proj_z(self, i):
        return f"model.layers.{i}.self_attn.q_a_proj.qzeros" #  I32
    def attn_q_a_proj_s(self, i):
        return f"model.layers.{i}.self_attn.q_a_proj.scales"  #  F16
    def attn_q_b_proj_q(self, i):
        return f"model.layers.{i}.self_attn.q_b_proj.qweight"      # I32
    def attn_q_b_proj_z(self, i):
        return f"model.layers.{i}.self_attn.q_b_proj.qzeros"       # I32
    def attn_q_b_proj_s(self, i):
        return f"model.layers.{i}.self_attn.q_b_proj.scales"       # F16
        
    def attn_kv_a_proj_q(self, i):
        return f"model.layers.{i}.self_attn.kv_a_proj_with_mqa.qweight"  #  I32
    def attn_kv_a_proj_z(self, i):
        return f"model.layers.{i}.self_attn.kv_a_proj_with_mqa.qzeros"   #  I32
    def attn_kv_a_proj_s(self, i):
        return f"model.layers.{i}.self_attn.kv_a_proj_with_mqa.scales"   #  F16
    def attn_kv_b_proj_q(self, i):
        return f"model.layers.{i}.self_attn.kv_b_proj.qweight"     #  I32
    def attn_kv_b_proj_z(self, i):
        return f"model.layers.{i}.self_attn.kv_b_proj.qzeros"      #  I32
    def attn_kv_b_proj_s(self, i):
        return f"model.layers.{i}.self_attn.kv_b_proj.scales"      #  F16

    def attn_o_proj_q(self, i):
        return f"model.layers.{i}.self_attn.o_proj.qweight"        #  I32
    def attn_o_proj_z(self, i):
        return f"model.layers.{i}.self_attn.o_proj.qzeros"         #   I32
    def attn_o_proj_s(self, i):
        return f"model.layers.{i}.self_attn.o_proj.scales"         #  F16
    
    def q_a_layernorm(self, i):
        return f"model.layers.{i}.self_attn.q_a_layernorm.weight"
    def kv_a_layernorm(self, i):
        return f"model.layers.{i}.self_attn.kv_a_layernorm.weight"
    def post_attn_norm(self, i):
        return f"model.layers.{i}.post_attention_layernorm.weight"
    # --- MLP FFN ---
    def gate_q(self, i):
        return f"model.layers.{i}.mlp.gate_proj.qweight"           
    def gate_z(self, i):
        return f"model.layers.{i}.mlp.gate_proj.qzeros"            
    def gate_s(self, i):
        return f"model.layers.{i}.mlp.gate_proj.scales"            
    def up_q(self, i):
        return f"model.layers.{i}.mlp.up_proj.qweight"             
    def up_z(self, i):
        return f"model.layers.{i}.mlp.up_proj.qzeros"              
    def up_s(self, i):
        return f"model.layers.{i}.mlp.up_proj.scales"   
    def down_q(self, i):
        return f"model.layers.{i}.mlp.down_proj.qweight"           
    def down_z(self, i):
        return f"model.layers.{i}.mlp.down_proj.qzeros"            
    def down_s(self, i):
        return f"model.layers.{i}.mlp.down_proj.scales"            
    # --- MoE ---
    # shared_experts
    def shared_gate_q(self, i):
        return f"model.layers.{i}.mlp.shared_experts.gate_proj.qweight"  # [7168,256] I32
    def shared_gate_z(self, i):
        return f"model.layers.{i}.mlp.shared_experts.gate_proj.qzeros"   # [112,256] I32
    def shared_gate_s(self, i):
        return f"model.layers.{i}.mlp.shared_experts.gate_proj.scales"   # [112,2048] F16

    def shared_up_q(self, i):
        return f"model.layers.{i}.mlp.shared_experts.up_proj.qweight"    # [7168,256] I32
    def shared_up_z(self, i):
        return f"model.layers.{i}.mlp.shared_experts.up_proj.qzeros"     # [112,256] I32
    def shared_up_s(self, i):
        return f"model.layers.{i}.mlp.shared_experts.up_proj.scales"     # [112,2048] F16

    def shared_down_q(self, i):
        return f"model.layers.{i}.mlp.shared_experts.down_proj.qweight"  # [2048,896] I32
    def shared_down_z(self, i):
        return f"model.layers.{i}.mlp.shared_experts.down_proj.qzeros"   # [32,896]  I32
    def shared_down_s(self, i):
        return f"model.layers.{i}.mlp.shared_experts.down_proj.scales"   # [32,7168] F16
    # router_experts
    def expert_gate_q(self, i, e):
        return f"model.layers.{i}.mlp.experts.{e}.gate_proj.qweight"
    def expert_gate_z(self, i, e):
        return f"model.layers.{i}.mlp.experts.{e}.gate_proj.qzeros"
    def expert_gate_s(self, i, e):
        return f"model.layers.{i}.mlp.experts.{e}.gate_proj.scales"

    def expert_up_q(self, i, e):
        return f"model.layers.{i}.mlp.experts.{e}.up_proj.qweight"
    def expert_up_z(self, i, e):
        return f"model.layers.{i}.mlp.experts.{e}.up_proj.qzeros"
    def expert_up_s(self, i, e):
        return f"model.layers.{i}.mlp.experts.{e}.up_proj.scales"

    def expert_down_q(self, i, e):
        return f"model.layers.{i}.mlp.experts.{e}.down_proj.qweight"
    def expert_down_z(self, i, e):
        return f"model.layers.{i}.mlp.experts.{e}.down_proj.qzeros"
    def expert_down_s(self, i, e):
        return f"model.layers.{i}.mlp.experts.{e}.down_proj.scales"

    #router_gate/score
    def moe_gate_weight(self, i):
        return f"model.layers.{i}.mlp.gate.weight"           # [256,7168] BF16
    def moe_gate_bias(self, i):
        return f"model.layers.{i}.mlp.gate.e_score_correction_bias"  # [256] BF16

class ModelMetaFromDeepseek(DeepseekV3MetaCStruct):
    def __init__(self, config, dtype=torch.float16, max_tokens=None):
        if dtype == torch.float16:
            dt_ = DataType.INFINI_DTYPE_F16
        elif dtype == torch.float32:
            dt_ = DataType.INFINI_DTYPE_F32
        elif dtype == torch.bfloat16:
            dt_ = DataType.INFINI_DTYPE_BF16
        else:
            dt_ = DataType.INFINI_DTYPE_F16
        super().__init__(
            dt_logits=dt_,
            nlayer=config["num_hidden_layers"],
            d=config["hidden_size"],
            nh=config["num_attention_heads"],
            nkvh=config["num_key_value_heads"],
            di=config["intermediate_size"], # dense FFN size
            dctx=(
                config["max_position_embeddings"] if max_tokens is None else max_tokens
            ),
            dvoc=config["vocab_size"],
            epsilon=config["rms_norm_eps"],
            theta=config["rope_theta"],
            end_token=config["eos_token_id"], 
            # --- MLA --- 
            qk_rope_head_dim=config["qk_rope_head_dim"], # 64
            qk_nope_head_dim=config["qk_nope_head_dim"], # 128
            v_head_dim=config["v_head_dim"],
            qk_head_dim=config["qk_rope_head_dim"]+config["qk_nope_head_dim"], # 192
            # --- MoE ---
            moe_intermediate_size=config["moe_intermediate_size"], 
            num_experts_per_tok=config["num_experts_per_tok"],     
            n_routed_experts=config["n_routed_experts"],           
            routed_scaling_factor=config["routed_scaling_factor"],
            n_group=config["n_group"],
            topk_group=config["topk_group"],
            norm_topk_prob=config["norm_topk_prob"],
            # --- LORA ---
            q_lora_rank=config["q_lora_rank"],                     
            kv_lora_rank=config["kv_lora_rank"]                     
        )
        self.torch_dtype_logits = dtype
        self.quant_config = getattr(config, "quantization_config", None)
        self.quant_method = self.quant_config.get("quant_method", None)

class DeepseekV3WeightsImpl(DeepseekV3WeightsCStruct):
    def __init__(
        self,
        meta,
        naming,
        state_dict,
        ndev=1,
        transpose_weight=True,
    ):
        nlayer = meta.nlayer
        nh = meta.nh
        nkvh = meta.nkvh
        d = meta.d
        di = meta.di
        assert nh % nkvh == 0
        assert nh % ndev == 0
        assert nkvh % ndev == 0
        assert di % ndev == 0
        torch_dt_origin = meta.torch_dtype_logits
        if meta.quant_method=="AWQ":
            torch_dt_quant = torch.int32 # qweight/qzero
            torch_dt_scale = torch.float16
            self.dt_quant = DataType.INFINI_DTYPE_I32
            self.dt_scale = DataType.INFINI_DTYPE_F16

        if torch_dt_origin == torch.bfloat16:
            self.dt_origin = DataType.INFINI_DTYPE_BF16
        else:
            raise ValueError("Unsupported proj weight data type")

        self.transpose_linear_weights = 1 if transpose_weight else 0
        self.nlayer = nlayer
        self.nexpert = meta.n_routed_experts
        input_embd_naming = (
            naming.input_embd()
            if naming.input_embd() in state_dict
            else naming.output_embd()
        )
        output_embd_naming = (
            naming.output_embd()
            if naming.output_embd() in state_dict
            else naming.input_embd()
        )
        # input_embed
        self.input_embd_tensor = (
            state_dict[input_embd_naming].to(torch_dt_origin)
        )
        self.input_embd = self.input_embd_tensor.data_ptr()
        # output_norm
        self.output_norm_tensor = (
            state_dict[naming.output_norm()].to(torch_dt_origin)
        )
        self.output_norm = self.output_norm_tensor.data_ptr()
        self.output_embd_tensor = state_dict[output_embd_naming].to(torch_dt_origin)
        if not transpose_weight:
            self.output_embd_tensor = self.output_embd_tensor.transpose(
                0, 1
            ).contiguous()
        self.output_embd = self.output_embd_tensor.data_ptr()
        # attn_input_layernorm
        self.attn_input_layernorm =  self._get_linear_cptr(state_dict, naming, "attn_norm", torch_dt_origin)
        # post_attn_norm
        self.post_attn_norm =  self._get_linear_cptr(state_dict, naming, "post_attn_norm", torch_dt_origin)

        # attn_q_a_proj Q/Z/S
        self.attn_q_a_proj_qweight =  self._get_linear_cptr(state_dict, naming, "attn_q_a_proj_q", torch_dt_quant)
        self.attn_q_a_proj_qzeros =  self._get_linear_cptr(state_dict, naming, "attn_q_a_proj_z", torch_dt_quant)
        self.attn_q_a_proj_scales =  self._get_linear_cptr(state_dict, naming, "attn_q_a_proj_s", torch_dt_scale)

        # q_a_layernorm
        self.q_a_layernorm =  self._get_linear_cptr(state_dict, naming, "q_a_layernorm", torch_dt_origin)

        #attn_q_b_proj
        self.attn_q_b_proj_qweight=  self._get_linear_cptr(state_dict, naming, "attn_q_b_proj_q", torch_dt_quant)
        self.attn_q_b_proj_qzeros =  self._get_linear_cptr(state_dict, naming, "attn_q_b_proj_z", torch_dt_quant)
        self.attn_q_b_proj_scales =  self._get_linear_cptr(state_dict, naming, "attn_q_b_proj_s", torch_dt_scale)

        # attn_kv_a_proj
        self.attn_kv_a_proj_qweight = self._get_linear_cptr(state_dict, naming, "attn_kv_a_proj_q", torch_dt_quant)
        self.attn_kv_a_proj_qzeros = self._get_linear_cptr(state_dict, naming, "attn_kv_a_proj_z", torch_dt_quant)
        self.attn_kv_a_proj_scales = self._get_linear_cptr(state_dict, naming, "attn_kv_a_proj_s", torch_dt_scale)

        # attn_kv_b_proj
        self.attn_kv_b_proj_qweight = self._get_linear_cptr(state_dict, naming, "attn_kv_b_proj_q", torch_dt_quant)
        self.attn_kv_b_proj_qzeros = self._get_linear_cptr(state_dict, naming, "attn_kv_b_proj_z", torch_dt_quant)
        self.attn_kv_b_proj_scales = self._get_linear_cptr(state_dict, naming, "attn_kv_b_proj_s", torch_dt_scale)
        # kv_a_layernorm
        self.kv_a_layernorm = self._get_linear_cptr(state_dict, naming, "kv_a_layernorm", torch_dt_origin)

        # attn_o_proj
        self.attn_o_proj_qweight = self._get_linear_cptr(state_dict, naming, "attn_o_proj_q", torch_dt_quant)
        self.attn_o_proj_qzeros = self._get_linear_cptr(state_dict, naming, "attn_o_proj_z", torch_dt_quant)
        self.attn_o_proj_scales = self._get_linear_cptr(state_dict, naming, "attn_o_proj_s", torch_dt_scale)

        # MLP
        self.ffn_gate_proj_qweight = self._get_linear_cptr(state_dict, naming, "gate_q", torch_dt_quant)
        self.ffn_gate_proj_qzeros = self._get_linear_cptr(state_dict, naming, "gate_z", torch_dt_quant)
        self.ffn_gate_proj_scales = self._get_linear_cptr(state_dict, naming, "gate_s", torch_dt_scale)
        self.ffn_up_proj_qweight = self._get_linear_cptr(state_dict, naming, "up_q", torch_dt_quant)
        self.ffn_up_proj_qzeros = self._get_linear_cptr(state_dict, naming, "up_z", torch_dt_quant)
        self.ffn_up_proj_scales = self._get_linear_cptr(state_dict, naming, "up_s", torch_dt_scale)
        self.ffn_down_proj_qweight = self._get_linear_cptr(state_dict, naming, "down_q", torch_dt_quant)
        self.ffn_down_proj_qzeros = self._get_linear_cptr(state_dict, naming, "down_z", torch_dt_quant)
        self.ffn_down_proj_scales = self._get_linear_cptr(state_dict, naming, "down_s", torch_dt_scale)
        
        # Moe
        self.moe_gate_weight = self._get_linear_cptr(state_dict, naming, "moe_gate_weight", torch_dt_origin)
        self.moe_gate_bias = self._get_linear_cptr(state_dict, naming, "moe_gate_bias", torch_dt_origin)

        # Shared experts
        self.moe_shared_gate_proj_qweight = self._get_linear_cptr(state_dict, naming, "shared_gate_q", torch_dt_quant)
        self.moe_shared_gate_proj_qzeros = self._get_linear_cptr(state_dict, naming, "shared_gate_z", torch_dt_quant)
        self.moe_shared_gate_proj_scales = self._get_linear_cptr(state_dict, naming, "shared_gate_s", torch_dt_scale)

        self.moe_shared_up_proj_qweight = self._get_linear_cptr(state_dict, naming, "shared_up_q", torch_dt_quant)
        self.moe_shared_up_proj_qzeros = self._get_linear_cptr(state_dict, naming, "shared_up_z", torch_dt_quant)
        self.moe_shared_up_proj_scales = self._get_linear_cptr(state_dict, naming, "shared_up_s", torch_dt_scale)

        self.moe_shared_down_proj_qweight = self._get_linear_cptr(state_dict, naming, "shared_down_q", torch_dt_quant)
        self.moe_shared_down_proj_qzeros = self._get_linear_cptr(state_dict, naming, "shared_down_z", torch_dt_quant)
        self.moe_shared_down_proj_scales = self._get_linear_cptr(state_dict, naming, "shared_down_s", torch_dt_scale)

        # Router experts
        self.moe_expert_gate_proj_qweight = self._get_moe_cptr(state_dict, naming, "expert_gate_q", torch_dt_quant)
        self.moe_expert_gate_proj_qzeros = self._get_moe_cptr(state_dict, naming, "expert_gate_z", torch_dt_quant)
        self.moe_expert_gate_proj_scales = self._get_moe_cptr(state_dict, naming, "expert_gate_s", torch_dt_scale)

        self.moe_expert_up_proj_qweight = self._get_moe_cptr(state_dict, naming, "expert_up_q", torch_dt_quant)
        self.moe_expert_up_proj_qzeros = self._get_moe_cptr(state_dict, naming, "expert_up_z", torch_dt_quant)
        self.moe_expert_up_proj_scales = self._get_moe_cptr(state_dict, naming, "expert_up_s", torch_dt_scale)

        self.moe_expert_down_proj_qweight = self._get_moe_cptr(state_dict, naming, "expert_down_q", torch_dt_quant)
        self.moe_expert_down_proj_qzeros = self._get_moe_cptr(state_dict, naming, "expert_down_z", torch_dt_quant)
        self.moe_expert_down_proj_scales = self._get_moe_cptr(state_dict, naming, "expert_down_s", torch_dt_scale)


    def _get_linear_cptr(self, param_dict, naming, weight_str, data_type):
        get_prefix_func = getattr(naming, weight_str)
        target_tensors = [param_dict[get_prefix_func(i)].to(data_type) for i in range(self.nlayer)]
        target_ptrs = [target_tensors[i].data_ptr() for i in range(self.nlayer)]
        c_ptr = (c_void_p * self.nlayer)(*target_ptrs)
        return c_ptr

    def _get_moe_cptr(self, param_dict, naming, weight_str, data_type):
        layer_ptrs = []
        get_prefix_func = getattr(naming, weight_str)
        for i in range(self.nlayer):
            expert_ptrs = []
            for e in range(self.nexpert):
                # 获取特定层和特定专家的张量
                
                # 5. 获取张量并转换数据类型，然后获取数据指针
                tensor = param_dict[get_prefix_func(i, e)].to(data_type)
                expert_ptrs.append(tensor.data_ptr())
            c_expert_ptr = (c_void_p * self.nexpert)(*expert_ptrs)
            layer_ptrs.append(ctypes.addressof(c_expert_ptr))
            
        # POINTER(POINTER(c_void_p))
        c_ptr = (c_void_p * self.nlayer)(*layer_ptrs)
        return c_ptr
