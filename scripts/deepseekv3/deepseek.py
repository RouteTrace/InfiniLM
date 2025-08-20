from typing import List, Sequence

from sympy import true
from libinfinicore_infer_dpsk import (
    DeepseekV3MetaCStruct,
    DeepseekV3WeightsCStruct,
    KVCacheCStruct,
    DataType,
    DeviceType,
    create_deepseekv3_model,
    destroy_deepseekv3_model,
    create_kv_cache,
    drop_kv_cache,
    infer_batch,
    forward_batch,
)
from infer_task import InferTask, KVCache

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
    def gate_weight(self, i):
        return f"model.layers.{i}.mlp.gate.weight"           # [256,7168] BF16
    def gate_bias(self, i):
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

class DeepseekV3WeightsImpl(DeepseekV3WeightsCStruct):
    def __init__(
        self,
        meta,
        naming,
        state_dict,
        torch_dt_mat=torch.float16,
        torch_dt_norm=torch.float32,
        ndev=1,
        transpose_weight=True,
    ):
        nlayer = meta.nlayer
        nh = meta.nh
        nkvh = meta.nkvh
        dh = meta.dh
        d = meta.d
        di = meta.di
        scale_input = meta.scale_input
        scale_output = meta.scale_output
        scale_o = meta.scale_o
        scale_down = meta.scale_down
        assert nh % nkvh == 0
        assert nh % ndev == 0
        assert nkvh % ndev == 0
        assert di % ndev == 0
        torch_dt_logits = meta.torch_dtype_logits
        if torch_dt_mat == torch.float16:
            self.dt_mat = DataType.INFINI_DTYPE_F16
        elif torch_dt_mat == torch.float32:
            self.dt_mat = DataType.INFINI_DTYPE_F32
        elif torch_dt_mat == torch.bfloat16:
            self.dt_mat = DataType.INFINI_DTYPE_BF16
        else:
            raise ValueError("Unsupported proj weight data type")
        if torch_dt_norm == torch.float16:
            self.dt_norm = DataType.INFINI_DTYPE_F16
        elif torch_dt_norm == torch.float32:
            self.dt_norm = DataType.INFINI_DTYPE_F32
        elif torch_dt_norm == torch.bfloat16:
            self.dt_norm = DataType.INFINI_DTYPE_BF16
        else:
            raise ValueError("Unsupported norm weight data type")

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
        self.transpose_linear_weights = 1 if transpose_weight else 0
        self.nlayer = nlayer
        self.input_embd_tensor = (
            state_dict[input_embd_naming].to(torch_dt_logits) * scale_input
        )
        self.input_embd = self.input_embd_tensor.data_ptr()
        self.output_norm_tensor = (
            state_dict[naming.output_norm()].to(torch_dt_norm) * scale_output
        )
        self.output_norm = self.output_norm_tensor.data_ptr()
        self.output_embd_tensor = state_dict[output_embd_naming].to(torch_dt_mat)
        if not transpose_weight:
            self.output_embd_tensor = self.output_embd_tensor.transpose(
                0, 1
            ).contiguous()
        self.output_embd = self.output_embd_tensor.data_ptr()

        self.attn_norm_tensors = [
            state_dict[naming.attn_norm(i)].to(torch_dt_norm) for i in range(nlayer)
        ]
        self.attn_norm_ptrs = [
            self.attn_norm_tensors[i].data_ptr() for i in range(nlayer)
        ]
        self.attn_norm = (c_void_p * nlayer)(*self.attn_norm_ptrs)

        def qkv_slices(_i):
            _Q = (
                state_dict[naming.attn_q(_i)]
                .reshape([nh, 2, dh // 2, d])
                .transpose(1, 2)
            )
            _K = (
                state_dict[naming.attn_k(_i)]
                .reshape([nkvh, 2, dh // 2, d])
                .transpose(1, 2)
            )
            _V = state_dict[naming.attn_v(_i)].reshape([nkvh, dh // 2, 2, d])
            _result = []
            _nh = nh // ndev
            _nkvh = nkvh // ndev
            for _idev in range(ndev):
                _result.append(_Q[_idev * _nh : (_idev + 1) * _nh, :, :, :])
                _result.append(_K[_idev * _nkvh : (_idev + 1) * _nkvh, :, :, :])
                _result.append(_V[_idev * _nkvh : (_idev + 1) * _nkvh, :, :])
            return _result

        self.qkv_tensor = [
            torch.concat(qkv_slices(i)).to(torch_dt_mat) for i in range(nlayer)
        ]
        if not transpose_weight:
            for i in range(nlayer):
                self.qkv_tensor[i] = (
                    self.qkv_tensor[i]
                    .reshape(ndev, (nh + 2 * nkvh) // ndev * dh, d)
                    .transpose(1, 2)
                    .contiguous()
                )
        self.qkv_tensor_ptrs = [self.qkv_tensor[i].data_ptr() for i in range(nlayer)]
        self.attn_qkv = (c_void_p * nlayer)(*self.qkv_tensor_ptrs)

        def qkv_b_slices(_i):
            _QB = (
                state_dict[naming.attn_q_b(_i)]
                .reshape([nh, 2, dh // 2])
                .transpose(1, 2)
            )
            _KB = (
                state_dict[naming.attn_k_b(_i)]
                .reshape([nkvh, 2, dh // 2])
                .transpose(1, 2)
            )
            _VB = state_dict[naming.attn_v_b(_i)].reshape([nkvh, dh // 2, 2])
            _result = []
            _nh = nh // ndev
            _nkvh = nkvh // ndev
            for _idev in range(ndev):
                _result.append(_QB[_idev * _nh : (_idev + 1) * _nh, :, :].flatten())
                _result.append(_KB[_idev * _nkvh : (_idev + 1) * _nkvh, :, :].flatten())
                _result.append(_VB[_idev * _nkvh : (_idev + 1) * _nkvh, :, :].flatten())
            return _result

        if naming.attn_q_b(0) in state_dict:
            self.qkv_b_tensors = [
                torch.concat(qkv_b_slices(i)).to(torch_dt_logits) for i in range(nlayer)
            ]
            self.qkv_b_tensor_ptrs = [
                self.qkv_b_tensors[i].data_ptr() for i in range(nlayer)
            ]
            self.attn_qkv_b = (c_void_p * nlayer)(*self.qkv_b_tensor_ptrs)
        else:
            self.attn_qkv_b = None

        self.attn_o_tensor = [
            (
                state_dict[naming.attn_o(i)]
                .to(torch_dt_mat)
                .reshape([d, ndev, nh // ndev * dh])
                .transpose(0, 1)
                .contiguous()
                if transpose_weight
                else state_dict[naming.attn_o(i)]
                .transpose(0, 1)
                .to(torch_dt_mat)
                .contiguous()
            )
            * scale_o
            for i in range(nlayer)
        ]
        self.attn_o_ptrs = [self.attn_o_tensor[i].data_ptr() for i in range(nlayer)]
        self.attn_o = (c_void_p * nlayer)(*self.attn_o_ptrs)

        self.ffn_norm_tensors = [
            state_dict[naming.ffn_norm(i)].to(torch_dt_norm) for i in range(nlayer)
        ]
        self.ffn_norm_ptrs = [
            self.ffn_norm_tensors[i].data_ptr() for i in range(nlayer)
        ]
        self.ffn_norm = (c_void_p * nlayer)(*self.ffn_norm_ptrs)

        def gate_up_slices(_i):
            _result = []
            _di = di // ndev
            for _idev in range(ndev):
                _start = _idev * _di
                _end = (_idev + 1) * _di
                _result.append(state_dict[naming.gate(_i)][_start:_end, :])
                _result.append(state_dict[naming.up(_i)][_start:_end, :])
            return _result

        self.gate_up_tensors = [
            torch.concat(gate_up_slices(i)).to(torch_dt_mat) for i in range(nlayer)
        ]
        if not transpose_weight:
            for i in range(nlayer):
                self.gate_up_tensors[i] = (
                    self.gate_up_tensors[i]
                    .reshape(ndev, 2 * di // ndev, d)
                    .transpose(1, 2)
                    .contiguous()
                )
        self.gate_up_ptrs = [self.gate_up_tensors[i].data_ptr() for i in range(nlayer)]
        self.ffn_gate_up = (c_void_p * nlayer)(*self.gate_up_ptrs)

        self.ffn_down_tensor = [
            (
                state_dict[naming.down(i)]
                .to(torch_dt_mat)
                .reshape([d, ndev, di // ndev])
                .transpose(0, 1)
                .contiguous()
                if transpose_weight
                else state_dict[naming.down(i)]
                .transpose(0, 1)
                .to(torch_dt_mat)
                .contiguous()
            )
            * scale_down
            for i in range(nlayer)
        ]
        self.ffn_down_ptrs = [self.ffn_down_tensor[i].data_ptr() for i in range(nlayer)]
        self.ffn_down = (c_void_p * nlayer)(*self.ffn_down_ptrs)


class DeepseekV3BatchedTask:
    def __init__(self, tasks: List[InferTask]):
        self.tasks = tasks
        self.nreq = len(tasks)

        # Precompute fields
        token_lists = [t.tokens for t in tasks]
        self.req_lens_list = [len(toks) for toks in token_lists]
        self.req_pos_list = [t.pos for t in tasks]
        self.kv_cache_ptrs = [t.kvcache().data() for t in tasks]
        self.temperaturas_list = [t.temperature for t in tasks]
        self.topks_list = [t.topk for t in tasks]
        self.topps_list = [t.topp for t in tasks]

        # Flatten token lists
        flat_tokens = [tok for toks in token_lists for tok in toks]
        self.ntok = len(flat_tokens)

        # Convert to ctypes arrays in one pass
        self.tokens = (c_uint * self.ntok)(*flat_tokens)
        self.req_lens = (c_uint * self.nreq)(*self.req_lens_list)
        self.req_pos = (c_uint * self.nreq)(*self.req_pos_list)
        self.kv_caches = (POINTER(KVCacheCStruct) * self.nreq)(*self.kv_cache_ptrs)
        self.temperaturas = (c_float * self.nreq)(*self.temperaturas_list)
        self.topks = (c_uint * self.nreq)(*self.topks_list)
        self.topps = (c_float * self.nreq)(*self.topps_list)

    def input_args(self):
        return (
            self.tokens,
            self.ntok,
            self.req_lens,
            self.nreq,
            self.req_pos,
            self.kv_caches,
            self.temperaturas,
            self.topks,
            self.topps,
        )


class DeepseekV3ForCauslLM:
    def __init__(
        self, model_dir_path, device=DeviceType.DEVICE_TYPE_CPU, ndev=1, max_tokens=None
    ):
        def load_all_safetensors_from_dir(dir_path_: str):
            tensors_ = {}
            dir_path_ = Path(dir_path_)
            for file in sorted(dir_path_.glob("*.safetensors")):
                data_ = safetensors.safe_open(file, "pt")
                for name_ in data_.keys():
                    tensors_[name_] = data_.get_tensor(name_)
            return tensors_

        print("Loading model weights to host...")
        load_start_time = time.time()

        with open(os.path.join(model_dir_path, "config.json"), "r") as f:
            config = json.load(f)
            self.config = config
        eos_token_id = self.config["eos_token_id"]
        self.eos_token_id = (
            [eos_token_id] if type(eos_token_id) == int else eos_token_id
        )
        transpose_weight = (
            device != DeviceType.DEVICE_TYPE_ASCEND
        )  # y = xW is faster than y=xW^T on Ascend
        if "deepseek_v3" == config["model_type"]:
            state_dict = load_all_safetensors_from_dir(model_dir_path)
            #TODO
            self.meta = ModelMetaFromDeepseek(config, max_tokens=max_tokens)
            self.weights = DeepseekV3WeightsImpl(
                    self.meta,
                    DeepseekV3AWQWeightsNaming(),
                    state_dict,
                    ndev=ndev,
                    transpose_weight=transpose_weight,
                )
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                    model_dir_path
                )
        else:
            raise ValueError("Unsupported model architecture")

        load_end_time = time.time()
        print(f"Time used: {load_end_time - load_start_time:.3f}s")

        print(f"Creating model on {ndev} devices...")
        load_start_time = time.time()
        dev_ids = (c_int * ndev)(*[i for i in range(ndev)])
        self.model_instance = create_deepseekv3_model(
            byref(self.meta),
            byref(self.weights),
            device,
            ndev,
            dev_ids,
        )
        load_end_time = time.time()
        print(f"Time used: {load_end_time - load_start_time:.3f}s")

    def max_context_len(self):
        return self.meta.dctx

    def create_kv_cache(self):
        return create_kv_cache(self.model_instance)

    def drop_kv_cache(self, kv_cache):
        drop_kv_cache(self.model_instance, kv_cache)

    def batch_infer_one_round(self, tasks: List[InferTask]):
        output = (c_uint * len(tasks))()
        batch_inputs = DeepseekV3BatchedTask(tasks)
        infer_batch(
            self.model_instance,
            *(batch_inputs.input_args()),
            output,
        )
        return list(output)

    def generate(self, input_content, max_steps, topp_=1.0, topk_=1, temperature_=1.0):
        input_content = self.tokenizer.apply_chat_template(
            conversation=[{"role": "user", "content": input_content}],
            add_generation_prompt=True,
            tokenize=False,
        )
        print(input_content, end="", flush=True)
        tokens = self.tokenizer.encode(input_content)
        infer_task = InferTask(
            0,
            tokens,
            self.max_context_len(),
            temperature_,
            topk_,
            topp_,
            self.eos_token_id,
        )
        infer_task.bind_kvcache(KVCache(self))

        steps = 0
        total_time = 0
        output_content = ""

        for step_i in range(max_steps):
            start_time = time.time()
            output_tokens = self.batch_infer_one_round([infer_task])
            end_time = time.time()
            steps += 1
            output_str = (
                self.tokenizer._tokenizer.id_to_token(output_tokens[0])
                .replace("▁", " ")
                .replace("<0x0A>", "\n")
            )
            output_content += output_str
            print(output_str, end="", flush=True)
            if output_tokens[0] in self.eos_token_id:
                break
            infer_task.next(output_tokens[0])

            if step_i > 0:
                total_time += end_time - start_time

        print("\n")
        avg_time = total_time * 1000 / (steps - 1)
        print(f"Time per step: {avg_time:.3f}ms")

        infer_task._kv_cache.drop(self)
        return output_content, avg_time

    def perplexity(self, test_sequences: List[Sequence[int]], batch_size=10):
        tasks = [
            InferTask(i, [], self.max_context_len(), 1.0, 1, 1.0, self.eos_token_id)
            for i in range(batch_size)
        ]
        kv_caches = [KVCache(self) for _ in range(batch_size)]

        nll = 0.0
        total_len = 0

        for i in range(0, len(test_sequences), batch_size):
            batch_id = 0
            true_tokens = []
            while batch_id < batch_size and batch_id + i < len(test_sequences):
                input_tokens = test_sequences[i + batch_id][:-1]
                true_tokens.extend(test_sequences[i + batch_id][1:])
                tasks[batch_id].tokens = input_tokens
                tasks[batch_id].bind_kvcache(kv_caches[batch_id])
                batch_id += 1

            batch_inputs = DeepseekV3BatchedTask(tasks[:batch_id])
            logits = torch.zeros(
                (batch_inputs.ntok, self.meta.dvoc), dtype=self.meta.torch_dtype_logits
            )
            forward_batch(
                self.model_instance,
                batch_inputs.tokens,
                batch_inputs.ntok,
                batch_inputs.req_lens,
                batch_inputs.nreq,
                batch_inputs.req_pos,
                batch_inputs.kv_caches,
                logits.data_ptr(),
            )

            logits = logits.float()
            token_ids = torch.tensor(true_tokens, dtype=torch.int64)  # [ntok,]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)  # (ntok, vocab)
            token_logprobs = log_probs[
                torch.arange(batch_inputs.ntok), token_ids
            ]  # (ntok,)

            start = 0
            for l in batch_inputs.req_lens_list:
                nll += -token_logprobs[start : start + l].sum().item()
                start += l
            total_len += token_logprobs.numel()

        for task in tasks:
            task.release_kvcache()

        return math.exp(nll / total_len)

    def destroy_model_instance(self):
        destroy_deepseekv3_model(self.model_instance)
        print("Model destroyed")


def test():
    if len(sys.argv) < 3:
        print(
            "Usage: python jiuge.py [--cpu | --nvidia| --cambricon | --ascend | --metax | --moore] <path/to/model_dir> [n_device]"
        )
        sys.exit(1)
    model_path = sys.argv[2]
    device_type = DeviceType.DEVICE_TYPE_CPU
    if sys.argv[1] == "--cpu":
        device_type = DeviceType.DEVICE_TYPE_CPU
    elif sys.argv[1] == "--nvidia":
        device_type = DeviceType.DEVICE_TYPE_NVIDIA
    elif sys.argv[1] == "--cambricon":
        device_type = DeviceType.DEVICE_TYPE_CAMBRICON
    elif sys.argv[1] == "--ascend":
        device_type = DeviceType.DEVICE_TYPE_ASCEND
    elif sys.argv[1] == "--metax":
        device_type = DeviceType.DEVICE_TYPE_METAX
    elif sys.argv[1] == "--moore":
        device_type = DeviceType.DEVICE_TYPE_MOORE
    elif sys.argv[1] == "--iluvatar":
        device_type = DeviceType.DEVICE_TYPE_ILUVATAR
    else:
        print(
            "Usage: python jiuge.py [--cpu | --nvidia| --cambricon | --ascend | --metax | --moore] <path/to/model_dir> [n_device]"
        )
        sys.exit(1)

    ndev = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    model = DeepseekV3ForCauslLM(model_path, device_type, ndev)
    model.generate("山东最高的山是？", 500)
    model.destroy_model_instance()


if __name__ == "__main__":
    test()
