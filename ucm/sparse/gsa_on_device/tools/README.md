# Attention Sparsity & Layer Skipping Profiler

This project analyzes large model Attention behaviors to support KV cache sparsification and layer skipping optimization:
- Token-block-level sparsity (Hash Ratio)
- Inter-layer attention overlap (Overlap Matrix)
- Automatic derivation of Skip and Rollback layer configurations

Full data loop： __vLLM → K/Q collection → offline analysis of sparsity and layer skipping__

 


$\quad$

## 📥 vLLM → K/Q collection 
Insert K/Q tensor-saving logic before the attention computation → run dahailaozhen prompt to collect the generated K/Q.
-  DeepSeekR1: vllm/vllm/v1/attention/backends/mla/flashmla.py 
-  Qwen3-32B: vllm/vllm/model_executor/models/qwen3.py 

  
```python
class FlashMLAImpl(MLACommonImpl[FlashMLAMetadata]):

    can_return_lse_for_decode: bool = True

    import threading

    def async_write(filename, data):
        with open(filename, 'a') as f:
            f.write(str(data) + '\n')

    def __init__(
            self,
            num_heads: int,
            head_size: int,
            scale: float,
            num_kv_heads: int,
            alibi_slopes: Optional[list[float]],
            sliding_window: Optional[int],
            kv_cache_dtype: str,
            logits_soft_cap: Optional[float],
            attn_type: str,
            kv_sharing_target_layer_name: Optional[str],
            # MLA Specific Arguments
            **mla_args) -> None:
        super().__init__(num_heads, head_size, scale, num_kv_heads,
                         alibi_slopes, sliding_window, kv_cache_dtype,
                         logits_soft_cap, attn_type,
                         kv_sharing_target_layer_name, **mla_args)
        self.topk_blocks_ratio = 0
        
        self.k_idx = 0
        
        self.q0_idx = 0
        self.q1_idx = 0
        self.q2_idx = 0
        self.q3_idx = 0
        self.q4_idx = 0
        self.q5_idx = 0
        self.q6_idx = 0
        self.q7_idx = 0

 
        assert is_flashmla_supported(), \
            "FlashMLA is not supported on this device"
        if CudaPlatform.has_device_capability(100):
            raise NotImplementedError(
                "FlashMLA is temporarily disabled on Blackwell (SM 10.0). "
                "Please use CUTLASS_MLA or TRITON_MLA instead. "
                "Example: `export VLLM_ATTENTION_BACKEND=CUTLASS_MLA`")

        unsupported_features = [alibi_slopes, sliding_window, logits_soft_cap]
        if any(unsupported_features):
            raise NotImplementedError(
                "FlashMLAImpl does not support one of the following: "
                "alibi_slopes, sliding_window, logits_soft_cap")

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "FlashMLAImpl")
    
    
    def _forward_decode(
        self,
        q: Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashMLAMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        assert kv_c_and_k_pe_cache.numel() > 0
        assert attn_metadata.decode is not None

        if type(q) is tuple:
            q_pe = q[1] 
            q = torch.cat(q, dim=-1)  

        assert isinstance(q, torch.Tensor)
     
        
        
        # attn_metadata.decode.seq_lens
        for b_idx in range(attn_metadata.decode.block_table.shape[0]):
            if attn_metadata.decode.seq_lens[b_idx] > 0:
                # 开始保存profiling数据
                cuda_idx = str(kv_c_and_k_pe_cache.device)  
                cuda_idx = cuda_idx.replace(":", "_")
                idx_num = int(cuda_idx.split("_")[1])
                
                cuda_idx = cuda_idx.replace("_", "")
                base_path = f"/output/profiling/DeepSeekR1/{cuda_idx}/dahailaozhen_openthink"
                os.makedirs(base_path, exist_ok=True)
                if idx_num == 0:
                    q_path = f"{base_path}/real_q_{layer.layer_name}_custom{self.q0_idx}.pt"
                    torch.save(q.cpu(), q_path)
                    self.q0_idx += 1
                elif idx_num == 1:
                    q_path = f"{base_path}/real_q_{layer.layer_name}_custom{self.q1_idx}.pt"
                    torch.save(q.cpu(), q_path)
                    self.q1_idx += 1
                elif idx_num == 2:
                    q_path = f"{base_path}/real_q_{layer.layer_name}_custom{self.q2_idx}.pt"
                    torch.save(q.cpu(), q_path)
                    self.q2_idx += 1
                elif idx_num == 3:
                    q_path = f"{base_path}/real_q_{layer.layer_name}_custom{self.q3_idx}.pt"
                    torch.save(q.cpu(), q_path)
                    self.q3_idx += 1
                elif idx_num == 4:
                    q_path = f"{base_path}/real_q_{layer.layer_name}_custom{self.q4_idx}.pt"
                    torch.save(q.cpu(), q_path)
                    self.q4_idx += 1
                elif idx_num == 5:
                    q_path = f"{base_path}/real_q_{layer.layer_name}_custom{self.q5_idx}.pt"
                    torch.save(q.cpu(), q_path)
                    self.q5_idx += 1
                elif idx_num == 6:
                    q_path = f"{base_path}/real_q_{layer.layer_name}_custom{self.q6_idx}.pt"
                    torch.save(q.cpu(), q_path)
                    self.q6_idx += 1
                elif idx_num == 7:
                    q_path = f"{base_path}/real_q_{layer.layer_name}_custom{self.q7_idx}.pt"
                    torch.save(q.cpu(), q_path)
                    self.q7_idx += 1
                cuda0_path = f"/output/profiling/DeepSeekR1/cuda0/dahailaozhen_openthink"
                os.makedirs(cuda0_path, exist_ok=True)
                if kv_c_and_k_pe_cache.device == torch.device('cuda:0'):
                    k_path = f"{cuda0_path}/real_k_{layer.layer_name}_custom{self.k_idx}.pt"
                    torch.save(kv_c_and_k_pe_cache.cpu(), k_path)
                    self.k_idx += 1
       
        
        """
        1、只保存0卡上 kv_c_and_k_pe_cache 和 hidden state
        2、保存 0-7卡上的 query
        """

 
        o, lse = flash_mla_with_kvcache(
            q=q.unsqueeze(1),   
            k_cache=kv_c_and_k_pe_cache.unsqueeze(-2),   
            block_table=attn_metadata.decode.block_table,
            cache_seqlens=attn_metadata.decode.seq_lens,
            head_dim_v=self.kv_lora_rank,
            tile_scheduler_metadata=attn_metadata.decode.
            tile_scheduler_metadata,
            num_splits=attn_metadata.decode.num_splits,
            softmax_scale=self.scale,
            causal=True,
            descale_q=layer._q_scale.reshape(1),
            descale_k=layer._k_scale.reshape(1),
        )

        return o, lse
```

```python
class Qwen3Attention(nn.Module):

    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 num_kv_heads: int,
                 max_position: int = 4096 * 32,
                 head_dim: Optional[int] = None,
                 rms_norm_eps: float = 1e-06,
                 qkv_bias: bool = False,
                 rope_theta: float = 10000,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 rope_scaling: Optional[tuple] = None,
                 prefix: str = "",
                 attn_type: str = AttentionType.DECODER) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=self.rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads,
                              cache_config=cache_config,
                              quant_config=quant_config,
                              prefix=f"{prefix}.attn",
                              attn_type=attn_type)
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
 
        self.ix=0
        self.prefix=prefix
        path='/output/profiling/Qwen3-32B'
        os.makedirs(path, exist_ok=True)
        self.save_path_q_npu0 = f"{path}/npu0/my_tensor_q_{self.prefix}.pt"
        self.save_path_k_npu0 = f"{path}/npu0/my_tensor_k_{self.prefix}.pt"
        self.my_dict_q_npu0={}
        self.my_dict_k_npu0={}
        
        self.save_path_q_npu1 = f"{path}/npu1/my_tensor_q_{self.prefix}.pt"
        self.save_path_k_npu1 = f"{path}/npu1/my_tensor_k_{self.prefix}.pt"
        self.my_dict_q_npu1={}
        self.my_dict_k_npu1={}
        
        self.save_path_q_npu2 = f"{path}/npu2/my_tensor_q_{self.prefix}.pt"
        self.save_path_k_npu2 = f"{path}/npu2/my_tensor_k_{self.prefix}.pt"
        self.my_dict_q_npu2={}
        self.my_dict_k_npu2={}
        
        self.save_path_q_npu3 = f"{path}/npu3/my_tensor_q_{self.prefix}.pt"
        self.save_path_k_npu3 = f"{path}/npu3/my_tensor_k_{self.prefix}.pt"
        self.my_dict_q_npu3={}
        self.my_dict_k_npu3={}
 
        
   
       


    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # Add qk-norm
        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim,
                           self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)
        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim,
                           self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)
        q, k = self.rotary_emb(positions, q, k)           
 
        
    
        global_rank = dist.get_rank()
        if global_rank==0:
            if self.ix>0:
                self.my_dict_q_npu0[(self.ix, self.prefix)] = q.detach().cpu()
                self.my_dict_k_npu0[(self.ix, self.prefix)] = k.detach().cpu()
                if self.ix>=128:
                    torch.save(self.my_dict_q_npu0, self.save_path_q_npu0)
                    torch.save(self.my_dict_k_npu0, self.save_path_k_npu0)
                print(self.ix, self.prefix, q.shape, k.shape)  
        elif global_rank==1:
            if self.ix>0:
                self.my_dict_q_npu1[(self.ix, self.prefix)] = q.detach().cpu()
                self.my_dict_k_npu1[(self.ix, self.prefix)] = k.detach().cpu()
                if self.ix>=128:
                    torch.save(self.my_dict_q_npu1, self.save_path_q_npu1)
                    torch.save(self.my_dict_k_npu1, self.save_path_k_npu1)
                print(self.ix, self.prefix, q.shape, k.shape) 
        elif global_rank==2:
            if self.ix>0:
                self.my_dict_q_npu2[(self.ix, self.prefix)] = q.detach().cpu()
                self.my_dict_k_npu2[(self.ix, self.prefix)] = k.detach().cpu()
                if self.ix>=128:
                    torch.save(self.my_dict_q_npu2, self.save_path_q_npu2)
                    torch.save(self.my_dict_k_npu2, self.save_path_k_npu2)
                print(self.ix, self.prefix, q.shape, k.shape) 
        elif global_rank==3:
            if self.ix>0:
                self.my_dict_q_npu3[(self.ix, self.prefix)] = q.detach().cpu()
                self.my_dict_k_npu3[(self.ix, self.prefix)] = k.detach().cpu()
                if self.ix>=128:
                    torch.save(self.my_dict_q_npu3, self.save_path_q_npu3)
                    torch.save(self.my_dict_k_npu3, self.save_path_k_npu3)
                print(self.ix, self.prefix, q.shape, k.shape) 
        elif  global_rank>3:
            print('TP should be 4! Please check it.')
        else:
            pass   
        self.ix+=1   
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output
    
```


$\quad$

## 📥 Model configuration
Modify the model configuration accordingly: 
```python
class Config:
    device = 'npu'  #推理设备名称，cuda/npu
    enable_mla = False  #是否启用 MLA Attention
    TP = 4   #tensor parallel 数
    kvheads = 8  #KV head 总数
    qhead = 8  #每 KV head对应 Q head 数
    chunk_size = 128   #block 粒度（token 聚合单位）
    layers = 64  #模型层数
    dim = 128  #head hidden size
    if enable_mla:
        scale  = 1/np.sqrt(128+64) * (0.1*np.log(40) + 1)**2 
    else:
        scale = 1/np.sqrt(dim)
    sink = 128  #强制保留开头 token
    recent = 512  #强制保留最近 token
    datapath = './output/_profiling/Qwen3-32B'  #profiling 数据路径
    calibration_data = 'dahailaozhen_openthink' #数据集名称
    skipLayerinput=['dahailaozhen_openthink']
```

$\quad$

## 📥 SKip Layer Config
Run main.py → change __(perserved_tokens, sparsity,computed_ratio_of_hamming_layers)__ to get the best skip-layer config.
```python
args = Config()
ins = RALS(args, reuse=False)
ins.run(perserved_tokens=2048, 
        sparsity=0.9, 
        computed_ratio_of_hamming_layers=0.3, 
        update_tokens_for_recompute=True)
```
