import transformers

from retake.qwen2_vl import (
    retake_Qwen2VLAttention_forward,
    retake_Qwen2VLSdpaAttention_forward,
    retake_Qwen2VLFlashAttention2_forward,
    retake_Qwen2VLForConditionalGeneration_compress_video_tokens,
    retake_Qwen2VLForConditionalGeneration_segment_input_ids,
    retake_Qwen2VLForConditionalGeneration_get_chunk_size,
    retake_Qwen2VLForConditionalGeneration_forge_input_chunks,
    retake_Qwen2VLForConditionalGeneration_forward,
)
from retake.qwen2_5_vl import (
    fixed_Qwen2_5_VLModel_prepare_4d_causal_attention_mask_with_cache_position,
    retake_Qwen2_5_VLAttention_forward,
    retake_Qwen2_5_VLSdpaAttention_forward,
    retake_Qwen2_5_VLFlashAttention2_forward,
    retake_Qwen2_5_VLForConditionalGeneration_segment_input_ids,
    retake_Qwen2_5_VLForConditionalGeneration_get_chunk_size,
    retake_Qwen2_5_VLForConditionalGeneration_forge_input_chunks,
    retake_Qwen2_5_VLForConditionalGeneration_forward,
)
from retake.llava_onevision import (
    retake_Qwen2Attention_init,
    retake_Qwen2Attention_forward,
    retake_LlavaOnevisionForConditionalGeneration_get_chunk_size,
    retake_LlavaOnevisionForConditionalGeneration_segment_input_ids,
    retake_LlavaOnevisionForConditionalGeneration_compress_video_tokens,
    retake_LlavaOnevisionForConditionalGeneration_forge_input_chunks,
    retake_LlavaOnevisionForConditionalGeneration_forward,
)


def patch_qwen2vl_config(config, exp_configs):
    # Rope Scaling
    if 'scaling_factor' in exp_configs:
        config.rope_scaling.pop('type')
        config.rope_scaling['rope_type'] = 'yarn'
        config.rope_scaling['factor'] = exp_configs['scaling_factor']
        config.rope_scaling['beta_fast'] = 32.0
        config.rope_scaling['beta_slow'] = 1.0
    # ReTaKe
    config.longvideo_kwargs = exp_configs.get('longvideo_kwargs', {})
    return config


def patch_qwen2_5_vl_config(config, exp_configs):
    # Rope Scaling
    if 'scaling_factor' in exp_configs:
        config.rope_scaling.pop('type')
        config.rope_scaling['rope_type'] = 'yarn'
        config.rope_scaling['factor'] = exp_configs['scaling_factor']
        config.rope_scaling['beta_fast'] = 32.0
        config.rope_scaling['beta_slow'] = 1.0
    # ReTaKe
    config.longvideo_kwargs = exp_configs.get('longvideo_kwargs', {})
    return config


def patch_llava_onevision_config(config, exp_configs):
    # Rope Scaling
    if 'scaling_factor' in exp_configs:
        config.text_config.rope_scaling = {
            'rope_type': 'yarn',
            'factor': exp_configs['scaling_factor'],
            'beta_fast': 32.0,
            'beta_slow': 1.0,
        }
    # ReTaKe
    config.longvideo_kwargs = exp_configs.get('longvideo_kwargs', {})
    return config


def patch_qwen2vl(method):

    if method == "retake":
        print("Using ReTaKe for Qwen2VLForConditionalGeneration!")
        transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLAttention.forward = retake_Qwen2VLAttention_forward
        transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLSdpaAttention.forward = retake_Qwen2VLSdpaAttention_forward
        transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLFlashAttention2.forward = retake_Qwen2VLFlashAttention2_forward
        transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLForConditionalGeneration.compress_video_tokens = retake_Qwen2VLForConditionalGeneration_compress_video_tokens
        transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLForConditionalGeneration.segment_input_ids = retake_Qwen2VLForConditionalGeneration_segment_input_ids
        transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLForConditionalGeneration.get_chunk_size = retake_Qwen2VLForConditionalGeneration_get_chunk_size
        transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLForConditionalGeneration.forge_input_chunks = retake_Qwen2VLForConditionalGeneration_forge_input_chunks
        transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLForConditionalGeneration.forward = retake_Qwen2VLForConditionalGeneration_forward
    else:
        raise NotImplementedError


def patch_qwen2_5_vl(method):

    if method == "retake":
        print("Using ReTaKe for Qwen2_5_VLForConditionalGeneration!")
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLModel._prepare_4d_causal_attention_mask_with_cache_position = fixed_Qwen2_5_VLModel_prepare_4d_causal_attention_mask_with_cache_position
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLAttention.forward = retake_Qwen2_5_VLAttention_forward
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLSdpaAttention.forward = retake_Qwen2_5_VLSdpaAttention_forward
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLFlashAttention2.forward = retake_Qwen2_5_VLFlashAttention2_forward
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.segment_input_ids = retake_Qwen2_5_VLForConditionalGeneration_segment_input_ids
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.get_chunk_size = retake_Qwen2_5_VLForConditionalGeneration_get_chunk_size
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forge_input_chunks = retake_Qwen2_5_VLForConditionalGeneration_forge_input_chunks
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = retake_Qwen2_5_VLForConditionalGeneration_forward
    else:
        raise NotImplementedError


def patch_llava_onevision(method):

    if method == "retake":
        print("Using ReTaKe for LlavaOnevisionForConditionalGeneration!")
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.__init__ = retake_Qwen2Attention_init
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = retake_Qwen2Attention_forward
        transformers.models.llava_onevision.modeling_llava_onevision.LlavaOnevisionForConditionalGeneration.get_chunk_size = retake_LlavaOnevisionForConditionalGeneration_get_chunk_size
        transformers.models.llava_onevision.modeling_llava_onevision.LlavaOnevisionForConditionalGeneration.segment_input_ids = retake_LlavaOnevisionForConditionalGeneration_segment_input_ids
        transformers.models.llava_onevision.modeling_llava_onevision.LlavaOnevisionForConditionalGeneration.compress_video_tokens = retake_LlavaOnevisionForConditionalGeneration_compress_video_tokens
        transformers.models.llava_onevision.modeling_llava_onevision.LlavaOnevisionForConditionalGeneration.forge_input_chunks = retake_LlavaOnevisionForConditionalGeneration_forge_input_chunks
        transformers.models.llava_onevision.modeling_llava_onevision.LlavaOnevisionForConditionalGeneration.forward = retake_LlavaOnevisionForConditionalGeneration_forward
    else:
        raise NotImplementedError
