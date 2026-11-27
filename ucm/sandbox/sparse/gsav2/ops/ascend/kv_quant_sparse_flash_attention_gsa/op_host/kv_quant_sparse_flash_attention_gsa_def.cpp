/*!
 * \file kv_quant_sparse_flash_attention_gsa_def.cpp
 * \brief GSA 定制改进版 KvQuantSparseFlashAttention 算子定义。
 *
 * 相比官方 v1 的改进：
 *   1. A5 全类型(含 hifloat8)支持 sparse_indices > 2048（最多 8192）
 *   2. 新增可选输出 softmax_max / softmax_sum（return_softmax_lse 控制）
 *   3. 新增可选输入 sinks
 *   4. kernel 入口 opFile 改为 kv_quant_sparse_flash_attention_gsa
 *
 * 参考:
 * https://gitcode.com/cann/ops-transformer/tree/master/attention/kv_quant_sparse_flash_attention_v2
 */

#include "register/op_def_registry.h"

namespace ops {
class KvQuantSparseFlashAttentionGsa : public OpDef {
public:
    explicit KvQuantSparseFlashAttentionGsa(const char *name) : OpDef(name)
    {
        this->Input("query")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("key")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("value").ParamType(REQUIRED).Follow("key").AutoContiguous();
        this->Input("sparse_indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("key_dequant_scale")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("value_dequant_scale")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("block_table")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("actual_seq_lengths_query")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("actual_seq_lengths_kv")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        // v2 新增: sinks 可选输入
        this->Input("sinks")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("actual_sparse_lengths")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("attention_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        // v2 新增: softmax_max / softmax_sum 可选输出
        this->Output("softmax_max")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("softmax_sum")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});

        this->Attr("scale_value").AttrType(REQUIRED).Float(1.0);
        this->Attr("key_quant_mode").AttrType(REQUIRED).Int(1);
        this->Attr("value_quant_mode").AttrType(REQUIRED).Int(1);
        this->Attr("sparse_block_size").AttrType(OPTIONAL).Int(1);
        this->Attr("layout_query").AttrType(OPTIONAL).String("BSND");
        this->Attr("layout_kv").AttrType(OPTIONAL).String("BSND");
        this->Attr("sparse_mode").AttrType(OPTIONAL).Int(3);  // 3:默认值，只计算下三角
        this->Attr("pre_tokens").AttrType(OPTIONAL).Int(INT64_MAX);
        this->Attr("next_tokens").AttrType(OPTIONAL).Int(INT64_MAX);
        this->Attr("attention_mode").AttrType(OPTIONAL).Int(0);
        this->Attr("quant_scale_repo_mode").AttrType(OPTIONAL).Int(1);
        this->Attr("tile_size").AttrType(OPTIONAL).Int(128);     // 128:默认值
        this->Attr("rope_head_dim").AttrType(OPTIONAL).Int(64);  // 64:默认值
        // v2 新增: 是否返回 softmax 统计结果
        this->Attr("return_softmax_lse").AttrType(OPTIONAL).Bool(false);

        // A2/A3 配置 (INT8 only)
        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
            .ExtendCfgInfo("jitCompile.flag", "static_false,dynamic_false");
        this->AICore().AddConfig("ascend910b", aicore_config);
        this->AICore().AddConfig("ascend910_93", aicore_config);

        // A5 配置 (INT8 + float8_e4m3 + hifloat8, 全类型支持 sparse_size > 2048)
        OpAICoreConfig aicore_config_95;
        aicore_config_95.Input("query")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_95.Input("key")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT8_E4M3FN, ge::DT_HIFLOAT8, ge::DT_INT8, ge::DT_FLOAT8_E4M3FN,
                       ge::DT_HIFLOAT8, ge::DT_INT8})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_95.Input("value")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT8_E4M3FN, ge::DT_HIFLOAT8, ge::DT_INT8, ge::DT_FLOAT8_E4M3FN,
                       ge::DT_HIFLOAT8, ge::DT_INT8})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_95.Input("sparse_indices")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_95.Input("key_dequant_scale")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_95.Input("value_dequant_scale")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_95.Input("block_table")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_95.Input("actual_seq_lengths_query")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_95.Input("actual_seq_lengths_kv")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_95.Input("sinks")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_95.Input("actual_sparse_lengths")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_95.Output("attention_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND});
        aicore_config_95.Output("softmax_max")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND});
        aicore_config_95.Output("softmax_sum")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND});
        aicore_config_95.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
            .ExtendCfgInfo("jitCompile.flag", "static_false,dynamic_false");
        this->AICore().AddConfig("ascend950", aicore_config_95);
    }
};
OP_ADD(KvQuantSparseFlashAttentionGsa);
}  // namespace ops
