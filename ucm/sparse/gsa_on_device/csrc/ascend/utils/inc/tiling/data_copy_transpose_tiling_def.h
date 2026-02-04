// from vllm-ascend, see
// https://github.com/vllm-project/vllm-ascend/blob/main/csrc/utils/inc/tiling/data_copy_transpose_tiling_def.h

/*!
 * \file data_copy_transpose_tiling_def.h
 * \brief
 */

#pragma once

#include <cstdint>
#include <register/tilingdata_base.h>

namespace optiling {

BEGIN_TILING_DATA_DEF(CopyTransposeTiling)
TILING_DATA_FIELD_DEF(uint32_t, dstShapeB);
TILING_DATA_FIELD_DEF(uint32_t, dstShapeN);
TILING_DATA_FIELD_DEF(uint32_t, dstShapeS);
TILING_DATA_FIELD_DEF(uint32_t, dstShapeHN);
TILING_DATA_FIELD_DEF(uint32_t, dstShapeH);
TILING_DATA_FIELD_DEF(uint32_t, srcShapeB);
TILING_DATA_FIELD_DEF(uint32_t, srcShapeN);
TILING_DATA_FIELD_DEF(uint32_t, srcShapeS);
TILING_DATA_FIELD_DEF(uint32_t, srcShapeHN);
TILING_DATA_FIELD_DEF(uint32_t, originalShapeNLen);
TILING_DATA_FIELD_DEF(uint32_t, shapeSHValue);
TILING_DATA_FIELD_DEF(uint32_t, shapeNsValue);
TILING_DATA_FIELD_DEF(uint32_t, shapeNsnValue);
TILING_DATA_FIELD_DEF(uint32_t, invalidParamCopyTransposeTiling);
TILING_DATA_FIELD_DEF(uint32_t, shapeBHValue);
TILING_DATA_FIELD_DEF(uint32_t, paramsAlign);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(CopyTransposeTilingOp, CopyTransposeTiling)

} // namespace optiling
