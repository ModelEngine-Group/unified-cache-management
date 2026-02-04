// from vllm-ascend, see
// https://github.com/vllm-project/vllm-ascend/blob/main/csrc/utils/inc/tiling/data_copy_transpose_tiling.h

/*!
 * \file data_copy_transpose_tiling.h
 * \brief
 */

#pragma once

#include <vector>
#include <graph/tensor.h>
#include "data_copy_transpose_tiling_def.h"

namespace optiling {

inline void GetDataCopyTransposeTiling(const ge::Shape &dstShape, const ge::Shape &srcShape, const uint32_t typeSize,
                                       optiling::CopyTransposeTiling &tiling)
{
    std::vector<int64_t> dstShapeInfo = dstShape.GetDims();
    std::vector<int64_t> srcShapeInfo = srcShape.GetDims();

    tiling.set_dstShapeB(dstShapeInfo[0]);
    tiling.set_dstShapeN(dstShapeInfo[1]);
    tiling.set_dstShapeS(dstShapeInfo[2]);
    tiling.set_dstShapeH(dstShapeInfo[3]);
    tiling.set_dstShapeHN(tiling.get_dstShapeH() / tiling.get_dstShapeN());

    tiling.set_srcShapeB(srcShapeInfo[0]);
    tiling.set_srcShapeN(srcShapeInfo[1]);
    tiling.set_srcShapeS(srcShapeInfo[2]);
    tiling.set_srcShapeHN(srcShapeInfo[3]);
    tiling.set_originalShapeNLen(tiling.get_srcShapeHN() * typeSize);
    tiling.set_shapeSHValue(tiling.get_dstShapeS() * tiling.get_dstShapeH());
    tiling.set_shapeNsValue(tiling.get_dstShapeN() * tiling.get_dstShapeS());
    tiling.set_shapeNsnValue(tiling.get_dstShapeN() * tiling.get_srcShapeS() * tiling.get_srcShapeN());
    tiling.set_shapeBHValue(tiling.get_dstShapeB() * tiling.get_dstShapeH());
}

} // namespace optiling
