#ifndef COMPRESS_TYPES_H
#define COMPRESS_TYPES_H

typedef enum {
    R100 = 32,
    R133 = 24,  // 32 / 24 = 1.33x
    R139 = 23,  // 32 / 23 = 1.39x
    R145 = 22,  // 32 / 22 = 1.45x
    R152 = 21,  // 32 / 21 = 1.52x
    R160 = 20,  // 32 / 20 = 1.60x
    R200 = 16   // 32 / 16 = 2.00x
} FixedRatio;

typedef enum {
    DT_BF16 = 0,
    DT_FP16 = 1,
    DT_FP8E5M2 = 2,
    DT_FP8E4M3 = 3,
    DT_INVALID = 100
} DataType;

#endif  // COMPRESS_TYPES_H
