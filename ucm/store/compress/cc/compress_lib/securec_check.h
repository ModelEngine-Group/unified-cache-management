#ifndef SECUREC_CHECK_H
#define SECUREC_CHECK_H

#include "securec.h"
#include "tunstall.h"

#define SECUREC_CHECK(call)                              \
    do {                                                 \
        errno_t secErr = (call);                         \
        if (secErr != EOK) { return R_ERR_COPY_FAILED; } \
    } while (0)

#endif  // SECUREC_CHECK_H
