#include <cstdio>

#include "fields/alt_bn128-fp2.hip"
#include "curves/jacobian_t.hip"
#include "curves/xyzz_t.hip"
#include "msm/msm.hip"
#include "ntt/ntt.hip"
#include "spmvm/spmvm.hip"

#include "utils.h"

namespace alt_bn128 {
    typedef jacobian_t<fp_t> g1_t;
    typedef jacobian_t<fp2_t> g2_t;
    typedef xyzz_t<fp_t> g1_bucket_t;
    typedef xyzz_t<fp2_t> g2_bucket_t;
}
