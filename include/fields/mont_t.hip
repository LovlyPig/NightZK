// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#if defined(__HIPCC__) && !defined(__SPPARK_FF_MONT_T_HIP__)
# define __SPPARK_FF_MONT_T_HIP__

# include <cstddef>
# include <cstdint>
# include "pow.hip"

//
// This is a trimmed-down version tailored for use in NTT only.
//

# if defined(__GFX10__) || defined(__GFX11__) || defined(__GFX12__)
#  define v_addc_co_u32 "v_add_co_ci_u32 "
#  define v_subb_co_u32 "v_sub_co_ci_u32 "
# elif defined(__GFX9__)
#  define v_addc_co_u32 "v_addc_co_u32 "
#  define v_subb_co_u32 "v_subb_co_u32 "
# elif !defined(__HIP_DEVICE_COMPILE__)
#  define v_addc_co_u32 "v_dummy "
#  define v_subb_co_u32 "v_dummy "
# else
#  error "unsupported GFX architecture"
# endif

# define __MONT_T_STR(x) #x
# define __MONT_T_XSTR(x) __MONT_T_STR(x)
# define S_OP(op) "s_" #op "_b" __MONT_T_XSTR(__AMDGCN_WAVEFRONT_SIZE) " "

# define inline __device__ __forceinline__

//
// To instantiate declare modulus as __device__ __constant___ const and
// complement it with its factual bit-length and the corresponding 32-bit
// Motgomery factor. Bit-length has to be such that (N+31)/32 is val
// and not less than 4.
//
template<const size_t N, const uint32_t MOD[(N+31)/32], const uint32_t& M0,
         const uint32_t RR[(N+31)/32], const uint32_t ONE[(N+31)/32],
         const uint32_t MODx[(N+31)/32] = MOD>
class alignas(((N+63)/64)&1 ? 8 : 16) mont_t {
public:
    static const size_t nbits = N;
    static constexpr size_t __device__ bit_length() { return N; }
    static const uint32_t degree = 1;
    using mem_t = mont_t;

    mont_t(uint64_t *mem) { memcpy(val, mem, sizeof(val)); } // TO BE JUSTIFIED
protected:
    static const size_t n = (N+31)/32;
    static_assert((n%2) == 0 && n >= 4, "unsupported bit length");

    template<typename vec_t>
    static inline vec_t shfl_xor(const vec_t& a, uint32_t idx = 1)
    {
        vec_t ret;
        for (size_t i = 0; i < sizeof(vec_t)/sizeof(uint32_t); i++)
            ret[i] = __shfl_xor(a[i], idx, 32);
        return ret;
    }

private:
    uint32_t val[n];

    static inline void mul_n(uint32_t* acc, const uint32_t* a, uint32_t bi,
                             size_t n_=n)
    {
        #pragma unroll
        for (size_t j = 0; j < n_; j+=2) {
            acc[j] = a[j]*bi;
            acc[j+1] = __umulhi(a[j], bi);
        }
	}

    static inline uint32_t cmad_n(uint32_t* acc, const uint32_t* a, uint32_t bi,
                              size_t n_=n)
    {
        uint32_t carry = 0;

        uint64_t p0 = (uint64_t)a[0] * bi;
        acc[0] = __builtin_addc((uint32_t)p0, acc[0], 0, &carry);
        acc[1] = __builtin_addc((uint32_t)(p0 >> 32), acc[1], carry, &carry);

        #pragma unroll
        for (size_t j = 2; j < n_; j += 2) {
            uint64_t p = (uint64_t)a[j] * bi;
            acc[j] = __builtin_addc((uint32_t)p, acc[j], carry, &carry);
            acc[j+1] = __builtin_addc((uint32_t)(p >> 32), acc[j+1], carry, &carry);
        }

        return carry;
    }

    static inline uint32_t cadd_n(uint32_t* acc, const uint32_t* a, size_t n_=n)
    {
        uint32_t carry = 0;
    	acc[0] = __builtin_addc(acc[0], a[0], 0, &carry);

        #pragma unroll
        for (size_t i = 1; i < n_; i++) {
            acc[i] = __builtin_addc(acc[i], a[i], carry, &carry);
        }
    	return carry;
    }

    class wide_t {
    private:
        union {
            uint32_t val[2*n];
            mont_t s[2];
        };

    public:
        inline uint32_t& operator[](size_t i)               { return val[i]; }
        inline const uint32_t& operator[](size_t i) const   { return val[i]; }
        inline operator mont_t()
        {
            s[0].mul_by_1();
            return s[0] + s[1];
        }
        inline void final_sub(uint32_t carry, uint32_t* tmp)
        {   s[1].final_sub(carry, tmp);   }

        inline wide_t() {}

    private:
        static inline void mad_row(uint32_t* odd, uint32_t* val,
                                   const uint32_t* a, uint32_t bi, size_t n_=n)
        {
            // cmad_n(odd, a+1, bi, n_-2);
            // asm("madc.lo.cc.u32 %0, %2, %3, 0; madc.hi.u32 %1, %2, %3, 0;"
            //     : "=r"(odd[n_-2]), "=r"(odd[n_-1])
            //     : "r"(a[n_-1]), "r"(bi));

            uint32_t carry_odd = cmad_n(odd, a+1, bi, n_-2);
            uint64_t p = (uint64_t)a[n_-1] * bi;
            odd[n_-2] = __builtin_addc((uint32_t)p, 0, carry_odd, &carry_odd);
            odd[n_-1] = __builtin_addc((uint32_t)(p >> 32), 0, carry_odd, &carry_odd);
            
            uint32_t carry_val = cmad_n(val, a, bi, n_);
            odd[n_-1] += carry_val;
        }

    public:
        inline wide_t(const mont_t& a, const mont_t& b)     //// |a|*|b|
        {
            size_t i = 0;
            uint32_t odd[2*n-2];

            mul_n(val, &a[0], b[0]);
            mul_n(odd,  &a[1], b[0]); 
	    ++i; mad_row(&val[i+1], &odd[i-1], &a[0], b[i]);

            #pragma unroll
            while (i < n-2) {
                ++i; mad_row(&odd[i],    &val[i],  &a[0], b[i]);
                ++i; mad_row(&val[i+1], &odd[i-1], &a[0], b[i]);
            }
            
            uint32_t carry = cadd_n(&val[1], &odd[0], 2*n-2);
     	    val[2*n-1] += carry;
        }

    private:
        static inline void qad_row(uint32_t* odd, uint32_t* val,
                                   const uint32_t* a, uint32_t bi, size_t n)
        {
            uint32_t carry_odd = cmad_n(odd, a, bi, n-2);
            uint64_t p = (uint64_t)a[n-2] * bi;
            odd[n-2] = __builtin_addc((uint32_t)p, 0, carry_odd, &carry_odd);
            odd[n-1] = __builtin_addc((uint32_t)(p >> 32), 0, carry_odd, &carry_odd);
            
            uint32_t carry_val = cmad_n(val, a+1, bi, n-2);
            odd[n-1] += carry_val;
        }

    public:
        inline wide_t(const mont_t& a)                      //// |a|**2
        {
            size_t i = 0, j;
            uint32_t odd[2*n-2];

            // perform |a[i]|*|a[j]| for all j>i
            mul_n(val+2, &a[2], a[0], n-2);
            mul_n(odd,    &a[1], a[0], n);

            #pragma unroll
            while (i < n-4) {
                ++i; mad_row(&val[2*i+2], &odd[2*i], &a[i+1], a[i], n-i-1);
                ++i; qad_row(&odd[2*i], &val[2*i+2], &a[i+1], a[i], n-i);
            }

            asm (
                "v_mul_lo_u32 %0, %2, %3\n\t"
                "v_mul_hi_u32 %1, %2, %3"
                : "=v"(val[2*n-4]), "=v"(val[2*n-3])
                : "v"(a[n-1]), "v"(a[n-3])
            );
            uint64_t p = (uint64_t)a[n-2] * a[n-3];
            uint32_t carry = 0;
            odd[2*n-6] = __builtin_addc((uint32_t)p, odd[2*n-6], 0, &carry);
            odd[2*n-5] = __builtin_addc((uint32_t)(p >> 32), odd[2*n-5], carry, &carry); 
            val[2*n-3] += carry;
            
            asm (
                "v_mul_lo_u32 %0, %2, %3\n\t"
                "v_mul_hi_u32 %1, %2, %3"
                : "=v"(odd[2*n-4]), "=v"(odd[2*n-3])
                : "v"(a[n-1]), "v"(a[n-2])
            );
            
            carry = cadd_n(&val[2], &odd[1], 2*n-4);
            val[2*n-2] = odd[2*n-3] + carry;

            val[0] = 0;
            val[1] = __builtin_addc(odd[0], odd[0], 0, &carry);
            #pragma unroll
	    for (j = 2; j < 2*n-1; j++)
                val[j] = __builtin_addc(val[j], val[j], carry, &carry);
            val[j] = carry;

            i = 0;
            p = (uint64_t)a[0] * a[0];
            val[0] = __builtin_addc((uint32_t)p, val[0], 0, &carry);
            val[1] = __builtin_addc((uint32_t)(p >> 32), val[1], carry, &carry);
	    #pragma unroll
            for (++i; i < n; i++) {
                p = (uint64_t)a[i] * a[i];
                val[2*i] = __builtin_addc((uint32_t)p, val[2*i], carry, &carry);
                val[2*i+1] = __builtin_addc((uint32_t)(p >> 32), val[2*i+1], carry, &carry);
            }
        }
    };

    inline void final_subc(uint32_t carry = 0)
    {
        uint32_t borrow = 0, tmp[n];
        #pragma unroll
        for (size_t i = 0; i < n; i++) {
            uint32_t diff = val[i] - MOD[i];
            tmp[i] = diff - borrow;
            borrow = (val[i] < MOD[i]) | (diff < borrow);
        }
        bool top = (carry == borrow);
        #pragma unroll
        for (size_t i = 0; i < n; i++) {
            val[i] = top ? tmp[i] : val[i];
        }
    }

# if __AMDGCN_WAVEFRONT_SIZE == 64
    using cond_t = uint64_t;
# else
    using cond_t = uint32_t;
# endif

    static __device__ __noinline__ void noop()          { asm("");       }

    inline operator const uint32_t*() const             { return val;    }
    inline operator uint32_t*()                         { return val;    }

public:
    inline uint32_t& operator[](size_t i)               { return val[i]; }
    inline const uint32_t& operator[](size_t i) const   { return val[i]; }
    inline constexpr size_t len() const                 { return n;      }

    inline mont_t() {}
    inline mont_t(const uint32_t p[n])
    {
        for (size_t i=0; i<n; i++)
            val[i] = p[i];
    }

    template<typename... Ts>
    constexpr mont_t(uint32_t a0, Ts... arr) : val{a0, arr...} {}

    __host__ mont_t(const uint64_t p[n/2])
    {
        for (size_t i=0; i<n/2; i++) {
            val[2*i]     = (uint32_t)(p[i]);
            val[2*i + 1] = (uint32_t)(p[i] >> 32);
        }
    }

    inline mont_t& operator<<=(unsigned l)
    {
        uint32_t carry = 0;
        while (l--) {
            val[0] = __builtin_addc(val[0], val[0], 0, &carry);
            for (size_t i = 1; i < n; i++)
                val[i] = __builtin_addc(val[i], val[i], carry, &carry);
            final_subc(carry);
        }

        return *this;
    }
    friend inline mont_t operator<<(mont_t a, unsigned l)
    {   return a <<= l;   }

    inline mont_t& operator>>=(unsigned r)
    {
	    size_t i;
        uint32_t tmp[n + 1];
	    uint32_t carry = 0;
        while (r--) {
            tmp[n] = 0 - (val[0] & 1);
            for (i = 0; i < n; i++) {
                tmp[i] = MOD[i] & tmp[n];
            }
            carry = cadd_n(&val[0], &tmp[0]);
            if (N % 32 == 0) {
                tmp[n] = carry;
            }
            #pragma unroll
            for (i = 0; i < n - 1; i++) {
                val[i] = (tmp[i] >> 1) | (tmp[i+1] << 31);
            }
            if (N % 32 == 0) {
                val[i] = (tmp[i] >> 1) | (tmp[i+1] << 31);
            } else {
                val[i] = tmp[i] >> 1;
            }
        }

        return *this;
    }
    friend inline mont_t operator>>(mont_t a, unsigned r)
    {   return a >>= r;   }

    inline mont_t shfl(uint32_t idx, uint32_t mask = 0xffffffff) const
    {
        mont_t ret;
        // 
        for (size_t i = 0; i < n; i++)
            ret[i] = __shfl(val[i], idx, 32);

        return ret;
    }

    __host__ inline bool is_zero() const
    {
        uint32_t is_zero = val[0] | val[1];

        for (size_t i = 2; i < n; i += 2)
            is_zero |= val[i] | val[i+1];

        return is_zero == 0;
    }

    inline bool is_zero(const mont_t& a) const
    {
        uint32_t is_zero = val[0] | a[0];

        for (size_t i = 1; i < n; i++)
            is_zero |= val[i] | a[i];

        return is_zero == 0;
    } 

    inline mont_t& cneg(bool flag)
    {
        size_t i;
        uint32_t tmp[n], is_zero = val[0];

        uint32_t carry = 0;
        tmp[0] = __builtin_subc(MOD[0], val[0], 0, &carry);
        #pragma unroll
        for (i = 0; i < n; i++) {
            tmp[i] = __builtin_subc(MOD[i], val[i], carry, &carry);
            is_zero |= val[i];
        }
        bool do_negate = flag && (is_zero != 0);
        #pragma unroll
        for (i = 0; i < n; i++)
            val[i] = do_negate ? tmp[i] : val[i];
        return *this;
    }
    static inline mont_t cneg(mont_t a, bool flag)
    {   return a.cneg(flag);   }

    inline void store(uint32_t p[n]) const
    {
        for (size_t i=0; i<n; i++)
            p[i] = val[i];
    }

    inline mont_t& operator+=(const mont_t& b)
    {
        cond_t carry;

        asm("v_add_co_u32   %0, %1, %0, %2"         : "+v"(val[0]), "=s"(carry)
                                                    : "v"(b[0]));
        for (size_t i=1; i<n; i++)
            asm( v_addc_co_u32 "%0, %1, %0, %2, %1" : "+v"(val[i]), "+s"(carry)
                                                    : "v"(b[i]));
        final_sub(carry);

        return *this;
    }
    friend inline mont_t operator+(mont_t a, const mont_t& b)
    {   return a += b;   }

    inline mont_t& operator-=(const mont_t& b)
    {
        cond_t borrow, carry;
        uint32_t tmp[n];

        asm("v_sub_co_u32   %0, %1, %0, %2"         : "+v"(val[0]), "=s"(borrow)
                                                    : "v"(b[0]));
        for (size_t i=1; i<n; i++)
            asm( v_subb_co_u32 "%0, %1, %0, %2, %1" : "+v"(val[i]), "+s"(borrow)
                                                    : "v"(b[i]));

        asm("v_add_co_u32   %0, %1, %2, %3"         : "=v"(tmp[0]), "=s"(carry)
                                                    : "v"(val[0]), "v"(MOD[0]));
        for (size_t i=1; i<n; i++) {
            asm( v_addc_co_u32 "%0, %1, %2, %3, %1" : "=v"(tmp[i]), "+s"(carry)
                                                    : "v"(val[i]), "v"(MOD[i]));
            asm("v_cndmask_b32  %0, %0, %1, %2"     : "+v"(val[i-1])
                                                    : "v"(tmp[i-1]), "s"(borrow));
        }
        asm("v_cndmask_b32  %0, %0, %1, %2"         : "+v"(val[n-1])
                                                    : "v"(tmp[n-1]), "s"(borrow));
        noop();

        return *this;
    }
    friend inline mont_t operator-(mont_t a, const mont_t& b)
    {   return a -= b;   }

    friend inline mont_t operator*(const mont_t& a, const mont_t& b)
    {
        union { uint64_t ul; uint32_t u[2]; };
        cond_t carry = 0;
        mont_t ret;
        size_t i;

        uint64_t mx = b[0];
        ul = mx * a[0];
        ret[0] = u[0];
        for (i=1; i<n; i++) {
            ul = mx * a[i] + u[1];
            ret[i] = u[0];
        }
        auto top = u[1];

        for (size_t j=0; ; ) {
            mx = M0 * ret[0];
            ul = mx * MOD[0] + ret[0];
            for (i=1; i<n; i++) {
                ul = (mx * MOD[i] + u[1]) + ret[i];
                ret[i-1] = u[0];
            }
            if (N%32 != 0) {
                ret[n-1] = u[1] + top;
            } else {
                asm( v_addc_co_u32 "%0, %1, %2, %3, %1" : "=v"(ret[n-1]), "+s"(carry)
                                                        : "v"(u[1]), "v"(top));
            }

            if (++j == n)
                break;

            mx = b[j];
            ul = mx * a[0] + ret[0];
            ret[0] = u[0];
            for (i=1; i<n; i++) {
                ul = (mx * a[i] + u[1]) + ret[i];
                ret[i] = u[0];
            }
            if (N%32 != 0) {
                top = u[1];
            } else {
                asm( v_addc_co_u32 "%0, %1, %2, 0, %1"  : "=v"(top), "+s"(carry)
                                                        : "v"(u[1]));
            }
        }

        ret.final_sub(carry);

        return ret;
    }
    inline mont_t& operator*=(const mont_t& a)
    {   return *this = *this * a;   }

    inline mont_t& sqr()
    {   return *this = *this * *this;   }

    // raise to a variable power, variable in respect to threadIdx,
    // but mind the ^ operator's precedence!
    inline mont_t& operator^=(uint32_t p)
    {   return pow_byref(*this, p);   }
    friend inline mont_t operator^(mont_t a, uint32_t p)
    {   return a ^= p;   }
    inline mont_t operator()(uint32_t p)
    {   return *this^p;   }

    // raise to a constant power, e.g. x^7, to be unrolled at compile time
    inline mont_t& operator^=(int p)
    {   return pow_byref(*this, p);   }
    friend inline mont_t operator^(mont_t a, int p)
    {   return p == 2 ? a *= a : a ^= p;   }
    inline mont_t operator()(int p)
    {   return *this^p;   }
    friend inline mont_t sqr(const mont_t& a)
    {   return a^2;   }

    inline void to()    { mont_t t = RR * *this; *this = t; }
    inline void from()  { mul_by_1(); }

    static inline const mont_t& one()
    {   return *reinterpret_cast<const mont_t*>(ONE);   }
    static inline mont_t one(int or_zero)
    {
        mont_t ret;
        bool flag = (or_zero != 0);

        #pragma unroll
        for (size_t i = 0; i < n; i++) {
            ret[i] = flag ? 0 : ONE[i];
        }
        
        return ret;
    }

    inline void zero()
    {
        if (n%4 == 0) {
            uint4* p = reinterpret_cast<uint4*>(val);
            for (size_t i=0; i<sizeof(val)/(sizeof(uint4)); i++)
                p[i] = uint4{0, 0, 0, 0};
        } else {
            uint64_t* p = reinterpret_cast<uint64_t*>(val);
            for (size_t i=0; i<sizeof(val)/(sizeof(uint64_t)); i++)
                p[i] = 0;
        }
    }

    friend inline mont_t czero(const mont_t& a, int set_z)
    {
        mont_t ret;
        cond_t cond;

        asm("v_cmp_ne_u32  %0, %1, 0"           : "=s"(cond)
                                                : "v"(set_z));
        for (size_t i=0; i<n; i++)
            asm("v_cndmask_b32 %0, %1, 0, %2"   : "=v"(ret[i])
                                                : "v"(a[i]), "s"(cond));
        return ret;
    }

    static inline mont_t csel(const mont_t& a, const mont_t& b, int sel_a)
    {
        mont_t ret;
        cond_t cond;

        asm("v_cmp_ne_u32  %0, %1, 0"           : "=s"(cond)
                                                : "v"(sel_a));
        for (size_t i=0; i<n; i++)
            asm("v_cndmask_b32 %0, %1, %2, %3"  : "=v"(ret[i])
                                                : "v"(b[i]), "v"(a[i]), "s"(cond));
        return ret;
    }

    inline void shfl_bfly(uint32_t laneMask)
    {
        uint32_t idx = (threadIdx.x ^ laneMask) << 2;

        for (size_t i=0; i<n; i++)
            val[i] = __builtin_amdgcn_ds_bpermute(idx, val[i]);
    }

private:
    inline void final_sub(cond_t carry)
    {
        cond_t borrow;
        uint32_t tmp[n];

        asm("v_sub_co_u32   %0, %1, %2, %3"         : "=v"(tmp[0]), "=s"(borrow)
                                                    : "v"(val[0]), "v"(MOD[0]));
        for (size_t i=1; i<n; i++)
            asm( v_subb_co_u32 "%0, %1, %2, %3, %1" : "=v"(tmp[i]), "+s"(borrow)
                                                    : "v"(val[i]), "v"(MOD[i]));
        asm(S_OP(orn2)     "%0, %0, %1"             : "+s"(carry)
                                                    : "s"(borrow));
        for (size_t i=0; i<n; i++)
            asm("v_cndmask_b32  %0, %0, %1, %2"     : "+v"(val[i])
                                                    : "v"(tmp[i]), "s"(carry));
        noop();
    }

    inline void mul_by_1()
    {
        union { uint64_t ul; uint32_t u[2]; };

        for (size_t j=0; j<n; j++) {
            uint64_t mx = M0 * val[0];
            ul = mx * MOD[0] + val[0];
            for (size_t i=1; i<n; i++) {
                ul = (mx * MOD[i] + u[1]) + val[i];
                val[i-1] = u[0];
            }
            val[n-1] = u[1];
        }
    }
private:
    // Euclidean inversion based on https://eprint.iacr.org/2020/972
    // and <blst>/src/no_asm.h.

    struct approx_t  { 
    	uint32_t lo, hi; 
    	__host__ __device__ approx_t& operator=(uint64_t val){
    	     this->lo = static_cast<uint32_t>(val & 0xFFFFFFFF);
    	     this->hi = static_cast<uint32_t>((val>>32) & 0xFFFFFFFF);
    	     return *this;
    	}
    };
    struct factorx_t { uint32_t f0, g0; };

    static inline uint32_t lshift_2(uint32_t hi, uint32_t lo, uint32_t i)
    {
        // uint32_t ret;
        // asm("shf.l.clamp.b32 %0, %1, %2, %3;" : "=r"(ret) : "r"(lo), "r"(hi), "r"(i));
        // return ret;
        if (i == 0) return hi;
	if (i < 32)
            return (hi << i) | (lo >> (32 - i));
	return lo << (i-32);
    }

    static inline void ab_approximation_n(approx_t& a_, const mont_t& a,
                                          approx_t& b_, const mont_t& b)
    {
        size_t i = n-1;

        uint32_t a_hi = a[i], a_lo = a[i-1];
        uint32_t b_hi = b[i], b_lo = b[i-1];
	
	bool flag;
	#pragma unroll
	for (i--; --i;) {
	    flag = (a_hi | b_hi) == 0;
	    a_hi = flag ? a_lo : a_hi;
	    b_hi = flag ? b_lo : b_hi;
    	    a_lo = flag ? a[i] : a_lo;
	    b_lo = flag ? b[i] : b_lo; 
	}	
        
        uint32_t off = __clz(a_hi | b_hi);
        /* |off| can be LIMB_T_BITS if all a[2..]|b[2..] were zeros */

        a_ = approx_t{a[0], lshift_2(a_hi, a_lo, off)};
        b_ = approx_t{b[0], lshift_2(b_hi, b_lo, off)};
    }

    static inline void cswap(uint32_t& a, uint32_t& b, uint32_t mask)
    {
        uint32_t xorm = (a ^ b) & mask;
        a ^= xorm;
        b ^= xorm;
    }

    static inline factorx_t inner_loop_x(approx_t a, approx_t b)
    {
        const uint32_t tid = threadIdx.x&1;
        const uint32_t odd = 0 - tid;

        // val thread calculates |f0|,|f1|, odd - |g0|,|g1|
        uint32_t fg0 = tid^1, fg1 = tid;
        uint32_t alo = a.lo, ahi = a.hi, blo = b.lo, bhi = b.hi;

        // in the odd thread |a| and |b| are in reverse order, compensate...
        cswap(alo, blo, odd);
        cswap(ahi, bhi, odd);

        uint64_t a_full = ((uint64_t)ahi << 32) | alo;
        uint64_t b_full = ((uint64_t)bhi << 32) | blo;

        #pragma unroll 2
        for (uint32_t n = 32-2; n--;) {
            bool is_odd = (a_full&1) != 0;
            bool brw = is_odd && (a_full < b_full);
            uint64_t next_a = a_full;
            if (is_odd) {
                next_a = brw ? (b_full - a_full) : (a_full - b_full);
            }

            b_full = brw ? a_full : b_full;

            uint32_t fgx = brw ? fg0 : fg1;
            fg0  = brw ? fg1 : fg0;
            fg0 = is_odd ? (fg0 - fgx) : fg0;
            fg1 = fgx << 1;
            a_full = next_a >> 1;
        }

        // val thread needs |f0|,|g0|, odd - |g1|,|f1|, in this order
        cswap(fg0, fg1, odd);

        fg1 = __shfl_xor(fg1, 1, 32);

        return factorx_t{fg0, fg1};
    }

    static inline factorx_t inner_loop_x(uint32_t a, uint32_t b, uint32_t n)
    {
        const uint32_t tid = threadIdx.x&1;
        const uint32_t odd = 0 - tid;

        // val thread calculates |f0|,|f1|, odd - |g0|,|g1|
        uint32_t fg0 = tid^1, fg1 = tid;

        // in the odd thread |a| and |b| are in reverse order, compensate...
        cswap(a, b, odd);

        #pragma unroll 1
        while (n--) {
            bool is_odd = (a&1) != 0;
            bool borrow = is_odd && (a < b);
            uint32_t a_next = a;

            if (is_odd) {
                a_next = borrow ? (b - a) : (a - b);
            }
            b = borrow ? a : b;
            uint32_t fgx = borrow ? fg0 : fg1;
            fg0  = borrow ? fg1 : fg0;
            fg0 = borrow ? (fg0 - fgx) : fg0;

            fg1 = fgx << 1;
            a = a_next >> 1;

            //asm("}");
        }

        // we care only about |f1| and |g1| in this subroutine
        fg0 = __shfl_xor(fg1, 1, 32);

        return factorx_t{fg1, fg0};
    }

    template<typename vec_t>
    static inline uint32_t cneg_v(vec_t& ret, const vec_t& a, uint32_t neg)
    {
        const size_t n = sizeof(vec_t)/sizeof(uint32_t);

        // asm("xor.b32 %0, %1, %2;"    : "=r"(ret[0]) : "r"(a[0]), "r"(neg));
        // asm("add.cc.u32 %0, %0, %1;" : "+r"(ret[0]) : "r"(neg&1));
        // for (size_t i=1; i<n; i++)
        //     asm("xor.b32 %0, %1, %2; addc.cc.u32 %0, %0, 0;"
        //         : "=r"(ret[i]) : "r"(a[i]), "r"(neg));

        // uint32_t sign;
        // asm("shr.s32 %0, %1, 31;" : "=r"(sign) : "r"(ret[n-1]));
        
        ret[0] = a[0] ^ neg;
        uint32_t carry = 0;
        ret[0] = __builtin_addc(ret[0], neg & 1, 0, &carry);
        #pragma unroll
        for (size_t i = 1; i < n; i++) {
            ret[i] = a[i] ^ neg;
            ret[i] = __builtin_addc(ret[i], 0, carry, &carry);
        }

        uint32_t sign = ((int32_t)ret[n-1]) >> 31;
        return sign;
    }

    static inline void smul_n_shift_x(mont_t& a, uint32_t& f_,
                                      mont_t& b, uint32_t& g_)
    {
        mont_t val, odd;
        uint32_t neg;
        size_t i;

        neg = ((int32_t)f_) >> 31;
        auto f = (f_ ^ neg) - neg;
        (void)cneg_v(a, a, neg);
        mul_n(&val[0], &a[0], f);
        mul_n(&odd[0], &a[1], f);
        odd[n-1] -= f & neg;

        neg = ((int32_t)g_) >> 31;
        auto g = (g_ ^ neg) - neg;
        (void)cneg_v(b, b, neg);
        uint32_t c1 = cmad_n(&val[0], &b[0], g);
        odd[n-1] += c1;
        cmad_n(&odd[0], &b[1], g);
        odd[n-1] -= g & neg;

        c1 = cadd_n(&val[1], &odd[0], n-1);
        odd[n-1] += c1;

        #pragma unroll
        for (i = 0; i < n-1; i++) {
            val[i] = (val[i] >> 30) | (val[i+1] << 2);
        }
        val[i] = (val[i] >> 30) | (odd[i] << 2);

        neg = ((int32_t)odd[i]) >> 31;
        f_ = (f_ ^ neg) - neg;
        g_ = (g_ ^ neg) - neg;
        (void)cneg_v(a, val, neg);
    }

    static inline uint32_t smul_2x(wide_t& u, uint32_t f,
                                   wide_t& v, uint32_t g)
    {
        wide_t val, odd;
        uint32_t neg;

        neg = ((int32_t)f) >> 31;
        f = (f ^ neg) - neg;
        neg = cneg_v(u, u, neg);
        mul_n(&val[0], &u[0], f, 2*n);
        mul_n(&odd[0], &u[1], f, 2*n);
        odd[2*n-1] -= f & neg;

        neg = ((int32_t)g) >> 31;
        g = (g ^ neg) - neg;
        neg = cneg_v(v, v, neg);
        uint32_t c1 = cmad_n(&val[0], &v[0], g, 2*n);
        odd[2*n-1] += c1;
        cmad_n(&odd[0], &v[1], g, 2*n);
        odd[2*n-1] -= g & neg;

        u[0] = val[0];
        u[1] = __builtin_addc(val[1], odd[0], 0, &c1);
        #pragma unroll
        for (size_t i = 2; i < 2*n; i++)
            u[i] = __builtin_addc(val[i], odd[i-1], c1, &c1);
        odd[2*n-1] += c1;

        return odd[2*n-1];
        
    }

protected:
    /*
     * val thread holds |a| and |u|, while odd one - |b| and |v|. They need
     * to exchange the values, but then perform the multiplications in
     * parallel. The improvement [for 38x-bit moduli] is >20% and ~5.5KB
     * code size reduction in comparison to a single-threaded version.
     */
    static inline mont_t ct_inverse_mod_x(const mont_t& inp)
    {
	/*if(threadIdx.x < 8)
            printf("lane %d partner %d\n", threadIdx.x,
                __shfl_xor(threadIdx.x,1,32));*/
        if (N%32 != 0 && MOD == MODx) asm("s_trap 2;");

        const uint32_t tid = threadIdx.x&1;
        const uint32_t nbits = 2*n*32;

        mont_t a_b = csel(MOD, inp, tid);
        wide_t u_v, v_u;

        u_v[0] = tid^1;
        v_u[0] = tid;
        #pragma unroll
        for (size_t i=1; i<2*n; i++)
            u_v[i] = v_u[i] = 0;

        #pragma unroll 1
        for (uint32_t i=0; i<nbits/(32-2); i++) {
            mont_t b_a = shfl_xor(a_b);
            approx_t a_, b_;
            ab_approximation_n(a_, a_b, b_, b_a);
            auto fg = inner_loop_x(a_, b_);
            smul_n_shift_x(a_b, fg.f0, b_a, fg.g0);
            (void)smul_2x(u_v, fg.f0, v_u, fg.g0);
            v_u = shfl_xor(u_v);
        }

        uint32_t b_a = __shfl_xor(a_b[0], 1, 32);
        auto fg = inner_loop_x(a_b[0], b_a, nbits%30);
        auto top = (int32_t)smul_2x(u_v, fg.f0, v_u, fg.g0);
        //if (threadIdx.x == 0)
	  //  printf("top in ct_inverse_mod_x %d\n", top);
	assert(top==0|top==1|top==-1 && "top error in ct_inverse_mod_x");
	bool flag = top < 0;
        if (flag) {
	    uint32_t carry = 0;
            u_v[n] = __builtin_addc(u_v[n], MODx[0], 0, &carry);
            #pragma unroll
            for (size_t i = 1; i < n; i++) {
                u_v[n+i] = __builtin_addc(u_v[n+i], MODx[i], carry, &carry);
            }
            top += (int32_t)carry;
        }
        
        auto sign = 0 - top;        /* top is 1, 0 or -1 */
        top |= sign;
        #pragma unroll
        for (size_t i=0; i<n; i++)
            a_b[i] = MODx[i] & top;
        sign >>= 31;
        (void)cneg_v(a_b, a_b, sign);
        cadd_n(&u_v[n], &a_b[0]);

        mont_t ret = u_v;
        ret.to();
        return ret;
    }

public:
    inline mont_t reciprocal() const
    {
        bool a_zero = is_zero();
        mont_t a = csel(ONE, val, a_zero);
	if (a_zero) {
            assert("csel should return 1 for zero input" && a.is_one());
        }
        mont_t b = shfl_xor(a);
        a *= b;                     // a*b
        a = ct_inverse_mod_x(a);    // 1/(a*b)
        a *= b;                     // b/(a*b) == 1/a
	if (!a_zero) {
	    assert("a*a^-1 should be 1" && (a*val).is_one());
	}
        return czero(a, a_zero);
    }
    friend inline mont_t operator/(int one, const mont_t& a)
    {   if (one != 1) asm("s_trap 2;"); return a.reciprocal();   }
    friend inline mont_t operator/(const mont_t& a, const mont_t& b)
    {   return a * b.reciprocal();   }
    inline mont_t& operator/=(const mont_t& a)
    {   return *this *= a.reciprocal();   }
    
};

# undef inline
# undef v_subb_co_u32
# undef v_addc_co_u32
# undef S_OP
# undef __MONT_T_XSTR
# undef __MONT_T_STR
#endif
