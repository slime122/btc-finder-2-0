// secp256k1 scanner kernel for OpenCL C 1.2.
//
// Conservative implementation for Apple OpenCL / Intel integrated GPUs:
// - 256-bit field values are 8 uint words, big-endian.
// - Field multiplication uses modular double-and-add to avoid 512-bit temporaries.
// - Scalar multiplication uses affine double-and-add with Fermat inversions.
//
// This is intentionally correctness-first. It is suitable as the real crypto
// baseline for the Python pipeline; the next performance step is replacing
// affine multiplication with Jacobian/windowed multiplication.

#define U32_MAXV 0xffffffffU

__constant uint FIELD_P[8] = {
    0xffffffffU, 0xffffffffU, 0xffffffffU, 0xffffffffU,
    0xffffffffU, 0xffffffffU, 0xfffffffeU, 0xfffffc2fU
};

__constant uint FIELD_P_MINUS_2[8] = {
    0xffffffffU, 0xffffffffU, 0xffffffffU, 0xffffffffU,
    0xffffffffU, 0xffffffffU, 0xfffffffeU, 0xfffffc2dU
};

__constant uint GROUP_N[8] = {
    0xffffffffU, 0xffffffffU, 0xffffffffU, 0xfffffffeU,
    0xbaaedce6U, 0xaf48a03bU, 0xbfd25e8cU, 0xd0364141U
};

__constant uint GX[8] = {
    0x79be667eU, 0xf9dcbbacU, 0x55a06295U, 0xce870b07U,
    0x029bfcdbU, 0x2dce28d9U, 0x59f2815bU, 0x16f81798U
};

__constant uint GY[8] = {
    0x483ada77U, 0x26a3c465U, 0x5da4fbfcU, 0x0e1108a8U,
    0xfd17b448U, 0xa6855419U, 0x9c47d08fU, 0xfb10d4b8U
};

inline void u256_copy(uint r[8], const uint a[8]) {
    for (int i = 0; i < 8; i++) r[i] = a[i];
}

inline void u256_zero(uint r[8]) {
    for (int i = 0; i < 8; i++) r[i] = 0U;
}

inline int u256_is_zero(const uint a[8]) {
    uint v = 0U;
    for (int i = 0; i < 8; i++) v |= a[i];
    return v == 0U;
}

inline int u256_cmp(const uint a[8], const uint b[8]) {
    for (int i = 0; i < 8; i++) {
        if (a[i] > b[i]) return 1;
        if (a[i] < b[i]) return -1;
    }
    return 0;
}

inline int u256_cmp_p(const uint a[8]) {
    for (int i = 0; i < 8; i++) {
        if (a[i] > FIELD_P[i]) return 1;
        if (a[i] < FIELD_P[i]) return -1;
    }
    return 0;
}

inline int u256_cmp_n(const uint a[8]) {
    for (int i = 0; i < 8; i++) {
        if (a[i] > GROUP_N[i]) return 1;
        if (a[i] < GROUP_N[i]) return -1;
    }
    return 0;
}

inline uint u256_add_raw(uint r[8], const uint a[8], const uint b[8]) {
    ulong carry = 0UL;
    for (int i = 7; i >= 0; i--) {
        ulong s = (ulong)a[i] + (ulong)b[i] + carry;
        r[i] = (uint)s;
        carry = s >> 32;
    }
    return (uint)carry;
}

inline uint u256_sub_raw(uint r[8], const uint a[8], const uint b[8]) {
    ulong borrow = 0UL;
    for (int i = 7; i >= 0; i--) {
        ulong av = (ulong)a[i];
        ulong bv = (ulong)b[i] + borrow;
        if (av >= bv) {
            r[i] = (uint)(av - bv);
            borrow = 0UL;
        } else {
            r[i] = (uint)((1UL << 32) + av - bv);
            borrow = 1UL;
        }
    }
    return (uint)borrow;
}

inline uint u256_sub_p(uint r[8], const uint a[8]) {
    ulong borrow = 0UL;
    for (int i = 7; i >= 0; i--) {
        ulong av = (ulong)a[i];
        ulong bv = (ulong)FIELD_P[i] + borrow;
        if (av >= bv) {
            r[i] = (uint)(av - bv);
            borrow = 0UL;
        } else {
            r[i] = (uint)((1UL << 32) + av - bv);
            borrow = 1UL;
        }
    }
    return (uint)borrow;
}

inline uint u256_add_p(uint r[8], const uint a[8]) {
    ulong carry = 0UL;
    for (int i = 7; i >= 0; i--) {
        ulong s = (ulong)a[i] + (ulong)FIELD_P[i] + carry;
        r[i] = (uint)s;
        carry = s >> 32;
    }
    return (uint)carry;
}

inline void fp_add(uint r[8], const uint a[8], const uint b[8]) {
    uint t[8];
    uint carry = u256_add_raw(t, a, b);
    if (carry || u256_cmp_p(t) >= 0) {
        uint psub[8];
        u256_sub_p(psub, t);
        u256_copy(r, psub);
    } else {
        u256_copy(r, t);
    }
}

inline void fp_sub(uint r[8], const uint a[8], const uint b[8]) {
    uint t[8];
    uint borrow = u256_sub_raw(t, a, b);
    if (borrow) {
        uint added[8];
        u256_add_p(added, t);
        u256_copy(r, added);
    } else {
        u256_copy(r, t);
    }
}

inline void fp_double(uint r[8], const uint a[8]) {
    fp_add(r, a, a);
}

inline int u256_get_bit(const uint a[8], int bit_index) {
    int word = 7 - (bit_index >> 5);
    int shift = bit_index & 31;
    return (int)((a[word] >> shift) & 1U);
}

inline int p_minus_2_get_bit(int bit_index) {
    int word = 7 - (bit_index >> 5);
    int shift = bit_index & 31;
    return (int)((FIELD_P_MINUS_2[word] >> shift) & 1U);
}

inline void fp_mul(uint r[8], const uint a_in[8], const uint b_in[8]) {
    uint acc[8];
    uint a[8];
    u256_zero(acc);
    u256_copy(a, a_in);

    for (int bit = 0; bit < 256; bit++) {
        if (u256_get_bit(b_in, bit)) {
            uint tmp[8];
            fp_add(tmp, acc, a);
            u256_copy(acc, tmp);
        }
        uint dbl[8];
        fp_double(dbl, a);
        u256_copy(a, dbl);
    }
    u256_copy(r, acc);
}

inline void fp_square(uint r[8], const uint a[8]) {
    fp_mul(r, a, a);
}

inline void fp_inv(uint r[8], const uint a[8]) {
    uint result[8];
    uint base[8];
    u256_zero(result);
    result[7] = 1U;
    u256_copy(base, a);

    for (int bit = 255; bit >= 0; bit--) {
        uint sq[8];
        fp_square(sq, result);
        u256_copy(result, sq);
        if (p_minus_2_get_bit(bit)) {
            uint mul[8];
            fp_mul(mul, result, base);
            u256_copy(result, mul);
        }
    }
    u256_copy(r, result);
}

inline void point_set_g(uint x[8], uint y[8], int *inf) {
    for (int i = 0; i < 8; i++) {
        x[i] = GX[i];
        y[i] = GY[i];
    }
    *inf = 0;
}

inline void point_double(uint rx[8], uint ry[8], int *rinf,
                         const uint x[8], const uint y[8], int inf) {
    if (inf || u256_is_zero(y)) {
        u256_zero(rx);
        u256_zero(ry);
        *rinf = 1;
        return;
    }

    uint x2[8], three_x2[8], two_y[8], inv[8], lambda[8];
    uint lambda2[8], two_x[8], x3[8], x_minus_x3[8], y3[8], tmp[8];

    fp_square(x2, x);
    fp_add(tmp, x2, x2);
    fp_add(three_x2, tmp, x2);
    fp_double(two_y, y);
    fp_inv(inv, two_y);
    fp_mul(lambda, three_x2, inv);

    fp_square(lambda2, lambda);
    fp_double(two_x, x);
    fp_sub(x3, lambda2, two_x);
    fp_sub(x_minus_x3, x, x3);
    fp_mul(tmp, lambda, x_minus_x3);
    fp_sub(y3, tmp, y);

    u256_copy(rx, x3);
    u256_copy(ry, y3);
    *rinf = 0;
}

inline void point_add(uint rx[8], uint ry[8], int *rinf,
                      const uint x1[8], const uint y1[8], int inf1,
                      const uint x2[8], const uint y2[8], int inf2) {
    if (inf1) {
        u256_copy(rx, x2);
        u256_copy(ry, y2);
        *rinf = inf2;
        return;
    }
    if (inf2) {
        u256_copy(rx, x1);
        u256_copy(ry, y1);
        *rinf = inf1;
        return;
    }

    if (u256_cmp(x1, x2) == 0) {
        if (u256_cmp(y1, y2) == 0) {
            point_double(rx, ry, rinf, x1, y1, inf1);
        } else {
            u256_zero(rx);
            u256_zero(ry);
            *rinf = 1;
        }
        return;
    }

    uint dy[8], dx[8], inv[8], lambda[8], lambda2[8], x3[8], y3[8], tmp[8];
    fp_sub(dy, y2, y1);
    fp_sub(dx, x2, x1);
    fp_inv(inv, dx);
    fp_mul(lambda, dy, inv);
    fp_square(lambda2, lambda);
    fp_sub(tmp, lambda2, x1);
    fp_sub(x3, tmp, x2);
    fp_sub(tmp, x1, x3);
    fp_mul(tmp, lambda, tmp);
    fp_sub(y3, tmp, y1);

    u256_copy(rx, x3);
    u256_copy(ry, y3);
    *rinf = 0;
}

inline void scalar_mul_g(uint rx[8], uint ry[8], int *rinf, const uint k[8]) {
    uint qx[8], qy[8], tx[8], ty[8];
    int qinf = 1;
    u256_zero(qx);
    u256_zero(qy);

    for (int bit = 255; bit >= 0; bit--) {
        point_double(tx, ty, &qinf, qx, qy, qinf);
        u256_copy(qx, tx);
        u256_copy(qy, ty);

        if (u256_get_bit(k, bit)) {
            uint gx[8], gy[8];
            int ginf;
            point_set_g(gx, gy, &ginf);
            point_add(tx, ty, &qinf, qx, qy, qinf, gx, gy, ginf);
            u256_copy(qx, tx);
            u256_copy(qy, ty);
        }
    }

    u256_copy(rx, qx);
    u256_copy(ry, qy);
    *rinf = qinf;
}

inline uint rotr32(uint x, uint n) {
    return (x >> n) | (x << (32U - n));
}

inline uint ch32(uint x, uint y, uint z) {
    return (x & y) ^ (~x & z);
}

inline uint maj32(uint x, uint y, uint z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

inline uint sha_ep0(uint x) {
    return rotr32(x, 2U) ^ rotr32(x, 13U) ^ rotr32(x, 22U);
}

inline uint sha_ep1(uint x) {
    return rotr32(x, 6U) ^ rotr32(x, 11U) ^ rotr32(x, 25U);
}

inline uint sha_sig0(uint x) {
    return rotr32(x, 7U) ^ rotr32(x, 18U) ^ (x >> 3);
}

inline uint sha_sig1(uint x) {
    return rotr32(x, 17U) ^ rotr32(x, 19U) ^ (x >> 10);
}

__constant uint SHA_K[64] = {
    0x428a2f98U,0x71374491U,0xb5c0fbcfU,0xe9b5dba5U,0x3956c25bU,0x59f111f1U,0x923f82a4U,0xab1c5ed5U,
    0xd807aa98U,0x12835b01U,0x243185beU,0x550c7dc3U,0x72be5d74U,0x80deb1feU,0x9bdc06a7U,0xc19bf174U,
    0xe49b69c1U,0xefbe4786U,0x0fc19dc6U,0x240ca1ccU,0x2de92c6fU,0x4a7484aaU,0x5cb0a9dcU,0x76f988daU,
    0x983e5152U,0xa831c66dU,0xb00327c8U,0xbf597fc7U,0xc6e00bf3U,0xd5a79147U,0x06ca6351U,0x14292967U,
    0x27b70a85U,0x2e1b2138U,0x4d2c6dfcU,0x53380d13U,0x650a7354U,0x766a0abbU,0x81c2c92eU,0x92722c85U,
    0xa2bfe8a1U,0xa81a664bU,0xc24b8b70U,0xc76c51a3U,0xd192e819U,0xd6990624U,0xf40e3585U,0x106aa070U,
    0x19a4c116U,0x1e376c08U,0x2748774cU,0x34b0bcb5U,0x391c0cb3U,0x4ed8aa4aU,0x5b9cca4fU,0x682e6ff3U,
    0x748f82eeU,0x78a5636fU,0x84c87814U,0x8cc70208U,0x90befffaU,0xa4506cebU,0xbef9a3f7U,0xc67178f2U
};

inline void sha256_33(const uchar msg[33], uchar out[32]) {
    uint w[64];
    for (int i = 0; i < 64; i++) w[i] = 0U;

    for (int i = 0; i < 33; i++) {
        int wi = i >> 2;
        int sh = 24 - ((i & 3) << 3);
        w[wi] |= ((uint)msg[i]) << sh;
    }
    w[8] |= 0x00800000U;
    w[15] = 33U * 8U;

    for (int i = 16; i < 64; i++) {
        w[i] = sha_sig1(w[i - 2]) + w[i - 7] + sha_sig0(w[i - 15]) + w[i - 16];
    }

    uint a = 0x6a09e667U, b = 0xbb67ae85U, c = 0x3c6ef372U, d = 0xa54ff53aU;
    uint e = 0x510e527fU, f = 0x9b05688cU, g = 0x1f83d9abU, h = 0x5be0cd19U;

    for (int i = 0; i < 64; i++) {
        uint t1 = h + sha_ep1(e) + ch32(e, f, g) + SHA_K[i] + w[i];
        uint t2 = sha_ep0(a) + maj32(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    uint hs[8] = {
        0x6a09e667U + a, 0xbb67ae85U + b, 0x3c6ef372U + c, 0xa54ff53aU + d,
        0x510e527fU + e, 0x9b05688cU + f, 0x1f83d9abU + g, 0x5be0cd19U + h
    };
    for (int i = 0; i < 8; i++) {
        out[i * 4 + 0] = (uchar)(hs[i] >> 24);
        out[i * 4 + 1] = (uchar)(hs[i] >> 16);
        out[i * 4 + 2] = (uchar)(hs[i] >> 8);
        out[i * 4 + 3] = (uchar)(hs[i]);
    }
}

inline uint rol32(uint x, uint n) {
    return (x << n) | (x >> (32U - n));
}

inline uint rip_f(int j, uint x, uint y, uint z) {
    if (j < 16) return x ^ y ^ z;
    if (j < 32) return (x & y) | (~x & z);
    if (j < 48) return (x | ~y) ^ z;
    if (j < 64) return (x & z) | (y & ~z);
    return x ^ (y | ~z);
}

inline uint rip_k(int j) {
    if (j < 16) return 0x00000000U;
    if (j < 32) return 0x5a827999U;
    if (j < 48) return 0x6ed9eba1U;
    if (j < 64) return 0x8f1bbcdcU;
    return 0xa953fd4eU;
}

inline uint rip_kk(int j) {
    if (j < 16) return 0x50a28be6U;
    if (j < 32) return 0x5c4dd124U;
    if (j < 48) return 0x6d703ef3U;
    if (j < 64) return 0x7a6d76e9U;
    return 0x00000000U;
}

__constant int RIP_R[80] = {
     0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,
     7, 4,13, 1,10, 6,15, 3,12, 0, 9, 5, 2,14,11, 8,
     3,10,14, 4, 9,15, 8, 1, 2, 7, 0, 6,13,11, 5,12,
     1, 9,11,10, 0, 8,12, 4,13, 3, 7,15,14, 5, 6, 2,
     4, 0, 5, 9, 7,12, 2,10,14, 1, 3, 8,11, 6,15,13
};

__constant int RIP_RR[80] = {
     5,14, 7, 0, 9, 2,11, 4,13, 6,15, 8, 1,10, 3,12,
     6,11, 3, 7, 0,13, 5,10,14,15, 8,12, 4, 9, 1, 2,
    15, 5, 1, 3, 7,14, 6, 9,11, 8,12, 2,10, 0, 4,13,
     8, 6, 4, 1, 3,11,15, 0, 5,12, 2,13, 9, 7,10,14,
    12,15,10, 4, 1, 5, 8, 7, 6, 2,13,14, 0, 3, 9,11
};

__constant int RIP_S[80] = {
    11,14,15,12, 5, 8, 7, 9,11,13,14,15, 6, 7, 9, 8,
     7, 6, 8,13,11, 9, 7,15, 7,12,15, 9,11, 7,13,12,
    11,13, 6, 7,14, 9,13,15,14, 8,13, 6, 5,12, 7, 5,
    11,12,14,15,14,15, 9, 8, 9,14, 5, 6, 8, 6, 5,12,
     9,15, 5,11, 6, 8,13,12, 5,12,13,14,11, 8, 5, 6
};

__constant int RIP_SS[80] = {
     8, 9, 9,11,13,15,15, 5, 7, 7, 8,11,14,14,12, 6,
     9,13,15, 7,12, 8, 9,11, 7, 7,12, 7, 6,15,13,11,
     9, 7,15,11, 8, 6, 6,14,12,13, 5,14,13,13, 7, 5,
    15, 5, 8,11,14,14, 6,14, 6, 9,12, 9,12, 5,15, 8,
     8, 5,12, 9,12, 5,14, 6, 8,13, 6, 5,15,13,11,11
};

inline void ripemd160_32(const uchar msg[32], uchar out[20]) {
    uint x[16];
    for (int i = 0; i < 16; i++) x[i] = 0U;
    for (int i = 0; i < 32; i++) {
        int wi = i >> 2;
        int sh = (i & 3) << 3;
        x[wi] |= ((uint)msg[i]) << sh;
    }
    x[8] = 0x00000080U;
    x[14] = 32U * 8U;

    uint h0 = 0x67452301U, h1 = 0xefcdab89U, h2 = 0x98badcfeU, h3 = 0x10325476U, h4 = 0xc3d2e1f0U;
    uint al = h0, bl = h1, cl = h2, dl = h3, el = h4;
    uint ar = h0, br = h1, cr = h2, dr = h3, er = h4;

    for (int j = 0; j < 80; j++) {
        uint t = rol32(al + rip_f(j, bl, cl, dl) + x[RIP_R[j]] + rip_k(j), (uint)RIP_S[j]) + el;
        al = el; el = dl; dl = rol32(cl, 10U); cl = bl; bl = t;

        t = rol32(ar + rip_f(79 - j, br, cr, dr) + x[RIP_RR[j]] + rip_kk(j), (uint)RIP_SS[j]) + er;
        ar = er; er = dr; dr = rol32(cr, 10U); cr = br; br = t;
    }

    uint t = h1 + cl + dr;
    h1 = h2 + dl + er;
    h2 = h3 + el + ar;
    h3 = h4 + al + br;
    h4 = h0 + bl + cr;
    h0 = t;

    uint hs[5] = {h0, h1, h2, h3, h4};
    for (int i = 0; i < 5; i++) {
        out[i * 4 + 0] = (uchar)(hs[i]);
        out[i * 4 + 1] = (uchar)(hs[i] >> 8);
        out[i * 4 + 2] = (uchar)(hs[i] >> 16);
        out[i * 4 + 3] = (uchar)(hs[i] >> 24);
    }
}

inline void make_pubkey_bytes(uchar out[33], const uint x[8], const uint y[8]) {
    out[0] = (uchar)((y[7] & 1U) ? 0x03U : 0x02U);
    for (int i = 0; i < 8; i++) {
        out[1 + i * 4 + 0] = (uchar)(x[i] >> 24);
        out[1 + i * 4 + 1] = (uchar)(x[i] >> 16);
        out[1 + i * 4 + 2] = (uchar)(x[i] >> 8);
        out[1 + i * 4 + 3] = (uchar)(x[i]);
    }
}

inline int hash160_equal_global(const uchar h[20], __global const uchar *target) {
    uint diff = 0U;
    for (int i = 0; i < 20; i++) diff |= (uint)(h[i] ^ target[i]);
    return diff == 0U;
}

inline void store_key(__global uint *found_flags, __global uint *found_keys, int index, const uint key[8]) {
    for (int i = 0; i < 8; i++) {
        found_keys[index * 8 + i] = key[i];
    }
    atomic_xchg((volatile __global int *)&found_flags[index], 1);
}

__kernel void scan_range(
    const uint start0,
    const uint start1,
    const uint start2,
    const uint start3,
    const uint start4,
    const uint start5,
    const uint start6,
    const uint start7,
    const ulong count,
    __global const uchar *target_hash160,
    __global const uchar *proof_hash160s,
    __global uint *found_flags,
    __global uint *found_keys
) {
    const ulong gid = get_global_id(0);
    if (gid >= count) return;

    uint key[8] = {start0, start1, start2, start3, start4, start5, start6, start7};
    ulong carry = gid;
    for (int i = 7; i >= 0; i--) {
        ulong sum = (ulong)key[i] + (carry & 0xffffffffUL);
        key[i] = (uint)sum;
        carry = (carry >> 32) + (sum >> 32);
    }
    if (u256_is_zero(key) || u256_cmp_n(key) >= 0) return;

    uint x[8], y[8];
    int inf;
    scalar_mul_g(x, y, &inf, key);
    if (inf) return;

    uchar pub[33];
    uchar sha[32];
    uchar hash160[20];
    make_pubkey_bytes(pub, x, y);
    sha256_33(pub, sha);
    ripemd160_32(sha, hash160);

    if (hash160_equal_global(hash160, target_hash160)) {
        store_key(found_flags, found_keys, 0, key);
    }

    for (int target = 0; target < 6; target++) {
        if (hash160_equal_global(hash160, proof_hash160s + target * 20)) {
            store_key(found_flags, found_keys, target + 1, key);
        }
    }
}
