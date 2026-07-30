// OpenCL contract for the BTC Puzzle scanner.
//
// This kernel is intentionally small and memory-bounded. The Python host passes
// seven hash160 targets: index 0 is the main puzzle target, indexes 1..6 are
// proof-of-work addresses. A production secp256k1 implementation should replace
// the placeholder body with:
//   private key -> public key -> HASH160(compressed/uncompressed pubkey)
// and store the matching private key limbs in found_keys[index * 8..].

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
    if (gid >= count) {
        return;
    }

    // Placeholder 256-bit increment so host/device plumbing can be validated.
    // Full secp256k1 math will consume these words as the candidate key.
    uint key[8] = {start0, start1, start2, start3, start4, start5, start6, start7};
    ulong carry = gid;
    for (int i = 7; i >= 0; --i) {
        ulong sum = (ulong)key[i] + (carry & 0xffffffffUL);
        key[i] = (uint)sum;
        carry = (carry >> 32) + (sum >> 32);
    }

    (void)target_hash160;
    (void)proof_hash160s;
    (void)found_flags;
    (void)found_keys;
}
