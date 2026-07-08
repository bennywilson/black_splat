// gaussian_splat_radix.wgsl
//
// GPU least-significant-digit radix sort of the splat index buffer, keyed by
// view-space depth so the draw can composite back-to-front. Reduce-Scan-Scan-Scatter
// radix: 32-bit keys, four 8-bit digit passes, each O(n).
//
// Note: One-Sweep / decoupled-look-back radix requires cross-workgroup forward-progress 
// guarantees the WebGPU spec does not provide and would risk deadlock on the web backend.  
// Instead every phase is its own dispatch (its own pass boundary), and all scans run in workgroup shared memory
// with no subgroup intrinsics (also not guaranteed on the web).
//
// Phases per sort (the host drives the dispatch order; see gaussian_splat.rs):
//   cs_compute_keys  - once: depth -> sortable u32 key, payload = splat index.
//   per digit (shift 0/8/16/24):
//     cs_histogram   - per-tile bucket counts, written bucket-major into g_hist.
//     cs_scan_reduce - exclusive-scan each 256-wide block of g_hist in place,
//                      emit each block's total to g_block_sums.
//     cs_scan_spine  - one workgroup: exclusive-scan g_block_sums (any length).
//     cs_scan_add    - add each block's scanned base back into g_hist.  g_hist now
//                      holds, at [bucket*num_tiles + tile], the global output base
//                      of that tile's run of `bucket` elements.
//     cs_scatter     - per tile: stable local sort (8x 1-bit split) then scatter
//                      to global positions; ping-pongs keys/vals to the *_out bufs.

const WG: u32 = 256u;        // workgroup size == tile size (one element per thread)
const RADIX: u32 = 256u;     // 8-bit digit
const RADIX_MASK: u32 = 255u;

struct SortGlobals {
    zc: vec4<f32>,        // third row of the view matrix: depth = dot(zc, vec4(pos, 1))
    num_elements: u32,    // real splat count (no padding)
    num_tiles: u32,       // ceil(num_elements / WG)
    _pad0: u32,
    _pad1: u32,
};

struct Splat {
    position: vec4<f32>,
    scale_opacity: vec4<f32>,
    rotation: vec4<f32>,
    sh0: vec4<f32>,
    sh_rest: array<f32, 24>,
};

// Per-pass uniform, supplied via dynamic offset: the digit's bit shift (0/8/16/24).
struct PassInfo {
    shift: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
};

@group(0) @binding(0) var<uniform> g: SortGlobals;
@group(0) @binding(1) var<storage, read> g_splats: array<Splat>;
@group(0) @binding(2) var<storage, read> g_keys_in: array<u32>;
@group(0) @binding(3) var<storage, read_write> g_keys_out: array<u32>;
@group(0) @binding(4) var<storage, read> g_vals_in: array<u32>;
@group(0) @binding(5) var<storage, read_write> g_vals_out: array<u32>;
@group(0) @binding(6) var<storage, read_write> g_hist: array<u32>;
@group(0) @binding(7) var<storage, read_write> g_block_sums: array<u32>;
@group(1) @binding(0) var<uniform> pass_info: PassInfo;

// Map an IEEE-754 float to a u32 whose unsigned order matches the float's order:
// flip the sign bit for positives, flip all bits for negatives.  An ascending u32
// sort then orders the floats ascending.
fn float_key(f: f32) -> u32 {
    let u = bitcast<u32>(f);
    let mask = select(0x80000000u, 0xFFFFFFFFu, (u & 0x80000000u) != 0u);
    return u ^ mask;
}

// Shared scratch used by several entry points.  Sizes are written as the literal
// 256 (== WG == RADIX): naga 0.20 otherwise decorates a const-sized workgroup
// array with an ArrayStride, which Vulkan rejects for the Workgroup storage class.
var<workgroup> s_scan: array<u32, 256>;       // scan working buffer
var<workgroup> s_keys: array<u32, 256>;
var<workgroup> s_vals: array<u32, 256>;
var<workgroup> s_count: array<u32, 256>;      // per-digit tile counts (histogram)

// Hillis-Steele exclusive scan of s_scan[0..WG].  On return s_scan holds the
// exclusive prefix sums and the function returns the total sum.
fn excl_scan_wg(li: u32) -> u32 {
    let mine = s_scan[li];
    workgroupBarrier();
    var offset = 1u;
    loop {
        if (offset >= WG) { break; }
        var add = 0u;
        if (li >= offset) { add = s_scan[li - offset]; }
        workgroupBarrier();
        if (li >= offset) { s_scan[li] = s_scan[li] + add; }
        workgroupBarrier();
        offset = offset << 1u;
    }
    let total = s_scan[WG - 1u];
    workgroupBarrier();
    let inclusive = s_scan[li];
    workgroupBarrier();
    s_scan[li] = inclusive - mine;   // each lane touches only its own slot
    workgroupBarrier();
    return total;
}

// Hillis-Steele inclusive max-scan of s_scan[0..WG], in place.
fn incl_max_scan_wg(li: u32) {
    workgroupBarrier();
    var offset = 1u;
    loop {
        if (offset >= WG) { break; }
        var v = 0u;
        if (li >= offset) { v = s_scan[li - offset]; }
        workgroupBarrier();
        if (li >= offset) { s_scan[li] = max(s_scan[li], v); }
        workgroupBarrier();
        offset = offset << 1u;
    }
}

// Stable local sort of s_keys/s_vals[0..WG] by the 8-bit digit at `shift`, as 8
// sequential 1-bit stable splits.  O(WG log WG); no atomics.
fn local_sort(li: u32, shift: u32) {
    var bit = 0u;
    loop {
        if (bit >= 8u) { break; }
        let b = shift + bit;
        let my_key = s_keys[li];
        let my_val = s_vals[li];
        let is_one = (my_key >> b) & 1u;

        s_scan[li] = 1u - is_one;                 // predicate: this lane is a zero
        let total_zeros = excl_scan_wg(li);
        let zeros_before = s_scan[li];            // exclusive prefix of zeros
        var dest = zeros_before;                  // zeros keep their relative order
        if (is_one == 1u) {
            dest = total_zeros + (li - zeros_before);  // ones follow, stably
        }
        workgroupBarrier();
        s_keys[dest] = my_key;
        s_vals[dest] = my_val;
        workgroupBarrier();
        bit = bit + 1u;
    }
}

// For the sorted tile, returns the run-start of position `li`: the index where
// li's digit run begins (so li - run_start is li's rank within its run).  Must be
// called by every lane (it contains workgroup barriers).
fn run_start_of(li: u32, active_count: u32, shift: u32) -> u32 {
    var boundary = 0u;
    if (li < active_count) {
        let d = (s_keys[li] >> shift) & RADIX_MASK;
        var prev = 0xFFFFFFFFu;                   // sentinel: lane 0 is always a boundary
        if (li > 0u) { prev = (s_keys[li - 1u] >> shift) & RADIX_MASK; }
        if (li == 0u || d != prev) { boundary = li; }
    }
    s_scan[li] = boundary;
    incl_max_scan_wg(li);
    return s_scan[li];
}

// -----------------------------------------------------------------------------
// Phase 0: compute keys + payloads (run once per sort, before the digit passes).
// -----------------------------------------------------------------------------
@compute @workgroup_size(256)
fn cs_compute_keys(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= g.num_elements) {
        return;
    }
    let p = g_splats[i].position.xyz;
    let depth = g.zc.x * p.x + g.zc.y * p.y + g.zc.z * p.z + g.zc.w;
    // Ascending by depth, exactly as the old bitonic sort did: index 0 ends up the
    // farthest splat, so the draw (alpha blending in index order) composites
    // back-to-front.
    g_keys_out[i] = float_key(depth);
    g_vals_out[i] = i;
}

// -----------------------------------------------------------------------------
// Phase 1: per-tile histogram (bucket-major output).  Sort the tile by digit, then
// read off each digit's run length -- O(WG log WG), atomic-free.
// -----------------------------------------------------------------------------
@compute @workgroup_size(256)
fn cs_histogram(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let li = lid.x;
    let tile = wid.x;
    let tile_start = tile * WG;
    var active_count = WG;
    if (tile_start + WG > g.num_elements) {
        active_count = g.num_elements - tile_start;
    }
    let shift = pass_info.shift;

    if (li < active_count) {
        s_keys[li] = g_keys_in[tile_start + li];
    } else {
        s_keys[li] = 0xFFFFFFFFu;                 // sorts to the tail, never counted
    }
    s_vals[li] = 0u;                              // unused here; keeps local_sort generic
    workgroupBarrier();

    local_sort(li, shift);
    let run_start = run_start_of(li, active_count, shift);

    // Each digit's count is written by the (single) lane at the end of its run.
    s_count[li] = 0u;
    workgroupBarrier();
    if (li < active_count) {
        let d = (s_keys[li] >> shift) & RADIX_MASK;
        var is_run_end = li == active_count - 1u;
        if (!is_run_end) {
            is_run_end = d != ((s_keys[li + 1u] >> shift) & RADIX_MASK);
        }
        if (is_run_end) {
            s_count[d] = li - run_start + 1u;
        }
    }
    workgroupBarrier();
    // Bucket-major: hist[bucket * num_tiles + tile].
    g_hist[li * g.num_tiles + tile] = s_count[li];
}

// -----------------------------------------------------------------------------
// Phase 2a: exclusive-scan each 256-wide block of g_hist in place; emit block total.
// g_hist length is RADIX * num_tiles (a multiple of 256), so there are exactly
// num_tiles full blocks.
// -----------------------------------------------------------------------------
@compute @workgroup_size(256)
fn cs_scan_reduce(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let li = lid.x;
    let block = wid.x;
    let idx = block * WG + li;
    s_scan[li] = g_hist[idx];
    let total = excl_scan_wg(li);
    g_hist[idx] = s_scan[li];
    if (li == 0u) {
        g_block_sums[block] = total;
    }
}

// -----------------------------------------------------------------------------
// Phase 2b: exclusive-scan g_block_sums in a single workgroup (any length).
// -----------------------------------------------------------------------------
@compute @workgroup_size(256)
fn cs_scan_spine(@builtin(local_invocation_id) lid: vec3<u32>) {
    let li = lid.x;
    let n = g.num_tiles;
    var carry = 0u;
    var base = 0u;
    loop {
        if (base >= n) { break; }
        let idx = base + li;
        var v = 0u;
        if (idx < n) { v = g_block_sums[idx]; }
        s_scan[li] = v;
        let total = excl_scan_wg(li);
        if (idx < n) { g_block_sums[idx] = s_scan[li] + carry; }
        carry = carry + total;
        workgroupBarrier();
        base = base + WG;
    }
}

// -----------------------------------------------------------------------------
// Phase 2c: add each block's scanned base back into g_hist.
// -----------------------------------------------------------------------------
@compute @workgroup_size(256)
fn cs_scan_add(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let li = lid.x;
    let block = wid.x;
    let idx = block * WG + li;
    g_hist[idx] = g_hist[idx] + g_block_sums[block];
}

// -----------------------------------------------------------------------------
// Phase 3: stable local sort + scatter to global positions.
// -----------------------------------------------------------------------------
@compute @workgroup_size(256)
fn cs_scatter(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let li = lid.x;
    let tile = wid.x;
    let i = gid.x;

    let tile_start = tile * WG;
    var active_count = WG;
    if (tile_start + WG > g.num_elements) {
        active_count = g.num_elements - tile_start;
    }

    // Load this tile.  Inactive lanes (past the end) get a max key so they sort to
    // the tail; they are never scattered.
    var key = 0xFFFFFFFFu;
    var val = 0u;
    if (li < active_count) {
        key = g_keys_in[i];
        val = g_vals_in[i];
    }
    s_keys[li] = key;
    s_vals[li] = val;
    workgroupBarrier();

    let shift = pass_info.shift;

    // Stably sort the tile by the current 8-bit digit, so equal digits are
    // contiguous; then li - run_start is this lane's stable rank within its run.
    local_sort(li, shift);
    let run_start = run_start_of(li, active_count, shift);

    if (li < active_count) {
        let my_digit = (s_keys[li] >> shift) & RADIX_MASK;
        // Global base for this tile's run of `my_digit` (from the scanned histogram)
        // plus the rank within that run.
        let global_pos = g_hist[my_digit * g.num_tiles + tile] + (li - run_start);
        g_keys_out[global_pos] = s_keys[li];
        g_vals_out[global_pos] = s_vals[li];
    }
}
