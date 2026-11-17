// Exhaustive NCC search of Bad Apple frames over resampled lunar image patches.
// Each candidate pose is resampled on the GPU (im2col over map-projected rasters), normalised to
// zero mean / unit norm, multiplied against all frames with cuBLAS (fp16 in, fp32 accumulate), and
// folded into a per-query top-K list.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#define API extern "C" __declspec(dllexport)
static thread_local std::string last_error;
static void check(cudaError_t code) { if (code != cudaSuccess) throw std::runtime_error(cudaGetErrorString(code)); }
static void blas(cublasStatus_t code) { if (code != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS error " + std::to_string(code)); }

constexpr int GW = 32, GH = 24, DIM = GW * GH;
constexpr int TOPK = 32;
constexpr int CAP = 1024;
constexpr float R_KM = 1737.4f;
constexpr float PI = 3.14159265358979f;
constexpr float MIN_STD = 3.0f;  // u8 units; flatter patches carry no usable shape
// Cells pinned at the top of the stretch are flat sensor-saturation plateaus (TC DN ~2880), not terrain.
constexpr float SAT_LEVEL = 254.5f, MAX_SAT_FRACTION = 0.10f;

struct Level {             // one pyramid level of one raster, u8 with 0 = nodata
    unsigned long long offset;
    int width, height;
    int kind;              // 0: global equirectangular (lon -180..180, lat 90..-90), 1: local quadratic
    int pad;
    float q[12];           // kind 1: tangent-plane km -> pixel-edge coords (col: q[0..5], row: q[6..11])
};
struct Pose {              // kind 0: a,b = lon,lat (rad) of the patch centre; kind 1: tangent-plane km
    unsigned level;
    float a, b, w, theta;
    unsigned flags;        // bit0: eligible for the uniform-frame (dark/bright) lists
};
struct Entry { float score; Pose pose; };

struct Frame {             // per-pose constants
    Level L; float c, s, sp, cp, half_w, half_h, step;
};

__device__ __forceinline__ bool fetch(const uint8_t* raster, const Level& L, float x, float y, float& out) {
    float fx = x - 0.5f, fy = y - 0.5f;
    float x0f = floorf(fx), y0f = floorf(fy);
    float tx = fx - x0f, ty = fy - y0f;
    int x0 = (int)x0f, y0 = (int)y0f, x1, y1;
    if (L.kind == 1) {
        if (x0 < 0 || y0 < 0 || x0 + 1 >= L.width || y0 + 1 >= L.height) return false;
        x1 = x0 + 1; y1 = y0 + 1;
    } else {
        x0 = ((x0 % L.width) + L.width) % L.width; x1 = (x0 + 1) % L.width;
        y0 = min(max(y0, 0), L.height - 1); y1 = min(y0 + 1, L.height - 1);
    }
    const uint8_t* base = raster + L.offset;
    float a = base[(size_t)y0 * L.width + x0], b = base[(size_t)y0 * L.width + x1];
    float c = base[(size_t)y1 * L.width + x0], d = base[(size_t)y1 * L.width + x1];
    if (a == 0.f || b == 0.f || c == 0.f || d == 0.f) return false;
    out = (a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty;
    return true;
}

// (u, v): patch-frame km, u to the right, v up. Rotated by theta into the map frame (x east, y north).
__device__ __forceinline__ bool point(const uint8_t* raster, const Frame& f, const Pose& p, float u, float v, float& out) {
    float dx = u * f.c - v * f.s, dy = u * f.s + v * f.c;
    float col, row;
    if (f.L.kind == 1) {
        float x = p.a + dx, y = p.b + dy;
        const float* q = f.L.q;
        col = q[0] + q[1] * x + q[2] * y + q[3] * x * x + q[4] * x * y + q[5] * y * y;
        row = q[6] + q[7] * x + q[8] * y + q[9] * x * x + q[10] * x * y + q[11] * y * y;
    } else {
        float x = dx / R_KM, y = dy / R_KM;
        float rho = sqrtf(x * x + y * y);
        float lat = p.b, lon = p.a;
        if (rho > 0.f) {
            float cc = asinf(fminf(rho, 1.f)), sc = sinf(cc), cosc = cosf(cc);
            lat = asinf(fmaxf(-1.f, fminf(1.f, cosc * f.sp + y * sc * f.cp / rho)));
            lon = p.a + atan2f(x * sc, rho * cosc * f.cp - y * sc * f.sp);
        }
        col = (lon + PI) * (1.f / (2.f * PI)) * f.L.width;
        row = (0.5f * PI - lat) * (1.f / PI) * f.L.height;
    }
    return fetch(raster, f.L, col, row, out);
}

__device__ __forceinline__ Frame make_frame(const Level* levels, const Pose& p, int gw) {
    Frame f;
    f.L = levels[p.level];
    f.c = cosf(p.theta); f.s = sinf(p.theta);
    f.sp = sinf(p.b); f.cp = cosf(p.b);
    f.half_w = 0.5f * p.w; f.half_h = 0.375f * p.w;
    f.step = p.w / gw;
    return f;
}

// One cell value: 2x2 supersampled bilinear taps approximate the box average over the cell.
__device__ __forceinline__ bool cell(const uint8_t* raster, const Frame& f, const Pose& p, int i, int j, float& out) {
    float u = -f.half_w + (i + 0.5f) * f.step, v = f.half_h - (j + 0.5f) * f.step;
    float q = 0.25f * f.step, sum = 0.f, t;
    if (!point(raster, f, p, u - q, v + q, t)) return false; sum += t;
    if (!point(raster, f, p, u + q, v + q, t)) return false; sum += t;
    if (!point(raster, f, p, u - q, v - q, t)) return false; sum += t;
    if (!point(raster, f, p, u + q, v - q, t)) return false; sum += t;
    out = 0.25f * sum;
    return true;
}

template <int THREADS>
__device__ __forceinline__ float block_sum(float value, float* shared) {
    for (int o = 16; o > 0; o >>= 1) value += __shfl_xor_sync(0xffffffffu, value, o);
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    __syncthreads();
    if (lane == 0) shared[warp] = value;
    __syncthreads();
    float total = 0.f;
    for (int w = 0; w < THREADS / 32; ++w) total += shared[w];
    return total;
}

// Resample one pose per block into a normalised fp16 row; write the uniform-frame scores.
constexpr int ST = 128, PER = DIM / ST;
__global__ void sample_kernel(const uint8_t* raster, const Level* levels, const Pose* poses, int n,
                              __half* patches, uint8_t* valid, __half* dark, __half* bright) {
    __shared__ float red[ST / 32];
    int c = blockIdx.x;
    if (c >= n) return;
    Pose p = poses[c];
    Frame f = make_frame(levels, p, GW);
    float vals[PER];
    bool ok = true;
    float sum = 0.f;
#pragma unroll
    for (int k = 0; k < PER; ++k) {
        int idx = threadIdx.x + k * ST;
        float t = 0.f;
        if (ok && !cell(raster, f, p, idx % GW, idx / GW, t)) ok = false;
        vals[k] = t; sum += t;
    }
    ok = __syncthreads_and(ok);
    float mean = block_sum<ST>(sum, red) / DIM;
    float sq = 0.f, sat = 0.f;
#pragma unroll
    for (int k = 0; k < PER; ++k) { float d = vals[k] - mean; sq += d * d; sat += vals[k] > SAT_LEVEL; }
    float var = block_sum<ST>(sq, red);
    float saturated = block_sum<ST>(sat, red) / DIM;
    float sd = sqrtf(var / DIM);
    bool textured = ok && sd >= MIN_STD && saturated <= MAX_SAT_FRACTION;
    float inv = textured ? rsqrtf(var) : 0.f;
#pragma unroll
    for (int k = 0; k < PER; ++k)
        patches[(size_t)c * DIM + threadIdx.x + k * ST] = __float2half((vals[k] - mean) * inv);
    if (threadIdx.x == 0) {
        valid[c] = textured ? 1 : 0;
        bool uniform_ok = ok && (p.flags & 1u);
        dark[c] = __float2half(uniform_ok ? -(mean + sd) / 255.f : -INFINITY);
        bright[c] = __float2half(uniform_ok ? (mean - sd) / 255.f : -INFINITY);
    }
}

// Patch samples (fp32, normalised) for refinement scoring and CPU cross-checks; any grid size.
__global__ void sample_float_kernel(const uint8_t* raster, const Level* levels, const Pose* poses, int n, int gw,
                                    float* out, uint8_t* valid, float* stats) {
    __shared__ float red[ST / 32];
    int c = blockIdx.x;
    if (c >= n) return;
    Pose p = poses[c];
    Frame f = make_frame(levels, p, gw);
    int gh = gw * 3 / 4, dim = gw * gh;
    float* row = out + (size_t)c * dim;
    bool ok = true;
    float sum = 0.f;
    for (int idx = threadIdx.x; idx < dim; idx += ST) {
        float t = 0.f;
        if (ok && !cell(raster, f, p, idx % gw, idx / gw, t)) ok = false;
        row[idx] = t; sum += t;
    }
    ok = __syncthreads_and(ok);
    float mean = block_sum<ST>(sum, red) / dim;
    float sq = 0.f, sat = 0.f;
    for (int idx = threadIdx.x; idx < dim; idx += ST) { float d = row[idx] - mean; sq += d * d; sat += row[idx] > SAT_LEVEL; }
    float var = block_sum<ST>(sq, red);
    float saturated = block_sum<ST>(sat, red) / dim;
    float sd = sqrtf(var / dim);
    bool textured = ok && sd >= MIN_STD && saturated <= MAX_SAT_FRACTION;
    float inv = textured ? rsqrtf(var) : 0.f;
    for (int idx = threadIdx.x; idx < dim; idx += ST) row[idx] = (row[idx] - mean) * inv;
    if (threadIdx.x == 0) { valid[c] = (ok ? 1 : 0) | (textured ? 2 : 0); stats[2 * c] = mean; stats[2 * c + 1] = sd; }
}

__global__ void dot_kernel(const float* patches, const float* queries, const int* qidx, int n, int dim, float* out) {
    __shared__ float red[ST / 32];
    int c = blockIdx.x;
    if (c >= n) return;
    const float* a = patches + (size_t)c * dim;
    const float* b = queries + (size_t)qidx[c] * dim;
    float s = 0.f;
    for (int i = threadIdx.x; i < dim; i += ST) s += a[i] * b[i];
    s = block_sum<ST>(s, red);
    if (threadIdx.x == 0) out[c] = s;
}

__device__ __forceinline__ unsigned key16(__half h) {
    unsigned short b = __half_as_ushort(h);
    return (b & 0x8000u) ? (unsigned)(~b & 0xffffu) : (unsigned)(b | 0x8000u);
}

// Merge one score column into that query's running top-K (sorted descending).
constexpr int KT = 256;
__global__ void topk_kernel(const __half* scores, size_t ldc, int n, const uint8_t* valid, int nq_gemm,
                            const Pose* poses, Entry* lists) {
    __shared__ float s_score[2 * CAP];
    __shared__ int s_index[2 * CAP];
    __shared__ unsigned hist[256];
    __shared__ int s_count;
    __shared__ unsigned s_bin, s_need;
    __shared__ float red[KT / 32];
    int q = blockIdx.x;
    bool masked = q < nq_gemm;  // dark/bright columns carry -inf for ineligible poses instead
    const __half* col = scores + (size_t)q * ldc;
    Entry* list = lists + (size_t)q * TOPK;
    float thr = list[TOPK - 1].score;

    float cnt = 0.f;
    for (int i = threadIdx.x; i < n; i += KT)
        if ((!masked || valid[i]) && __half2float(col[i]) > thr) cnt += 1.f;
    int total = (int)block_sum<KT>(cnt, red);
    if (total == 0) return;

    unsigned lo_key = 0;      // gather keys >= lo_key (and > thr)
    if (total > CAP) {        // exact radix select of the TOPK-th largest key among candidates
        unsigned prefix = 0, need = TOPK;
        for (int pass = 0; pass < 2; ++pass) {
            for (int b = threadIdx.x; b < 256; b += KT) hist[b] = 0;
            __syncthreads();
            for (int i = threadIdx.x; i < n; i += KT) {
                if ((masked && !valid[i]) || !(__half2float(col[i]) > thr)) continue;
                unsigned k = key16(col[i]);
                if (pass == 1 && (k >> 8) != prefix) continue;
                atomicAdd(&hist[pass == 0 ? (k >> 8) : (k & 255u)], 1u);
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                unsigned acc = 0; int b = 255;
                for (; b > 0; --b) { if (acc + hist[b] >= need) break; acc += hist[b]; }
                s_bin = (unsigned)b; s_need = need - acc;
            }
            __syncthreads();
            if (pass == 0) prefix = s_bin; else lo_key = (prefix << 8) | s_bin;
            need = s_need;
            __syncthreads();
        }
    }
    if (threadIdx.x == 0) s_count = 0;
    __syncthreads();
    // Strictly-above-cutoff keys first (always fewer than CAP), then ties up to capacity.
    for (int tie = 0; tie < 2; ++tie) {
        for (int i = threadIdx.x; i < n; i += KT) {
            if (masked && !valid[i]) continue;
            __half h = col[i];
            float v = __half2float(h);
            unsigned k = key16(h);
            bool take = tie ? (lo_key != 0 && k == lo_key) : (lo_key == 0 || k > lo_key);
            if (!(v > thr) || !take) continue;
            int slot = atomicAdd(&s_count, 1);
            if (slot < CAP) { s_score[slot] = v; s_index[slot] = i; }
        }
        __syncthreads();
    }
    int m = min(s_count, CAP);
    for (int k = threadIdx.x; k < TOPK; k += KT) { s_score[m + k] = list[k].score; s_index[m + k] = -1 - k; }
    int size = 1;
    while (size < m + TOPK) size <<= 1;
    for (int k = m + TOPK + threadIdx.x; k < size; k += KT) { s_score[k] = -INFINITY; s_index[k] = INT32_MIN; }
    __syncthreads();
    for (int len = 2; len <= size; len <<= 1) {
        for (int stride = len >> 1; stride > 0; stride >>= 1) {
            for (int i = threadIdx.x; i < size; i += KT) {
                int j = i ^ stride;
                if (j > i) {
                    bool desc = (i & len) == 0;
                    float a = s_score[i], b = s_score[j];
                    if (desc ? (a < b) : (a > b)) {
                        s_score[i] = b; s_score[j] = a;
                        int t = s_index[i]; s_index[i] = s_index[j]; s_index[j] = t;
                    }
                }
            }
            __syncthreads();
        }
    }
    // Rebuild the list; every old entry is read before any slot is overwritten.
    Entry e;
    int slot = threadIdx.x;
    if (slot < TOPK) {
        int idx = s_index[slot];
        if (idx >= 0) { e.score = s_score[slot]; e.pose = poses[idx]; }
        else if (idx != INT32_MIN) e = list[-1 - idx];
        else { e.score = -INFINITY; memset(&e.pose, 0, sizeof(Pose)); }
    }
    __syncthreads();
    if (slot < TOPK) list[slot] = e;
}

__global__ void init_lists(Entry* lists, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) { lists[i].score = -INFINITY; memset(&lists[i].pose, 0, sizeof(Pose)); }
}
__global__ void to_half(const float* in, __half* out, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(in[i]);
}

struct Context {
    uint8_t* raster = nullptr;
    Level* levels = nullptr;    // this shard's levels
    Level* indexed = nullptr;   // levels - level_base: poses carry global level ids
    int n_levels = 0;
    __half* q16 = nullptr;      // nq x DIM (GEMM operand)
    float* q32 = nullptr;       // nq x DIM
    float* q64 = nullptr;       // nq x 4*DIM
    int nq = 0, chunk = 0, cols = 0;
    Pose* poses[2] = {nullptr, nullptr};
    Pose* host[2] = {nullptr, nullptr};
    cudaEvent_t done[2] = {nullptr, nullptr};
    int turn = 0;
    __half* patches = nullptr;
    uint8_t* valid = nullptr;
    __half* scores = nullptr;   // chunk x cols, column major
    Entry* lists = nullptr;     // cols x TOPK
    cudaStream_t stream = nullptr;
    cublasHandle_t handle = nullptr;
    // scratch for the float paths
    Pose* sp = nullptr; float* sf = nullptr; uint8_t* sv = nullptr; float* ss = nullptr; int* si = nullptr; float* so = nullptr;
    size_t scratch = 0;
};

static void ensure_scratch(Context* c, size_t n, int dim) {
    size_t need = n * dim;
    if (n <= c->scratch && need <= c->scratch * 4 * DIM) return;
    cudaFree(c->sp); cudaFree(c->sf); cudaFree(c->sv); cudaFree(c->ss); cudaFree(c->si); cudaFree(c->so);
    size_t cap = n < 65536 ? 65536 : n;
    check(cudaMalloc(&c->sp, cap * sizeof(Pose)));
    check(cudaMalloc(&c->sf, cap * 4 * DIM * sizeof(float)));
    check(cudaMalloc(&c->sv, cap));
    check(cudaMalloc(&c->ss, cap * 2 * sizeof(float)));
    check(cudaMalloc(&c->si, cap * sizeof(int)));
    check(cudaMalloc(&c->so, cap * sizeof(float)));
    c->scratch = cap;
}

static int fail(const std::exception& e) { last_error = e.what(); return -1; }

API const char* mb_error() { return last_error.c_str(); }
API int mb_dims() { return DIM; }
API int mb_topk() { return TOPK; }
API int mb_pose_size() { return (int)sizeof(Pose); }
API int mb_level_size() { return (int)sizeof(Level); }

// A context holds one shard of the rasters: levels [level_base, level_base + n_levels) of the global
// list, with offsets relative to the shard's own byte range.
API void mb_destroy(void* ctx);

API void* mb_create(const uint8_t* raster, unsigned long long raster_bytes, const Level* levels, int n_levels,
                    unsigned level_base, const float* q32, const float* q64, int nq, int chunk) {
    Context* c = nullptr;
    try {
        if (n_levels <= 0 || nq <= 0 || chunk <= 0) throw std::runtime_error("empty shard, queries or chunk");
        c = new Context();
        c->n_levels = n_levels; c->nq = nq; c->chunk = chunk; c->cols = nq + 2;
        check(cudaStreamCreateWithFlags(&c->stream, cudaStreamNonBlocking));
        blas(cublasCreate(&c->handle));
        blas(cublasSetStream(c->handle, c->stream));
        blas(cublasSetMathMode(c->handle, CUBLAS_DEFAULT_MATH));
        check(cudaMalloc(&c->raster, raster_bytes));
        check(cudaMemcpy(c->raster, raster, raster_bytes, cudaMemcpyHostToDevice));
        check(cudaMalloc(&c->levels, n_levels * sizeof(Level)));
        check(cudaMemcpy(c->levels, levels, n_levels * sizeof(Level), cudaMemcpyHostToDevice));
        c->indexed = c->levels - level_base;  // only ever indexed with ids >= level_base
        check(cudaMalloc(&c->q32, (size_t)nq * DIM * sizeof(float)));
        check(cudaMemcpy(c->q32, q32, (size_t)nq * DIM * sizeof(float), cudaMemcpyHostToDevice));
        check(cudaMalloc(&c->q64, (size_t)nq * 4 * DIM * sizeof(float)));
        check(cudaMemcpy(c->q64, q64, (size_t)nq * 4 * DIM * sizeof(float), cudaMemcpyHostToDevice));
        check(cudaMalloc(&c->q16, (size_t)nq * DIM * sizeof(__half)));
        size_t total = (size_t)nq * DIM;
        to_half<<<(unsigned)((total + 255) / 256), 256, 0, c->stream>>>(c->q32, c->q16, total);
        check(cudaGetLastError());
        for (int i = 0; i < 2; ++i) {
            check(cudaMalloc(&c->poses[i], (size_t)chunk * sizeof(Pose)));
            check(cudaMallocHost(&c->host[i], (size_t)chunk * sizeof(Pose)));
            check(cudaEventCreateWithFlags(&c->done[i], cudaEventDisableTiming));
            check(cudaEventRecord(c->done[i], c->stream));
        }
        check(cudaMalloc(&c->patches, (size_t)chunk * DIM * sizeof(__half)));
        check(cudaMalloc(&c->valid, chunk));
        check(cudaMalloc(&c->scores, (size_t)chunk * c->cols * sizeof(__half)));
        check(cudaMalloc(&c->lists, (size_t)c->cols * TOPK * sizeof(Entry)));
        int count = c->cols * TOPK;
        init_lists<<<(count + 255) / 256, 256, 0, c->stream>>>(c->lists, count);
        check(cudaGetLastError());
        check(cudaStreamSynchronize(c->stream));
        return c;
    } catch (const std::exception& e) {
        last_error = e.what();
        mb_destroy(c);  // release whatever was allocated before the failure
        return nullptr;
    }
}

// Queue one chunk of poses; returns once the poses are staged (the GPU work runs asynchronously).
API int mb_search(void* ctx, const Pose* poses, int n) {
    try {
        auto* c = (Context*)ctx;
        if (n <= 0) return 0;
        if (n > c->chunk) throw std::runtime_error("chunk too large");
        int t = c->turn; c->turn ^= 1;
        check(cudaEventSynchronize(c->done[t]));
        std::memcpy(c->host[t], poses, (size_t)n * sizeof(Pose));
        check(cudaMemcpyAsync(c->poses[t], c->host[t], (size_t)n * sizeof(Pose), cudaMemcpyHostToDevice, c->stream));
        size_t ldc = (size_t)c->chunk;
        __half* dark = c->scores + (size_t)c->nq * ldc;
        __half* bright = dark + ldc;
        sample_kernel<<<n, ST, 0, c->stream>>>(c->raster, c->indexed, c->poses[t], n, c->patches, c->valid, dark, bright);
        check(cudaGetLastError());
        float alpha = 1.f, beta = 0.f;
        blas(cublasGemmEx(c->handle, CUBLAS_OP_T, CUBLAS_OP_N, n, c->nq, DIM, &alpha,
                          c->patches, CUDA_R_16F, DIM, c->q16, CUDA_R_16F, DIM, &beta,
                          c->scores, CUDA_R_16F, (int)ldc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        topk_kernel<<<c->cols, KT, 0, c->stream>>>(c->scores, ldc, n, c->valid, c->nq, c->poses[t], c->lists);
        check(cudaGetLastError());
        check(cudaEventRecord(c->done[t], c->stream));
        return 0;
    } catch (const std::exception& e) { return fail(e); }
}

API int mb_results(void* ctx, Entry* out) {
    try {
        auto* c = (Context*)ctx;
        check(cudaStreamSynchronize(c->stream));
        check(cudaMemcpy(out, c->lists, (size_t)c->cols * TOPK * sizeof(Entry), cudaMemcpyDeviceToHost));
        return 0;
    } catch (const std::exception& e) { return fail(e); }
}

// Restore running lists from a checkpoint.
API int mb_set_results(void* ctx, const Entry* in) {
    try {
        auto* c = (Context*)ctx;
        check(cudaStreamSynchronize(c->stream));
        check(cudaMemcpy(c->lists, in, (size_t)c->cols * TOPK * sizeof(Entry), cudaMemcpyHostToDevice));
        return 0;
    } catch (const std::exception& e) { return fail(e); }
}

// Normalised fp32 patches for arbitrary poses (grid width gw, height 3/4 gw); valid bit0 = in data, bit1 = textured.
API int mb_sample(void* ctx, const Pose* poses, int n, int gw, float* out, uint8_t* valid, float* stats) {
    try {
        auto* c = (Context*)ctx;
        int dim = gw * gw * 3 / 4;
        // The scratch buffers hold up to 64x48 samples per pose.
        if (gw != 32 && gw != 64) throw std::runtime_error("grid must be 32 or 64");
        if (n <= 0) return 0;
        ensure_scratch(c, n, dim);
        check(cudaMemcpyAsync(c->sp, poses, (size_t)n * sizeof(Pose), cudaMemcpyHostToDevice, c->stream));
        sample_float_kernel<<<n, ST, 0, c->stream>>>(c->raster, c->indexed, c->sp, n, gw, c->sf, c->sv, c->ss);
        check(cudaGetLastError());
        if (out) check(cudaMemcpyAsync(out, c->sf, (size_t)n * dim * sizeof(float), cudaMemcpyDeviceToHost, c->stream));
        check(cudaMemcpyAsync(valid, c->sv, n, cudaMemcpyDeviceToHost, c->stream));
        if (stats) check(cudaMemcpyAsync(stats, c->ss, (size_t)n * 2 * sizeof(float), cudaMemcpyDeviceToHost, c->stream));
        check(cudaStreamSynchronize(c->stream));
        return 0;
    } catch (const std::exception& e) { return fail(e); }
}

// fp32 NCC of each pose against one query (gw = 32 or 64); invalid/flat poses score -2.
API int mb_score(void* ctx, const Pose* poses, const int* qidx, int n, int gw, float* scores) {
    try {
        auto* c = (Context*)ctx;
        int dim = gw * gw * 3 / 4;
        if (gw != 32 && gw != 64) throw std::runtime_error("grid must be 32 or 64");
        if (n <= 0) return 0;
        ensure_scratch(c, n, dim);
        check(cudaMemcpyAsync(c->sp, poses, (size_t)n * sizeof(Pose), cudaMemcpyHostToDevice, c->stream));
        check(cudaMemcpyAsync(c->si, qidx, (size_t)n * sizeof(int), cudaMemcpyHostToDevice, c->stream));
        sample_float_kernel<<<n, ST, 0, c->stream>>>(c->raster, c->indexed, c->sp, n, gw, c->sf, c->sv, c->ss);
        dot_kernel<<<n, ST, 0, c->stream>>>(c->sf, gw == 32 ? c->q32 : c->q64, c->si, n, dim, c->so);
        check(cudaGetLastError());
        std::vector<uint8_t> valid(n);
        check(cudaMemcpyAsync(scores, c->so, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost, c->stream));
        check(cudaMemcpyAsync(valid.data(), c->sv, n, cudaMemcpyDeviceToHost, c->stream));
        check(cudaStreamSynchronize(c->stream));
        for (int i = 0; i < n; ++i) if ((valid[i] & 2) == 0) scores[i] = -2.f;
        return 0;
    } catch (const std::exception& e) { return fail(e); }
}

API void mb_destroy(void* ctx) {
    auto* c = (Context*)ctx;
    if (!c) return;
    if (c->stream) cudaStreamSynchronize(c->stream);
    cudaFree(c->raster); cudaFree(c->levels); cudaFree(c->q16); cudaFree(c->q32); cudaFree(c->q64);
    for (int i = 0; i < 2; ++i) {
        cudaFree(c->poses[i]); cudaFreeHost(c->host[i]);
        if (c->done[i]) cudaEventDestroy(c->done[i]);
    }
    cudaFree(c->patches); cudaFree(c->valid); cudaFree(c->scores); cudaFree(c->lists);
    cudaFree(c->sp); cudaFree(c->sf); cudaFree(c->sv); cudaFree(c->ss); cudaFree(c->si); cudaFree(c->so);
    if (c->handle) cublasDestroy(c->handle);
    if (c->stream) cudaStreamDestroy(c->stream);
    delete c;
}
