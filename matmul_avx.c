/**
 * matmul_avx.c — 矩阵乘法从朴素到 AVX 的 5 级优化
 *
 * 编译: gcc -O2 -mavx -mfma -o matmul_avx.exe matmul_avx.c
 * 运行: matmul_avx.exe
 *
 * 环境: MinGW (GCC), Windows
 * 计时: QueryPerformanceCounter
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>

/* ======================== 配置 ======================== */

#ifndef N
#define N 512         /* 矩阵维度: N x N */
#endif

#ifndef TILE
#define TILE 64       /* 分块大小 */
#endif

#ifndef WARMUP
#define WARMUP 2      /* 预热轮数 */
#endif

#ifndef ITERS
#define ITERS 5       /* 正式测试轮数 */
#endif

/* ======================== 计时工具 ======================== */

static double now_sec(void)
{
    LARGE_INTEGER freq, cnt;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&cnt);
    return (double)cnt.QuadPart / (double)freq.QuadPart;
}

/* GFLOPS = 2*N^3 / 时间(秒) / 1e9 */
static double gflops(double sec, int n)
{
    return (2.0 * n * n * n) / sec / 1e9;
}

/* ======================== 工具函数 ======================== */

static float rand_float(void)
{
    return (float)(rand() % 100) / 50.0f - 1.0f; /* [-1, 1] */
}

static int alloc_matrices(float **A, float **B, float **C_naive, float **C_opt,
                          int n)
{
    *A       = (float *)_aligned_malloc(n * n * sizeof(float), 32);
    *B       = (float *)_aligned_malloc(n * n * sizeof(float), 32);
    *C_naive = (float *)_aligned_malloc(n * n * sizeof(float), 32);
    *C_opt   = (float *)_aligned_malloc(n * n * sizeof(float), 32);
    if (!*A || !*B || !*C_naive || !*C_opt) {
        fprintf(stderr, "内存分配失败\n");
        return -1;
    }
    return 0;
}

static void free_matrices(float *A, float *B, float *C_naive, float *C_opt)
{
    _aligned_free(A);
    _aligned_free(B);
    _aligned_free(C_naive);
    _aligned_free(C_opt);
}

static void init_matrices(float *A, float *B, int n)
{
    int i, j;
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            A[i * n + j] = rand_float();

    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            B[i * n + j] = rand_float();
}

static void clear_matrix(float *C, int n)
{
    memset(C, 0, n * n * sizeof(float));
}

/* 验证: C_opt 和 C_naive 是否近似相等 */
static int verify(const float *C_naive, const float *C_opt, int n)
{
    int i;
    for (i = 0; i < n * n; i++) {
        float diff = (float)fabs(C_naive[i] - C_opt[i]);
        if (diff > 1.0f) {
            printf("  验证失败: idx=%d  naive=%.2f  opt=%.2f  diff=%.2f\n",
                   i, C_naive[i], C_opt[i], diff);
            return -1;
        }
    }
    return 0;
}

/* ======================== 各版本实现 ======================== */

/* ---- v1: 朴素三重循环 i-j-k ---- */
void matmul_v1_ijk(const float *A, const float *B, float *C, int n)
{
    int i, j, k;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            float sum = 0.0f;
            for (k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

/* ---- v2: 循环重排 i-k-j，利用 A 行的空间局部性 ---- */
void matmul_v2_ikj(const float *A, const float *B, float *C, int n)
{
    int i, j, k;
    for (i = 0; i < n; i++) {
        for (k = 0; k < n; k++) {
            float aik = A[i * n + k];
            for (j = 0; j < n; j++) {
                C[i * n + j] += aik * B[k * n + j];
            }
        }
    }
}

/* ---- v3: 分块 (tiling)，提高 cache 命中率 ---- */
void matmul_v3_tiling(const float *A, const float *B, float *C, int n, int tile)
{
    int i, j, k, ii, jj, kk;
    for (ii = 0; ii < n; ii += tile) {
        int i_max = ii + tile < n ? ii + tile : n;
        for (kk = 0; kk < n; kk += tile) {
            int k_max = kk + tile < n ? kk + tile : n;
            for (jj = 0; jj < n; jj += tile) {
                int j_max = jj + tile < n ? jj + tile : n;
                for (i = ii; i < i_max; i++) {
                    for (k = kk; k < k_max; k++) {
                        float aik = A[i * n + k];
                        for (j = jj; j < j_max; j++) {
                            C[i * n + j] += aik * B[k * n + j];
                        }
                    }
                }
            }
        }
    }
}

/* ---- v4: AVX intrinsic (i-k-j 顺序 + FMA) ---- */
#include <immintrin.h>

void matmul_v4_avx(const float *A, const float *B, float *C, int n)
{
    int i, j, k;
    for (i = 0; i < n; i++) {
        for (k = 0; k < n; k++) {
            __m256 aik = _mm256_set1_ps(A[i * n + k]);
            for (j = 0; j + 8 <= n; j += 8) {
                __m256 c   = _mm256_load_ps(&C[i * n + j]);
                __m256 b   = _mm256_load_ps(&B[k * n + j]);
                __m256 res = _mm256_fmadd_ps(aik, b, c);
                _mm256_store_ps(&C[i * n + j], res);
            }
            /* 处理余数 */
            for (; j < n; j++) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}

/* ---- v5: AVX + Tiling 组合 ---- */
void matmul_v5_combined(const float *A, const float *B, float *C,
                        int n, int tile)
{
    int i, j, k, ii, jj, kk;
    for (ii = 0; ii < n; ii += tile) {
        int i_max = ii + tile < n ? ii + tile : n;
        for (kk = 0; kk < n; kk += tile) {
            int k_max = kk + tile < n ? kk + tile : n;
            for (jj = 0; jj < n; jj += tile) {
                int j_max = jj + tile < n ? jj + tile : n;
                for (i = ii; i < i_max; i++) {
                    for (k = kk; k < k_max; k++) {
                        __m256 aik = _mm256_set1_ps(A[i * n + k]);
                        for (j = jj; j + 8 <= j_max; j += 8) {
                            __m256 c   = _mm256_load_ps(&C[i * n + j]);
                            __m256 b   = _mm256_load_ps(&B[k * n + j]);
                            __m256 res = _mm256_fmadd_ps(aik, b, c);
                            _mm256_store_ps(&C[i * n + j], res);
                        }
                        /* 余数 */
                        for (; j < j_max; j++) {
                            C[i * n + j] += A[i * n + k] * B[k * n + j];
                        }
                    }
                }
            }
        }
    }
}

/* ======================== 测试框架 ======================== */

typedef void (*matmul_func)(const float *, const float *, float *, int);

struct bench_entry {
    const char  *name;
    matmul_func  func;
    int          use_tile;   /* 0 表示不用 tile */
};

static double run_bench(const float *A, const float *B, float *C,
                        const struct bench_entry *entry, int n)
{
    int t;
    double start, end;

    /* 预热 */
    for (t = 0; t < WARMUP; t++) {
        clear_matrix(C, n);
        if (entry->use_tile)
            matmul_v5_combined(A, B, C, n, TILE);
        else
            entry->func(A, B, C, n);
    }

    /* 正式 */
    clear_matrix(C, n);
    start = now_sec();
    for (t = 0; t < ITERS; t++) {
        clear_matrix(C, n);
        if (entry->use_tile)
            matmul_v5_combined(A, B, C, n, TILE);
        else
            entry->func(A, B, C, n);
    }
    end = now_sec();

    return (end - start) / ITERS;
}

/* ======================== 主程序 ======================== */

int main(void)
{
    float *A, *B, *C_naive, *C_opt;
    int n = N;
    int i;

    printf("========================================\n");
    printf("  矩阵乘法性能优化对比   N=%d\n", n);
    printf("========================================\n\n");

    /* 分配 + 初始化 */
    if (alloc_matrices(&A, &B, &C_naive, &C_opt, n) != 0)
        return 1;

    srand(42);
    init_matrices(A, B, n);

    /* ---------- 先跑 v1 作为基准 ---------- */
    printf("[v1] 朴素三重循环 i-j-k ...\n");
    clear_matrix(C_naive, n);
    double t1 = run_bench(A, B, C_naive,
        &(struct bench_entry){"v1_ijk", matmul_v1_ijk, 0}, n);
    printf("  耗时: %.4f sec  GFLOPS: %.2f\n\n", t1, gflops(t1, n));

    /* ---------- v2: 循环重排 ---------- */
    printf("[v2] 循环重排 i-k-j ...\n");
    clear_matrix(C_opt, n);
    double t2 = run_bench(A, B, C_opt,
        &(struct bench_entry){"v2_ikj", matmul_v2_ikj, 0}, n);
    printf("  耗时: %.4f sec  GFLOPS: %.2f", t2, gflops(t2, n));
    printf("  加速比: %.2fx\n\n", t1 / t2);

    /* 拿 v2 的结果验证 */
    if (verify(C_naive, C_opt, n) != 0) goto fail;

    /* ---------- v3: 分块 ---------- */
    printf("[v3] 分块 tiling (tile=%d) ...\n", TILE);
    clear_matrix(C_opt, n);
    double t3_start = now_sec();
    for (i = 0; i < ITERS; i++) {
        clear_matrix(C_opt, n);
        matmul_v3_tiling(A, B, C_opt, n, TILE);
    }
    double t3 = (now_sec() - t3_start) / ITERS;
    printf("  耗时: %.4f sec  GFLOPS: %.2f", t3, gflops(t3, n));
    printf("  加速比: %.2fx\n\n", t1 / t3);

    if (verify(C_naive, C_opt, n) != 0) goto fail;

    /* ---------- v4: AVX ---------- */
    printf("[v4] AVX 向量化 (8 flops/cycle) ...\n");
    clear_matrix(C_opt, n);
    double t4 = run_bench(A, B, C_opt,
        &(struct bench_entry){"v4_avx", matmul_v4_avx, 0}, n);
    printf("  耗时: %.4f sec  GFLOPS: %.2f", t4, gflops(t4, n));
    printf("  加速比: %.2fx\n\n", t1 / t4);

    if (verify(C_naive, C_opt, n) != 0) goto fail;

    /* ---------- v5: AVX + Tiling ---------- */
    printf("[v5] AVX + Tiling 组合 ...\n");
    clear_matrix(C_opt, n);
    double t5 = run_bench(A, B, C_opt,
        &(struct bench_entry){"v5_combined", NULL, 1}, n);
    printf("  耗时: %.4f sec  GFLOPS: %.2f", t5, gflops(t5, n));
    printf("  加速比: %.2fx\n\n", t1 / t5);

    if (verify(C_naive, C_opt, n) != 0) goto fail;

    /* ---------- 汇总 ---------- */
    printf("========================================\n");
    printf("  汇总 (N=%d, tile=%d)\n", n, TILE);
    printf("========================================\n");
    printf("  v1 朴素          : %7.2f GFLOPS  1.00x (基准)\n",   gflops(t1, n));
    printf("  v2 循环重排      : %7.2f GFLOPS  %.2fx\n",          gflops(t2, n), t1/t2);
    printf("  v3 分块          : %7.2f GFLOPS  %.2fx\n",          gflops(t3, n), t1/t3);
    printf("  v4 AVX           : %7.2f GFLOPS  %.2fx\n",          gflops(t4, n), t1/t4);
    printf("  v5 AVX+Tiling    : %7.2f GFLOPS  %.2fx\n",          gflops(t5, n), t1/t5);
    printf("========================================\n");
    printf("  所有版本计算结果已验证正确 ✓\n");

    free_matrices(A, B, C_naive, C_opt);
    return 0;

fail:
    printf("  验证失败，终止\n");
    free_matrices(A, B, C_naive, C_opt);
    return 1;
}
