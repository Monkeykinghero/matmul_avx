# matmul_avx — 矩阵乘法 5 级优化实战

从朴素三重循环到 AVX + Tiling，逐步优化矩阵乘法。

## 项目结构

```
matmul_avx/
├── matmul_avx.c    # 主程序（5 个版本）
├── Makefile        # 编译脚本
├── README.md       # 本文件
└── .gitignore      # 忽略编译产物
```

## 5 级优化路线图

| 版本 | 函数 | 优化手段 | 说明 |
|------|------|---------|------|
| v1 | `matmul_v1_ijk` | 朴素三重循环 | 基准线，复杂度 O(N³) |
| v2 | `matmul_v2_ikj` | 循环重排 (i-k-j) | 利用缓存行空间局部性 |
| v3 | `matmul_v3_tiling` | 分块 (cache blocking) | 减少 cache miss |
| v4 | `matmul_v4_avx` | AVX FMA intrinsic | SIMD 向量化，一次 8 个 float |
| v5 | `matmul_v5_combined` | AVX + Tiling 组合 | 两倍加持 |

## 编译与运行

```bash
# 默认 N=512, TILE=64
make clean
make run

# 自定义矩阵大小和分块
make MATN=1024 MAT_TILE=128 run

# 仅编译
make
```

### 环境要求

- Windows (MinGW-w64 GCC)
- CPU 支持 AVX2 + FMA (2013 年后的 Intel/AMD 都可以)

```bash
# 检查 CPU 是否支持
gcc -mavx -mfma -dM -E - < /dev/null | grep -i avx
```

## 输出示例

```
========================================
  矩阵乘法性能优化对比   N=512
========================================

[v1] 朴素三重循环 i-j-k ...
  耗时: 0.1158 sec  GFLOPS: 2.32

[v2] 循环重排 i-k-j ...
  耗时: 0.0478 sec  GFLOPS: 5.61  加速比: 2.42x

[v3] 分块 tiling (tile=64) ...
  耗时: 0.0608 sec  GFLOPS: 4.41  加速比: 1.90x

[v4] AVX 向量化 (8 flops/cycle) ...
  耗时: 0.0070 sec  GFLOPS: 38.23  加速比: 16.49x

[v5] AVX + Tiling 组合 ...
  耗时: 0.0101 sec  GFLOPS: 26.51  加速比: 11.44x

========================================
  汇总 (N=512, tile=64)
========================================
  v1 朴素          :    2.32 GFLOPS  1.00x (基准)
  v2 循环重排      :    5.61 GFLOPS  2.42x
  v3 分块          :    4.41 GFLOPS  1.90x
  v4 AVX           :   38.23 GFLOPS  16.49x
  v5 AVX+Tiling    :   26.51 GFLOPS  11.44x
========================================
  所有版本计算结果已验证正确 ✓
```

> 实际加速比取决于你的 CPU 型号、内存带宽和 L1/L2/L3 缓存大小。

## GitHub

- 仓库: https://github.com/Monkeykinghero/matmul_avx
- 作者: [Monkeykinghero](https://github.com/Monkeykinghero)
- 许可: MIT

## 下一步

- [ ] 跑通并记录你自己 CPU 上的加速比
- [ ] 尝试调整 `TILE` 大小观察性能变化
- [ ] 扩展到 N=1024/2048 看内存带宽瓶颈
- [ ] 对比 OpenBLAS / Intel MKL 的理论峰值
