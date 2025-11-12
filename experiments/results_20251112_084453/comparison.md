# BERT 快速训练优化对比实验

实验时间: 2025-11-12 08:45:07

## 实验结果对比

| 配置 | 速度 (tokens/s) | 损失 | 时间 (s) | 状态 |
|------|----------------|------|----------|------|
| 01_baseline | 0 | - | 0.9 | ✗ |
| 02_bf16 | 0 | - | 0.9 | ✗ |
| 03_bf16_compile | 0 | - | 0.9 | ✗ |
| 04_bf16_muon | 0 | - | 0.9 | ✗ |
| 05_bf16_flash | 0 | - | 0.9 | ✗ |
| 06_bf16_compile_muon | 0 | - | 0.9 | ✗ |
| 07_bf16_compile_flash | 0 | - | 0.9 | ✗ |
| 08_bf16_muon_flash | 0 | - | 0.9 | ✗ |
| 09_bf16_compile_muon_flash | 0 | - | 0.9 | ✗ |
| 10_bf16_compile_muon_flash_qknorm | 0 | - | 0.9 | ✗ |
| 11_recommended | 0 | - | 0.9 | ✗ |
| 12_full_optimized | 0 | - | 0.9 | ✗ |
| 13_bf16_compile_muon_asyncdata | 0 | - | 0.9 | ✗ |
| 14_bf16_flash_qknorm_yarn | 0 | - | 0.9 | ✗ |
| 15_bf16_compile_flash_fp8head | 0 | - | 0.9 | ✗ |

## 优化建议
