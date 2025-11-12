# BERT 快速训练优化对比实验

实验时间: 2025-11-12 08:48:15

## 实验结果对比

| 配置 | 速度 (tokens/s) | 损失 | 时间 (s) | 状态 |
|------|----------------|------|----------|------|
| 02_bf16 | 96188 | 44.583841705322264 | 2.1 | ✓ |
| 05_bf16_flash | 96172 | 44.89303550720215 | 2.1 | ✓ |
| 14_bf16_flash_qknorm_yarn | 96165 | 44.51131591796875 | 2.1 | ✓ |
| 08_bf16_muon_flash | 94958 | 66.90504302978516 | 2.2 | ✓ |
| 04_bf16_muon | 94762 | 85.6109718322754 | 2.2 | ✓ |
| 01_baseline | 78471 | 41.022804260253906 | 2.6 | ✓ |
| 07_bf16_compile_flash | 65423 | 44.544044494628906 | 3.1 | ✓ |
| 03_bf16_compile | 65314 | 44.3335464477539 | 3.1 | ✓ |
| 15_bf16_compile_flash_fp8head | 65167 | 44.03547821044922 | 3.1 | ✓ |
| 10_bf16_compile_muon_flash_qknorm | 64664 | 63.79208679199219 | 3.2 | ✓ |
| 09_bf16_compile_muon_flash | 64617 | 67.20613632202148 | 3.2 | ✓ |
| 11_recommended | 64602 | 64.38664245605469 | 3.2 | ✓ |
| 06_bf16_compile_muon | 64550 | 65.2066146850586 | 3.2 | ✓ |
| 13_bf16_compile_muon_asyncdata | 64435 | 69.32614669799804 | 3.2 | ✓ |
| 12_full_optimized | 0 | - | 23.5 | ✗ |

## 优化建议

**最快配置**: 02_bf16
- 速度: 96188 tokens/s
- 配置摘要: BF16


## 相对基线加速比

| 配置 | 加速比 |
|------|--------|
| 02_bf16 | 1.23x |
| 05_bf16_flash | 1.23x |
| 14_bf16_flash_qknorm_yarn | 1.23x |
| 08_bf16_muon_flash | 1.21x |
| 04_bf16_muon | 1.21x |
| 01_baseline | 1.00x |
| 07_bf16_compile_flash | 0.83x |
| 03_bf16_compile | 0.83x |
| 15_bf16_compile_flash_fp8head | 0.83x |
| 10_bf16_compile_muon_flash_qknorm | 0.82x |
| 09_bf16_compile_muon_flash | 0.82x |
| 11_recommended | 0.82x |
| 06_bf16_compile_muon | 0.82x |
| 13_bf16_compile_muon_asyncdata | 0.82x |