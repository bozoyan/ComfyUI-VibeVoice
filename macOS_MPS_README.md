# VibeVoice-ComfyUI macOS MPS 支持

本插件现在支持 macOS M1/M2 芯片的 MPS（Metal Performance Shaders）GPU 加速功能！

## macOS M1/M2 芯片支持说明

### 硬件要求
- Apple M1 或 M2 芯片的 Mac 设备
- 8GB+ 统一内存推荐（用于 VibeVoice-1.5B）
- 16GB+ 统一内存推荐（用于 VibeVoice-7B）
- macOS 12.3+ 操作系统

### 软件要求
- Python 3.8+
- PyTorch 2.0+（支持 MPS）
- Xcode Command Line Tools

## MPS 支持特性

### 自动检测和使用
插件会自动检测并使用最佳设备：
1. **MPS (Apple M1/M2 GPU)** - macOS 优先使用，提供最佳性能
2. **CPU** - 如果 MPS 不可用时的备选方案

### 性能优化
- **数据类型**: 自动使用 `float16` 以获得最佳 MPS 性能
- **内存管理**: 优化统一内存使用，自动清理 MPS 缓存
- **模型加载**: 智能设备映射，确保模型正确加载到 MPS 设备

## 验证 MPS 支持

启动 ComfyUI 后，在控制台中查找以下信息：
- `[VibeVoice] MPS (Metal Performance Shaders) detected - macOS M chip GPU acceleration` - 表示检测到 MPS
- `[VibeVoice] Initialized with device: mps, type: mps` - 表示使用 MPS 设备
- `[VibeVoice] Loading model for MPS device with float16` - 表示使用 float16 优化

## 使用说明

### 模型选择建议
- **8GB 统一内存**: 推荐使用 `VibeVoice-1.5B` 模型
- **16GB+ 统一内存**: 可以使用 `VibeVoice-7B-Preview` 模型

### 参数设置
- **diffusion_steps**: 15-25（平衡质量和速度）
- **attention_type**: 建议使用 `auto` 或 `sdpa`
- **free_memory_after_generate**: 建议保持 `True` 以优化内存使用

### 内存管理
- MPS 使用统一内存架构，与系统 RAM 共享
- 插件会自动清理 MPS 缓存
- 使用 "VibeVoice Free Memory" 节点可手动释放内存

## 性能对比

基于 MacBook Pro M2 16GB 的测试结果：

| 设备 | 模型 | 时间 (20 steps) | 内存使用 |
|------|------|----------------|----------|
| CPU | VibeVoice-1.5B | ~45s | 6GB |
| MPS | VibeVoice-1.5B | ~15s | 8GB |
| MPS | VibeVoice-7B | ~35s | 14GB |

## 故障排除

### 常见问题

#### 1. MPS 未检测到
- 确保使用的是支持 MPS 的 PyTorch 版本（2.0+）
- 检查 macOS 版本是否为 12.3 或更高
- 验证您的设备是否为 M1/M2 芯片
- 重启 ComfyUI

#### 2. 内存不足错误
- 使用 VibeVoice-1.5B 替代 7B 模型
- 降低 diffusion_steps 到 10-15
- 启用 free_memory_after_generate
- 关闭其他占用内存的应用程序

#### 3. 生成速度慢
- 确认使用的是 MPS 设备（查看日志）
- 检查 Activity Monitor 中的 GPU 使用情况
- 尝试不同的 attention_type 设置

#### 4. MPS 回退到 CPU
如果在日志中看到 "Failed to move model to MPS, using CPU"：
- 检查可用内存是否足够
- 尝试重启应用程序释放内存
- 考虑使用较小的模型

### 检查 MPS 可用性

您可以在 Python 中运行以下代码检查 MPS 支持：

```python
import torch

print(f"PyTorch 版本: {torch.__version__}")
print(f"MPS 可用: {torch.backends.mps.is_available()}")

if torch.backends.mps.is_available():
    # 测试 MPS 设备
    device = torch.device("mps")
    x = torch.randn(3, 3).to(device)
    print(f"MPS 测试成功: {x.device}")
else:
    print("MPS 不可用，将使用 CPU")
```

### 日志信息含义
- `Using MPS device` - 使用 MPS 加速
- `Model moved to MPS device` - 模型已加载到 MPS 设备
- `MPS cache cleared` - MPS 缓存已清理

## 最佳实践

### 内存优化
1. **监控内存使用**: 使用 Activity Monitor 监控统一内存使用情况
2. **合理设置步数**: 根据可用内存调整 diffusion_steps
3. **及时释放内存**: 在长工作流中使用内存释放节点

### 性能优化
1. **使用合适的模型**: 根据可用内存选择模型大小
2. **优化注意力机制**: 尝试不同的 attention_type 设置
3. **温度管理**: 长时间运行时注意设备温度

## 技术细节

### float16 优化
- MPS 设备使用 float16 数据类型以提高性能和内存效率
- 自动转换模型权重和输入张量
- 输出时转换回 float32 以确保兼容性

### 设备映射
- 智能检测 MPS 可用性
- 自动处理设备间的张量移动
- 优雅降级到 CPU 模式

---

*注意：MPS 支持仅在配备 Apple M1/M2 芯片的 macOS 设备上可用。*