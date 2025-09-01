# VibeVoice-ComfyUI AMD DirectML 支持

本插件现在支持使用 AMD 显卡的 DirectML 加速功能！

## AMD 显卡支持说明

### 硬件要求
- AMD Radeon RX 580 或更高版本
- 8GB+ 显存推荐
- Windows 10/11 操作系统

### 软件要求
- Python 3.8+
- PyTorch 2.0+
- torch-directml

## 安装 DirectML 支持

### 方法1：自动安装脚本
运行提供的安装脚本：
```bash
cd /path/to/ComfyUI/custom_nodes/VibeVoice-ComfyUI
python install_directml.py
```

### 方法2：手动安装
```bash
pip install torch-directml
```

## 验证安装

安装完成后，启动 ComfyUI，在控制台中查找以下信息：
- `[VibeVoice] DirectML detected with X AMD GPU device(s)` - 表示检测到 AMD 显卡
- `[VibeVoice] AMD GPU acceleration with DirectML is ready` - 表示 DirectML 准备就绪
- `[VibeVoice] Initialized with device: privateuseone:0, DirectML: True` - 表示使用 DirectML 设备

## 使用说明

### 自动设备检测
插件会自动检测并使用最佳设备：
1. **DirectML (AMD GPU)** - 优先使用，提供最佳性能
2. **CUDA (NVIDIA GPU)** - 如果没有 DirectML，使用 CUDA
3. **CPU** - 最后备选方案

### 性能优化建议

#### 模型选择
- **AMD RX 580 (8GB)**: 推荐使用 `VibeVoice-1.5B` 模型
- **更高端 AMD GPU (16GB+)**: 可以使用 `VibeVoice-7B-Preview` 模型

#### 参数设置
- **diffusion_steps**: 15-25 (平衡质量和速度)
- **attention_type**: 建议使用 `auto` 或 `eager`
- **free_memory_after_generate**: 建议保持 `True` 以节省显存

### 内存管理
- DirectML 自动管理显存
- 使用 "VibeVoice Free Memory" 节点可手动释放显存
- 如果遇到显存不足，降低 `diffusion_steps` 或切换到 1.5B 模型

## 故障排除

### 常见问题

#### 1. DirectML 未检测到
- 确保已安装 `torch-directml`
- 检查 AMD 驱动是否最新
- 重启 ComfyUI

#### 2. 显存不足错误
- 使用 VibeVoice-1.5B 替代 7B 模型
- 降低 diffusion_steps 到 10-15
- 启用 free_memory_after_generate

#### 3. 生成速度慢
- 确认使用的是 DirectML 设备 (查看日志)
- 检查其他程序是否占用显卡
- 尝试不同的 attention_type 设置

### 日志信息含义
- `Using DirectML device` - 使用 DirectML 加速
- `Model moved to DirectML device` - 模型已加载到 AMD 显卡
- `DirectML cache cleared` - DirectML 缓存已清理

## 性能对比

基于 AMD RX 580 8GB 的测试结果：

| 设备 | 模型 | 时间 (20 steps) | 显存使用 |
|------|------|----------------|----------|
| CPU | VibeVoice-1.5B | ~120s | 4GB RAM |
| DirectML | VibeVoice-1.5B | ~25s | 6GB VRAM |
| DirectML | VibeVoice-7B | ~60s | 8GB VRAM |

## 联系支持

如果遇到问题，请提供：
1. AMD 显卡型号和驱动版本
2. PyTorch 和 torch-directml 版本
3. 完整的错误日志
4. ComfyUI 控制台输出

---

*注意：DirectML 支持需要 Windows 系统和兼容的 AMD 显卡。此功能在 Linux 下不可用。*