# VibeVoice ComfyUI 节点

这是一个综合的 ComfyUI 集成插件，用于 Microsoft 的 VibeVoice 文本转语音模型，在您的 ComfyUI 工作流中直接实现高质量的单人和多人语音合成。

## 功能特性

### 核心功能
- 🎤 **单人语音合成**: 生成自然语音，支持可选的语音克隆
- 👥 **多人对话**: 支持最多 4 个不同说话人
- 🎯 **语音克隆**: 从音频样本克隆语音
- 📝 **文本文件加载**: 从文本文件加载脚本

### 模型选项
- 🚀 **两种模型大小**: 1.5B（更快）和 7B（更高质量）
- 🔧 **灵活配置**: 控制温度、采样和引导比例

### 性能与优化
- ⚡ **注意力机制**: 可选择 auto、eager、sdpa 或 flash_attention_2
- 🎛️ **扩散步数**: 可调节的质量与速度平衡（默认: 20）
- 💾 **内存管理**: 生成后自动清理 VRAM 的开关
- 🧹 **释放内存节点**: 复杂工作流的手动内存控制

## 视频演示
<p align="center">
  <a href="https://www.youtube.com/watch?v=fIBMepIBKhI">
    <img src="https://img.youtube.com/vi/fIBMepIBKhI/maxresdefault.jpg" alt="VibeVoice ComfyUI Wrapper Demo" />
  </a>
  <br>
  <strong>点击观看演示视频</strong>
</p>

## 安装方法

### 自动安装（推荐）
1. 将此仓库克隆到您的 ComfyUI 自定义节点文件夹：
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Enemyx-net/VibeVoice-ComfyUI
```

2. 重启 ComfyUI - 节点将在首次使用时自动安装 VibeVoice

### 手动安装
如果自动安装失败：
```bash
cd ComfyUI
python_embeded/python.exe -m pip install git+https://github.com/microsoft/VibeVoice.git
```

## 可用节点

### 1. VibeVoice 加载文本文件
从 ComfyUI 的 input/output/temp 目录中加载文本内容。
- **支持格式**: .txt
- **输出**: 用于 TTS 节点的文本字符串

### 2. VibeVoice 单人说话
使用单个语音从文本生成语音。
- **文本输入**: 直接文本或从加载文本节点的连接
- **模型**: VibeVoice-1.5B 或 VibeVoice-7B-Preview
- **语音克隆**: 用于语音克隆的可选音频输入
- **参数**（按顺序）：
  - `text`: 要转换为语音的输入文本
  - `model`: VibeVoice-1.5B 或 VibeVoice-7B-Preview
  - `attention_type`: auto、eager、sdpa 或 flash_attention_2（默认: auto）
  - `free_memory_after_generate`: 生成后释放显存（默认: True）
  - `diffusion_steps`: 去噪步数（5-100，默认: 20）
  - `seed`: 用于可复现性的随机种子（默认: 42）
  - `cfg_scale`: 分类器无关引导（1.0-2.0，默认: 1.3）
  - `use_sampling`: 启用/禁用确定性生成（默认: False）
- **可选参数**:
  - `voice_to_clone`: 用于语音克隆的音频输入
  - `temperature`: 采样温度（0.1-2.0，默认: 0.95）
  - `top_p`: 核采样参数（0.1-1.0，默认: 0.95）

### 3. VibeVoice 多人说话
生成具有不同语音的多人对话。
- **说话人格式**: 使用 `[N]:` 记号，其中 N 为 1-4
- **语音分配**: 每个说话人的可选语音样本
- **推荐模型**: VibeVoice-7B-Preview 以获得更好的多人说话质量
- **参数**（按顺序）：
  - `text`: 带有说话人标签的输入文本
  - `model`: VibeVoice-1.5B 或 VibeVoice-7B-Preview
  - `attention_type`: auto、eager、sdpa 或 flash_attention_2（默认: auto）
  - `free_memory_after_generate`: 生成后释放显存（默认: True）
  - `diffusion_steps`: 去噪步数（5-100，默认: 20）
  - `seed`: 用于可复现性的随机种子（默认: 42）
  - `cfg_scale`: 分类器无关引导（1.0-2.0，默认: 1.3）
  - `use_sampling`: 启用/禁用确定性生成（默认: False）
- **可选参数**:
  - `speaker1_voice` 到 `speaker4_voice`: 用于语音克隆的音频输入
  - `temperature`: 采样温度（0.1-2.0，默认: 0.95）
  - `top_p`: 核采样参数（0.1-1.0，默认: 0.95）

### 4. VibeVoice 释放内存
手动释放所有已加载的 VibeVoice 模型内存。
- **输入**: `audio` - 连接音频输出以触发内存清理
- **输出**: `audio` - 原样传递输入音频
- **使用场景**: 在节点之间插入以在特定工作流点释放 VRAM/RAM
- **示例**: `[VibeVoice 节点] → [释放内存] → [保存音频]`

## 多人说话文本格式

对于多人说话生成，使用 `[N]:` 记号格式化您的文本：

```
[1]: 你好，你今天怎么样？
[2]: 我很好，谢谢你的关心！
[1]: 听到这个真太好了。
[3]: 大家好，介意我加入谈话吗？
[2]: 当然不介意，欢迎！
```

**重要注意事项**：
- 使用 `[1]:`、`[2]:`、`[3]:`、`[4]:` 作为说话人标签
- 最多支持 4 个说话人
- 系统会自动从您的文本中检测说话人数量
- 每个说话人都可以有一个可选的语音样本用于克隆

## 模型信息

### VibeVoice-1.5B
- **大小**: ~5GB 下载
- **速度**: 推理更快
- **质量**: 适合单人说话
- **使用场景**: 快速原型设计、单个语音

### VibeVoice-7B-Preview
- **大小**: ~17GB 下载
- **速度**: 推理较慢
- **质量**: 优秀，尤其适合多人说话
- **使用场景**: 生产质量、多人对话

模型在首次使用时会自动下载，并缓存在 `ComfyUI/models/vibevoice/` 中。

## 生成模式

### 确定性模式（默认）
- `use_sampling = False`
- 产生一致、稳定的输出
- 推荐用于生产使用

### 采样模式
- `use_sampling = True`
- 输出更多变化
- 使用 temperature 和 top_p 参数
- 适合创意探索

## 语音克隆

要克隆语音：
1. 将音频节点连接到 `voice_to_clone` 输入（单人说话）
2. 或连接到 `speaker1_voice`、`speaker2_voice` 等（多人说话）
3. 模型将尝试匹配语音特征

**语音样本要求**：
- 背景噪音最小的清晰音频
- 最少 3-10 秒。建议至少 30 秒以获得更好的质量
- 自动重采样到 24kHz

## 最佳实践建议

1. **文本准备**：
   - 使用正确的标点符号以实现自然停顿
   - 将长文本分成段落
   - 对于多人说话，确保清晰的说话人转换

2. **模型选择**：
   - 使用 1.5B 用于快速单人说话任务
   - 使用 7B 用于多人说话或对质量有较高要求时

3. **种子管理**：
   - 默认种子（42）适用于大多数情况
   - 保存好的种子以获得一致的角色语音
   - 如果默认值效果不好，尝试随机种子

4. **性能**：
   - 首次运行会下载模型（5-17GB）
   - 后续运行使用缓存模型
   - 推荐使用 GPU 以获得更快的推理速度

## 系统要求

### 硬件
- **最低要求**: VibeVoice-1.5B 需要 8GB 显存
- **推荐配置**: VibeVoice-7B 需要 16GB+ 显存
- **内存**: 16GB+ 系统内存

### 软件
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+（用于 GPU 加速）
- ComfyUI（最新版本）

## 故障排除

### 安装问题
- 确保您使用的是 ComfyUI 的 Python 环境
- 如果自动安装失败，尝试手动安装
- 安装后重启 ComfyUI

### 生成问题
- 如果语音听起来不稳定，尝试确定性模式
- 对于多人说话，确保文本具有正确的 `[N]:` 格式
- 检查说话人编号是否连续（1,2,3 而不是 1,3,5）

### 内存问题
- 7B 模型需要约 16GB 显存
- 低显存系统使用 1.5B 模型
- 模型使用 bfloat16 精度以提高效率

## 示例

### 单人说话
```
文本: "欢迎参加我们的演示。今天我们将探索人工智能的迷人世界。"
模型: VibeVoice-1.5B
cfg_scale: 1.3
use_sampling: False
```

### 两人对话
```
[1]: 你看到新的 AI 发展了吗？
[2]: 是的，非常令人印象深刻！
[1]: 我觉得语音合成技术取得了很大进步。
[2]: 绝对的，听起来非常自然。
```

### 四人对话
```
[1]: 欢迎大家参加我们的会议。
[2]: 谢谢邀请我们！
[3]: 很高兴能在这里。
[4]: 期待这次讨论。
[1]: 让我们开始今天的议程吧。
```

## 性能基准

| 模型 | 显存使用 | 上下文长度 | 最大音频时长 |
|-------|------------|----------------|-------------------|
| VibeVoice-1.5B | ~8GB | 64K tokens | ~90 分钟 |
| VibeVoice-7B | ~16GB | 32K tokens | ~45 分钟 |

## 已知限制

- 多人说话模式最多支持 4 个说话人
- 英文和中文文本效果最佳
- 某些种子可能产生不稳定的输出
- 无法直接控制背景音乐生成

## 许可证

此 ComfyUI 包装器在 MIT 许可证下发布。详情请查看 LICENSE 文件。

**注意**：VibeVoice 模型本身受 Microsoft 许可证条款约束：
- VibeVoice 仅用于研究目的
- 请查看 Microsoft 的 VibeVoice 仓库以了解完整的模型许可证详情

## 相关链接

- [原始 VibeVoice 仓库](https://github.com/microsoft/VibeVoice) - Microsoft 官方 VibeVoice 仓库

## 致谢

- **VibeVoice 模型**: Microsoft Research
- **ComfyUI 集成**: Fabio Sarracino
- **基础模型**: 基于 Qwen2.5 架构构建

## 技术支持

如遇到问题或疑问：
1. 查看故障排除部分
2. 查看 ComfyUI 日志中的错误消息
3. 确保 VibeVoice 已正确安装
4. 提交包含详细错误信息的问题报告

## 贡献

欢迎贡献！请：
1. 彻底测试更改
2. 遵循现有的代码风格
3. 根据需要更新文档
4. 提交具有清晰描述的拉取请求

## 更新日志

### 版本 1.0.4
- 改进了分词器依赖处理

### 版本 1.0.3
- 在单人说话和多人说话节点中添加了 `attention_type` 参数以优化性能
  - auto（默认）：自动选择最佳实现
  - eager：没有优化的标准实现
  - sdpa：PyTorch 的优化缩放点积注意力
  - flash_attention_2：最大性能的 Flash Attention 2（需要兼容 GPU）
- 添加了 `diffusion_steps` 参数以控制生成质量与速度的平衡
  - 默认值：20（VibeVoice 默认值）
  - 更高的值：更好的质量，但生成时间更长
  - 更低的值：更快的生成，但质量可能会降低

### 版本 1.0.2
- 在单人说话和多人说话节点中添加了 `free_memory_after_generate` 开关
- 新增专用的“释放内存节点”，用于工作流中的手动内存管理
- 改进了 VRAM/RAM 使用优化
- 增强了长时间生成会话的稳定性
- 用户现在可以在自动和手动内存管理之间选择

### 版本 1.0.1
- 修复了说话人文本中换行符的问题（单人和多人说话节点）
- 单个说话人文本内的换行符现在在生成前会自动删除
- 改进了所有生成模式的文本格式化处理

### 版本 1.0.0
- 初始发布
- 带有语音克隆的单人说话节点
- 带有自动说话人检测的多人说话节点
- 从 ComfyUI 目录加载文本文件
- 确定性和采样生成模式
- 支持 VibeVoice 1.5B 和 7B 模型