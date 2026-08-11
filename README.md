# lora-lab

用 LoRA 微调 `Qwen/Qwen2.5-0.5B-Instruct`，教会它自贡井盐与地方史问答。
全流程在本机 Apple Silicon（MPS）上运行，代码默认从项目根目录解析路径，可在任意目录直接执行。

## 目录结构

```
lora-lab/
├── data/
│   ├── train.jsonl          # 原始 78 条问答（两个主题共用）
│   └── train_goap.jsonl     # goap_lora/make_data.py 生成的 GOAP 格式数据
├── tech_lora/               # 主题一：井盐知识 LoRA
│   ├── train.py             #   训练 -> outputs/tech-lora-v2
│   ├── inference.py         #   加载 adapter 交互问答（CONCEPT/ANSWER）
│   └── inference_base.py    #   未微调底座对比
├── goap_lora/               # 主题二：GOAP 强化（顺序训练）
│   ├── make_data.py         #   由 train.jsonl 生成 GOAP 训练数据
│   ├── merge_base.py        #   合并 tech-lora-v2 进 base -> outputs/tech-lora-merged
│   ├── train.py             #   在合并底座上训练 GOAP 格式 -> outputs/goap-lora
│   └── inference.py         #   加载 合并底座+goap adapter 交互问答
└── outputs/                 # 模型产物（已 gitignore）
    ├── tech-lora-v2/        #   tech LoRA adapter
    ├── tech-lora-merged/    #   合并后的完整底座（Qwen + tech-lora-v2 权重）
    └── goap-lora/           #   GOAP LoRA adapter
```

## 用法

主题一（井盐知识）：

```
uv run python tech_lora/train.py
uv run python tech_lora/inference.py
uv run python tech_lora/inference_base.py   # 未微调底座对比
```

主题二（GOAP 强化）：

```
uv run python goap_lora/make_data.py        # 生成 data/train_goap.jsonl
uv run python goap_lora/merge_base.py       # 合并底座（已跑过可跳过）
uv run python goap_lora/train.py            # 顺序训练 GOAP 格式
uv run python goap_lora/inference.py        # 验证
```

## 说明

- **goap-lora 的 adapter 只能配 `outputs/tech-lora-merged` 底座使用**，不能配原始 Qwen 权重。
- 有 pytorch MPS 在长时间训练上偶发内核挂死的现象；训练慢多半是系统后台服务（Spotlight 等）争用 CPU，可稍后再跑。
- 设备默认 `mps`，可用 `DEVICE=cpu uv run python ...` 强制 CPU（更稳但更慢）。
- 生成用采样解码（`temperature=0.7, top_p=0.9`）；小模型上贪婪解码 + 重复惩罚容易退化。