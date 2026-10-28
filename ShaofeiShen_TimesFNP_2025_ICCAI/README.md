# TimesFNP

- **论文全名**：TimesFNP: Contrastive Learning for Financial Domain with Noise-Resilient Prediction
- **作者**：Shaofei Shen, Qing Li*, Yan Chen, Rui Cheng, Yu Zheng（西南财经大学）
- **发表年份**：2025
- **期刊/会议**：ICCAI 2025（2025 11th International Conference on Computing and Artificial Intelligence, pp. 697-704）
- **论文链接**：https://doi.org/10.1109/ICCAI66501.2025.00111
- **代码负责人**：Shaofei Shen（GitHub 用户名：shensf0522）

## 项目说明

本文件夹为上述论文的配套代码，包含数据处理、训练与评估全流程。方法由三个模块构成：

1. **数据增强**：正样本——频率自适应采样（FFT 按能量保留主频成分重构低噪序列）；负样本——高斯/均匀/对数正态混合噪声按原序列 std 缩放注入（`src/augment.py`）。
2. **对比学习**：temporal-wise InfoNCE（式 2，频率采样序列为正样本、同序列其他时间片与加噪序列为负样本）+ instance-wise InfoNCE（式 3，同行业公司为正样本对，cosine 相似度，温度 τ=0.1）（`src/losses.py`）。
3. **编解码预测**：patch 嵌入 + 可学习位置编码 → Transformer Encoder（2 层/4 头/FFN=128）；H 个可学习 query + 宏观协变量投影 → cross-attention Decoder 输出 H 步预测（`src/model.py`；`--model base` 为去掉对比损失的同结构对照组）。

## 环境与依赖

- Python 3.10；核心框架 torch 2.5.1+cu121（CUDA 12.1，无 GPU 时 CPU 版亦可运行）
- 安装（二选一）：

```bash
pip install -r requirements.txt                                       # CPU 版 torch
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121   # CUDA 版 (先装此行再装其余)
# 国内镜像可加速: 加 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 数据获取

以下数据均为公开数据，下载后放置于 `data/raw/`（该目录不入库，见 .gitignore）。

### a. GKX datashare（主数据，CRSP 源个股月度面板）

- **数据集名称**：Empirical Asset Pricing via Machine Learning 公开数据（Gu, Kelly & Xiu, 2020, RFS；源自 CRSP）
- **下载链接**：https://dachxiu.chicagobooth.edu/download/datashare.zip （约 1.5GB，解压后 datashare.csv 约 3.8GB）
- **获取条件**：无（公开下载）
- **本地放置路径**：`data/raw/gkx_datashare.zip`，解压得 `data/raw/datashare.csv`
  （慢速网络可用多线程下载器：`python data/fast_download.py`，自动放至该路径）
- **使用范围**：1957-03 ~ 2016-12，NYSE/AMEX/NASDAQ 约 3 万只股票。
  **字段对应关系**：该数据无 RET 列，`mom1m`（1 个月动量）按 Green-Hand-Zhang 定义即当月 CRSP 总收益率，本代码以其为月收益率；`sic2`（2 位 SIC 码）作为行业标签用于实例级对比。
- **数据划分**：train 1957-1989 / val 1990-1999 / test 2000-2016（按预测目标起始月归属，输入窗口可回溯，无泄露）。
- **预处理**：`python data/prepare_gkx.py`（面板化、减无风险利率得超额收益、活跃期缺失 >30% 剔除、缺失填 0、宏观变量 train 段标准化 → `data/processed/gkx_panel.npz`）

### b. Welch-Goyal 宏观数据（8 个宏观协变量 + 无风险利率）

- **数据集名称**：Welch & Goyal (2008) PredictorData（Amit Goyal 主页公开数据）
- **下载链接**：https://sites.google.com/view/agoyal145 （页面内 Google Sheets 导出 xlsx，取 **Monthly** 表）
- **获取条件**：无
- **本地放置路径**：`data/raw/goyal_1qwpl2R_DNujpU5YUkk8lacP1tTeMb9iJ.xlsx`（文件名随版本可能不同，需与 `data/extract_macro.py` 中 `SRC` 一致，可修改该变量）
- **预处理**：`python data/extract_macro.py` → 生成 `data/raw/macro_gw.csv`（dp, ep, bm, ntis, tbl, tms, dfy, svar 八变量）与 `data/raw/riskfree.csv`（月度无风险利率）

### c. ETTh1（辅助实验，可选）

- **数据集名称**：ETTh1（Electricity Transformer Temperature，小时级，7 通道）
- **下载链接**：https://github.com/zhouhaoyi/ETDataset （ETT-small/ETTh1.csv）
- **本地放置路径**：`data/raw/ETTh1.csv`
- **数据划分**：标准协议 train/val/test = 12/4/4 个月；seq_len=96, pred_len=96

## 运行命令（均从本文件夹根目录执行）

```bash
# 0. 数据准备 (按上节放置原始文件后)
python data/extract_macro.py        # 生成宏观变量与无风险利率 csv
python data/prepare_gkx.py          # 生成 GKX 面板 (数分钟)

# 1. 主实验: TimesFNP (+CL) vs base (无 CL), 主配置 (path_winz 口径, L=120, H=1)
python src/train.py --dataset gkx --model base     --horizon 1 --target path_winz --lookback 120 --epochs 40
python src/train.py --dataset gkx --model timesfnp --horizon 1 --target path_winz --lookback 120 --epochs 40 --lambda1 0.1 --lambda2 0.1

# 2. 评估与汇总 (最优权重自动存 checkpoints/, 日志在 results/logs/)
python src/evaluate.py --tag gkx_timesfnp_l01-01_L120p3D64_path_winz_H1   # 单次评估
python src/evaluate.py --aggregate                                        # 汇总 results/results.md, results.csv

# 3. 其他 horizon / 口径 (示例)
python src/train.py --dataset gkx --model timesfnp --horizon 3 --target path_winz --lookback 60 --epochs 40
python src/train.py --dataset gkx --model timesfnp --horizon 1 --scale z --lookback 60     # ret+z 收益口径

# 4. ETT 辅助实验 (可选, 无行业标签, 仅 temporal-wise 损失)
python src/train.py --dataset ett --model base     --horizon 96 --batch 32 --lambda1 0 --lambda2 0
python src/train.py --dataset ett --model timesfnp --horizon 96 --batch 32 --lambda1 0 --lambda2 0.1

# 一键运行主实验 (base + CL)
bash run_experiments.sh
```

训练超参（`python src/train.py --help`）：`--dataset {ett,gkx}`、`--target {ret,rank,path_winz,path_minmax,path_z}`、`--scale {raw,pct,z,winz}`、`--lookback`、`--patch`、`--dmodel`、`--lambda1/--lambda2`（对比损失权重）、`--tau`、`--keep-ratio`、`--noise-scale`、`--patience` 等。

**训练权重未提交**（体积考虑，见 .gitignore），用上述命令即可重新训练得到；单次训练在 RTX 4080 上约 3-15 分钟。
