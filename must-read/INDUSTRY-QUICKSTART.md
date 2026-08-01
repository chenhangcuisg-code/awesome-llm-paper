# 业界 LLM 快速入门 · Industry LLM Quickstart

> 面向「想快速摸清业界 LLM 是怎么做出来、怎么调出来、怎么部署、怎么变成 agent 的」读者。
> 不求全，只求**每个环节挑最经典 / 最能定调的那几篇**，按读的顺序排。
>
> 与 [`README.md`](./README.md)（学术向 must-read 全集）的关系：那份是广度目录，这份是**一条可以从头读到尾的主线**。

**四条主线**

| 模块 | 主题 | 论文数 | 大概读完时间 |
|---|---|---|---|
| [A](#a-训练全流程入门) | 训练全流程：pretrain → midtrain → post-train (PPO / GRPO / DPO / OPD) | 16 | 2–3 周 |
| [B](#b-经典技术报告--system-card) | 经典 technical report 与 system card | 14 | 1–2 周 |
| [C](#c-轻量化部署) | 轻量化部署：算法 + 工程项目 | 17 | 1–2 周 |
| [D](#d-agent-harness--self-evolving) | Agent harness + self-evolving harness | 16 | 1–2 周 |

**图例**：⭐ = 必读（每模块 3–5 篇，只读这些也能建立骨架） · 📄 = 论文 · 📕 = 技术报告 / system card · 🔧 = 工程项目（代码为主） · 📝 = 博客 / 非论文但影响力大

**⚠️ 关于 2026 年条目**：本文件收录了 2026 年新发布的报告（Kimi K3、GLM-5、Qwen3.5/3.6、Opus 5、GPT-5.6 等）。这些链接均经逐条访问核实，但版本迭代很快，引用前请再确认最新版本号。

---

## A. 训练全流程入门

一个现代 LLM 的生命周期：**预训练 → 中期训练 → 后训练（SFT / RLHF / RLVR）→ 蒸馏压缩**。这一节按这个顺序走。

### A0. 先搞清楚「规模」这件事

不理解 scaling law，后面所有的资源分配决策都无从谈起。

| | 论文 | 为什么读 | 链接 |
|---|---|---|---|
| ⭐📄 | **Scaling Laws for Neural Language Models** (Kaplan et al., OpenAI, 2020) | 第一次把 loss 写成参数量 / 数据量 / 算力的幂律函数。整个行业「加钱就能变强」的信念来源。 | [2001.08361](https://arxiv.org/abs/2001.08361) |
| ⭐📄 | **Training Compute-Optimal Large Language Models**（Chinchilla, Hoffmann et al., DeepMind, 2022） | 推翻 Kaplan 的配比结论：给定算力，**数据量应该和参数量同比例增长**。今天所有模型的 token 预算都是这篇的后代。读完这两篇再看任何技术报告，参数/数据配比就不再是黑箱。 | [2203.15556](https://arxiv.org/abs/2203.15556) |
| 📄 | **Language Models are Few-Shot Learners**（GPT-3, Brown et al., 2020） | 规模带来 in-context learning。历史坐标原点，快速扫一遍即可。 | [2005.14165](https://arxiv.org/abs/2005.14165) |

### A1. Pretrain：经典的「预训练对模型影响」研究

> 这一段回答用户的核心问题：**什么样的预训练决定了模型最终是什么样**。
> 结论先说：**2022 年以后，业界共识是「数据 > 架构」**。下面四篇是把这个共识钉死的四篇。

| | 论文 | 为什么读 | 链接 |
|---|---|---|---|
| ⭐📄 | **The Pile: An 800GB Dataset of Diverse Text**（Gao et al., EleutherAI, 2020） | 开源预训练语料的起点，第一次系统论证「语料**多样性**（22 个来源混合）本身就是一种能力来源」，而不只是堆量。后续所有数据配比研究的参照系。 | [2101.00027](https://arxiv.org/abs/2101.00027) |
| ⭐📄 | **The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale**（Penedo et al., HuggingFace, 2024） | **最推荐的一篇「预训练数据如何影响模型」实证研究**。把去重强度、质量过滤器、分类器筛选这些工程决策逐个做消融，用真实下游分数说话。是目前公开资料里，把「数据处理 pipeline 的每一步值多少分」讲得最透的。 | [2406.17557](https://arxiv.org/abs/2406.17557) |
| ⭐📄 | **DataComp-LM: In search of the next generation of training sets**（Li et al., 2024） | 把数据构造做成**受控 benchmark**：固定模型和算力，只变数据，看谁赢。让「数据配方」第一次变成可复现的科学问题而非玄学。 | [2406.11794](https://arxiv.org/abs/2406.11794) |
| 📄 | **Rho-1: Not All Tokens Are What You Need**（Lin et al., 2024） | 更细的粒度：不只筛文档，而是在**token 级别**筛。用参考模型给每个 token 打分，只在高价值 token 上算 loss。数学任务上小模型追平大模型。「不是所有 token 都值得学」这个观念很有启发。 | [2404.07965](https://arxiv.org/abs/2404.07965) |
| 📄 | **Physics of Language Models: Part 3.1, Knowledge Storage and Extraction**（Allen-Zhu & Li, 2023） | 用合成数据做受控实验，证明**知识能不能被提取出来，取决于预训练时数据是怎么呈现的**（同一事实要有多种改写）。解释了为什么模型「知道但答不出来」。整个 Physics of LM 系列都值得追。 | [2309.14316](https://arxiv.org/abs/2309.14316) |
| 📄 | **Phi-3 Technical Report**（Microsoft, 2024） | 「教科书级数据」路线的代表：3.8B 模型靠精选+合成数据打到 Mixtral 8x7B 水平。数据质量能换多少参数量，这篇给了个上界感觉。 | [2404.14219](https://arxiv.org/abs/2404.14219) |

### A2. Midtrain：中期训练

> Midtrain（中期训练 / continual pretraining / annealing）= 预训练主阶段之后、SFT 之前，用高质量或特定领域数据再训一段。
> 这个概念 2024–2025 才被正式命名，**目前最经典、被引用最多的定调论文是 OctoThinker**。

| | 论文 | 为什么读 | 链接 |
|---|---|---|---|
| ⭐📄 | **OctoThinker: Mid-training Incentivizes Reinforcement Learning Scaling**（Wang, Zhou, Li, Liu, 2025） | **这一段的必读**。第一篇系统回答「为什么 Qwen 系 base model 一 RL 就涨，Llama 系就不涨」——差别在 midtrain。提出 Stable-then-Decay 两阶段配方（200B tokens 恒定学习率 → 20B tokens 三分支 CoT + 衰减学习率），并开源 70B+ token 的 MegaMath-Web-Pro-Max。**结论：base model 的「RL 可训性」是 midtrain 阶段决定的，不是 RL 阶段能补救的。** | [2506.20512](https://arxiv.org/abs/2506.20512) |
| 📄 | **The Llama 3 Herd of Models**（Meta, 2024） | 不是专讲 midtrain，但它的 annealing 阶段（最后 40M token 上调高质量数据权重）是业界最早公开的完整 midtrain 实操记录之一。配合 OctoThinker 看，一个讲原理一个讲落地。 | [2407.21783](https://arxiv.org/abs/2407.21783) |
| 📄 | **2 OLMo 2 Furious**（AI2, 2025） | 全流程完全开源（数据 / 代码 / 中间 checkpoint / 训练曲线）。它的两阶段课程（web 为主 → 高质量混合退火）是可以**逐步复现**的 midtrain 参考实现。想动手的话从这个 repo 起步。 | [2501.00656](https://arxiv.org/abs/2501.00656) |

### A3. 四个后训练算法：PPO / GRPO / DPO / OPD

> 用户点名的四个。建议顺序：**PPO → DPO → GRPO → OPD**（先理解 RL 基线，再看去掉 RL 的简化，再看去掉 critic 的简化，最后看不用 RL 的蒸馏路线）。

#### PPO — 在线 RL 的基线

| | 论文 | 为什么读 | 链接 |
|---|---|---|---|
| ⭐📄 | **Proximal Policy Optimization Algorithms**（Schulman et al., OpenAI, 2017） | 原始 PPO。核心是 clipped surrogate objective——限制每步策略更新幅度，换取训练稳定。是后面所有 LLM RL 算法的共同祖先。**注意这是 2017 年的 RL 论文，不是 LLM 论文**，读的时候把 action 理解成 token 即可。 | [1707.06347](https://arxiv.org/abs/1707.06347) |
| ⭐📄 | **Training language models to follow instructions with human feedback**（InstructGPT, Ouyang et al., 2022） | PPO **怎么用在 LLM 上**的教科书。三阶段 pipeline：SFT → 训 reward model → PPO 优化。ChatGPT 的直接前身，「对齐」这个词的工业定义就出自这里。 | [2203.02155](https://arxiv.org/abs/2203.02155) |

#### DPO — 把 RL 消掉

| | 论文 | 为什么读 | 链接 |
|---|---|---|---|
| ⭐📄 | **Direct Preference Optimization: Your Language Model is Secretly a Reward Model**（Rafailov et al., Stanford, 2023） | 数学上证明「RLHF 的最优解有闭式表达」，于是**不用训 reward model、不用采样、不用 RL**，直接在偏好对上做分类损失。工程复杂度断崖式下降，一度成为开源社区默认对齐方法。理解它「为什么能消掉 RL」是关键。 | [2305.18290](https://arxiv.org/abs/2305.18290) |

#### GRPO — 把 critic 消掉

| | 论文 | 为什么读 | 链接 |
|---|---|---|---|
| ⭐📄 | **DeepSeekMath: Pushing the Limits of Mathematical Reasoning**（Shao et al., DeepSeek, 2024） | **GRPO 的原始出处**（很多人误引 R1，R1 是应用不是提出）。做法：同一 prompt 采样一组回答，用**组内相对得分**当 advantage，直接扔掉 value network——显存省一大截。今天几乎所有开源 RLVR 训练都在跑 GRPO 或它的变体。 | [2402.03300](https://arxiv.org/abs/2402.03300) |
| ⭐📕 | **DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL**（DeepSeek, 2025） | GRPO 的规模化验证。纯 RL（无 SFT 冷启动的 R1-Zero）就能自发涌现长 CoT 和「aha moment」。**「可验证奖励 RL（RLVR）」这条路线的定调之作**。 | [2501.12948](https://arxiv.org/abs/2501.12948) |
| 📄 | **DAPO: An Open-Source LLM Reinforcement Learning System at Scale**（Yu et al., ByteDance, 2025） | GRPO 的四个工程修补（clip-higher / 动态采样 / token 级损失 / 超长奖励整形）。想真正把 GRPO 跑稳，这篇是必要补充——原始 GRPO 在实践中会遇到的坑基本都在这里。 | [2503.14476](https://arxiv.org/abs/2503.14476) |
| 📄 | **GSPO: Group Sequence Policy Optimization**（Zheng et al., Qwen Team, 2025） | GRPO 的**序列级**修正。指出 token 级重要性采样在 MoE 上会累积方差导致崩溃，改到序列级即可稳定。Qwen3 的实际训练算法，MoE + RL 场景下建议直接看这篇。 | [2507.18071](https://arxiv.org/abs/2507.18071) |

#### OPD — On-Policy Distillation

> **OPD = On-Policy Distillation**：让**学生模型自己采样**，再由教师对学生生成的每个 token 打分。
> 解决的是普通蒸馏的分布错配问题——学生训练时看的是教师的句子，推理时走的是自己的句子，两者分布不一致。

| | 论文 | 为什么读 | 链接 |
|---|---|---|---|
| ⭐📄 | **On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes**（GKD, Agarwal et al., Google DeepMind, ICLR 2024） | **OPD 的原始论文**。提出 GKD（Generalized Knowledge Distillation）：在学生自采样的序列上用教师反馈训练，并支持替换散度函数（学生太小学不像教师时，forward KL 不是好选择）。还能和 RLHF 无缝组合。 | [2306.13649](https://arxiv.org/abs/2306.13649) |
| 📄 | **MiniLLM: On-Policy Distillation of Large Language Models**（Gu et al., 2023） | 同期工作，主张用**反向 KL** 代替正向 KL——避免学生把概率质量摊到教师的低概率区域（即「什么都学一点、什么都不像」）。和 GKD 对照读，能看清散度选择的意义。 | [2306.08543](https://arxiv.org/abs/2306.08543) |
| ⭐📝 | **On-Policy Distillation**（Kevin Lu, Thinking Machines Lab, 2025-10） | **最好的入门材料**，比论文好读。一句话概括 OPD 的定位：RL 是「稀疏奖励 + on-policy」，SFT 蒸馏是「稠密监督 + off-policy」，**OPD = 稠密监督 + on-policy，两者优点兼得**。文中给出 AIME'24 从 60% → 70% 只用约 150 步、相比 off-policy 蒸馏省 9–30× 算力的实测。注意页面有 2026-06 的更新插注。 | [thinkingmachines.ai](https://thinkingmachines.ai/blog/on-policy-distillation/) |

### A 模块读法建议

```
第 1 周：A0 (Kaplan + Chinchilla) → A1 (FineWeb + DataComp-LM)
第 2 周：A2 (OctoThinker) → A3 PPO 线 (PPO + InstructGPT) → DPO
第 3 周：A3 GRPO 线 (DeepSeekMath + R1 + DAPO) → OPD (TML 博客 + GKD)
```
只有一天：**Chinchilla → FineWeb → OctoThinker → DeepSeekMath → TML 的 OPD 博客**，五篇能把主线走完。

---

## B. 经典技术报告 / System Card

> **technical report** 和 **system card** 的区别：
> - **Technical report**：讲**怎么造的**（架构、数据、训练基础设施、消融）。中国实验室 + Meta 一般发这个。
> - **System card**：讲**安全评估结果**（能力上限、危险能力测试、缓解措施、红队结论）。OpenAI / Anthropic 主要发这个，架构细节基本不给。
>
> 想学怎么训模型 → 读 technical report。想了解前沿模型的能力边界和安全治理 → 读 system card。

### B1. 用户点名的五个（2026 最新）

| | 报告 | 组织 / 日期 | 亮点 | 链接 |
|---|---|---|---|---|
| ⭐📕 | **Kimi K3: Open Frontier Intelligence** | Moonshot AI · 2026-07-27 | 2.8T 总参 / 104B 激活 MoE，原生视觉 + 1M 上下文。三个架构创新：**Kimi Delta Attention (KDA)**（线性 attention delta rule，常数级单 token 解码开销）、**Attention Residuals**（跨深度信息流）、**Stable LatentMoE**（896 专家选 16）。相比 K2 整体 scaling 效率约 2.5×。后训练是覆盖 general / agentic / coding 的多档 reasoning effort RL。报告坦承仍落后 Claude Fable 5 与 GPT-5.6 Sol。**目前公开权重里工程细节最丰富的一份。** | [2607.24653](https://arxiv.org/abs/2607.24653) · [GitHub](https://github.com/MoonshotAI/Kimi-K3) |
| ⭐📕 | **GLM-5: from Vibe Coding to Agentic Engineering** | Z.ai (原智谱) · 2026-02 | ~744B MoE / ~40B 激活，200K 上下文，28.5T token 预训练。主题是从「vibe coding」转向「agentic engineering」。两个后训练贡献：**解耦生成与训练的异步 RL 基础设施**、**面向长程交互的异步 agent RL 算法**。**GLM-5.2**（2026-06-16 开源）在此基础上主打 long-horizon 任务与 FrontierSWE。 | [2602.15763](https://arxiv.org/abs/2602.15763) · [5.2 blog](https://z.ai/blog/glm-5.2) · [GitHub](https://github.com/zai-org/GLM-5) |
| ⭐📕 | **Qwen3 Technical Report** | Alibaba Qwen Team · 2025-05 | **最适合入门的中文实验室报告**：dense 与 MoE 双系列、thinking / non-thinking 双模式统一、thinking budget 机制、大小模型间的 strong-to-weak 蒸馏、119 语种。写得系统且可复现，是国内技术报告的模板级作品。<br>**后续版本**：Qwen3.5（2026-02-16，397B-A17B，稀疏 MoE + Gated Delta Networks + 早融合多模态预训练 + 百万级 agent 环境 RL + 201 语种）与 Qwen3.6（2026-04，主打 agentic coding 与 Thinking Preservation）**目前只有博客，尚无 arXiv 技术报告**。 | [2505.09388](https://arxiv.org/abs/2505.09388) · [Qwen3.5 blog](https://qwen.ai/blog?id=qwen3.5) · [GitHub](https://github.com/QwenLM/Qwen3.6) |
| ⭐📕 | **Claude Opus 5 System Card** | Anthropic · 2026 | Anthropic system card 的标准结构：能力评估 → 对齐评估 → 危险能力（CBRN / 网络安全 / 自主性）→ RSP 认定 → 缓解措施。**读 system card 主要是学「怎么系统评估一个模型能干什么、不能干什么」这套方法论**，架构细节不会给。<br>相关：[Sonnet 5](https://www-cdn.anthropic.com/9e6a1044980d8c4ed85669faf9c2a8342e2e9f1e/Claude%20Sonnet%205%20System%20Card.pdf) · [Fable 5 & Mythos 5](https://www-cdn.anthropic.com/2f9323abbcc4abe219577539efe19a623c9ca2bd/Claude%20Fable%205%20&%20Claude%20Mythos%205%20System%20Card.pdf) · [Transparency Hub](https://www.anthropic.com/transparency)（全部 system card 索引） | [Opus 5 PDF](https://www.anthropic.com/document/claude-opus-5-system-card) |
| ⭐📕 | **OpenAI GPT-5 System Card** | OpenAI · 2025-08（arXiv 2025-12，v2 2026-05） | 讲清了 GPT-5 的**路由器架构**（快模型 + 推理模型 + 实时 router，按复杂度/工具需求/用户意图分流），以及 **safe-completions** 安全训练法。生物/化学域按 Preparedness Framework 预防性判定为 High。<br>**最新版 GPT-5.6 System Card**（2026-07-09）覆盖 Sol / Terra / Luna 三档；三者在生化与网络安全均为 High、AI 自我改进低于 High；新增 CoT 可监控性、evaluation metagaming、agentic coding「超出用户意图」倾向等章节——**对做 agent 的人特别值得看**。 | [2601.03267](https://arxiv.org/abs/2601.03267) · [GPT-5.6 card](https://deploymentsafety.openai.com/gpt-5-6) |

### B2. 建议一并读的历史坐标

| | 报告 | 为什么读 | 链接 |
|---|---|---|---|
| ⭐📕 | **DeepSeek-V3 Technical Report** | **工程细节最硬的一份开源报告**：671B MoE / 37B 激活、MLA、无辅助损失负载均衡、MTP、原生 FP8 训练，2.788M H800 GPU 小时完成。想知道「低成本训大模型」到底怎么做，这是唯一一份写到底的。 | [2412.19437](https://arxiv.org/abs/2412.19437) |
| 📕 | **The Llama 3 Herd of Models** | 西方开源阵营的完整流程记录：数据、并行策略、训练故障处理（多少张卡多久坏一次都写了）、后训练。工程运维视角最全。 | [2407.21783](https://arxiv.org/abs/2407.21783) |
| 📕 | **Kimi K2: Open Agentic Intelligence** | K3 的前作，先读它再读 K3 能看清演进路线。MuonClip 优化器、大规模 agentic 数据合成。 | [2507.20534](https://arxiv.org/abs/2507.20534) |
| 📕 | **LongCat-Flash Technical Report** | 美团 LongCat：560B MoE，零计算专家（按 token 重要性动态分配算力）+ shortcut-connected MoE（通信与计算重叠）。「动态算力分配」这条思路值得单独关注。 | [2509.01322](https://arxiv.org/abs/2509.01322) |
| 📕 | **Qwen2.5 Technical Report** | Qwen3 的前作，18T token 预训练。想看一个系列如何迭代，2.5 → 3 的对照很有价值。 | [2412.15115](https://arxiv.org/abs/2412.15115) |
| 📕 | **GPT-4 Technical Report** | 历史意义大于技术含量（架构完全不公开）。作为「技术报告开始不讲技术」这一转折点的标志读一下。 | [2303.08774](https://arxiv.org/abs/2303.08774) |
| 📄 | **DeepSeek-R1 Thoughtology: Let's think about LLM Reasoning** | 第三方对 R1 推理链的系统解剖：推理长度的最优区间、长上下文管理、安全性。**读报告之外也要读别人怎么审报告。** | [2504.07128](https://arxiv.org/abs/2504.07128) |

### 怎么读技术报告（方法论）

1. **先看架构表**：参数量 / 激活量 / 层数 / 上下文 / 词表 —— 30 秒建立量级直觉。
2. **再看数据段**：token 总量、配比、清洗流程 —— 这里决定了模型上限。
3. **然后看训练基础设施**：并行策略、精度、故障率 —— 这段最难抄也最有价值。
4. **后训练段落**：SFT 数据从哪来、RL 用什么算法什么奖励。
5. **最后才看 benchmark 表**：**benchmark 数字是最不可信的部分**，各家评测设置不一致，横向对比要极其小心。
6. **System card 反过来读**：先看危险能力评估的方法设计，再看结论。

---

## C. 轻量化部署

> 分三层：**量化算法**（怎么把模型压小）→ **推理系统优化**（怎么把它跑快）→ **工程项目**（业界实际在用什么）。
> 本仓库另有 [`quantization/`](../quantization/) 目录，收录了 2025 年以后约 135 篇量化论文按 11 个方向分类。**这一节是入口，那里是纵深。**

### C1. 量化算法经典（按提出顺序）

| | 论文 | 一句话 | 链接 |
|---|---|---|---|
| ⭐📄 | **GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers**（Frantar et al., ICLR 2023） | **用户点名的这篇**。基于近似二阶信息（Hessian）逐层求解最优量化，175B 模型 4 小时量到 3–4 bit 且几乎无损。**PTQ 的事实标准起点**，今天 HuggingFace / vLLM / TensorRT-LLM 全都内置。要理解「为什么量化不是简单地 round」，读这篇。 | [2210.17323](https://arxiv.org/abs/2210.17323) |
| ⭐📄 | **AWQ: Activation-aware Weight Quantization**（Lin et al., MLSys 2024 best paper） | GPTQ 的主要对手，思路更简单：**只有约 1% 的权重通道是「显著」的，按激活幅度而非权重幅度来判定**，对这些通道做 per-channel scaling 保护。不需要反向传播、不过拟合校准集，边缘部署上尤其常用。**GPTQ vs AWQ 是部署选型的第一个岔路口。** | [2306.00978](https://arxiv.org/abs/2306.00978) |
| ⭐📄 | **SmoothQuant: Accurate and Efficient Post-Training Quantization for LLMs**（Xiao et al., ICML 2023） | W8A8 的经典解法。发现激活比权重难量化（有 outlier），于是**把量化难度从激活「平移」到权重**（数学等价变换）。让 INT8 激活量化真正可用，是所有 activation quantization 工作的共同起点。 | [2211.10438](https://arxiv.org/abs/2211.10438) |
| 📄 | **LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale**（Dettmers et al., 2022） | 第一次系统揭示 **emergent outlier features** 现象（超过 6.7B 后出现极端激活值，破坏朴素量化）。混合精度分解方案。**理解量化为什么难，必读这篇。** | [2208.07339](https://arxiv.org/abs/2208.07339) |

> **纵深阅读**：旋转类方法（QuaRot / SpinQuant）、FP4 / MXFP4（Blackwell 时代的默认精度）、KV cache 量化、MoE 量化、推理模型量化——见 [`quantization/`](../quantization/) 目录的 11 个专题文件。

### C2. 推理系统优化经典

| | 论文 | 一句话 | 链接 |
|---|---|---|---|
| ⭐📄 | **Efficient Memory Management for LLM Serving with PagedAttention**（Kwon et al., SOSP 2023） | **vLLM 的原始论文，本节最该读的一篇**。借鉴操作系统虚拟内存分页，把 KV cache 分块管理，消除碎片、支持共享前缀，吞吐提升 2–4×。**PagedAttention 已经是所有主流推理引擎的标配**，不懂这个就不算懂 LLM serving。 | [2309.06180](https://arxiv.org/abs/2309.06180) |
| ⭐📄 | **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness**（Dao et al., 2022） | 从 GPU 显存层级出发（HBM 慢、SRAM 快），用 tiling + 重计算把 attention 的显存复杂度从 O(N²) 降到 O(N)，**且结果精确不近似**。今天没有任何生产推理栈不用它。 | [2205.14135](https://arxiv.org/abs/2205.14135) |
| ⭐📄 | **Fast Inference from Transformers via Speculative Decoding**（Leviathan et al., ICML 2023） | 小模型快速起草若干 token，大模型一次并行验证——**输出分布严格不变**的前提下 2–3× 加速。所有 draft-verify 类方法（Medusa / EAGLE / MTP）的源头。 | [2211.17192](https://arxiv.org/abs/2211.17192) |
| 📄 | **SGLang: Efficient Execution of Structured Language Model Programs**（Zheng et al., 2023） | RadixAttention（前缀树式 KV 复用）+ 结构化输出约束。**多轮对话 / agent 场景下前缀复用率极高，SGLang 往往比 vLLM 更合适**——选型时要知道这个差别。 | [2312.07104](https://arxiv.org/abs/2312.07104) |
| 📄 | **Mixtral of Experts**（Jiang et al., Mistral, 2024） | MoE 部署的入门样本：47B 总参只激活 13B。**推理时显存看总参、算力看激活参**——这个不对称是 MoE 部署所有麻烦的根源。 | [2401.04088](https://arxiv.org/abs/2401.04088) |
| 📄 | **Small Language Models are the Future of Agentic AI**（Belcak et al., NVIDIA, 2025） | 立场论文：agent 系统里大多数调用是重复的、格式化的窄任务，用 SLM 更划算，只在必要时升级到大模型。**部署成本优化的架构级思路**，不是模型级。 | [2506.02153](https://arxiv.org/abs/2506.02153) |

### C3. 业界实际在用的工程项目

> 论文讲原理，项目讲落地。下面按**用途**而非流行度排列。

| | 项目 | 定位 · 什么时候选它 | 链接 |
|---|---|---|---|
| ⭐🔧 | **vLLM** | **在线服务事实标准**。PagedAttention + continuous batching，OpenAI 兼容 API，量化格式支持最全（GPTQ / AWQ / FP8 / MXFP4 / NVFP4）。**不确定用什么就用它。** | [github.com/vllm-project/vllm](https://github.com/vllm-project/vllm) |
| ⭐🔧 | **SGLang** | **agent / 多轮 / 高前缀复用场景的更优解**。RadixAttention 让共享前缀近乎免费，结构化输出（JSON schema / 正则）是一等公民。DeepSeek 官方部署栈。 | [github.com/sgl-project/sglang](https://github.com/sgl-project/sglang) |
| ⭐🔧 | **llama.cpp / GGUF** | **端侧与 CPU 部署的唯一主流答案**。纯 C/C++ 无依赖，GGUF 格式 + k-quant 系列量化，Mac / 手机 / 树莓派都能跑。Ollama、LM Studio 底层都是它。 | [github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) |
| 🔧 | **TensorRT-LLM** | NVIDIA 官方，**极致单卡/单机延迟**。编译式优化，FP8 / FP4 支持最早最深。代价是灵活性差、迭代模型支持慢。追求极限性能且锁定 NVIDIA 时用。 | [github.com/NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) |
| 🔧 | **Ollama** | **本地开发体验最好**。一行 `ollama run` 拉起模型，包装 llama.cpp。原型验证 / 个人使用首选，不适合生产高并发。 | [github.com/ollama/ollama](https://github.com/ollama/ollama) |
| 🔧 | **llm-compressor**（原 SparseML） | vLLM 官方配套的**压缩工具链**：GPTQ / SmoothQuant / SparseGPT / FP8 一站式，产出直接被 vLLM 加载。**从「有个 HF 模型」到「有个量化好的可部署模型」这一步用它。** | [github.com/vllm-project/llm-compressor](https://github.com/vllm-project/llm-compressor) |
| 🔧 | **AutoAWQ / GPTQModel** | 单算法量化工具。要精细控制量化过程（分组大小、校准集、act-order）时用，比 llm-compressor 更贴近论文原始实现。 | [AutoAWQ](https://github.com/casper-hansen/AutoAWQ) · [GPTQModel](https://github.com/ModelCloud/GPTQModel) |
| 🔧 | **FlashInfer** | attention / sampling kernel 库，被 vLLM 与 SGLang 共同依赖。**想改 kernel 层的话，改这里而不是改引擎。** | [github.com/flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer) |
| 🔧 | **bitsandbytes** | QLoRA 的底层（NF4 量化 + 分页优化器）。**训练侧**省显存用它，推理侧性能不如上面几个。 | [github.com/bitsandbytes-foundation/bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) |
| 🔧 | **KTransformers** | 异构（CPU+GPU）推理，**单卡跑超大 MoE** 的方案：专家权重放 CPU/内存，激活部分上 GPU。消费级硬件跑 DeepSeek-V3 级模型靠它。 | [github.com/kvcache-ai/ktransformers](https://github.com/kvcache-ai/ktransformers) |

### 部署选型决策树（速查）

```
要在服务器上开 API 服务？
├── 高并发 / 通用            → vLLM
├── agent、多轮、共享长前缀   → SGLang
└── 锁 NVIDIA 且要极限延迟    → TensorRT-LLM

要在个人设备 / CPU 上跑？     → llama.cpp（生产）/ Ollama（开发）
显存不够跑超大 MoE？          → KTransformers（CPU offload）
需要先把模型压小？
├── 走 vLLM 生态             → llm-compressor
└── 要精细控制单个算法        → AutoAWQ / GPTQModel

量化格式怎么选？
├── W4，GPU，通用            → AWQ 或 GPTQ（4-bit，group 128）
├── W8A8，追吞吐             → SmoothQuant / FP8
├── Blackwell (B100/B200)   → NVFP4 / MXFP4
└── CPU / 端侧               → GGUF k-quant (Q4_K_M 是通用甜点)
```

---

## D. Agent Harness + Self-Evolving

> **Harness（脚手架）** = 包在模型外面的那层代码：工具定义、上下文管理、循环控制、错误恢复、记忆。
> 同一个模型换个 harness，SWE-bench 分数能差十几个点——**harness 是 agent 工程的主战场**。
> **Self-evolving harness** = 让 agent 自己改自己的 harness / 自己积累技能，而不是靠人手调。

### D1. Agent 的四块基础拼图

| | 论文 | 贡献的那块拼图 | 链接 |
|---|---|---|---|
| ⭐📄 | **ReAct: Synergizing Reasoning and Acting in Language Models**（Yao et al., ICLR 2023） | **循环结构**。Thought → Action → Observation 交替。这个模式简单到今天几乎所有 agent 框架都还在用它的变体。**agent 的「Hello World」，必读。** | [2210.03629](https://arxiv.org/abs/2210.03629) |
| ⭐📄 | **Reflexion: Language Agents with Verbal Reinforcement Learning**（Shinn et al., NeurIPS 2023） | **失败恢复**。把失败的执行结果转成自然语言反思存进记忆，下一轮带着教训重试——不更新任何参数的「语言强化学习」。 | [2303.11366](https://arxiv.org/abs/2303.11366) |
| 📄 | **Self-Refine: Iterative Refinement with Self-Feedback**（Madaan et al., 2023） | **自我批评**。同一个模型生成 → 自评 → 修订的闭环。**注意读它的局限：没有外部信号时，自我批评的收益有上限**（这条限制直接催生了后面 D3 的验证器研究）。 | [2303.17651](https://arxiv.org/abs/2303.17651) |
| 📄 | **Voyager: An Open-Ended Embodied Agent with Large Language Models**（Wang et al., 2023） | **技能库**。在 Minecraft 里让 agent 把成功的行为写成可复用代码存起来，能力随时间单调累积。**self-evolving 的思想源头之一**，D3 的很多工作都在引它。 | [2305.16291](https://arxiv.org/abs/2305.16291) |

### D2. 真正跑起来的 harness（工程经典）

| | 工作 | 关键洞见 | 链接 |
|---|---|---|---|
| ⭐📄🔧 | **SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering**（Yang et al., NeurIPS 2024） | **本节最重要的一篇**。提出 **ACI（Agent-Computer Interface）** 概念并给出实证：**同一个模型，仅仅改进给它的工具接口设计（文件查看器怎么分页、编辑命令怎么反馈语法错误、搜索结果怎么截断），SWE-bench 分数就能翻倍。** 这篇把「harness 工程」正式确立为一个独立学科——不是提示词工程，是接口设计。 | [2405.15793](https://arxiv.org/abs/2405.15793) · [GitHub](https://github.com/SWE-agent/SWE-agent) |
| ⭐📄🔧 | **OpenHands: An Open Platform for AI Software Developers as Generalist Agents**（Wang et al., 2024） | **最完整的开源 agent harness 实现**（原 OpenDevin）。沙箱执行、浏览器、多 agent 委派、事件流架构。想读 harness 源码，读这个仓库。 | [2407.16741](https://arxiv.org/abs/2407.16741) · [GitHub](https://github.com/All-Hands-AI/OpenHands) |
| ⭐📄 | **SWE-bench: Can Language Models Resolve Real-World GitHub Issues?**（Jimenez et al., ICLR 2024） | **评测基准，但必须放进来**——因为 harness 的所有改进都是被这个 benchmark 拽着走的。理解它的构造方式（真实 PR + 单元测试验证）和已知缺陷（测试泄漏、环境依赖），才能正确解读所有 agent 分数。 | [2310.06770](https://arxiv.org/abs/2310.06770) |
| 📕 | **Claude Code / Anthropic 工程实践** | Anthropic 关于「怎么构建有效 agent」的工程博客，主张**优先用简单可组合的模式，而不是复杂框架**。业界影响力很大，读完前面论文后看这个能校正很多过度设计的冲动。 | [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents) |
| 📄 | **Harness Handbook**（2026-07） | 以行为为中心给 harness 代码画地图，让 agent 能定位「该改哪里」。**为 self-evolving harness 服务的基础设施工作。** | [2607.13285](https://arxiv.org/abs/2607.13285) |
| 📄 | **ToFu: 开放、token 高效的白盒研究型 harness**（2026-07） | 既能当助手用、也能当研究对象拆的 harness。**想做 harness 研究而不只是用 harness，从这个入手。** | [2607.11423](https://arxiv.org/abs/2607.11423) |

### D3. Self-Evolving：让 agent 改进自己

> 按「改什么」分成三层，**越往下自主性越高、风险越大**：
> **① 改上下文/记忆**（不动代码）→ **② 改 harness 代码**（动脚手架）→ **③ 改自己整体**（动一切）

| | 工作 | 层级 · 核心机制 | 链接 |
|---|---|---|---|
| ⭐📄 | **STaR: Bootstrapping Reasoning With Reasoning**（Zelikman et al., NeurIPS 2022） | **自举训练的鼻祖**。模型生成推理链 → 只保留答对的 → 拿去微调自己 → 循环。**所有 self-improvement 工作的共同祖先**，虽然它改的是权重不是 harness。 | [2203.14465](https://arxiv.org/abs/2203.14465) |
| ⭐📄 | **ADAS: Automated Design of Agentic Systems**（Hu et al., ICLR 2025） | **②**。让 meta-agent **用代码写出新的 agent**，在图灵完备的代码空间里搜索架构，维护发现档案避免重复。「agent 设计本身可以自动化」这个命题的奠基论文。 | [2408.08435](https://arxiv.org/abs/2408.08435) |
| ⭐📄 | **Darwin Gödel Machine: Open-Ended Evolution of Self-Improving Agents**（Zhang, Hu, Lu, Lange, Clune, 2025） | **③，本节最该读的一篇**。agent **迭代修改自己的代码**，用 benchmark 实证验证改动（而非像原始 Gödel machine 那样要求形式化证明）。维护 agent 档案 + 分支式并行探索（达尔文式开放演化）。结果：SWE-bench **20.0% → 50.0%**，Polyglot **14.2% → 30.7%**。论文明确讨论了沙箱与人类监督等安全措施——**做这个方向必须一并读安全部分。** | [2505.22954](https://arxiv.org/abs/2505.22954) · [GitHub](https://github.com/jennyzzt/dgm) |
| ⭐📄 | **Self-Authored Verification Is Unreliable in Heuristic Self-Improving Agents**（2026-07） | **必读的反面证据**。指出 self-evolving 系统的核心失效模式：**「verifier–deployment gap」——agent 自己写的验证器会系统性地高估自己的改动**，导致在真实部署中回退。提出 SEAL（外部密封验收环）阻断回归。**在被 DGM 的漂亮数字说服之前，先读这篇。** | [2607.24300](https://arxiv.org/abs/2607.24300) |
| 📄 | **Self-Evolving Agent Harnesses via Gated Semantic Quality-Diversity**（2026-07） | **②**。把「提出补丁」和「归因功劳」解耦，用按失效模式索引的门控档案维护多样性。报告 +9 到 +15.5pp。**目前 harness 自演化方向最直接的一篇。** | [2607.13683](https://arxiv.org/abs/2607.13683) |
| 📄 | **MemoHarness: 从经验中学习的 harness**（2026-07） | **①**。双层经验库，跨六个控制维度积累。**风险最低的一层自演化**（不改代码只改记忆），落地性最强，建议实践从这层开始。 | [2607.14159](https://arxiv.org/abs/2607.14159) |
| 📄 | **SEED: Self-Evolving On-Policy Distillation**（2026-07） | **跨界**：把完成的轨迹转成 hindsight 技能，再蒸馏回 agentic RL 策略。**A 模块的 OPD 和 D 模块的 self-evolving 在这里合流**——值得注意的趋势。 | [2607.14777](https://arxiv.org/abs/2607.14777) |
| 📄 | **MetaEvolve: Teaching LLMs to Self-Evolve**（2026-07） | 用 RL 训练「多轮自我改进」这项元技能本身，报告在分布外编码任务上有较大增益。 | [2607.21971](https://arxiv.org/abs/2607.21971) |
| 📄 | **AREX: Recursively Self-Improving Agent for Deep Research**（2026-07） | 内层研究循环 + 外层审计精炼循环的双环设计，面向长程任务。 | [2607.21461](https://arxiv.org/abs/2607.21461) |
| 📄 | **Agent Harness Distillation**（2026-07） | 反过来的视角：**harness 可以被黑盒提取**，所以它是可泄漏的 IP。附带防御方案。做商业 agent 产品的话这篇有直接影响。 | [2607.28147](https://arxiv.org/abs/2607.28147) |

### D 模块读法建议

```
入门（1 天）：ReAct → SWE-agent（读 ACI 那一节就够）→ Anthropic「Building Effective Agents」
进阶（3 天）：Reflexion + Voyager → OpenHands（读代码）→ SWE-bench（理解评测）
前沿（1 周）：STaR → ADAS → Darwin Gödel Machine → 立刻接 Self-Authored Verification（反面）
              → MemoHarness（①）→ Gated Semantic QD（②）
```

**给做 self-evolving 的人三条提醒**（都来自上面论文的实测教训）：
1. **验证器是瓶颈，不是生成器。** 自己写的验证器会自我偏袒（见 SEAL 那篇），必须有外部或密封的验收信号。
2. **档案 / 多样性机制不是可选项。** DGM 和 ADAS 都强调：没有档案就会过早收敛到局部最优。
3. **从改记忆开始，不要一上来就改代码。** ①→②→③ 的风险阶梯是真实的，沙箱和人类监督在每一层都要有。

---

## 全局速通路线

**只有一周**，想把四个模块都摸一遍：

| 天 | 读什么 |
|---|---|
| D1 | Chinchilla → FineWeb（规模 + 数据） |
| D2 | OctoThinker → InstructGPT（midtrain + 后训练全景） |
| D3 | DeepSeekMath (GRPO) → DPO → TML 的 OPD 博客（四个算法） |
| D4 | DeepSeek-V3 报告 → Kimi K3 报告（怎么造模型） |
| D5 | GPTQ → AWQ → vLLM/PagedAttention（怎么部署） |
| D6 | ReAct → SWE-agent → Building Effective Agents（怎么做 agent） |
| D7 | Darwin Gödel Machine → Self-Authored Verification（前沿与它的反面） |

---

## 术语速查

| 缩写 | 全称 | 含义 |
|---|---|---|
| **PPO** | Proximal Policy Optimization | 在线 RL 基线算法，用 clip 限制策略更新幅度 |
| **GRPO** | Group Relative Policy Optimization | 去掉 value network，用组内相对得分作 advantage |
| **DPO** | Direct Preference Optimization | 去掉 RL，直接在偏好对上做分类损失 |
| **OPD** | On-Policy Distillation | 学生自采样、教师逐 token 打分的蒸馏 |
| **GKD** | Generalized Knowledge Distillation | OPD 的原始论文名称 |
| **RLHF** | RL from Human Feedback | 用人类偏好训 reward model 再做 RL |
| **RLVR** | RL with Verifiable Rewards | 用可自动验证的奖励（如数学答案对错）做 RL，R1 路线 |
| **SFT** | Supervised Fine-Tuning | 监督微调 |
| **CoT** | Chain of Thought | 思维链 |
| **PTQ** | Post-Training Quantization | 训练后量化，不需重训 |
| **QAT** | Quantization-Aware Training | 量化感知训练 |
| **W4A16** | Weight 4-bit, Activation 16-bit | 量化配置记法：权重 4 bit、激活 16 bit |
| **KV cache** | Key-Value cache | 自回归解码缓存的历史 K/V，长上下文的主要显存瓶颈 |
| **MoE** | Mixture of Experts | 混合专家，总参大但每 token 只激活一部分 |
| **MLA** | Multi-head Latent Attention | DeepSeek 提出的低秩 KV 压缩注意力 |
| **MTP** | Multi-Token Prediction | 一次预测多个 token，可用于加速解码 |
| **ACI** | Agent-Computer Interface | agent 与计算机交互的接口设计（SWE-agent 提出） |
| **Harness** | — | 包在模型外的脚手架代码：工具、循环、上下文、记忆 |
| **midtrain** | mid-training | 预训练主阶段与 SFT 之间的高质量数据再训练阶段 |
| **annealing** | — | midtrain 的一种：末期上调高质量数据权重并衰减学习率 |

---

## 说明

- **收录标准**：① 该子方向公认的奠基工作，或 ② 业界实际大规模在用的方法/项目，或 ③ 2026 年最新且方向性明确的工作。宁缺毋滥，每个小节控制在 3–7 条。
- **链接核实**：所有 arXiv ID、标题、作者均通过 arXiv API 或直接访问页面逐条核对（2026-08-01）。GitHub 项目链接为访问核实的官方仓库。
- **已知不确定**：Qwen3.5 / Qwen3.6 截至核实时**仅有官方博客、无 arXiv 技术报告**；GLM-5.2 同理（arXiv 上只有 GLM-5）。Thinking Machines 的 OPD 博客页面含 2026-06 的更新插注，与 2025-10 原版可能有出入。
- **欢迎 PR**：补充遗漏的经典工作、修正链接、更新新版本报告。
