# 别人是怎么做 baseline 对比的：ZeroPrint、AdaptPrint、BReF

> 文献学习报告，design-side，2026-10-07。
> 目的：在决定我们的 MET / ZeroPrint baseline 怎么跑之前，先看已发表/近期的论文怎么比这几个方法。
> 数字全部转述自原文（arXiv HTML 版），未独立复现。AdaptPrint、BReF 为预印本，未经同行评审。

---

## 0. 一页结论（先读这个）

1. **同一个 baseline，换一篇论文，排名就翻。** MET 在 AdaptPrint 里几乎垫底（Top-1 4–8%），在 BReF 里是最强的黑盒 baseline（Top-1 64%）；ZeroPrint 在 AdaptPrint 里很强，在 BReF 里跌到 32%。**别人论文里的 baseline 数字不能搬过来用，必须在我们的设置下重跑。**
2. **翻转的方向和我们之前的预判一致：**
   - MET 对*任何*分布变化都敏感：BReF 的任务是在兄弟模型里找亲本，含 6 条量化边，MET 表现好；AdaptPrint 的目标挂了 system prompt，温度也和参考不一样，MET 就崩了。
   - ZeroPrint 是为"对衍生不敏感"设计的：粗粒度溯源好，精确区分近亲差。
   - **我们的设置同时有"配置变化"和"近亲"两个难点，两个方法很可能各输一半。** 这是要验证的假设，不是结论。
3. **LLMmap 在别人论文里几乎都是它最弱的用法：** ZeroPrint、BReF 明确用的是**官方预训练的 open-set encoder**，AdaptPrint 没说清楚。没有一篇按 LLMmap 自己的 closed-set 重训练模式跑。
4. **三篇里方法学最严谨的是 BReF**，值得照抄它的做法：
   - 每个 baseline 保留"原生协议"，只用它的最终分数排序；
   - 附录逐个写清复现细节（解码参数、token 上限、官方代码的 commit）；
   - 明说预算*不对齐*、只做透明报告；
   - 只用量纲无关的指标；
   - 按 suspect 做 cluster bootstrap，配对 McNemar 加 Holm 校正；
   - 另设一张"内部对照"表（随机选 K 个 50 个种子、Bottom-K、全池），和"已有方法"分开。
5. **三篇都没有比"query 选择"本身。** BReF 的 Random-25 / Bottom-25 / Query-only Top-25 对照最接近我们的选择消融，但只是它自己方法的内部 ablation。

---

## 1. 三篇论文速览

| | **ZeroPrint**（WWW'26，arXiv 2510.06605） | **AdaptPrint**（预印本，arXiv 2608.22213） | **BReF**（预印本，arXiv 2609.06330） |
|---|---|---|---|
| 任务 | 溯源：可疑模型是否衍生自源模型（**微调、system prompt、RAG 都算"同一个"**） | 识别：黑盒服务背后是 27 个候选中的哪一个 | 直接亲本检索：给定衍生模型，在 19 个候选亲本里排出真亲本（兄弟模型作为 hard negative 保留） |
| 模型 | LeaFBench：7 个基座，149 个实例，1.1B–14B | 27 个 API 模型（2025–26），12 个作为目标 | 34 个开源 checkpoint，22 个 suspect，411 个 pair |
| 自己的信号 | 文本输出 → mpnet embedding → 扰动-响应 Jacobian | 辅助 LLM 生成的领域问题 + 续写 / 追问，cosine + n-gram Jaccard | **A/B/C/D 选项概率**（需要 logprob，属于灰盒） |
| baseline | LLMmap、MET、SEF、TRAP（半白盒）、REEF（白盒，仅作参考） | MET、LLMmap、ZeroPrint | REEF、LLMmap、MET、ZeroPrint，外加自建 QA-Agreement |
| 指标 | AUC、pAUC（FPR≤0.05）、TPR@1%FPR、Mahalanobis 距离 | Top-1/3/5 | Top-1/3、MRR、ROC-AUC（DP–DF、DP–All） |
| 统计 | 均值±std（没说是几个种子） | 无区间 | suspect-cluster bootstrap 10k，配对 McNemar + Holm，按 family 聚类做敏感性检验 |

---

## 2. ZeroPrint 怎么比 MET 和 LLMmap

**设定**

- 所有方法设"查询上限 q=200"，沿用 REEF 的做法，号称公平。
- 实际执行：
  - **LLMmap**：用 8 个原版 query，加官方仓库里**预训练的 open-set 特征提取器**，没有重训练。实际只用了 8 次查询，不是 200 次。
  - **MET**："原版用 25 个样本，所以跑 8 次，凑满 200"。8 次的结果怎么合成一个分数，文中没说。
  - **SEF**：Qwen3-Embedding-4B，从 DROP/Ethics/PubMedQA/HumanEval 采 200 个 query。
  - **TRAP**：GCG 优化 100 步。属于半白盒，文中单独标注。
  - **REEF**：白盒，200 条 TruthfulQA，只作为参考上限。

**结果**（LeaFBench，Table 1）

| 方法 | AUC | pAUC | TPR@1%FPR | MD |
|---|---|---|---|---|
| REEF（白盒） | .896 | .832 | .634 | 2.091 |
| LLMmap | .632 | .635 | .253 | 0.521 |
| MET | .661 | .658 | .329 | 1.457 |
| SEF | .581 | .646 | .282 | 0.172 |
| TRAP | .712 | .625 | .227 | 1.382 |
| **ZeroPrint** | **.720** | **.683** | **.366** | 1.457 |

**值得注意的地方**

- ZeroPrint 对 TRAP 的 AUC 领先（.720 vs .712）小于各自的 std。MD 和 MET 打平。论文没做显著性检验。
- 只有一张总表：没有按衍生类型（微调、合并、蒸馏、system prompt、RAG）拆分，看不出 baseline 输在哪类衍生上。
- 鲁棒性实验（改写输入、扰动输出）只测了 ZeroPrint 自己，baseline 没测。
- 预算 ablation 只扫了 ZeroPrint 自己（t、m 变大，AUC 最高到 .778），没有在相同预算下和 baseline 对比。
- 开销：在 4 个 LLM 上，ZeroPrint 424.9 s，比 MET、SEF、TRAP 快，比 REEF 和 LLMmap 慢。
- LeaFBench 和 SEF 都来自 Shao et al. 2025a，看起来和 ZeroPrint 是同一作者组，所以 benchmark 不算独立第三方。这一点是从引用和代码仓库推断的，没有核实。

**对我们的启示：** 名义上的"同一预算上限"并不等于真的公平，LLMmap 只用了 8 次查询。这种写法审稿人一眼就能看穿。

---

## 3. AdaptPrint 怎么比 MET、LLMmap、ZeroPrint

**设定**

- **目标**：12 个模型 × 18 个 system prompt：10 个来自 GitHub 的服务风格 prompt（452–2731 字符），6 个来自 LLMmap（含 CoT），2 个自建 RAG。目标解码 T=0.7、top-p=0.95。
- **AdaptPrint 自己的候选**：T=0.0。
- **baseline 统一当作"模板库匹配"来用**：先从候选模型抽指纹，再和目标逐一比。参考库有两种：
  - **Method-Gen**：参考指纹只用一个通用 system prompt 下的；
  - **Method-All**：参考指纹用*其他所有* system prompt 下的实例（留一法式的）。
  - 为了和 All 对齐，AdaptPrint 自己的候选也限制在这 12 个目标模型里。
- baseline 用官方代码还是重写、LLMmap 是否重训练、MET 的 MMD 怎么变成 27 选 1 的分数，**文中都没说**。

**结果**（Table IV）

| | LLMmap | MET | ZeroPrint | AdaptPrint-Triple |
|---|---|---|---|---|
| Top-1 Gen | 36.6% | **7.5%** | 48.1% | 80.6% |
| Top-1 All | 57.4% | **4.2%** | 78.2% | 87.0% |
| Top-5 All | 81.5% | 9.9% | 88.9% | 94.4% |

- 作者对 MET 的解释是："许多模型被打成相同的分数"。
- **红旗：MET 在 All 下比 Gen 下更差**（7.5% → 4.2%）。参考数据多了反而变差，基本说明适配方式本身有问题，不是方法真的这么弱。
- 另一个可能的原因：MET 的 Hamming 核是逐字符比较，目标挂了长 system prompt、T=0.7，参考是另一个配置，逐字符对齐就失效了。这正是我们预判的 MET 在"未知部署配置"下的失败方式。

**guardrail 对比**（Table V）

- 被 off-topic 拦截的比例：LLMmap 133/144、MET 399/450、ZeroPrint 112/180、AdaptPrint 3/54。
- 分母等于"固定 query 数 × 18 个 prompt"。AdaptPrint 的问题是按目标领域*专门生成*的，所以这一比天然对它有利。这本来就是它想论证的点，但不算同等条件下的对比。

**其他**

- 开放集识别只做了自己的方法，baseline 以"没讨论阈值"为由略过。
- 微调实验只有 2 个 LoRA 模型。**认成其基座模型就算对**，这是溯源口径，和我们的版本识别口径相反。
- 没有任何置信区间。

**对我们的启示**

- Gen/All 是个好设计：它把"参考库覆盖了多少部署配置"单独当成一个变量来考察，和我们 D024 的 SINGLE→CENTROID 是同一个思想。
- 但 baseline 的适配细节不公开，导致 MET 的数字没法解释。我们必须把适配细节写全。

---

## 4. BReF：为什么选它作为第三篇

选它是因为：
- 它**同时复现了 MET、ZeroPrint、LLMmap**；
- 任务是**在兄弟模型中做细粒度区分**，最接近我们的近亲难点；
- 方法学在三篇里最严谨。

Helm et al. 2605.07878 和我们一样做 query 选择，但它不是指纹 baseline 的对比范例，留到起草 F 时单独读。

**复现原则**（§5.3、附录 C）

- "保留每个方法原生的探测、表示和打分"，只拿最终分数在同一个候选池上排序；
- 明说"各协议信号和预算不同，比的是检索性能，不是等成本效率"；
- 预算单独列表（Table 12），只是为了透明。

**逐方法复现细节**（附录 C，我们可以直接参考）

| 方法 | 每个 checkpoint 的协议 | 分数 |
|---|---|---|
| LLMmap | 8 个固定 query，greedy，≤256 新 token，截断到 650 字符（照官方实现）；e5 embedding → **官方 open-set encoder** → 384 维 | cosine 距离 |
| MET | 25 条多语言 Wikipedia prompt × 10 个样本；T=1.0、top-p=1、≤50 新 token；Unicode 字符序列，L=1000；prompt-aware 归一化 Hamming 核；无偏 MMD；100 次置换 | MMD（越小越像） |
| ZeroPrint | **官方代码，冻结在某个 commit**；HumanEval；2 条 query × (1+4 扰动) × 20 次采样 = 200 次生成；T=0.7、top-p=0.9、≤512 新 token；mpnet；ridge α=0.001 | Pearson |
| REEF（白盒） | 200 条 TruthfulQA，第 18 层最后一个 token 的激活，linear CKA；用一个已验证的参考值做回归检查 | CKA |

**指标与统计**

- 只用量纲无关的指标（Top-1/3、MRR、AUC）。距离型分数取负号统一方向。各方法原始分数的差距**不做跨方法比较**。
- 按 suspect 做 cluster bootstrap（因为 411 个 pair 共享同一批 suspect，不是独立同分布）。
- 配对 McNemar 加 Holm 校正，再按 6 个 family 聚类做敏感性检验。

**结果**（Table 2）

| 方法 | 信号 | Top-1 | MRR | AUC DP–DF |
|---|---|---|---|---|
| LLMmap | 文本 | 22.7% | .410 | .703 |
| ZeroPrint | 文本 | 31.8% | .423 | .714 |
| MET | 采样 | **63.6%** | .696 | .779 |
| QA-Agreement | 硬标签 | 63.6% | .720 | .857 |
| REEF | 隐层表示 | 68.2% | .817 | .968 |
| BReF | 选项概率 | 100% | 1.000 | 1.000 |

**内部对照表**（Table 4、5，单独成表，不和已有方法混在一起）

- Random-25（50 个种子，报均值±std）、Bottom-25、全部 971 条、只换选择器的版本，以及几种去掉"方向"信息、只留"幅度"的对照。
- 它们的核心发现："cross-family 的 AUC 高，不等于能精确找出亲本。" Random-25 的 DP–DF AUC 有 .98，但 Top-1 只有 18/22。**这和我们"均值掩盖 hard tail"（I1）是同一个现象。**

**它的局限**

- 依赖选项 logprob，属于灰盒，和 MET、ZeroPrint 的纯文本威胁模型不同。文中按"信号"一列标出来了，没有回避。
- 只有 22 个 suspect，样本小。

---

## 5. 横向对比：同一个 baseline 在三篇里的成绩

| baseline | ZeroPrint 论文（溯源，AUC） | AdaptPrint（27 选 1，Top-1 Gen / All） | BReF（找亲本，Top-1 / MRR） |
|---|---|---|---|
| LLMmap | .632 | 36.6% / 57.4% | 22.7% / .410 |
| MET | .661 | **7.5% / 4.2%** | **63.6% / .696** |
| ZeroPrint | .720（作者自报） | 48.1% / 78.2% | 31.8% / .423 |

**怎么读这张表（后两条是假设，待检验）**

- **MET 和 ZeroPrint 的相对排名在 BReF 里翻转了。**
  - BReF 的任务要求区分兄弟模型和量化版本。MET 当初就是为检测量化、微调这类分布偏移设计的，所以占优。
  - AdaptPrint 的目标和参考处在不同配置下，MET 对配置偏移同样敏感，于是崩盘。
  - ZeroPrint 反过来：对衍生不敏感，所以适合粗溯源、不适合精确区分。
- **我们的问题同时包含两个难点。** 部署配置未知，这会伤 MET；近亲 pair 是论文支柱之一，这会伤 ZeroPrint。如果实测确实如此，就可以写成："已发表的两类方法各自在我们问题的一半上失败。"
- **LLMmap 三篇都偏低。** 但它们用的都是（或很可能是）预训练 open-set encoder，不是 LLMmap 论文报告 95.3% 的那种 closed-set 重训练模式。

---

## 6. 公平性杠杆清单：每篇怎么处理

| 杠杆 | ZeroPrint | AdaptPrint | BReF |
|---|---|---|---|
| 查询预算 | 名义上限 q=200，实际不对齐（LLMmap 只用 8） | 不对齐，没讨论 | **明确不对齐，列表透明报告** |
| baseline 代码 | "我们实现"，没说是否官方 | 没说 | 官方代码 + commit；REEF 做回归检查 |
| baseline 原生协议 | 部分保留（MET 被"跑 8 次"） | 改造成模板库匹配 | **完整保留** |
| LLMmap 模式 | 预训练 open-set | 没说 | 预训练 open-set |
| 参考 vs 目标配置 | 没涉及 | Gen / All 两种参考库；目标 T=.7，自己的候选 T=0 | 各方法用各自的原生解码 |
| 威胁模型差异 | 白盒、半白盒单独标注 | 没涉及 | 按"信号"一列标注 |
| 不确定性 | ±std | 无 | cluster bootstrap + McNemar/Holm |
| 细分分析 | 无 | 按模型、按配置画热图（只画自己的方法） | 按衍生类型、按 family |
| 内部对照 | 自己方法的 ablation | 自己方法的 ablation | **单独一张对照表**（随机选 K 多种子等） |

---

## 7. 对我们的启示（供讨论，不是决定）

前面提出的三个待决问题（MET 碰 I5、ZeroPrint 新生成、F 用哪个语料），文献给出的信息如下。

1. **两层 baseline 结构。** 和 BReF 一样，分开报两张表：
   - (a) **已有方法整条 pipeline、原生协议**：MET、ZeroPrint、LLMmap；
   - (b) **模块级对照**：D024 那一组、MET 统计量替换、random/top-k/MMR。

   我上一轮只建议 MET 做模块级。**BReF 和 AdaptPrint 的对比说明，审稿人期待看到原生协议的 MET**，否则模块级的结果容易被说成 strawman。所以 MET 可能两层都要：
   - 原生协议版需要新生成：25 条 prompt × 10 个样本，T=1；
   - 模块版用冻结数据。
2. **MET 原生协议放进我们的设置时，要按 AdaptPrint 的 Gen/All 思路处理参考库。**
   - 参考端分两种：只用单一配置，或覆盖 build 配置；
   - 目标端用我们 S_test 的配置。
   - 同时加一行"目标也在 MET 原生配置下"的对照，用来区分"方法弱"和"配置偏移"这两种原因。
3. **ZeroPrint 有官方代码，BReF 已经用固定 commit 复现过一次。** 它的协议是每个 checkpoint 200 次生成、T=0.7、≤512 新 token（我们的语料是 100 token）。成本可以按"模型数 × 200 × (1 + 目标配置数)"估。
4. **LLMmap 两种模式都要报。**
   - 我们现有的 paper8 结果是 closed-set 重训练；
   - 再加一行**官方预训练 open-set encoder**：别人引用 LLMmap 时都用这个模式，否则读者没法把我们的数字和文献里的数字对上。
5. **预算只做透明报告，不做名义对齐。** 我们的方法是 k=8，原生 MET 是 250 个样本，原生 ZeroPrint 是 200 次生成。要报一张预算表，必要时再加一行"预算对齐"的结果。不要学 ZeroPrint 那种"q=200 上限"的写法。
6. **统计方法我们已经领先**（预注册 + config×seed 层级 bootstrap）。对 Top-1 的配对比较可以补一个 McNemar + Holm，作为和文献同口径的报告。
7. **指标。** 我们的主指标是 37/85 选 1 的 top-1 和 pair 尾部 CVaR。补报 MRR 和 Top-3 很便宜，也能和 AdaptPrint、BReF 直接对照。
8. **口径要写清楚。** AdaptPrint 和 ZeroPrint 都把"认成基座模型"算成正确或阳性，我们把它算成错误。论文里必须单列一段说明任务口径的差异，否则这两个 baseline 在我们设置下的低分会被读成"故意用错"。

---

## 8. 未核实 / 注意事项

- ZeroPrint 附录 B 的 LeaFBench 构造细节（正负 pair 的数量）在 HTML 版里没找到。
- AdaptPrint 没有说明 baseline 是否用官方代码、LLMmap 是否重训练。
- LeaFBench、SEF 与 ZeroPrint 是同一作者组，这是从引用格式和 GitHub 用户名推断的，没有核实。
- BReF 只有 22 个 suspect，它的排名本身也有不小的不确定性（它自己报了区间）。
- 第 5 节对排名翻转的解释是假设，需要在我们自己的设置下测。

## 9. 测试集精度：这是大家共同的问题吗？（2026-10-09 补充）

我们记录过的问题是：每个模型只有 25 个 S_test 配置。由此 v1 有 925 条测试 trace，每个 pair 只有 50 条，per-pair 的最小步长是 2pp。强方法之间相差 1–2pp 时分辨不出来。

下面用同样的眼光看另外三篇。区间是我按二项分布粗算的，**没有考虑聚类，所以真实的区间只会更宽**。

| | 独立测试单元 | 测试规模 | 报了什么不确定性 | 粗算的精度 |
|---|---|---|---|---|
| ZeroPrint | 模型 pair（149 个实例，pair 数没报） | 不明 | ±std，没说是对什么取的 std | 无法评估；它对 TRAP 的领先（.720 vs .712）小于 std |
| AdaptPrint | 12 个目标模型 × 18 个 system prompt = **216** 条（Top-1 都是 1/216 的整数倍，可以验证） | 216 | **无** | Top-1 约 80% 时 ±5.3pp；它对 ZeroPrint-All 的 +8.8pp，差值大约 ±7pp。只有 12 个模型可以当聚类单元，有效样本更小 |
| BReF | **22** 个 suspect | 22 | suspect-cluster bootstrap、McNemar + Holm | Top-1 步长 4.5pp。MET 14/22 的 Wilson 区间约 [43%, 80%]，ZeroPrint 7/22 约 [16%, 53%]，两者略有重叠 |
| **我们** | 模型 × 配置，按配置 slot 做 bootstrap | v1 **925** / v2 **2,125** | 预注册 + config×seed 层级 bootstrap | Δ mean top-1 约 ±1.3–3.5pp（D024、D029）；单个 pair 约 ±12pp |

**结论**

- **精度不够是整个领域的共同问题**，不是我们独有的。我们的测试规模是四篇里最大的，也是统计报告最完整的之一。
- 别人"没有这个问题"，是因为**没有去测**：AdaptPrint 不报区间，ZeroPrint 只报 std。
- **这个问题只在比较两个接近的强方法时才要命。** 如果文献中的差距在我们的设置下也成立，D030 里我们和 MET 的差距大概率会大到不受精度限制。和 ZeroPrint 比、以及看单个 pair 时，精度仍会起作用。
- **对 D030 的直接影响：** 如果 ZeroPrint 只跑 5 个 test 配置，它的比较就只有 5 个 bootstrap 聚类单元，每个 pair 只有 10 条 trace，区间大约宽 2.2 倍。所以 D030 已经写明：选 T_ZP 时要把精度和成本并排列出来。

## Sources

- [ZeroPrint — arXiv 2510.06605v2](https://arxiv.org/html/2510.06605v2)
- [AdaptPrint — arXiv 2608.22213v1](https://arxiv.org/html/2608.22213v1)
- [BReF — arXiv 2609.06330v1](https://arxiv.org/html/2609.06330v1)
- [MET — arXiv 2410.20247v2](https://arxiv.org/html/2410.20247v2)
