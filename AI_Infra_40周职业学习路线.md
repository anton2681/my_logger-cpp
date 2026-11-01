# AI Infra 职业学习路线（40 周）

> **适合人群**：Python
> 能写简单程序；学过高等数学、线性代数和概率论但遗忘较多；愿意深入学习
> C++ 和底层系统编程。\
> **目标**：建立完整的 AI Infrastructure（AI
> Infra）知识体系，能够理解、开发、调试和优化 AI
> 系统，并完成可展示的工程项目。\
> **学习投入**：每天 3～4 小时，每周 6 天，计划 40 周，约 9～10 个月。

------------------------------------------------------------------------

## 目录

1.  [学习目标与路线总览](#1-学习目标与路线总览)
2.  [AI Infra 四大方向](#2-ai-infra-四大方向)
3.  [学习节奏与方法](#3-学习节奏与方法)
4.  [第 1 阶段：编程与计算机系统基础（第 1～6
    周）](#4-第-1-阶段编程与计算机系统基础第-1～6-周)
5.  [第 2 阶段：机器学习、深度学习与 GPT（第 7～12
    周）](#5-第-2-阶段机器学习深度学习与-gpt第-7～12-周)
6.  [第 3 阶段：GPU 编程与性能优化（第 13～19
    周）](#6-第-3-阶段gpu-编程与性能优化第-13～19-周)
7.  [第 4 阶段：分布式训练（第 20～24
    周）](#7-第-4-阶段分布式训练第-20～24-周)
8.  [第 5 阶段：LLM 推理系统（第 25～30
    周）](#8-第-5-阶段llm-推理系统第-25～30-周)
9.  [第 6 阶段：AI 平台与集群基础设施（第 31～34
    周）](#9-第-6-阶段ai-平台与集群基础设施第-31～34-周)
10. [第 7 阶段：方向探索与求职准备（第 35～40
    周）](#10-第-7-阶段方向探索与求职准备第-35～40-周)
11. [推荐 GitHub 开源项目](#11-推荐-github-开源项目)
12. [技术栈与掌握程度](#12-技术栈与掌握程度)
13. [简历项目建议](#13-简历项目建议)
14. [每日与每周学习安排](#14-每日与每周学习安排)
15. [第一个月行动清单](#15-第一个月行动清单)
16. [学习原则与注意事项](#16-学习原则与注意事项)

------------------------------------------------------------------------

# 1. 学习目标与路线总览

这份计划不是单纯的深度学习课程表，而是以 **AI Systems / AI
Infrastructure Engineer** 为长期目标的工程路线。

40 周结束时，争取达到以下阶段性成果：

-   能使用 Python 和 C++ 完成中小型工程项目。
-   理解 Linux、操作系统、内存、进程线程、并发和网络基础。
-   理解神经网络、PyTorch、Transformer 和 GPT 的基本运行过程。
-   能编写简单 CUDA / Triton GPU Kernel，并进行正确性测试和性能
    Benchmark。
-   理解 DDP、FSDP、NCCL 等分布式训练基础。
-   能使用 vLLM 部署模型，并对推理服务进行压力测试和性能分析。
-   能使用 Docker、Kubernetes 和监控工具部署及观察模型服务。
-   完成 3～5 个有代码、测试、文档和性能报告的 GitHub 项目。

**重要预期管理**：40
周可以建立扎实的入门基础、完成有说服力的项目并开始准备相关岗位，但不代表自动达到所有公司的招聘要求。实际求职竞争力还取决于项目深度、代码质量、系统知识、面试表现和持续实践。

## 40 周总览

  ------------------------------------------------------------------------------------------------------
  周次              阶段                   核心内容                                   阶段产出
  ----------------- ---------------------- ------------------------------------------ ------------------
  1～6              编程与计算机系统基础   Python 工程化、C++、Linux、OS、并发、网络  C++ Thread Pool

  7～12             AI 基础与 GPT          数学复习、机器学习、PyTorch、Transformer   Mini GPT

  13～19            GPU 编程与性能优化     GPU 架构、CUDA、Triton、Profiling          GPU Kernel
                                                                                      Optimization

  20～24            分布式训练             DDP、FSDP2、NCCL、并行策略                 Distributed GPT
                                                                                      Training

  25～30            LLM 推理系统           KV Cache、vLLM、Batching、量化、Benchmark  LLM Inference
                                                                                      Benchmark

  31～34            AI 平台与集群          Docker、Kubernetes、GPU 调度、监控         LLM Deployment
                                                                                      Platform

  35～40            方向探索与求职         四方向体验、源码阅读、作品集和面试         专精方向与作品集
  ------------------------------------------------------------------------------------------------------

------------------------------------------------------------------------

# 2. AI Infra 四大方向

先建立共同基础，再通过小项目体验不同方向，不必一开始就选定岗位。

## A. GPU Kernel / Performance Engineering

**工作重点**：优化 GPU 计算，让算子运行得更快、更省显存。

常见技术：

-   C++、CUDA、Triton
-   CUTLASS、FlashAttention
-   Nsight Systems、Nsight Compute
-   GPU Memory Hierarchy、Kernel Fusion、Profiling

典型工作：

-   实现和优化 GPU 算子。
-   优化矩阵乘法、Attention、Softmax 等计算。
-   分析显存访问、计算吞吐和 Kernel Launch 开销。
-   提高 GPU 利用率。

## B. LLM Training Infrastructure

**工作重点**：让大模型在多 GPU、多节点上高效、稳定地训练。

常见技术：

-   PyTorch Distributed
-   DDP、FSDP2
-   Tensor Parallel、Pipeline Parallel
-   NCCL、Megatron-LM、DeepSpeed
-   Checkpoint、通信优化、显存优化

典型工作：

-   多 GPU / 多机训练。
-   训练任务性能分析。
-   参数、梯度和优化器状态分片。
-   通信开销和训练稳定性优化。

## C. LLM Inference Infrastructure

**工作重点**：让模型服务以更低延迟、更高吞吐、更低成本运行。

常见技术：

-   vLLM、SGLang
-   KV Cache、PagedAttention
-   Continuous Batching
-   Quantization、Speculative Decoding
-   TTFT、TPOT、ITL、Throughput

典型工作：

-   推理引擎开发和优化。
-   请求调度与批处理。
-   KV Cache 管理。
-   延迟、吞吐和显存优化。

## D. AI Platform / Cluster Infrastructure

**工作重点**：构建、部署和管理 AI 服务及 GPU 集群。

常见技术：

-   Linux、C++、Go
-   Docker、Kubernetes
-   GPU Scheduling、Device Plugin
-   Prometheus、Grafana
-   Networking、Monitoring、Observability

典型工作：

-   GPU 资源调度。
-   模型服务部署。
-   集群监控和故障处理。
-   容器、网络和平台工程。

## 建议的探索策略

先完成共同基础，再体验四个方向：

-   第 35 周：GPU Kernel。
-   第 36 周：分布式训练。
-   第 37 周：推理系统。
-   第 38 周：AI 平台。

第 39 周根据兴趣和项目体验选择一个方向深入。

------------------------------------------------------------------------

# 3. 学习节奏与方法

## 每天 3～4 小时安排

  内容                       时间 做法
  --------------------- --------- -----------------------------
  课程视频 / 理论         45 分钟 看课程，记录核心概念
  官方文档 / 代码阅读     45 分钟 对照文档理解 API 和系统设计
  编程 / 项目开发         90 分钟 独立写代码、测试、调试
  复习 / 技术笔记         30 分钟 总结问题、记录中英文术语

合计约 3 小时 30 分钟。

## 每周安排

-   周一～周五：学习新知识并完成小练习。
-   周六：集中做项目、写测试、整理笔记和复盘。
-   周日：休息或补漏。

## 学习原则

1.  **代码优先**：看懂视频不等于掌握，能独立实现才算真正学会。
2.  **先理解再读大型源码**：先实现简化版本，再读 PyTorch、vLLM
    等复杂项目。
3.  **每周有可交付产物**：代码、测试、实验报告或技术笔记至少完成一项。
4.  **重视性能测量**：不要只说"更快"，要记录基线、测试条件和实际结果。
5.  **逐步阅读英文资料**：每天 15～20 分钟阅读官方文档、README、Issue 或
    PR。
6.  **不要急着购买 GPU**：前期可使用 CPU；需要 CUDA 时再租用云 GPU。

------------------------------------------------------------------------

# 4. 第 1 阶段：编程与计算机系统基础（第 1～6 周）

**阶段目标**：建立 C++、Linux 和计算机系统基础，为后续
GPU、分布式和推理系统学习打底。

## 推荐课程与资料

### C++

-   [LearnCpp](https://www.learncpp.com/)：系统学习现代 C++。
-   [The Cherno C++ YouTube
    播放列表](https://www.youtube.com/playlist?list=PLlrATfBNZ98dudnM48yfGUldqGD0S4FFb)：视频入门。
-   [B站搜索：The Cherno
    C++](https://search.bilibili.com/all?keyword=The%20Cherno%20C%2B%2B)
-   [CppCon Back to
    Basics](https://www.youtube.com/playlist?list=PLHTh1InhhwT7gQEuYznhhvAYTel0qzl72)：专题讲座。

### Linux、Shell、Git 和开发工具

-   [MIT Missing Semester](https://missing.csail.mit.edu/)
-   [B站搜索：MIT Missing
    Semester](https://search.bilibili.com/all?keyword=MIT%20Missing%20Semester)
-   [Pro Git 中文版](https://git-scm.com/book/zh/v2)

### 操作系统

-   [OSTEP：Operating Systems: Three Easy
    Pieces](https://pages.cs.wisc.edu/~remzi/OSTEP/)
-   [MIT 6.S081 操作系统课程](https://pdos.csail.mit.edu/6.S081/2020/)
-   [B站搜索：MIT
    6.S081](https://search.bilibili.com/all?keyword=MIT%206.S081)

### 网络基础

-   [Beej's Guide to Network Programming](https://beej.us/guide/bgnet/)
-   [B站搜索：Socket
    网络编程](https://search.bilibili.com/all?keyword=C%2B%2B%20Socket%20网络编程)

## 每周任务

### 第 1 周：Linux、Shell、Git、Python 工程化

学习内容：

-   Linux 文件系统、目录和权限。
-   Shell 常用命令、管道、重定向。
-   Git 基本工作流：clone、add、commit、branch、merge、rebase。
-   Python 虚拟环境、依赖管理、日志和 pytest。
-   VS Code、终端和调试工具。

任务：

-   [ ] 安装并熟悉 Linux 开发环境（原生 Linux 或 WSL2 均可）。
-   [ ] 熟悉
    `cd`、`ls`、`grep`、`find`、`ps`、`top`、`kill`、`chmod`、`ssh`
    等命令。
-   [ ] 创建 GitHub 仓库 `ai-infra-learning`。
-   [ ] 建立 Python 虚拟环境并运行 pytest。
-   [ ] 写一份开发环境配置说明。

### 第 2 周：C++ 基础

学习内容：

-   变量、类型、函数、控制流。
-   类和对象。
-   指针与引用。
-   Stack 和 Heap。
-   基础内存管理。
-   编译、链接和头文件。

任务：

-   [ ] 完成 5～10 个 C++ 小练习。
-   [ ] 用指针和引用实现简单数据结构。
-   [ ] 解释 Stack 和 Heap 的区别。
-   [ ] 使用编译器编译多文件程序。

### 第 3 周：现代 C++

学习内容：

-   STL：vector、string、map、unordered_map、queue。
-   RAII。
-   智能指针：unique_ptr、shared_ptr、weak_ptr。
-   Move Semantics。
-   Template。
-   CMake。

任务：

-   [ ] 使用 STL 完成数据结构练习。
-   [ ] 使用 RAII 管理资源。
-   [ ] 使用 CMake 编译一个多文件项目。
-   [ ] 阅读并理解简单的模板代码。

### 第 4 周：操作系统基础

学习内容：

-   进程和线程。
-   系统调用。
-   虚拟内存。
-   地址空间。
-   栈、堆和内存映射。
-   Linux 进程管理。

任务：

-   [ ] 使用 `ps`、`top`、`htop` 观察进程。
-   [ ] 使用 `pmap` 或 `/proc` 观察进程内存。
-   [ ] 编写创建进程或线程的小程序。
-   [ ] 理解用户态和内核态的基本区别。

### 第 5 周：并发编程

学习内容：

-   C++ std::thread。
-   Mutex。
-   Condition Variable。
-   Atomic。
-   Race Condition。
-   Deadlock。
-   Producer-Consumer。

任务：

-   [ ] 实现生产者---消费者队列。
-   [ ] 使用 mutex 保护共享数据。
-   [ ] 使用 atomic 实现简单计数器。
-   [ ] 构造并定位一次数据竞争或死锁问题。

### 第 6 周：网络与综合项目

学习内容：

-   TCP/IP 基础。
-   Socket。
-   HTTP 基础。
-   Client / Server 模型。
-   基础网络调试。

任务：

-   [ ] 使用 C++ 实现 TCP Echo Server。
-   [ ] 使用客户端连接并发送消息。
-   [ ] 处理客户端断开和错误。
-   [ ] 写明编译、运行和测试方法。

## 第 6 周毕业项目：C++ Thread Pool

项目仓库建议：`cpp-thread-pool`

``` text
cpp-thread-pool/
├── include/
│   └── thread_pool.h
├── src/
│   └── thread_pool.cpp
├── tests/
│   └── test_thread_pool.cpp
├── CMakeLists.txt
└── README.md
```

功能要求：

-   支持多个工作线程。
-   支持任务队列。
-   支持提交任务并获取结果。
-   支持线程同步。
-   支持优雅关闭。
-   使用 CMake 编译。
-   使用 GDB 调试。
-   测试并发情况下的正确性。

项目重点：并发、锁、队列、线程生命周期和资源管理。

------------------------------------------------------------------------

# 5. 第 2 阶段：机器学习、深度学习与 GPT（第 7～12 周）

**阶段目标**：理解模型计算什么、如何训练，以及 Transformer / GPT
的基本执行过程。

本阶段不是为了成为算法研究员，而是为了理解 AI Infra 所服务的模型计算。

## 推荐课程

-   [李宏毅 2021
    机器学习课程（B站搜索）](https://search.bilibili.com/all?keyword=李宏毅2021机器学习)
-   [动手学深度学习（中文版）](https://zh.d2l.ai/)
-   [Andrej Karpathy：Neural Networks Zero to
    Hero](https://karpathy.ai/zero-to-hero.html)
-   [PyTorch
    官方入门教程](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)
-   [PyTorch 60 Minute
    Blitz](https://docs.pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)

## 每周任务

### 第 7 周：数学复习与机器学习基础

学习内容：

-   向量、矩阵、矩阵乘法。
-   导数、偏导数、链式法则。
-   概率、期望、方差。
-   线性回归、逻辑回归。
-   损失函数、梯度下降。

任务：

-   [ ] 使用 NumPy 实现线性回归。
-   [ ] 实现 MSE Loss。
-   [ ] 实现梯度下降。
-   [ ] 观察学习率对收敛的影响。

### 第 8 周：神经网络基础

学习内容：

-   神经元、权重、偏置。
-   前向传播。
-   激活函数。
-   反向传播。
-   计算图。

任务：

-   [ ] 理解 MLP 的数据流。
-   [ ] 使用 PyTorch 实现 MNIST 分类。
-   [ ] 画出一个简单神经网络的前向传播流程。
-   [ ] 解释反向传播如何计算梯度。

### 第 9 周：PyTorch 核心机制

学习内容：

-   Tensor 和 Shape。
-   Autograd。
-   `requires_grad`、`grad`、`backward()`。
-   `nn.Module`、`nn.Linear`、`nn.Embedding`。
-   Optimizer 和训练循环。

任务：

-   [ ] 熟悉 Tensor 创建、索引和矩阵运算。
-   [ ] 手动验证一次自动求导结果。
-   [ ] 不看教程写出完整训练循环。
-   [ ] 记录训练和验证 Loss。

### 第 10 周：Tokenizer 与语言模型入门

学习内容：

-   Token、Token ID。
-   字符级 Tokenizer。
-   Bigram Language Model。
-   Next Token Prediction。
-   Cross Entropy。
-   自回归语言模型。

任务：

-   [ ] 实现字符级 Tokenizer。
-   [ ] 实现 Bigram 模型。
-   [ ] 训练一个小型字符级语言模型。
-   [ ] 观察模型如何预测下一个字符。

### 第 11 周：Attention、Transformer 与 GPT

学习内容：

-   Query、Key、Value。
-   Scaled Dot-Product Attention。
-   Causal Mask。
-   Multi-Head Attention。
-   Embedding、LayerNorm、Residual。
-   Decoder-only 架构。

任务：

-   [ ] 使用 PyTorch 矩阵运算实现 Self-Attention。
-   [ ] 实现 Causal Mask。
-   [ ] 实现 Transformer Block。
-   [ ] 画出 GPT 数据流图。

### 第 12 周：Mini GPT 与 DIY-LLM 入门

学习内容：

-   GPT 完整结构。
-   训练和生成。
-   Tokenizer 与模型接口。
-   DIY-LLM 项目目录和 Assignment 1。

任务：

-   [ ] 从零实现一个小型 GPT。
-   [ ] 使用小型文本数据训练。
-   [ ] 实现文本生成。
-   [ ] 阅读 DIY-LLM 的 Assignment 1。
-   [ ] 尝试完成 Tokenizer 和小型语言模型相关任务。

推荐资料：

-   [Karpathy：Let's build GPT from
    scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY)
-   [nanoGPT GitHub](https://github.com/karpathy/nanoGPT)
-   [Datawhale DIY-LLM
    GitHub](https://github.com/datawhalechina/diy-llm)
-   [DIY-LLM 文档](https://datawhalechina.github.io/diy-llm/)

**阶段完成标准**：能够解释 GPT 的基本数据流，理解
Attention、Embedding、Transformer Block 和 Next Token
Prediction，并能训练一个小型 GPT。

------------------------------------------------------------------------

# 6. 第 3 阶段：GPU 编程与性能优化（第 13～19 周）

**阶段目标**：从"会使用 PyTorch"过渡到"理解 GPU
如何执行计算，并能实现和优化简单算子"。

## 推荐课程与资料

-   [CUDA C++ Programming
    Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
-   [GPU MODE Lectures GitHub](https://github.com/gpu-mode/lectures)
-   [GPU MODE YouTube
    搜索](https://www.youtube.com/results?search_query=GPU+MODE+CUDA+Triton+lectures)
-   [Triton
    官方教程](https://triton-lang.org/main/getting-started/tutorials/)
-   [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems)
-   [NVIDIA Nsight
    Compute](https://developer.nvidia.com/nsight-compute/)

## GPU 必须理解的概念

  概念                           学习重点
  ------------------------------ --------------------------
  GPU Architecture               GPU 整体结构
  SM                             Streaming Multiprocessor
  Warp                           GPU 线程执行组织
  Thread / Block / Grid          CUDA 线程组织
  Register                       寄存器
  Shared Memory                  共享内存
  Global Memory                  全局显存
  Memory Bandwidth               显存带宽
  Occupancy                      GPU 占用率
  Memory Coalescing              合并内存访问
  Compute Bound / Memory Bound   计算瓶颈和访存瓶颈
  Roofline Model                 性能上界分析
  Kernel Fusion                  算子融合

## 每周任务

### 第 13 周：GPU 架构

-   [ ] 理解 SM、Warp、CUDA Core 的基本概念。
-   [ ] 理解 Thread、Block、Grid。
-   [ ] 理解 Register、Shared Memory、Global Memory。
-   [ ] 画出 GPU 内存层级图。
-   [ ] 理解 Memory Bandwidth 和 Occupancy。

### 第 14 周：CUDA 入门

-   [ ] 配置 CUDA 开发环境。
-   [ ] 编写第一个 CUDA Kernel。
-   [ ] 实现 Vector Addition。
-   [ ] 理解 Kernel Launch。
-   [ ] 使用 CUDA Event 测量执行时间。

### 第 15 周：内存和并行优化

-   [ ] 理解 Shared Memory。
-   [ ] 理解 Memory Coalescing。
-   [ ] 实现 Reduction。
-   [ ] 比较不同 Block Size。
-   [ ] 观察内存访问对性能的影响。

### 第 16 周：矩阵乘法

-   [ ] 理解矩阵乘法的计算过程。
-   [ ] 实现基础 CUDA Matmul。
-   [ ] 学习 Tiling。
-   [ ] 尝试使用 Shared Memory。
-   [ ] 与 PyTorch Matmul 比较性能和数值结果。

### 第 17 周：Profiling

-   [ ] 学习 Nsight Systems。
-   [ ] 学习 Nsight Compute。
-   [ ] 使用 PyTorch Profiler。
-   [ ] 分析 CPU / GPU 时间线。
-   [ ] 找出一个可复现的性能瓶颈。

### 第 18 周：Triton

-   [ ] 学习 Triton 编程模型。
-   [ ] 实现 Vector Addition。
-   [ ] 实现 Softmax。
-   [ ] 实现 Matrix Multiplication。
-   [ ] 实现 LayerNorm。
-   [ ] 与 PyTorch 结果比较。

### 第 19 周：Attention 与 Kernel Fusion

-   [ ] 理解 Attention 的计算和显存开销。
-   [ ] 学习 FlashAttention 的基本思想。
-   [ ] 研究 Kernel Fusion。
-   [ ] 尝试优化 Attention 或相关算子。
-   [ ] 阅读 DIY-LLM Assignment 2 的性能分析和 Triton 任务。

## 第 19 周毕业项目：GPU Kernel Optimization

项目仓库建议：`gpu-kernel-optimization`

``` text
gpu-kernel-optimization/
├── cuda/
│   ├── vector_add.cu
│   └── reduction.cu
├── triton/
│   ├── softmax.py
│   ├── matmul.py
│   └── layernorm.py
├── benchmarks/
│   └── benchmark.py
├── tests/
│   └── test_ops.py
└── README.md
```

要求：

-   实现 Vector Addition、Reduction、Softmax、Matmul、LayerNorm。
-   与 PyTorch 结果进行数值比较。
-   测试不同 Tensor Shape。
-   记录执行时间。
-   分析显存访问和计算瓶颈。
-   对比优化前后的性能。

**报告必须记录真实实验结果**，包括 GPU
型号、软件版本、输入规模、测试方法和执行时间。不要只写"性能提升很多"。

------------------------------------------------------------------------

# 7. 第 4 阶段：分布式训练（第 20～24 周）

**阶段目标**：理解多个 GPU
如何协同训练模型，以及通信和显存如何影响性能。

## 核心知识

  技术                核心思想
  ------------------- ----------------------------
  Data Parallelism    不同 GPU 处理不同数据
  DDP                 数据并行训练
  FSDP2               分片参数、梯度和优化器状态
  Tensor Parallel     将模型计算拆到多 GPU
  Pipeline Parallel   将模型层分配到不同 GPU
  NCCL                GPU 集合通信
  AllReduce           跨 GPU 聚合数据
  AllGather           收集分散数据
  ReduceScatter       聚合并分发数据
  Checkpoint          保存和恢复训练状态

## 推荐课程与资料

-   [PyTorch Distributed
    Tutorials](https://docs.pytorch.org/tutorials/distributed.html)
-   [NVIDIA NCCL
    文档](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)
-   [Megatron-LM GitHub](https://github.com/NVIDIA/Megatron-LM)
-   [DeepSpeed GitHub](https://github.com/deepspeedai/DeepSpeed)

## 每周任务

### 第 20 周：分布式计算基础

-   [ ] 理解进程间通信。
-   [ ] 理解 Rank、World Size。
-   [ ] 理解 Process Group。
-   [ ] 理解 AllReduce、AllGather、ReduceScatter。
-   [ ] 使用 PyTorch Distributed 启动多个进程。

### 第 21 周：DDP

-   [ ] 将单 GPU GPT 改成 DDP。
-   [ ] 理解梯度同步。
-   [ ] 理解 DistributedSampler。
-   [ ] 比较单 GPU 和多 GPU 训练吞吐量。
-   [ ] 分析通信开销。

### 第 22 周：FSDP2

-   [ ] 理解参数、梯度和优化器状态分片。
-   [ ] 学习 FSDP2 基本 API。
-   [ ] 尝试使用 FSDP2 训练 Transformer。
-   [ ] 比较 DDP 与 FSDP2 的显存占用和训练速度。

### 第 23 周：模型并行

-   [ ] 理解 Tensor Parallel。
-   [ ] 理解 Pipeline Parallel。
-   [ ] 阅读相关示例代码。
-   [ ] 理解模型切分与通信之间的关系。
-   [ ] 了解 NCCL 通信拓扑和带宽影响。

### 第 24 周：分布式训练优化

-   [ ] 实现训练 Checkpoint 保存和恢复。
-   [ ] 比较不同 GPU 数量下的训练吞吐量。
-   [ ] 记录 GPU 显存和利用率。
-   [ ] 分析扩展效率。
-   [ ] 写一份分布式训练性能报告。

## 第 24 周毕业项目：Distributed GPT Training

项目仓库建议：`distributed-gpt-training`

要求：

-   基于 PyTorch 实现 GPT 训练。
-   支持单 GPU 和 DDP。
-   尝试 FSDP2。
-   实现 Checkpoint。
-   记录训练时间、Tokens/sec 和显存。
-   比较不同并行策略的性能。

需要回答：

-   为什么 GPU 数量增加后，训练速度不一定线性增加？
-   AllReduce 为什么会产生通信开销？
-   FSDP2 为什么可以降低显存占用？
-   网络带宽如何影响多机训练？

**资源说明**：没有多 GPU 时，可以先用 CPU / Gloo
验证分布式通信逻辑；真正的 GPU 扩展效率测试需要多张 GPU。

------------------------------------------------------------------------

# 8. 第 5 阶段：LLM 推理系统（第 25～30 周）

**阶段目标**：理解大模型如何被部署成服务，以及如何优化延迟、吞吐和显存。

## 核心知识

  概念                   作用
  ---------------------- --------------------------------
  Prefill                处理输入 Prompt
  Decode                 逐个生成 Token
  KV Cache               缓存 Attention 的 Key 和 Value
  PagedAttention         更高效地管理 KV Cache
  Continuous Batching    动态组织推理请求
  Quantization           降低模型权重存储成本
  Speculative Decoding   利用草稿预测等方式加速生成
  Prefix Caching         复用相同前缀的计算结果

## 推荐课程与资料

-   [vLLM 官方文档](https://docs.vllm.ai/)
-   [vLLM GitHub](https://github.com/vllm-project/vllm)
-   [B站搜索：vLLM
    推理框架](https://search.bilibili.com/all?keyword=vLLM%20推理框架)
-   [SGLang GitHub](https://github.com/sgl-project/sglang)

## 关键性能指标

  指标             含义
  ---------------- --------------------------------------------------
  TTFT             Time To First Token，首 Token 延迟
  ITL              Inter-Token Latency，Token 间隔
  TPOT             Time Per Output Token，平均每个输出 Token 的耗时
  Throughput       单位时间生成的 Token 数量
  P95 Latency      95 分位请求延迟
  GPU Memory       GPU 显存占用
  KV Cache Usage   KV Cache 使用情况

## 每周任务

### 第 25 周：推理原理

-   [ ] 理解 Prefill 和 Decode。
-   [ ] 理解 KV Cache 的作用。
-   [ ] 自己实现一个简化版 KV Cache。
-   [ ] 比较有无 KV Cache 时的重复计算。
-   [ ] 理解输入长度和输出长度对推理的影响。

### 第 26 周：vLLM 入门

-   [ ] 配置推理环境。
-   [ ] 使用 vLLM 部署一个开源模型。
-   [ ] 通过 API 发送请求。
-   [ ] 理解模型加载和服务启动流程。
-   [ ] 阅读 vLLM 的高层架构文档。

### 第 27 周：调度与 KV Cache

-   [ ] 理解 PagedAttention 的基本思想。
-   [ ] 理解 Continuous Batching。
-   [ ] 研究并发请求调度。
-   [ ] 观察不同并发量下的显存变化。
-   [ ] 了解 Prefix Caching。

### 第 28 周：推理优化

-   [ ] 了解 Quantization。
-   [ ] 了解 Speculative Decoding。
-   [ ] 比较不同模型精度或量化配置。
-   [ ] 观察质量、显存和吞吐之间的权衡。
-   [ ] 记录实验结果。

### 第 29 周：服务工程

-   [ ] 使用 API 提供模型服务。
-   [ ] 实现流式响应。
-   [ ] 学习异步请求处理。
-   [ ] 使用 Docker 部署。
-   [ ] 处理超时、错误和健康检查。

### 第 30 周：Benchmark 与性能分析

-   [ ] 使用 `vllm bench serve` 或其他工具压测。
-   [ ] 测试不同并发数。
-   [ ] 测试不同 Prompt 长度。
-   [ ] 测试不同生成长度。
-   [ ] 记录 TTFT、ITL、TPOT、Throughput 和 P95。
-   [ ] 写性能分析报告。

## 第 30 周毕业项目：LLM Inference Benchmark

项目仓库建议：`llm-inference-benchmark`

要求：

-   使用 vLLM 部署开源模型。
-   提供 API 服务。
-   支持并发请求测试。
-   测试不同 Prompt 和生成长度。
-   记录 TTFT、ITL、TPOT、Throughput、P95 Latency。
-   记录 GPU 显存和 KV Cache 使用情况。
-   写明硬件、模型、软件版本和测试条件。

------------------------------------------------------------------------

# 9. 第 6 阶段：AI 平台与集群基础设施（第 31～34 周）

**阶段目标**：理解模型服务如何在容器和集群环境中部署、调度与监控。

## 推荐课程与资料

-   [Docker 官方入门](https://docs.docker.com/get-started/)
-   [B站搜索：Docker
    入门](https://search.bilibili.com/all?keyword=Docker%20入门教程)
-   [Kubernetes 官方教程](https://kubernetes.io/docs/tutorials/)
-   [B站搜索：Kubernetes
    入门](https://search.bilibili.com/all?keyword=Kubernetes%20入门教程)
-   [Kubernetes GPU
    Scheduling](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/)
-   [NVIDIA GPU
    Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/)
-   [Prometheus 文档](https://prometheus.io/docs/introduction/overview/)
-   [Grafana 文档](https://grafana.com/docs/grafana/latest/)

## 每周任务

### 第 31 周：Docker

-   [ ] 学习 Image、Container、Volume、Network。
-   [ ] 编写 Dockerfile。
-   [ ] 将 vLLM 或简化模型服务容器化。
-   [ ] 使用 Docker Compose 管理服务。
-   [ ] 理解 GPU 容器运行时的基本作用。

### 第 32 周：Kubernetes

-   [ ] 学习 Pod、Deployment、Service。
-   [ ] 学习 ConfigMap 和 Secret。
-   [ ] 使用本地集群部署简单服务。
-   [ ] 理解控制平面和 Worker Node。
-   [ ] 了解 Kubernetes Scheduler。

### 第 33 周：GPU 资源调度

-   [ ] 理解 GPU Device Plugin。
-   [ ] 学习 Pod 的 GPU 资源申请。
-   [ ] 理解 GPU Operator 的作用。
-   [ ] 了解 GPU 节点的资源管理。
-   [ ] 观察容器内 GPU 可见性和资源状态。

### 第 34 周：监控和可观测性

-   [ ] 学习 Prometheus 基本概念。
-   [ ] 学习 Grafana Dashboard。
-   [ ] 采集模型服务指标。
-   [ ] 监控 GPU 利用率和显存。
-   [ ] 监控请求延迟和吞吐。
-   [ ] 配置简单告警。

## 第 34 周毕业项目：LLM Deployment Platform

建议架构：

``` text
             User
               |
               v
          API Service
               |
               v
          Kubernetes
               |
               v
         Model Service
               |
               v
             vLLM
               |
               v
              GPU

Monitoring:
Prometheus + Grafana
```

项目要求：

-   使用 Docker 构建服务镜像。
-   使用 Kubernetes 部署模型服务。
-   配置 GPU 资源申请。
-   配置 Service。
-   使用 Prometheus 采集指标。
-   使用 Grafana 展示 GPU 和服务指标。
-   写清楚部署、监控和故障排查步骤。

先完成单机或本地集群实践，不要求一开始搭建大规模 GPU 集群。

------------------------------------------------------------------------

# 10. 第 7 阶段：方向探索与求职准备（第 35～40 周）

## 第 35～38 周：四方向轮岗体验

### 第 35 周：GPU Kernel

-   [ ] 使用 Triton 实现 Softmax。
-   [ ] 与 PyTorch 实现进行正确性和性能比较。
-   [ ] 使用 Profiler 分析性能。
-   [ ] 观察内存访问和计算开销。

参考：[Triton
Tutorials](https://triton-lang.org/main/getting-started/tutorials/)

### 第 36 周：分布式训练

-   [ ] 使用 DDP 训练 GPT。
-   [ ] 测量训练吞吐量。
-   [ ] 观察 AllReduce。
-   [ ] 分析通信开销。

参考：[PyTorch
Distributed](https://docs.pytorch.org/tutorials/distributed.html)

### 第 37 周：推理系统

-   [ ] 使用 vLLM 部署模型。
-   [ ] 测试不同并发量。
-   [ ] 研究 KV Cache。
-   [ ] 分析推理吞吐量和延迟。

参考：[vLLM](https://docs.vllm.ai/)

### 第 38 周：AI 平台

-   [ ] 使用 Docker 部署模型。
-   [ ] 使用 Kubernetes 管理服务。
-   [ ] 配置 Prometheus。
-   [ ] 使用 Grafana 展示监控数据。

参考：[Kubernetes](https://kubernetes.io/docs/tutorials/)

## 第 39 周：选择专精方向

  兴趣和优势                           推荐方向
  ------------------------------------ -----------------
  喜欢研究内存、指针、矩阵计算和性能   GPU Kernel
  喜欢研究通信、并发和多 GPU           Training Infra
  喜欢研究调度、延迟和吞吐量           Inference Infra
  喜欢研究 Linux、网络和系统架构       AI Platform

选择后，重点阅读该方向的开源项目源码，寻找一个能够复现、分析或解决的问题。

## 第 40 周：作品集和求职准备

-   [ ] 整理 GitHub 项目。
-   [ ] 补充 README 和架构图。
-   [ ] 补充可复现的运行步骤。
-   [ ] 整理 Benchmark 数据和测试环境。
-   [ ] 写性能分析报告。
-   [ ] 总结遇到的问题和优化过程。
-   [ ] 准备 C++、操作系统、网络、GPU 和分布式系统面试题。
-   [ ] 选择目标岗位并对照招聘要求查漏补缺。

------------------------------------------------------------------------

# 11. 推荐 GitHub 开源项目

不要同时阅读所有项目。先理解小项目，再逐步阅读工业级代码。

  ------------------------------------------------------------------------------------------------
  项目                    GitHub                                      重点
  ----------------------- ------------------------------------------- ----------------------------
  PyTorch                 https://github.com/pytorch/pytorch          Tensor、Autograd、CUDA
                                                                      Backend、Distributed

  nanoGPT                 https://github.com/karpathy/nanoGPT         GPT 结构、训练循环

  Triton                  https://github.com/triton-lang/triton       GPU Kernel、编译器、算子优化

  vLLM                    https://github.com/vllm-project/vllm        推理引擎、KV Cache、调度

  SGLang                  https://github.com/sgl-project/sglang       推理运行时、请求调度

  Megatron-LM             https://github.com/NVIDIA/Megatron-LM       Tensor Parallel、Pipeline
                                                                      Parallel

  DeepSpeed               https://github.com/deepspeedai/DeepSpeed    ZeRO、显存优化、分布式训练

  DIY-LLM                 https://github.com/datawhalechina/diy-llm   LLM 训练、GPU
                                                                      优化、推理和分布式
  ------------------------------------------------------------------------------------------------

建议阅读顺序：

1.  nanoGPT。
2.  PyTorch 的相关教程和局部源码。
3.  Triton 教程和示例。
4.  DIY-LLM。
5.  vLLM。
6.  Megatron-LM / DeepSpeed。
7.  SGLang。

源码阅读方法：

-   先运行最小示例。
-   画出模块调用关系。
-   选择一个核心功能追踪代码。
-   加日志或断点观察运行过程。
-   写自己的简化实现。
-   阅读相关 Issue 和 PR。

------------------------------------------------------------------------

# 12. 技术栈与掌握程度

这不是要求在 40 周内把所有技术学到专家水平，而是一个优先级清单。

## 编程语言与工具

  技术       目标
  ---------- --------------------------
  Python     熟练，能独立完成工程开发
  C++17      熟练，掌握内存和并发
  CUDA C++   能编写基础 GPU Kernel
  Triton     能实现常见算子
  Go         了解；平台方向再深入
  Git        熟练
  CMake      熟练
  GDB        会调试
  Linux      熟练

## AI 与 GPU

  技术             目标
  ---------------- --------------------
  PyTorch          熟练
  Transformer      理解底层结构
  CUDA             掌握基础编程模型
  Triton           能写 GPU Kernel
  Nsight Systems   能分析系统时间线
  Nsight Compute   能分析 Kernel 性能
  NCCL             理解集合通信
  FlashAttention   理解原理和优化方法

## 分布式与推理

  技术                目标
  ------------------- --------------------
  DDP                 熟练使用
  FSDP2               掌握基本原理和实践
  Tensor Parallel     理解并能实践
  Pipeline Parallel   理解基本原理
  vLLM                能部署和分析
  SGLang              了解架构
  Megatron-LM         了解分布式训练实现
  DeepSpeed           了解 ZeRO

## 平台与工程系统

  技术         目标
  ------------ ------------------
  Docker       熟练
  Kubernetes   能部署和管理服务
  Prometheus   会配置监控
  Grafana      能制作监控面板
  TCP/IP       理解
  HTTP / RPC   理解
  Redis        了解
  PostgreSQL   了解
  CI/CD        掌握基础

**核心优先级**：

1.  C++ / Python。
2.  Linux / 操作系统。
3.  PyTorch / Transformer。
4.  CUDA / Triton / Profiling。
5.  分布式训练。
6.  LLM 推理。
7.  Docker / Kubernetes。

Go、Redis、数据库等根据后续职业方向再深入。

------------------------------------------------------------------------

# 13. 简历项目建议

简历不能只写"学习过 PyTorch""了解 CUDA""使用过
vLLM"。要用项目证明工程能力。

建议重点打造三个项目。

## 项目 A：GPU Kernel Optimization

**目标**：证明 GPU 底层计算和性能分析能力。

项目内容：

-   使用 CUDA 实现基础算子。
-   使用 Triton 实现 Softmax、Matmul、LayerNorm。
-   使用 PyTorch 作为正确性基准。
-   使用 Nsight 分析性能瓶颈。
-   对比不同 Tensor Shape 下的执行性能。

简历描述示例：

> 基于 CUDA 和 Triton 实现多个 GPU 计算算子，完成数值正确性验证及性能
> Benchmark，使用 Nsight 分析 Kernel
> 执行和显存访问瓶颈，并对计算过程进行优化。

需要补充真实实验数字：

-   相对于 PyTorch 基线的执行时间。
-   不同输入规模的性能变化。
-   显存带宽利用率。
-   Kernel Launch 开销。

## 项目 B：Distributed GPT Training

**目标**：证明分布式训练和系统分析能力。

项目内容：

-   从零实现 GPT。
-   使用 DDP 训练。
-   使用 FSDP2 进行参数分片。
-   研究 NCCL 集合通信。
-   实现 Checkpoint。
-   测试多 GPU 训练效率。

简历描述示例：

> 基于 PyTorch Distributed 构建 GPT 分布式训练系统，实现 DDP 和 FSDP2
> 训练流程，分析多 GPU
> 训练过程中的通信开销、显存使用和训练吞吐量，并完成不同并行策略的性能对比。

## 项目 C：LLM Inference Serving Platform

**目标**：证明推理系统和工程部署能力。

项目内容：

-   使用 vLLM 部署模型。
-   实现 OpenAI 兼容 API。
-   使用 Docker 容器化。
-   使用并发请求进行压力测试。
-   记录 TTFT、TPOT、Throughput、P95。
-   使用 Prometheus 和 Grafana 监控。

简历描述示例：

> 基于 vLLM 构建 LLM
> 推理服务，完成容器化部署、并发请求压测和性能监控，分析不同请求负载下的首
> Token 延迟、生成吞吐量、显存占用及服务性能瓶颈。

## 每个项目应具备的材料

-   GitHub 源代码。
-   清晰的 README。
-   系统架构图。
-   可复现的运行步骤。
-   单元测试。
-   Benchmark 脚本。
-   性能对比报告。
-   技术问题和优化过程。

如果能为 PyTorch、Triton 或 vLLM 提交有价值的
PR，或复现并修复一个真实性能问题，会更有说服力。

------------------------------------------------------------------------

# 14. 每日与每周学习安排

## 每日安排

  时间      内容
  --------- ---------------------
  45 分钟   课程视频 / 理论
  45 分钟   官方文档 / 代码阅读
  90 分钟   编程 / 项目开发
  30 分钟   复习 / 技术笔记

## 每周安排

  日期         任务
  ------------ -----------------------------------
  周一～周五   学习新内容、完成小练习
  周六         项目开发、测试、复盘、整理 GitHub
  周日         休息或补漏

## 建议的 GitHub 总仓库结构

``` text
ai-infra-learning/
├── 01-cpp/
├── 02-linux-os/
├── 03-networking/
├── 04-machine-learning/
├── 05-pytorch/
├── 06-transformer/
├── 07-cuda/
├── 08-triton/
├── 09-distributed-training/
├── 10-llm-inference/
├── 11-docker-k8s/
├── projects/
│   ├── cpp-thread-pool/
│   ├── mini-gpt/
│   ├── gpu-kernel-optimization/
│   ├── distributed-gpt-training/
│   └── llm-inference-benchmark/
└── notes/
```

------------------------------------------------------------------------

# 15. 第一个月行动清单

不要同时开始 C++、CUDA、PyTorch 和 Kubernetes。第一个月先完成 Linux、C++
和操作系统入门。

## 第 1 周：Linux + Git + Python 工程化

-   [ ] 安装并熟悉 Linux 开发环境。
-   [ ] 学习 Shell 常用命令。
-   [ ] 掌握 Git 基本工作流。
-   [ ] 创建 `ai-infra-learning` 仓库。

## 第 2 周：C++ 基础

-   [ ] 学习变量、函数和类。
-   [ ] 掌握指针和引用。
-   [ ] 理解 Stack 和 Heap。
-   [ ] 完成 5～10 个 C++ 小练习。

## 第 3 周：现代 C++

-   [ ] 学习 STL 容器。
-   [ ] 掌握 RAII 和智能指针。
-   [ ] 学习 CMake。
-   [ ] 编写一个多文件 C++ 项目。

## 第 4 周：操作系统入门

-   [ ] 理解进程和线程。
-   [ ] 理解虚拟内存。
-   [ ] 学习 Linux 进程管理命令。
-   [ ] 使用 GDB 调试 C++ 程序。

第一个月结束时，应当有一个 GitHub 仓库、一些可运行的 C++ 练习，以及对
Linux 和操作系统基本概念的初步理解。

------------------------------------------------------------------------

# 16. 学习原则与注意事项

## 1. 不要急着购买 GPU

-   前期数学、C++、Linux、操作系统和 PyTorch 基础学习可以使用 CPU。
-   需要 CUDA 和 GPU Kernel 实践时，可以使用云 GPU。
-   多 GPU 分布式实验需要多 GPU 环境，可先用 CPU / Gloo 理解通信逻辑。
-   先确认自己的学习方向和实际资源需求，再考虑购买硬件。

可选云端环境：

-   [Google Colab](https://colab.research.google.com/)
-   [Kaggle Notebooks](https://www.kaggle.com/code)
-   [RunPod](https://www.runpod.io/)

## 2. 不要只把 AI Infra 理解成部署模型

会 Docker、Kubernetes、调用 vLLM，只代表掌握了一部分部署能力。

更有竞争力的工程能力包括：

-   分析性能瓶颈。
-   理解 GPU 计算、内存和通信。
-   理解模型训练和推理执行过程。
-   通过代码和系统设计解决性能问题。
-   进行可复现的性能测试。

## 3. 不要同时读太多大型源码

建议先实现简化版本，再阅读大型项目。阅读源码时，先从入口、数据流和一个核心模块开始，不要试图一次看完整个仓库。

## 4. 坚持阅读英文技术资料

每天 15～20 分钟：

-   阅读官方文档。
-   阅读 GitHub README。
-   阅读 Issue 和 PR。
-   整理中英文专业术语。

不要求一开始流利交流，但需要逐步具备独立阅读英文技术资料的能力。

## 5. 40 周后的下一步

完成基础路线后，根据兴趣选择一个方向深入：

-   GPU Kernel：深入
    CUDA、Triton、CUTLASS、FlashAttention、编译器和性能分析。
-   Training Infra：深入 FSDP、Megatron-LM、NCCL、多机网络和训练容错。
-   Inference Infra：深入 vLLM、SGLang、KV Cache、调度、量化和推理优化。
-   AI Platform：深入 Kubernetes、Go、GPU 调度、资源管理和可观测性。

长期目标是理解 AI 系统从模型计算、GPU
执行、分布式通信到在线服务的完整链路。

------------------------------------------------------------------------

## 核心资源速查表

  -----------------------------------------------------------------------------------------------------------------------------------------
  领域                                主资源
  ----------------------------------- -----------------------------------------------------------------------------------------------------
  C++                                 [LearnCpp](https://www.learncpp.com/)、[The
                                      Cherno](https://www.youtube.com/playlist?list=PLlrATfBNZ98dudnM48yfGUldqGD0S4FFb)

  Linux / 工具                        [MIT Missing Semester](https://missing.csail.mit.edu/)

  操作系统                            [OSTEP](https://pages.cs.wisc.edu/~remzi/OSTEP/)

  机器学习                            [李宏毅课程](https://search.bilibili.com/all?keyword=李宏毅2021机器学习)

  深度学习                            [动手学深度学习](https://zh.d2l.ai/)

  神经网络 / GPT                      [Karpathy Zero to Hero](https://karpathy.ai/zero-to-hero.html)

  PyTorch                             [官方教程](https://docs.pytorch.org/tutorials/)

  GPT 实现                            [nanoGPT](https://github.com/karpathy/nanoGPT)

  LLM 课程                            [DIY-LLM](https://github.com/datawhalechina/diy-llm)

  CUDA                                [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)

  GPU 性能                            [GPU MODE Lectures](https://github.com/gpu-mode/lectures)

  Triton                              [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/)

  分布式训练                          [PyTorch Distributed](https://docs.pytorch.org/tutorials/distributed.html)

  推理系统                            [vLLM](https://docs.vllm.ai/)

  集群部署                            [Docker](https://docs.docker.com/get-started/)、[Kubernetes](https://kubernetes.io/docs/tutorials/)
  -----------------------------------------------------------------------------------------------------------------------------------------

**最终建议**：先完成第一个月，不要因为后面还有 CUDA、分布式训练和 vLLM
就产生压力。每周持续产出代码和实验记录，逐步建立系统工程能力。40
周是一条学习路线，不是必须机械遵守的截止日期。
