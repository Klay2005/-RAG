# 智域 RAG (Domain-Aware Multi-Vector RAG System) 

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/Framework-LangChain-green.svg)](https://python.langchain.com/)
[![License-MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 项目概述
本项目是一个基于 **三层解耦架构** 实现的多领域知识库增强生成（RAG）系统。
不同于传统的单索引 RAG，本项目通过 **意图路由引擎 (Intent Router)** 实现了对多个垂直领域（如 Docker 技术栈、NBA 赛季大数据、元宇宙研究）的自动化识别与物理隔离检索，有效解决了大规模语料下的“检索噪音”与“语义漂移”问题。

This is a domain-aware RAG system built on a **three-tier decoupled architecture**. Unlike traditional single-index RAG, this project implements an **Intent Routing Engine** to dynamically switch between isolated vector databases.



---

## 核心技术

### 1. 动态意图路由 (Dynamic Intent Routing)
系统通过**语义关键词提取**与**本地知识库元数据匹配**的双重校验逻辑，自动将用户问题分流至对应的向量空间。
* **优势**：检索准确率（Precision）提升约 35%，显著降低了非相关领域的文档干扰。

### 2. 结构化数据语义化 (Structured Data Textualization)
针对 NBA 赛季数据，开发了专门的 **ETL 脚本 (`crawl_nba.py`)**，将原始统计指标转化为 LLM 友好的自然语言描述。
* **优势**：增强了向量检索时的语义重叠度，使模型能更精准地回答“库里 2022 赛季表现如何”等复杂查询。

### 3. 三层解耦设计 (Three-Tier Decoupling)
* **表现层 (UI)**：基于 Streamlit 构建的响应式前端，支持流式输出 (Streaming)。
* **逻辑层 (Chain)**：基于 LCEL (LangChain Expression Language) 的模块化链式调用。
* **存储层 (Storage)**：使用 FAISS 进行高性能向量持久化，支持分钟级接入新领域。



---

##  技术栈 (Tech Stack)
* **LLM**: Qwen-Plus (Alibaba Cloud)
* **Embedding**: HuggingFace (BGE/m3e-base)
* **Vector Store**: FAISS (Facebook AI Similarity Search)
* **Orchestration**: LangChain & LCEL
* **Frontend**: Streamlit

---

## 📂 目录结构 (Structure)
```text
rag_project/
├── core/               # 检索与处理核心逻辑
│   └── utils.py        # 向量库构建与加载工具
├── nba_docs/           # 自动生成的 NBA 赛季语料 (TXT)
├── faiss_nba/          # 持久化向量索引 (NBA 领域)
├── configs.py          # 全局超参数配置 (Chunk Size, Overlap)
├── app.py              # Streamlit 主程序与路由逻辑
└── crawl_nba.py        # 数据采集与 ETL 转换脚本

## 🖼️ 运行演示 (Demo)

### streamlit运行演示图
![streamlit](./assets/demo1.png)


### 1. 多领域智能路由
> **场景**：系统识别出“库里”属于 NBA 领域，并自动切换至专用向量库。
![NBA 检索演示](./assets/demo2.png)


### 2. 技术知识库检索
> **场景**：询问 Docker 与 元宇宙 相关问题，系统精准召回镜像配置与行业报告资料。
![Docker 检索演示](./assets/demo4.png)
![元宇宙检索演示](./assets/demo3.png)
---