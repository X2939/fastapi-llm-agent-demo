# FastAPI LLM Tool Calling Agent Demo

一个基于 FastAPI + OpenAI-compatible API + SQLite 的最小可用 Tool Calling Agent Demo。

本项目不使用 LangChain 等复杂框架，而是手写最小 Agent 流程，重点展示：

- FastAPI 后端接口开发
- 本地 vLLM / OpenAI-compatible API 调用
- LLM 工具选择
- Python 工具函数执行
- SQLite 查询
- SSE 流式返回 Agent 执行过程
- 基础异常处理

## 1. 项目背景

普通聊天机器人只能生成文本，不能真正执行动作。

Agent 的核心区别是：它可以根据用户任务选择工具，并通过后端代码执行真实动作，例如计算、读取文件、查询数据库。

本项目实现了一个最小 Agent 流程：

```text
用户输入
-> LLM 判断应该调用哪个工具
-> LLM 生成工具调用参数
-> Python 后端执行工具函数
-> LLM 根据工具结果生成最终回答
-> FastAPI 返回结构化响应
