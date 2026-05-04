# FastAPI LLM Tool Calling Agent Demo

一个基于 **FastAPI + OpenAI-compatible API + SQLite** 的最小可用 Tool Calling Agent Demo。

本项目不依赖 LangChain，而是手写 Agent 主流程，用来展示大模型应用后端中常见的工具选择、工具执行、结构化响应和步骤级流式输出能力。

## 项目亮点

- 使用 FastAPI 对外提供 Agent HTTP 服务。
- 接入本地 vLLM 或其他 OpenAI-compatible LLM API。
- 手写最小 Tool Calling 流程：LLM 选择工具，后端执行真实工具函数。
- 支持 `calculator`、`file_summary`、`sql_query` 三类工具。
- 提供 `/agent/chat` 普通调用接口和 `/agent/stream` SSE 步骤级流式接口。
- calculator 使用 AST 安全解析表达式，避免直接使用 `eval`。
- SQL 工具限制只允许 `SELECT` 查询，降低危险写操作风险。
- 提供 SQLite 初始化脚本和接口测试脚本，方便本地验证。

## 技术栈

- Python 3.10+
- FastAPI / Uvicorn
- httpx / Pydantic / python-dotenv
- SQLite
- vLLM OpenAI-compatible API
- SSE StreamingResponse

## 项目结构

```text
fast-llm-agent-demo/
├── app/
│   ├── main.py          # FastAPI 入口，定义 HTTP 接口
│   ├── agent.py         # Agent 主流程：工具选择、工具执行、最终回答
│   ├── tools.py         # calculator / file_summary / sql_query
│   ├── llm_client.py    # 调用 OpenAI-compatible LLM API
│   └── schemas.py       # 请求和响应数据结构
├── data/
│   ├── demo.db          # SQLite 示例数据库，运行脚本后生成
│   └── sample.txt       # 文件摘要示例文本
├── scripts/
│   ├── init_db.py       # 初始化 SQLite 数据库
│   ├── test_llm_client.py
│   └── test_agent.py
├── .env.example
├── requirements.txt
└── README.md
```

## Agent 流程

```text
用户输入
  ↓
FastAPI /agent/chat
  ↓
LLM 选择工具并生成 JSON 参数
  ↓
Python 后端执行工具函数
  ↓
LLM 根据工具结果生成最终回答
  ↓
FastAPI 返回结构化响应
```

LLM 不直接读取文件、查询数据库或执行计算。它只输出工具名和参数，真正的动作由后端 Python 工具函数完成。

## 快速启动

### 1. 创建环境

```bash
cd ~/fast-llm-agent-demo
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 配置 LLM 服务

复制环境变量模板：

```bash
cp .env.example .env
```

`.env.example` 示例：

```env
LLM_BASE_URL=http://127.0.0.1:8000/v1
LLM_API_KEY=token-abc123
LLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct
```

如果使用本地 vLLM，可以先启动模型服务：

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --host 127.0.0.1 \
  --port 8000 \
  --api-key token-abc123
```

### 3. 初始化数据库

```bash
python3 scripts/init_db.py
```

### 4. 启动 FastAPI

```bash
uvicorn app.main:app --reload --port 9000
```

接口文档：

```text
http://127.0.0.1:9000/docs
```

## 接口示例

### 健康检查

```bash
curl http://127.0.0.1:9000/health
```

返回示例：

```json
{
  "status": "ok",
  "message": "FastAPI service is running"
}
```

### 普通 Agent 调用

```bash
curl -X POST http://127.0.0.1:9000/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"请帮我计算 20 + 4 * (-3)"}'
```

返回结构示例：

```json
{
  "user_message": "请帮我计算 20 + 4 * (-3)",
  "selected_tool": "calculator",
  "tool_arguments": {
    "expression": "20 + 4 * (-3)"
  },
  "tool_result": 8.0,
  "final_answer": "20 + 4 * (-3) 的计算结果是 8。"
}
```

### SSE 步骤级流式接口

```bash
curl -N -X POST http://127.0.0.1:9000/agent/stream \
  -H "Content-Type: application/json" \
  -d '{"message":"请查询价格大于100的商品"}'
```

流式事件示例：

```text
data:{"event":"start","user_message":"请查询价格大于100的商品"}

data:{"event":"tool_selected","selected_tool":"sql_query","tool_arguments":{"query":"SELECT ..."}}

data:{"event":"tool_result","tool_result":[...]}

data:{"event":"final_answer","final_answer":"..."}

data:{"event":"done"}
```

当前 `/agent/stream` 是步骤级 SSE 流式，不是 token-by-token 流式。它主要用于展示 Agent 执行过程。

## 工具设计

### calculator

- 用于简单数学计算。
- 使用 Python AST 解析表达式。
- 支持数字、加减乘除、括号、幂运算和负号。
- 不使用 `eval`，避免任意代码执行风险。

### file_summary

- 读取本地 `data/sample.txt`。
- Demo 中固定读取示例文件，避免任意文件读取。
- 返回截断后的文本摘要。

### sql_query

- 查询 SQLite 示例数据库中的 `products` 表。
- 只允许 `SELECT` 查询。
- 拒绝 `UPDATE`、`DELETE`、`DROP` 等写操作。

## 测试脚本

```bash
python3 scripts/test_llm_client.py
python3 scripts/test_agent.py
```

`scripts/test_agent.py` 覆盖以下场景：

- 数学计算
- 文件摘要
- SQLite 查询
- 异常输入

## 面试可讲点

- 为什么 Agent 不是普通聊天机器人：Agent 会选择工具并通过后端执行真实动作。
- 为什么不用 LangChain：本项目目标是理解底层流程，所以手写工具选择、参数解析、工具执行和最终回答生成。
- 为什么 LLM 不能直接查数据库：LLM 只能生成文本或结构化参数，真实数据库访问必须由后端代码完成。
- 工具调用安全怎么做：calculator 不用 `eval`，SQL 只允许 `SELECT`，文件工具限制示例路径。
- SSE 和 token 级流式的区别：当前项目展示步骤级事件，后续可以把 LLM Client 改为 `stream=True` 实现 token 级输出。

## 后续扩展

- 增加工具注册表，减少 `if` 分支。
- 增加 JSON Schema 参数校验。
- 增加 pytest 自动化测试。
- 将 SQLite 替换为 MySQL 或 PostgreSQL。
- 支持多轮对话历史。
- 接入 RAG 文档检索能力。
- 实现 token-by-token 流式输出。
