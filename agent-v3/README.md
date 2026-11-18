# EnterpriseAgent

企业知识库与工单处理智能体。项目以 LangGraph 官方 Agentic RAG 教程的状态图模式为参考：Agent 判断是否需要检索，知识检索和建单作为工具，失败时转人工并留下可跟踪的工单。

当前实现是一个可直接运行的完整 MVP：

- FastAPI：聊天、检索、文档摄取、工单和运营指标 API。
- LangGraph：`分类 → 检索 → 带依据回答 → 条件建单` 状态图。
- Qdrant：文档切块、向量持久化和相似度检索。
- 工单闭环：优先级/SLA、状态迁移、备注与事件时间线。
- Web 控制台：知识问答、来源展示、文档上传和工单队列。
- 双模式推理：默认本地无密钥 RAG；配置 OpenAI 兼容接口后切换 LLM 生成。
- Docker Compose：API 与 Qdrant 一键部署。

## 快速启动

需要 Docker 24+ 与 Docker Compose v2。

```bash
docker compose up -d --build
docker compose ps
```

首次构建默认使用清华 PyPI 镜像。在海外网络或企业内部源可覆盖：

```bash
PIP_INDEX_URL=https://pypi.org/simple docker compose build api
```

打开：

- 控制台：<http://localhost:8000>
- OpenAPI：<http://localhost:8000/docs>
- Qdrant 控制台：<http://localhost:6333/dashboard>

查看启动日志：

```bash
docker compose logs -f api
```

停止服务但保留知识库和工单数据：

```bash
docker compose down
```

## 演示问题

```text
VPN 连不上，怎么排查？
差旅报销需要哪些材料？
打印机一直卡纸，请帮我处理
P1 工单的响应时间是多少？
你好，你是谁？
```

“打印机一直卡纸”会同时返回知识库处理依据并自动创建工单。可以在右侧工单队列看到新记录。

## API 示例

知识问答：

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"VPN 无法连接，请帮我处理","requester":"demo","auto_create_ticket":true}'
```

仅检索：

```bash
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query":"年假可以怎么申请","limit":3}'
```

新增知识：

```bash
curl -X POST http://localhost:8000/api/documents \
  -H "Content-Type: application/json" \
  -d '{"title":"会议室预订规则","content":"会议室需提前一天预订。","source":"internal://admin/meeting","tags":["行政"]}'
```

创建工单：

```bash
curl -X POST http://localhost:8000/api/tickets \
  -H "Content-Type: application/json" \
  -d '{"title":"无法登录","description":"MFA 验证失败","requester":"张三","priority":"high","category":"账号与权限"}'
```

完整接口列表见 `/docs`。

## LLM 与 Embedding 配置

默认 `LLM_PROVIDER=local`、`EMBEDDING_PROVIDER=local`，无需密钥。本地模式使用多语言字符 n-gram 哈希向量和抽取式带依据回答，适合演示、离线和验收；它不等同于高质量生产模型。

切换到 OpenAI 兼容服务：

```bash
cp .env.example .env
```

在 `.env` 中设置：

```dotenv
LLM_PROVIDER=openai
LLM_API_KEY=your-key
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4.1-mini
EMBEDDING_PROVIDER=openai
EMBEDDING_DIMENSION=1536
EMBEDDING_MODEL=text-embedding-3-small
```

`LLM_BASE_URL` 可指向任何兼容 OpenAI Chat Completions 与 Embeddings API 的网关。切换 embedding 维度后，应用会自动重建 Qdrant collection，然后从内置种子文档重新建索引。

## 架构

```text
Browser / API Client
        |
        v
    FastAPI
        |
        v
   LangGraph StateGraph (每个节点都产生 trace span)
   classify -> retrieve -> answer
       |                     |
       +-> smalltalk         +-> [support request] create_ticket
        |
        +------ Tool: search_knowledge_base ----> Qdrant
        |
        +------ Tool: create_support_ticket ----> SQLite

   Traces -> JSON log + /api/traces + OTLP collector (可选)
```

工单当前使用 SQLite 持久化到 Docker volume，适合单实例 MVP。生产多实例应迁移到 PostgreSQL，并接入企业 SSO、RBAC、知识密级过滤与审计系统。

## 测试

本地运行测试：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pytest
```

Docker 内运行测试：

```bash
docker run --rm --user root -v "$PWD:/workspace" -w /workspace \
  -e PYTHONPATH=/workspace/src enterprise-agent-api \
  sh -c "pip install pytest && pytest"
```

若 Docker Hub 不可达，可先通过可用的镜像代理拉取基础镜像，再按原标签重命名后构建：

```bash
docker pull docker.m.daocloud.io/library/python:3.11-slim
docker tag docker.m.daocloud.io/library/python:3.11-slim python:3.11-slim
docker pull docker.m.daocloud.io/qdrant/qdrant:v1.12.4
docker tag docker.m.daocloud.io/qdrant/qdrant:v1.12.4 qdrant/qdrant:v1.12.4
```

集成冒烟：

```bash
curl http://localhost:8000/healthz
curl http://localhost:8000/api/dashboard
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"P1 工单多久响应？","auto_create_ticket":false}'
```

## 可观测性与故障定位

每个请求生成一条 trace，由带耗时的 span 组成，覆盖图节点、工具调用和 LLM 调用：

```text
agent.run                     (整条请求)
├── node.classify
├── node.retrieve
│   └── tool.search_knowledge_base
├── node.answer
│   ├── llm.generate_answer   (仅 LLM 模式)
│   └── llm_failure           (仅失败时记录)
└── node.create_ticket
    └── tool.create_support_ticket
```

三个出口：

- 结构化日志：每个 span 和 trace 各输出一行 JSON，含 `trace_id`、`status`、`error_type`、`duration_ms`
- HTTP 接口：`GET /api/traces` 列出最近 trace，`GET /api/traces/{trace_id}` 返回完整 span 树
- OTLP 导出：设置 `TRACING_OTLP_ENDPOINT` 后推送到任意兼容 OTLP 的 collector（Jaeger、Tempo、SigNoz 等）

LangSmith 无需改代码：LangGraph 节点是 LangChain runnable，设置 `LANGCHAIN_TRACING_V2=true` 与 `LANGCHAIN_API_KEY` 即可导出每个节点的 run。

```bash
curl -s localhost:8000/api/traces | python -m json.tool
curl -s localhost:8000/api/traces/<trace_id> | python -m json.tool
```

原来的 `except Exception: pass` 会静默吞掉 LLM 异常。现在失败会记录 `llm_failure` 事件（含错误类型、模型、已回退到本地回答），节点级 span 也带 `status=error` 与错误信息，因此"回答质量下降"这类静默故障可直接定位到具体节点。

## 安全边界

这个 MVP 默认面向内网演示，不应直接暴露到公网。正式上线前至少完成：

- 在反向代理层启用 HTTPS，并关闭跨域通配或限制可信来源。
- 为 API 与工单写操作接入 OAuth2/SSO、RBAC 和审计日志。
- 在检索前按部门、租户和文档密级添加 Qdrant payload filter。
- 对上传内容做病毒扫描、文件类型验证和敏感信息检测。
- 对模型输入输出做 PII 脱敏、提示注入防护和内容安全策略。
- 将 SQLite 替换为 PostgreSQL，并为 Qdrant 配置快照、备份与恢复演练。

## 官方参考

- [LangGraph 官方仓库](https://github.com/langchain-ai/langgraph)
- [Build a custom RAG agent with LangGraph](https://docs.langchain.com/oss/python/langgraph/agentic-rag)
- [LangGraph Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api)
- [Qdrant Quickstart](https://qdrant.tech/documentation/quickstart/)

7 天落地计划与生产化路线见 [docs/SPRINT.md](docs/SPRINT.md)。
