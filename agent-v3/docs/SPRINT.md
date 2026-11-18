# EnterpriseAgent 7 天冲刺

## 目标

第 7 天结束时，交付一个可演示、可测试、可通过 Docker Compose 部署的企业知识库与工单处理智能体。MVP 必须覆盖 RAG 问答、来源引用、故障分诊、自动建单、工单状态流转和基础运营指标。

## Day 1：工程基线与验收口径

- 建立 FastAPI、LangGraph、Qdrant、pytest 和 Docker Compose 骨架。
- 确认三类黄金问题：政策咨询、故障自助、知识缺失转人工。
- 定义 API 契约、工单字段、优先级和 SLA。
- 验收：`docker compose up --build` 后 `/healthz` 返回正常。

## Day 2：知识摄取与检索

- 支持纯文本、Markdown、CSV 上传及按来源重新索引。
- 完成切块、向量化、Qdrant collection 和环境隔离。
- 准备账号、VPN、网络、报销、人事、行政、安全种子知识。
- 验收：关键词与语义查询均能返回 Top-K 片段及来源。

## Day 3：LangGraph Agentic RAG

- 实现分类、检索、生成、直接回复和建单节点。
- 将知识检索和建单注册为 LangChain tools。
- 使用条件边控制检索与建单，不在寒暄时调用检索。
- 验收：图结构可通过 `/api/agent/graph` 查看，节点轨迹可追踪。

## Day 4：工单闭环

- 实现建单、列表、详情、状态更新、备注和 SLA 截止时间。
- 校验非法状态迁移，保存工单事件时间线。
- 对故障请求提供“知识解答 + 自动建单”组合体验。
- 验收：open → in_progress → resolved 全链路通过。

## Day 5：评测与稳定性

- 建立不少于 20 条带标注答案和来源的评测集。
- 统计 Recall@K、来源命中率、无依据率、建单准确率和 P95 延迟。
- 增加 Qdrant 重试、模型失败降级、输入长度和上传大小限制。
- 验收：单元/集成测试全部通过，故障注入时 API 可解释降级。

## Day 6：安全、权限与运维

- 接入企业 SSO，按部门/密级过滤知识片段。
- 工单 API 增加 API Key 或 OAuth2，审计写操作。
- 引入 PostgreSQL 替代 SQLite，配置备份、日志和指标。
- 验收：跨密级检索不发生越权，所有写操作有审计记录。

## Day 7：上线彩排与演示

- 用真实制度文档完成 10 问演示脚本。
- 完成 100 并发只读压测并记录容量边界。
- 配置反向代理、TLS、健康检查和进程重启策略。
- 验收：新机器按 README 可在 10 分钟内复现部署。

## 规模化路线

1. 用 PostgreSQL + pgvector 或 Qdrant 集群承载多租户知识空间。
2. 用 LangGraph checkpointer 持久化会话与人工审批中断点。
3. 用消息队列异步执行文档解析和大批量重建索引。
4. 接入企业 IAM、数据分类标签、审计日志与脱敏网关。
5. 用 LangSmith 或 OpenTelemetry 做节点级追踪和在线质量监控。

