# neoagent 版本路线图

> 最后更新：2026-04-12

---

## 已发布版本

### v1 — 核心骨架

**状态**：已完成  
**测试**：173 tests

核心功能：

- `NeoAgent` 主类，统一入口
- Provider 抽象层（Anthropic / OpenAI 双 provider）
- `QueryLoop` 执行引擎（turn 管理、工具调用循环）
- Tool System：`BaseTool` ABC + `ToolRegistry`
- Prompt System：`PromptBuilder`（系统提示组装）
- 基础类型：`Message`、`Turn`、`ConversationResult`

---

### v2 — 智能增强

**状态**：已完成  
**测试**：265 tests

核心功能：

- **Context Compression**：LLM 摘要 + 截断回退；token 使用率超 70% 自动触发
- **Memory System**：`MemoryStore`（JSON 文件持久化）、`MemoryExtractor`（LLM 提取关键记忆）、`MemoryRetriever`（词法检索）
- **Skill System**：`PromptBuilder` 动态段管理、skill 懒加载机制

---

### v3.1 — 架构重构

**状态**：已完成  
**Tag**：`v3.1.0`（commit `e3594007`）  
**测试**：386 tests

核心功能：

- **Session & SessionState**：持久化存储、`fork`/`resume` 语义、`JsonFileStorage` 后端
- **EventBus**：10+ 事件类型、pub/sub 模式，解耦各模块间通信
- **ToolExecutor**：权限检查、并发分区执行、结果截断保护
- **BashTool**：安全执行环境（120s 超时、正则黑名单过滤危险命令）

---

### v3.2a — 可扩展性

**状态**：已完成  
**Tag**：`v3.2a`（commit `ae74d0c`）  
**测试**：577 tests

核心功能：

- **Hook System**：`HookManager` + 4 个拦截点（`pre_tool_call` / `post_tool_call` / `pre_provider_call` / `post_provider_call`）
- **HookResult 三态**：`allow` / `deny` / `modify`；hook 参数用 frozen dataclass 防篡改
- **MCP 集成**：`MCPClient`、`MCPTool`、`StdioTransport`；MCP server 声明式配置
- **DeferredToolRegistry**：工具延迟加载；`ToolSearchTool` 按需提升工具到上下文

---

### v3.2b — 多智能体

**状态**：已完成  
**测试**：732 tests（修复后稳定）

核心功能：

- **Orchestrator + WorkerPool + TaskTracker**：三层 multi-agent 架构
- **Task Envelope**：frozen dataclass，携带 `timeout` / `max_turns` 约束
- **5 个 Orchestrator 内置工具**：`spawn_worker`、`delegate_task`、`cancel_task`、`list_tasks`、`list_workers`
- **asyncio 并发调度**：任务级并发、取消链路（`CancelledError` 安全传播）
- **WorkerCard**：markdown 格式 worker 能力描述解析器

---

### v3.2c — 通道抽象

**状态**：已完成  
**Tag**：`v3.2c`（commits `8ee94b7` + `adc52c9`）  
**测试**：760 tests

核心功能：

- **Channel ABC**：`start` / `stop` / `serve_forever` 三方法生命周期接口
- **FastAPIChannel**：
  - `GET /v1/health` — 健康检查
  - `POST /v1/run` — 同步 JSON 响应
  - `POST /v1/run/stream` — SSE 流式响应
- **SSE 流式支持**：`asyncio.Queue` 桥接 agent 事件流，`streaming=True/False` 开关
- **可选依赖**：`pip install neoagent[fastapi]`，不安装 fastapi 时核心功能不受影响
- **类型兼容**：PEP 563 + FastAPI 类型解析 workaround（`update_forward_refs`）

---

### v3.2d — Session Recovery + Evaluation & Observability（当前版本）

**状态**：已完成  
**Tag**：`v3.2d`  
**测试**：818 tests

核心功能：

- **Session Recovery**：
  - Per-turn auto-save：QueryLoop 每轮 turn 边界自动保存 session（崩溃后从最后完成的 turn 恢复）
  - Resume validation：`resume(validate=True)` 检验 workspace 存在性 + 24h 活跃度，发出 `SessionResumeWarningEvent`
  - Storage cleanup：`JsonFileStorage.cleanup(max_age_days, max_sessions)` 自动清理过期/超量 session 文件
- **Evaluation & Observability**：
  - `UsageTracker`：按模型分组统计 input/output token 用量
  - `MetricsCollector`：per-turn 结构化指标（tokens、latency、tool count）
  - `EvalRunner`：批量评测框架（EvalCase + assertion + 异常隔离 + EvalReport）
- **11 个组件维度全部覆盖**，原始 master spec 终态目标达成

---

## 下一阶段规划

### v3.3 — 双通道支持（计划中）

目标：在 Channel ABC 基础上添加消息队列通道实现，让单个 NeoAgent 实例可以同时挂载 HTTP 和 MQ 两种对外服务方式。

具体条目：

- `KafkaChannel` / `RabbitMQChannel`：实现 Channel ABC 的三个生命周期方法
- JSON wire format：消息序列化/反序列化（请求入 → agent 处理 → 结果出）
- Broker 生命周期管理：连接、断线重连、优雅关闭
- 消息确认与重试：ACK 机制、失败重试、死信队列
- 双通道并行运行：同一 NeoAgent 同时挂载 HTTP + MQ，互不干扰
- 通道级配置：不同通道可独立设置 `max_turns`、`timeout` 等参数
- 消息幂等性：相同 `message_id` 不重复处理

设计前提：Channel ABC 的 `start/stop/serve_forever` 对 MQ 语义天然适用；MQ 场景以异步任务模式为主（请求 → ACK → 后台处理 → 结果回写），与 HTTP 同步/SSE 模式正交，互不影响。

---

## 远期方向

### v3.4 — 安全与鉴权

- API Key 鉴权（FastAPI dependency 注入）
- 有状态 Session Channel（`/v1/sessions/{id}/run`）
- 请求限速与配额管理

### v3.5 — A2A 协议

- Agent-to-Agent Protocol 支持
- Teams mode（多 agent 协作编排）
- Remote Workers（跨进程/跨机器 worker 注册与调度）

### 远期 — create-agent Skill

将 neoagent 的 agent 创建范式封装为 superpowers 插件 skill：

- 输入：需求描述 + 约束条件
- 输出：完整 NeoAgent 配置 + 自定义工具 + 部署方案
- 核心逻辑：基于 ai-knowledge 知识库的架构决策引擎，匹配最优配置方案
- 关系链：`ai-knowledge`（架构对比 + 设计权衡）→ `create-agent-skill`（生成配置）→ 用户 agent 项目

---

## 版本一览

| 版本 | 状态 | Tag / Commit | Tests | 核心主题 |
|------|------|-------------|-------|---------|
| v1 | 已完成 | — | 173 | 核心骨架 |
| v2 | 已完成 | — | 265 | 智能增强（压缩 + 记忆 + skill） |
| v3.1 | 已完成 | `v3.1.0` / `e3594007` | 386 | 架构重构（Session + EventBus + ToolExecutor） |
| v3.2a | 已完成 | `v3.2a` / `ae74d0c` | 577 | 可扩展性（Hooks + MCP + 延迟加载） |
| v3.2b | 已完成 | — | 732 | 多智能体（Orchestrator + WorkerPool） |
| v3.2c | 已完成 | `v3.2c` / `adc52c9` | 760 | 通道抽象（FastAPI + SSE） |
| v3.2d | 已完成 | `v3.2d` | 818 | Session Recovery + Evaluation & Observability |
| v3.3 | 计划中 | — | — | 双通道（MQ 支持） |
| v3.4 | 远期 | — | — | 安全与鉴权 |
| v3.5 | 远期 | — | — | A2A 协议 |
