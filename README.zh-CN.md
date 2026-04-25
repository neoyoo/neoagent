# neoagent

> 以可插拔抽象和单一会话状态为核心的 Python 异步 Agent SDK。

[English](README.md) | [中文]

## 是什么

neoagent 是一个面向生产级需求的 Python Agent SDK，覆盖 Agent 主循环、工具调度、上下文压缩、跨会话记忆、Hook 拦截、MCP 集成、多 Agent 编排和 HTTP Channel 暴露，所有功能通过单一 `NeoAgent` 类统一接入，全链路 async/await。

SDK 同时支持 Anthropic 和 OpenAI provider，严格类型注解，测试覆盖 577+ 用例（TDD 优先开发）。

这是个人基础设施项目，不在 PyPI 上发布，需从本仓库直接安装。当前活跃分支为 `v2.0`。

## 设计理念

**每个存储决策都有可插拔抽象。** 会话持久化、工作记忆、压缩消息存储、压缩策略、记忆 Review 策略各自对应一个 Protocol。默认实现开箱即用，需要定制后端（Postgres、Elasticsearch、Qdrant、S3）时只需实现对应 Protocol，其余代码不用动。

**SessionState 是唯一事实来源。** Agent 恢复、内省、交接所需的一切状态都在 `SessionState` 里。没有散落在各处的单例或模块级全局变量，会话持久化、崩溃恢复和断点续传因此是自然的，而不是事后打补丁。

**默认事件驱动。** 工具调用、对话轮次完成、压缩事件、工作记忆变更等所有有意义的动作都通过 `EventBus` 发布类型化 Event。Observer 订阅无需侵入核心代码。Hook handler 可在工具执行前拦截并阻断。Hook 系统和事件系统可组合、相互独立。

**双触发压缩让上下文预算可控。** `ContextCompressor` 在两个独立信号触发：每 10 个用户轮次，或 token 用量超过 `context_budget` 的 70%。压缩执行时，原始消息体以 `msg_id` 为索引写入 `CompressedMessageStore`。内置 `recall_turn` 工具允许 Agent 按需取回完整原文——信息不会永久丢失。

**延迟工具注册防止上下文膨胀。** 工具可存放在 `DeferredToolRegistry`（对 LLM 不可见），通过内置 `tool_search` 工具按需提升到活跃注册表。大型 MCP 工具集默认走这条路——LLM 只在需要时才看到对应工具。

## 快速开始

从本地仓库安装：

```bash
pip install -e /path/to/neoagent
```

最简 Agent：

```python
import asyncio
import os
from neoagent import NeoAgent, NeoAgentConfig

config = NeoAgentConfig(
    api_key=os.environ["ANTHROPIC_API_KEY"],
    model="claude-sonnet-4-6",
    system_prompt="You are a helpful assistant.",
    max_turns=30,
    context_budget=80_000,
)
agent = NeoAgent(config)

async def main() -> None:
    reply: str = await agent.chat("Hello")
    print(reply)

asyncio.run(main())
```

两个入口：`agent.chat(message)` 返回 `str`，适合单轮交互；`agent.run(messages, max_turns=N)` 返回 `ConversationResult`，适合完整对话控制。

## 架构

SDK 分为 12 个模块，简要结构如下：

```
NeoAgent (agent.py)
  ├── QueryLoop (core/loop.py)            — 执行引擎，状态机
  │     ├── Provider (providers/)         — Anthropic / OpenAI 适配器
  │     ├── ToolExecutor                  — ToolRegistry + DeferredToolRegistry
  │     ├── HookManager (hooks.py)        — pre/post 拦截
  │     └── ContextCompressor             — 按轮次或 token 阈值自动压缩
  ├── MemoryManager (memory/)             — 跨会话记忆提取与检索
  ├── Session + SessionState (session.py) — 持久会话、断点续传
  ├── EventBus (events.py)               — 发布/订阅可观测总线
  └── Observer (observe.py)              — 结构化日志 + 评估

独立扩展模块：
  mcp/          — MCPClient, StdioTransport, MCPTool
  multi/        — Orchestrator, WorkerCard
  channels/     — FastAPIChannel（POST /v1/run, /v1/run/stream）
  eval/         — EvalRunner, EvalCase, EvalReport

v2 内部模块（neoagent/v2/）：
  schema.py           — WorkingMemory, Batch, CompressionDelta, Layer
  compressed_store.py — CompressedMessageStore Protocol + 默认 InMemory 实现
  abc.py              — WorkingMemoryStore Protocol
  strategies/         — OneShotCompressionStrategy, OneShotMemoryReviewStrategy
```

完整模块图、所有场景代码骨架和生产就绪检查清单见 `skills/neoagent/SKILL.md`。

## 可插拔扩展点

所有五个扩展点遵循相同模式：实现 Protocol，通过 `NeoAgentConfig` 传入实例。

| Protocol | 配置字段 | 默认实现 |
|---|---|---|
| `SessionStorage` | `session_dir` | `JsonFileStorage` |
| `WorkingMemoryStore` | `working_memory_store` | `InMemoryWorkingMemoryStore` |
| `CompressedMessageStore` | `compressed_message_store` | `InMemoryCompressedMessageStore` |
| `CompressionStrategy` | `compression_strategy` | `OneShotCompressionStrategy` |
| `MemoryReviewStrategy` | `memory_review_strategy` | `OneShotMemoryReviewStrategy` |

`CompressedMessageStore` 是 v2.0 新增的扩展点。它将压缩后消息体的存放位置与 Session 记录本身解耦。默认实现包装 `session_state.compressed_messages`，使 `JsonFileStorage` 继续在单文件内往返序列化所有内容。指向 Postgres 或 Elasticsearch 可在规模化场景下独立存储消息体。

## 示例

**`examples/v2_repl.py`** — 交互式终端 REPL。完整演练 v2.0 功能集：`PromptBuilder`、工作记忆快照、压缩、free/recall、自定义工具、逐轮请求日志。支持 `--list` 和 `--resume <SESSION_ID>`。

**`examples/migrate_log_to_session.py`** — 重放 `logs/<id>/events.jsonl` 以重建 Session JSON。适用于在配置 `session_dir` 之前已捕获的旧会话迁移。

## 配套 Skill

`skills/neoagent/SKILL.md` 是使用本 SDK 的权威详细指南，包含完整模块图、所有场景代码骨架（从最简 Agent 到评估框架）、反模式参考和生产就绪检查清单。将其链接或复制到 `~/.claude/skills/` 即可在 Claude Code 中启用。

```bash
ln -s /path/to/neoagent/skills/neoagent ~/.claude/skills/neoagent
```

## 许可证

MIT。见 `LICENSE`。

## 状态

个人基础设施项目，API 可能持续演进。`v2.0` 是当前活跃分支。不在 PyPI 上发布，请使用 `pip install -e /path/to/neoagent` 安装。
