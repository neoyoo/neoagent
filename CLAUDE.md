# neoagent

## 项目概述

neoagent 是一个 Python AI Agent SDK，目标是提供生产级的 agent 开发框架。当前版本包含核心 agent loop、工具系统、Hook 拦截系统、MCP 集成、会话管理、上下文压缩、记忆系统、可观测性。

支持 Anthropic 和 OpenAI 双 provider，核心链路全部 async/await，严格类型注解。

## 架构层次

| 模块 | 职责 |
|------|------|
| `agent.py` | 主入口 NeoAgent 类，协调各子系统 |
| `core/` | QueryLoop（主循环）、PromptBuilder（提示构建）、ContextCompressor（上下文压缩） |
| `tools/` | BaseTool、ToolRegistry、ToolExecutor、DeferredToolRegistry、builtin tools |
| `hooks.py` | HookManager —— pre/post tool_call、pre/post provider_call 四个拦截点 |
| `mcp/` | MCPClient、StdioTransport、MCPTool —— MCP 协议集成 |
| `session.py` | Session、SessionState、SessionStorage —— 会话持久化 |
| `events.py` | EventBus，10 种事件类型，发布/订阅 |
| `memory/` | MemoryManager、MemoryStore、extractor、retriever |
| `providers/` | OpenAI、Anthropic 适配器 |
| `observe.py` | Observer 可观测性框架 |

## 版本演进路线图

| 版本 | 状态 | 内容 |
|------|------|------|
| v3.1 | 完成 | 核心架构重构（Session / EventBus / ToolExecutor / BashTool），300+ tests |
| v3.2a | 完成 | Hooks 拦截系统 + MCP 集成（DeferredToolRegistry, tool_search），577 tests |
| v3.2b | 规划中 | Multi-Agent（Orchestrator + Workers 模式） |
| v3.2c | 待定 | Channel 接口层 |

### 远期演进方向

- **A2A（Agent-to-Agent Protocol）**：Worker 接口扩展为远程 agent 通信，支持 Google A2A 协议标准
- **Teams 模式**：多 agent 共享项目上下文 + 协调策略，类似 Claude Code Teams
- **关键设计原则**：Worker 是协议不是类，本地/远程 worker 只是传输层不同，核心调度逻辑不变

## 开发规范

- **测试**：pytest，TDD 优先，当前 577+ tests，新功能须先写测试
- **异步**：核心链路全部 `async/await`，禁止在异步上下文中使用同步阻塞调用
- **类型**：严格类型注解，使用 `mypy` 检查
- **不可变**：事件/Hook payload 使用 `frozen dataclass` + `MappingProxyType`，禁止在 hook 内修改 payload
- **提交前**：确保 `pytest` 全绿，不跳 hook，不强推 main
