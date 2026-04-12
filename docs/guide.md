# neoagent 使用指南

## 1. 概述

neoagent 是一个 Python 异步 agent 框架，将 LLM 调用、工具执行、上下文压缩、记忆持久化、多智能体协调和 HTTP 服务整合到一个可组合的流水线中。

**核心设计理念：**

- **Async-first**：全链路 `asyncio`，工具并发执行，HTTP 服务不阻塞主循环
- **类型安全**：Pydantic v2 贯穿消息、工具入参、配置等所有数据边界
- **可组合**：每个子系统（Provider / Tool / Hook / Memory / Channel）都是独立模块，可单独替换

**架构概览（文字版）：**

```
用户 / HTTP Client
        ↓
  Channel (FastAPIChannel)
        ↓
    NeoAgent
   ┌────────────────────────────────────────────┐
   │  PromptBuilder  →  system prompt           │
   │  QueryLoop (执行引擎)                        │
   │    ├── ContextCompressor (压缩)             │
   │    ├── Provider (Anthropic / OpenAI)        │
   │    ├── ToolExecutor → ToolRegistry          │
   │    ├── HookManager (前后拦截)               │
   │    ├── MemoryManager (持久记忆)             │
   │    └── EventBus (可观测性)                  │
   └────────────────────────────────────────────┘
        ↓
    Session / JsonFileStorage (持久化)
```

---

## 2. 安装

```bash
# 核心依赖（Anthropic + OpenAI + Pydantic + tiktoken）
pip install neoagent

# 可选：HTTP 服务（FastAPI + uvicorn）
pip install neoagent[fastapi]

# 开发依赖（pytest + pytest-asyncio + httpx）
pip install neoagent[dev]
```

**要求：Python >= 3.11**

---

## 3. 快速上手

```python
import asyncio
from neoagent.agent import NeoAgent
from neoagent.config import NeoAgentConfig

async def main():
    config = NeoAgentConfig(api_key="sk-ant-...")
    agent = NeoAgent(config)

    reply = await agent.chat("请介绍一下你自己")
    print(reply)

asyncio.run(main())
```

**运行说明：**

- `chat()` 创建一个临时 Session，发送消息，返回 assistant 的纯文本回复
- 不传 session 时，对话结束后状态自动丢弃（无持久化）
- 默认使用 `claude-sonnet-4-20250514`，最多 30 轮对话

---

## 4. 核心概念

### NeoAgent

主入口类。内部按以下顺序完成初始化：

1. 根据 `config.provider` 实例化 `AnthropicProvider` 或 `OpenAIProvider`
2. 创建 `ToolRegistry`、`ToolExecutor`、`PermissionChecker`
3. 创建 `PromptBuilder`，注入 `identity` 段（来自 `config.system_prompt`）
4. 创建 `QueryLoop`（核心执行引擎）
5. 若 `config.session_dir` 不为 None，自动创建 `JsonFileStorage`

```python
class NeoAgent:
    def __init__(self, config: NeoAgentConfig, storage: SessionStorage | None = None) -> None: ...

    # 高层 API：发送消息，返回纯文本
    async def chat(self, message: str, session: Session | None = None) -> str: ...

    # 低层 API：传入完整消息列表，返回 ConversationResult
    async def run(self, messages: list[Message], session: Session | None = None, *, max_turns: int | None = None) -> ConversationResult: ...

    # 工具注册
    def register_tool(self, tool: BaseTool) -> None: ...

    # Session 管理
    def new_session(self, session_id: str | None = None) -> Session: ...
    def resume(self, session_id: str) -> Session: ...

    # 记忆系统
    def enable_memory(self, memory_dir: Path | None = None, project_key: str | None = None) -> None: ...

    # 可观测性
    def enable_logging(self, log_dir: Path | None = None, console: bool = True) -> Observer: ...
    def disable_logging(self) -> None: ...

    # Hook 系统
    def hook(self, hook_type: HookType, handler: HookHandler, priority: int = 0) -> None: ...
    def on(self, hook_type: HookType, priority: int = 0): ...  # 装饰器形式
    def unhook(self, hook_type: HookType, handler: HookHandler) -> None: ...

    # MCP
    async def add_mcp_server(self, name: str, command: list[str], env: dict[str, str] | None = None) -> None: ...
    async def remove_mcp_server(self, name: str) -> None: ...
    def list_mcp_servers(self) -> list[str]: ...

    # 生命周期
    async def close(self) -> None: ...

    @property
    def event_bus(self) -> EventBus: ...
```

### Message / Turn / ConversationResult

这三个类型描述对话的数据流：

```python
# 一条消息（来自 pydantic BaseModel）
class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str | list[ContentBlock]

# content 可以是以下三种 block 的列表：
class TextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str

class ToolUseBlock(BaseModel):
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict

class ToolResultBlock(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str
    is_error: bool = False

# 一轮对话（一次 LLM 调用 + 可能的工具调用）
class Turn(BaseModel):
    response: Message            # assistant 回复
    tool_calls: list[ToolCall]   # 本轮发起的工具调用
    tool_results: list[ToolResult]  # 工具执行结果
    stop_reason: Literal["end_turn", "tool_use", "max_tokens"]

# 完整对话结果
class ConversationResult(BaseModel):
    turns: list[Turn]
    reason: Literal["completed", "max_turns"]
```

### Provider

Provider 抽象对接 LLM 供应商。通过 `config.provider` 字段选择：

```python
class Provider(ABC):
    @abstractmethod
    async def create(self, system: str, messages: list, tools: list, **kwargs) -> Response: ...

    @abstractmethod
    def get_context_window(self) -> int: ...
```

- **AnthropicProvider**：默认，上下文窗口 200,000 tokens，模型 `claude-sonnet-4-20250514`
- **OpenAIProvider**：上下文窗口随模型变化（gpt-4o 为 128,000），模型 `gpt-4o`

### QueryLoop

执行引擎，内部循环逻辑：

1. 检查是否需要压缩上下文（超过 `context_budget` 的 70%）
2. 构建 system prompt（`PromptBuilder.build()`）
3. 触发 `pre_provider_call` Hook
4. 调用 Provider 获取响应
5. 触发 `post_provider_call` Hook
6. 若 `stop_reason == "end_turn"`：触发记忆提取，返回结果
7. 若 `stop_reason == "tool_use"`：执行工具（含 pre/post_tool_call Hook），追加消息，继续循环
8. 超过 `max_turns`：以 `reason="max_turns"` 返回

---

## 5. 工具系统

### BaseTool ABC

所有工具继承 `BaseTool`：

```python
from abc import ABC, abstractmethod
from typing import Literal
from pydantic import BaseModel
from neoagent.tools.base import BaseTool
from neoagent.core.types import ToolResult

class BaseTool(ABC):
    name: str                                        # 工具名，必须唯一
    description: str                                 # 给 LLM 看的描述
    input_model: type[BaseModel]                     # Pydantic 模型，定义入参 schema
    permission: Literal["auto", "ask", "deny"] = "ask"  # 权限模式
    is_concurrent_safe: bool = False                 # 是否可与其他工具并发执行

    @abstractmethod
    async def execute(self, input: BaseModel) -> ToolResult: ...

    def get_schema(self) -> dict: ...  # 自动生成 Anthropic 风格 tool schema
```

### 注册工具

```python
agent = NeoAgent(config)
agent.register_tool(MyTool())
```

`ToolRegistry` 在注册时校验名称唯一性（重复注册抛 `ValueError`）。

### 权限模型

| permission | 行为 |
|-----------|------|
| `"auto"` | 自动执行，无需用户确认 |
| `"ask"` | 默认值；若 `auto_approve_tools=True` 则自动批准，否则拒绝 |
| `"deny"` | 永久拒绝，且不暴露给 LLM（不出现在 tool schema 列表中） |

全局开关：`NeoAgentConfig(auto_approve_tools=True)` 让所有 `"ask"` 工具自动批准。

### 并发安全

`ToolExecutor` 将同一轮次的工具调用分为两组：
- `is_concurrent_safe=True`：用 `asyncio.gather()` 并发执行
- `is_concurrent_safe=False`（默认）：串行执行

### 内置工具概览

| 工具名 | 描述 |
|-------|------|
| `read` | 读取文件内容 |
| `write` | 写入文件 |
| `edit` | 精确字符串替换 |
| `bash` | 执行 shell 命令（受限沙箱） |
| `grep` | 正则搜索文件内容 |
| `glob` | 文件路径模式匹配 |

BashTool 安全特性：正则黑名单拦截危险命令（`rm -rf`、`curl | sh` 等），环境变量白名单过滤，`bash --restricted` 受限模式。

### 自定义工具完整示例

```python
import asyncio
from pydantic import BaseModel, Field
from neoagent.tools.base import BaseTool
from neoagent.core.types import ToolResult
from neoagent.agent import NeoAgent
from neoagent.config import NeoAgentConfig


class FetchUrlInput(BaseModel):
    url: str = Field(description="要获取的 URL")
    timeout: int = Field(default=10, ge=1, le=60, description="超时秒数")


class FetchUrlTool(BaseTool):
    name: str = "fetch_url"
    description: str = "获取指定 URL 的 HTTP 响应内容"
    input_model: type[BaseModel] = FetchUrlInput
    permission: str = "auto"          # 自动批准
    is_concurrent_safe: bool = True   # 多个 URL 可并发

    async def execute(self, input: BaseModel) -> ToolResult:
        assert isinstance(input, FetchUrlInput)
        import aiohttp
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(input.url, timeout=aiohttp.ClientTimeout(total=input.timeout)) as resp:
                    text = await resp.text()
                    return ToolResult(call_id="", output=text[:5000])
        except Exception as e:
            return ToolResult(call_id="", output=str(e), is_error=True)


async def main():
    config = NeoAgentConfig(api_key="sk-ant-...", auto_approve_tools=True)
    agent = NeoAgent(config)
    agent.register_tool(FetchUrlTool())

    reply = await agent.chat("请获取 https://example.com 并总结其内容")
    print(reply)

asyncio.run(main())
```

---

## 6. Prompt 系统

### PromptBuilder

PromptBuilder 将多个 `PromptSection` 按优先级拼接成 system prompt。

```python
from neoagent.core.prompt import PromptBuilder, PromptSection

class PromptSection:
    name: str                       # 段名，唯一
    content: str | Callable[[], str]  # 静态字符串或动态函数（每次 build() 时调用）
    priority: int                   # 排序权重，数字越小越靠前（静态段先于动态段）
    is_static: bool = True
```

**排序规则：** 先静态段（`is_static=True`），再动态段；同类内按 `priority` 升序排列。最终格式为：

```
# identity
You are neoagent, a helpful AI assistant.

# memory
（动态内存内容）
```

### Skill 懒加载

Skill 是预先注册但未激活的 prompt 段：

```python
builder = agent._prompt_builder

# 注册 skill（不立即激活）
builder.register_skill("code_review", PromptSection(
    name="code_review",
    content="你是一名代码审查专家，重点关注安全性和可维护性。",
    priority=10,
    is_static=True,
))

# 按需激活
builder.activate_skill("code_review")

# 停用
builder.deactivate_skill("code_review")

# 查询状态
builder.is_skill_active("code_review")  # -> bool
```

### 添加自定义 system prompt

```python
from neoagent.config import NeoAgentConfig
from neoagent.agent import NeoAgent
from neoagent.core.prompt import PromptSection

# 方式 1：通过 config（最简单）
config = NeoAgentConfig(
    api_key="...",
    system_prompt="你是一名专业的数据分析师，擅长 Python 和 SQL。",
)

# 方式 2：在 NeoAgent 初始化后追加动态段
agent = NeoAgent(config)
agent._prompt_builder.add_section(PromptSection(
    name="context",
    content=lambda: f"当前工作目录：{Path.cwd()}",
    priority=10,
    is_static=False,
))
```

---

## 7. Session 管理

Session 封装对话历史和跨轮次状态（token 统计、压缩计数、记忆基线等）。

### Session API

```python
from neoagent.session import Session, SessionState, JsonFileStorage

# 创建新 session
session = Session.create()                   # 自动生成 UUID
session = Session.create("my-session-id")   # 指定 ID

# fork（深拷贝，产生独立会话）
new_session = session.fork()
new_session = session.fork("fork-id")

# 通过 agent 创建
session = agent.new_session()
session = agent.new_session("session-123")

# 恢复已存储的 session（需要配置 storage）
session = agent.resume("session-123")   # 若不存在抛 KeyError
```

### SessionState 字段

```python
class SessionState:
    previous_summary: str | None    # 上次压缩产生的摘要
    compression_failures: int       # 连续压缩失败次数
    memory_tool_calls: int          # 当前记忆周期内的工具调用次数
    memory_token_baseline: int      # 上次记忆提取时的 token 数
    total_input_tokens: int         # 累计输入 tokens
    total_output_tokens: int        # 累计输出 tokens
    promoted_tools: set[str]        # 本 session 已提升的 MCP 工具
```

### JsonFileStorage 持久化

```python
from pathlib import Path
from neoagent.session import JsonFileStorage

storage = JsonFileStorage(Path("./sessions"))

# 通过 config 自动配置
config = NeoAgentConfig(api_key="...", session_dir=Path("./sessions"))
agent = NeoAgent(config)

# 使用持久化 session
session = agent.new_session("user-42")
await agent.chat("你好", session=session)       # 自动保存
await agent.chat("继续上面的话题", session=session)  # 追加历史

# 跨进程恢复
session = agent.resume("user-42")
await agent.chat("我们之前聊到哪里了？", session=session)

# 列出所有 session
storage.list_ids()   # -> list[str]
storage.delete("user-42")
```

Session 文件以 `{session_id}.json` 存储，包含消息历史、SessionState 和时间戳。

---

## 8. 上下文压缩

### 触发条件

每轮对话开始前，`QueryLoop` 计算当前 token 用量：

```
(消息 tokens + 工具 schema tokens) > context_budget × 0.7
```

`context_budget` 默认取 Provider 的上下文窗口大小（Anthropic: 200,000；gpt-4o: 128,000）。

### 工作原理

1. **LLM 摘要**：保留首条消息（anchor）和最近 6 条消息，对中间部分让 LLM 生成结构化摘要（GOAL / PROGRESS / DECISIONS / FILES / NEXT STEPS / KEY CONTEXT）
2. **增量更新**：若 `session_state.previous_summary` 不为空，摘要会在原基础上增量更新
3. **截断回退**：LLM 摘要连续失败 3 次后，切换为直接截断（保留 anchor + 最近 6 条）
4. **配对清理**：压缩后自动修复 `tool_use` / `tool_result` 孤对问题

### 配置

```python
config = NeoAgentConfig(
    api_key="...",
    context_budget=100000,  # 手动设置预算（默认取 provider 上下文窗口）
)
```

设为 `0` 时自动使用 Provider 上下文窗口大小。

---

## 9. 记忆系统

记忆系统让 agent 在对话结束后仍能记住关键信息，下次会话时自动召回。

### MemoryStore：文件存储

```python
from neoagent.memory.store import MemoryStore
from pathlib import Path

store = MemoryStore(Path("~/.neoagent/memory/my-project"))
# 目录结构：
# MEMORY.md        ← 索引，格式：- [描述](文件名.md)
# user_prefs.md    ← 按主题分文件
# project_info.md
```

### MemoryManager：触发条件

| 条件 | 阈值 |
|-----|------|
| 工具调用次数 | >= 5 次 |
| Token 增量 | >= 4,000 tokens |

两个条件满足其一即触发提取。

提取流程：LLM 分析对话 → 输出 JSON 数组（filename / description / content）→ 写入 MemoryStore → 重建 MEMORY.md 索引。

### MemoryRetriever：词法检索

检索时按关键词匹配度对所有主题文件评分，返回索引 + Top-5 相关文件内容（每文件最多 2,000 字符）。

### 启用记忆

```python
from pathlib import Path

# 方式 1：自动路径（按工作目录 hash 决定）
agent.enable_memory()

# 方式 2：指定目录
agent.enable_memory(memory_dir=Path("./memory"))

# 方式 3：指定项目 key（便于跨目录共享）
agent.enable_memory(project_key="my-project")
```

记忆内容会自动注入 system prompt 的动态段，在每次 LLM 调用前刷新。

---

## 10. Hook 系统

Hook 在关键节点拦截执行流程，支持日志审计、安全过滤、请求改写。

### 4 个拦截点

| Hook 类型 | 触发时机 | 支持 deny |
|----------|---------|---------|
| `pre_tool_call` | 工具执行前（权限检查通过后） | 是 |
| `post_tool_call` | 工具执行后 | 否（忽略） |
| `pre_provider_call` | Provider 调用前 | 是 |
| `post_provider_call` | Provider 调用后 | 否（忽略） |

### HookResult 三态

```python
from neoagent.hooks import HookResult

HookResult.allow()                           # 放行，继续正常流程
HookResult.deny(reason="危险命令")            # 中止（仅 pre hook 有效）
HookResult.modify({"tool_input": {...}})     # 修改后放行
```

### 注册 Hook 示例

```python
from neoagent.hooks import PreToolCallEvent, PostToolCallEvent, HookResult
from neoagent.agent import NeoAgent

agent = NeoAgent(config)

# 方式 1：函数式注册
async def audit_tool(event: PreToolCallEvent) -> HookResult:
    print(f"[audit] 调用工具 {event.tool_name}，参数：{dict(event.tool_input)}")
    return HookResult.allow()

agent.hook("pre_tool_call", audit_tool)

# 方式 2：装饰器
@agent.on("pre_tool_call", priority=0)
async def block_dangerous(event: PreToolCallEvent) -> HookResult:
    if event.tool_name == "bash":
        cmd = event.tool_input.get("command", "")
        if "rm -rf" in cmd:
            return HookResult.deny(reason="禁止删除操作")
    return HookResult.allow()

# 方式 3：改写工具输出
@agent.on("post_tool_call")
async def mask_secrets(event: PostToolCallEvent) -> HookResult:
    result = event.result
    result = result.replace("sk-ant-", "sk-ant-***")
    return HookResult.modify({"result": result})

# 注销
agent.unhook("pre_tool_call", audit_tool)
```

**优先级**：`priority` 数字越小越先执行（默认 0），相同优先级按注册顺序执行。

**PreToolCallEvent 字段：**

```python
class PreToolCallEvent:
    tool_name: str
    tool_input: MappingProxyType   # 只读，不可原地修改
    call_id: str
```

**PreProviderCallEvent 字段：**

```python
class PreProviderCallEvent:
    system: str
    messages: tuple    # 只读 tuple
    tools: tuple
```

---

## 11. 事件系统

### EventBus

```python
from neoagent.events import EventBus, ToolCallEvent

bus = agent.event_bus

# 订阅单个事件类型
def on_tool_call(event: ToolCallEvent) -> None:
    print(f"工具调用：{event.name}，参数：{dict(event.input_data)}")

bus.subscribe(ToolCallEvent, on_tool_call)
bus.unsubscribe(ToolCallEvent, on_tool_call)

# 订阅所有事件（debug 用）
bus.subscribe_all(lambda e: print(type(e).__name__, e))
```

事件 handler 是**同步函数**（EventBus 同步广播）。Handler 抛异常时会记录日志但不中断其他 handler。

### 所有事件类型

| 事件类 | 触发时机 | 关键字段 |
|-------|---------|---------|
| `ProviderRequestEvent` | Provider 调用前 | `system, messages, tools, turn` |
| `ProviderResponseEvent` | Provider 返回后 | `content, stop_reason, input_tokens, output_tokens, turn` |
| `ToolCallEvent` | 工具被调用时 | `name, input_data, call_id` |
| `ToolResultEvent` | 工具返回后 | `name, call_id, output, is_error` |
| `CompressCheckEvent` | 每轮压缩检查后 | `msg_tokens, tool_tokens, budget, should_compress` |
| `CompressDoneEvent` | LLM 压缩完成 | `summary, previous_summary` |
| `CompressFallbackEvent` | 压缩回退截断 | `reason` |
| `MemoryExtractEvent` | 记忆提取尝试后 | `triggered, tool_calls, token_delta, items_stored` |
| `SkillChangeEvent` | Skill 激活/停用 | `name, active` |
| `TurnCompleteEvent` | 每轮结束 | `turn_index, stop_reason, tool_call_count` |
| `WorkerEvent` | Worker 事件冒泡到 Orchestrator | `worker_name, task_id, depth, inner` |
| `TaskDispatchEvent` | 任务派发给 Worker | `task_id, worker_name, instruction, depth` |
| `TaskCompleteEvent` | Worker 完成任务 | `task_id, worker_name, status, turns_completed, usage` |

### Observer 可观测性

Observer 通过 `EventBus` 订阅所有事件，实现双输出：

- **控制台**：ANSI 彩色，内容截断（500 字符）
- **日志文件**：完整内容，按时间命名

```python
# 启用日志
observer = agent.enable_logging(
    log_dir=Path("./logs"),  # None 时默认 ./logs
    console=True,
)
# observer.log_path -> Path  当前日志文件路径

# 停用日志
agent.disable_logging()

# 也可手动关闭
observer.close()
```

---

## 12. MCP 集成

MCP（Model Context Protocol）让 agent 通过标准协议接入外部工具服务器。

### 连接 MCP 服务器

```python
import asyncio
from neoagent.agent import NeoAgent
from neoagent.config import NeoAgentConfig

async def main():
    config = NeoAgentConfig(api_key="...", auto_approve_tools=True)
    agent = NeoAgent(config)

    # 连接 MCP 服务器（stdio 传输）
    await agent.add_mcp_server(
        name="github",
        command=["npx", "@anthropic/mcp-server-github"],
        env={"GITHUB_TOKEN": "ghp_..."},  # 额外环境变量
    )

    reply = await agent.chat("列出我的所有 GitHub repository")
    print(reply)
    await agent.close()

asyncio.run(main())
```

**`add_mcp_server()` 内部流程：**
1. 启动 `StdioTransport` 子进程
2. 执行 MCP initialize 握手
3. 获取工具列表，注册到 `ToolRegistry` 和 `DeferredToolRegistry`
4. 首次连接时自动注册 `tool_search` 内置工具

### DeferredToolRegistry：延迟加载

MCP 服务器可能提供几十个工具，全部暴露给 LLM 会消耗大量 token。延迟加载机制：

- 所有 MCP 工具默认**不暴露**给 LLM（deferred 状态）
- system prompt 中注入工具名列表（`<deferred-tools>` 块）
- LLM 可用 `tool_search` 工具按需提升（promote）工具

### ToolSearchTool：按需提升

`tool_search` 的查询语法：

| 查询形式 | 含义 |
|--------|------|
| `select:create_issue,list_repos` | 精确名称匹配 |
| `+github keyword` | 名称含 github 关键词 |
| `issue` | 正则搜索名称+描述 |

工具提升后在当前 session 内持续可见（写入 `session_state.promoted_tools`）。

### 断开连接

```python
await agent.remove_mcp_server("github")
agent.list_mcp_servers()  # -> list[str]
```

---

## 13. 多智能体

Orchestrator 协调多个独立 NeoAgent 实例（workers）完成复杂任务。

### 核心组件

```python
from neoagent.multi.orchestrator import Orchestrator
from neoagent.multi.worker import WorkerCard

class Orchestrator:
    def __init__(
        self,
        config: NeoAgentConfig,
        max_depth: int = 2,              # worker 嵌套最大深度
        max_concurrent_workers: int = 5, # 最大并发 worker 数
    ) -> None: ...

    def register_worker(self, card: WorkerCard) -> None: ...
    def load_workers(self, directory: str | Path) -> None: ...  # 从目录加载 .md 文件
    def register_tool(self, tool: BaseTool) -> None: ...        # 加入工具池

    async def run(self, message: str) -> str: ...               # 执行
    async def close(self) -> None: ...

    # 支持 async context manager
    async def __aenter__(self) -> Orchestrator: ...
    async def __aexit__(self, *_) -> None: ...
```

### WorkerCard：Worker 定义

WorkerCard 可以用 Python 代码或 `.md` 文件定义：

**Python 方式：**

```python
from neoagent.multi.worker import WorkerCard

card = WorkerCard(
    name="code_reviewer",
    description="专注于代码安全审查和最佳实践",
    instruction="你是一名资深代码审查工程师，重点检查安全漏洞、SQL 注入、XSS 等问题。",
    tags=("security", "review"),
    model=None,          # None = 继承 Orchestrator 的模型
    tools=("bash", "read"),  # 授权使用的工具名
)
```

**Markdown 文件方式（YAML frontmatter）：**

```markdown
---
name: code_reviewer
description: 专注于代码安全审查和最佳实践
tags: [security, review]
model: claude-haiku-4-20250514  # 可选，指定更轻量的模型
tools: [bash, read]
---

你是一名资深代码审查工程师。
审查代码时重点关注：
- SQL 注入漏洞
- XSS 攻击面
- 权限边界问题
```

从目录加载：`orchestrator.load_workers("./workers/")`

### Task / TaskResult

```python
from neoagent.multi.task import Task, TaskResult, TokenUsage

# 任务信封
class Task:
    task_id: str
    instruction: str
    context: tuple[str, ...]   # 额外上下文（JSON 或纯文本）
    max_turns: int = 20
    timeout: int = 1800        # 秒，默认 30 分钟
    max_output_tokens: int = 2000

    @classmethod
    def create(cls, instruction: str, context: tuple[str, ...] = (), ...) -> Task: ...

# 任务结果
class TaskResult:
    task_id: str
    status: Literal["completed", "failed", "cancelled"]
    output: str | None
    error: str | None
    usage: TokenUsage | None
    work_summary: str | None    # 工具调用摘要 + 最后输出
    turns_completed: int

    @property
    def is_success(self) -> bool: ...
```

### 5 个内置工具

Orchestrator brain 自动注册以下工具（LLM 用于协调）：

| 工具名 | 作用 |
|-------|-----|
| `spawn_worker` | 创建并启动一个 worker 执行任务 |
| `delegate_task` | 派发任务给已存在的 worker |
| `cancel_task` | 取消正在执行的任务 |
| `list_workers` | 列出所有已注册的 worker |
| `list_tasks` | 列出所有任务及状态 |

### 完整示例

```python
import asyncio
from pathlib import Path
from neoagent.multi.orchestrator import Orchestrator
from neoagent.multi.worker import WorkerCard
from neoagent.config import NeoAgentConfig
from neoagent.tools.builtin.read import ReadTool
from neoagent.tools.builtin.bash import BashTool


async def main():
    config = NeoAgentConfig(
        api_key="sk-ant-...",
        auto_approve_tools=True,
        system_prompt="你是一名工程任务协调员，负责将复杂任务分解并分发给专业 worker。",
    )

    async with Orchestrator(config, max_concurrent_workers=3) as orch:
        # 注册工具到全局工具池（workers 按授权使用）
        orch.register_tool(ReadTool())
        orch.register_tool(BashTool())

        # 注册 workers
        orch.register_worker(WorkerCard(
            name="file_reader",
            description="读取和分析文件内容",
            instruction="你专门负责读取文件并提取关键信息。",
            tags=("file", "read"),
            model=None,
            tools=("read",),
        ))
        orch.register_worker(WorkerCard(
            name="shell_runner",
            description="执行 shell 命令并返回结果",
            instruction="你专门负责执行 shell 命令并汇报结果。",
            tags=("shell", "exec"),
            model=None,
            tools=("bash",),
        ))

        result = await orch.run(
            "请先读取 README.md 的内容，然后运行 'python --version' 并综合两个结果给我一个报告"
        )
        print(result)

asyncio.run(main())
```

---

## 14. Channel 系统

Channel 将 NeoAgent 暴露为外部可调用的服务端点。

### Channel ABC

```python
from neoagent.channels.base import Channel

class Channel(ABC):
    def __init__(self, agent: NeoAgent) -> None: ...

    @abstractmethod
    async def start(self) -> None: ...          # 初始化（绑定端口等）

    @abstractmethod
    async def stop(self) -> None: ...           # 优雅关闭

    @abstractmethod
    async def serve_forever(self) -> None: ...  # 阻塞直到 stop() 被调用
```

### FastAPIChannel

```python
from neoagent.channels.fastapi_channel import FastAPIChannel

class FastAPIChannel(Channel):
    def __init__(
        self,
        agent: NeoAgent,
        host: str = "0.0.0.0",
        port: int = 8000,
        streaming: bool = True,    # 是否开启 SSE 流式端点
    ) -> None: ...
```

**HTTP 端点：**

| 方法 | 路径 | 描述 |
|-----|------|-----|
| `GET` | `/v1/health` | 健康检查，返回 `{"status": "ok"}` |
| `POST` | `/v1/run` | 同步请求，返回完整 ConversationResult |
| `POST` | `/v1/run/stream` | SSE 流式请求（`streaming=True` 时可用） |

**请求格式（`/v1/run`）：**

```json
{
  "messages": [
    {"role": "user", "content": "你好"}
  ],
  "max_turns": null
}
```

**SSE 事件格式（`/v1/run/stream`）：**

```
data: {"type": "tool_call", "name": "bash", "call_id": "...", "input": {...}}
data: {"type": "tool_result", "name": "bash", "call_id": "...", "output": "...", "is_error": false}
data: {"type": "turn_complete", "turn_index": 0, "stop_reason": "tool_use", "tool_call_count": 1}
data: {"type": "response", "stop_reason": "end_turn", "input_tokens": 100, "output_tokens": 50, "turn": 1}
data: {"type": "done", "reason": "completed"}
```

每个请求创建**独立的临时 session**（无状态设计），并发请求互不干扰。

### 安装依赖

```bash
pip install neoagent[fastapi]
```

### 完整示例：启动 HTTP 服务

```python
import asyncio
from neoagent.agent import NeoAgent
from neoagent.config import NeoAgentConfig
from neoagent.channels.fastapi_channel import FastAPIChannel
from neoagent.tools.builtin.bash import BashTool


async def main():
    config = NeoAgentConfig(
        api_key="sk-ant-...",
        auto_approve_tools=True,
    )
    agent = NeoAgent(config)
    agent.register_tool(BashTool())

    channel = FastAPIChannel(
        agent=agent,
        host="0.0.0.0",
        port=8000,
        streaming=True,
    )
    await channel.serve_forever()   # serve_forever() 会自动调用 start()


asyncio.run(main())
```

**客户端调用：**

```bash
# 同步调用
curl -X POST http://localhost:8000/v1/run \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "运行 echo hello"}]}'

# SSE 流式调用
curl -X POST http://localhost:8000/v1/run/stream \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "运行 echo hello"}]}'
```

**在测试或嵌入时直接获取 ASGI app：**

```python
from httpx import AsyncClient, ASGITransport
app = channel.app
async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
    resp = await client.get("/v1/health")
```

---

## 15. 配置参考

### NeoAgentConfig 所有字段

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

@dataclass
class NeoAgentConfig:
    api_key: str                                           # 必填，API Key
    model: str = "claude-sonnet-4-20250514"                # 模型名
    provider: Literal["anthropic", "openai"] = "anthropic" # LLM 供应商
    base_url: str | None = None                            # 自定义 API Base URL
    max_turns: int = 30                                    # 最大对话轮数
    context_budget: int = 0                                # 上下文预算（0=自动）
    max_result_size: int = 50000                           # 工具结果最大字符数
    auto_approve_tools: bool = False                       # 自动批准 "ask" 工具
    memory_dir: Path | None = None                         # 记忆存储目录（暂未自动使用，用 enable_memory()）
    memory_project_key: str | None = None                  # 记忆项目 key
    session_dir: Path | None = None                        # Session 持久化目录
    system_prompt: str | None = None                       # 自定义 system prompt
```

### Provider 配置对比

| 字段 | Anthropic 默认 | OpenAI 默认 |
|-----|--------------|------------|
| `model` | `claude-sonnet-4-20250514` | `gpt-4o` |
| `max_tokens` | 8192（内部） | 4096（内部） |
| `context_window` | 200,000 | 128,000（gpt-4o） |
| API Key 环境变量 | `ANTHROPIC_API_KEY` | `OPENAI_API_KEY` |

**使用 OpenAI：**

```python
config = NeoAgentConfig(
    api_key="sk-...",
    provider="openai",
    model="gpt-4o",
)
```

**使用兼容 OpenAI 接口的第三方服务（如 DeepSeek）：**

```python
config = NeoAgentConfig(
    api_key="sk-...",
    provider="openai",
    model="deepseek-chat",
    base_url="https://api.deepseek.com/v1",
)
```

---

## 16. 架构图

### 模块依赖关系

```
NeoAgent (agent.py)
├── NeoAgentConfig (config.py)
├── Provider: AnthropicProvider / OpenAIProvider (providers/)
│     └── Provider ABC (providers/base.py)
├── ToolRegistry (tools/registry.py)
│     └── BaseTool ABC (tools/base.py)
│           └── 内置工具: read/write/edit/bash/grep/glob (tools/builtin/)
├── ToolExecutor (tools/executor.py)
│     ├── PermissionChecker (tools/permission.py)
│     └── HookManager (hooks.py)
├── PromptBuilder (core/prompt.py)
│     └── PromptSection
├── QueryLoop (core/loop.py)
│     ├── ContextCompressor (core/compress.py)
│     ├── MemoryManager (memory/manager.py)
│     │     ├── MemoryStore (memory/store.py)
│     │     ├── MemoryExtractor (memory/extractor.py)
│     │     └── MemoryRetriever (memory/retriever.py)
│     └── EventBus (events.py)
│           └── Observer (observe.py)
├── Session / JsonFileStorage (session.py)
├── DeferredToolRegistry (tools/deferred.py)
├── MCPClient + StdioTransport (mcp/)
└── Orchestrator (multi/orchestrator.py)
      ├── WorkerPool (multi/worker.py)
      └── TaskTracker (multi/task.py)

Channel (channels/base.py)
└── FastAPIChannel (channels/fastapi_channel.py)
```

### 数据流：用户消息到响应

```
1. 用户消息
        ↓
2. Channel（可选）: FastAPIChannel 接收 HTTP 请求
        ↓
3. NeoAgent.chat() / NeoAgent.run()
        ↓
4. Session 创建 / 消息追加
        ↓
5. QueryLoop.run(session)
   ├─ 5a. ContextCompressor.should_compress()
   │       → 超过阈值：LLM 摘要 → 更新 session.messages
   ├─ 5b. PromptBuilder.build() → system prompt
   ├─ 5c. HookManager.run_pre("pre_provider_call")
   ├─ 5d. Provider.create() → Response
   ├─ 5e. HookManager.run_post("post_provider_call")
   │
   ├─ [stop_reason == "tool_use"]
   │   ├─ 5f. HookManager.run_pre("pre_tool_call")
   │   ├─ 5g. ToolExecutor.execute(tool_calls) → ToolResult[]
   │   │       （is_concurrent_safe 工具并发，其他串行）
   │   ├─ 5h. HookManager.run_post("post_tool_call")
   │   └─ 5i. 追加 assistant + tool_result 消息，继续循环
   │
   └─ [stop_reason == "end_turn"]
       ├─ 5j. MemoryManager.maybe_extract() → 持久化记忆
       └─ 5k. 返回 ConversationResult
        ↓
6. EventBus 全程广播事件 → Observer 记录日志
        ↓
7. NeoAgent.chat() 提取最后一轮文本 → 返回 str
        ↓
8. Session.save()（若配置了 storage）
        ↓
9. Channel 将响应序列化返回给用户
```

---

*本文档基于 neoagent v3.2c 源码生成，所有 API 签名均来自实际代码。*
