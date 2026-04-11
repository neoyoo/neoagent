# neoagent Cross Review

日期: 2026-04-11

## 严重（必须修复 — 安全、数据丢失或崩溃风险）

- `ask` 工具在默认配置下等于“自动批准”，这直接把 `bash`/`write`/`edit` 变成了默认可执行能力。见 `neoagent/tools/permission.py:16`、`neoagent/agent.py:25`。这不是“集成方需知”，而是库默认安全边界失守。任何忘记注入 `ask_callback` 的调用方，都会在无交互模式下放行危险工具。建议默认 `auto_approve=False`，并把非交互自动批准改成显式 opt-in。

```python
class PermissionMode(str, Enum):
    AUTO = "auto"
    ASK = "ask"
    DENY = "deny"

class PermissionChecker:
    def __init__(..., auto_approve: bool = False): ...
```

- `BashTool` 仍然本质上是无沙箱的远程 shell 执行器，当前 regex blocklist 很容易绕过。见 `neoagent/tools/builtin/bash.py:36`、`neoagent/tools/builtin/bash.py:44`。例如 `python -c`, `perl -e`, `find . -delete`, `sh -c`, 变量拼接、`eval`、重定向、网络外传都不在规则里；而且没有 `cwd` 限制、没有环境隔离、没有资源限制。结合上一条，这是当前最大的残留漏洞。建议不要继续扩 blocklist，改成能力收敛：

```python
proc = await asyncio.create_subprocess_exec(
    "bash", "-lc", cmd,
    cwd=str(workspace_root),
    env=safe_env,
    stdout=PIPE, stderr=STDOUT,
)
```

同时至少增加：固定工作目录、可选禁网、最大输出字节数、命令 allowlist/runner 抽象。

- 文件工具存在典型 TOCTOU 问题：`validate_path()` 校验的是 resolved path，但真正打开/写入的是原始输入路径。见 `neoagent/tools/builtin/read.py:29`、`neoagent/tools/builtin/write.py:28`、`neoagent/tools/builtin/edit.py:29`、`neoagent/tools/builtin/grep.py:30`、`neoagent/tools/builtin/glob.py:28`。如果校验后 symlink 被替换，后续操作可能落到允许目录外。建议所有工具都使用 `validate_path()` 的返回值继续操作。

```python
path = validate_path(input.file_path, self._allowed)
content = path.read_text(encoding="utf-8")
```

- `MemoryStore` 信任 `MEMORY.md` 中的文件名，允许通过篡改索引读取/删除 memory 目录外文件。见 `neoagent/memory/store.py:37`、`neoagent/memory/store.py:46`、`neoagent/memory/store.py:62`。`MemoryExtractor` 的文件名清洗只覆盖“写入路径”，没有覆盖“读取已有索引”。一旦 `MEMORY.md` 被手工或恶意写入 `../secret.txt`，`retrieve()` 会把外部文件内容注入 prompt。建议在 `MemoryStore` 层统一做 basename 校验/resolve containment，别把安全性交给上游。

```python
def _topic_path(self, filename: str) -> Path:
    path = (self._dir / filename).resolve()
    if not path.is_relative_to(self._dir.resolve()):
        raise ValueError("invalid memory filename")
    return path
```

## 重要（应该修复 — 正确性、健壮性或 API 质量）

- `QueryLoop`/`ContextCompressor`/`MemoryManager` 的状态跨 `run()` 持续存在，导致会话污染和潜在数据泄漏。见 `neoagent/core/loop.py:30`、`neoagent/core/compress.py:57`、`neoagent/memory/manager.py:22`、`neoagent/agent.py:44`。`chat()` 看起来是单轮无状态 API，但底层 summary 和 memory baseline 会沿用上一次运行结果。对多用户或多任务复用同一个 `NeoAgent` 实例时，这是隐性串话。建议把这些状态移到 `Session`/`RunContext`，或至少在 `run()` 开头重置。

- `NeoAgent` 通过直接改私有属性给 `QueryLoop` 热插 memory/logging，破坏了后续 v3 扩展的依赖边界。见 `neoagent/agent.py:82`、`neoagent/agent.py:97`。Multi-Agent、Hooks、Channel 一上来，这种做法会变成全局状态拼装器。建议让 `NeoAgent` 只负责构建 `SessionRuntime`，所有可选能力通过构造参数或 builder 注入。

- `enable_logging()` 重复调用会泄漏旧文件句柄。见 `neoagent/agent.py:96`、`neoagent/agent.py:97`。新 `Observer` 直接覆盖旧实例，没有先 `close()`。这是明确的资源管理缺陷。建议在替换前关闭旧 observer，并让 `enable_logging()` 幂等。

- `MemoryExtractor` API 契约是返回 `description`，但 `_rebuild_index()` 完全忽略它，改用文件首行重建索引。见 `neoagent/memory/extractor.py:105`、`neoagent/memory/extractor.py:122`。这会导致检索质量漂移，且接口语义不可信。建议把 description 持久化到 frontmatter 或 sidecar metadata，而不是丢弃。

- `ToolRegistry` 对并发安全工具是无上限 `gather()`，对大工具集会造成资源尖峰；同时结果截断发生在工具完成之后，超大输出仍会先进内存。见 `neoagent/tools/registry.py:35`、`neoagent/tools/registry.py:74`。50+ 工具或大 grep/bash 输出时，这会成为 v3 扩展瓶颈。建议加 semaphore 和 streaming/result cap。

- `OpenAIProvider` 把内部消息模型硬编码成单一 chat-completions 形态，混合内容处理也不完整。见 `neoagent/providers/openai.py:55`、`neoagent/providers/openai.py:82`、`neoagent/providers/openai.py:110`。`user` 消息里一旦同时有文本和 `ToolResultBlock`，文本会被丢掉；`tool_result.is_error` 也没有传递。v3 加 MCP/Channel 后，这种“内部块模型 <-> 特定供应商格式”直连会很难改。建议引入独立的 provider-normalized request/response translator。

- 观察器的 memory hooks 设计了但没接上，属于“承诺的可观测性不存在”。见 `neoagent/observe.py:197`、`neoagent/memory/manager.py:28`。这会误导调用方，也说明事件模型目前不是一等抽象。

## 代码质量（应该改进 — 工程规范、可维护性）

- `PromptBuilder.build()` 的排序规则是 `(not is_static, priority)`，这意味着任何 dynamic section 无论 priority 多高，都会排在所有 static section 后面。见 `neoagent/core/prompt.py:48`。当前做法和“priority”直觉不一致，后面加 hooks/channel 指令时很容易踩坑。建议改成显式 phase 枚举，例如 `SYSTEM_STATIC / CONTEXT / MEMORY / EPHEMERAL`，不要用布尔值暗编码。

- `BaseTool.execute()` 和 `Provider.create()` 类型过弱，导致全项目到处 `assert isinstance(...)` 和裸 `list`/`dict`。见 `neoagent/tools/base.py:15`、`neoagent/providers/base.py:23`。建议用 `Generic[InputT]` 和结构化请求对象替代。

- 广泛的 `except Exception` 让故障被压平为 `str(e)`，缺少分类、上下文和 telemetry。见 `neoagent/tools/registry.py:77`、`neoagent/core/compress.py:85`、`neoagent/memory/extractor.py:97`。建议至少区分 validation / permission / execution / provider failures，并在 observer 中记录 fallback 原因。

- `ReadTool` 整文件 `readlines()` 再切片，热路径上对大文件不友好。见 `neoagent/tools/builtin/read.py:33`。建议用流式读取或 `itertools.islice()`。

- `ContextCompressor.should_compress()` 和 observer 都会重复对整段消息重新编码，长对话下是明显的 O(total_history) 热点。见 `neoagent/core/loop.py:39`、`neoagent/core/compress.py:62`。建议在 `SessionState` 中缓存 token 估算并增量更新。

- `ToolResult.call_id` 的两段式初始化是坏味道。见 `neoagent/core/types.py:38`、`neoagent/tools/registry.py:73`。当前做法让每个 tool 都返回 `call_id=""`。建议让 tool 返回不含 call id 的 payload，registry 封装最终 `ToolResult`。

- 测试盲区最明显的几处：
  - 没有覆盖同一 `QueryLoop`/`NeoAgent` 连续 `run()` 的状态泄漏，见 `tests/core/test_loop.py`，之后全是单次运行。
  - 没有覆盖“工具必须使用 resolved path 而非原始 path”的 race/链接切换场景，见 `tests/tools/test_pathguard.py`，只测了 `validate_path()` 本身。
  - 没有覆盖 `enable_logging()`/`enable_memory()` 生命周期，见 `tests/test_agent.py`。
  - 没有覆盖 observer memory hooks 实际接线，见 `tests/test_observe.py`，只测了 `Observer` 独立行为。
  - 没有覆盖高并发工具执行和输出背压。

## 轻微（改了更好 — 一致性、美观）

- `pathguard.py` 里的 `_DEFAULT_ALLOWED` 未使用，是死代码。见 `neoagent/tools/pathguard.py:4`。

- `NeoAgentConfig` 导入了未使用的 `field`。见 `neoagent/config.py:2`。

- `compress()` 的 `context_budget` 参数没有实际使用。见 `neoagent/core/compress.py:76`。

- `Turn` 和 `ConversationResult` 使用空列表默认值，虽然 Pydantic 一般会处理，但风格上仍建议 `Field(default_factory=list)`。见 `neoagent/core/types.py:46`、`neoagent/core/types.py:52`。

## v3 架构建议

- 先引入 `Session` / `RunContext`。把 compressor summary、memory counters、observer、token cache 全部从 `QueryLoop` 实例状态移到 session 状态。`NeoAgent` 保持无状态工厂，否则 Multi-Agent 一定串状态。

- 把 `Observer` 升级成真正的事件总线。需要 `before_tool`, `after_tool`, `before_provider`, `after_provider`, `before_memory_extract` 等 typed events；logging 只是其中一个 subscriber。这样 Hooks 才能落地。

- 拆分 `ToolRegistry` 和 `ToolExecutor`。前者只管注册与 schema，后者负责权限、并发、限流、结果截断、审计。MCP tool、本地 tool、AgentTool 才能统一挂载。

- 建立 `Workspace` / `Capability` 抽象。文件访问、shell、network 都不要直接散落在 tool 内部。v3 的 Multi-Agent 和 MCP 都需要共享同一套能力边界。

- Provider 层改成结构化协议，而不是 `create(system, messages, tools, **kwargs)`。至少需要 capability flags、normalized content blocks、tool result semantics、streaming 预留位。否则 MCP/Channel 一接进来，provider 适配会越来越硬编码。

## 总结评价

- 整体评分：6.5/10

- 最优先的 5 件事：
  1. 把 `PermissionChecker.auto_approve` 默认改为 `False`。
  2. 收紧 `BashTool`，停止依赖 regex blocklist 作为主安全模型。
  3. 修复所有文件工具的 TOCTOU，统一使用 resolved path。
  4. 给 `MemoryStore` 增加目录 containment 校验，堵住篡改 `MEMORY.md` 的越界读取。
  5. 引入 `SessionState`，消除 `QueryLoop`/compressor/memory 的跨运行状态泄漏。

- 做得好的地方：
  - 模块划分清楚，v2 的 `core/tools/providers/memory` 分层是成立的。
  - `ToolRegistry` 的并发安全默认 fail-closed 思路是对的。
  - `ContextCompressor` 的工具对清洗和角色交替修复是认真做过故障复盘的。
  - 测试数量不少，而且 provider/tool/core 基础面覆盖已经成体系。
