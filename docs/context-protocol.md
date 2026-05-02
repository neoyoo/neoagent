# neoagent 上下文协议（Context Protocol）

> spec § 15.1 + decision G — 7 层扁平架构
>
> 本文记录 neoagent 每次调用 LLM 时实际发出的上下文结构，是开发、调试、扩展的第一手参考。

---

## 1. 架构总览

```
┌─────────────────────────────────────────────────────────┐
│                     LLM API 请求                         │
│                                                         │
│  system: <静态层 1-4> + "\n\n" + <临时层 5-7>            │
│  messages: [ {role, content}, ... ]                     │
│  tools: [ {name, description, input_schema}, ... ]      │
└─────────────────────────────────────────────────────────┘
```

### 7 层分类

| 层 | 枚举值 | 类型 | 构建时机 | 渲染方式 |
|----|--------|------|---------|---------|
| 1 | `IDENTITY` | 静态 | 启动时 | 纯文本 |
| 2 | `PERSISTENT_MEMORY` | 静态 | 启动时 | 纯文本 |
| 3 | `CAPABILITIES` | 静态 | skill 激活/停用时 | 纯文本 |
| 4 | `SECURITY` | 静态 | 启动时 | 纯文本 |
| 5 | `WORKING_MEMORY` | 临时 | 每轮重建 | XML 块 |
| 6 | `COMPRESSED_HISTORY` | 临时 | 每轮重建 | XML 块 |
| 7 | `MEMORY_CONTEXT` | 临时 | 每轮重建 | XML 块 |

**静态层**由 `LayeredPromptBuilder.build_system_prompt()` 构建，按 `priority` 降序拼接。  
**临时层**由 `LayeredPromptBuilder.build_ephemeral()` 每轮追加到 system 末尾。

---

## 2. system 字段完整示例

以下是一次典型对话第 5 轮时 LLM 实际收到的 `system` 内容：

```
# Layer 1 — IDENTITY
You are neoagent, a capable AI assistant with access to tools.
You can read files, run shell commands, search code, and help with complex tasks.
Always think step-by-step before acting.

# Layer 2 — PERSISTENT_MEMORY
User preferences:
- Prefer concise responses
- Always use async/await in Python code
- Project root: /Users/neo/project

# Layer 3 — CAPABILITIES
## Available Skills

### code-search
Search and navigate codebases efficiently.
Use Glob to list files, Grep to search content, Read to inspect files.

### bash-execution
Execute shell commands via the Bash tool.
Use for: running tests, checking git status, listing directories.

# Layer 4 — SECURITY
## Security Constraints
- Do not execute commands that modify system files outside the project directory
- Do not expose API keys or secrets in outputs
- Confirm before running destructive operations (rm, reset --hard, DROP TABLE)

<working_memory version="3" at_turn="5">
  CONSTRAINTS_AND_PREFERENCES:
    - c01: 使用 pytest，TDD 优先
    - c02: async/await 核心链路，禁止同步阻塞

  PROGRESS:
    已完成 HookManager 基础框架，正在实现 pre_tool_call 拦截点

  KEY_DECISIONS:
    - d01: Hook payload 使用 frozen dataclass，禁止在 hook 内修改
    - d02: 选择 4 个拦截点：pre/post tool_call + pre/post provider_call

  RELEVANT_FILES:
    - f01: neoagent/hooks.py
    - f02: neoagent/core/loop.py
    - f03: tests/test_hooks.py

  NEXT_STEPS:
    - n01: 实现 post_tool_call 拦截点
    - n02: 补充 HookManager 的 async hook 支持测试

  CRITICAL_CONTEXT:
    当前任务：为 neoagent v3.2a 实现完整的 Hooks 拦截系统
</working_memory>

<compressed_history>
  <batch id="cm_1" turns="1-3">
    <recoverable>
      <turn n="1">
        <msg id="u1" role="user" preview="请帮我设计 neoagent 的 Hook 系统" />
        <msg id="a1" role="assistant" preview="好的，我来设计 Hook 拦截架构..." />
      </turn>
      <turn n="2">
        <msg id="u2" role="user" preview="先实现 HookManager 基础类" />
        <msg id="t1" role="tool" preview="Read hooks.py → 文件不存在" />
        <msg id="a2" role="assistant" preview="创建 hooks.py，定义 HookManager..." />
      </turn>
      <turn n="3">
        <msg id="u3" role="user" preview="加上 pre_tool_call 拦截点" />
        <msg id="t2" role="tool" preview="Write hooks.py → 成功写入 45 行" />
        <msg id="a3" role="assistant" preview="pre_tool_call 已实现，payload 为 frozen..." />
      </turn>
    </recoverable>
  </batch>
</compressed_history>

<memory-context>
  <entry type="preference" confidence="0.9">用户偏好 TDD，每个功能先写测试再实现</entry>
  <entry type="fact" confidence="0.85">neoagent 项目路径：/Users/neo/Desktop/project/git/neoagent</entry>
  <entry type="goal" confidence="0.95">当前目标是完成 v3.2a Hooks + MCP 集成，保持 500+ tests 绿色</entry>
</memory-context>
```

---

## 3. messages 数组结构

`messages` 是标准 Anthropic/OpenAI 格式，包含当前轮（未压缩）的对话历史：

```json
[
  {
    "role": "user",
    "content": "现在实现 post_tool_call 拦截点"
  },
  {
    "role": "assistant",
    "content": [
      {
        "type": "text",
        "text": "好的，我来实现 post_tool_call。"
      },
      {
        "type": "tool_use",
        "id": "toolu_01",
        "name": "Read",
        "input": { "file_path": "/Users/neo/.../hooks.py" }
      }
    ]
  },
  {
    "role": "user",
    "content": [
      {
        "type": "tool_result",
        "tool_use_id": "toolu_01",
        "content": "class HookManager:\n    ..."
      }
    ]
  }
]
```

### tool_result 被 free 后的形态

当 `free_tool_result` 工具被调用后，该 tool_result 的 content 替换为占位符，原始内容存入 session：

```json
{
  "type": "tool_result",
  "tool_use_id": "toolu_01",
  "content": "[FREED:toolu_01 — use recall_tool_result to restore]"
}
```

---

## 4. 各层详细规范

### Layer 1 — IDENTITY（身份）

```
priority: 高（建议 100）
is_static: true
内容: 角色定义 + 核心能力说明 + 行为原则
```

注册方式：
```python
from neoagent.v2.schema import Layer

agent.prompt_builder.register_layer_section(
    Layer.IDENTITY,
    PromptSection(
        name="identity",
        content="You are neoagent, a capable AI assistant...",
        priority=100,
    )
)
```

---

### Layer 2 — PERSISTENT_MEMORY（持久记忆）

```
priority: 中（建议 80）
is_static: true
内容: 跨会话的用户偏好、项目背景、长期事实
来源: MemoryManager 从 MemoryStore 检索后注入
```

---

### Layer 3 — CAPABILITIES（能力/技能）

```
priority: 中（建议 60）
is_static: true（但可动态激活/停用）
内容: 当前激活的 skill 说明
```

skill 生命周期：
```python
# 注册（不激活）
agent.prompt_builder.register_skill("code-search", section)

# 激活（加入 system prompt）
agent.prompt_builder.activate_skill("code-search")

# 停用（从 system prompt 移除，但保持注册）
agent.prompt_builder.deactivate_skill("code-search")
```

---

### Layer 4 — SECURITY（安全约束）

```
priority: 低（建议 40）
is_static: true
内容: 禁止行为清单、确认策略、沙箱边界
来源: neoagent/v2/prompts/security_blocks.py
```

当 `config.enable_security_prompt_blocks = True`（默认）时自动注入。

---

### Layer 5 — WORKING_MEMORY（工作记忆，XML）

每轮由 `LayeredPromptBuilder.build_ephemeral()` 生成，对应 `WorkingMemory` dataclass：

```xml
<working_memory version="{wm.version}" at_turn="{wm.at_turn}">
  CONSTRAINTS_AND_PREFERENCES:
    - c01: ...
    - c02: ...

  PROGRESS:
    {wm.progress}

  KEY_DECISIONS:
    - d01: ...

  RELEVANT_FILES:
    - f01: ...

  NEXT_STEPS:
    - n01: ...

  CRITICAL_CONTEXT:
    {wm.critical_context}
</working_memory>
```

LLM 可通过 `update_working_memory` 工具更新任意字段。

---

### Layer 6 — COMPRESSED_HISTORY（压缩历史，XML）

messages 数组超过 `context_budget × 70%` 时触发压缩，历史批次以 XML 注入：

```xml
<compressed_history>
  <batch id="cm_1" turns="1-3">
    <recoverable>
      <turn n="1">
        <msg id="u1" role="user" preview="用户消息摘要（20-40字）" />
        <msg id="a1" role="assistant" preview="助手响应摘要（20-40字）" />
      </turn>
      <turn n="2">
        <msg id="u2" role="user" preview="..." />
        <msg id="t1" role="tool" preview="工具调用摘要" />
        <msg id="a2" role="assistant" preview="..." />
      </turn>
    </recoverable>
  </batch>
  <batch id="cm_2" turns="4-6">
    ...
  </batch>
</compressed_history>
```

LLM 可通过 `recall_turn` 工具按 turn id 恢复原始消息。

---

### Layer 7 — MEMORY_CONTEXT（记忆检索，XML）

从 `MemoryStore` 检索到与当前对话相关的记忆条目后注入：

```xml
<memory-context>
  <entry type="preference" confidence="0.9">内容</entry>
  <entry type="fact" confidence="0.85">内容</entry>
  <entry type="goal" confidence="0.95">内容</entry>
</memory-context>
```

`type` 取值：`preference` / `fact` / `goal` / `event` / `skill`  
`confidence` 范围：0.0 ~ 1.0

---

## 5. 上下文组装流程（loop.py）

```
每轮 LLM 调用前：

1. msgs_for_llm = 截断后的 messages（超出 budget 的旧消息已压缩）
2. system = prompt_builder.build()          ← 静态层 1-4
3. ephemeral = layered_prompt_builder
               .build_ephemeral(wm, batches) ← 临时层 5-7
4. if ephemeral:
       system = system + "\n\n" + ephemeral
5. 临时 section（_deferred, _freed）按需注入，调用后清理
6. → provider.call(system, messages, tools)
```

---

## 6. 扩展指引

### 添加新的静态层内容

```python
from neoagent.core.prompt import PromptSection
from neoagent.v2.schema import Layer

agent.prompt_builder.register_layer_section(
    Layer.IDENTITY,
    PromptSection(
        name="project-context",
        content="Current project: neoagent v3.2b\nFocus: Multi-Agent orchestration.",
        priority=90,  # 低于 identity(100)，高于其他 IDENTITY 内容
    )
)
```

### 动态更新工作记忆

LLM 通过 `update_working_memory` 工具调用，框架执行后更新 `session.state._current_wm`，下一轮自动反映在 Layer 5 中。

### 检查某轮实际发出的上下文

订阅 `ProviderRequestEvent`：

```python
from neoagent.events import ProviderRequestEvent

def on_request(event: ProviderRequestEvent):
    print("=== SYSTEM ===")
    print(event.system)
    print("=== MESSAGES ===")
    for m in event.messages:
        print(m)

agent.event_bus.subscribe(ProviderRequestEvent, on_request)
```
