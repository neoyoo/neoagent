# neoagent/v2/strategies/_memory_review_prompt.py
"""Memory review prompt template for OneShotMemoryReviewStrategy.

Spec refs:
  § 11.3  (lines 1899-1964) — OneShotMemoryReviewStrategy implementation details
"""

MEMORY_REVIEW_PROMPT_TEMPLATE = """\
你是 memory reviewer，负责从最近的对话中提炼对用户长期有用的 memory entries。
你的任务是分析用户对话和当前工作记忆，输出一组新增或需强化的 memory 条目。

═══════════════════════════════════════════════════════════════
输入上下文
═══════════════════════════════════════════════════════════════

【用户 ID（user_id）】
{user_id}

【最近对话消息（messages）】
{messages_text}

【当前工作记忆（working_memory）】
{working_memory_json}

═══════════════════════════════════════════════════════════════
输出 JSON Schema（严格格式）
═══════════════════════════════════════════════════════════════

输出必须是一个 JSON 数组（list），每个元素结构如下：

```json
[
  {{
    "type": "preference" | "fact" | "goal" | "constraint" | "habit",
    "category": "<可选，如 'travel' / 'code-style' / null>",
    "content": "<自然语言描述 memory 的完整内容>",
    "confidence": 0.0,
    "evidence": {{"msg_ids": ["m3", "m5"]}}
  }}
]
```

如果对话中没有值得记录的内容，请直接返回空数组：[]

═══════════════════════════════════════════════════════════════
合法示例
═══════════════════════════════════════════════════════════════

```json
[
  {{
    "type": "preference",
    "category": "travel",
    "content": "用户偏好坐靠窗座位",
    "confidence": 0.9,
    "evidence": {{"msg_ids": ["m1"]}}
  }},
  {{
    "type": "fact",
    "category": "finance",
    "content": "用户旅行预算约为 3000 美元",
    "confidence": 0.75,
    "evidence": {{"msg_ids": ["m3", "m5"]}}
  }},
  {{
    "type": "goal",
    "category": null,
    "content": "用户希望在樱花季节赴日旅行",
    "confidence": 0.85,
    "evidence": null
  }}
]
```

═══════════════════════════════════════════════════════════════
输出规则（必须严格遵守）
═══════════════════════════════════════════════════════════════

规则 1：type 字段只能是以下值之一
  → "preference" | "fact" | "goal" | "constraint" | "habit"
  → 不得使用其他任何字符串

规则 2：只输出新增或需强化的 memory
  → 不要重复当前工作记忆（working_memory）中已经明确记录的内容
  → 只有从对话中发现的新信息才值得输出

规则 3：confidence 必须在 [0.0, 1.0] 范围内
  → 0.0 = 极不确定，1.0 = 非常确定
  → 从对话中明确陈述的信息 confidence 可以较高，隐含推断的较低

规则 4：content 必须是非空字符串
  → 使用自然语言完整描述这条 memory 的内容
  → 不得为 null 或空字符串

规则 5：保守策略
  → 如果对话中没有明显可学的长期信息，请返回空数组 []
  → 不要强行输出低质量的 memory 条目

规则 6：不得输出 JSON 数组以外的任何文字
  → 响应必须是纯 JSON 数组，不得包含 Markdown 代码块标记、解释文字等

规则 7：evidence 字段可选
  → 如果可以溯源到具体消息，填写 {{"msg_ids": [...]}}
  → 无法溯源时填写 null

═══════════════════════════════════════════════════════════════
输出要求
═══════════════════════════════════════════════════════════════

现在请直接输出符合上述规则的纯 JSON 数组，不要有任何其他文字。
如无可学的内容，返回 []。
"""
