---
title: "规避 DeepSeek Anthropic 端点 issue #1397"
date: 2026-07-06 14:29:39
updated: 2026-07-06 14:29:39
categories:
  - Claude Code
tags:
  - DeepSeek
  - Claude Code
  - Anthropic API
  - Reverse Proxy
description: Claude Code 接入 DeepSeek /anthropic 兼容端点后，子 Agent 请求因 thinking 与 reasoning_effort 互斥被返回 400。用一个本地反向代理条件性剥离 thinking 字段解决。
---

## 问题

Claude Code `2.1.166+` 接入 DeepSeek 的 `/anthropic` 兼容端点时，所有子 Agent 请求(`/deep-research`、Workflow 等)返回：

```
HTTP 400  thinking options type cannot be disabled when reasoning_effort is set
```

成因：

- Claude Code 对子 Agent 请求硬编码 `thinking:{type:"disabled"}`。
- `CLAUDE_CODE_EFFORT_LEVEL=max` 经 `anthropic-beta` 请求头中的 `effort-2025-11-24` capability flag(非 request body 字段)启用 reasoning effort。
- DeepSeek 兼容层判定二者互斥，拒绝请求。

参考：[deepseek-ai/DeepSeek-V3#1397](https://github.com/deepseek-ai/DeepSeek-V3/issues/1397)。

<!-- more -->

## 解决方案

在 `claude` 与上游之间插入本地反向代理：当请求满足「`anthropic-beta` 含 `effort-` flag 且 body `thinking.type=="disabled"`」时，剥离 `thinking` 字段后透传，其余请求原样转发。代理与 `claude` 进程共生命周期。

## 实施

### 代理 `/root/.claude/deepseek-proxy.py`

纯标准库实现，要点：

```python
def _should_strip(body_bytes: bytes, effort_active: bool) -> bytes:
    if not effort_active:
        return body_bytes
    data = json.loads(body_bytes)
    thinking = data.get("thinking")
    if isinstance(thinking, dict) and thinking.get("type") == "disabled":
        data.pop("thinking", None)
        return json.dumps(data).encode("utf-8")
    return body_bytes

# 转发前从请求头提取 effort 信号
effort_active = "effort-" in self.headers.get("anthropic-beta", "")
body = _should_strip(raw_body, effort_active) if method == "POST" and raw_body else raw_body
```

实现透传层需注意:

- 转发路径拼接 `/anthropic` 前缀。
- 用 `http.client.HTTPSConnection` 逐块 `read/write/flush`,支持 SSE 流式响应。
- 响应头注入 `Connection: close` 并置 `close_connection=True`,避免 keep-alive 阻塞。
- `allow_reuse_address/port=True`,支持端口快速复用。
- 读取上游响应的循环需捕获 `BrokenPipeError`、`ConnectionResetError`、`http.client.IncompleteRead`:客户端取消 in-flight 请求或上游流式响应中途断开属正常现象,静默忽略以避免刷堆栈。

### 启动脚本 `/root/env_config_ds.sh`

将代理生命周期内联:

```sh
DS_PROXY_PORT=18979
lsof -ti :${DS_PROXY_PORT} 2>/dev/null | xargs kill 2>/dev/null
sleep 0.2
python3 /root/.claude/deepseek-proxy.py &
DS_PROXY_PID=$!
sleep 0.3
kill -0 "$DS_PROXY_PID" 2>/dev/null || { echo "[ds-proxy] FATAL"; exit 1; }
trap 'kill "$DS_PROXY_PID" 2>/dev/null; wait "$DS_PROXY_PID" 2>/dev/null' EXIT

export CLAUDE_CODE_EFFORT_LEVEL=max
export ANTHROPIC_BASE_URL="http://127.0.0.1:${DS_PROXY_PORT}"
# ... 其余环境变量 ...
claude --dangerously-skip-permissions --disallowedTools WebSearch
```

请求路径:

```
Claude Code → 127.0.0.1:18979 (条件性 strip thinking) → api.deepseek.com/anthropic
```
