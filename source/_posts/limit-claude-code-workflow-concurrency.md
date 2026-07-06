---
title: 如何限制 Claude Code 的 workflow 并发：一次从环境变量挖到 glibc syscall 的调试
date: 2026-06-08 12:00:00
updated: 2026-06-08 12:00:00
categories:
  - Claude Code
tags:
  - Claude Code
  - Workflow
  - Concurrency
  - LD_PRELOAD
  - Debugging
description: Claude Code 的 workflow 并发上限来自 os.cpus()。四个「显然有效」的办法全部失效，最终用 LD_PRELOAD 覆盖 sysconf 一击即中。
---

## 起因

我用 Claude Code 接了第三方模型（GLM / DeepSeek）跑 deep research。问题是:**它一启动 dynamic workflow 就会同时拉起远超我预期数量的 agent**,把后端打满。我试过设 `CLAUDE_CODE_MAX_TOOL_USE_CONCURRENCY`,没用。

于是有了这趟debug。最终结论很短,但中间踩的坑值得记下来——因为**直觉上"显然有效"的办法,实测全部失败**,只有最后一个能用。

<!-- more -->

## 先搞清楚:并发上限到底从哪来

Claude Code 现在是个 **245MB 的 Node SEA 独立 ELF**(自带运行时),不是普通 node 脚本。直接 `grep` 它的二进制:

```bash
F=/root/.local/share/claude/versions/2.1.168
grep -ao 'function Br5(H){return Math.min(16,Math.max(2,H-2))}' "$F"
grep -ao '.\{15\}Br5(rw4.cpus().length).\{15\}' "$F"
# => ...uire("os");pr5=Br5(rw4.cpus().length),Fr5=`Workflow ...
```

公式清清楚楚:

```
workflow 并发 = min(16, max(2, 核数 - 2))     // 核数 = os.cpus().length
```

注意三点:
1. 它取的是 **`os.cpus().length`**,跟 `CLAUDE_CODE_MAX_TOOL_USE_CONCURRENCY` 无关(那个管的是普通 Task 子代理,workflow 引擎根本不读)。
2. 有**下限 2**。
3. 我这台机器 **256 核** → `min(16, max(2, 254)) = 16`。所以 deep research 总是同时跑到 16 个。

目标因此变成:**让 claude 以为自己只有 6 个核**(→ 并发 4)。

## 关键:怎么验证「有没有生效」

这一步是整篇的核心方法论。**strace 看不出并行度**,光看二进制改没改也不算数。我搭了个**透明计数代理**:只把请求转发到真实端点,同时统计同一时刻在途的 `POST /messages` 数量,记录峰值。

```python
# count_proxy.py 核心逻辑
counted = method=="POST" and "/messages" in path and "count_tokens" not in path
if counted:
    with lock: cur += 1; peak = max(peak, cur)
... 转发 ...
finally:
    if counted:
        with lock: cur -= 1
```

然后用一个**强制触发 workflow** 的 prompt,让它 `parallel()` 拉 10 个最简单的 agent:

```
ultracode. You MUST call the Workflow tool now. Write a workflow whose body does:
await parallel(Array.from({length:10},(_,i)=>()=>agent('return done '+i))).
```

- 如果并发上限是 16 → 10 个全并发 → **峰值 11**(10 agent + 主循环)。
- 如果上限被压到 4 → 同时最多 4 个 → **峰值 ~5**。

峰值 11 vs 5,一目了然。**基线实测:峰值 11。** 有了这把尺子,下面每个方案都能客观判定。

## 踩坑实录:四个「显然有效」的方案全部失败

### ❌ 1. NODE_OPTIONS 预加载 shim 伪造 `os.cpus()`

最自然的想法:`--require` 一个 shim 把 `os.cpus` monkey-patch 掉。

```js
const os = require("os"), real = os.cpus.bind(os);
os.cpus = () => real().slice(0, 6);
```

在系统 node 上 `os.cpus().length` 确实从 256 变 6。但挂到 claude 上——**debug 行根本没打印**。一测才知道:

```bash
NODE_OPTIONS="--bogus-flag-xyz" claude --version   # 不报错 → NODE_OPTIONS 被完全忽略
```

**Node SEA 独立二进制不读 `NODE_OPTIONS`。** shim 这条路直接死。

### ❌ 2. LD_PRELOAD 重定向 `/proc/stat`

老 libuv 的 `os.cpus()` 从 `/proc/stat` 数 `cpuN` 行。写个 `.so` 把对 `/proc/stat` 的打开重定向到一份只有 6 行的假文件——在系统 node 上立刻 256→6。

挂到 claude 上:**峰值还是 11**。strace 一看,claude 整个 workflow 期间**根本没读 `/proc/stat`**。版本不同,死。

### ❌ 3. taskset 限制 CPU 亲和性

strace 显示 claude 调了 `sched_getaffinity`,返回完整 `[0..255]`。那 `taskset -c 0-5` 把亲和性砍到 6 个总行了吧?

```bash
taskset -c 0-5 claude ...   # 峰值仍是 11
```

死。核数不是从亲和性来的。

### ❌ 4. LD_PRELOAD 重定向 `/sys/devices/system/cpu/online`

strace 抓到唯一的 CPU 相关文件:`/sys/devices/system/cpu/online`,读了 10 次,内容 `0-255`。把它重定向到 `0-5`:

```bash
grep -aoE '"/sys/devices/system/cpu[^"]*"' trace.txt | sort | uniq -c
#  10 "/sys/devices/system/cpu/online"
```

**峰值还是 11。** 这就很诡异了——明明读的就是这个文件啊?

### ✅ 对照实验:改二进制确认模型没错

为了确认「核数确实是那个杠杆」,我在副本上把 `Br5(rw4.cpus().length)` 直接改成 `Br5(6)`(等字节填充,不改文件大小):

```python
b.replace(b'Br5(rw4.cpus().length)', b'Br5(6'+b' '*16+b')')
```

**峰值 = 5。** 公式和测量方法都对,只是前面所有「喂假核数」的手段都没真正改到那个值。

## 卡住了:那就上网查 `os.cpus()` 到底怎么实现

与其继续猜,直接查 libuv 源码。结论:

> **libuv 取核数用的是 `sysconf(_SC_NPROCESSORS_ONLN)`**,`/proc/stat`、`/proc/cpuinfo` 只用来取时间和型号。

而**现代 glibc 的 `sysconf(_SC_NPROCESSORS_ONLN)` 内部读的正是 `/sys/devices/system/cpu/online`**——对上了 strace 里那 10 次打开!

那为什么重定向那个文件没用?**因为 glibc 内部用的是私有 syscall(`__open64_nocancel` 之类),不走 `openat` 的 PLT 符号**,所以我的 LD_PRELOAD hook 拦不到。

正解浮出水面:**别拦文件,拦 `sysconf` 这个公开符号本身。**

## ✅ 最终方案:LD_PRELOAD 覆盖 sysconf

```c
// fakecpu.c
#define _GNU_SOURCE
#include <unistd.h>
#include <stdlib.h>
#include <dlfcn.h>

static int fake_n(void){ const char* s=getenv("FAKE_CPUS"); return s?atoi(s):0; }

long sysconf(int name){
  static long(*r)(int)=0; if(!r) r=dlsym(RTLD_NEXT,"sysconf");
  int n=fake_n();
  if(n>0 && (name==_SC_NPROCESSORS_ONLN || name==_SC_NPROCESSORS_CONF)) return n;
  return r(name);
}
int get_nprocs(void){ int n=fake_n(); if(n>0) return n;
  static int(*r)(void)=0; if(!r) r=dlsym(RTLD_NEXT,"get_nprocs"); return r(); }
int get_nprocs_conf(void){ int n=fake_n(); if(n>0) return n;
  static int(*r)(void)=0; if(!r) r=dlsym(RTLD_NEXT,"get_nprocs_conf"); return r(); }
```

```bash
gcc -shared -fPIC -O2 -o /root/.claude/fakecpu.so fakecpu.c -ldl
```

接入启动脚本:

```bash
export FAKE_CPUS=6                                # = 想要的并发数 + 2
export LD_PRELOAD="/root/.claude/fakecpu.so${LD_PRELOAD:+:$LD_PRELOAD}"
```

**实测:峰值 11 → 5。** workflow 并发被锁到 4。不改二进制、不需要 namespace、不怕 auto-update。

## 实测矩阵汇总

| 方法 | 峰值 | 结论 |
|---|---|---|
| 不干预(基线) | 11 | cap=16 |
| `NODE_OPTIONS` + os.cpus shim | — | ❌ SEA 不读 NODE_OPTIONS |
| LD_PRELOAD 重定向 `/proc/stat` | 11 | ❌ claude 不读此文件 |
| `taskset -c 0-5` | 11 | ❌ 核数不来自亲和性 |
| LD_PRELOAD 重定向 `/sys/.../online` | 11 | ❌ glibc 私有 syscall 绕过 hook |
| 改二进制 `Br5(6)`(对照) | 5 | ✅ 但会被升级覆盖 |
| **LD_PRELOAD 覆盖 `sysconf`** | **5** | ✅ **最终方案** |

## 经验教训

1. **不要相信「显然有效」,要有客观的测量手段。** 这次四个方案我都觉得稳了,结果全 11。计数代理这把尺子才是整件事的关键。
2. **猜不动的时候去查实现/源码。** 在 `/proc/stat` vs `/sys/online` vs 亲和性之间瞎试,不如直接确认 libuv 用的是 `sysconf`。
3. **LD_PRELOAD 拦文件 vs 拦函数是两回事。** glibc 内部 syscall 绕开 PLT,重定向文件路径无效;但只要目标走的是公开符号(`sysconf`),覆盖符号一击即中。
4. **SEA 独立二进制吃掉了一堆常规旋钮**(NODE_OPTIONS 等),排查时要先确认环境变量到底有没有被读。

## 已知副作用

LD_PRELOAD 会传染给 claude 派生的所有子进程,它们的 `sysconf` 也会返回 6 核。对「限 workflow 并发」无害,但若某个子工具按核数开线程池,会偏小。可接受。

## 调并发

`FAKE_CPUS = 想要的并发数 + 2`,范围让结果落在 2~16(公式 `min(16, max(2, n-2))`)。例:并发 4 → `FAKE_CPUS=6`;并发 8 → `FAKE_CPUS=10`。

---

*参考:[libuv misc docs](https://docs.libuv.org/en/v1.x/misc.html)、[libuv #2351](https://github.com/libuv/libuv/issues/2351)*
