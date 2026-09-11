#!/usr/bin/env python3
"""Sync blogs/ drafts into source/_posts/ Hexo posts.

用法:
  python3 scripts/sync_blogs.py             # 同步并本地构建验证
  python3 scripts/sync_blogs.py --push      # 同步 + 构建 + 提交推送（触发 Pages 部署）
  python3 scripts/sync_blogs.py --dry-run   # 只打印计划，不写文件
  python3 scripts/sync_blogs.py --no-build  # 跳过 hexo generate

配置: blogs/sync.yml（手工维护，见文件内注释）。

同步是无状态的: 每次从 blogs/ 源文件 + 配置在内存中重新生成文章,
与 source/_posts/ 现有文件逐字节比对——
  新文章  -> 写入, date/updated 取配置 date 或源文件 mtime
  有变化  -> 重写, 保留原 date, updated 置为当前时间
  相同    -> 跳过
不会删除文章; 删除草稿后请手动删除对应 post。
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
BLOGS = ROOT / "blogs"
POSTS = ROOT / "source" / "_posts"
IMAGES = ROOT / "source" / "images"
CONFIG_PATH = BLOGS / "sync.yml"

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg"}
EXTERNAL_RE = re.compile(r"^([a-z][a-z0-9+.-]*:|#|mailto:)", re.I)

errors = []


def die_on_errors():
    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


class Entry:
    """一个同步条目: 一个系列目录, 或顶层单个 .md."""

    def __init__(self, path, meta):
        self.path = path                      # 目录或 .md 文件
        self.meta = meta
        self.base_dir = path if path.is_dir() else BLOGS

    def chapters(self):
        """yield (章节号或 None, 源文件, 该章配置, slug)."""
        raise NotImplementedError

    def link_targets(self):
        """{解析后的源文件绝对路径: slug}, 供互链改写."""
        return {src: slug for _n, src, _ch, slug in self.chapters()}

    def image_cfgs(self):
        """{图片文件名: {dest, alt}} — 合并所有章节的 images 配置."""
        cfgs = {}
        for _n, _src, ch, _slug in self.chapters():
            for name, ic in (ch.get("images") or {}).items():
                cfgs[name] = ic
        return cfgs


class Series(Entry):
    CHAPTER_RE = re.compile(r"\((\d+)\)\.md$")

    def __init__(self, path, meta):
        super().__init__(path, meta)
        self.prefix = meta["prefix"]
        self.pad = int(meta.get("pad", 0))
        self.chapters_cfg = meta["chapters"]
        self.files = {}
        for f in sorted(path.glob("*.md")):
            m = self.CHAPTER_RE.search(f.name)
            if not m:
                errors.append(f"{f}: 文件名不符合 <名>(N).md 章节模式")
                continue
            n = int(m.group(1))
            if n in self.files:
                errors.append(f"{f}: 章节号 {n} 重复")
            self.files[n] = f
        for n in self.files:
            if n not in self.chapters_cfg:
                errors.append(f"{path.name}: 章节 {n} 缺少 chapters.{n} 配置")
        for n in self.chapters_cfg:
            if n not in self.files:
                errors.append(f"{path.name}: 配置了章节 {n}, 但没有对应文件")

    def slug_for(self, n):
        return f"{self.prefix}-{n:0{self.pad}d}-{self.chapters_cfg[n]['slug']}"

    def chapters(self):
        for n, f in sorted(self.files.items()):
            yield n, f, self.chapters_cfg[n], self.slug_for(n)


class Single(Entry):
    def chapters(self):
        yield None, self.path, self.meta, self.meta["slug"]


def discover(items_cfg):
    entries = {}
    for p in sorted(BLOGS.iterdir()):
        if p.name.startswith(".") or p.name == "sync.yml":
            continue
        meta = items_cfg.get(p.name)
        if meta is None:
            errors.append(f"blogs/{p.name}: 在 blogs/sync.yml 中没有配置")
            continue
        if meta.get("skip"):
            print(f"skip (配置了 skip): blogs/{p.name}")
            continue
        kind = meta.get("type") or ("series" if p.is_dir() else "single")
        if kind == "series":
            if not p.is_dir():
                errors.append(f"blogs/{p.name}: type: series 但不是目录")
                continue
            entries[p.name] = Series(p, meta)
        elif kind == "single":
            if not (p.is_file() and p.suffix == ".md"):
                errors.append(f"blogs/{p.name}: type: single 需要是顶层 .md 文件")
                continue
            entries[p.name] = Single(p, meta)
        else:
            errors.append(f"blogs/{p.name}: 未知 type: {kind}")
    return entries


# ---------------- 正文变换 ----------------

MD_LINK_RE = re.compile(r"(!?)\[([^\]]*)\]\((?:<([^>]+)>|([^<>()\s]+))\)")


def default_title(text, src):
    first = text.split("\n", 1)[0]
    if not first.startswith("# "):
        errors.append(f"{src}: 第一行不是 H1 标题")
        return ""
    return first[2:].strip()


def transform_body(text, src, entry, cur_n, link_targets, site_root, dry):
    """H1 剥离 / [TOC] 移除 / 互链与图片路径改写 / 插入 more 标记."""
    lines = text.split("\n")
    if not lines[0].startswith("# "):
        errors.append(f"{src}: 第一行不是 H1 标题")
        return ""
    lines = [l for l in lines[1:] if l.strip() != "[TOC]"]
    body = "\n".join(lines).lstrip("\n")

    def sub(m):
        bang, text_part, t_angle, t_plain = m.groups()
        target = t_angle if t_angle is not None else t_plain
        if EXTERNAL_RE.match(target):
            return m.group(0)
        resolved = (entry.base_dir / target).resolve()
        if not resolved.exists():
            errors.append(f"{src}: 链接目标不存在: {target}")
            return m.group(0)
        if resolved.suffix == ".md":
            slug = link_targets.get(resolved)
            if slug is None:
                errors.append(f"{src}: 链接目标不是已配置的文章: {target}")
                return m.group(0)
            return f"{bang}[{text_part}]({site_root}/posts/{slug}/)"
        if resolved.suffix.lower() in IMAGE_SUFFIXES:
            cfg = entry.image_cfgs().get(resolved.name, {})
            alt = cfg.get("alt") or text_part
            if cfg.get("dest"):
                dest = cfg["dest"]
            elif isinstance(entry, Series):
                dest = f"{entry.prefix}-{cur_n}-{resolved.stem}{resolved.suffix}"
            else:
                dest = f"{resolved.stem}{resolved.suffix}"
            if not dry:
                IMAGES.mkdir(parents=True, exist_ok=True)
                shutil.copy2(resolved, IMAGES / dest)
            return f"{bang}[{alt}]({site_root}/images/{dest})"
        errors.append(f"{src}: 不支持链接的文件类型: {target}")
        return m.group(0)

    body = MD_LINK_RE.sub(sub, body)

    first, sep, rest = body.partition("\n\n")
    if not sep:
        errors.append(f"{src}: 正文没有空行分段, 无法插入 more 标记")
        return ""
    return f"{first}\n\n<!-- more -->\n\n{rest}"


# ---------------- front matter 与写盘 ----------------

def compose_post(title, date, updated, categories, tags, description, body):
    fm = "---\n"
    fm += f"title: {json.dumps(title, ensure_ascii=False)}\n"
    fm += f"date: {date}\n"
    fm += f"updated: {updated}\n"
    fm += "categories:\n" + "".join(f"  - {c}\n" for c in categories)
    fm += "tags:\n" + "".join(f"  - {t}\n" for t in tags)
    if description:
        if ": " in description or description[0] in "[]{}>&*#|`@%\"'":
            errors.append(f"description 含 YAML 不安全字符, 请改写: {description[:30]}...")
            return ""
        fm += f"description: {description}\n"
    fm += "---\n"
    return fm + body


def existing_field(post_path, field):
    m = re.search(rf"^{field}: (.+)$", post_path.read_text(encoding="utf-8"), re.M)
    if not m:
        errors.append(f"{post_path}: 已有文章缺 {field} 字段")
        return None
    return m.group(1).strip()


def as_list(v):
    return v if isinstance(v, list) else [v]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--push", action="store_true", help="同步后提交并推送 (触发 Pages 部署)")
    ap.add_argument("--dry-run", action="store_true", help="只打印计划, 不写文件")
    ap.add_argument("--no-build", action="store_true", help="跳过 hexo generate 验证")
    args = ap.parse_args()

    site_root = yaml.safe_load((ROOT / "_config.yml").read_text(encoding="utf-8"))["root"].rstrip("/")
    items_cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))["items"]
    entries = discover(items_cfg)
    die_on_errors()

    link_targets = {}
    for e in entries.values():
        link_targets.update(e.link_targets())
    # 也允许链接到已有但非 blogs 生成的文章: 按文件名匹配 slug
    for p in POSTS.glob("*.md"):
        link_targets.setdefault((BLOGS / (p.stem + ".md")).resolve(), p.stem)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pending_writes, written, unchanged = [], [], []
    for name, e in entries.items():
        cats = as_list(e.meta["categories"])
        tags = as_list(e.meta["tags"])
        for n, src, ch, slug in e.chapters():
            text = src.read_text(encoding="utf-8")
            if not text.strip():
                errors.append(f"{src}: 空文件 (填写内容或在 sync.yml 中 skip)")
                continue
            title = ch.get("title") or default_title(text, src)
            body = transform_body(text, src, e, n, link_targets, site_root, args.dry_run)
            post = POSTS / f"{slug}.md"
            if post.exists():
                date = existing_field(post, "date")
                # 先用原有 updated 比对, 真有变化才刷新时间戳, 避免每次全量重写
                updated = existing_field(post, "updated")
            else:
                date = ch.get("date") or datetime.fromtimestamp(
                    src.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                updated = date
            content = compose_post(title, date, updated, cats, tags, ch.get("description"), body)
            if not content:
                continue
            if post.exists() and post.read_text(encoding="utf-8") == content:
                unchanged.append(slug)
                continue
            if post.exists():
                content = compose_post(title, date, now, cats, tags, ch.get("description"), body)
            pending_writes.append((post, content))
    # 两阶段: 全部校验通过后才落盘, 避免把有问题的内容写进 _posts
    die_on_errors()

    for post, content in pending_writes:
        print(f"{'would write' if args.dry_run else 'write'}: {post.relative_to(ROOT)}")
        if not args.dry_run:
            POSTS.mkdir(parents=True, exist_ok=True)
            post.write_text(content, encoding="utf-8")
            written.append(post.stem)
    die_on_errors()

    print(f"\nunchanged: {len(unchanged)}, written: {len(written)}")
    if not written:
        return

    if not args.no_build and not args.dry_run:
        print("\n== hexo generate ==")
        subprocess.run(["pnpm", "build"], cwd=ROOT, check=True)

    if args.push and not args.dry_run:
        print("\n== git ==")
        subprocess.run(["git", "add", "source/_posts", "source/images", "blogs/sync.yml"],
                       cwd=ROOT, check=True)
        msg = "Sync blogs/ drafts: " + ", ".join(written)
        subprocess.run(["git", "commit", "-m", msg], cwd=ROOT, check=True)
        subprocess.run(["git", "push", "origin", "main"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
