# Faccoco's Blog

这是部署在 <https://faccoco.github.io/blog/> 的 Hexo 博客源代码。

## 本地运行

```bash
pnpm install
pnpm server
```

## 构建

```bash
pnpm build
```

推送到 `main` 分支后，GitHub Actions 会自动构建并部署至 GitHub Pages。

## 发布草稿

草稿放在 `blogs/` 下（系列目录或单篇 .md），在 `blogs/sync.yml` 中配好元数据后：

```bash
pnpm sync:blogs          # 同步到 source/_posts/ 并本地构建验证
pnpm sync:blogs --push   # 同步 + 构建 + 提交推送（触发 Pages 部署）
```

脚本是无状态同步：每次从 `blogs/` 源文件 + 配置重新生成文章并与现有文件比对，新增写入、变更更新（保留原 date、刷新 updated）、相同跳过；章节互链与图片路径自动改写。配置说明见 `blogs/sync.yml` 文件内注释。注意：脚本不会删除文章，删除草稿后需手动删除对应 post。
