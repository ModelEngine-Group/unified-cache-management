# docs-next

UCM 文档站的重建工程,基于 **MkDocs Material**。与现网 Sphinx 站(`../docs/`)隔离建设,三项评审模块(切到 MkDocs、AI 中文生成、内容调整)全部通过前不切换正式入口。

> 状态:本地预览可用;严格构建通过(中文镜像页缺失时 fallback 英文,有 WARNING 但不阻塞)。

## 快速开始

```bash
cd docs-next
pip install -r requirements.txt  # 首次
mkdocs serve                    # 或: python tools/site.py serve
```

打开 http://127.0.0.1:8000 —— **英文在 `/`(默认语言无前缀),中文在 `/zh/`**。`Ctrl+C` 停。

> 改普通 `.md` 是热更新(live reload);改了 `overrides/main.html` 这类主题模板后必须**重启** `mkdocs serve`(livereload 不重载模板)。

## 更新文档

日常更新就两步:**加 md + 登记到 nav**。

### 1. 加 md 文件

放到 `docs/en/<路径>.md`(英文)。中文镜像放 `docs/zh-cn/` 同路径(不建则 fallback 显示英文)。文件名建议 kebab-case。

### 2. 登记到导航

在 `mkdocs.yml` 的 `nav:` 加条目,否则页面能直接访问但不进左侧导航树(mkdocs 会提示 "not included in nav"):

```yaml
nav:
  - User Guide:
      - My Page: user-guide/my-page.md   # 路径相对 docs/en/
```

### 3. 静态资源(图片等)

- 图片放 `docs/assets/images/`,引用用**绝对路径** `/assets/images/xxx.png`(en 和 zh 下都生效)
- 计算器 JS/HTML 在 `docs/assets/` 根,iframe 用 `/assets/kv_cache_calculator.html`

### 写作注意点(踩过的坑)

- **表格前必须有空行**:GFM 表格前一行若是段落,表格不渲染(会变成段落文字)
- **HTML 容器带 `markdown` 属性**:`<div align="center" markdown>` 才渲染 div 里的 `![]()` 图片/徽章,没 `markdown` 属性则忽略
- **折叠块**用 admonition `??? note "标题"` 而非 `<details>`
- **图片宽度**用 attr_list `![](){ width=60% }` 而非 `<img width>`

## 构建与校验

| 命令 | 作用 |
| --- | --- |
| `mkdocs build` | 普通构建 |
| `mkdocs build --strict` | 严格构建(WARNING 升错误) |
| `python tools/site.py validate` | 全语言严格构建 |
| `python tools/site.py build --lang en --strict` | 单语言严格构建 |

严格构建通过(退出 0)。中文首页指向尚未创建的中文镜像页会产生若干 WARNING(i18n fallback 固有,不阻塞);AI 生成完整中文后 WARNING 消失。本地检查用 `mkdocs build` 即可。

`tools/site.py` 子命令:`serve` / `build --lang {en,zh-cn} [--strict] [--clean]` / `validate` / `translate --changed`(CI,本地未接通) / `generate`(占位)。

## 多版本预览

版本选择器需 mike 多版本站点:

```bash
mike deploy 0.5.0 latest -u --ignore-remote-status  # 本地构建(不推远程)
mike serve                                          # 访问 /latest/
```

- `mkdocs serve`:dev 模式,有 live reload,**无版本选择器**
- `mike serve`:多版本静态预览,有版本选择器,**无 live reload**(改内容后需重新 `mike deploy`)
- 正式发布走 CI:`mike deploy <version> -u -p`(推 gh-pages)

## 项目结构

```
docs-next/
├── mkdocs.yml          站点配置:nav、Material 主题、i18n、markdown 扩展
├── docs/               docs_dir(站点内容根)
│   ├── en/             英文(默认语言,URL 无前缀)
│   ├── zh-cn/          中文镜像(当前仅 index/about,其余待 AI 生成)
│   └── assets/         共享静态资源(images/ css/ js/ calculator、model-configs)
├── overrides/
│   └── main.html       主题覆盖:KaTeX CDN + header 白色 + 字体分层(Jost 侧栏 / Inter 正文)
├── tools/
│   ├── site.py         统一入口(serve/build/validate/translate/generate)
│   └── gen_model_tour.py  生成 Model Tour 具体模型拉起页
├── requirements.txt    Python 依赖(mkdocs / mkdocs-material / mkdocs-static-i18n / pymdown-extensions)
├── .venv/              本地虚拟环境(不提交)
└── site/               构建产物(不提交)
```

## 国际化(i18n)

- 插件 `mkdocs-static-i18n`,**folder 模式**(`docs/en/`、`docs/zh-cn/`)
- 英文默认语言,**URL 无前缀**(在 `/`,不是 `/en/`);中文在 `/zh/`
- `fallback_to_default: true`:中文页缺失时回退英文内容
- 共享静态资源用根绝对路径 `/assets/...`(避免 i18n 静态资产路径问题;构建时这些绝对链接留 INFO 提示属正常)

## 内容规范

- **命令优先,非散文**:Model Tour / Engines / Deploy 尤其;先给可执行命令,再补说明
- **参数单一事实源**:任务页只内联相关子集并链 `reference/api-parameters.md`;全量参数表只在 Reference 维护
- **无 emoji**:表格状态用 `Yes` / `No` / `Untested`,评级用 `n/5`;架构图保留 ASCII 框线
- **占位页**:待补页面在 md 里写 "What to add" 规范(读者意图/必含/不要做/验收/负责人),由模块负责人填内容;已有内容页不改

## 数学公式

KaTeX 经 `overrides/main.html` 注入(CDN),不随 i18n 静态资产走。正文用 `$$...$$`(块)或 `$...$`(行内)。

## 相关文件

- 重构技术评审:`../docs/ucm-mkdocs-site-rearchitecture-technical-review.md`(§4.7.3 定义六模块顶层导航与负责人矩阵)
- 现网 Sphinx 站:`../docs/`——**切换前不动**
