# UCM documentation

文档使用 MkDocs Material，目标托管平台为 Read the Docs（RTD）。英文是内容来源，中文位于同路径的 `docs/zh/`。切换采用 Fork 预览 → 官方 `ucm` 的顺序；旧 Sphinx 历史版本和 GitHub Pages 历史下载入口继续保留。

## 本地开发与验证

使用 Python 3.12：

```bash
cd docs/docs-site
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python tools/site.py serve
```

开发服务的英文在 `/`，中文在 `/zh/`。默认不联网获取 Release。

共享资源位于 `docs/assets/`。`tools/shared_assets.py` 在 i18n 筛选后保留这些资源，
不为缺失的中文页面提供英文回退。计算器的 iframe 使用模板的 `url` 过滤器，
兼容双语本地预览和独立语言构建。

```bash
# 与 RTD 相同的独立语言构建；输出 site/en 和 site/zh
python tools/site.py validate
python tools/site.py build --lang en --strict
python tools/site.py build --lang zh --strict

# 从指定仓库选择真实安装清单，或者传入本地已验证的发布清单
python tools/site.py build --lang en --strict --repository OWNER/REPO
python tools/site.py build --lang zh --strict --manifest /path/to/release-manifest.json

python -m pytest -q tests
node --test tests/install-ui.test.cjs
```

`requirements.txt` 固定站点依赖；`requirements-dev.txt` 添加测试工具。

`build --lang` 选择一种语言并直接读取源码。中英文页面必须完整配对，不复制英文、不提供缺译回退。严格构建检查页面与资源，双语检查比较当前完整目录。

## 内容维护

- 英文放在 `docs/en/`，中文使用 `docs/zh/` 下相同路径。新增公开页面登记到 `mkdocs.yml`；图片及脚本使用相对链接。
- 安装包、镜像及 Chart 坐标只来自当前构建的发布清单；Quickstart 顶部确定环境，正文使用相应的完整制品命令，再完成配置、启动与验证；不另写裸 `pip install uc-manager` 或镜像 `latest`。SGLang 也通过同一清单选择器展示当前制品。
- Quickstart 保留 Docker 启动、设备与目录挂载、UCM 安装、引擎启动及请求验证的完整路径；安装选择器负责制品坐标，不能替代启动命令。迁移操作步骤时保留可到达的对应入口。
- 参数参考以实际配置读取位置为依据。任务页引用参考，避免重复维护默认值。
- 新页面必须有对读者有效的内容。内部待办留在任务或 Issue 中；Model Tour 保留 GLM、Qwen、DeepSeek、MiniMax、Kimi 五个家族；内容较少时保留栏目供后续补充。
- 用户指南提供快速开始、部署、Model Tour 和运行排障。快速开始负责安装、最小配置、启动和验证，并链接开发者指南中的后端选择与缓存配置方法，以及参考中的参数定义；不另设“存储选择”页面，不重复维护参数表。Store 详细页放在开发者指南；指标清单以旧文档 metrics_list.md 为基线，保留条目并注明已确认的版本差异。
- 删除或合并页面时同步更新导航、正文链接和图片，不生成历史 URL 跳转。新站不提供 UCM 稀疏注意力和 ReRoPE 文档；计算器中的模型结构与容量计算继续保留。
- Benchmark 占位源码暂不参与站点构建，也不显示在导航中；内容补齐后再恢复。各使用指南保留必要验证。
- 所有 Markdown 页面在 `docs/en/` 和 `docs/zh/` 下必须有相同相对路径；新增、删除和重命名时同步维护两种语言。
- 新站只提供当前文档路径。移动页面时更新当前导航和正文链接，不生成历史 URL 跳转页。

架构和请求流程的可编辑 HTML 图稿放在 `diagrams/`，SVG 从对应图稿导出到
`docs/assets/images/`。中英文分别配文；更新图稿后同步导出，并在正文宽度下检查，
不要直接修改 SVG 后遗留不一致的源图。

## RTD 构建与安装清单

根 `.readthedocs.yaml` 使用 Python 3.12，调用 `python docs/docs-site/tools/site.py rtd`。构建读取 RTD 的语言、Git identifier、commit hash、canonical URL 和输出目录，生成一个语言/版本根。英文使用 `/en/<version>/`，中文 RTD 语言为 `zh-cn`，对应源码目录 `zh`。

顶部沿用 Material 的版本和语言菜单，通过 RTD Addons 数据列出已构建的版本与翻译，切换时保留当前页面路径。顶部菜单就绪后隐藏默认悬浮菜单；线上搜索继续使用 RTD，本地保留 MkDocs 搜索。源码链接绑定实际仓库和 Git ref。

快速开始按引擎、真实引擎版本、CUDA/CANN、Ascend 设备、操作系统和 CPU 架构选择环境。同一选择同时决定“标准引擎镜像”和“UCM 镜像”两个页签的制品与命令：前者在容器中安装 UCM，后者已包含 UCM。两者统一使用 `/workspace/model`、`/workspace/storage` 和 `/workspace/ucm.yaml`，不在正文重复选择环境。标准镜像仓库从 `.github/release/release.yaml` 注入，完整上游标签来自发布清单的镜像 family ID，不从 UCM 发布标签猜测。SGLang 直接展示现有指南。Helm 下载只在 Helm 部署页的“获取 Chart”中提供。 系统元数据未报告发行版时，仅保留镜像 ID 显式声明的 `ubuntuNNNN` 标签来区分变体，不推断默认 Linux 镜像的发行版。

选择器读取本版本根目录的 `release-manifest.json`。PyPI 发布完成时使用对应 index；未发布时直接安装清单中的 backend Wheel URL，不假定存在 PyPI 包：

| 构建类型 | 清单来源 |
| --- | --- |
| Tag / Stable | 同仓库、对应 Git 标签的完整发布清单；支持正式版和 RC |
| Latest | 优先使用同仓库最高版本、已完成且具有有效发布清单的 Stable Release；没有合格正式版时，使用最高版本的已完成预发布 Release |
| 没有合格 Release | 页面明确显示安装数据不可用，提供源码构建入口 |

公开清单的生成和校验由 `.github/release/ucm_release/manifest.py` 统一负责。Schema 10 支持一个 Wheel 声明多个 Runtime 能力；仍兼容已发布的 Schema 9 清单。枚举时跳过不支持的格式；精确指定不支持的 Release 时直接报错，不作为待发布重试。已有清单损坏、标签/仓库不匹配、文件集合或下载 URL 与 Release 不一致时构建失败。Latest 页面显示实际安装制品的版本，避免把开发文档版本当作发布版本。

RTD 可能在 Tag 推送时先于产物完成启动构建；此时返回 RTD 的取消码 `183`，不发布不完整页面。Release 流水线完成清单上传、回读及保留策略后，再触发中英文项目的对应 Tag 和 Latest，等待构建成功、核对源码 SHA 并回读公开页面及清单；仅在 RTD 当前 active Stable 对应该 Tag 时重建 Stable，重建旧标签不会回退别名。

日常更新由 `docs-check.yml` 负责：文档、站点配置或发布构建相关路径的变更推送到 `develop` 后，先运行发布契约检查和中英文文档构建，再用 `RTD_API_TOKEN` 调用两个项目的 Latest 构建 API。在 `develop` 上手动运行该 workflow 也会执行同一路径。PR、其他分支和 Tag 不执行 Latest 发布任务。

发布前校验两个项目的仓库、语言、翻译关系及默认分支 `develop`，Fork 必须使用绑定自身仓库的 RTD 项目。任务等待 RTD 构建完成，回读公开页面和安装清单，将实际 build ID、commit 和 URL 保存到 `ucm-docs-latest-receipt-run-<run_id>`。Latest 跟随 RTD 构建时的 `develop`，实际 commit 可能晚于触发 workflow 的提交；`source_sha` 记录触发提交，`builds[].commit` 记录实际构建提交。失败或超时会使 Actions 失败，API 接受触发不等于发布成功。

这条路径不需要 GitHub App，也不依赖旧 webhook。旧 webhook 若保留，仍可能额外触发构建。RTD PR 在线预览和预览提示应关闭，PR 的生成检查和双语门控继续由 GitHub Actions 执行，构建结果在 Actions 中查看。

## PR 中英文文件门控

`docs-bilingual.yml` 在 PR 上检查当前中英文目录的全部 Markdown 路径。新增、删除和重命名都必须在另一语言有对应结果，已有缺译页面也会报错。图片等非 Markdown 文件不参与路径配对；内容质量由 review 确认。

检查只使用 Python 标准库，不读取 Git 历史、不调用模型，也不修改仓库：

```bash
python tools/check_bilingual_docs.py
```

将 `Docs · Bilingual files` 设置为目标分支的必需状态检查后，缺少配对文件的 PR 才会被 GitHub 阻止合并。

## Fork 验收及官方切换

1. 在 RTD 创建英文父项目及中文 Translation 项目，均绑定 Fork，默认分支设为 `develop`，使用该分支的根 RTD 配置。无需连接 GitHub App；关闭两个项目的 Pull request builds 和 PR 预览提示。界面无法编辑时，可通过项目更新 API 将 `external_builds_enabled` 设为 `false`。
2. 语言分别设置为 English 和 Simplified Chinese (`zh-cn`)，版本模式使用带翻译的多版本模式。启用 Addons 的版本/语言和搜索。
3. 将候选修改同步到 Fork `develop`，验证 push 自动更新和手动补跑、中英文 Latest、实际安装选择器、当前 URL、favicon 和计算器。记录 RTD build ID、源码 SHA 与公开 URL；取消构建不等于已部署。
4. 验证中英文文件门控，将 `Docs · Bilingual files` 设置为目标分支的必需状态检查。
5. Fork 验收后，通过官方开发分支集成切换 `ucm` 项目；英文父项目仍使用现有 `ucm`，关联中文项目。新内容只维护 `docs/docs-site`。
6. GitHub Repository Variables 设置 `RTD_PROJECT_EN`、`RTD_PROJECT_ZH`，Repository Secret 设置 `RTD_API_TOKEN`。项目仓库必须与当前发布仓库一致。官方 Latest 和 Stable/Prerelease 发布必须配置两个项目和 Token。Fork 三项均未配置时跳过 RTD 发布；配置齐全后执行文档发布及验证，只配置部分参数会在触发 RTD 构建前失败。
7. 官方先切换 Latest。首个包含新配置、完整发布清单且 RTD Tag 构建通过后启用 Stable。旧 Git 标签仍按原配置构建，不改写历史标签。
8. 验收通过后停止新 Pages 发布，保留原 `gh-pages` 内容及自定义域名，尤其历史下载索引；本轮不修改 DNS。若正式切换失败，恢复上一版 RTD 配置即可继续旧站构建。

RTD 管理和 API Token 通过对应后台配置，不能写入源码或日志。

联合验收在 `verify-release-delivery` 任务汇总 Toolkit 安装、Chart/镜像回读和中英文 RTD 结果。Fork 未配置 RTD 时，验收记录中的 `docs.status` 为 `skipped`，不要求文档回执；其余验收条件保持不变。`ucm-release-acceptance-run-<run_id>` 是内部 Actions artifact，不上传到 GitHub Release。
