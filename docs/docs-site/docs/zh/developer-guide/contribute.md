# UCM 贡献指南

UCM 社区欢迎问题反馈、功能开发和文档改进。每一种贡献都能帮助项目服务更多用户。

## 参与方式

- **报告问题或缺陷**：描述遇到的问题和异常行为，帮助改进稳定性与性能。
- **支持新的硬件或组件**：提出需求或提交实现，将 UCM 扩展到新的设备和平台。
- **提出或实现新功能**：分享功能建议，或直接贡献代码。
- **改进文档与指南**：完善现有文档、补充新指南，帮助其他用户理解和使用 UCM。

## 许可证

许可证信息参见 [LICENSE](https://github.com/ModelEngine-Group/unified-cache-management/blob/develop/LICENSE)。

## 提交 Issue

发现缺陷或希望提出新功能时，请先检查[已有 Issue](https://github.com/ModelEngine-Group/unified-cache-management/issues)。如果没有对应的问题，请[新建 Issue](https://github.com/ModelEngine-Group/unified-cache-management/issues/new/choose)，并提供：

- **标题**：准确、简洁地概括问题。
- **环境**：设备、推理框架版本、软件包版本和操作系统等。
- **复现步骤**：按顺序说明如何稳定触发问题。
- **预期与实际行为**：说明预期结果，以及实际发生的情况。
- **错误信息与日志**：提供相关控制台输出或日志片段。
- **可视化证据**：必要时提供截图或短录屏。
- **严重程度**：标明影响级别，如 Critical、High、Medium 或 Low。

清晰、客观、简洁的报告有助于更快定位问题。

## 提交代码变更

### 第一步：Fork 仓库

将 [UCM 仓库](https://github.com/ModelEngine-Group/unified-cache-management) Fork 到自己的 GitHub 账号。

### 第二步：创建分支

创建一个名称清晰、能说明用途的新分支。

### 第三步：实现变更

在分支上修改代码或文档，保持变更聚焦，并遵循项目现有风格与约定。

### 第四步：执行代码风格检查

UCM 遵循以下风格规范：

- Python：[PEP 8](https://peps.python.org/pep-0008/)。
- C++：[Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)。

项目使用以下工具保持格式一致：

- Python 检查和格式化：[black](https://black.readthedocs.io/en/stable/the_black_code_style) 与 [isort](https://pycqa.github.io/isort/)。
- 拼写检查：[codespell](https://github.com/codespell-project/codespell)。
- C++ 格式化：[clang-format](https://clang.llvm.org/docs/ClangFormat.html)。

建议在本地准备开发环境，并在提交 PR 前运行检查、修复问题：

```bash
# Run the following commands to format your code before submitting.
# Using a virtual environment is optional but recommended to avoid dependency conflicts.

# Choose a workspace dir (e.g., ~/vllm-project/) and set up venv (optional)
cd ~/vllm-project/
python3 -m venv .venv
source ./.venv/bin/activate

# Clone UCM and install
git clone https://github.com/ModelEngine-Group/unified-cache-management.git
cd unified-cache-management

# Install lint requirement and enable pre-commit hook
pip install -r requirements-lint.txt

# Run lint (You need install pre-commits deps via proxy network at first time)
bash format.sh
```

### 第五步：创建 Pull Request

将变更推送到自己的 Fork，再向主仓库创建 PR。提供清晰的标题、变更说明、关联 Issue，以及必要的上下文或截图。

PR 标题使用以下前缀说明变更类型。涉及多个类别时，应包含相应前缀，例如 `[Feat][Test]`：

- `[Feat]`：新功能或功能增强。
- `[Bugfix]`：修复缺陷。
- `[Opt]`：在不改变功能的前提下改进性能、效率或资源使用。
- `[Build]`：构建系统、依赖或工具变更。
- `[CI]`：持续集成配置与工作流变更。
- `[Doc]`：文档改进或修正。
- `[Test]`：新增、修改或重构测试。
- `[Misc]`：不属于上述类别的变更，请谨慎使用。

### 第六步：代码评审

面向受保护分支（例如 `develop;release`）的 PR 应遵循以下评审要求：

- **Code Owner 审批**：根据修改的文件自动请求相关 Code Owner 评审。作者不能审批自己的 PR，至少需要一位指定 Code Owner 批准。
- **额外评审**：除作者外，还需获得充分的同行评审，由熟悉相关代码的维护者或贡献者验证。
- **CI 通过**：合并前所有持续集成检查，包括风格检查、单元测试和构建，都应通过。
- **禁止绕过**：受保护分支不允许直接推送，也不允许强制合并或自行批准。

满足要求后可以使用 “Squash and merge”。请求评审前，请确保变更有相应说明和验证，并符合项目编码规范。

## 改进文档

可以通过修正笔误、澄清表述、补充缺失功能和配置步骤、改进示例或增加实际用例来贡献文档。

### 构建文档

新文档位于 `docs/docs-site/`，使用 MkDocs Material 构建。中英文页面保持相同相对路径，新增或移动页面时同步更新导航和链接。

```shell
# Install documentation dependencies.
cd unified-cache-management/docs/docs-site
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt

# Serve the site locally with live reload.
mkdocs serve
# Or via the unified entry point:
python tools/site.py serve

# Build a strict production build for both languages.
python tools/site.py build --lang en --strict
python tools/site.py build --lang zh --strict

# Validate the whole site (strict build across languages).
python tools/site.py validate
```

提交 PR 前，请在浏览器中检查页面：

- 英文：[http://127.0.0.1:8000/](http://127.0.0.1:8000/)。
- 中文：[http://127.0.0.1:8000/zh/](http://127.0.0.1:8000/zh/)。
