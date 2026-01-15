# UCM Contributing Guide

Thank you for your interest in contributing to the UCM project! Our community is open to all, and we welcome various contributions—your efforts help the project grow and benefit more users.
## 🧩 Ways to Contribute
You can contribute in the following ways, and we value each type of participation:
- **🐛 Report issues or bugs**  
  Report any issues or unexpected behavior you encounter. Your feedback helps us improve stability and performance.
- **⚙️ Add support for new hardware or components**  
  Extend UCM to new devices or platforms by proposing requirements or submitting implementations. All ideas and contributions are welcome.
- **✨ Suggest or implement new features**  
  Share feature ideas or contribute code to help UCM better meet user needs.
- **📚 Improve documentation and guides**  
  Enhance existing documentation or create new guides to help others get started quickly and effectively.

Every contribution matters—thank you for being part of the UCM community.


## 📄 License

License information is available in the [LICENSE](https://github.com/ModelEngine-Group/unified-cache-management/blob/develop/LICENSE) file.


## 🐞 How to Submit an Issue

If you find a bug or want to suggest a new feature, please first check [existing issues](https://github.com/ModelEngine-Group/unified-cache-management/issues). If it’s a new issue, please [open a new issue](https://github.com/ModelEngine-Group/unified-cache-management/issues/new/choose   ) and include the following information:

- **Title**: A clear and concise summary of the issue.  
- **Environment**: Device, inference framework version, package version, OS, etc. 
- **Steps to Reproduce**: Clear, numbered steps to reliably trigger the issue.  
- **Expected vs. Actual Behavior**: What should happen versus what actually happens.  
- **Error Messages / Logs**: Relevant console output or log snippets.  
- **Visual Evidence** (if helpful): Screenshots or short screen recordings.  
- **Severity**: Indicate impact level (*Critical*, *High*, *Medium*, *Low*).

Keep your report clear, factual, and concise—this helps us address your issue faster!


## 💻 Submitting Code Changes
**To submit your code changes, please follow the workflow outlined below.**
### Step 1: Fork the Repository

Fork the [UCM repository](https://github.com/ModelEngine-Group/unified-cache-management) to your GitHub account.

### Step 2: Create a Branch

Create a new branch with a clear and descriptive name.

### Step 3: Implement Your Changes

Make your code or documentation changes in the branch, keeping them focused and consistent with the project’s style and conventions. 

### Step 4: Run Lint Checks

It’s recommended to set up a local development environment and run the linter before submitting a PR to help ensure consistent code style.

- **Run lint locally:** run code style checks and fix any issues.

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

### Step 5: Open a Pull Request (PR)

Push your changes to your fork and open a PR in the main repository. Include:

- A **clear title** and **description** of your changes.
- A reference to the related issue.
- Any additional context or screenshots.

The PR title is prefixed appropriately to indicate the type of change, according to the following categories: 

💡 **Reminder:** If your PR spans multiple categories, include all relevant prefixes (e.g., `[Feat][Test]`).

- [Feat] for new features or enhancements
- [Bugfix] for bug fixes
- [Opt] for code optimizations that improve performance, efficiency, or resource usage without changing functionality
- [Build] for build system changes, dependency updates, or tooling improvements
- [CI] for continuous integration configuration and workflow updates
- [Doc] for documentation improvements or corrections
- [Test] for adding, updating, or refactoring tests
- [Misc] for PRs that do not fit other categories — please use sparingly

### Step 6: Code Review Process

All pull requests targeting protected branches (e.g., `develop;release`) must adhere to the following review policy:

- **Code Owner Approval Required**: Pull requests automatically request reviews from relevant code owners based on the files changed. You cannot approve your own pull request, and approval from at least one designated code owner is mandatory.
- **Additional Approvals**: The pull request must receive sufficient peer review beyond the author, including validation from maintainers or experienced contributors familiar with the affected code.
- **Passing CI Checks**: All continuous integration checks—including linting, unit tests, and builds—must pass before merging.
- **No Bypassing**: Direct pushes to protected branches are disabled, and force merges or self-approvals are not permitted.

Once all requirements are satisfied, the “Squash and merge” button will become available. Please ensure your changes are well-documented, tested, and aligned with the project’s coding standards before requesting review.

## 📚 Improving Documentation

**Great docs grow stronger together. You can help by:**

- Fixing typos or clarifying unclear text.
- Documenting missing features or setup steps.
- Improving examples or adding practical use cases.

### Building the docs
```shell
# Install dependencies.
cd unified-cache-management/docs
pip install -r requirements-docs.txt

# Build the docs.
make clean
make html

# Open the docs with your browser
python3 -m http.server -d build/html/
```

Review the documentation in your browser before submitting a pull request.

- English version: [http://localhost:8000](http://localhost:8000/)
