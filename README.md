<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/source/logos/UCM.png">
    <img alt="UCM" src="docs/source/logos/UCM-light.png" width=50%>
  </picture>
</p>

<p align="center">
| <a href="docs/source/index.md"><b>Documentation</b></a> | <a href="https://github.com/ModelEngine-Group/unified-cache-management/issues/16"><b>Roadmap</b></a> |
</p>

---

*Latest News* 🔥
- [2025/08/01] We are excited to announce the alpha release of Unified Cache Manager.

---

## Performance
nfs connector has reached about 4x TTFT accelerate.

![perf](docs/source/_static/images/nfs_performance.png)

## About
UCM is rooted in KV Cache, with the goal of reducing inference costs and building commercially viable inference solutions. It enhances throughput through methods such as Prefix Cache, sparsification, and PD Disaggregation.

---

## Quick Start
please refer to [Quick Start](/develop/docs/source/getting-started/quick_start.md).

---

## Support Features
please refer to [features matrix](docs/source/feature/support.md).

---

## Branch Policy
Unified Cache has main branch, develop branch and release branch.
- **main**: The main branch is the stable line; it receives merges from develop once basic tests pass. In principle, there should be no direct check-ins to the main branch.
- **develop**: The develop branch is the daily development branch where new features are merged.
- **release**: Each time a new version release process begins, release branch is merged from the main branch. This branch only accepts bug fixes, and all bug fixes will be picked back to the develop branch. After the release testing passes, a release tag (x.x.x) will be created on the release branch. The release branch will be retained for the next release.

---

## Contributing
We welcome and value any contributions and collaborations. Please check out [Contributing](docs/source/developer-guide/contribute.md) to vLLM for how to get involved.

---

## License

UCM is licensed under the MIT with additional conditions. Please read the [LICENSE](./LICENSE) file for details.
