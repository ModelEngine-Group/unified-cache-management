# 安装

UCM 以 Docker 镜像、Helm Chart、Python wheel 包和源码构建的形式交付。使用下方的安装器获取适用于您环境的精确命令，然后按照特定引擎指南完成剩余步骤。

!!! tip "使用说明"

    在下方选择您的 UCM 版本、引擎、设备、操作系统、架构和安装方式 —— 匹配的安装命令会自动显示。

## 安装器

<div id="ucm-install-selector" class="ucm-install">
  <div class="ucm-install__rows" id="ucm-install-rows"></div>
  <div class="ucm-install__output">
    <div class="ucm-install__cmdhead">
      <span class="ucm-install__cmdtitle" id="ucm-cmd-label">运行此命令</span>
      <button class="ucm-install__copy" id="ucm-copy" type="button" title="复制命令" aria-label="复制命令">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="16" height="16" aria-hidden="true"><path fill="currentColor" d="M19 3h-4.18A3 3 0 0 0 12 1a3 3 0 0 0-2.82 2H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V5a2 2 0 0 0-2-2m-7 0a1 1 0 0 1 1 1 1 1 0 0 1-1 1 1 1 0 0 1-1-1 1 1 0 0 1 1-1m7 16H5V5h2v2h10V5h2z"/></svg>
        <span class="ucm-install__copy-label">复制</span>
      </button>
    </div>
    <pre class="ucm-install__cmd"><code id="ucm-cmd"></code></pre>
    <p class="ucm-install__note" id="ucm-note"></p>
  </div>
</div>

<script>
(function () {
  var VERSION = "0.5.0";
  var DIMS = {
    version: ["0.5.0"],
    engine:  ["vLLM", "vLLM Ascend", "SGLang", "MindIE"],
    device:  ["GPU (CUDA)", "NPU (Ascend)"],
    os:      ["Ubuntu", "openEuler"],
    arch:    ["amd64", "arm64"],
    method:  ["Docker", "Helm chart", "pip wheel", "源码构建"]
  };
  var DEFAULTS = { version:"0.5.0", engine:"vLLM", device:"GPU (CUDA)", os:"Ubuntu", arch:"amd64", method:"Docker" };
  var LABELS = { version:"UCM 版本", engine:"引擎", device:"设备", os:"操作系统", arch:"架构", method:"安装方式" };
  var ORDER = ["version","engine","device","os","arch","method"];
  var ENGINE_DEVICE = { "vLLM":"GPU (CUDA)", "vLLM Ascend":"NPU (Ascend)", "SGLang":"GPU (CUDA)", "MindIE":"NPU (Ascend)" };

  function cmd(state) {
    var v = state.version, e = state.engine, d = state.device, o = state.os, a = state.arch, m = state.method;
    if (ENGINE_DEVICE[e] && ENGINE_DEVICE[e] !== d) return null;
    if (m === "Docker") {
      if (e === "vLLM" && d === "GPU (CUDA)") {
        return "docker run --gpus all --rm -p 8000:8000 \\\n  -e HICACHE_ENGINE=vllm \\\n  -v $(pwd)/config.yaml:/app/config.yaml \\\n  ghcr.io/modelengine-group/ucm-vllm:" + v + " \\\n  --model glm-4.5 --hicache-storage-backend ucmconn";
      }
      if (e === "vLLM Ascend" && d === "NPU (Ascend)") {
        return "docker run --device /dev/davinci0 --device /dev/davinci_manager \\\n  --device /dev/hisi_hdc --rm -p 8000:8000 \\\n  -e HICACHE_ENGINE=vllm-ascend \\\n  -v $(pwd)/config.yaml:/app/config.yaml \\\n  ghcr.io/modelengine-group/ucm-vllm-ascend:" + v;
      }
      if (e === "SGLang" && d === "GPU (CUDA)") {
        return "docker run --gpus all --rm -p 30000:30000 \\\n  -v $(pwd)/config.yaml:/app/config.yaml \\\n  ghcr.io/modelengine-group/ucm-sglang:" + v + " \\\n  --model glm-4.5";
      }
      if (e === "MindIE" && d === "NPU (Ascend)") {
        return "docker run --device /dev/davinci0 --device /dev/davinci_manager \\\n  --device /dev/hisi_hdc --rm -p 1024:1024 \\\n  ghcr.io/modelengine-group/ucm-mindie:" + v;
      }
    }
    if (m === "Helm chart") {
      return "helm repo add ucm https://modelengine-group.github.io/charts\nhelm install ucm ucm/unified-cache-pd \\\n  --version " + v.replace("rc1","-rc.1") + " \\\n  --set engine=" + e.toLowerCase().replace(/\s/g,"-") + " \\\n  --set device=" + (d.indexOf("NPU")===0?"npu":"gpu") + " \\\n  --set architecture=" + a;
    }
    if (m === "pip wheel") {
      if (e === "vLLM")        return "pip install ucm-vllm==" + v + " --find-links https://github.com/ModelEngine-Group/unified-cache-management/releases";
      if (e === "vLLM Ascend") return "pip install ucm-vllm-ascend==" + v;
      if (e === "SGLang")      return "pip install ucm-sglang==" + v;
      return null;
    }
    if (m === "源码构建") {
      return "git clone https://github.com/ModelEngine-Group/unified-cache-management.git\ncd unified-cache-management\npip install -e .\n# 然后应用引擎集成补丁（参见引擎指南）";
    }
    return null;
  }

  var NOTE = {
    "Helm chart": "使用 Kubernetes 部署 UCM。完整 Helm values 参考请参见部署指南。",
    "源码构建": "推荐用于开发。应用引擎指南中记录的引擎集成补丁。",
    "pip wheel": "预构建 wheel 包随每个版本发布。需要已安装匹配的引擎。",
    "Docker": "预构建镜像包含引擎。运行时配置请参见引擎指南。"
  };

  function buildRows() {
    var root = document.getElementById("ucm-install-rows");
    root.innerHTML = "";
    ORDER.forEach(function (dim) {
      var row = document.createElement("div");
      row.className = "ucm-install__row";
      var label = document.createElement("span");
      label.className = "ucm-install__rowlabel";
      label.textContent = LABELS[dim];
      var group = document.createElement("div");
      group.className = "ucm-install__group";
      DIMS[dim].forEach(function (opt) {
        var id = "ucm-" + dim + "-" + opt.replace(/[^a-z0-9]/gi,"");
        var lab = document.createElement("label");
        lab.className = "ucm-install__opt";
        lab.htmlFor = id;
        var inp = document.createElement("input");
        inp.type = "radio"; inp.name = "ucm-" + dim; inp.value = opt; inp.id = id;
        if (opt === DEFAULTS[dim]) inp.checked = true;
        var sp = document.createElement("span");
        sp.textContent = opt;
        lab.appendChild(inp); lab.appendChild(sp);
        group.appendChild(lab);
      });
      row.appendChild(label); row.appendChild(group);
      root.appendChild(row);
    });
  }

  function state() {
    var s = {};
    ORDER.forEach(function (dim) {
      var c = document.querySelector('input[name="ucm-'+dim+'"]:checked');
      s[dim] = c ? c.value : null;
    });
    return s;
  }

  function updateActive(inp) {
    if (!inp) return;
    var group = inp.closest(".ucm-install__group");
    if (!group) return;
    group.querySelectorAll(".ucm-install__opt").forEach(function(l){ l.classList.remove("is-active"); });
    inp.closest(".ucm-install__opt").classList.add("is-active");
  }

  function render() {
    var s = state();
    var code = cmd(s);
    var codeEl = document.getElementById("ucm-cmd");
    var labelEl = document.getElementById("ucm-cmd-label");
    var noteEl = document.getElementById("ucm-note");
    var copy = document.getElementById("ucm-copy");
    if (code) {
      codeEl.textContent = code;
      codeEl.parentElement.style.display = "";
      copy.style.display = "";
      labelEl.textContent = s.method + " · " + s.engine + " · " + s.device;
      noteEl.textContent = NOTE[s.method] || "";
      noteEl.classList.remove("ucm-install__note--err");
    } else {
      codeEl.textContent = "";
      codeEl.parentElement.style.display = "none";
      copy.style.display = "none";
      labelEl.textContent = "不可用";
      noteEl.textContent = "此组合不可用。请尝试不同的引擎、设备或方式。";
      noteEl.classList.add("ucm-install__note--err");
    }
  }

  function onEngineChange() {
    var s = state();
    var want = ENGINE_DEVICE[s.engine];
    if (want && s.device !== want) {
      var target = document.querySelector('input[name="ucm-device"][value="'+want+'"]');
      if (target) { target.checked = true; updateActive(target); }
    }
  }

  function init() {
    var root = document.getElementById("ucm-install-selector");
    if (!root || root.dataset.ready) return;
    root.dataset.ready = "1";
    buildRows();
    ORDER.forEach(function(dim){
      var c = document.querySelector('input[name="ucm-'+dim+'"]:checked');
      updateActive(c);
    });
    root.addEventListener("change", function(ev){
      if (ev.target.matches('input[type=radio]')) {
        updateActive(ev.target);
        var dim = ev.target.name.replace("ucm-","");
        if (dim === "engine") onEngineChange();
        render();
      }
    });
    document.getElementById("ucm-copy").addEventListener("click", function(){
      var text = document.getElementById("ucm-cmd").textContent;
      if (navigator.clipboard) navigator.clipboard.writeText(text);
      var lbl = this.querySelector(".ucm-install__copy-label");
      var prev = lbl.textContent;
      lbl.textContent = "已复制";
      setTimeout(function(){ lbl.textContent = prev; }, 1500);
    });
    render();
  }

  if (document.readyState !== "loading") init();
  else document.addEventListener("DOMContentLoaded", init);
  if (typeof document$ !== "undefined") document$.subscribe(init);
})();
</script>

<style>
.ucm-install {
  margin: 1.2em 0;
  border: 1px solid var(--md-default-fg-color--lighter);
  border-radius: 4px;
  overflow: hidden;
  background: var(--md-default-bg-color);
}
.ucm-install__rows { display: flex; flex-direction: column; }
.ucm-install__row {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 6px 16px;
  border-bottom: 1px solid var(--md-default-fg-color--lightest);
}
.ucm-install__rowlabel {
  flex: 0 0 140px;
  font-size: 0.82rem;
  font-weight: 600;
  color: var(--md-default-fg-color--light);
}
.ucm-install__group { display: flex; flex-wrap: wrap; flex: 1 1 auto; min-width: 0; }
.ucm-install__opt {
  position: relative;
  flex: 1 1 0;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  text-align: center;
  padding: 6px 16px;
  font-size: 0.82rem;
  font-weight: 500;
  border: 1px solid var(--md-default-fg-color--lighter);
  border-radius: 0;
  background: var(--md-default-bg-color);
  color: var(--md-default-fg-color--light);
  cursor: pointer;
  margin-left: -1px;
  transition: all 0.12s ease;
  user-select: none;
}
.ucm-install__group .ucm-install__opt:first-child { margin-left: 0; }
.ucm-install__opt:hover {
  color: var(--md-primary-fg-color);
  border-color: var(--md-primary-fg-color);
  z-index: 1;
}
.ucm-install__opt input { position: absolute; opacity: 0; pointer-events: none; }
.ucm-install__opt.is-active {
  background: var(--md-primary-fg-color);
  border-color: var(--md-primary-fg-color);
  color: var(--md-primary-bg-color, #fff);
  z-index: 2;
}
.ucm-install__output { padding: 0; background: var(--md-default-fg-color--lightest); }
.ucm-install__cmdhead {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px 16px;
  background: var(--md-default-fg-color--lighter);
}
.ucm-install__cmdtitle {
  font-size: 0.72rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--md-default-fg-color--light);
}
.ucm-install__copy {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 10px;
  font-size: 0.75rem;
  border-radius: 4px;
  border: 1px solid var(--md-default-fg-color);
  background: transparent;
  color: var(--md-default-fg-color--light);
  cursor: pointer;
  transition: all 0.12s ease;
}
.ucm-install__copy:hover {
  background: var(--md-primary-fg-color);
  border-color: var(--md-primary-fg-color);
  color: #fff;
}
.ucm-install__cmd {
  margin: 0;
  padding: 16px;
  background: var(--md-default-fg-color--lightest);
  border-radius: 0;
  overflow-x: auto;
}
.ucm-install__cmd code {
  font-family: var(--md-code-font-family, monospace);
  font-size: 0.82rem;
  line-height: 1.6;
  white-space: pre;
  color: var(--md-default-fg-color);
}
.ucm-install__note {
  margin: 0;
  padding: 10px 16px;
  font-size: 0.78rem;
  color: var(--md-default-fg-color--light);
  background: var(--md-default-bg-color);
  border-top: 1px solid var(--md-default-fg-color--lightest);
}
.ucm-install__note--err { color: var(--md-accent-fg-color); font-weight: 500; }
@media (max-width: 600px) {
  .ucm-install__row { flex-direction: column; align-items: stretch; gap: 6px; }
  .ucm-install__rowlabel { flex: none; }
}
</style>

## 后续步骤

安装器提供启动命令。关于引擎集成补丁、运行时配置和服务选项，请按照您选择的引擎指南操作：

- [vLLM (CUDA)](../engines/index.md)
- [vLLM Ascend (NPU)](../engines/index.md)
- [SGLang](../engines/index.md)
- [MindIE](../engines/index.md)

关于使用 Helm 的 Kubernetes 部署，请参见 [GLM PD 最佳实践](../model-tour/index.md)。