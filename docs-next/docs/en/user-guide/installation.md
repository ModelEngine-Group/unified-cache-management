# Installation

UCM is delivered as Docker images, a Helm chart, Python wheels, and source
builds. Use the installer below to get the exact command for your stack, then
follow the engine-specific guide for the remaining steps.

!!! tip "Single page, dynamic commands"

    This page follows the PyTorch "local installer" pattern: one page holds a
    static install matrix and a small front-end script renders the matching
    command for the selected combination — there is **one page per task, not one
    page per combination**. The same "static data + front-end script" approach
    already drives the [KV Cache Calculator](../toolkit/kv-cache-calculator.md).

## Installer

<div id="ucm-install-selector" class="ucm-install">
  <div class="ucm-install__rows" id="ucm-install-rows"></div>
  <div class="ucm-install__output">
    <div class="ucm-install__cmdhead">
      <span class="ucm-install__cmdtitle" id="ucm-cmd-label">Run this command</span>
      <button class="ucm-install__copy" id="ucm-copy" type="button" title="Copy command" aria-label="Copy command">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="16" height="16" aria-hidden="true"><path fill="currentColor" d="M19 3h-4.18A3 3 0 0 0 12 1a3 3 0 0 0-2.82 2H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V5a2 2 0 0 0-2-2m-7 0a1 1 0 0 1 1 1 1 1 0 0 1-1 1 1 1 0 0 1-1-1 1 1 0 0 1 1-1m7 16H5V5h2v2h10V5h2z"/></svg>
        <span class="ucm-install__copy-label">Copy</span>
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
    method:  ["Docker", "Helm chart", "pip wheel", "Source build"]
  };
  var DEFAULTS = { version:"0.5.0", engine:"vLLM", device:"GPU (CUDA)", os:"Ubuntu", arch:"amd64", method:"Docker" };
  var LABELS = { version:"UCM version", engine:"Engine", device:"Device", os:"Operating system", arch:"Architecture", method:"Install method" };
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
    if (m === "Source build") {
      return "git clone https://github.com/ModelEngine-Group/unified-cache-management.git\ncd unified-cache-management\npip install -e .\n# Then apply the engine integration patch (see engine guide)";
    }
    return null;
  }

  var NOTE = {
    "Helm chart": "Deploys UCM with Kubernetes. See the Deploy guide for the full Helm values reference.",
    "Source build": "Recommended for development. Apply the engine integration patch documented in the engine guide.",
    "pip wheel": "Prebuilt wheels are published with each release. Requires the matching engine already installed.",
    "Docker": "Prebuilt images bundle the engine. See the engine guide for runtime configuration."
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
      labelEl.textContent = s.method + " \u00b7 " + s.engine + " \u00b7 " + s.device;
      noteEl.textContent = NOTE[s.method] || "";
      noteEl.classList.remove("ucm-install__note--err");
    } else {
      codeEl.textContent = "";
      codeEl.parentElement.style.display = "none";
      copy.style.display = "none";
      labelEl.textContent = "Not available";
      noteEl.textContent = "This combination is not available. Try a different engine, device, or method.";
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
      lbl.textContent = "Copied";
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
/* Segmented control: buttons joined into one bar, divided by borders */
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

## Next steps

The installer gives you the bootstrap command. For engine integration patches,
runtime configuration, and serving options, follow the engine guide that
matches your selection:

- [vLLM (CUDA)](engines/vllm.md)
- [vLLM Ascend (NPU)](engines/vllm-ascend.md)
- [SGLang](engines/sglang.md)
- [MindIE](engines/mindie.md)

For Kubernetes deployment with Helm, see
[GLM PD Best Practice](model-tour/glm/glm-5-1.md).
