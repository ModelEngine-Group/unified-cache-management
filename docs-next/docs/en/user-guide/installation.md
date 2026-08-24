# Installation

UCM is delivered as Docker images, a Helm chart, Python wheels, and source
builds. Use the installer below to get the exact command for your stack, then
follow the engine-specific guide for the remaining steps. The available
integrated artifacts are the `0.7.58 Preview/Fork` release.

!!! tip "How to use"

    Select your UCM version, engine, compute platform, architecture, and
    install method below — the matching install command appears automatically.

## Installer

<div id="ucm-install-selector" class="ucm-install">
  <div class="ucm-install__rows" id="ucm-install-rows"></div>
  <section class="ucm-install__output" aria-labelledby="ucm-cmd-label">
    <div class="ucm-install__cmdhead">
      <span class="ucm-install__cmdtitle" id="ucm-cmd-label">Run this command</span>
      <button class="ucm-install__copy" id="ucm-copy" type="button" title="Copy command" aria-label="Copy command">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="16" height="16" aria-hidden="true"><path fill="currentColor" d="M19 3h-4.18A3 3 0 0 0 12 1a3 3 0 0 0-2.82 2H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V5a2 2 0 0 0-2-2m-7 0a1 1 0 0 1 1 1 1 1 0 0 1-1 1 1 1 0 0 1-1-1 1 1 0 0 1 1-1m7 16H5V5h2v2h10V5h2z"/></svg>
        <span class="ucm-install__copy-label">Copy</span>
      </button>
    </div>
    <pre class="ucm-install__cmd"><code id="ucm-cmd"></code></pre>
    <p class="ucm-install__note" id="ucm-note"></p>
    <p class="ucm-install__status" id="ucm-status" aria-live="polite" aria-atomic="true"></p>
  </section>
</div>

<script>
(function () {
  var VERSION = "0.7.58";
  var DIMS = {
    version: ["0.7.58 Preview/Fork"],
    engine:  ["vLLM", "vLLM Ascend", "SGLang"],
    device:  ["CUDA", "NPU A2", "NPU A3", "NPU A5"],
    arch:    ["amd64", "arm64"],
    method:  ["Docker Image", "Helm chart", "pip wheel", "Source build"]
  };
  var DEFAULTS = { version:"0.7.58 Preview/Fork", engine:"vLLM", device:"CUDA", arch:"amd64", method:"Docker Image" };
  var LABELS = { version:"UCM version", engine:"Engine", device:"Compute platform", arch:"Architecture", method:"Install method" };
  var ORDER = ["version","engine","device","arch","method"];
  var ENGINE_DEVICES = {
    "vLLM": ["CUDA"],
    "vLLM Ascend": ["NPU A2", "NPU A3", "NPU A5"],
    "SGLang": ["CUDA"]
  };
  var UNAVAILABLE_COMPUTE = {
    "NPU A5": "No UCM-integrated A5 image, wheel profile, Dockerfile, or setup platform is published."
  };
  var DOCKER_IMAGES = {
    "vLLM|CUDA": "ghcr.io/supermarioyl/vllm-openai:v0.21.0-ucm-0.7.58-r1",
    "vLLM Ascend|NPU A2": "ghcr.io/supermarioyl/vllm-ascend:v0.22.1rc1-ucm-0.7.58-r1",
    "vLLM Ascend|NPU A3": "ghcr.io/supermarioyl/vllm-ascend:v0.22.1rc1-a3-ucm-0.7.58-r1"
  };
  var RELEASE_URL = "https://github.com/SuperMarioYL/unified-cache-management/releases/download/v" + VERSION + "/";

 function cmd(state) {
   var e = state.engine, d = state.device, a = state.arch, m = state.method;
    if (UNAVAILABLE_COMPUTE[d]) return null;
   if (!ENGINE_DEVICES[e] || ENGINE_DEVICES[e].indexOf(d) === -1) return null;
    if (m === "Docker Image") {
      var image = DOCKER_IMAGES[e + "|" + d];
      if (!image) return null;
      return "docker pull " + image;
    }
    if (m === "Helm chart") {
      return "helm install ucm " + RELEASE_URL + "unified-cache-pd-0.7.58.tgz";
    }
    if (m === "pip wheel") {
      var wheelArch = a === "amd64" ? "x86_64" : "aarch64";
      var wheel = d === "CUDA" ? "uc_manager_cuda-0.7.58-cp312-cp312-manylinux_2_28_" + wheelArch + ".whl" :
        d === "NPU A2" ? "uc_manager_cann_a2-0.7.58-cp312-cp312-linux_" + wheelArch + ".whl" :
        d === "NPU A3" ? "uc_manager_cann_a3-0.7.58-cp312-cp312-linux_" + wheelArch + ".whl" : null;
      return wheel ? "pip install " + RELEASE_URL + wheel : null;
    }
    if (m === "Source build") {
      var platform = d === "CUDA" ? "cuda" : d === "NPU A2" ? "ascend" : d === "NPU A3" ? "ascend-a3" : null;
      return platform ? "git clone --branch v0.7.58 https://github.com/SuperMarioYL/unified-cache-management.git\ncd unified-cache-management\nPLATFORM=" + platform + " pip install -e ." : null;
    }
    return null;
  }

  var NOTE = {
    "Helm chart": "Uses the verified Preview/Fork chart artifact. See the Deploy guide for runtime values.",
    "Source build": "Builds the Preview/Fork tag with the setup platform selected by compute platform.",
    "pip wheel": "Uses the exact Preview/Fork GitHub Release wheel matching the selected compute platform and CPU architecture.",
    "Docker Image": "Verified Preview/Fork images: the OCI index covers amd64 and arm64, the image userland is fixed by the image, and no official ModelEngine-Group UCM GHCR package was publicly readable on 2026-08-20."
  };

  function unavailableReason(s) {
    if (s.method === "Docker Image" && s.engine === "SGLang") {
      return "SGLang Docker Image is unavailable because no published UCM SGLang image was found.";
    }
    return "This combination is not available. Try a different engine, compute platform, or install method.";
  }

  function buildRows() {
    var root = document.getElementById("ucm-install-rows");
    root.innerHTML = "";
    ORDER.forEach(function (dim) {
      var row = document.createElement("fieldset");
      row.className = "ucm-install__row";
      var label = document.createElement("legend");
      label.className = "ucm-install__rowlabel";
      label.textContent = LABELS[dim];
      var group = document.createElement("div");
      group.className = "ucm-install__group ucm-install__group--" + dim;
      DIMS[dim].forEach(function (opt) {
        var id = "ucm-" + dim + "-" + opt.replace(/[^a-z0-9]/gi,"");
        var lab = document.createElement("label");
        lab.className = "ucm-install__opt";
        lab.htmlFor = id;
        var inp = document.createElement("input");
        inp.type = "radio"; inp.name = "ucm-" + dim; inp.value = opt; inp.id = id;
        if (opt === DEFAULTS[dim]) inp.checked = true;
        var sp = document.createElement("span");
        if (dim === "device" && UNAVAILABLE_COMPUTE[opt]) {
          inp.disabled = true;
          lab.classList.add("ucm-install__opt--disabled");
          lab.title = UNAVAILABLE_COMPUTE[opt];
          inp.setAttribute("aria-label", opt + " unavailable: " + UNAVAILABLE_COMPUTE[opt]);
          sp.textContent = opt + " · N/A";
        } else {
          sp.textContent = opt;
        }
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

  var copyResetTimer;

  function render() {
    var s = state();
    var code = cmd(s);
    var codeEl = document.getElementById("ucm-cmd");
    var labelEl = document.getElementById("ucm-cmd-label");
    var noteEl = document.getElementById("ucm-note");
    var copy = document.getElementById("ucm-copy");
    var status = document.getElementById("ucm-status");
    if (code) {
      codeEl.textContent = code;
      codeEl.parentElement.hidden = false;
      copy.hidden = false;
      labelEl.textContent = s.method + " \u00b7 " + s.engine + " \u00b7 " + s.device;
      noteEl.textContent = NOTE[s.method] || "";
      noteEl.classList.remove("ucm-install__note--err");
      status.textContent = "Command updated: " + s.method + " for " + s.engine + " on " + s.device + ".";
    } else {
      codeEl.textContent = "";
      codeEl.parentElement.hidden = true;
      copy.hidden = true;
      labelEl.textContent = "Not available";
      noteEl.textContent = unavailableReason(s);
      noteEl.classList.add("ucm-install__note--err");
      status.textContent = noteEl.textContent;
    }
  }

  function onEngineChange() {
    var s = state();
    var compatible = ENGINE_DEVICES[s.engine] || [];
    if (compatible.indexOf(s.device) === -1) {
      var target = document.querySelector('input[name="ucm-device"][value="'+compatible[0]+'"]');
      if (target) target.checked = true;
    }
  }

  async function copyCommand() {
    var text = document.getElementById("ucm-cmd").textContent;
    var label = this.querySelector(".ucm-install__copy-label");
    var status = document.getElementById("ucm-status");
    var feedback;
    if (!navigator.clipboard || !navigator.clipboard.writeText) {
      feedback = "Copy unavailable";
    } else {
      try {
        await navigator.clipboard.writeText(text);
        feedback = "Copied";
      } catch (err) {
        feedback = "Copy failed";
      }
    }
    label.textContent = feedback;
    status.textContent = feedback;
    clearTimeout(copyResetTimer);
    copyResetTimer = setTimeout(function(){
      label.textContent = "Copy";
      status.textContent = "";
      copyResetTimer = null;
    }, 1500);
  }

  function init() {
    var root = document.getElementById("ucm-install-selector");
    if (!root || root.dataset.ready) return;
    root.dataset.ready = "1";
    buildRows();
    root.addEventListener("change", function(ev){
      if (ev.target.matches('input[type=radio]')) {
        var dim = ev.target.name.replace("ucm-","");
        if (dim === "engine") onEngineChange();
        render();
      }
    });
    document.getElementById("ucm-copy").addEventListener("click", copyCommand);
    render();
  }

  if (document.readyState !== "loading") init();
  else document.addEventListener("DOMContentLoaded", init);
  if (typeof document$ !== "undefined") document$.subscribe(init);
})();
</script>

<style>
.ucm-install {
  --ucm-install-muted: var(--md-default-fg-color--light);
  --ucm-install-accent: #00695c;
  --ucm-install-accent-contrast: #fff;
  --ucm-install-focus: #00695c;
  --ucm-install-control-height: 2.15rem;
  --ucm-install-option-bg: #edf1f0;
  --ucm-install-option-fg: #4e5c59;
  --ucm-install-terminal: #08201f;
  --ucm-install-terminal-rule: #1f504c;
  --ucm-install-terminal-fg: #e3f5f0;
  margin: 1.2em 0;
  max-width: 100%;
  border: 0;
  border-radius: 0;
  overflow: visible;
  background: transparent;
}
[data-md-color-scheme="slate"] .ucm-install {
  --ucm-install-muted: var(--md-default-fg-color--light);
  --ucm-install-accent: #62d9c8;
  --ucm-install-accent-contrast: #08201f;
  --ucm-install-focus: #8be7d8;
  --ucm-install-option-bg: #263431;
  --ucm-install-option-fg: #d2e3df;
}
.ucm-install *, .ucm-install *::before, .ucm-install *::after { box-sizing: border-box; }
.ucm-install__rows { display: flex; flex-direction: column; gap: .08rem; }
.ucm-install__row {
  display: block;
  position: relative;
  min-inline-size: 0;
  margin: 0;
  padding: .16rem 0 .16rem 8.75rem;
  border: 0;
}
.ucm-install__rowlabel {
  position: absolute;
  inset-block-start: 50%;
  inset-inline-start: 0;
  transform: translateY(-50%);
  min-width: 0;
  padding: 0;
  font-size: 0.82rem;
  font-weight: 600;
  color: var(--ucm-install-muted);
}
.ucm-install__group { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: .35rem; min-width: 0; }
.ucm-install__group--version { display: flex; }
.ucm-install__group--engine { grid-template-columns: repeat(3, minmax(0, 1fr)); }
.ucm-install__group--device,
.ucm-install__group--method { grid-template-columns: repeat(4, minmax(0, 1fr)); }
.ucm-install__opt {
  position: relative;
  height: var(--ucm-install-control-height);
  min-width: 0;
  cursor: pointer;
}
.ucm-install__group--version .ucm-install__opt { width: max-content; }
.ucm-install__opt span {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 100%;
  height: 100%;
  padding: 0 .5rem;
  border: 0;
  border-radius: 4px;
  background: var(--ucm-install-option-bg);
  color: var(--ucm-install-option-fg);
  font-size: .78rem;
  font-weight: 500;
  line-height: 1;
  text-align: center;
  white-space: nowrap;
  transition: background-color .16s ease, border-color .16s ease, color .16s ease, box-shadow .16s ease;
  user-select: none;
}
.ucm-install__opt:hover span {
  color: var(--ucm-install-accent);
  background: color-mix(in srgb, var(--ucm-install-accent) 10%, var(--ucm-install-option-bg));
}
.ucm-install__opt input {
  position: absolute;
  inline-size: 1px;
  block-size: 1px;
  margin: -1px;
  opacity: 0;
  clip-path: inset(50%);
}
.ucm-install__opt input:checked + span {
  background: var(--ucm-install-accent);
  color: var(--ucm-install-accent-contrast);
}
.ucm-install__opt--disabled { cursor: not-allowed; }
.ucm-install__opt--disabled span {
  color: var(--ucm-install-muted);
  opacity: .62;
  text-decoration: line-through;
}
.ucm-install label:focus-within span {
  outline: 2px solid var(--ucm-install-focus);
  outline-offset: 2px;
  position: relative;
  z-index: 1;
}
.ucm-install__output { margin-top: .7rem; overflow: hidden; border-radius: 6px; background: var(--ucm-install-terminal); color: var(--ucm-install-terminal-fg); }
.ucm-install__cmdhead {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  padding: .75rem 1.15rem;
  border-bottom: 1px solid var(--ucm-install-terminal-rule);
}
.ucm-install__cmdtitle {
  font-size: 0.72rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: #a8d6ce;
}
.ucm-install__copy {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  flex: 0 0 auto;
  padding: .32rem .62rem;
  font-size: 0.75rem;
  border: 1px solid #4b827a;
  border-radius: 4px;
  background: transparent;
  color: #d8f1eb;
  cursor: pointer;
  transition: background-color .16s ease, border-color .16s ease, color .16s ease;
}
.ucm-install__copy:hover {
  background: var(--ucm-install-accent);
  border-color: var(--ucm-install-accent);
  color: var(--ucm-install-accent-contrast);
}
.ucm-install__copy:focus-visible { outline: 2px solid #b8f1e6; outline-offset: 2px; }
.ucm-install__cmd {
  margin: 0;
  max-width: 100%;
  padding: 1rem 1.15rem;
  background: transparent;
  border-radius: 0;
  overflow-x: auto;
  overflow-y: hidden;
  overflow-wrap: normal;
  word-break: normal;
}
.ucm-install__cmd .md-clipboard, .ucm-install__cmd .md-code__button { display: none; }
.ucm-install__cmd code {
  display: block;
  width: max-content;
  min-width: 100%;
  overflow: visible !important;
  background: transparent;
  font-family: var(--md-code-font-family, monospace);
  font-size: 0.82rem;
  line-height: 1.6;
  white-space: pre;
  word-break: normal;
  color: var(--ucm-install-terminal-fg);
}
.ucm-install__note {
  margin: 0;
  padding: .72rem 1.15rem;
  font-size: 0.78rem;
  color: #b9d5cf;
  border-top: 1px solid var(--ucm-install-terminal-rule);
}
.ucm-install__note--err { color: #ffc6c1; font-weight: 500; }
.ucm-install__status {
  position: absolute;
  inline-size: 1px;
  block-size: 1px;
  overflow: hidden;
  clip: rect(0 0 0 0);
  clip-path: inset(50%);
  white-space: nowrap;
}
@media (max-width: 720px) {
  .ucm-install__row { padding: .2rem 0; }
  .ucm-install__rowlabel { position: static; transform: none; margin-bottom: .22rem; }
  .ucm-install__group { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: .35rem; }
  .ucm-install__group--version { display: flex; }
  .ucm-install__group--engine,
  .ucm-install__group--method { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
@media (max-width: 360px) {
  .ucm-install__cmdhead, .ucm-install__cmd, .ucm-install__note { padding-inline: .85rem; }
  .ucm-install__group { grid-template-columns: 1fr; }
}
@media (prefers-reduced-motion: reduce) {
  .ucm-install__opt span, .ucm-install__copy { transition: none; }
}
</style>

## Next steps

The installer gives you the bootstrap command. For engine integration patches,
runtime configuration, and serving options, follow the engine guide that
matches your selection:

- [vLLM (CUDA)](engines/index.md)
- [vLLM Ascend (NPU)](engines/index.md)
- [SGLang](engines/index.md)

For Kubernetes deployment with Helm, see
[GLM PD Best Practice](model-tour/index.md).
