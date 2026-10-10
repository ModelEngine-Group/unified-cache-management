(function (root, factory) {
  "use strict";

  var manifestApi = root.UcmReleaseManifest;
  if (!manifestApi && typeof module === "object" && module.exports) {
    manifestApi = require("./manifest.js");
  }
  var api = factory(manifestApi);
  if (typeof module === "object" && module.exports) module.exports = api;
  root.UcmInstallSelector = api;

  if (typeof document !== "undefined") {
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", api.initialize);
    } else {
      api.initialize();
    }
    if (typeof document$ !== "undefined") document$.subscribe(api.initialize);
  }
})(typeof globalThis !== "undefined" ? globalThis : this, function (Manifest) {
  "use strict";

  if (!Manifest) throw new Error("UcmReleaseManifest must be loaded before install.js");

  var COPY_ICON = "\u2398";
  var ENGINE_ORDER = ["vllm", "vllm-ascend", "sglang"];
  var ROW_ORDER = ["engine", "engineVersion", "runtime", "variant", "os", "architecture"];
  var TEXT = {
    en: {
      loading: "Loading the current release manifest...",
      unavailable: "No release manifest is published for this documentation version yet.",
      invalid: "The published release manifest is invalid.",
      release: "Release", engine: "Engine", engineVersion: "Engine Version",
      runtime: "CUDA / CANN", variant: "Ascend Device", os: "OS", architecture: "Architecture",
      copy: "Copy", copied: "Copied", copyFailed: "Copy failed",
      noChart: "This release does not include a Helm Chart.",
      noToolkit: "This release has no Toolkit package. Use the source installation command below.",
    },
    zh: {
      loading: "正在加载当前版本的 Release Manifest……",
      unavailable: "此文档版本尚未发布 Release Manifest。",
      invalid: "已发布的 Release Manifest 格式无效。",
      release: "Release", engine: "推理引擎", engineVersion: "引擎版本",
      runtime: "CUDA / CANN", variant: "昇腾设备", os: "操作系统", architecture: "CPU 架构",
      copy: "复制", copied: "已复制", copyFailed: "复制失败",
      noChart: "本次发布未提供 Helm Chart。",
      noToolkit: "本次发布未提供 Toolkit 包，请使用下方的源码安装命令。",
    },
  };

  function element(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function architectureLabel(value) {
    return { amd64: "x86_64", arm64: "aarch64" }[value] || value;
  }

  function runtimeLabel(value) {
    return value.replace("-", " ").toUpperCase();
  }

  function selectableProduct(artifact) {
    return ["vllm", "vllm-ascend"].includes(artifact.product) &&
      ![artifact.accelerator.variant, artifact.accelerator.soc_version].some(function (value) {
        return /(^|[-_.])a5($|[-_.])/i.test(String(value));
      });
  }

  function wheelInstallCommand(wheel, manifest) {
    if (!wheel) return null;
    if (!manifest.python.pypi) {
      return 'pip install "' + wheel.url + '"' +
        (manifest.toolkit ? ' "' + manifest.toolkit.url + '"' : "");
    }
    var dependencyIndex = "https://pypi.org/simple";
    var indexOption = manifest.python.pypi.index_url !== dependencyIndex
      ? " --index-url " + manifest.python.pypi.index_url + " --extra-index-url " + dependencyIndex
      : "";
    return "pip install" + indexOption + ' "' + manifest.python.distribution +
      "[" + wheel.extra + (manifest.toolkit ? ",toolkit" : "") + "]==" + manifest.python.version + '"';
  }

  function standardImageReference(image, repositories) {
    var prefix = image.product + "-";
    var repository = repositories[image.product];
    // Schema 9 family IDs retain the product prefix and full upstream runtime tag.
    if (!repository || !image.id.startsWith(prefix)) return null;
    var tag = image.id.slice(prefix.length);
    if (!/^(v\d|nightly-releases-v\d)/.test(tag)) return null;
    return repository + ":" + tag;
  }

  function buildSelectorModel(manifest, runtimeRepositories) {
    Manifest.validateManifest(manifest);
    var combinations = [];
    manifest.images.filter(selectableProduct).forEach(function (image) {
      var os = Manifest.osKey(image.os);
      var osLabel = image.os.version === "unreported"
        ? {linux: "Linux", openeuler: "openEuler"}[image.os.id] || image.os.id
        : Manifest.osLabel(image.os);
      // Published default/ubuntu2404 tags can share Linux/unreported metadata.
      // Preserve the explicit tag distinction without guessing the default distro.
      var namedVariant = image.os.id === "linux" && image.os.version === "unreported" &&
        image.id.match(/(?:^|-)(ubuntu[0-9]+)(?:-|$)/);
      if (namedVariant) {
        os += "|" + namedVariant[1];
        osLabel += " / " + namedVariant[1];
      }
      // A registry fallback may cover a different architecture, but never a different image.
      var targets = new Map();
      ["ghcr", "dockerhub"].forEach(function (channel) {
        var publication = image.publications[channel];
        if (!publication) return;
        publication.members.forEach(function (member) {
          if (!targets.has(member.architecture)) targets.set(member.architecture, publication.pull);
        });
      });
      targets.forEach(function (pull, architecture) {
        var wheels = manifest.wheels.filter(function (wheel) {
          return wheel.product === image.product && wheel.architecture === architecture &&
            wheel.accelerator.runtime === image.accelerator.runtime &&
            wheel.accelerator.variant === image.accelerator.variant;
        });
        if (wheels.length > 1) throw new TypeError("Multiple Wheels match image " + image.id + " / " + architecture);
        var channel = image.upstream.channel;
        var channelLabel = {stable: "Stable", rc: "RC", nightly: "Nightly"}[channel] || channel;
        combinations.push({
          engine: image.product,
          engineVersion: image.upstream.version + "|" + channel,
          engineVersionLabel: image.upstream.version + " / " + channelLabel,
          runtime: image.accelerator.runtime,
          runtimeLabel: runtimeLabel(image.accelerator.runtime),
          variant: image.accelerator.variant,
          variantLabel: image.accelerator.variant.toUpperCase(),
          os: os,
          osLabel: osLabel,
          architecture: architecture,
          architectureLabel: architectureLabel(architecture),
          image: image, wheel: wheels[0] || null, imageReference: pull,
          standardImageReference: standardImageReference(image, runtimeRepositories || {}),
        });
      });
    });
    return { manifest: manifest, combinations: combinations };
  }

  function choose(requested, options) {
    var available = options.filter(function (option) { return !option.disabled; });
    return available.some(function (option) { return option.value === requested; })
      ? requested : available.length ? available[0].value : null;
  }

  function deriveSelection(model, requestedState) {
    var requested = requestedState || {};
    var engineOptions = ENGINE_ORDER.map(function (engine) {
      return {value: engine, label: engine === "sglang" ? "SGLang" : Manifest.productLabel(engine)};
    });
    var state = {engine: choose(requested.engine, engineOptions)};
    var rows = {engine: {visible: true, options: engineOptions}};
    var universe = model.combinations.filter(function (item) { return item.engine === state.engine; });
    var candidates = universe;
    ROW_ORDER.slice(1).forEach(function (field) {
      var labels = new Map(universe.map(function (item) { return [item[field], item[field + "Label"]]; }));
      var options = Array.from(labels, function (pair) {
        return {value: pair[0], label: pair[1], disabled: !candidates.some(function (item) { return item[field] === pair[0]; })};
      }).sort(function (a, b) { return a.label.localeCompare(b.label, undefined, {numeric: true}); });
      state[field] = choose(requested[field], options);
      rows[field] = {
        visible: universe.length > 0 && (field !== "variant" || state.engine === "vllm-ascend"),
        options: options,
      };
      candidates = candidates.filter(function (item) { return item[field] === state[field]; });
    });
    if (candidates.length > 1) throw new TypeError("Multiple images match the selected environment");
    var combination = candidates[0] || null;
    return {
      state: state, rows: rows,
      combination: combination,
      image: combination ? combination.image : null,
      wheel: combination ? combination.wheel : null,
      imageReference: combination ? combination.imageReference : null,
      standardImageReference: combination ? combination.standardImageReference : null,
      standard: !!(combination && combination.standardImageReference && combination.wheel),
      toolkit: !!(combination && model.manifest.toolkit),
      pipCommand: combination ? wheelInstallCommand(combination.wheel, model.manifest) : null,
    };
  }

  function commandBlock(command, messages) {
    var wrapper = element("div", "ucm-install__command");
    var pre = element("pre", "ucm-install__pre");
    pre.appendChild(element("code", "", command));
    var button = element("button", "ucm-install__copy");
    button.type = "button";
    button.setAttribute("aria-label", messages.copy);
    button.appendChild(element("span", "ucm-install__copy-icon", COPY_ICON));
    button.appendChild(element("span", "ucm-install__copy-label", messages.copy));
    button.addEventListener("click", function () {
      var label = button.querySelector(".ucm-install__copy-label");
      var operation =
        navigator.clipboard && navigator.clipboard.writeText
          ? navigator.clipboard.writeText(command)
          : Promise.reject(new Error("Clipboard API unavailable"));
      operation.then(
        function () {
          label.textContent = messages.copied;
        },
        function () {
          label.textContent = messages.copyFailed;
        }
      );
      window.setTimeout(function () {
        label.textContent = messages.copy;
      }, 1600);
    });
    wrapper.appendChild(pre);
    wrapper.appendChild(button);
    return wrapper;
  }

  function directLink(url, label) {
    var link = element("a", "ucm-install__link", label);
    link.href = url;
    link.rel = "noopener";
    return link;
  }

  function renderRow(name, row, selected, messages, onSelect) {
    var wrapper = element("div", "ucm-selector__row");
    wrapper.dataset.selectorRow = name;
    if (!row.visible) wrapper.hidden = true;
    wrapper.appendChild(element("div", "ucm-selector__label", messages[name]));
    var options = element("div", "ucm-selector__options");
    options.setAttribute("role", "radiogroup");
    options.setAttribute("aria-label", messages[name]);
    row.options.forEach(function (option) {
      var labelText = option.label;
      var optionLabel = element("label", "ucm-selector__option");
      var input = element("input", "ucm-selector__input");
      input.type = "radio";
      input.name = "ucm-selector-" + name;
      input.value = option.value;
      input.dataset.selectorOption = option.value;
      input.setAttribute("aria-label", labelText);
      input.checked = option.value === selected;
      input.disabled = option.disabled;
      optionLabel.title = labelText;
      if (input.checked) optionLabel.classList.add("ucm-selector__option--selected");
      if (input.disabled) optionLabel.classList.add("ucm-selector__option--disabled");
      input.addEventListener("change", function () {
        if (input.checked) onSelect(name, option.value);
      });
      optionLabel.appendChild(input);
      var optionText = element("span", "ucm-selector__option-text", labelText);
      optionLabel.appendChild(optionText);
      options.appendChild(optionLabel);
    });
    wrapper.appendChild(options);
    return wrapper;
  }

  function renderSelector(app, selection, messages, onSelect) {
    var controls = app.querySelector("[data-install-selector]");
    controls.replaceChildren();
    ROW_ORDER.forEach(function (name) {
      controls.appendChild(renderRow(name, selection.rows[name], selection.state[name], messages, onSelect));
    });
  }

  function toolkitInstallCommands(manifest) {
    if (!manifest.toolkit) return [];
    var packageInfo = manifest.toolkit;
    if (!manifest.python.pypi) return ['pip install "' + packageInfo.url + '"'];
    var index = manifest.python.pypi.index_url;
    var indexOption = index === "https://pypi.org/simple" ? "" : " --index-url " + index;
    return [
      'pip install' + indexOption + ' "' + packageInfo.distribution + '==' + packageInfo.version + '"',
      'pip install' + indexOption + ' "' + manifest.python.distribution + '[toolkit]==' + packageInfo.version + '"'
    ];
  }

  function initializeToolkit() {
    var app = document.querySelector("[data-toolkit-install]");
    if (!app || app.dataset.manifestLoading) return;
    app.dataset.manifestLoading = "1";
    var messages = TEXT[app.dataset.locale === "zh" ? "zh" : "en"];
    Manifest.loadManifest(Manifest.defaultManifestUrl()).then(function (manifest) {
      var commands = toolkitInstallCommands(manifest);
      app.replaceChildren();
      if (!commands.length) {
        app.appendChild(element("p", "", messages.noToolkit));
        return;
      }
      app.appendChild(directLink(manifest.release.url, messages.release + ": " + manifest.release.version));
      commands.forEach(function (command) { app.appendChild(commandBlock(command, messages)); });
      app.dataset.manifestReady = "1";
    }).catch(function () { app.textContent = messages.unavailable; });
  }

  function chartDownloadCommand(manifest) {
    if (!manifest.chart) return null;
    return 'helm pull "' + manifest.chart.url + '" --untar\ncd ' + manifest.chart.name;
  }

  function initializeChart() {
    var app = document.querySelector("[data-chart-install]");
    if (!app || app.dataset.manifestLoading) return;
    app.dataset.manifestLoading = "1";
    var messages = TEXT[app.dataset.locale === "zh" ? "zh" : "en"];
    Manifest.loadManifest().then(function (manifest) {
      var command = chartDownloadCommand(manifest);
      if (!command) {
        app.textContent = messages.noChart;
        return;
      }
      app.replaceChildren(
        directLink(manifest.chart.url, manifest.chart.filename),
        commandBlock(command, messages)
      );
      app.dataset.manifestReady = "1";
    }).catch(function () { app.textContent = messages.unavailable; });
  }

  function initialize() {
    initializeToolkit();
    initializeChart();
  }

  return {
    TEXT: TEXT,
    ROW_ORDER: ROW_ORDER.slice(),
    architectureLabel: architectureLabel,
    toolkitInstallCommands: toolkitInstallCommands,
    chartDownloadCommand: chartDownloadCommand,
    buildSelectorModel: buildSelectorModel,
    deriveSelection: deriveSelection,
    renderSelector: renderSelector,
    commandBlock: commandBlock,
    initialize: initialize,
  };
});
