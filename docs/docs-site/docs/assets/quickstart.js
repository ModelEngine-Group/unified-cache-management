/* One environment selection drives both installation paths and the Markdown guide. */
(function () {
  var Selector = window.UcmInstallSelector;
  var Manifest = window.UcmReleaseManifest;
  var controllers = new WeakMap();

  function linkedGuide() {
    var target = document.getElementById(decodeURIComponent(location.hash.slice(1)));
    return target && target.closest("[data-quickstart-guide]");
  }

  function initialize() {
    var app = document.getElementById("ucm-install-app");
    if (!app || controllers.has(app)) return;
    var messages = Selector.TEXT[app.dataset.locale === "zh" ? "zh" : "en"];
    var guides = Array.from(document.querySelectorAll("[data-quickstart-guide]"));
    var templates = new Map();
    document.querySelectorAll("[data-command-template]").forEach(function (container) {
      templates.set(container, container.querySelector("code").textContent);
    });
    var model = {manifest: null, combinations: []};
    var linked = linkedGuide();
    var requested = linked ? {engine: linked.dataset.quickstartGuide} : {};
    var status = app.querySelector("[data-install-status]");

    function render() {
      var selection = Selector.deriveSelection(model, requested);
      requested = selection.state;
      status.hidden = requested.engine === "sglang";
      Selector.renderSelector(app, selection, messages, function (name, value) {
        requested = Object.assign({}, requested);
        requested[name] = value;
        render();
        if (name === "engine") history.replaceState(null, "", "#" + value);
        // Re-rendering radio rows should not lose keyboard focus.
        var input = Array.from(app.querySelectorAll("input")).find(function (node) {
          return node.name === "ucm-selector-" + name && node.value === requested[name];
        });
        if (input) input.focus({preventScroll: true});
      });
      var combination = selection.combination;
      var pythonVersion = selection.wheel
        ? selection.wheel.python_abi.replace(/^cp(\d)(\d+)$/, "$1.$2") : "";
      var values = combination ? {
        image: selection.imageReference,
        standard_image: selection.standardImageReference,
        docker_platform: "linux/" + requested.architecture,
        engine_package: requested.engine,
        engine_version: combination.engineVersionLabel,
        runtime: combination.runtimeLabel,
        variant: combination.variantLabel,
        os: combination.osLabel,
        architecture: combination.architectureLabel,
        python_version: pythonVersion || "—",
        pip_install: selection.pipCommand
          ? selection.pipCommand.replace(/^pip /, "python" + pythonVersion + " -m pip ") : "",
      } : {};
      guides.forEach(function (guide) {
        guide.hidden = guide.dataset.quickstartGuide !== requested.engine;
        if (guide.dataset.quickstartGuide === "sglang" || guide.hidden) return;
        guide.querySelector("[data-environment-summary]").hidden = !combination;
        guide.querySelectorAll("[data-env-value]").forEach(function (node) {
          node.textContent = values[node.dataset.envValue] || "—";
        });
        guide.querySelectorAll("[data-requires]").forEach(function (node) {
          node.hidden = !selection[node.dataset.requires];
        });
        guide.querySelectorAll("[data-artifact-missing]").forEach(function (node) {
          node.hidden = !!selection[node.dataset.artifactMissing];
        });
        guide.querySelectorAll("[data-command-template]").forEach(function (container) {
          // Missing artifacts keep their command sections hidden, including stale commands.
          if (!selection[container.closest("[data-requires]").dataset.requires]) return;
          var command = templates.get(container).replace(/\{\{ (\w+) \}\}/g, function (_, key) {
            return values[key];
          });
          container.replaceChildren(Selector.commandBlock(command, messages));
        });
      });
      app.dataset.selectedEngine = requested.engine;
    }

    controllers.set(app, function (engine) {
      requested = Object.assign({}, requested, {engine: engine});
      render();
    });
    render();
    Manifest.loadManifest().then(function (manifest) {
      var repositories = document.getElementById("ucm-runtime-repositories");
      model = Selector.buildSelectorModel(
        manifest, repositories ? JSON.parse(repositories.textContent) : {}
      );
      render();
      var releaseLink = document.createElement("a");
      releaseLink.href = manifest.release.url;
      releaseLink.textContent = manifest.release.version;
      status.replaceChildren(document.createTextNode(messages.release + ": "), releaseLink);
      status.className = "ucm-install__status ucm-install__status--ready";
      app.dataset.manifestReady = "1";
      openLinkedGuide();
    }).catch(function (error) {
      status.textContent = error instanceof TypeError ? messages.invalid : messages.unavailable;
      status.className = "ucm-install__status ucm-install__status--unavailable";
    });
  }

  function openLinkedGuide() {
    var guide = linkedGuide();
    var select = controllers.get(document.getElementById("ucm-install-app"));
    if (guide && select) {
      select(guide.dataset.quickstartGuide);
      var target = document.getElementById(decodeURIComponent(location.hash.slice(1)));
      var block = target.closest(".tabbed-block");
      if (block) {
        var tabs = block.closest(".tabbed-set");
        var index = Array.from(block.parentElement.children).indexOf(block);
        var input = tabs.querySelectorAll(":scope > input")[index];
        if (!input.checked) input.click();
      }
      target.scrollIntoView();
    }
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", initialize);
  else initialize();
  if (typeof document$ !== "undefined") document$.subscribe(initialize);
  window.addEventListener("hashchange", openLinkedGuide);
})();
