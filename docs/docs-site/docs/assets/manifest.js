(function (root, factory) {
  "use strict";

  var api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  root.UcmReleaseManifest = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  var KIND = "ucm-release-manifest";
  var SCHEMA_VERSION = 9;
  var manifestScriptSource =
    typeof document !== "undefined" && document.currentScript
      ? document.currentScript.src
      : null;

  function validateManifest(value) {
    if (!value || value.kind !== KIND || value.schema_version !== SCHEMA_VERSION) {
      throw new TypeError("release manifest schema_version must be " + SCHEMA_VERSION);
    }
    if (!value.release || !value.python || !("chart" in value) ||
        !Array.isArray(value.wheels) || !Array.isArray(value.images)) {
      throw new TypeError("release manifest is incomplete");
    }
    return value;
  }

  function defaultManifestUrl(scriptSource) {
    var source = scriptSource;
    if (!source && typeof document !== "undefined" && document.currentScript) {
      source = document.currentScript.src;
    }
    var base =
      typeof document !== "undefined" ? document.baseURI : "https://example.invalid/assets/";
    var scriptUrl = new URL(source || manifestScriptSource || "assets/manifest.js", base);
    return new URL("../release-manifest.json", scriptUrl);
  }

  function loadManifest(url, fetchImplementation) {
    var target = url ? new URL(String(url), defaultManifestUrl()) : defaultManifestUrl();
    var request = fetchImplementation || (typeof fetch === "function" ? fetch : null);
    if (!request) return Promise.reject(new Error("Fetch API unavailable"));
    return request(target.href, { cache: "no-store" })
      .then(function (response) {
        if (!response.ok) throw new Error("Release Manifest unavailable");
        return response.json();
      })
      .then(validateManifest);
  }

  function acceleratorKey(accelerator) {
    return [accelerator.runtime, accelerator.variant].join("|");
  }

  function acceleratorLabel(accelerator) {
    var runtime = accelerator.runtime.replace("-", " ").toUpperCase();
    return accelerator.variant === "default"
      ? runtime
      : runtime + " / " + accelerator.variant.toUpperCase();
  }

  function osKey(os) {
    return os.id + "|" + os.version;
  }

  function osLabel(os) {
    return os.id.charAt(0).toUpperCase() + os.id.slice(1) + " " + os.version;
  }

  function productLabel(product) {
    if (product === "vllm-ascend") return "vLLM-Ascend";
    if (product === "vllm") return "vLLM";
    return product;
  }

  function preferredPublication(image) {
    if (image.publications.ghcr) {
      return { channel: "ghcr", publication: image.publications.ghcr };
    }
    if (image.publications.dockerhub) {
      return { channel: "dockerhub", publication: image.publications.dockerhub };
    }
    return null;
  }

  return {
    KIND: KIND,
    SCHEMA_VERSION: SCHEMA_VERSION,
    validateManifest: validateManifest,
    defaultManifestUrl: defaultManifestUrl,
    loadManifest: loadManifest,
    acceleratorKey: acceleratorKey,
    acceleratorLabel: acceleratorLabel,
    osKey: osKey,
    osLabel: osLabel,
    productLabel: productLabel,
    preferredPublication: preferredPublication,
  };
});
