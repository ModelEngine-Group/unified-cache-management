const assert = require("node:assert/strict");
const test = require("node:test");

const Manifest = require("../docs/assets/manifest.js");
const Selector = require("../docs/assets/install.js");

function publication(reference, architectures) {
  return {
    pull: reference,
    multi_arch: architectures.length > 1,
    members: architectures.map((architecture) => ({
      architecture,
      reference:
        architectures.length === 1 ? reference : reference + "-" + architecture,
    })),
  };
}

function wheel({ id, product, extra, runtime, variant, soc, architecture }) {
  const distribution =
    product === "vllm" ? "uc-manager-cuda-" + extra : "uc-manager-" + extra;
  const platformTag =
    (product === "vllm" ? "manylinux_2_28_" : "manylinux_2_34_") +
    (architecture === "amd64" ? "x86_64" : "aarch64");
  const filename =
    distribution.replaceAll("-", "_") +
    "-0.9.3-cp312-cp312-" +
    platformTag +
    ".whl";
  return {
    id,
    extra,
    capabilities: [{
      product,
      accelerator: { runtime, variant, soc_version: soc },
    }],
    distribution,
    version: "0.9.3",
    python_abi: "cp312",
    architecture,
    platform_tags: [platformTag],
    filename,
    url: "https://github.com/example/ucm/releases/download/v0.9.3/" + filename,
    sha256: "a".repeat(64),
    dependencies: ["wrapt==1.17.2"],
  };
}

function image({
  id,
  product,
  runtime,
  variant,
  soc,
  architectures,
  version = "0.10.2",
  channel = "stable",
  osId = "ubuntu",
  osVersion = "22.04",
}) {
  return {
    id,
    product,
    upstream: { version, channel },
    accelerator: { runtime, variant, soc_version: soc },
    os: { id: osId, version: osVersion },
    publications: {
      ghcr: publication("ghcr.io/example/" + id + ":v0.9.3", architectures),
      dockerhub: null,
    },
  };
}

function refreshPythonAssets(manifest) {
  manifest.python.extras = Object.fromEntries(
    manifest.wheels.map((item) => [item.extra, item.distribution])
  );
  manifest.github_release_assets = [
    "release-manifest.json",
    manifest.chart.filename,
    ...manifest.wheels.map((item) => item.filename),
  ].sort();
  return manifest;
}

function fixture({ pypi = true } = {}) {
  const manifest = {
    kind: "ucm-release-manifest",
    schema_version: 10,
    release: {
      tag: "v0.9.3",
      type: "stable",
      version: "0.9.3",
      url: "https://github.com/example/ucm/releases/tag/v0.9.3",
      actions_run_id: 33087700398,
    },
    python: {
      distribution: "uc-manager",
      version: "0.9.3",
      extras: {},
      pypi: pypi
        ? {
            index_url: "https://pypi.org/simple",
            project_url: "https://pypi.org/project/uc-manager/0.9.3/",
          }
        : null,
    },
    wheels: [
      wheel({
        id: "wheel-cuda-amd64",
        product: "vllm",
        extra: "cu130",
        runtime: "cuda-13.0",
        variant: "default",
        soc: "na",
        architecture: "amd64",
      }),
      wheel({
        id: "wheel-cann-arm64",
        product: "vllm-ascend",
        extra: "cann901-a2",
        runtime: "cann-9.0.1",
        variant: "a2",
        soc: "ascend910b1",
        architecture: "arm64",
      }),
    ],
    images: [
      image({
        id: "vllm-cuda",
        product: "vllm",
        runtime: "cuda-13.0",
        variant: "default",
        soc: "na",
        architectures: ["amd64", "arm64"],
      }),
      image({
        id: "vllm-ascend-a2",
        product: "vllm-ascend",
        runtime: "cann-9.0.1",
        variant: "a2",
        soc: "ascend910b1",
        architectures: ["arm64"],
      }),
    ],
    chart: {
      name: "unified-cache-chart",
      version: "0.9.3",
      filename: "unified-cache-chart-0.9.3.tgz",
      url: "https://github.com/example/ucm/releases/download/v0.9.3/chart.tgz",
      oci: "ghcr.io/example/charts/unified-cache-chart:0.9.3",
    },
    github_release_assets: [],
  };
  return refreshPythonAssets(manifest);
}

function option(row, value) {
  return row.options.find((candidate) => candidate.value === value);
}

function select(manifest, state = {}) {
  return Selector.deriveSelection(Selector.buildSelectorModel(manifest), state);
}

test("loader accepts Schemas 9 and 10 and rejects unsupported or incomplete data", () => {
  const manifest = fixture();
  assert.equal(Manifest.validateManifest(manifest), manifest);
  assert.throws(() => Manifest.validateManifest({...manifest, schema_version: 8}), /schema_version/);
  assert.throws(() => Manifest.validateManifest({...manifest, wheels: null}), /incomplete/);
});

test("one environment resolves both installation methods from the same release", () => {
  const manifest = fixture();
  const result = select(manifest, {architecture: "amd64"});
  assert.deepEqual(Selector.ROW_ORDER, ["engine", "engineVersion", "runtime", "variant", "os", "architecture"]);
  assert.deepEqual(result.state, {engine: "vllm", engineVersion: "0.10.2|stable", runtime: "cuda-13.0", variant: "default", os: "ubuntu|22.04", architecture: "amd64"});
  assert.equal(result.imageReference, manifest.images[0].publications.ghcr.pull);
  assert.equal(result.wheel, manifest.wheels[0]);
  assert.equal(result.pipCommand, 'pip install "uc-manager[cu130]==0.9.3"');
  assert.equal(result.rows.variant.visible, false);
  for (const field of ["runtime", "os", "architecture"]) assert.equal(result.rows[field].visible, true);
});

test("pip preserves official, fork and direct Wheel publication contracts", () => {
  const manifest = fixture();
  manifest.python.distribution = "supermarioyl-uc-manager";
  manifest.python.pypi.index_url = "https://test.pypi.org/simple";
  assert.equal(select(manifest, {architecture: "amd64"}).pipCommand,
    'pip install --index-url https://test.pypi.org/simple --extra-index-url https://pypi.org/simple "supermarioyl-uc-manager[cu130]==0.9.3"');
  manifest.python.pypi = null;
  assert.equal(select(manifest, {architecture: "amd64"}).pipCommand, 'pip install "' + manifest.wheels[0].url + '"');
  assert.equal(select(manifest, {engine: "vllm-ascend"}).pipCommand, 'pip install "' + manifest.wheels[1].url + '"');
});

test("Quickstart installs Toolkit together with the selected backend when published", () => {
  const manifest = fixture();
  manifest.toolkit = {
    distribution: "ucm-toolkit", version: "0.9.3",
    url: "https://github.com/example/ucm/releases/download/v0.9.3/ucm_toolkit-0.9.3-py3-none-any.whl",
  };
  assert.equal(select(manifest, {architecture: "amd64"}).pipCommand,
    'pip install "uc-manager[cu130,toolkit]==0.9.3"');
  assert.equal(select(manifest, {engine: "vllm-ascend"}).pipCommand,
    'pip install "uc-manager[cann901-a2,toolkit]==0.9.3"');
  assert.equal(select(manifest, {architecture: "amd64"}).toolkit, true);

  manifest.python.distribution = "supermarioyl-uc-manager";
  manifest.python.pypi.index_url = "https://test.pypi.org/simple";
  assert.equal(select(manifest, {architecture: "amd64"}).pipCommand,
    'pip install --index-url https://test.pypi.org/simple --extra-index-url https://pypi.org/simple "supermarioyl-uc-manager[cu130,toolkit]==0.9.3"');
  manifest.python.pypi = null;
  assert.equal(select(manifest, {architecture: "amd64"}).pipCommand,
    'pip install "' + manifest.wheels[0].url + '" "' + manifest.toolkit.url + '"');
  delete manifest.toolkit;
  assert.equal(select(manifest, {architecture: "amd64"}).toolkit, false);
  assert.equal(select(manifest, {architecture: "amd64"}).pipCommand,
    'pip install "' + manifest.wheels[0].url + '"');
});

test("a shared Wheel is selectable for every declared Runtime capability", () => {
  const manifest = fixture({pypi: false});
  const sharedWheel = manifest.wheels[0];
  const secondCapability = {
    product: "vllm",
    accelerator: {
      runtime: "cuda-13.1",
      variant: "default",
      soc_version: "na",
    },
  };
  sharedWheel.capabilities.push(secondCapability);
  manifest.images.push(image({
    id: "vllm-cuda-131",
    product: "vllm",
    runtime: "cuda-13.1",
    variant: "default",
    soc: "na",
    architectures: ["amd64"],
  }));

  const result = select(manifest, {engine: "vllm", runtime: "cuda-13.1"});
  assert.equal(result.wheel, sharedWheel);
  assert.equal(result.image, manifest.images[2]);
});

function ascendFixture() {
  const manifest = fixture();
  manifest.wheels = [];
  manifest.images = [];
  for (const runtime of ["cann-9.0.1", "cann-9.1.0"]) {
    for (const variant of ["a2", "a3"]) {
      for (const architecture of ["arm64", "amd64"]) {
        manifest.wheels.push(wheel({id: runtime + variant + architecture, product: "vllm-ascend", extra: runtime.replace(/[^0-9]/g, "") + "-" + variant, runtime, variant, soc: "ascend910", architecture}));
      }
    }
  }
  for (const [version, channel, runtime] of [["0.23.0", "stable", "cann-9.1.0"], ["0.24.0rc0", "nightly", "cann-9.0.1"], ["0.26.0rc1", "rc", "cann-9.1.0"]]) {
    for (const variant of ["a2", "a3"]) {
      for (const osId of ["linux", "openeuler"]) {
        manifest.images.push(image({id: version + variant + osId, product: "vllm-ascend", version, channel, runtime, variant, soc: "ascend910", architectures: ["amd64", "arm64"], osId, osVersion: "unreported"}));
      }
    }
  }
  return refreshPythonAssets(manifest);
}

test("Ascend keeps runtime, device, OS and CPU distinct and matches both artifacts", () => {
  const manifest = ascendFixture();
  const model = Selector.buildSelectorModel(manifest);
  for (const combination of model.combinations) {
    const result = Selector.deriveSelection(model, combination);
    assert.equal(result.image, combination.image);
    const capability = result.wheel.capabilities.find(item => item.product === result.image.product);
    assert.equal(capability.accelerator.runtime, result.state.runtime);
    assert.equal(capability.accelerator.variant, result.state.variant);
    assert.equal(result.wheel.architecture, result.state.architecture);
    assert.equal(result.image.os.id + "|" + result.image.os.version, result.state.os);
    assert.ok(result.pipCommand.includes(result.wheel.extra));
  }
  const result = select(manifest, {engine: "vllm-ascend"});
  assert.deepEqual(result.rows.engineVersion.options.map(o => o.label), ["0.23.0 / Stable", "0.24.0rc0 / Nightly", "0.26.0rc1 / RC"]);
  assert.deepEqual(result.rows.runtime.options.map(o => o.label), ["CANN 9.0.1", "CANN 9.1.0"]);
  assert.deepEqual(result.rows.variant.options.map(o => o.label), ["A2", "A3"]);
  assert.deepEqual(result.rows.os.options.map(o => o.label), ["Linux", "openEuler"]);
  assert.equal(result.rows.variant.visible, true);
});

test("changing engine version preserves valid choices and resets incompatible runtime and OS", () => {
  const manifest = ascendFixture();
  manifest.images = manifest.images.filter(i => i.upstream.version !== "0.24.0rc0" || i.os.id !== "openeuler");
  const previous = select(manifest, {engine: "vllm-ascend", variant: "a3", os: "openeuler|unreported", architecture: "arm64"});
  const next = select(manifest, {...previous.state, engineVersion: "0.24.0rc0|nightly"});
  assert.equal(next.state.runtime, "cann-9.0.1");
  assert.equal(next.state.os, "linux|unreported");
  assert.equal(next.state.variant, "a3");
  assert.equal(next.state.architecture, "arm64");
  assert.equal(next.wheel.capabilities[0].accelerator.runtime, "cann-9.0.1");
  assert.equal(option(next.rows.runtime, "cann-9.1.0").disabled, true);
  assert.equal(option(next.rows.os, "openeuler|unreported").disabled, true);
});

test("version and channel identify distinct images; multiple CANN versions remain selectable", () => {
  const manifest = ascendFixture();
  const extra = structuredClone(manifest.images[0]);
  extra.upstream.channel = "rc";
  manifest.images.push(extra);
  const anotherRuntime = structuredClone(extra);
  anotherRuntime.accelerator.runtime = "cann-9.0.1";
  manifest.images.push(anotherRuntime);
  const result = select(manifest, {engine: "vllm-ascend", engineVersion: "0.23.0|rc", runtime: "cann-9.0.1"});
  assert.equal(result.state.engineVersion, "0.23.0|rc");
  assert.equal(result.state.runtime, "cann-9.0.1");
  assert.equal(result.image, anotherRuntime);
});

test("registry preference follows architecture availability without changing image", () => {
  const manifest = fixture();
  manifest.images[0].publications.dockerhub = publication("docker.io/example/vllm:v0.9.3", ["amd64", "ppc64le"]);
  assert.equal(select(manifest, {architecture: "amd64"}).imageReference, manifest.images[0].publications.ghcr.pull);
  const otherArch = select(manifest, {architecture: "ppc64le"});
  assert.equal(otherArch.imageReference, manifest.images[0].publications.dockerhub.pull);
  assert.equal(otherArch.wheel, null);
  assert.equal(otherArch.pipCommand, null);
});

test("explicit Ubuntu image variants remain distinct when OS metadata is unreported", () => {
  const manifest = fixture();
  manifest.images[0].os = {id: "linux", version: "unreported"};
  const ubuntu = structuredClone(manifest.images[0]);
  ubuntu.id += "-ubuntu2404";
  ubuntu.publications.ghcr = publication("ghcr.io/example/vllm:ubuntu2404", ["amd64"]);
  manifest.images.push(ubuntu);
  const result = select(manifest, {os: "linux|unreported|ubuntu2404", architecture: "amd64"});
  assert.equal(result.image, ubuntu);
  assert.equal(result.imageReference, ubuntu.publications.ghcr.pull);
  assert.deepEqual(result.rows.os.options.map(o => o.label), ["Linux", "Linux / ubuntu2404"]);
  assert.equal(select(manifest).image, manifest.images[0]);
});

test("missing or ambiguous artifacts never silently choose a different environment", () => {
  const manifest = fixture();
  const absentWheel = select(manifest, {architecture: "arm64"});
  assert.ok(absentWheel.image);
  assert.equal(absentWheel.pipCommand, null);
  manifest.images = [];
  assert.equal(select(manifest).pipCommand, null);
  const duplicate = fixture();
  duplicate.images.push(structuredClone(duplicate.images[0]));
  assert.throws(() => select(duplicate), /Multiple images/);
  const duplicateWheel = fixture();
  duplicateWheel.wheels.push(structuredClone(duplicateWheel.wheels[0]));
  assert.throws(() => select(duplicateWheel), /Multiple Wheels/);
});

test("SGLang remains selectable without a manifest and has no environment or artifact choices", () => {
  for (const model of [{manifest: null, combinations: []}, Selector.buildSelectorModel(fixture())]) {
    const result = Selector.deriveSelection(model, {engine: "sglang"});
    assert.equal(result.state.engine, "sglang");
    for (const field of Selector.ROW_ORDER.slice(1)) assert.equal(result.rows[field].visible, false);
    assert.equal(result.image, null);
    assert.equal(result.pipCommand, null);
    assert.deepEqual(result.rows.engine.options.map(o => o.label), ["vLLM", "vLLM-Ascend", "SGLang"]);
  }
});

test("Helm page downloads and extracts its own release Chart", () => {
  const manifest = fixture();
  assert.equal(Selector.chartDownloadCommand(manifest), 'helm pull "' + manifest.chart.url + '" --untar\ncd unified-cache-chart');
});

test("standard images use policy repositories and preserve the full upstream tag", () => {
  const manifest = fixture();
  manifest.images[0].id = "vllm-v0.10.2-cu130-ubuntu2204";
  const repositories = {vllm: "docker.io/example/standard-vllm"};
  const result = Selector.deriveSelection(Selector.buildSelectorModel(manifest, repositories), {architecture: "amd64"});
  assert.equal(result.standardImageReference, "docker.io/example/standard-vllm:v0.10.2-cu130-ubuntu2204");
  assert.equal(result.imageReference, manifest.images[0].publications.ghcr.pull);
  assert.equal(result.standard, true);
  assert.ok(result.pipCommand.includes("uc-manager[cu130]==0.9.3"));

  const ascend = ascendFixture();
  const image = ascend.images.find(i => i.upstream.version === "0.24.0rc0" && i.accelerator.variant === "a3" && i.os.id === "openeuler");
  image.id = "vllm-ascend-nightly-releases-v0.24.0rc0-a3-openeuler";
  const selected = Selector.deriveSelection(Selector.buildSelectorModel(ascend, {"vllm-ascend": "quay.io/example/standard-ascend"}), {
    engine: "vllm-ascend", engineVersion: "0.24.0rc0|nightly", variant: "a3", os: "openeuler|unreported",
  });
  assert.equal(selected.standardImageReference, "quay.io/example/standard-ascend:nightly-releases-v0.24.0rc0-a3-openeuler");
  assert.equal(selected.standard, true);
});

test("standard image installation requires both a known upstream reference and a matching Wheel", () => {
  const manifest = fixture();
  manifest.images[0].id = "vllm-v0.10.2";
  assert.equal(select(manifest, {architecture: "amd64"}).standard, false);
  const missingWheel = Selector.deriveSelection(Selector.buildSelectorModel(manifest, {vllm: "docker.io/example/vllm"}), {architecture: "arm64"});
  assert.equal(missingWheel.standard, false);
  assert.ok(missingWheel.image);
});

test("Toolkit install commands use this release's namespace, version and index", () => {
  const manifest = {
    toolkit: {distribution: "supermarioyl-ucm-toolkit", version: "0.7.0rc10", url: "https://github.com/example/toolkit.whl"},
    python: {distribution: "supermarioyl-uc-manager", pypi: {index_url: "https://test.pypi.org/simple"}}
  };
  assert.deepEqual(Selector.toolkitInstallCommands(manifest), [
    'pip install --index-url https://test.pypi.org/simple "supermarioyl-ucm-toolkit==0.7.0rc10"',
    'pip install --index-url https://test.pypi.org/simple "supermarioyl-uc-manager[toolkit]==0.7.0rc10"'
  ]);
  assert.deepEqual(Selector.toolkitInstallCommands({python: manifest.python}), []);
  manifest.python.pypi = null;
  assert.deepEqual(Selector.toolkitInstallCommands(manifest), ['pip install "https://github.com/example/toolkit.whl"']);
});


test("releases without a Chart retain Wheel commands and omit Helm downloads", () => {
  const manifest = fixture();
  manifest.github_release_assets = manifest.github_release_assets.filter(
    (name) => name !== manifest.chart.filename
  );
  manifest.chart = null;
  assert.equal(Manifest.validateManifest(manifest), manifest);
  assert.equal(select(manifest, {architecture: "amd64"}).pipCommand,
    'pip install "uc-manager[cu130]==0.9.3"');
  assert.equal(Selector.chartDownloadCommand(manifest), null);
});
