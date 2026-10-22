"""Exercise the registration contract without importing UCM or requiring hardware."""

import ast
import copy
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

SCRIPT = Path(__file__).resolve().parents[1] / "tools/generate_kv_metrics.py"
SPEC = importlib.util.spec_from_file_location("generate_kv_metrics", SCRIPT)
generator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generator)


@pytest.fixture
def definitions():
    return {
        "counter": [{"name": "kv_count", "documentation": 'Count "quoted" \\ path'}],
        "gauge": [{"name": "kv_gauge", "documentation": "Current value"}],
        "histogram": [
            {"name": "kv_time", "documentation": "Duration", "buckets": [1e-5, 0.5]}
        ],
    }


def template(python):
    if python:
        return (
            '# user comment\n_COUNTER_METRICS = [\n    ("custom", "Keep"),\n]\n'
            "_GAUGE_METRICS = [\n]\n_HISTOGRAM_METRICS = [\n]\n"
            'ENABLED = ["custom"]\n'
        )
    return (
        '# user comment\ncounter:\n  - {name: "custom", documentation: "Keep"}\n'
        'gauge:\nhistogram:\nenabled: ["custom"]\n'
    )


def document(definitions, language, totals):
    summary = (
        "3 个 KV 指标（1 个 Counter、1 个 Gauge、1 个 Histogram）"
        if language == "zh"
        else "3 KV metrics (1 Counter, 1 Gauge, and 1 Histogram)"
    )
    lines = ["# Metrics"]
    if totals:
        lines.append("Defaults: 2 Counters, 1 Gauge, and 1 Histogram.")
    lines.extend(["### 1.6 ASU / KV metrics", summary])
    for section in generator.SECTIONS:
        lines.extend(
            [
                f"#### {section.title()}s",
                "| Metric | Unit | Description |",
                "| --- | --- | --- |",
            ]
        )
        for metric in definitions[section]:
            lines.append(f"| `ucm:{metric['name']}` | unit | Translated description |")
    lines.append("## 2. Examples")
    return "\n".join(lines) + "\n"


@pytest.mark.parametrize("section", ["counter", "gauge", "histogram"])
@pytest.mark.parametrize("python", [False, True])
def test_render_block_round_trips_types_escaping_and_small_buckets(
    definitions, section, python
):
    block = generator.render_ucm_block(definitions, section, python)
    assert block.count(f"BEGIN GENERATED KV {section}") == 1
    assert block.count(f"END GENERATED KV {section}") == 1
    metric = definitions[section][0]
    if python:
        values = ast.literal_eval("[\n" + block + "\n]")
        expected = (metric["name"], metric["documentation"])
        if section == "gauge":
            expected += ({},)
        elif section == "histogram":
            expected += ([0.00001, 0.5],)
        assert values == [expected]
    else:
        values = yaml.safe_load(f"{section}:\n{block}\n")[section]
        assert values == [metric]
        if section == "histogram":
            assert all(
                isinstance(value, (int, float)) for value in values[0]["buckets"]
            )


@pytest.mark.parametrize("python", [False, True])
def test_render_config_inserts_replaces_and_preserves_custom_content(
    definitions, python
):
    original = template(python)
    rendered = generator.render_ucm_config(original, definitions, python)
    assert generator.render_ucm_config(rendered, definitions, python) == rendered
    changed = copy.deepcopy(definitions)
    changed["counter"][0]["documentation"] = "Replacement"
    replaced = generator.render_ucm_config(rendered, changed, python)
    assert "Replacement" in replaced
    assert "quoted" not in replaced
    # Removing only the generated blocks must recover all user content verbatim.
    remaining = replaced
    for section in generator.SECTIONS:
        assert replaced.count(f"BEGIN GENERATED KV {section}") == 1
        remaining = remaining.replace(
            generator.render_ucm_block(changed, section, python) + "\n", ""
        )
    assert remaining == original


@pytest.mark.parametrize("section", ["counter", "gauge", "histogram"])
@pytest.mark.parametrize("python", [False, True])
def test_render_config_rejects_missing_section(definitions, section, python):
    anchor = f"_{section.upper()}_METRICS = [\n" if python else f"{section}:\n"
    content = template(python).replace(anchor, "")
    with pytest.raises(ValueError, match=f"missing UCM section: {section}"):
        generator.render_ucm_config(content, definitions, python)


@pytest.mark.parametrize("python", [False, True])
def test_empty_definitions_need_no_anchors(python):
    definitions = {section: [] for section in generator.SECTIONS}
    assert (
        generator.render_ucm_config("# custom\n", definitions, python) == "# custom\n"
    )


def test_default_header_renders_all_types_and_buckets(definitions):
    header = generator.render_default_header(definitions)
    assert (
        '{"kv_count", MetricType::COUNTER, "Count \\"quoted\\" \\\\ path", {}}'
        in header
    )
    assert '{"kv_gauge", MetricType::GAUGE, "Current value", {}}' in header
    assert '{"kv_time", MetricType::HISTOGRAM, "Duration", {1e-05, 0.5}}' in header


FAILURES = [
    ("missing", "missing KV metric: kv_count"),
    ("duplicate", "duplicate KV metric: kv_count"),
    ("cross_duplicate", "duplicate KV metric: kv_count"),
    ("type", "type mismatch for kv_count: expected counter, got gauge"),
    ("documentation", "documentation mismatch for kv_count"),
    ("buckets", "bucket mismatch for kv_time"),
]


def incompatible_config(definitions, failure):
    config = copy.deepcopy(definitions)
    if failure == "missing":
        config["counter"] = []
    elif failure == "duplicate":
        config["counter"].append(copy.deepcopy(config["counter"][0]))
    elif failure == "cross_duplicate":
        config["gauge"].append(config["counter"][0])
    elif failure == "type":
        config["gauge"].append(config["counter"].pop())
    elif failure == "documentation":
        config["counter"][0]["documentation"] = "Wrong"
    elif failure == "buckets":
        config["histogram"][0]["buckets"] = [0.1]
    return config


@pytest.mark.parametrize("failure, expected", FAILURES)
def test_check_definitions_reports_each_incompatibility(definitions, failure, expected):
    assert generator.check_ucm_definitions(
        definitions, incompatible_config(definitions, failure)
    ) == [expected]


def test_check_allows_custom_metrics_and_does_not_mutate_config(definitions):
    config = copy.deepcopy(definitions)
    config["counter"].append({"name": "custom", "documentation": "User metric"})
    config["enabled"] = ["custom"]
    before = copy.deepcopy(config)
    assert generator.check_ucm_definitions(definitions, config) == []
    assert config == before


def test_check_reports_all_missing_metrics_for_empty_sections(definitions):
    errors = generator.check_ucm_definitions(definitions, {"counter": None})
    assert errors == [
        "missing KV metric: kv_count",
        "missing KV metric: kv_gauge",
        "missing KV metric: kv_time",
    ]


@pytest.fixture
def sandbox(tmp_path, definitions):
    script = tmp_path / "kv_semantics/metrics/tools/generate_kv_metrics.py"
    script.parent.mkdir(parents=True)
    shutil.copyfile(SCRIPT, script)
    source = tmp_path / "kv_semantics/metrics/config/kv_metrics.yaml"
    source.parent.mkdir(parents=True)
    source.write_text(
        'counter:\n  - {name: "kv_count", documentation: "Count"}\n'
        'gauge:\n  - {name: "kv_gauge", documentation: "Current value"}\n'
        'histogram:\n  - {name: "kv_time", documentation: "Duration", '
        "buckets: [0.00001, 0.5]}\n",
        encoding="utf-8",
    )
    header = (
        tmp_path
        / "kv_semantics/metrics/include/kv_metrics/default_metric_descriptors.h"
    )
    header.parent.mkdir(parents=True)
    ucm_yaml = tmp_path / "examples/metrics/metrics_configs.yaml"
    defaults = tmp_path / "ucm/default_metrics_config.py"
    for path, python in ((ucm_yaml, False), (defaults, True)):
        path.parent.mkdir(parents=True)
        path.write_text(template(python), encoding="utf-8")
    for original, language, totals in generator.METRIC_DOCS:
        path = tmp_path / original.relative_to(generator.ROOT)
        path.parent.mkdir(parents=True)
        path.write_text(document(definitions, language, totals), encoding="utf-8")
    return script, header, ucm_yaml, defaults


def run_cli(sandbox, *args):
    return subprocess.run(
        [sys.executable, str(sandbox[0]), *map(str, args)],
        capture_output=True,
        text=True,
        timeout=20,
    )


def snapshot(sandbox):
    root = sandbox[0].parents[3]
    return {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_cli_default_generation_does_not_sync_ucm(sandbox):
    before = [path.read_bytes() for path in sandbox[2:]]
    result = run_cli(sandbox)
    assert result.returncode == 0, result.stderr
    assert '"kv_count", MetricType::COUNTER' in sandbox[1].read_text()
    assert [path.read_bytes() for path in sandbox[2:]] == before


def test_cli_sync_is_idempotent_and_check_is_read_only(sandbox):
    before = snapshot(sandbox)
    result = run_cli(sandbox, "--check")
    assert result.returncode == 1
    assert result.stdout.count("out of date:") == 3
    assert snapshot(sandbox) == before
    result = run_cli(sandbox, "--sync-ucm")
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("generated ") == 3
    assert "kv_count" in sandbox[2].read_text()
    ast.parse(sandbox[3].read_text())
    synced = snapshot(sandbox)
    for flag in ("--sync-ucm", "--check"):
        result = run_cli(sandbox, flag)
        assert result.returncode == 0, result.stderr
        assert result.stdout == ""
        assert snapshot(sandbox) == synced


@pytest.mark.parametrize("index", [1, 2, 3])
def test_cli_check_detects_each_stale_output_without_writing(sandbox, index):
    assert run_cli(sandbox, "--sync-ucm").returncode == 0
    path = sandbox[index]
    path.write_text(path.read_text().replace('"Count"', '"Stale"'), encoding="utf-8")
    before = snapshot(sandbox)
    result = run_cli(sandbox, "--check")
    assert result.returncode == 1
    assert f"out of date: {path.relative_to(sandbox[0].parents[3])}" in result.stdout
    assert snapshot(sandbox) == before


@pytest.mark.parametrize("failure, expected", [(None, ""), *FAILURES])
def test_cli_ucm_config_exit_status_diagnostics_and_no_writes(
    sandbox, definitions, failure, expected
):
    definitions["counter"][0]["documentation"] = "Count"
    config = incompatible_config(definitions, failure)
    config["enabled"] = ["custom"]
    path = sandbox[0].parents[3] / "deployment.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    before = snapshot(sandbox)
    result = run_cli(sandbox, "--ucm-config", path)
    assert result.returncode == int(failure is not None), result.stderr
    assert result.stdout.strip() == expected
    assert snapshot(sandbox) == before


@pytest.mark.parametrize("content", [None, "counter: [", "- not-a-mapping", ""])
def test_cli_ucm_config_rejects_unreadable_invalid_or_empty_config(sandbox, content):
    path = sandbox[0].parents[3] / "deployment.yaml"
    if content is not None:
        path.write_text(content, encoding="utf-8")
    before = snapshot(sandbox)
    result = run_cli(sandbox, "--ucm-config", path)
    assert result.returncode != 0
    assert result.stdout or result.stderr
    assert snapshot(sandbox) == before


@pytest.mark.parametrize(
    "flags", [("--check",), ("--sync-ucm",), ("--check", "--sync-ucm")]
)
def test_cli_ucm_config_rejects_conflicting_flags_before_reading_or_writing(
    sandbox, flags
):
    # Neither the deployment file nor generated header exists: reject usage first.
    path = sandbox[0].parents[3] / "missing.yaml"
    before = snapshot(sandbox)
    result = run_cli(sandbox, "--ucm-config", path, *flags)
    assert result.returncode == 2
    assert "usage:" in result.stderr
    assert "--ucm-config cannot be combined with --check or --sync-ucm" in result.stderr
    assert "Traceback" not in result.stderr
    assert snapshot(sandbox) == before


@pytest.mark.parametrize("literal", ["1e-5", '"0.00001"', "0.00001"])
def test_cli_ucm_config_explains_string_buckets(sandbox, literal):
    assert run_cli(sandbox, "--sync-ucm").returncode == 0
    path = sandbox[0].parents[3] / "deployment.yaml"
    path.write_text(
        sandbox[2].read_text().replace("[0.00001, 0.5]", f"[{literal}, 0.5]"),
        encoding="utf-8",
    )
    before = snapshot(sandbox)
    result = run_cli(sandbox, "--ucm-config", path)
    if literal == "0.00001":
        assert result.returncode == 0, result.stderr
        assert result.stdout == ""
    else:
        assert result.returncode == 1
        assert "bucket mismatch for kv_time: buckets[0]=" in result.stdout
        assert "(str); expected numeric values" in result.stdout
        assert "YAML 1.1" in result.stdout
        assert "use unquoted decimal literals such as 0.00001" in result.stdout
    assert snapshot(sandbox) == before


def test_cli_ucm_config_requires_yaml_but_check_does_not(sandbox):
    assert run_cli(sandbox, "--sync-ucm").returncode == 0
    # Simulate an absent optional dependency independently of the host environment.
    bootstrap = """
import builtins, runpy, sys
original_import = builtins.__import__
def without_yaml(name, *args, **kwargs):
    if name == 'yaml':
        raise ModuleNotFoundError('No module named yaml')
    return original_import(name, *args, **kwargs)
builtins.__import__ = without_yaml
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    before = snapshot(sandbox)
    for args, expected in ((["--check"], 0), (["--ucm-config", str(sandbox[2])], 1)):
        result = subprocess.run(
            [sys.executable, "-c", bootstrap, str(sandbox[0]), *args],
            capture_output=True,
            text=True,
            timeout=20,
        )
        assert result.returncode == expected
        if expected:
            assert "No module named yaml" in result.stderr
    assert snapshot(sandbox) == before


@pytest.mark.parametrize("language", ["en", "zh"])
@pytest.mark.parametrize(
    "failure, expected",
    [
        ("missing", "missing documented KV metric: kv_count"),
        ("duplicate", "duplicate documented KV metric: kv_count"),
        ("type", "documented type mismatch for kv_count"),
        ("obsolete", "unexpected documented KV metric: kv_obsolete"),
        ("kv_counts", "KV metric count summary out of date"),
        ("default_counts", "default metric counts out of date"),
        ("section", "expected exactly one ASU / KV documentation section"),
    ],
)
def test_document_contract_detects_drift(definitions, language, failure, expected):
    content = document(definitions, language, True)
    row = "| `ucm:kv_count` | unit | Translated description |\n"
    if failure == "missing":
        content = content.replace(row, "")
    elif failure == "duplicate":
        content = content.replace(row, row + row)
    elif failure == "type":
        content = content.replace("#### Counters", "#### Gauges")
    elif failure == "obsolete":
        content = content.replace(row, row + row.replace("kv_count", "kv_obsolete"))
    elif failure == "kv_counts":
        content = content.replace("3 KV metrics", "4 KV metrics").replace(
            "3 个 KV 指标", "4 个 KV 指标"
        )
    elif failure == "default_counts":
        content = content.replace("2 Counters", "99 Counters")
    elif failure == "section":
        content = content.replace("ASU / KV", "Unrelated")
    errors = generator.check_metric_document(
        content, definitions, {"counter": 2, "gauge": 1, "histogram": 1}, language, True
    )
    assert any(expected in error for error in errors), errors


@pytest.mark.parametrize("language", ["en", "zh"])
def test_document_contract_keeps_translated_descriptions(definitions, language):
    content = document(definitions, language, True).replace(
        "Translated description", "人工维护的说明 / manually maintained prose"
    )
    assert (
        generator.check_metric_document(
            content,
            definitions,
            {"counter": 2, "gauge": 1, "histogram": 1},
            language,
            True,
        )
        == []
    )


@pytest.mark.parametrize("index", [0, 1, 2])
@pytest.mark.parametrize("remove_file", [False, True])
def test_cli_check_covers_all_documents_without_writes(sandbox, index, remove_file):
    assert run_cli(sandbox, "--sync-ucm").returncode == 0
    root = sandbox[0].parents[3]
    path = root / generator.METRIC_DOCS[index][0].relative_to(generator.ROOT)
    if remove_file:
        path.unlink()
    else:
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "`ucm:kv_count`", "`ucm:kv_obsolete`"
            ),
            encoding="utf-8",
        )
    before = snapshot(sandbox)
    result = run_cli(sandbox, "--check")
    assert result.returncode == 1, result.stderr
    assert str(path.relative_to(root)) in result.stdout
    expected = (
        "missing metric documentation"
        if remove_file
        else "missing documented KV metric: kv_count"
    )
    assert expected in result.stdout
    assert snapshot(sandbox) == before


def test_sync_requires_manual_document_update_after_definition_changes(sandbox):
    assert run_cli(sandbox, "--sync-ucm").returncode == 0
    root = sandbox[0].parents[3]
    source = root / "kv_semantics/metrics/config/kv_metrics.yaml"
    source.write_text(
        source.read_text(encoding="utf-8").replace("kv_count", "kv_replacement"),
        encoding="utf-8",
    )
    assert run_cli(sandbox, "--sync-ucm").returncode == 0
    before = snapshot(sandbox)
    result = run_cli(sandbox, "--check")
    assert result.returncode == 1, result.stderr
    assert "missing documented KV metric: kv_replacement" in result.stdout
    assert "unexpected documented KV metric: kv_count" in result.stdout
    assert snapshot(sandbox) == before


def test_default_totals_include_non_kv_metrics_without_importing_ucm(definitions):
    content = generator.render_ucm_config(template(True), definitions, True)
    counts = generator.default_metric_counts(
        content + "\nraise RuntimeError('not executed')\n"
    )
    assert counts == {"counter": 2, "gauge": 1, "histogram": 1}


def test_default_totals_expand_static_comprehensions_without_executing_them():
    content = """
METHODS = ["load", "save", "lookup"]
_COUNTER_METRICS = []
_GAUGE_METRICS = []
_HISTOGRAM_METRICS = [
    ("base", "Base", []),
    *[must_not_execute(method) for method in METHODS],
    *[("extra", "Extra", [])],
]
"""
    assert generator.default_metric_counts(content) == {
        "counter": 0,
        "gauge": 0,
        "histogram": 5,
    }


@pytest.mark.parametrize(
    "expression",
    [
        "[entry for entry in METHODS if enabled(entry)]",
        "load_metrics()",
    ],
)
def test_default_totals_reject_dynamic_sequences(expression):
    content = (
        'METHODS = ["load", "save"]\n'
        "_COUNTER_METRICS = []\n_GAUGE_METRICS = []\n"
        f"_HISTOGRAM_METRICS = [*{expression}]\n"
    )
    with pytest.raises(ValueError, match="cannot statically count _HISTOGRAM_METRICS"):
        generator.default_metric_counts(content)


def test_repository_documentation_matches_generated_catalog():
    definitions = generator.load_definitions()
    defaults = generator.render_ucm_config(
        generator.UCM_DEFAULTS.read_text(encoding="utf-8"), definitions, True
    )
    assert generator.check_metric_documents(definitions, defaults) == []


@pytest.mark.parametrize("python", [False, True])
def test_removing_last_metric_clears_existing_generated_block(definitions, python):
    content = generator.render_ucm_config(template(python), definitions, python)
    definitions["gauge"] = []
    updated = generator.render_ucm_config(content, definitions, python)
    assert "kv_gauge" not in updated
    assert generator.render_ucm_config(updated, definitions, python) == updated
