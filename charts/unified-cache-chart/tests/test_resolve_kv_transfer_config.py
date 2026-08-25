#!/usr/bin/env python3
"""Unit tests for the Pod-local KV-transfer identity resolver."""

from __future__ import annotations

import importlib.util
import ctypes
import io
import json
import os
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
RESOLVER_PATH = ROOT / "files" / "resolve-kv-transfer-config.py"
SPEC = importlib.util.spec_from_file_location("kv_transfer_resolver", RESOLVER_PATH)
assert SPEC is not None and SPEC.loader is not None
resolver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(resolver)


def mooncake_template(
    *,
    connector: str = "MooncakeConnectorV1",
    role: str = "producer",
) -> dict:
    result = {
        "kv_connector": connector,
        "kv_role": f"kv_{role}",
        "engine_id": resolver.ENGINE_ID_SENTINEL,
        "kv_port": resolver.KV_PORT_SENTINEL,
        "kv_connector_extra_config": {
            "prefill": {"dp_size": 1, "tp_size": 1},
            "decode": {"dp_size": 1, "tp_size": 1},
        },
    }
    if connector == "MooncakeConnectorV1":
        result["kv_rank"] = 0 if role == "producer" else 1
    return result


def nixl_template(*, role: str = "producer") -> dict:
    return {
        "kv_connector": "NixlConnector",
        "kv_role": f"kv_{role}",
        "engine_id": resolver.ENGINE_ID_SENTINEL,
        "kv_load_failure_policy": "fail",
    }


def multi_template(
    *, connector: str = "MooncakeConnectorV1"
) -> dict:
    mooncake = {
        "kv_connector": connector,
        "kv_role": "kv_producer",
        "kv_port": resolver.KV_PORT_SENTINEL,
        "kv_connector_extra_config": {
            "prefill": {"dp_size": 1, "tp_size": 1},
            "decode": {"dp_size": 1, "tp_size": 1},
        },
    }
    if connector == "MooncakeConnectorV1":
        mooncake["kv_rank"] = 0
    return {
        "kv_connector": "MultiConnector",
        "kv_role": "kv_producer",
        "engine_id": resolver.ENGINE_ID_SENTINEL,
        "kv_connector_extra_config": {
            "connectors": [
                mooncake,
                {
                    "kv_connector": "UCMConnector",
                    "kv_role": "kv_both",
                    "kv_connector_module_path": (
                        "ucm.integration.vllm.ucm_connector"
                    ),
                    "kv_connector_extra_config": {
                        "UCM_CONFIG_FILE": (
                            "/vllm-workspace/UnifiedCache/config/"
                            "ucm_config.runtime.yaml"
                        )
                    },
                },
            ]
        },
    }


def pd_meta(
    *,
    connector: str = "MooncakeConnectorV1",
    role_kind: str = "producer",
    role_name: str | None = None,
    prefill_replicas: int = 1,
    decode_replicas: int = 1,
    engine_id_base: int = 0,
    kv_port_base: int | None = 36000,
    instance_stride: int | None = 100,
    port_span: int | None = 1,
) -> dict:
    identity = {"engineIdBase": engine_id_base}
    if kv_port_base is not None:
        identity["kvPortBase"] = kv_port_base
    if instance_stride is not None:
        identity["instanceStride"] = instance_stride
    result = {
        "schemaVersion": 1,
        "dynamicIdentity": True,
        "connector": connector,
        "roleKind": role_kind,
        "roleName": role_name
        or ("prefill" if role_kind == "producer" else "decode"),
        "groupNamePrefix": "demo-",
        "prefillReplicas": prefill_replicas,
        "decodeReplicas": decode_replicas,
        "identity": identity,
    }
    if port_span is not None:
        result["portSpan"] = port_span
    return result


def resolve(template: dict, meta: dict, group: str, role_id: str) -> dict:
    return resolver.resolve_kv_transfer_config(
        template,
        meta,
        group_name=group,
        role_id=role_id,
        probe_registry=False,
    )


class IdentityFormulaTests(unittest.TestCase):
    def test_1p1d(self) -> None:
        producer = resolve(
            mooncake_template(role="producer"),
            pd_meta(role_kind="producer"),
            "demo-0",
            "prefill-0",
        )
        consumer = resolve(
            mooncake_template(role="consumer"),
            pd_meta(role_kind="consumer"),
            "demo-0",
            "decode-0",
        )
        self.assertEqual((producer["engine_id"], producer["kv_port"]), ("0", 36000))
        self.assertEqual((consumer["engine_id"], consumer["kv_port"]), ("1", 36100))
        self.assertEqual(producer["kv_rank"], 0)
        self.assertEqual(consumer["kv_rank"], 1)

    def test_2p2d_exact_layout(self) -> None:
        cases = [
            ("producer", "prefill", 0, "0", 36000),
            ("producer", "prefill", 1, "1", 36100),
            ("consumer", "decode", 0, "2", 36200),
            ("consumer", "decode", 1, "3", 36300),
        ]
        for role_kind, role_name, ordinal, engine_id, kv_port in cases:
            with self.subTest(role_kind=role_kind, ordinal=ordinal):
                output = resolve(
                    mooncake_template(role=role_kind),
                    pd_meta(
                        role_kind=role_kind,
                        role_name=role_name,
                        prefill_replicas=2,
                        decode_replicas=2,
                    ),
                    "demo-0",
                    f"{role_name}-{ordinal}",
                )
                self.assertEqual(output["engine_id"], engine_id)
                self.assertEqual(output["kv_port"], kv_port)

    def test_second_serving_group_uses_next_identity_block(self) -> None:
        producer = resolve(
            mooncake_template(role="producer"),
            pd_meta(
                role_kind="producer", prefill_replicas=2, decode_replicas=2
            ),
            "demo-1",
            "prefill-0",
        )
        consumer = resolve(
            mooncake_template(role="consumer"),
            pd_meta(
                role_kind="consumer", prefill_replicas=2, decode_replicas=2
            ),
            "demo-1",
            "decode-1",
        )
        self.assertEqual((producer["engine_id"], producer["kv_port"]), ("4", 36400))
        self.assertEqual((consumer["engine_id"], consumer["kv_port"]), ("7", 36700))

    def test_sparse_serving_group_ordinal_remains_part_of_identity(self) -> None:
        output = resolve(
            mooncake_template(role="producer"),
            pd_meta(role_kind="producer"),
            "demo-10",
            "prefill-0",
        )
        # Kthena can retain sparse/increasing group ordinals across scale cycles.
        self.assertEqual((output["engine_id"], output["kv_port"]), ("20", 38000))

    def test_worker_index_is_ignored(self) -> None:
        template = mooncake_template(role="producer")
        meta = pd_meta(role_kind="producer", prefill_replicas=2, decode_replicas=2)
        with patch.dict(os.environ, {"WORKER_INDEX": "0"}):
            entry = resolve(template, meta, "demo-0", "prefill-1")
        with patch.dict(os.environ, {"WORKER_INDEX": "37"}):
            worker = resolve(template, meta, "demo-0", "prefill-1")
        self.assertEqual(entry, worker)

    def test_engine_id_base_is_applied(self) -> None:
        output = resolve(
            nixl_template(role="consumer"),
            pd_meta(
                connector="NixlConnector",
                role_kind="consumer",
                engine_id_base=20,
                kv_port_base=None,
                instance_stride=None,
                port_span=None,
            ),
            "demo-2",
            "decode-0",
        )
        # 1P1D: group 2 starts at index 4, decode offset is 1.
        self.assertEqual(output["engine_id"], "25")
        self.assertNotIn("kv_port", output)


class ValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.template = mooncake_template(role="producer")
        self.meta = pd_meta(role_kind="producer", prefill_replicas=2)

    def assert_resolution_error(
        self,
        pattern: str,
        *,
        template: dict | None = None,
        meta: dict | None = None,
        group: str | None = "demo-0",
        role_id: str | None = "prefill-0",
    ) -> None:
        with self.assertRaisesRegex(resolver.ResolutionError, pattern):
            resolver.resolve_kv_transfer_config(
                template if template is not None else self.template,
                meta if meta is not None else self.meta,
                group_name=group,
                role_id=role_id,
                probe_registry=False,
            )

    def test_missing_and_malformed_labels(self) -> None:
        cases = [
            (None, "prefill-0", "missing Pod label"),
            ("other-0", "prefill-0", "malformed Pod label"),
            ("demo-01", "prefill-0", "malformed Pod label"),
            ("demo-0", None, "missing Pod label"),
            ("demo-0", "prefill-x", "malformed Pod label"),
        ]
        for group, role_id, pattern in cases:
            with self.subTest(group=group, role_id=role_id):
                self.assert_resolution_error(pattern, group=group, role_id=role_id)

    def test_role_ordinal_must_be_in_replica_range(self) -> None:
        self.assert_resolution_error("outside producer replica range", role_id="prefill-2")

    def test_stride_has_absolute_and_parallel_span_minimum(self) -> None:
        meta = pd_meta(instance_stride=99)
        self.assert_resolution_error("must be >= 100", meta=meta)

        meta = pd_meta(instance_stride=100, port_span=101)
        self.assert_resolution_error("must be >= 101", meta=meta)

    def test_resolved_port_range_must_fit_65535(self) -> None:
        meta = pd_meta(kv_port_base=65500, instance_stride=100, port_span=40)
        self.assert_resolution_error("exceeds 65535", meta=meta)

        meta = pd_meta(
            kv_port_base=65400,
            instance_stride=100,
            port_span=40,
            prefill_replicas=2,
        )
        self.assert_resolution_error(
            "exceeds 65535", meta=meta, group="demo-0", role_id="prefill-1"
        )

    def test_port_fields_are_all_or_nothing(self) -> None:
        meta = pd_meta(instance_stride=None)
        self.assert_resolution_error("must be set together", meta=meta)

    def test_meta_connector_must_match_selected_transport(self) -> None:
        meta = pd_meta(connector="MooncakeHybridConnector")
        self.assert_resolution_error("does not match", meta=meta)

    def test_exact_sentinel_counts_are_required(self) -> None:
        missing = deepcopy(self.template)
        missing["engine_id"] = "fixed"
        self.assert_resolution_error("sentinel count mismatch", template=missing)

        duplicate = deepcopy(self.template)
        duplicate["kv_connector_extra_config"]["copy"] = resolver.KV_PORT_SENTINEL
        self.assert_resolution_error("sentinel count mismatch", template=duplicate)

    def test_embedded_and_unknown_sentinels_are_rejected(self) -> None:
        embedded = deepcopy(self.template)
        embedded["note"] = f"prefix-{resolver.ENGINE_ID_SENTINEL}"
        self.assert_resolution_error("unresolved KV-transfer sentinel", template=embedded)

        unknown = deepcopy(self.template)
        unknown["note"] = "__UC_UNKNOWN_FIELD__"
        self.assert_resolution_error("unresolved KV-transfer sentinel", template=unknown)

    def test_nixl_rejects_port_identity(self) -> None:
        meta = pd_meta(connector="NixlConnector", role_kind="producer")
        self.assert_resolution_error(
            "must not contain port fields", template=nixl_template(), meta=meta
        )

    def test_template_role_must_match_metadata_role_kind(self) -> None:
        self.assert_resolution_error(
            "kv_role must be 'kv_producer'",
            template=mooncake_template(role="consumer"),
        )

    def test_v1_rank_is_fixed_by_transfer_role(self) -> None:
        wrong_rank = mooncake_template(role="producer")
        wrong_rank["kv_rank"] = 1
        self.assert_resolution_error("kv_rank must be 0", template=wrong_rank)

        missing_rank = mooncake_template(role="producer")
        missing_rank.pop("kv_rank")
        self.assert_resolution_error("kv_rank must be 0", template=missing_rank)

    def test_hybrid_rejects_v1_only_fields(self) -> None:
        for field, value in {
            "kv_rank": 0,
            "kv_buffer_device": "npu",
            "kv_parallel_size": 8,
        }.items():
            with self.subTest(field=field):
                template = mooncake_template(
                    connector="MooncakeHybridConnector", role="producer"
                )
                template[field] = value
                self.assert_resolution_error(
                    "V1-only.*" + field,
                    template=template,
                    meta=pd_meta(connector="MooncakeHybridConnector"),
                )

    def test_nixl_rejects_residual_port_span(self) -> None:
        self.assert_resolution_error(
            "portSpan must be omitted",
            template=nixl_template(),
            meta=pd_meta(
                connector="NixlConnector",
                kv_port_base=None,
                instance_stride=None,
                port_span=1,
            ),
        )

    def test_static_ucm_requires_no_labels(self) -> None:
        template = {
            "kv_connector": "UCMConnector",
            "kv_role": "kv_both",
            "kv_connector_module_path": "ucm.integration.vllm.ucm_connector",
        }
        output = resolver.resolve_kv_transfer_config(
            template,
            {
                "schemaVersion": 1,
                "dynamicIdentity": False,
                "connector": "UCMConnector",
            },
            group_name=None,
            role_id=None,
            probe_registry=False,
        )
        self.assertEqual(output, template)


class MultiConnectorTests(unittest.TestCase):
    def test_identity_is_root_engine_and_mooncake_child_port(self) -> None:
        output = resolve(
            multi_template(connector="MooncakeHybridConnector"),
            pd_meta(
                connector="MooncakeHybridConnector",
                role_kind="producer",
            ),
            "demo-0",
            "prefill-0",
        )
        self.assertEqual(output["engine_id"], "0")
        self.assertNotIn("kv_port", output)
        children = output["kv_connector_extra_config"]["connectors"]
        mooncake, ucm = children
        self.assertEqual(mooncake["kv_port"], 36000)
        self.assertNotIn("engine_id", mooncake)
        self.assertNotIn("engine_id", ucm)
        self.assertNotIn("kv_port", ucm)
        self.assertNotIn("kv_rank", mooncake)

    def test_child_engine_identity_is_rejected_even_if_not_a_sentinel(self) -> None:
        template = multi_template()
        template["kv_connector_extra_config"]["connectors"][0]["engine_id"] = "fixed"
        with self.assertRaisesRegex(
            resolver.ResolutionError, "exactly one engine_id at the root"
        ):
            resolve(
                template,
                pd_meta(connector="MooncakeConnectorV1", role_kind="producer"),
                "demo-0",
                "prefill-0",
            )

    def test_multi_and_children_roles_are_fixed(self) -> None:
        cases = []
        wrong_root = multi_template()
        wrong_root["kv_role"] = "kv_consumer"
        cases.append((wrong_root, "producer.*root.kv_role|root.kv_role.*producer"))

        wrong_transport = multi_template()
        wrong_transport["kv_connector_extra_config"]["connectors"][0][
            "kv_role"
        ] = "kv_consumer"
        cases.append((wrong_transport, "kv_role must be 'kv_producer'"))

        wrong_ucm = multi_template()
        wrong_ucm["kv_connector_extra_config"]["connectors"][1][
            "kv_role"
        ] = "kv_producer"
        cases.append((wrong_ucm, "kv_role must be 'kv_both'"))

        for template, pattern in cases:
            with self.subTest(pattern=pattern):
                with self.assertRaisesRegex(resolver.ResolutionError, pattern):
                    resolve(
                        template,
                        pd_meta(connector="MooncakeConnectorV1"),
                        "demo-0",
                        "prefill-0",
                    )

    def test_multi_is_rejected_for_consumer_metadata(self) -> None:
        with self.assertRaisesRegex(resolver.ResolutionError, "only for a producer"):
            resolve(
                multi_template(),
                pd_meta(
                    connector="MooncakeConnectorV1",
                    role_kind="consumer",
                    role_name="decode",
                ),
                "demo-0",
                "decode-0",
            )


class RegistryProbeTests(unittest.TestCase):
    @staticmethod
    def fake_vllm_modules(
        failing_name: str | None = None,
        *,
        native_stdout: bool = False,
        buffered_c_stdout: bool = False,
    ):
        observed: list[str] = []
        plugins = types.ModuleType("vllm.plugins")

        def load_general_plugins() -> None:
            print("plugin diagnostic")
            if native_stdout:
                os.write(1, b"native plugin diagnostic\n")
            if buffered_c_stdout:
                ctypes.CDLL(None).printf(b"buffered C plugin diagnostic")

        plugins.load_general_plugins = load_general_plugins

        factory_module = types.ModuleType(
            "vllm.distributed.kv_transfer.kv_connector.factory"
        )

        class FakeFactory:
            @classmethod
            def get_connector_class(cls, config):
                observed.append(config.kv_connector)
                if config.kv_connector == failing_name:
                    raise ValueError(f"{failing_name} is not registered")
                return object

        factory_module.KVConnectorFactory = FakeFactory
        modules = {
            "vllm": types.ModuleType("vllm"),
            "vllm.plugins": plugins,
            "vllm.distributed": types.ModuleType("vllm.distributed"),
            "vllm.distributed.kv_transfer": types.ModuleType(
                "vllm.distributed.kv_transfer"
            ),
            "vllm.distributed.kv_transfer.kv_connector": types.ModuleType(
                "vllm.distributed.kv_transfer.kv_connector"
            ),
            "vllm.distributed.kv_transfer.kv_connector.factory": factory_module,
        }
        return modules, observed

    def test_root_and_nested_connectors_are_checked_through_factory(self) -> None:
        modules, observed = self.fake_vllm_modules()
        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch.dict(sys.modules, modules), redirect_stdout(stdout), redirect_stderr(stderr):
            resolver._probe_connector_registry(multi_template())
        self.assertEqual(
            observed, ["MultiConnector", "MooncakeConnectorV1", "UCMConnector"]
        )
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("plugin diagnostic", stderr.getvalue())

    def test_missing_hybrid_is_a_hard_failure_without_fallback(self) -> None:
        modules, observed = self.fake_vllm_modules("MooncakeHybridConnector")
        with patch.dict(sys.modules, modules), redirect_stderr(io.StringIO()):
            with self.assertRaisesRegex(
                resolver.ResolutionError, "MooncakeHybridConnector.*not registered"
            ):
                resolver._probe_connector_registry(
                    mooncake_template(connector="MooncakeHybridConnector")
                )
        self.assertEqual(observed, ["MooncakeHybridConnector"])

    def test_native_plugin_stdout_is_redirected_to_stderr_fd(self) -> None:
        modules, observed = self.fake_vllm_modules(
            native_stdout=True, buffered_c_stdout=True
        )
        saved_stdout = os.dup(1)
        saved_stderr = os.dup(2)
        try:
            with tempfile.TemporaryFile(mode="w+b") as stdout_file, tempfile.TemporaryFile(
                mode="w+b"
            ) as stderr_file:
                sys.stdout.flush()
                sys.stderr.flush()
                os.dup2(stdout_file.fileno(), 1)
                os.dup2(stderr_file.fileno(), 2)
                try:
                    with patch.dict(sys.modules, modules):
                        resolver._probe_connector_registry(mooncake_template())
                    # Force out anything the resolver failed to flush before it
                    # restored fd 1; it must still not reach captured stdout.
                    ctypes.CDLL(None).fflush(None)
                    sys.stdout.flush()
                    sys.stderr.flush()
                finally:
                    os.dup2(saved_stdout, 1)
                    os.dup2(saved_stderr, 2)

                stdout_file.seek(0)
                stderr_file.seek(0)
                self.assertEqual(stdout_file.read(), b"")
                stderr_bytes = stderr_file.read()
                self.assertIn(b"native plugin diagnostic", stderr_bytes)
                self.assertIn(b"buffered C plugin diagnostic", stderr_bytes)
        finally:
            os.close(saved_stdout)
            os.close(saved_stderr)
        self.assertEqual(observed, ["MooncakeConnectorV1"])


class CliTests(unittest.TestCase):
    def test_cli_emits_one_compact_json_line_and_no_stderr(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            template_path = Path(directory) / "template.json"
            meta_path = Path(directory) / "meta.json"
            template_path.write_text(json.dumps(mooncake_template()), encoding="utf-8")
            meta_path.write_text(json.dumps(pd_meta()), encoding="utf-8")
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                result = resolver.main(
                    [
                        "--template",
                        str(template_path),
                        "--meta",
                        str(meta_path),
                        "--group-name",
                        "demo-0",
                        "--role-id",
                        "prefill-0",
                        "--skip-registry-probe",
                    ]
                )
        self.assertEqual(result, 0)
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(stdout.getvalue().count("\n"), 1)
        self.assertNotIn(" ", stdout.getvalue())
        self.assertEqual(json.loads(stdout.getvalue())["kv_port"], 36000)

    def test_cli_reads_pod_identity_from_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            template_path = Path(directory) / "template.json"
            meta_path = Path(directory) / "meta.json"
            template_path.write_text(json.dumps(nixl_template()), encoding="utf-8")
            meta_path.write_text(
                json.dumps(
                    pd_meta(
                        connector="NixlConnector",
                        kv_port_base=None,
                        instance_stride=None,
                        port_span=None,
                    )
                ),
                encoding="utf-8",
            )
            stdout = io.StringIO()
            with patch.dict(
                os.environ,
                {"UC_PD_GROUP_NAME": "demo-0", "UC_PD_ROLE_ID": "prefill-0"},
            ), redirect_stdout(stdout):
                result = resolver.main(
                    [
                        "--template",
                        str(template_path),
                        "--meta",
                        str(meta_path),
                        "--skip-registry-probe",
                    ]
                )
        self.assertEqual(result, 0)
        self.assertEqual(json.loads(stdout.getvalue())["engine_id"], "0")

    def test_cli_output_file_is_independent_from_plugin_diagnostics(self) -> None:
        modules, _ = RegistryProbeTests.fake_vllm_modules()
        with tempfile.TemporaryDirectory() as directory:
            template_path = Path(directory) / "template.json"
            meta_path = Path(directory) / "meta.json"
            output_path = Path(directory) / "resolved.json"
            template_path.write_text(json.dumps(mooncake_template()), encoding="utf-8")
            meta_path.write_text(json.dumps(pd_meta()), encoding="utf-8")
            stdout = io.StringIO()
            stderr = io.StringIO()
            with patch.dict(sys.modules, modules), redirect_stdout(stdout), redirect_stderr(
                stderr
            ):
                result = resolver.main(
                    [
                        "--template",
                        str(template_path),
                        "--meta",
                        str(meta_path),
                        "--group-name",
                        "demo-0",
                        "--role-id",
                        "prefill-0",
                        "--output",
                        str(output_path),
                    ]
                )
            output_payload = output_path.read_text(encoding="utf-8")
        self.assertEqual(result, 0)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("plugin diagnostic", stderr.getvalue())
        self.assertEqual(json.loads(output_payload)["kv_port"], 36000)


if __name__ == "__main__":
    unittest.main()
