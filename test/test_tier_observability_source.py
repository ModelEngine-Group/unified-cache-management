import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


class TierObservabilitySourceTest(unittest.TestCase):
    def test_yuanrong_records_final_shard_sources(self):
        source = (REPO_ROOT / "ucm/store/yuanrongstore/cc/load_queue.cc").read_text(
            encoding="utf-8"
        )

        self.assertIn('"yuanrong_load_success_shards_total"', source)
        self.assertIn('"yuanrong_lookup_miss_posix_load_success_shards_total"', source)
        self.assertIn(
            '"yuanrong_load_fallback_posix_load_success_shards_total"', source
        )
        self.assertIn("RecoverySource::LOOKUP_MISS", source)
        self.assertIn("RecoverySource::LOAD_FALLBACK", source)
        self.assertIn("stats.yuanrongSuccess +=", source)
        self.assertIn("if (status.Success())", source)

    def test_cache_records_each_shard_without_tp_adjustment(self):
        source = (REPO_ROOT / "ucm/store/cache/cc/load_queue.cc").read_text(
            encoding="utf-8"
        )

        source_assignment = "shardTask.fromPosix = !shardTask.bufferHandle.Ready();"
        owner_branch = (
            "if (shardTask.bufferHandle.Owner() && " "!shardTask.bufferHandle.Ready())"
        )
        self.assertLess(source.index(source_assignment), source.index(owner_branch))
        self.assertIn('"cache_load_success_shards_total"', source)
        self.assertIn('"cache_posix_load_success_shards_total"', source)
        self.assertNotIn("tpSize", source)

    def test_posix_gc_exports_logical_capacity(self):
        source = (REPO_ROOT / "ucm/store/posix/cc/shard_gc.cc").read_text(
            encoding="utf-8"
        )

        self.assertIn('"posix_store_used_bytes"', source)
        self.assertIn('"posix_store_capacity_bytes"', source)
        self.assertIn('"posix_store_usage_ratio"', source)
        self.assertIn("estimatedFiles", source)

    def test_yuanrong_resource_reporter_runs_in_scheduler(self):
        source = (REPO_ROOT / "ucm/integration/vllm/ucm_connector.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("role == KVConnectorRole.SCHEDULER and consumer_enabled(", source)
        self.assertIn("self._start_yuanrong_resource_reporter()", source)
        self.assertNotIn(
            "role == KVConnectorRole.WORKER and consumer_enabled(\n"
            "            self.metrics_config, VLLM_CONNECTOR_CONSUMER\n"
            "        ):\n"
            "            self._start_yuanrong_resource_reporter()",
            source,
        )


if __name__ == "__main__":
    unittest.main()
