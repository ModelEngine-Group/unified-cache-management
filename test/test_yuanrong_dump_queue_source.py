import re
import unittest
from pathlib import Path
from typing import Optional


class YuanRongDumpQueueSourceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source_path = (
            Path(__file__).resolve().parents[1]
            / "ucm"
            / "store"
            / "yuanrongstore"
            / "cc"
            / "dump_queue.cc"
        )
        load_source_path = source_path.with_name("load_queue.cc")
        backfill_source_path = source_path.with_name("backfill_queue.cc")
        store_source_path = source_path.with_name("yuanrong_store.cc")
        cls.source = source_path.read_text()
        cls.load_source = load_source_path.read_text()
        cls.store_source = store_source_path.read_text()
        cls.backfill_source = (
            backfill_source_path.read_text() if backfill_source_path.exists() else ""
        )

    def _function_body(self, name: str, source: Optional[str] = None) -> str:
        source = self.source if source is None else source
        pattern = re.escape(name) + r"\([^)]*\)\s*(?:override\s*)?\{"
        match = re.search(pattern, source)
        self.assertIsNotNone(match, f"{name} not found")
        start = match.end()
        depth = 1
        index = start
        while index < len(source) and depth:
            if source[index] == "{":
                depth += 1
            elif source[index] == "}":
                depth -= 1
            index += 1
        self.assertEqual(depth, 0, f"{name} body is not balanced")
        return source[start : index - 1]

    def test_d2h_device_is_initialized_in_worker_thread(self):
        setup_body = self._function_body("DumpQueue::Setup")
        d2h_body = self._function_body("DumpQueue::D2HStage")

        self.assertIn("&DumpQueue::D2HStage", setup_body)
        self.assertIn("Trans::Device device", d2h_body)
        self.assertIn("device.Setup(config_.deviceId)", d2h_body)
        self.assertIn("ready_.ConsumerLoop", d2h_body)

    def test_mset_d2h_overwrites_existing_yuanrong_objects(self):
        dump_body = self._function_body("DumpQueue::DumpReadyTask")

        self.assertIn("setParam.existence = datasystem::ExistenceOpt::NONE", dump_body)
        self.assertNotIn("setParam.existence = datasystem::ExistenceOpt::NX", dump_body)

    def test_yuanrong_dump_uses_payload_address_after_metadata_header(self):
        persist_body = self._function_body("DumpQueue::PersistBatch")
        load_recover_body = self._function_body(
            "BackfillQueue::RunOne", self.backfill_source
        )

        self.assertNotIn(
            "GetSize() != static_cast<int64_t>(config_.objectSize)", persist_body
        )
        self.assertNotIn(
            "size != static_cast<int64_t>(config_.objectSize)", load_recover_body
        )
        self.assertIn("GetYuanRongPayloadAddress", persist_body)
        self.assertIn("payloadAddress", persist_body)
        self.assertIn("YuanRongComposedObjectSize", load_recover_body)
        self.assertIn("InitYuanRongComposedBuffer", load_recover_body)

    def test_posix_dump_validates_yuanrong_payload_layout_before_using_buffer(self):
        persist_body = self._function_body("DumpQueue::PersistBatch")

        self.assertIn("GetMetaInfo(keys, false", persist_body)
        self.assertIn("ValidateYuanRongBlobSizes", persist_body)
        self.assertLess(
            persist_body.index("ValidateYuanRongBlobSizes"),
            persist_body.index("kvClient_->Get"),
        )

    def test_dump_persists_only_mset_confirmed_local_keys(self):
        dump_body = self._function_body("DumpQueue::DumpReadyTask")

        self.assertIn(
            "MSetD2H(keys, blobLists, setParam, &localSetKeys)",
            dump_body,
        )
        self.assertIn(
            "FilterKeysByLocalSetKeys(localSetKeys, keys, task->desc)",
            dump_body,
        )
        self.assertIn("PersistenceTask", dump_body)
        self.assertIn("persistence_.TryPush", dump_body)
        self.assertNotIn("heteroClient_->Exist", dump_body)
        self.assertNotIn("postExist", dump_body)

    def test_mset_error_without_confirmed_local_key_fails_dump(self):
        dump_body = self._function_body("DumpQueue::DumpReadyTask")

        empty_pos = dump_body.index("if (localSetKeys.empty())")
        error_pos = dump_body.index("if (dumpStatus.IsError())", empty_pos)
        selection_pos = dump_body.index("FilterKeysByLocalSetKeys")
        partial_pos = dump_body.index("partially failed", selection_pos)
        self.assertLess(empty_pos, error_pos)
        self.assertLess(error_pos, selection_pos)
        self.assertLess(selection_pos, partial_pos)

    def test_partial_kv_get_persists_only_latched_buffers(self):
        persist_body = self._function_body("DumpQueue::PersistBatch")

        self.assertIn("if (!buffer)", persist_body)
        self.assertIn("lockedBuffers.push_back(std::move(buffer))", persist_body)
        self.assertIn("if (backendTask.empty())", persist_body)
        self.assertIn("std::move(lockedBuffers)", persist_body)
        self.assertIn("persisting readable buffers only", persist_body)

    def test_background_persistence_logs_queue_and_submit_timing(self):
        persist_body = self._function_body("DumpQueue::PersistBatch")

        self.assertIn("queue_wait={:.3f}ms", persist_body)
        self.assertIn("(persistenceStart - task.enqueueTime) * 1e3", persist_body)
        self.assertIn("prepare_submit={:.3f}ms", persist_body)
        self.assertIn("(backendSubmitEnd - kvGetEnd) * 1e3", persist_body)

    def test_background_persistence_logs_mebibyte_values(self):
        persist_body = self._function_body("DumpQueue::PersistBatch")

        self.assertIn("mb={:.3f}", persist_body)
        self.assertIn("inflight_mb={:.3f}", persist_body)
        self.assertEqual(persist_body.count("/ (1024.0 * 1024.0)"), 2)
        self.assertNotIn("inflight_bytes=", persist_body)

    def test_yuanrong_load_probes_miss_with_zero_timeout(self):
        load_body = self._function_body("LoadQueue::LoadThenRecover", self.load_source)

        self.assertIn("constexpr int32_t mgetTimeoutMs = 0", load_body)
        self.assertIn("MGetH2D(keys, blobLists, failedKeys, mgetTimeoutMs)", load_body)
        self.assertNotIn("firstGetTimeoutMs", load_body)
        self.assertNotIn("missTimeoutMs", load_body)

    def test_yuanrong_load_logs_total_mget_mb(self):
        load_body = self._function_body("LoadQueue::LoadThenRecover", self.load_source)

        self.assertIn(
            "config_.objectSize) * keys.size() / (1024.0 * 1024.0)", load_body
        )
        self.assertIn("mode=mget_first", load_body)
        self.assertIn("h2d_keys={}, h2d_mb={:.3f}", load_body)

    def test_yuanrong_load_emits_one_summary_per_execution_path(self):
        load_one_body = self._function_body("LoadQueue::LoadOne", self.load_source)
        fallback_body = self._function_body(
            "LoadQueue::LoadThenRecover", self.load_source
        )

        self.assertEqual(load_one_body.count("mode=parallel"), 1)
        self.assertEqual(fallback_body.count("mode=mget_first"), 1)
        self.assertNotIn("recovered keys=", fallback_body)

    def test_yuanrong_posix_keys_h2d_precedes_async_backfill(self):
        recover_body = self._function_body(
            "LoadQueue::RecoverFromBackend", self.load_source
        )
        finalize_body = self._function_body(
            "LoadQueue::FinalizeHostBatch", self.load_source
        )

        self.assertIn("config_.recoveryBatchSize", recover_body)
        self.assertIn("PrepareHostBatch", recover_body)
        self.assertIn("FinalizeHostBatch", recover_body)
        wait_pos = finalize_body.index("backend_->Wait")
        h2d_pos = finalize_body.index("HostToDeviceScatterAsync")
        sync_pos = finalize_body.index("stream.Synchronize")
        backfill_pos = finalize_body.index("backfillQueue_.Submit")
        self.assertLess(wait_pos, h2d_pos)
        self.assertLess(h2d_pos, sync_pos)
        self.assertLess(sync_pos, backfill_pos)
        self.assertNotIn("MGetH2D", finalize_body)

    def test_yuanrong_async_backfill_does_not_fail_front_load(self):
        run_body = self._function_body("BackfillQueue::RunOne", self.backfill_source)

        self.assertIn("kvClient_->MCreate", run_body)
        self.assertIn("kvClient_->MSet", run_body)
        self.assertIn("InitYuanRongComposedBuffer", run_body)
        self.assertNotIn("failureSet_", run_body)

    def test_invalid_queue_config_identifies_the_parameter_and_values(self):
        check_body = self._function_body("CheckConfig", self.store_source)

        self.assertNotIn("invalid YuanRong queue depth", check_body)
        self.assertIn("yuanrong_waiting_queue_depth({})", check_body)
        self.assertIn("yuanrong_load_worker_count", check_body)
        self.assertIn("yuanrong_recovery_batch_size", check_body)
        self.assertIn("yuanrong_host_buffer_count({})", check_body)
        self.assertIn("config.hostBufferCount, config.recoveryBatchSize", check_body)
        self.assertIn("yuanrong_h2d_stream_count", check_body)
        self.assertIn("yuanrong_backfill_worker_count", check_body)
        self.assertIn("yuanrong_backfill_queue_depth", check_body)
        self.assertNotIn("yuanrong_reaper_queue_depth", check_body)

    def test_device_memory_preregistration_runs_during_store_setup(self):
        setup_body = self._function_body("Setup", self.store_source)
        register_body = self._function_body("RegisterKvBuffers", self.store_source)
        parse_body = self._function_body("ParseConfig", self.store_source)

        register_pos = setup_body.index("RegisterKvBuffers(config_)")
        task_manager_pos = setup_body.index("taskManager_.Setup")
        self.assertLess(register_pos, task_manager_pos)
        self.assertIn("gpu_kv_buffer_addrs", parse_body)
        self.assertIn("gpu_kv_buffer_sizes", parse_body)
        self.assertIn("PreRegisterDeviceMemory(addrs, sizes)", register_body)
        self.assertNotIn("Status RegisterMemory", self.store_source)

    def test_lookup_rejects_any_yuanrong_exist_error(self):
        lookup_body = self._function_body("LookupYuanRong", self.store_source)

        self.assertIn("if (status.IsError())", lookup_body)
        self.assertNotIn("status.IsError() && exists.empty()", lookup_body)
        self.assertLess(
            lookup_body.index("if (status.IsError())"),
            lookup_body.index("if (exists.size() != num)"),
        )

    def test_host_buffer_count_is_derived_and_explicit_value_is_preserved(self):
        parse_body = self._function_body("ParseConfig", self.store_source)
        derive_body = self._function_body("DeriveHostBufferCount", self.store_source)

        self.assertIn("yuanrong_host_buffer_capacity_gb", parse_body)
        self.assertIn("config.hostBufferCount != 0", parse_body)
        self.assertIn("DeriveHostBufferCount(config)", parse_body)
        self.assertIn("config.hostBufferCountExplicit", derive_body)
        self.assertIn("DeriveYuanRongHostBufferCount", derive_body)

    def test_pure_yuanrong_skips_recovery_resources(self):
        setup_body = self._function_body("LoadQueue::Setup", self.load_source)
        derive_body = self._function_body("DeriveHostBufferCount", self.store_source)

        self.assertIn("if (backend_ != nullptr)", setup_body)
        self.assertIn("hostBufferPool_.Setup", setup_body)
        self.assertIn("backfillQueue_.Setup", setup_body)
        self.assertIn("config.storeBackend == nullptr", derive_body)
        self.assertIn("config.hostBufferCount = 0", derive_body)

    def test_yuanrong_posix_accepts_aio_only_with_direct_io(self):
        check_body = self._function_body("CheckConfig", self.store_source)

        self.assertIn('config.posixIoEngine != "aio"', check_body)
        self.assertIn('config.posixIoEngine == "aio"', check_body)
        self.assertIn("!config.ioDirect", check_body)
        self.assertIn("posix_io_engine=aio requires io_direct=true", check_body)
        self.assertNotIn("return Status::Unsupported()", check_body)


if __name__ == "__main__":
    unittest.main()
