from vllm.v1.outputs import KVConnectorOutput


class KVOutputAggregator:
    def aggregate(self, outputs: list, output_rank: int = 0):
        # Aggregate kv_connector_output from all workers

        def update_finished_set(req_ids, remaining_count_dict, finished_set):
            for req_id in req_ids or ():
                remaining_count_dict[req_id] -= 1
                if remaining_count_dict[req_id] == 0:
                    finished_set.add(req_id)
                    del remaining_count_dict[req_id]

        finished_sending = set[str]()
        finished_recving = set[str]()
        aggregated_kv_connector_stats = None
        invalid_block_ids: set[int] = set()
        for model_runner_output in outputs:
            output = model_runner_output.kv_connector_output
            if not output:
                continue
            update_finished_set(
                output.finished_sending, self._send_remaining_count, finished_sending
            )
            update_finished_set(
                output.finished_recving, self._recv_remaining_count, finished_recving
            )

            # Aggregate kv_connector_stats from all workers.
            if aggregated_kv_connector_stats is None:
                # Use the first worker's kv_connector_stats as accumulator.
                aggregated_kv_connector_stats = output.kv_connector_stats
            elif kv_connector_stats := output.kv_connector_stats:
                if aggregated_kv_connector_stats is None:
                    aggregated_kv_connector_stats = kv_connector_stats
                else:
                    assert isinstance(
                        aggregated_kv_connector_stats, type(kv_connector_stats)
                    )
                    aggregated_kv_connector_stats = (
                        aggregated_kv_connector_stats.aggregate(kv_connector_stats)
                    )

            invalid_block_ids |= getattr(output, "invalid_block_ids", set())

        # select output of the worker specified by output_rank
        output = outputs[output_rank]

        output.kv_connector_output = KVConnectorOutput(
            finished_sending=finished_sending or None,
            finished_recving=finished_recving or None,
            kv_connector_stats=aggregated_kv_connector_stats or None,
            invalid_block_ids=invalid_block_ids,
        )
        # if invalid_block_ids:
        #    logger.warning(
        #        f"[kv-load] aggregate invalid_block_ids={len(invalid_block_ids)} "
        #        f"sample={list(invalid_block_ids)[:10]}"
        #    )

        return output
