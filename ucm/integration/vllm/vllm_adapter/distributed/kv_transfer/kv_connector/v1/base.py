from typing import Optional

import vllm.distributed.kv_transfer.kv_connector.v1.base as kvconnector_v1_base


class KVConnectorBase_V1(kvconnector_v1_base.KVConnectorBase_V1):
    def get_block_ids_with_load_errors(self) -> Optional[set[int]]:
        """
        Get the set of block IDs that failed to load.
        Returns:
            Optional[set[int]]: A set of block IDs that encountered load errors.
            Returns None if no errors occurred during load.
        """
        return None


kvconnector_v1_base.KVConnectorBase_V1 = KVConnectorBase_V1
