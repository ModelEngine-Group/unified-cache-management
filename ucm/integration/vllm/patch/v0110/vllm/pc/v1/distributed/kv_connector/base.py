from abc import ABC, abstractmethod


class KVConnectorBase_V1(ABC):
    def get_block_ids_with_load_errors(self) -> set[int]:
        return set()

    def has_connector_metadata(self) -> bool:
        """Check whether the connector metadata is currently set.

        Returns:
            bools: True if connector metadata exists, False otherwise.
        """
        return self._connector_metadata is not None
