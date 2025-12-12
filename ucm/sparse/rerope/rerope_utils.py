import os


class VllmPatchConfig:

    @property
    def rerope_window(self) -> int:
        return int(os.getenv("REROPE_WINDOW", "32768"))

    @property
    def training_length(self) -> int:
        return int(os.getenv("TRAINING_LENGTH", "32768"))


default_config = VllmPatchConfig()
