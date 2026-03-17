from ucm.integration.vllm.patch.utils import patch_or_inject, when_imported


@when_imported("vllm.logger")
def patch_logger(mod):
    from ucm import logger

    patch_or_inject(mod, "init_logger", logger.init_logger)
    patch_or_inject(mod, "current_formatter_type", logger.current_formatter_type)
