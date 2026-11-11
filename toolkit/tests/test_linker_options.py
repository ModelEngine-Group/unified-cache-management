"""Check linker selection without importing the inference framework."""

from ucm_toolkit.tools.sglang_model_check.adapter import SglangModelCheckTool
from ucm_toolkit.tools.sglang_model_check.config import CheckConfig


def test_linker_option_and_config_roundtrip():
    parser = SglangModelCheckTool()._parser()
    assert parser.parse_args(["--model", "local"]).integration == "hicache"
    args = parser.parse_args([
        "--model", "local", "--integration", "linker",
        "--tp-size", "2", "--pp-size", "2",
    ])
    config = CheckConfig.from_mapping(vars(args))
    assert config.integration == "linker"
    assert (config.tp_size, config.pp_size) == (2, 2)
