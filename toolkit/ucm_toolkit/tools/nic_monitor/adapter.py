"""NIC monitor adapter."""

from __future__ import annotations

import argparse

from ... import registry
from ...errors import CommandNotFoundError, ScriptNotFoundError
from ...registry import ToolAdapter
from ...runner import command_exists, run_command


class NicMonitorTool(ToolAdapter):
    """Adapter for passive NIC load monitoring."""

    name = "nic-monitor"
    aliases = ("nic_monitor",)
    description = "Run passive NIC load monitoring."
    buildable = False
    script_path = "toolkit/src/nic_monitor/nic_monitor_pro.sh"

    def add_run_args(self, parser: argparse.ArgumentParser) -> None:
        """Register NIC load run arguments."""
        parser.add_argument("mode", choices=("fg", "bg"), help="Monitor mode")
        parser.add_argument("args", nargs="*", help="Arguments forwarded to the monitor script")

    def run(self, tool_args: list[str]) -> int:
        """Run passive NIC load monitoring."""
        if not tool_args or tool_args[0] in ("-h", "--help"):
            self._print_run_help()
            return 0
        script = registry.resolve_repo_path(self.script_path or "")
        if not script.exists():
            raise ScriptNotFoundError(str(script))
        if not command_exists("bash"):
            raise CommandNotFoundError("bash")
        return run_command(["bash", str(script), *tool_args])

    def doctor(self, args: argparse.Namespace | None = None) -> int:
        """Inspect NIC monitor availability."""
        script = registry.resolve_repo_path(self.script_path or "")
        script_ok = script.exists()
        bash_ok = command_exists("bash")
        ethtool_ok = command_exists("ethtool")
        print(f"{self.name}:")
        print(f"  script:  {script} {'OK' if script_ok else 'MISSING'}")
        print(f"  bash:    {'OK' if bash_ok else 'MISSING'}")
        print(f"  ethtool: {'OK' if ethtool_ok else 'MISSING'}")
        print("  note:    the monitor script must be run as root or with sudo")
        return 0 if script_ok and bash_ok else 1

    def _print_run_help(self) -> None:
        print("usage: ucm-toolkit run nic-monitor { fg [interval_sec] | bg [duration_hours] [interval_sec] } [options]")
        print()
        print("Passive physical NIC traffic monitor.")
        print()
        print("Options:")
        print("  --log-dir PATH                 Background log directory (default: ./net_log)")
        print("  --stat-cycle-seconds SECONDS   Background summary interval (default: 3600)")
        print()
        print("Examples:")
        print("  ucm-toolkit run nic-monitor fg")
        print("  ucm-toolkit run nic-monitor fg 5")
        print("  ucm-toolkit run nic-monitor bg")
        print("  ucm-toolkit run nic-monitor bg 24 5")
        print("  ucm-toolkit run nic-monitor bg 24 5 --log-dir /mnt/test/net_log")
        print("  ucm-toolkit run nic-monitor bg 24 5 --stat-cycle-seconds 600")
        print()
        print("The underlying script requires root privileges for ethtool access.")
