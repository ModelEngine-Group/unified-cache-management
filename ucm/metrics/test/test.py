import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ucm.metrics import ucmmonitor

# import ucmmonitor

mon = ucmmonitor.UCMStatsMonitor.get_instance()
mon.update_stats(
    "UCMStats",
    {
        "save_duration": 1.2,
        "save_speed": 300.5,
        "load_duration": 0.8,
        "load_speed": 450.0,
        "interval_lookup_hit_rates": 0.95,
    },
)
mon.update_stats(
    "UCMStats",
    {
        "save_duration": 1.2,
        "save_speed": 300.5,
        "load_duration": 0.8,
        "load_speed": 450.0,
        "interval_lookup_hit_rates": 0.95,
    },
)

data = mon.get_stats("UCMStats")
print(data)
