from ucm.shared.metrics import ucmmonitor


class ConnStats:
    def __init__(self, name: str = "PyStats1"):
        self._name = name
        self._data = {}

    def Name(self) -> str:
        return self._name

    def Update(self, params):
        for k, v in params.items():
            self._data.setdefault(k, []).append(v)

    def Reset(self):
        self._data.clear()

    def Data(self):
        return self._data.copy()
