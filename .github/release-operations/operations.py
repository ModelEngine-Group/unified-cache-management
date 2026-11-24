"""Operation lifecycle for the single Release body status line."""

from contextlib import contextmanager
from functools import wraps


def tracked(label):
    def decorate(function):
        @wraps(function)
        def invoke(self, release, *args, **kwargs):
            kind = label
            if label == "initialize":
                kind = (
                    "Release initialization"
                    if (args[0] if args else kwargs.get("command")) == "release_init"
                    else "Release synchronization"
                )
                if kwargs.get("dry_run", len(args) > 1 and args[1]):
                    kind += " preview"
            with self.snapshots():
                with self.operation(release, kind):
                    return function(self, release, *args, **kwargs)

        return invoke

    return decorate


class Operations:
    @contextmanager
    def operation(self, release, kind):
        if getattr(self, "_passive", False) and kind == "Release synchronization":
            yield
            return
        active = getattr(self, "_active_operations", None)
        if active is None:
            active = self._active_operations = set()
        if release in active:
            yield
            return
        _, state = self.load(release)
        if kind == "Release synchronization" and not state.get("initialized"):
            yield
            return
        active.add(release)
        state["operation"] = {"kind": kind, "url": self.run_url, "status": "running"}
        try:
            self.save(release, state, "Operation started")
            yield
        except Exception as error:
            self.finish_operation(release, False, str(error))
            raise
        else:
            self.finish_operation(release, True)
        finally:
            active.discard(release)

    def finish_operation(self, release, success, error=None):
        _, state = self.load(release)
        operation = state.get("operation", {})
        if operation.get("url") != self.run_url:
            return
        operation["status"] = "success" if success else "failure"
        if error:
            operation["error"] = " ".join(error.split())[:240]
        self.save(release, state, state.get("last_result", ""))

    def finalize_operations(self, job_status):
        if job_status == "success":
            return
        for release, row in self.directory().rows().items():
            operation = row.get("operation", {})
            if (
                operation.get("url") == self.run_url
                and operation.get("status") == "running"
            ):
                with self.snapshots():
                    self.finish_operation(release, False, "Workflow " + job_status)
