"""
Jenkins Pipeline Client for UCM vLLM Service Pipeline.

Usage:
    client = JenkinsPipelineClient(
        jenkins_url="https://your-jenkins.com",
        username="admin",
        api_token="your-api-token",
        job_name="your-multibranch-job",
        branch="jenkins-dev-yhq"
    )
    
    build_number = client.trigger(
        platform="vllm-ascend",
        image_tag="some-tag",
        model_folder_name="Qwen3-1.7B"
    )
    
    client.wait_for_completion(build_number)
    print(client.get_build_status(build_number))
    client.download_artifacts(build_number, output_dir="./artifacts")
"""

import json
import os
import re
import time
import urllib.parse
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import requests
from requests.auth import HTTPBasicAuth


class BuildStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    UNSTABLE = "UNSTABLE"
    ABORTED = "ABORTED"
    NOT_BUILT = "NOT_BUILT"
    BUILDING = "BUILDING"  # still running
    QUEUED = "QUEUED"  # in the queue, not started
    UNKNOWN = "UNKNOWN"


@dataclass
class PipelineParameters:
    """
    Maps 1-to-1 to the pipeline's `parameters` block.
    Every field name matches the pipeline parameter name exactly.
    """

    # ── General ──────────────────────────────────────────────
    BUILD_NAME: str = ""
    PLATFORM: str = "vllm-ascend"  # vllm-ascend | vllm-cuda | mindie
    IMAGE_BUILD_TYPE: str = "黄区日构建"  # 黄区日构建 | 蓝区日构建 | 蓝区手动构建
    ImageTag: str = ""
    OVERRIDE_IMAGE: str = ""  # 完整镜像地址，优先级高于 ImageTag

    # ── Environment / Node ───────────────────────────────────
    GPU_COUNT: str = "1"
    DEPLOY_MODE: str = "single"  # single | multi

    # ── vLLM ─────────────────────────────────────────────────
    SERVER_PORT: str = "9527"
    VLLM_COMMAND_MASTER: str = (
        "vllm serve /home/models/Qwen3-1.7B\n"
        "--served-model-name Qwen3-1.7B\n"
        "--block-size 128\n"
        "--tensor-parallel-size 1\n"
        "--data-parallel-size 1\n"
        "--gpu-memory-utilization 0.87\n"
        "--trust-remote-code\n"
    )
    VLLM_COMMAND_WORKER: str = (
        "vllm serve /home/models/Qwen3-1.7B\n"
        "--served-model-name Qwen3-1.7B\n"
        "--block-size 128\n"
        "--tensor-parallel-size 1\n"
        "--data-parallel-size 1\n"
        "--gpu-memory-utilization 0.87\n"
        "--trust-remote-code\n"
    )

    # ── UCM ──────────────────────────────────────────────────
    ENABLE_UCM: str = "True"  # True | False
    UCM_CONFIG_YAML: str = ""
    USE_LAYERWISE: str = "false"  # true | false

    # ── Test ─────────────────────────────────────────────────
    API_MODEL_NAME: str = "Qwen3-1.7B"
    MODEL_FOLDER_NAME: str = "Qwen3-1.7B"
    PERF_TEST_CASE: str = "[[80, 10, 8, 8, 0, 80]]"
    TEST_PARAMS: str = "--feature=uc_performance_test --continue-on-collection-errors"

    def to_jenkins_params(self) -> List[Dict[str, str]]:
        """Convert to the JSON list that Jenkins' REST API expects."""
        return [
            {"name": k, "value": str(v)}
            for k, v in asdict(self).items()
            if v is not None
        ]


@dataclass
class BuildInfo:
    """Parsed result of a Jenkins build."""

    number: int
    url: str
    status: BuildStatus
    duration_ms: int = 0
    timestamp: int = 0
    description: Optional[str] = None
    parameters: Dict[str, str] = field(default_factory=dict)
    stages: List[Dict[str, Any]] = field(default_factory=list)
    artifacts: List[Dict[str, str]] = field(default_factory=list)
    environment_variables: Dict[str, str] = field(default_factory=dict)
    console_url: str = ""

    @property
    def is_running(self) -> bool:
        return self.status in (BuildStatus.BUILDING, BuildStatus.QUEUED)

    @property
    def is_successful(self) -> bool:
        return self.status == BuildStatus.SUCCESS


class JenkinsPipelineError(Exception):
    """Base exception for pipeline client errors."""


class BuildTriggerError(JenkinsPipelineError):
    """Raised when a build cannot be triggered."""


class BuildNotFoundError(JenkinsPipelineError):
    """Raised when a build number cannot be resolved."""


class JenkinsPipelineClient:
    """
    High-level client to interact with a Jenkins Pipeline.

    Supports both **Multibranch Pipeline** (when ``branch`` is provided) and
    **regular Pipeline** (when ``branch`` is omitted or empty).

    Parameters
    ----------
    jenkins_url : str
        Root URL of the Jenkins instance, e.g. ``https://jenkins.example.com``.
    username : str
        Jenkins user with Job/Build + Job/Read permissions.
    api_token : str
        Jenkins API token (generate from *User → Configure → API Token*).
    job_name : str
        Full *folder/job* path of the Pipeline item,
        e.g. ``"my-folder/UCM-Integration-Pipeline-MB"``.
    branch : str, optional
        Git branch for Multibranch Pipeline, e.g. ``"jenkins-dev-yhq"``.
        If empty or ``None``, the client uses regular pipeline URL pattern.
    verify_ssl : bool
        Whether to verify TLS certificates (default ``True``).
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        jenkins_url: str,
        username: str,
        api_token: str,
        job_name: str,
        branch: str = "",
        verify_ssl: bool = True,
    ):
        self.jenkins_url = jenkins_url.rstrip("/")
        self.username = username
        self.api_token = api_token
        self.job_name = job_name
        self.branch = branch or ""
        self.verify_ssl = verify_ssl

        self._session = requests.Session()
        self._session.auth = HTTPBasicAuth(username, api_token)
        self._session.verify = verify_ssl
        self._session.headers.update(
            {
                "Accept": "application/json",
            }
        )

        # Pre-compute the base URL for this job
        job_path = "/job/".join(self.job_name.split("/"))
        if self.branch:
            # Multibranch pipeline URL pattern:
            #   <jenkins>/job/<folder>/job/<pipeline>/job/<branch>/
            encoded_branch = urllib.parse.quote(self.branch, safe="")
            self._job_url = f"{self.jenkins_url}/job/{job_path}/job/{encoded_branch}"
        else:
            # Regular pipeline URL pattern:
            #   <jenkins>/job/<folder>/job/<pipeline>/
            self._job_url = f"{self.jenkins_url}/job/{job_path}"

    # ------------------------------------------------------------------
    # Public API – Trigger
    # ------------------------------------------------------------------
    def trigger(
        self,
        params: Optional[PipelineParameters] = None,
        *,
        wait_for_queue: bool = True,
        queue_poll_interval: float = 2.0,
        queue_timeout: float = 300.0,
        **param_overrides,
    ) -> int:
        """
        Trigger a new build and return the build number.

        Parameters
        ----------
        params : PipelineParameters, optional
            Full parameter set. If ``None``, defaults are used.
        wait_for_queue : bool
            Block until Jenkins assigns a build number (default ``True``).
        queue_poll_interval : float
            Seconds between queue polls.
        queue_timeout : float
            Maximum seconds to wait for the build to leave the queue.
        **param_overrides
            Convenient overrides applied on top of *params*, e.g.
            ``client.trigger(PLATFORM="vllm-cuda", ImageTag="v1.2")``.

        Returns
        -------
        int
            The Jenkins build number.

        Raises
        ------
        BuildTriggerError
            If the build could not be enqueued.
        BuildNotFoundError
            If the build number could not be resolved from the queue.
        """
        if params is None:
            params = PipelineParameters()

        # Apply overrides
        if param_overrides:
            d = asdict(params)
            d.update(param_overrides)
            params = PipelineParameters(**d)

        jenkins_params = params.to_jenkins_params()

        # Jenkins expects: buildWithParameters?json=...  OR  form-encoded params
        url = f"{self._job_url}/buildWithParameters"

        # Use form-encoded parameters (simpler, works with all Jenkins versions)
        form_data = {p["name"]: p["value"] for p in jenkins_params}

        resp = self._session.post(url, data=form_data)
        if resp.status_code not in (200, 201, 302):
            raise BuildTriggerError(
                f"Failed to trigger build: HTTP {resp.status_code}\n{resp.text}"
            )

        # Jenkins returns the queue item URL in the Location header
        queue_url = resp.headers.get("Location")
        if not queue_url:
            raise BuildTriggerError("Jenkins did not return a queue Location header.")

        if not wait_for_queue:
            return -1  # caller doesn't care about the number yet

        # Poll the queue until we get a build number
        queue_api = f"{queue_url}api/json"
        start = time.time()
        while time.time() - start < queue_timeout:
            try:
                qdata = self._get_json(queue_api)
            except Exception:
                time.sleep(queue_poll_interval)
                continue

            if "executable" in qdata and qdata["executable"]:
                build_number = qdata["executable"]["number"]
                print(f"✅ Build #{build_number} started.")
                return int(build_number)

            if qdata.get("cancelled"):
                raise BuildTriggerError("Queue item was cancelled.")

            why = qdata.get("why", "unknown reason")
            print(f"⏳ Waiting in queue: {why}")
            time.sleep(queue_poll_interval)

        raise BuildNotFoundError(f"Build number not resolved within {queue_timeout}s.")

    # ------------------------------------------------------------------
    # Public API – Status / Info
    # ------------------------------------------------------------------
    def get_build_url(self, build_number: int) -> str:
        """Return the human-readable URL for a build."""
        return f"{self._job_url}/{build_number}/"

    def get_console_url(self, build_number: int) -> str:
        """Return the URL to the console (log) output page."""
        return f"{self._job_url}/{build_number}/console"

    def get_build_status(self, build_number: int) -> BuildStatus:
        """
        Return the current status of a build.

        While the build is running, ``result`` is ``None`` in the API response,
        so we return ``BuildStatus.BUILDING``.
        """
        data = self._get_json(f"{self._job_url}/{build_number}/api/json")
        result = data.get("result")
        if result is None:
            if data.get("building"):
                return BuildStatus.BUILDING
            return BuildStatus.UNKNOWN
        try:
            return BuildStatus(result)
        except ValueError:
            return BuildStatus.UNKNOWN

    def get_build_info(self, build_number: int) -> BuildInfo:
        """
        Retrieve comprehensive information about a build.
        """
        data = self._get_json(
            f"{self._job_url}/{build_number}/api/json"
            "?tree=number,url,result,building,duration,timestamp,"
            "description,actions[parameters[name,value]],"
            "artifacts[relativePath,fileName]"
        )

        # Parse status
        result = data.get("result")
        if result is None:
            status = (
                BuildStatus.BUILDING if data.get("building") else BuildStatus.UNKNOWN
            )
        else:
            try:
                status = BuildStatus(result)
            except ValueError:
                status = BuildStatus.UNKNOWN

        # Parse parameters from actions
        params_dict: Dict[str, str] = {}
        for action in data.get("actions", []):
            for p in action.get("parameters", []):
                params_dict[p["name"]] = str(p.get("value", ""))

        # Parse artifacts
        artifacts = [
            {
                "fileName": a["fileName"],
                "relativePath": a["relativePath"],
                "downloadUrl": (
                    f"{self._job_url}/{build_number}" f"/artifact/{a['relativePath']}"
                ),
            }
            for a in data.get("artifacts", [])
        ]

        # Fetch pipeline stages via Workflow API (best-effort)
        stages = self._get_stages(build_number)

        # Fetch injected env vars (best-effort)
        env_vars = self._get_env_vars(build_number)

        return BuildInfo(
            number=data["number"],
            url=data.get("url", self.get_build_url(build_number)),
            status=status,
            duration_ms=data.get("duration", 0),
            timestamp=data.get("timestamp", 0),
            description=data.get("description"),
            parameters=params_dict,
            stages=stages,
            artifacts=artifacts,
            environment_variables=env_vars,
            console_url=self.get_console_url(build_number),
        )

    def get_console_log(
        self,
        build_number: int,
        start_offset: int = 0,
    ) -> str:
        """
        Retrieve the full (or partial) console text.

        Parameters
        ----------
        start_offset : int
            Byte offset to start reading from (useful for incremental reads).
        """
        url = (
            f"{self._job_url}/{build_number}"
            f"/logText/progressiveText?start={start_offset}"
        )
        resp = self._session.get(url)
        resp.raise_for_status()
        return resp.text

    def stream_console_log(
        self,
        build_number: int,
        poll_interval: float = 2.0,
        print_output: bool = True,
    ):
        """
        Generator that yields new console output chunks as they appear.

        Parameters
        ----------
        poll_interval : float
            Seconds between polls.
        print_output : bool
            If ``True``, also print each chunk to stdout.

        Yields
        ------
        str
            New chunk of console text.
        """
        offset = 0
        while True:
            url = (
                f"{self._job_url}/{build_number}"
                f"/logText/progressiveText?start={offset}"
            )
            resp = self._session.get(url)
            resp.raise_for_status()

            text = resp.text
            new_offset = int(resp.headers.get("X-Text-Size", offset))
            more_data = resp.headers.get("X-More-Data", "false") == "true"

            if text:
                if print_output:
                    print(text, end="")
                yield text

            offset = new_offset

            if not more_data:
                break

            time.sleep(poll_interval)

    # ------------------------------------------------------------------
    # Public API – Wait
    # ------------------------------------------------------------------
    def wait_for_completion(
        self,
        build_number: int,
        poll_interval: float = 15.0,
        timeout: float = 43200.0,  # 12 hours (matches pipeline timeout)
        stream_log: bool = False,
    ) -> BuildInfo:
        """
        Block until the build finishes and return the final ``BuildInfo``.

        Parameters
        ----------
        poll_interval : float
            Seconds between status checks.
        timeout : float
            Maximum wait time in seconds.
        stream_log : bool
            If ``True``, stream console output to stdout while waiting.

        Returns
        -------
        BuildInfo
            Final build information.

        Raises
        ------
        TimeoutError
            If the build hasn't finished within *timeout* seconds.
        """
        start = time.time()
        log_offset = 0

        while time.time() - start < timeout:
            status = self.get_build_status(build_number)

            if stream_log:
                url = (
                    f"{self._job_url}/{build_number}"
                    f"/logText/progressiveText?start={log_offset}"
                )
                resp = self._session.get(url)
                if resp.ok and resp.text:
                    print(resp.text, end="")
                    log_offset = int(resp.headers.get("X-Text-Size", log_offset))

            if status not in (BuildStatus.BUILDING, BuildStatus.QUEUED):
                return self.get_build_info(build_number)

            elapsed = int(time.time() - start)
            if not stream_log:
                print(
                    f"⏳ Build #{build_number} still {status.value} "
                    f"({elapsed}s elapsed)…"
                )
            time.sleep(poll_interval)

        raise TimeoutError(f"Build #{build_number} did not complete within {timeout}s.")

    # ------------------------------------------------------------------
    # Public API – Artifacts
    # ------------------------------------------------------------------
    def list_artifacts(self, build_number: int) -> List[Dict[str, str]]:
        """
        Return a list of artifact metadata dicts with keys:
        ``fileName``, ``relativePath``, ``downloadUrl``.
        """
        info = self.get_build_info(build_number)
        return info.artifacts

    def download_artifacts(
        self,
        build_number: int,
        output_dir: str = ".",
        filename_filter: Optional[str] = None,
    ) -> List[str]:
        """
        Download all (or filtered) build artifacts to *output_dir*.

        Parameters
        ----------
        output_dir : str
            Local directory to save files into.
        filename_filter : str, optional
            Regex pattern; only artifacts whose ``fileName`` matches are
            downloaded.

        Returns
        -------
        list of str
            Paths to the downloaded files.
        """
        os.makedirs(output_dir, exist_ok=True)
        artifacts = self.list_artifacts(build_number)
        downloaded: List[str] = []

        for art in artifacts:
            if filename_filter and not re.search(filename_filter, art["fileName"]):
                continue

            url = art["downloadUrl"]
            dest = os.path.join(output_dir, art["fileName"])
            print(f"⬇️  Downloading {art['fileName']} → {dest}")

            resp = self._session.get(url, stream=True)
            resp.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            downloaded.append(dest)

        if not downloaded:
            print("ℹ️  No artifacts found (or none matched the filter).")
        return downloaded

    # ------------------------------------------------------------------
    # Public API – Abort
    # ------------------------------------------------------------------
    def abort_build(self, build_number: int) -> bool:
        """
        Attempt to abort a running build.

        Returns ``True`` if the abort request was accepted.
        """
        url = f"{self._job_url}/{build_number}/stop"
        resp = self._session.post(url)
        return resp.status_code in (200, 302)

    # ------------------------------------------------------------------
    # Public API – Latest builds
    # ------------------------------------------------------------------
    def get_last_build_number(self) -> Optional[int]:
        """Return the latest build number, or ``None``."""
        data = self._get_json(f"{self._job_url}/api/json?tree=lastBuild[number]")
        lb = data.get("lastBuild")
        return lb["number"] if lb else None

    def get_last_n_builds(self, n: int = 10) -> List[BuildInfo]:
        """Return ``BuildInfo`` for the last *n* builds."""
        data = self._get_json(
            f"{self._job_url}/api/json" f"?tree=builds[number]{{0,{n}}}"
        )
        builds = data.get("builds", [])
        return [self.get_build_info(b["number"]) for b in builds]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _get_json(self, url: str) -> dict:
        resp = self._session.get(url)
        resp.raise_for_status()
        return resp.json()

    def _get_stages(self, build_number: int) -> List[Dict[str, Any]]:
        """
        Use the Pipeline: Stage View / Workflow API to fetch stage info.
        Endpoint: ``<build>/wfapi/describe``
        """
        try:
            data = self._get_json(f"{self._job_url}/{build_number}/wfapi/describe")
            return [
                {
                    "name": s.get("name"),
                    "status": s.get("status"),
                    "durationMillis": s.get("durationMillis", 0),
                }
                for s in data.get("stages", [])
            ]
        except Exception:
            return []

    def _get_env_vars(self, build_number: int) -> Dict[str, str]:
        """
        Fetch injected environment variables via the ``injectedEnvVars`` API.
        Requires the *EnvInject* plugin.
        """
        try:
            data = self._get_json(
                f"{self._job_url}/{build_number}" "/injectedEnvVars/api/json"
            )
            return data.get("envMap", {})
        except Exception:
            return {}

    def __repr__(self) -> str:
        return (
            f"JenkinsPipelineClient("
            f"job={self.job_name!r}, branch={self.branch!r}, "
            f"url={self._job_url!r})"
        )
