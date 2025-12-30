from typing import Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import base64
import json
import os
# import sys
import urllib.request

from pythia.http_utils import http_get, http_post
from pythia.types import Result

HOME = os.environ["HOME"]
FOOBAR_DIR = os.path.join(HOME, ".pythia", "foobar")
STATE_DIR = os.path.join(FOO_DIR, "state")

@dataclass
class ForgeInstance:
    name: str
    host: str = None
    protocol: str = None
    tokens: dict[str, str] = None

    def api_base_url(self) -> str:
        return f"{self.host}/api"

@dataclass
class ForgeRegistry:
    environ: dict[str, str] = field(default_factory=dict)
    instances: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self._register_all()

    def get_env(self, key: str) -> Optional[str]:
        v = self.environ.get(key, None)
        if v is None:
            v = os.environ.get(key, None)
            if v is None:
                pass
            else:
                self.environ[key] = v
        return v

    def find_instance(self, name: str) -> Optional[ForgeInstance]:
        if name not in self.instances:
            return None
        return self.instances[name]

    def register_instance(self, name: str, **kwargs):
        if name in self.instances:
            raise KeyError
        self.instances[name] = ForgeInstance(name=name, **kwargs)

    def _register_all(self):
        api_token = self.get_env("CODEBERG_API_KEY")
        register_token = self.get_env("CODEBERG_RUNNER_TOKEN")
        # register_token = self.get_env("CODEBERG_RUNNER_REGISTER_TOKEN")
        self.register_instance(
            "codeberg.org",
            host="https://codeberg.org",
            protocol="forgejo",
            tokens={
                "api": api_token,
                "runner.register": register_token,
            },
        )

@dataclass
class ForgeStateJournal:
    _log_path: str = None

    def __post_init__(self) -> None:
        if self._log_path is None:
            self._log_path = os.path.join(STATE_DIR, "_log.jsonl")

    def get(self, type_: str, key=None):
        item = None
        with open(self._log_path, "r") as log_file:
            for line in log_file:
                entry = json.loads(line)
                if (
                    type_ == entry["type"] and
                    key is None or key == entry["key"]
                ):
                    item = entry["item"]
        return item

    def put(self, type_: str, item, key=None):
        entry = {
            "type": type_,
            "key": key,
            "item": item,
        }
        with open(self._log_path, "a") as log_file:
            print(json.dumps(entry), file=log_file, flush=True)

@dataclass
class ForgeAPIClient:
    token: str
    _instance: ForgeInstance = None
    _registry: ForgeRegistry = None
    _state: ForgeStateJournal = None

    @classmethod
    def default(cls) -> "ForgeAPIClient":
        registry = ForgeRegistry()
        instance = registry.find_instance("codeberg.org")
        token = instance.tokens["api"]
        if False:
            api = ForgeAPI(
                name="autopythia",
                token=token,
            )
        return cls(token, instance, registry)

    def __post_init__(self) -> None:
        if self._registry is None:
            self._registry = ForgeRegistry()
        if self._state is None:
            self._state = ForgeStateJournal()

    def _http_get(self, endpoint: str, params: Optional[dict] = None) -> Result:
        endpoint = endpoint.removeprefix("/")
        base_url = self._instance.api_base_url()
        req_url = f"{base_url}/v1/{endpoint}"
        print(f"DEBUG: ForgeAPIClient._http_get: req url  = {req_url}")
        req_headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"token {self.token}",
        }
        print(f"DEBUG: ForgeAPIClient._http_get: req hdrs = {req_headers}")
        req_params = params
        print(f"DEBUG: ForgeAPIClient._http_post: req prms = {req_params}")
        res = http_get(req_url, req_headers, req_params)
        if "err" in res:
            print(f"DEBUG: ForgeAPIClient._http_get: res exc  = {res['exc']}")
            return res
        print(f"DEBUG: ForgeAPIClient._http_get: res body = {res['ok']}")
        return res

    def _http_post(self, endpoint: str, params: Optional[dict], body: dict) -> Result:
        endpoint = endpoint.removeprefix("/")
        base_url = self._instance.api_base_url()
        req_url = f"{base_url}/v1/{endpoint}"
        print(f"DEBUG: ForgeAPIClient._http_post: req url  = {req_url}")
        req_headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"token {self.token}",
        }
        print(f"DEBUG: ForgeAPIClient._http_post: req hdrs = {req_headers}")
        req_params = params
        print(f"DEBUG: ForgeAPIClient._http_post: req prms = {req_params}")
        req_body = body
        print(f"DEBUG: ForgeAPIClient._http_post: req body = {req_body}")
        res = http_post(req_url, req_headers, req_params, req_body)
        if "err" in res:
            print(f"DEBUG: ForgeAPIClient._http_post: res exc  = {res['exc']}")
            return res
        print(f"DEBUG: ForgeAPIClient._http_post: res body = {res['ok']}")
        return res

    def _issues(self, owner: str, repo: str) -> Result:
        res = self._http_get(f"/repos/{owner}/{repo}/issues")
        if "err" in res:
            return res
        res_body = res["ok"]
        item = res_body
        if False:
            print(f"DEBUG: ForgeAPIClient._issues: item = {item}")
        return {"ok": item}

@dataclass
class ForgeRunner:
    name: str
    token: str
    version: Optional[str] = None
    labels: Optional[list[str]] = None

@dataclass
class ForgeRunnerClient:
    runner: ForgeRunner
    _instance: ForgeInstance = None
    _registry: ForgeRegistry = None
    _state: ForgeStateJournal = None
    # _workflow: ForgeWorkflowCache = None

    @classmethod
    def default(cls) -> "ForgeRunnerClient":
        registry = ForgeRegistry()
        instance = registry.find_instance("codeberg.org")
        register_token = instance.tokens["runner.register"]
        runner = ForgeRunner(
            name="autopythia-runner-0",
            token=register_token,
            labels=["autopythia"],
        )
        return cls(runner, instance, registry)

    def __post_init__(self) -> None:
        if self._registry is None:
            self._registry = ForgeRegistry()
        if self._state is None:
            self._state = ForgeStateJournal()

    def _http_post(self, endpoint: str, req_body: dict, register: bool = False) -> Result:
        base_url = self._instance.api_base_url()
        req_url = f"{base_url}/actions/runner.v1.RunnerService/{endpoint}"
        print(f"DEBUG: ForgeRunnerClient._http_post: req url  = {req_url}")
        req_headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if not register:
            register_item = self._state.get("runner.register", key=self.runner.name)
            runner_uuid = register_item["uuid"]
            runner_token = register_item["token"]
            req_headers["X-Runner-UUID"] = runner_uuid
            req_headers["X-Runner-Token"] = runner_token
        print(f"DEBUG: ForgeRunnerClient._http_post: req hdrs = {req_headers}")
        print(f"DEBUG: ForgeRunnerClient._http_post: req body = {req_body}")
        res = http_post(req_url, req_headers, req_body)
        if "err" in res:
            print(f"DEBUG: ForgeRunnerClient._http_post: res exc  = {res['exc']}")
            return res
        print(f"DEBUG: ForgeRunnerClient._http_post: res body = {res['ok']}")
        return res

    def _register(self) -> Result:
        item = self._state.get("runner.register", key=self.runner.name)
        if item is not None:
            print(f"DEBUG: ForgeRunnerClient._register: get item = {item}")
            return {"ok": item}
        req_body = {
            "name": self.runner.name,
            "token": self.runner.token,
        }
        if self.runner.version is not None:
            req_body["version"] = self.runner.version
        if self.runner.labels is not None:
            req_body["labels"] = self.runner.labels
        res = self._http_post("Register", req_body, register=True)
        if "err" in res:
            return res
        res_body = res["ok"]
        item = res_body["runner"]
        self._state.put("runner.register", item, key=self.runner.name)
        print(f"DEBUG: ForgeRunnerClient._register: put item = {item}")
        return {"ok": item}

    def _declare(self) -> Result:
        register_item = self._state.get("runner.register", key=self.runner.name)
        runner_labels = register_item["labels"]
        req_body = {
            "labels": runner_labels,
        }
        res = self._http_post("Declare", req_body)
        if "err" in res:
            return res
        res_body = res["ok"]
        item = res_body
        if False:
            self._state.put("runner.declare", item, key=self.runner.name)
            print(f"DEBUG: ForgeRunnerClient._declare: put item = {item}")
        return {"ok": item}

    def _fetch_task(self) -> Result:
        req_body = {
            # "tasks_version": "1",
            # "tasksVersion": "1",
        }
        res = self._http_post("FetchTask", req_body)
        if "err" in res:
            return res
        res_body = res["ok"]
        item = res_body.get("task", None)
        if item is not None:
            workflow_payload_b64 = item.get("workflowPayload", None)
        else:
            workflow_payload_b64 = None
        if workflow_payload_b64 is not None:
            workflow_payload_bytes = base64.b64decode(workflow_payload_b64)
            workflow_payload = workflow_payload_bytes.decode("utf-8")
            print(f"DEBUG: ForgeRunnerClient._fetch_task: dump workflow payload...")
            print(workflow_payload)
            print(f"DEBUG: ForgeRunnerClient._fetch_task: dump workflow payload: done")
            # workflow = pythia.yaml.loads(workflow_payload)
            # print(f"DEBUG: ForgeRunnerClient._fetch_task: parsed workflow: {workflow}")
        if False:
            self._state.put("runner.fetch_task", item, key=self.runner.name)
            print(f"DEBUG: ForgeRunnerClient._fetch_task: put item = {item}")
        return {"ok": item}

    def _update_task(self, task_item: dict) -> Result:
        t0 = f"{datetime.utcnow().isoformat()}Z"
        t1 = f"{datetime.utcnow().isoformat()}Z"
        req_body = {
            "state": {
                "id": task_item["id"],
                "result": "RESULT_SUCCESS",
                "started_at": t0,
                "stopped_at": t1,
            },
        }
        res = self._http_post("UpdateTask", req_body)
        if "err" in res:
            return res
        res_body = res["ok"]
        item = res_body
        if False:
            self._state.put("runner.update_task", item, key=self.runner.name)
            print(f"DEBUG: ForgeRunnerClient._update_task: put item = {item}")
        return {"ok": item}

# if False:
if __name__ == "__main__":
    data = """
"on": [push]
jobs:
    test:
        name: test
        runs-on: autopythia
        steps:
            - name: hi
              run: echo hi
            - name: hello world
              run: echo hello world
"""
    import pythia.yaml
    import sys
    print(pythia.yaml.loads(data))
    sys.exit(0)

if __name__ == "__main__":
    client = ForgeRunnerClient.default()
    client._register()
    client._declare()
    if True:
    # if False:
        task_item = {
            # "id": "2373834",
            "id": "2383452",
        }
    task_item = client._fetch_task()["ok"]
    if task_item:
        client._update_task(task_item)
