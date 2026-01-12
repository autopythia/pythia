from typing import Any, Optional
from dataclasses import dataclass
import json
import subprocess
import urllib.parse
import urllib.request

from pythia.types import Result

@dataclass
class CurlHTTP:
    @staticmethod
    def http_get(url: str, headers: Optional[dict], params: Optional[dict]) -> Result:
        req_url = url
        if params:
            req_params = urllib.parse.urlencode(params)
            req_url = f"{req_url}?{req_params}"
        cmd = ["curl"]
        cmd.append(req_url)
        cmd.append("-L")
        cmd.append("-X")
        cmd.append("GET")
        if headers:
            for header_key, header_value in headers.items():
                cmd.append("-H")
                cmd.append(f"{header_key}: {header_value}")
        cmd.append("-s")
        cmd.append("-w")
        cmd.append("%{stderr}%{http_code}")
        print(f"DEBUG: CurlAPIClientWorker._http_get: cmd = {cmd}")
        # print(f"DEBUG: CurlAPIClientWorker._http_get: err = {repr(err)}")
        # res.t0 = datetime.utcnow().isoformat()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            text=True,
            encoding="utf-8",
        )
        out, err = proc.communicate()
        # print(f"DEBUG: CurlAPIClientWorker._http_get: out = {repr(out)}")
        # print(f"DEBUG: CurlAPIClientWorker._http_get: err = {repr(err)}")
        # res.t1 = datetime.utcnow().isoformat()
        res_status = -1
        try:
            res_status = int(err)
        except ValueError:
            pass
        if not (res_status >= 200 and res_status < 300):
            return {"err": {
                "exc_type": type(e).__name__,
                "exc_str": f"{e}",
                "status": res_status,
            }}
        res_data = out.strip()
        res_content = json.loads(res_data)
        return {"ok": res_content}

    @staticmethod
    def http_post(url: str, headers: Optional[dict], params: Optional[dict], content: Any) -> Result:
        req_url = url
        if params:
            req_params = urllib.parse.urlencode(params)
            req_url = f"{req_url}?{req_params}"
        cmd = ["curl"]
        cmd.append(req_url)
        cmd.append("-L")
        cmd.append("-X")
        cmd.append("POST")
        if headers:
            for header_key, header_value in headers.items():
                cmd.append("-H")
                cmd.append(f"{header_key}: {header_value}")
        cmd.append("-d")
        cmd.append(json.dumps(content))
        cmd.append("-s")
        cmd.append("-w")
        cmd.append("%{stderr}%{http_code}")
        # res.t0 = datetime.utcnow().isoformat()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            text=True,
            encoding="utf-8",
        )
        out, err = proc.communicate()
        # print(f"DEBUG: CurlAPIClientWorker._http_post: out = {repr(out)}")
        # print(f"DEBUG: CurlAPIClientWorker._http_post: err = {repr(err)}")
        # res.t1 = datetime.utcnow().isoformat()
        res_status = -1
        try:
            res_status = int(err)
        except ValueError:
            pass
        if not (res_status >= 200 and res_status < 300):
            return {"err": {
                "exc_type": type(e).__name__,
                "exc_str": f"{e}",
                "status": res_status,
            }}
        res_data = out.strip()
        res_content = json.loads(res_data)
        return {"ok": res_content}

@dataclass
class UrllibHTTP:
    @staticmethod
    def http_get(url: str, headers: Optional[dict], params: Optional[dict]) -> Result:
        req_url = url
        if params:
            req_params = urllib.parse.urlencode(params)
            req_url = f"{req_url}?{req_params}"
        hreq = urllib.request.Request(
            req_url,
            headers=headers,
        )
        res_status = -1
        try:
            with urllib.request.urlopen(hreq) as hres:
                res_status = hres.status
                res_data = hres.read()
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
        ) as e:
            return {"err": {
                "exc_type": type(e).__name__,
                "exc_str": f"{e}",
                "status": res_status,
            }}
        res_content = json.loads(res_data.decode("utf-8"))
        return {"ok": res_content}

    @staticmethod
    def http_post(url: str, headers: Optional[dict], params: Optional[dict], content: Any) -> Result:
        req_url = url
        if params:
            req_params = urllib.parse.urlencode(params)
            req_url = f"{req_url}?{req_params}"
        data = json.dumps(content).encode("utf-8")
        hreq = urllib.request.Request(
            req_url,
            headers=headers,
            data=data,
        )
        res_status = -1
        try:
            with urllib.request.urlopen(hreq) as hres:
                res_status = hres.status
                res_data = hres.read()
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
        ) as e:
            return {"err": {
                "exc_type": type(e).__name__,
                "exc_str": f"{e}",
                "status": res_status,
            }}
        res_content = json.loads(res_data.decode("utf-8"))
        return {"ok": res_content}

DefaultHTTP = UrllibHTTP

def http_get(url: str, headers: Optional[dict], params: Optional[dict]) -> Result:
    return DefaultHTTP.http_get(url, headers, params)

def http_post(url: str, headers: Optional[dict], params: Optional[dict], content: Any) -> Result:
    return DefaultHTTP.http_post(url, headers, params, content)
