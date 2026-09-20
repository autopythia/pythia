"""Non-authoritative selected-model provenance; never a source of live policy."""

import json
import os
from pathlib import Path
import tempfile
import re

from .model_catalog import parse_json_value, EndpointSpec
from ._config_file import read_config_bytes
from .save import SaveError


MAX_MANIFEST_BYTES = 262_144


def catalog_manifest_path(log_path):
    path = Path(log_path)
    return path.with_name(path.name + ".catalog.json")


def check_catalog_manifest(path, bindings, *, reselected=()):
    try:
        data = read_config_bytes(path, MAX_MANIFEST_BYTES)
    except FileNotFoundError:
        # Interaction logs written before catalog provenance remain resumable.
        return ()
    except (OSError, ValueError):
        raise SaveError("Could not read model-catalog provenance.") from None
    try:
        document = parse_json_value(data.decode("utf-8"))
        if (not isinstance(document, dict) or set(document) != {"version", "bindings"}
                or type(document["version"]) is not int or document["version"] != 2
                or not isinstance(document["bindings"], dict)
                or set(document["bindings"]) != set(bindings)):
            raise ValueError()
        for item in document["bindings"].values():
            expected = {
                "api", "selector", "model", "canonical", "source",
                "fingerprint", "endpoint",
            }
            if (not isinstance(item, dict) or set(item) != expected
                    or any(not isinstance(item[key], str) for key in ("api", "source", "fingerprint"))
                    or any(item[key] is not None and not isinstance(item[key], str)
                           for key in ("selector", "model", "canonical"))):
                raise ValueError()
            endpoint = EndpointSpec(**item["endpoint"])
            if endpoint.api != item["api"] or endpoint.model != item["model"]:
                raise ValueError()
            if (item["api"] not in {"codex", "responses", "messages", "chat-completions"}
                    or re.fullmatch(r"[0-9a-f]{64}", item["fingerprint"]) is None):
                raise ValueError()
    except (ValueError, UnicodeError, TypeError):
        raise SaveError("Invalid model-catalog provenance.") from None
    notices = []
    for key, binding in bindings.items():
        old, new = document["bindings"][key], binding.manifest_entry()
        changed_selection = (old["selector"] != new["selector"] or (
            binding.api_explicit and old["api"] != new["api"]
        ))
        if old["canonical"] is not None and (old["api"], old["canonical"]) != (new["api"], new["canonical"]):
            if key not in reselected or not changed_selection:
                raise SaveError(
                    "Saved model preset is unavailable or has changed identity; "
                    "explicitly select a different model/API to resume."
                )
        changes = {field for field in new["endpoint"]
                   if old["endpoint"].get(field) != new["endpoint"][field]}
        if changes - binding.endpoint_overrides and not (key in reselected and changed_selection):
            raise SaveError(
                "Saved endpoint changed; explicitly select endpoint URL/model/auth or a new model/API to resume."
            )
        if old["fingerprint"] != new["fingerprint"]:
            notices.append(f"Model binding changed for {key}; using current catalog/launch settings.")
    return tuple(notices)


def save_catalog_manifest(path, bindings):
    path = Path(path)
    temporary = None
    try:
        payload = json.dumps({"version": 2, "bindings": {
            key: binding.manifest_entry() for key, binding in bindings.items()
        }}, indent=2, ensure_ascii=False) + "\n"
        if len(payload.encode("utf-8")) > MAX_MANIFEST_BYTES:
            raise ValueError("Model-catalog provenance exceeds size limit.")
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".catalog-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except (OSError, ValueError, TypeError):
        raise SaveError("Could not save model-catalog provenance; no further effects may start.") from None
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
