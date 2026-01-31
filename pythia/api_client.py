from typing import Any, Optional
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio
import concurrent.futures
import json
import os
import subprocess
import threading
import time
import traceback
import urllib.request

from pythia.api_registry import APIEndpoint, APIModel, APIRegistry
from pythia.utils import merge_dicts

@dataclass
class APIExceptionInfo:
    exc_type: str
    exc_str: str
    stack_trace: str

@dataclass
class APIChatCompletionResponse:
    status: int = None
    og_status: int = None
    payload: Any = None
    except_: Any = None
    t0: str = None
    t1: str = None
    debug_messages: Any = None
    debug_sampling_params: Any = None
    debug_tools: Any = None
    debug_request: Any = None
    debug_response: Any = None

    def result(self) -> Any:
        return self.payload

    def message(self) -> Optional[dict]:
        return self.payload["choices"][0]["message"]

    def content(self) -> Any:
        return self.result().get("content", None)

@dataclass
class _APIChatCompletionWorkItem:
    model: APIModel
    messages: Any
    sampling_params: Any
    tools: Any
    debug: Optional[bool]
    res: Any = None
    exc: Any = None

    def _prepare_deprecated(self):
        if isinstance(self.messages, str):
            self.messages = [{"content": self.messages, "role": "user"}]
        elif isinstance(self.messages, APIMessage):
            self.messages = [self.messages]
        if isinstance(self.messages, list):
            for message_idx in range(len(self.messages)):
                if isinstance(self.messages[message_idx], APIMessage):
                    self.messages[message_idx] = self.messages[message_idx].to_dict()
                elif isinstance(self.messages[message_idx], dict):
                    pass
                else:
                    raise NotImplementedError
        else:
            raise NotImplementedError

    def _prepare(self):
        if isinstance(self.messages, str):
            self.messages = [{"content": self.messages, "role": "user"}]
        elif isinstance(self.messages, dict):
            self.messages = [self.messages]
        elif isinstance(self.messages, list):
            pass
        else:
            raise NotImplementedError

    def _finalize(self):
        # FIXME(20250824): res is None case.
        self.res.og_status = self.res.status
        if self.exc is not None:
            print(f"DEBUG: _APIChatCompletionWorkItem._finalize: exc = {self.exc}")
            self.res.status = 500
            self.res.except_ = self.exc
            return self.res
        return self.res

@dataclass
class APIMessageResult:
    status: int = None
    og_status: int = None
    payload: Any = None
    except_: Any = None
    t0: str = None
    t1: str = None

    def result(self) -> Any:
        return self.payload

    def message(self) -> Optional[dict]:
        return {
            "role": self.payload["role"],
            "content": self.payload["content"],
        }

    def content(self) -> Any:
        return self.message()["content"]

@dataclass
class _APIMessageWorkItem:
    model: APIModel
    messages: Any
    sampling_params: Any
    tools: Any
    api_version: Optional[str]
    # debug: Optional[bool]
    res: Any = None
    exc: Any = None

    def _prepare(self):
        pass

    def _finalize(self):
        # FIXME(20250824): res is None case.
        if self.res is None:
            print(f"DEBUG: _APIMessageWorkItem._finalize: res is None")
            self.res = APIMessageResult()
            self.res.status = 500
            return self.res
        self.res.og_status = self.res.status
        if self.exc is not None:
            print(f"DEBUG: _APIMessageWorkItem._finalize: exc = {self.exc}")
            self.res.status = 500
            self.res.except_ = self.exc
            return self.res
        return self.res

@dataclass
class APIClientModelEndpoint:
    model: APIModel
    endpoint: APIEndpoint = None
    prefer_protocol: Optional[str] = None
    max_new_tokens: Optional[int] = None
    throttle_delay: Optional[int] = None
    throttle_concurrency: Optional[int] = None

    _user_agent: str = "fastapi"
    _endpoint_url: str = None
    _endpoint_headers: Any = None

    def __post_init__(self) -> None:
        if self.endpoint is None:
            self.endpoint = self.model.endpoint
        if (
            self.throttle_delay is None and
            self.model.throttle_rps is not None
        ):
            self.throttle_delay = 1.0 / self.model.throttle_rps
        if (
            self.throttle_delay is None and
            self.endpoint.throttle_rps is not None
        ):
            self.throttle_delay = 1.0 / self.endpoint.throttle_rps
        if (
            self.throttle_concurrency is None and
            self.model.throttle_concurrency is not None
        ):
            self.throttle_concurrency = self.model.throttle_concurrency
        if (
            self.throttle_concurrency is None and
            self.endpoint.throttle_concurrency is not None
        ):
            self.throttle_concurrency = self.endpoint.throttle_concurrency
        protocol, api_url = self.protocol_api_url()
        if protocol in (
            "deepseek",
            "openai",
        ):
            # TODO: proper urllib formatting.
            if protocol == "deepseek":
                self._endpoint_url = "{}/chat/completions".format(api_url)
            elif protocol == "openai":
                if self.model.model_path.startswith("deepinfra/"):
                    self._endpoint_url = "{}/v1/openai/chat/completions".format(api_url)
                elif self.model.model_path.startswith("z-ai/"):
                    self._endpoint_url = "{}/api/paas/v4/chat/completions".format(api_url)
                else:
                    self._endpoint_url = "{}/v1/chat/completions".format(api_url)
            else:
                raise NotImplementedError
            self._endpoint_headers = {
                "User-Agent": self._user_agent,
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            if self.endpoint.api_key is not None:
                self._endpoint_headers["Authorization"] = (
                    "Bearer {}".format(self.endpoint.api_key)
                )
        elif protocol == "anthropic":
            # TODO: proper urllib formatting.
            self._endpoint_url = "{}/v1/messages".format(api_url)
            self._endpoint_headers = {
                "User-Agent": self._user_agent,
                "Anthropic-Version": "2023-06-01",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            if self.endpoint.api_key is not None:
                self._endpoint_headers["X-API-Key"] = (
                    "{}".format(self.endpoint.api_key)
                )
        elif protocol == "gemini":
            self._endpoint_url = "{}/v1beta/{}:generateContent".format(
                api_url,
                self.model.endpoint_model_path,
            )
            self._endpoint_headers = {
                "User-Agent": self._user_agent,
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            if self.endpoint.api_key is not None:
                self._endpoint_headers["x-goog-api-key"] = (
                    "{}".format(self.endpoint.api_key)
                )
        else:
            raise NotImplementedError

    def protocol_api_url(self) -> str:
        protocol = None
        api_url = None
        if self.model.endpoint_protocol is not None:
            protocol = self.model.endpoint_protocol
            api_url = self.model.endpoint_api_url
        if protocol is None and self.prefer_protocol is not None:
            protocol = self.prefer_protocol
            if protocol == self.endpoint.protocol:
                api_url = self.endpoint.api_url
            elif (
                self.endpoint.protocol_api_urls is not None and
                protocol in self.endpoint.protocol_api_urls
            ):
                api_url = self.endpoint.protocol_api_urls[protocol]
            if api_url is None:
                protocol = None
        if protocol is None:
            protocol = self.endpoint.protocol
            api_url = self.endpoint.api_url
        assert protocol is not None
        assert api_url is not None
        return protocol, api_url

    def chat_completion(
        self,
        messages: list[dict[str, Any]] = None,
        sampling_params: dict[str, Any] = None,
        tools: Optional[list] = None,
        debug: Optional[bool] = None,
        res: Any = None,
    ):
        protocol, api_url = self.protocol_api_url()
        og_messages = deepcopy(messages)
        og_sampling_params = deepcopy(sampling_params)
        if sampling_params is None:
            sampling_params = dict()
        system_prompt = None
        if protocol in ("anthropic", "gemini"):
            if messages and messages[0]["role"] == "system":
                system_prompt = messages[0]["content"]
                messages = messages[1:]
        if protocol in (
            "anthropic",
            "deepseek",
            "openai",
        ):
            req_body = dict()
            req_body["model"] = self.model.endpoint_model_path
            if system_prompt is not None:
                req_body["system"] = system_prompt
            if (
                protocol == "anthropic" and
                messages is not None
            ):
                newmessages = []
                for message in messages:
                    role = message["role"]
                    if role == "assistant":
                        newcontent = []
                        if "content" in message:
                            newcontent.append({
                                "type": "text",
                                "text": message["content"],
                            })
                        # NB(peter): reasoning block must be the last block
                        # before tool call blocks.
                        if "reasoning_content" in message:
                            reasoning = message["reasoning_content"]
                            newblock = {
                                "type": "thinking",
                                "thinking": reasoning,
                            }
                            if "reasoning_id" in message:
                                newblock["signature"] = message["reasoning_id"]
                            newcontent.append(newblock)
                        if "tool_calls" in message:
                            for tool_call in (message["tool_calls"] or []):
                                tool_call_id = tool_call["id"]
                                fun_call = tool_call["function"]
                                tool_call_name = fun_call["name"]
                                tool_call_args = fun_call["arguments"]
                                newcontent.append({
                                    "type": "tool_use",
                                    "id": tool_call_id,
                                    "name": tool_call_name,
                                    "input": json.loads(tool_call_args),
                                })
                        newmessage = {
                            "role": role,
                            "content": newcontent,
                        }
                    elif role == "tool":
                        newmessage = {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": message["tool_call_id"],
                                    "content": message["content"],
                                },
                            ],
                        }
                    elif role == "user":
                        newmessage = {
                            "role": role,
                            "content": message["content"],
                        }
                    else:
                        raise ValueError(f"not implemented: anthropic support for role: {repr(role)}")
                    newmessages.append(newmessage)
                req_body["messages"] = newmessages
            elif messages is not None:
                req_body["messages"] = messages
            if (
                protocol == "anthropic" and
                tools is not None
            ):
                newtools = []
                for tool in tools:
                    if tool["type"] != "function":
                        raise ValueError(f"not implemented: anthropic version of tool: {tool}")
                    fun = tool["function"]
                    newtool = {
                        "name": fun["name"],
                        "description": fun["description"],
                        "input_schema": fun["parameters"],
                    }
                    newtools.append(newtool)
                req_body["tools"] = newtools
            elif tools is not None:
                req_body["tools"] = tools
            req_body["stream"] = False
            if protocol == "deepseek":
                max_completion_tokens = sampling_params.pop("max_completion_tokens", None)
                if max_completion_tokens is not None:
                    assert "max_new_tokens" not in sampling_params
                    sampling_params["max_new_tokens"] = max_completion_tokens
                max_tokens = sampling_params.pop("max_tokens", None)
                if max_tokens is not None:
                    assert "max_new_tokens" not in sampling_params
                    sampling_params["max_new_tokens"] = max_tokens
            if protocol in ("anthropic", "openai"):
                max_new_tokens = sampling_params.pop("max_new_tokens", None)
                if max_new_tokens is not None:
                    assert "max_tokens" not in sampling_params
                    sampling_params["max_tokens"] = max_new_tokens
                max_completion_tokens = sampling_params.pop("max_completion_tokens", None)
                if max_completion_tokens is not None:
                    assert "max_tokens" not in sampling_params
                    sampling_params["max_tokens"] = max_completion_tokens
                if (
                    self.model.model_path.startswith("openai/o3") or
                    self.model.model_path.startswith("openai/o4") or
                    self.model.model_path.startswith("openai/gpt-5")
                ):
                    max_tokens = sampling_params.pop("max_tokens", None)
                    if max_tokens is not None:
                        assert "max_completion_tokens" not in sampling_params
                        sampling_params["max_completion_tokens"] = max_tokens
                    sampling_params.pop("stop", None)
            if protocol == "anthropic":
                stop = sampling_params.pop("stop", None)
                if stop is None:
                    pass
                elif isinstance(stop, str):
                    sampling_params["stop_sequences"] = [stop]
                elif isinstance(stop, list):
                    sampling_params["stop_sequences"] = stop
                else:
                    raise ValueError(f"invalid conversion to stop_sequences: {stop}")
                sampling_params.pop("frequency_penalty", None)
                sampling_params.pop("presence_penalty", None)
                sampling_params.pop("logprobs", None)
                sampling_params.pop("top_logprobs", None)
        elif protocol == "gemini":
            contents = None
            for message_idx, message in enumerate(messages or []):
                role = message["role"]
                if role == "assistant":
                    role = "model"
                    parts = []
                    if "reasoning_id" in message:
                        part = {
                            "thought": True,
                            "thoughtSignature": message["reasoning_id"],
                        }
                        if "reasoning_content" in message:
                            part["text"] = message["reasoning_content"]
                        parts.append(part)
                    if "content" in message:
                        msg_content = message["content"]
                        if isinstance(msg_content, str):
                            part = {
                                "text": msg_content,
                            }
                        else:
                            raise ValueError(f"not implemented: gemini support for assistant message: {message}")
                        parts.append(part)
                    if "tool_calls" in message:
                        for tool_call in message["tool_calls"]:
                            if tool_call["type"] != "function":
                                raise ValueError(f"not implemented: gemini support for non-function tool call: {tool_call}")
                            tool_call_id = tool_call["id"]
                            fun_call = tool_call["function"]
                            tool_call_name = fun_call["name"]
                            tool_call_args = json.loads(fun_call["arguments"])
                            parts.append({
                                "functionCall": {
                                    "id": tool_call_id,
                                    "name": tool_call_name,
                                    "args": tool_call_args,
                                },
                            })
                elif role == "tool":
                    role = "user"
                    parts = []
                    tool_call_id = message["tool_call_id"]
                    tool_call_name = None
                    for prev_message_idx in reversed(range(message_idx)):
                        prev_message = messages[prev_message_idx]
                        if "tool_calls" in prev_message:
                            for tool_call in prev_message["tool_calls"]:
                                if tool_call["type"] != "function":
                                    raise ValueError(f"not implemented: gemini support for non-function tool call: {tool_call}")
                                fun_call = tool_call["function"]
                                if tool_call_id == tool_call["id"]:
                                    tool_call_name = fun_call["name"]
                                    break
                            if tool_call_name is not None:
                                break
                    tool_call_result = json.loads(message["content"])
                    parts.append({
                        "functionResponse": {
                            "id": tool_call_id,
                            "name": tool_call_name,
                            "response": tool_call_result,
                        },
                    })
                elif role == "user":
                    parts = []
                    msg_content = message["content"]
                    if isinstance(msg_content, str):
                        parts.append({
                            "text": msg_content,
                        })
                    else:
                        raise ValueError(f"not implemented: gemini support for user message: {message}")
                else:
                    raise ValueError(f"not implemented: gemini support for role: {repr(role)}")
                content = {
                    "role": role,
                    "parts": parts,
                }
                if contents is None:
                    contents = []
                contents.append(content)
            newtools = None
            for tool in (tools or []):
                newtool = tool
                if newtools is None:
                    newtools = []
                newtools.append(newtool)
            req_body = dict()
            if system_prompt is not None:
                req_body["system_instruction"] = {
                    "parts": [
                        {
                            "text": system_prompt,
                        },
                    ],
                }
            if contents is not None:
                req_body["contents"] = contents
            if newtools is not None:
                req_body["tools"] = [
                    {
                        "functionDeclarations": newtools,
                    }
                ]
            # TODO: tool config.
            # req_body["toolConfig"] = _
            max_new_tokens = sampling_params.pop("max_new_tokens", None)
            if max_new_tokens is not None:
                assert "max_tokens" not in sampling_params
                sampling_params["max_tokens"] = max_new_tokens
            max_completion_tokens = sampling_params.pop("max_completion_tokens", None)
            if max_completion_tokens is not None:
                assert "max_tokens" not in sampling_params
                sampling_params["max_tokens"] = max_completion_tokens
            def _convert_stop(v):
                if v is None:
                    return []
                if isinstance(v, str):
                    return [v]
                if isinstance(v, list):
                    return v
                raise ValueError
            sampling_key_map = [
                ("stop", "stopSequences", _convert_stop),
                ("temperature",),
                ("top_p", "topP"),
                ("top_k", "topK"),
                ("n", "candidateCount"),
                ("max_tokens", "maxOutputTokens"),
                ("logprobs", "responseLogprobs"),
                ("top_logprobs", "logprobs"),
                ("seed",),
                # ("presence_penalty", "presencePenalty"),
                # ("frequency_penalty", "frequencyPenalty"),
                ("presence_penalty", None),
                ("frequency_penalty", None),
                ("stream", None),
            ]
            generation_cfg = dict()
            for k in sampling_key_map:
                if len(k) == 1:
                    ok, = k
                    nk = ok
                    convert = None
                elif len(k) == 2:
                    ok, nk = k
                    convert = None
                elif len(k) == 3:
                    ok, nk, convert = k
                else:
                    raise NotImplementedError
                if ok in sampling_params:
                    v = sampling_params.pop(ok)
                    if convert is not None:
                        v = convert(v)
                    if nk is not None:
                        generation_cfg[nk] = v
            if sampling_params:
                raise ValueError(f"unsupported sampling params: {sampling_params.keys()}")
            sampling_params = {
                "generationConfig": generation_cfg,
            }
        else:
            raise NotImplementedError
        if (
            self.model.model_path.startswith("openai/o3") or
            self.model.model_path.startswith("openai/o4")
        ):
            sampling_params.pop("temperature", None)
            sampling_params.pop("top_p", None)
        res.sampling_params = sampling_params
        req_body |= sampling_params
        # req_body = merge_dicts(req_body, sampling_params)
        if self.model.endpoint_extra_params is not None:
            req_body = merge_dicts(req_body, deepcopy(self.model.endpoint_extra_params))
        if protocol == "anthropic":
            if (
                "max_tokens" in req_body and
                "thinking" in req_body and
                "budget_tokens" in req_body["thinking"]
            ):
                max_tokens = req_body["max_tokens"]
                thinking_tokens = req_body["thinking"]["budget_tokens"]
                # print(f"DEBUG: APIClientModelEndpoint.chat_completion: anthropic: max tokens = {max_tokens} thinking tokens = {thinking_tokens}")
                req_body["thinking"]["budget_tokens"] = max(0, min(max_tokens - 1, thinking_tokens))
        print(f"DEBUG: APIClientModelEndpoint.chat_completion: req body = {req_body}")
        if debug:
            res.debug_messages = og_messages
            res.debug_sampling_params = og_sampling_params
            res.debug_tools = tools
            res.debug_request = req_body
        req_data = json.dumps(req_body).encode("utf-8")
        hreq = urllib.request.Request(
            self._endpoint_url,
            headers = self._endpoint_headers.copy(),
            data = req_data,
        )
        res.t0 = datetime.utcnow().isoformat()
        try:
            with urllib.request.urlopen(hreq) as hres:
                res_status = hres.status
                res_data = hres.read()
        except urllib.error.HTTPError as e:
            err_status = e.code
            res.t1 = datetime.utcnow().isoformat()
            res.status = err_status
            try:
                err_body = json.loads(e.read().decode("utf-8"))
                print(f"DEBUG: APIClientModelEndpoint.chat_completion: err body = {err_body}")
                res.payload = err_body
            except Exception:
                pass
            return res
        except urllib.error.URLError as e:
            err_status = e.code
            res.t1 = datetime.utcnow().isoformat()
            res.status = err_status
            return res
        res.t1 = datetime.utcnow().isoformat()
        res.status = res_status
        res_body = json.loads(res_data.decode("utf-8"))
        print(f"DEBUG: APIClientModelEndpoint.chat_completion: res body = {res_body}")
        if debug:
            res.debug_response = deepcopy(res_body)
        res.payload = res_body
        if protocol == "anthropic":
            res_type = res.payload.pop("type")
            if res_type != "message":
                raise ValueError(f"not implemented: anthropic response type = {repr(res_type)}")
            role = res.payload.pop("role")
            res_content = res.payload.pop("content")
            _res_stop_sequence = res.payload.pop("stop_sequence", None)
            res_stop_reason = res.payload.pop("stop_reason", None)
            if res_stop_reason is None:
                finish_reason = None
            elif res_stop_reason == "refusal":
                finish_reason = "content_filter"
            elif res_stop_reason in ("max_tokens", "pause_turn"):
                finish_reason = "length"
            elif res_stop_reason in ("end_turn", "stop_sequence"):
                finish_reason = "stop"
            elif res_stop_reason == "tool_use":
                finish_reason = "tool_calls"
            else:
                raise ValueError(f"not implemented: anthropic stop reason = {repr(res_stop_reason)}")
            res_usage = res.payload.pop("usage", None)
            res.payload["object"] = "chat.completion"
            res.payload["created"] = int(datetime.fromisoformat(res.t1).timestamp())
            reasoning_id = None
            reasoning = None
            text = None
            tool_calls = None
            for content in res_content:
                content_type = content["type"]
                if content_type == "thinking":
                    assert reasoning is None
                    reasoning = content["thinking"]
                    if "signature" in content:
                        assert reasoning_id is None
                        reasoning_id = content["signature"]
                elif content_type == "text":
                    assert text is None
                    text = content["text"]
                elif content_type == "tool_use":
                    tool_call_id = content["id"]
                    tool_call_name = content["name"]
                    tool_call = {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": tool_call_name,
                        },
                    }
                    if "input" in content:
                        tool_call["function"]["arguments"] = json.dumps(content["input"])
                    if tool_calls is None:
                        tool_calls = []
                    tool_calls.append(tool_call)
                else:
                    raise ValueError(f"not implemented: anthropic response content: {res_content}")
            message = {
                "role": role,
            }
            if reasoning_id is not None:
                message["reasoning_id"] = reasoning_id
            if reasoning is not None:
                message["reasoning_content"] = reasoning
            if text is not None:
                message["content"] = text
            if tool_calls is not None:
                message["tool_calls"] = tool_calls
            res.payload["choices"] = [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ]
            if res_usage is not None:
                usage = dict()
                total_tokens = 0
                if "input_tokens" in res_usage:
                    usage["prompt_tokens"] = res_usage["input_tokens"]
                    total_tokens += usage["prompt_tokens"]
                if "output_tokens" in res_usage:
                    usage["completion_tokens"] = res_usage["output_tokens"]
                    total_tokens += usage["completion_tokens"]
                usage["total_tokens"] = total_tokens
                res.payload["usage"] = usage
                service_tier = res_usage.get("service_tier", None)
                if service_tier is not None:
                    res.payload["service_tier"] = service_tier
        elif protocol == "gemini":
            def _convert_choice(v):
                content_dict = v["content"]
                role = content_dict["role"]
                if role == "model":
                    role = "assistant"
                reasoning_id = None
                reasoning = None
                text = None
                tool_calls = None
                for part in content_dict["parts"]:
                    if "thought" in part and part["thought"]:
                        if "thoughtSignature" in part:
                            assert reasoning_id is None
                            reasoning_id = part["thoughtSignature"]
                        if "text" in part:
                            assert reasoning is None
                            reasoning = part["text"]
                    elif "functionCall" in part:
                        fun_call = part["functionCall"]
                        tool_call_id = fun_call["id"]
                        tool_call_name = fun_call["name"]
                        tool_call = {
                            "id": tool_call_id,
                            "type": "function",
                            "function": {
                                "name": tool_call_name,
                            },
                        }
                        if "args" in fun_call:
                            tool_call["function"]["arguments"] = json.dumps(fun_call["args"])
                        if tool_calls is None:
                            tool_calls = []
                        tool_calls.append(tool_call)
                    elif "text" in part:
                        assert text is None
                        text = part["text"]
                    else:
                        raise NotImplementedError("not implemented: gemini response choice: {v}")
                message = {
                    "role": role,
                }
                if reasoning_id is not None:
                    message["reasoning_id"] = reasoning_id
                if reasoning is not None:
                    message["reasoning_content"] = reasoning
                if text is not None:
                    message["content"] = text
                if tool_calls is not None:
                    message["tool_calls"] = tool_calls
                finish_reason = v["finishReason"]
                if finish_reason is not None:
                    finish_reason = finish_reason.lower()
                index = v["index"]
                return {
                    "index": index,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            def _convert_choices(vs):
                return [_convert_choice(v) for v in vs]
            def _convert_usage(v):
                nv = dict()
                if "promptTokenCount" in v:
                    nv["prompt_tokens"] = v["promptTokenCount"]
                if "candidatesTokenCount" in v:
                    nv["completion_tokens"] = v["candidatesTokenCount"]
                if "totalTokenCount" in v:
                    nv["total_tokens"] = v["totalTokenCount"]
                if "thoughtsTokenCount" in v:
                    nv["completion_tokens_details"] = {
                        "reasoning_tokens": v["thoughtsTokenCount"],
                    }
                return nv
            response_key_map = [
                ("responseId", "id"),
                ("modelVersion", "model"),
                ("candidates", "choices", _convert_choices),
                ("usageMetadata", "usage", _convert_usage),
            ]
            payload = dict()
            payload["id"] = None
            payload["object"] = "chat.completion"
            payload["created"] = int(datetime.fromisoformat(res.t1).timestamp())
            for k in response_key_map:
                if len(k) == 1:
                    ok, = k
                    nk = ok
                    convert = None
                elif len(k) == 2:
                    ok, nk = k
                    convert = None
                elif len(k) == 3:
                    ok, nk, convert = k
                else:
                    raise ValueError
                if ok in res.payload:
                    v = res.payload.pop(ok)
                    if convert is not None:
                        v = convert(v)
                    payload[nk] = v
            res.payload = payload
        res.payload["model"] = self.model.model_path
        return res

    def message(
        self,
        messages: list[dict[str, Any]] = None,
        sampling_params: dict[str, Any] = None,
        tools: Optional[list] = None,
        api_version: Optional[str] = None,
        # debug: Optional[bool] = None,
        worker: Any = None,
        res: Any = None,
    ):
        # print(f"DEBUG: APIClientModelEndpoint.message: model    = {self.model.model_path}")
        # print(f"DEBUG: APIClientModelEndpoint.message: messages = {messages}")
        # print(f"DEBUG: APIClientModelEndpoint.message: sampling = {sampling_params}")
        protocol, api_url = self.protocol_api_url()
        # print(f"DEBUG: APIClientModelEndpoint.message: protocol = {protocol} api url = {api_url}")
        # print(f"DEBUG: APIClientModelEndpoint.message: req url  = {self._endpoint_url}", flush=True)
        req_headers = self._endpoint_headers.copy()
        if protocol == "anthropic":
            if api_version is not None:
                req_headers["Anthropic-Version"] = api_version
            else:
                req_headers["Anthropic-Version"] = "2023-06-01"
        if protocol == "anthropic":
            # print(f"DEBUG: APIClientModelEndpoint.message: anthropic? True")
            req_body = dict()
            system_prompt = None
            if messages and messages[0]["role"] == "system":
                system_prompt = messages[0]["content"]
                messages = messages[1:]
            if system_prompt is not None:
                req_body["system"] = system_prompt
            # print(f"DEBUG: APIClientModelEndpoint.message: system   = {repr(system_prompt)}")
            # print(f"DEBUG: APIClientModelEndpoint.message: messages = {messages}")
            req_body["messages"] = messages
            if tools is not None:
                req_body["tools"] = tools
            if sampling_params is not None:
                if self.model.default_sampling_params is not None:
                    sampling_params = (
                        self.model.default_sampling_params |
                        sampling_params
                    )
            else:
                sampling_params = self.model.default_sampling_params
            if sampling_params is not None:
                req_body |= sampling_params
            if "max_new_tokens" in req_body:
                req_body["max_tokens"] = req_body.pop("max_new_tokens")
            req_body["model"] = self.model.endpoint_model_path
        elif protocol in (
            "deepseek",
            "openai",
        ):
            req_body = dict()
            newmessages = []
            for message in messages:
                role = message["role"]
                content = None
                reasoning_content = None
                newmessage = dict()
                newmessage["role"] = role
                if "content" in message:
                    if isinstance(message["content"], str):
                        content = message["content"]
                    elif isinstance(message["content"], list):
                        for block in message["content"]:
                            if block["type"] == "text":
                                assert content is None
                                content = block["text"]
                            elif block["type"] == "thinking":
                                assert reasoning_content is None
                                reasoning_content = block["thinking"]
                            elif block["type"] == "tool_use":
                                raise NotImplementedError("'type': 'tool_use'")
                            elif block["type"] == "tool_result":
                                raise NotImplementedError("'type': 'tool_result'")
                            else:
                                raise NotImplementedError
                    else:
                        raise NotImplementedError
                if reasoning_content is not None:
                    newmessage["reasoning_content"] = reasoning_content
                if content is not None:
                    newmessage["content"] = content
                newmessages.append(newmessage)
            req_body["messages"] = newmessages
            if tools is not None:
                req_body["tools"] = tools
            if sampling_params is not None:
                req_body |= sampling_params
            if "max_new_tokens" in req_body:
                req_body["max_tokens"] = req_body.pop("max_new_tokens")
            if (
                "top_k" in req_body and
                isinstance(req_body["top_k"], int) and
                req_body["top_k"] <= 0
            ):
                req_body.pop("top_k")
            req_body["model"] = self.model.endpoint_model_path
        else:
            raise NotImplementedError
        if self.model.endpoint_extra_params is not None:
            req_body = merge_dicts(req_body, deepcopy(self.model.endpoint_extra_params))
        if protocol == "anthropic":
            if self.model.default_thinking:
                if "thinking" not in req_body:
                    req_body["thinking"] = {
                        "type": "enabled",
                        "budget_tokens": -1,
                    }
            if (
                "max_tokens" in req_body and
                "thinking" in req_body and
                "budget_tokens" in req_body["thinking"]
            ):
                max_tokens = req_body["max_tokens"]
                thinking_tokens = req_body["thinking"]["budget_tokens"]
                if thinking_tokens >= 0:
                    req_body["thinking"]["budget_tokens"] = max(0, min(max_tokens - 1, thinking_tokens))
                else:
                    req_body["thinking"]["budget_tokens"] = max(0, max_tokens - 1)
        # print(f"DEBUG: APIClientModelEndpoint.message: req body = {req_body}", flush=True)
        if False:
            req_data = json.dumps(req_body).encode("utf-8")
            hreq = urllib.request.Request(
                self._endpoint_url,
                headers=req_headers,
                data=req_data,
            )
            res.t0 = datetime.utcnow().isoformat()
            try:
                with urllib.request.urlopen(hreq) as hres:
                    res_status = hres.status
                    res_data = hres.read()
            except urllib.error.HTTPError as e:
                err_status = e.code
                res.t1 = datetime.utcnow().isoformat()
                res.status = err_status
                try:
                    err_body = json.loads(e.read().decode("utf-8"))
                    print(f"DEBUG: APIClientModelEndpoint.message: http err body = {err_body}", flush=True)
                    res.payload = err_body
                except Exception:
                    pass
                return res
            except urllib.error.URLError as e:
                err_status = e.code
                res.t1 = datetime.utcnow().isoformat()
                res.status = err_status
                try:
                    err_body = json.loads(e.read().decode("utf-8"))
                    print(f"DEBUG: APIClientModelEndpoint.message: url err body = {err_body}", flush=True)
                    res.payload = err_body
                except Exception:
                    pass
                return res
            res.t1 = datetime.utcnow().isoformat()
            res.status = res_status
            res_body = json.loads(res_data.decode("utf-8"))
        worker._http_post(
            self._endpoint_url,
            req_headers,
            req_body,
            res,
        )
        if not (res.status >= 200 and res.status < 300):
            return res
        res_body = res.payload
        # print(f"DEBUG: APIClientModelEndpoint.message: res body = {res_body}", flush=True)
        if protocol == "anthropic":
            res.payload = res_body
            res.payload["model"] = self.model.model_path
        elif protocol in (
            "deepseek",
            "openai",
        ):
            new_payload = dict()
            new_payload["id"] = res_body["id"]
            new_payload["type"] = "message"
            res_choices = res_body.get("choices", None)
            res_usage = res_body.get("usage", None)
            if res_choices is not None:
                role = None
                newcontent = []
                stop_reason = None
                for choice_item in res_choices:
                    role = choice_item["message"]["role"]
                    content = choice_item["message"].get("content", None)
                    reasoning_content = choice_item["message"].get("reasoning_content", None)
                    tool_calls = choice_item["message"].get("tool_calls", None)
                    if content is not None:
                        thinking_start = None
                        if (
                            # FIXME: workaround for no reasoning parser
                            # (deepinfra deepseek-R1).
                            reasoning_content is None and
                            content.startswith("<think>")
                        ) or (
                            # FIXME: hack for sglang without reasoning parser.
                            self.endpoint.name == "__local__" and
                            content.startswith("<think>")
                        ):
                            thinking_start = len("<think>")
                            if thinking_start < len(content) and content[thinking_start] == "\n":
                                thinking_start += 1
                        # FIXME: hack for sglang without reasoning parser.
                        elif False:
                        # elif self.endpoint.name == "__local__":
                            if content.startswith("<think>"):
                                thinking_start = len("<think>")
                                if thinking_start < len(content) and content[thinking_start] == "\n":
                                    thinking_start += 1
                            else:
                                thinking_start = 0
                        if thinking_start is not None:
                            if thinking_start < len(content) and content[thinking_start] == "\n":
                                thinking_start += 1
                            thinking_end = content.find("</think>")
                            if thinking_end >= 0:
                                reasoning_content = content[thinking_start:thinking_end]
                                thinking_end += len("</think>")
                                for _ in range(2):
                                    if thinking_end < len(content) and content[thinking_end] == "\n":
                                        thinking_end += 1
                                content = content[thinking_end:]
                        new_block = {
                            "type": "text",
                            "text": content,
                        }
                        newcontent.append(new_block)
                    if reasoning_content is not None:
                        new_block = {
                            "type": "thinking",
                            "thinking": reasoning_content,
                        }
                        reasoning_id = choice_item["message"].get("reasoning_id", None)
                        if reasoning_id is not None:
                            new_block["signature"] = reasoning_id
                        newcontent.append(new_block)
                    if tool_calls:
                        raise NotImplementedError
                    finish_reason = choice_item.get("finish_reason", None)
                    if finish_reason == "stop":
                        stop_reason = "end_turn"
                    elif finish_reason == "length":
                        stop_reason = "max_tokens"
                    elif finish_reason == "content_filter":
                        stop_reason = "refusal"
                    break
                new_payload["role"] = role
                new_payload["model"] = self.model.model_path
                new_payload["content"] = newcontent
                new_payload["stop_reason"] = stop_reason
                new_payload["stop_sequence"] = None
                usage = dict()
                if res_usage is not None:
                    total_tokens = res_usage.get("total_tokens", None)
                    prompt_tokens = res_usage.get("prompt_tokens", None)
                    completion_tokens = res_usage.get("completion_tokens", None)
                    input_output_tokens = 0
                    if total_tokens is not None:
                        input_output_tokens = max(input_output_tokens, total_tokens)
                    if prompt_tokens is not None:
                        input_output_tokens = max(input_output_tokens, prompt_tokens)
                    if completion_tokens is not None:
                        input_output_tokens = max(input_output_tokens, completion_tokens)
                    if prompt_tokens is not None and completion_tokens is not None:
                        input_output_tokens = max(input_output_tokens, prompt_tokens + completion_tokens)
                    input_tokens = 0
                    output_tokens = 0
                    if prompt_tokens is not None:
                        input_tokens = prompt_tokens
                    if completion_tokens is not None:
                        output_tokens = completion_tokens
                    if input_tokens + output_tokens < input_output_tokens:
                        output_tokens = max(output_tokens, input_output_tokens - input_tokens)
                    usage["input_tokens"] = input_tokens
                    usage["output_tokens"] = output_tokens
                new_payload["usage"] = usage
            if "model" not in new_payload:
                new_payload["model"] = self.model.model_path
            res.payload = new_payload
        else:
            raise NotImplementedError
        # print(f"DEBUG: APIClientModelEndpoint.message: final res body = {res.payload}", flush=True)
        return res

class async_nullcontext:
    async def __aenter__(self):
        pass

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

@dataclass
class APIJournal:
    root: str = "data/_api_journal"
    enable: bool = True

    _log_path: str = None
    _lock: Any = None

    def __post_init__(self):
        if not self.enable:
            return
        env_key = "PYTHIA_API_JOURNAL_ROOT"
        if self.root is None:
            self.root = os.environ.get(env_key, None)
        if self.root is None:
            print(f"DEBUG: APIJournal: warning: missing env var {repr(env_key)}")
        else:
            if self._log_path is None:
                self._log_path = os.path.join(self.root, "_log.jsonl")
            if self._lock is None:
                self._lock = asyncio.Lock()

    async def get(self, model, messages=None, sampling_params=None, tools=None):
        raise NotImplementedError

    async def put(self, model, messages=None, sampling_params=None, tools=None, res=None):
        if not self._log_path:
            return
        item = {
            "model_path": model.model_path,
        }
        if messages is not None:
            item["messages"] = messages
        if sampling_params is not None:
            item["sampling_params"] = sampling_params
        if tools is not None:
            item["tools"] = tools
        item |= {
            "status": res.status,
            "og_status": res.og_status,
            "result": res.payload,
            "except_": res.except_,
            "t0": res.t0,
            "t1": res.t1,
        }
        async with self._lock:
            with open(self._log_path, "a") as file:
                print(json.dumps(item), file=file)

@dataclass
class APIClientEndpointState:
    _lock: Any = None
    _next_t0: Any = None
    _max_concurrency: Any = None

    def __post_init__(self):
        if self._lock is None:
            self._lock = threading.Lock()
        if self._max_concurrency is None:
            self._max_concurrency = async_nullcontext()

@dataclass
class APIClientWorker:
    max_concurrency: int
    _poolexec: Any = None

    def __post_init__(self) -> None:
        if self._poolexec is None:
            self._poolexec = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_concurrency,
            )

@dataclass
class CurlAPIClientWorker(APIClientWorker):
    def _http_post(self, url: str, headers: dict[str, str], payload, res):
        cmd = ["curl"]
        cmd.append("-L")
        cmd.append(url)
        cmd.append("-X")
        cmd.append("POST")
        for header_key, header_value in headers.items():
            cmd.append("-H")
            cmd.append(f"{header_key}: {header_value}")
        cmd.append("-d")
        cmd.append(json.dumps(payload))
        cmd.append("-s")
        cmd.append("-w")
        cmd.append("%{stderr}%{http_code}")
        res.t0 = datetime.utcnow().isoformat()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            text=True,
            encoding="utf-8",
        )
        out, err = proc.communicate()
        print(f"DEBUG: CurlAPIClientWorker._http_post: out = {repr(out)}")
        print(f"DEBUG: CurlAPIClientWorker._http_post: err = {repr(err)}")
        res.t1 = datetime.utcnow().isoformat()
        try:
            res.status = int(err)
        except ValueError:
            res.status = -1
        res_data = out.strip()
        res_body = json.loads(res_data)
        res.payload = res_body

@dataclass
class UrllibAPIClientWorker(APIClientWorker):
    def _http_post(self, url: str, headers: dict[str, str], payload, res):
        req_headers = headers
        req_body = payload
        req_data = json.dumps(req_body).encode("utf-8")
        hreq = urllib.request.Request(
            url,
            headers = req_headers,
            data = req_data,
        )
        res.t0 = datetime.utcnow().isoformat()
        try:
            with urllib.request.urlopen(hreq) as hres:
                res_status = hres.status
                res_data = hres.read()
        except (urllib.error.HTTPError, urllib.error.URLError) as e:
            err_status = e.code
            res.t1 = datetime.utcnow().isoformat()
            res.status = err_status
            try:
                err_body = json.loads(e.read().decode("utf-8"))
                print(f"DEBUG: UrllibAPIClientWorker._http_post: err body = {err_body}")
                res.payload = err_body
            except Exception:
                pass
            return
        except ValueError as e:
            res.t1 = datetime.utcnow().isoformat()
            res.status = -1
            return
        res.t1 = datetime.utcnow().isoformat()
        res.status = res_status
        res_body = json.loads(res_data.decode("utf-8"))
        res.payload = res_body

def _chat_completion(work_item, endpoint_state: APIClientEndpointState):
    endpoint = APIClientModelEndpoint(work_item.model)
    t = datetime.utcnow()
    t0 = None
    if endpoint.throttle_delay is not None:
        delta_t = timedelta(seconds=endpoint.throttle_delay)
        with endpoint_state._lock:
            t0 = endpoint_state._next_t0
            if t0 is not None:
                endpoint_state._next_t0 = max(t0, t) + delta_t
            else:
                endpoint_state._next_t0 = t + delta_t
        while t0 is not None and t0 > t:
            time.sleep((t0 - t).total_seconds())
            t = datetime.utcnow()
    print(f"DEBUG: APIClient.chat_completion: t = {t.isoformat()} t0 = {t0.isoformat() if t0 is not None else None}")
    work_item.res = APIChatCompletionResponse()
    endpoint.chat_completion(
        messages=work_item.messages,
        sampling_params=work_item.sampling_params,
        tools=work_item.tools,
        debug=work_item.debug,
        res=work_item.res,
    )
    return work_item

def _try_chat_completion(work_item, endpoint_state: APIClientEndpointState):
    try:
        _chat_completion(work_item, endpoint_state)
    # TODO: exc reporting.
    except Exception as e:
        # print(f"DEBUG: _try_chat_completion: except = {e}")
        work_item.exc = APIExceptionInfo(
            exc_type=f"{type(e).__name__}",
            exc_str=str(e),
            stack_trace=traceback.format_exc(),
        )
        print(f"DEBUG: _try_chat_completion: except = {work_item.exc}")
    return work_item

def _message(work_item, endpoint_state: APIClientEndpointState, worker: APIClientWorker):
    endpoint = APIClientModelEndpoint(work_item.model, prefer_protocol="anthropic")
    t = datetime.utcnow()
    t0 = None
    if endpoint.throttle_delay is not None:
        delta_t = timedelta(seconds=endpoint.throttle_delay)
        with endpoint_state._lock:
            t0 = endpoint_state._next_t0
            if t0 is not None:
                endpoint_state._next_t0 = max(t0, t) + delta_t
            else:
                endpoint_state._next_t0 = t + delta_t
        while t0 is not None and t0 > t:
            time.sleep((t0 - t).total_seconds())
            t = datetime.utcnow()
    # print(f"DEBUG: APIClient.message: t = {t.isoformat()} t0 = {t0.isoformat() if t0 is not None else None}")
    work_item.res = APIMessageResult()
    endpoint.message(
        messages=work_item.messages,
        sampling_params=work_item.sampling_params,
        tools=work_item.tools,
        api_version=work_item.api_version,
        # debug=work_item.debug,
        worker=worker,
        res=work_item.res,
    )
    return work_item

def _try_message(work_item, endpoint_state: APIClientEndpointState, worker: APIClientWorker):
    try:
        _message(work_item, endpoint_state, worker)
    # TODO: exc reporting.
    except Exception as e:
        # print(f"DEBUG: _try_message: except = {e}")
        work_item.exc = APIExceptionInfo(
            exc_type=f"{type(e).__name__}",
            exc_str=str(e),
            stack_trace=traceback.format_exc(),
        )
        # print(f"DEBUG: _try_message: except = {work_item.exc}")
        print(f"DEBUG: _try_message: except: {work_item.exc.exc_type} {work_item.exc.exc_str}")
        print(work_item.exc.stack_trace)
    return work_item

@dataclass
class APIClient:
    registry: APIRegistry
    max_concurrency: int = 192
    timeout: int = 1800

    _journal: APIJournal = None
    _worker: APIClientWorker = None
    _endpoint_state: dict[str, APIClientEndpointState] = None
    _model_endpoint_state: dict[str, APIClientEndpointState] = None

    def __post_init__(self) -> None:
        # print(f"DEBUG: APIClient: max concurrency = {self.max_concurrency}")
        # print(f"DEBUG: APIClient: timeout         = {self.timeout}")
        if self._journal is None:
            self._journal = APIJournal()
        if self._worker is None:
            # TODO
            # self._worker = CurlAPIClientWorker(self.max_concurrency)
            self._worker = UrllibAPIClientWorker(self.max_concurrency)
        if self._endpoint_state is None:
            self._endpoint_state = dict()

    async def chat_completion(
        self,
        model: APIModel,
        messages: Any,
        sampling_params: Optional[dict] = None,
        tools: Optional[list] = None,
        payload: Any = None,
        debug: Optional[bool] = None,
    ):
        assert payload is None, (
            "APIClient.chat_completion: not implemented: passthrough payload"
        )
        if model.endpoint.name not in self._endpoint_state:
            self._endpoint_state[model.endpoint.name] = APIClientEndpointState()
        endpoint_state = self._endpoint_state[model.endpoint.name]
        work_item = _APIChatCompletionWorkItem(
            model=model,
            messages=messages,
            sampling_params=sampling_params,
            tools=tools,
            debug=debug,
        )
        work_item._prepare()
        def _chat_completion_work_item():
            _try_chat_completion(work_item, endpoint_state)
            return work_item
        loop = asyncio.get_running_loop()
        w = loop.run_in_executor(self._worker._poolexec, _chat_completion_work_item)
        work_item = await w
        res = work_item._finalize()
        if self._journal is not None:
            print(f"DEBUG: APIClient.chat_completion: journal put...")
            raise NotImplementedError
            put_ret = await self._journal.put(item)
            print(f"DEBUG: APIClient.chat_completion: journal put: ret = {repr(put_ret)}")
        return res

    async def message(
        self,
        model: APIModel,
        messages: Any = None,
        sampling_params: Optional[dict] = None,
        tools: Optional[list] = None,
        api_version: Optional[str] = None,
        fresh: Optional[bool] = None,
        debug: Optional[bool] = None,
    ):
        if False:
            assert payload is None, (
                "APIClient.message: not implemented: passthrough payload"
            )
        if fresh is None:
            fresh = True
        if self._journal is not None and not fresh:
            res = await self._journal.get(
                model,
                messages=messages,
                sampling_params=sampling_params,
                tools=tools,
            )
            if res is not None:
                return res
        if model.endpoint.name not in self._endpoint_state:
            self._endpoint_state[model.endpoint.name] = APIClientEndpointState()
        endpoint_state = self._endpoint_state[model.endpoint.name]
        if model.endpoint.throttle_concurrency is not None:
            if isinstance(endpoint_state._max_concurrency, async_nullcontext):
                # print(f"DEBUG: APIClient.message: concurrency = {model.endpoint.throttle_concurrency}")
                endpoint_state._max_concurrency = asyncio.Semaphore(model.endpoint.throttle_concurrency)
        work_item = _APIMessageWorkItem(
            model=model,
            messages=messages,
            sampling_params=sampling_params,
            tools=tools,
            api_version=api_version,
            # debug=debug,
        )
        def _message_work_item(work_item, endpoint_state, worker):
            _try_message(work_item, endpoint_state, worker)
            return work_item
        loop = asyncio.get_running_loop()
        async with endpoint_state._max_concurrency:
            w = loop.run_in_executor(self._worker._poolexec, _message_work_item, work_item, endpoint_state, self._worker)
            work_item = await w
        res = work_item._finalize()
        if self._journal is not None:
            if res is not None and res.except_ is None:
                await self._journal.put(
                    model,
                    messages=messages,
                    sampling_params=sampling_params,
                    tools=tools,
                    res=res,
                )
        return res
