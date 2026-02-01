from typing import Any, Optional
from dataclasses import dataclass, field
import os

@dataclass
class APIEndpoint:
    name: str = None
    domain: str = None
    protocol: str = None
    api_url: str = None
    api_key: str = None
    protocol_api_urls: Optional[dict[str, str]] = None
    throttle_rps: Optional[int] = None
    throttle_concurrency: Optional[int] = None

@dataclass
class APIModel:
    model_path: str = None
    default_sampling_params: dict = None
    default_thinking: Optional[bool] = None
    endpoint_protocol: Optional[str] = None
    endpoint_api_url: Optional[str] = None
    endpoint_model_path: str = None
    endpoint_extra_params: dict = None
    endpoint: APIEndpoint = None
    throttle_rps: Optional[int] = None
    throttle_concurrency: Optional[int] = None

@dataclass
class APIRegistry:
    environ: dict[str, str] = field(default_factory=dict)
    endpoints: dict[str, Any] = field(default_factory=dict)
    endpoint_aliases: dict[str, str] = field(default_factory=dict)
    models: dict[str, Any] = field(default_factory=dict)
    model_aliases: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        self._register_all()

    def get_env(self, key: str, rstrip: bool = True) -> Optional[str]:
        v = self.environ.get(key, None)
        if v is None:
            v = os.environ.get(key, None)
            if v is None:
                # print(f"DEBUG: APIRegistry: warning: missing env var {repr(key)}")
                pass
            else:
                if rstrip:
                    v = v.rstrip()
                self.environ[key] = v
        return v

    def find_endpoint(self, name: str) -> Optional[APIEndpoint]:
        if name in self.endpoint_aliases:
            name = self.endpoint_aliases[name]
        return self.endpoints.get(name, None)

    def find_model(self, model_path: str) -> Optional[APIModel]:
        model_key = model_path.lower()
        if model_key in self.model_aliases:
            model_key = self.model_aliases[model_key]
        return self.models.get(model_key, None)

    def register_endpoint(self, name: str, aliases: list[str] = [], **kwargs):
        if name in self.endpoints:
            raise KeyError
        for alias_name in aliases:
            if alias_name in self.endpoint_aliases:
                raise KeyError
            self.endpoint_aliases[alias_name] = name
        self.endpoints[name] = APIEndpoint(name=name, **kwargs)

    def register_model(self, endpoint_name: str, model_path: str, aliases: list[str] = [], **kwargs):
        model_key = model_path.lower()
        if model_key in self.models:
            raise KeyError
        for alias_path in aliases:
            alias_key = alias_path.lower()
            if alias_key in self.model_aliases:
                raise KeyError
            self.endpoint_aliases[alias_key] = model_key
        endpoint = self.find_endpoint(endpoint_name)
        self.models[model_key] = APIModel(
            endpoint=endpoint,
            model_path=model_path,
            **kwargs
        )

    def _register_all(self):
        self._register_all_endpoints()
        if self.find_endpoint("anthropic"):
            self._register_anthropic_models()
        if self.find_endpoint("deepinfra"):
            self._register_deepinfra_models()
        if self.find_endpoint("deepseek"):
            self._register_deepseek_models()
        if self.find_endpoint("gemini"):
            self._register_gemini_models()
        if self.find_endpoint("moonshot"):
            self._register_moonshot_models()
        if self.find_endpoint("openai"):
            self._register_openai_models()
        if self.find_endpoint("together"):
            self._register_together_models()
        if self.find_endpoint("x.ai"):
            self._register_xai_models()
        self._register_local_models()

    def _register_all_endpoints(self):
        api_key = self.get_env("ANTHROPIC_API_KEY")
        if api_key is not None:
            self.register_endpoint(
                "anthropic", ["anthropic.com"],
                domain="anthropic.com",
                protocol="anthropic",
                api_url="https://api.anthropic.com",
                api_key=api_key,
            )
        api_key = self.get_env("DEEPINFRA_API_KEY")
        if api_key is not None:
            self.register_endpoint(
                "deepinfra", ["deepinfra.com"],
                domain="deepinfra.com",
                protocol="openai",
                api_url="https://api.deepinfra.com",
                api_key=api_key,
                throttle_concurrency=192,
            )
        api_key = self.get_env("DEEPSEEK_API_KEY")
        if api_key is not None:
            self.register_endpoint(
                "deepseek", ["deepseek-ai", "deepseek.com"],
                domain="deepseek.com",
                protocol="deepseek",
                api_url="https://api.deepseek.com",
                api_key=api_key,
                protocol_api_urls={
                    "deepseek": "https://api.deepseek.com",
                    "anthropic": "https://api.deepseek.com/anthropic",
                },
            )
        api_key = self.get_env("GEMINI_API_KEY")
        if api_key is not None:
            self.register_endpoint(
                "gemini",
                domain="google.com",
                protocol="gemini",
                api_url="https://generativelanguage.googleapis.com",
                api_key=api_key,
                throttle_rps=2,
            )
        api_key = self.get_env("MOONSHOT_API_KEY")
        if api_key is not None:
            self.register_endpoint(
                "moonshot", ["moonshot.ai"],
                domain="moonshot.ai",
                protocol="openai",
                api_url="https://api.moonshot.ai",
                api_key=api_key,
                protocol_api_urls={
                    "openai": "https://api.moonshot.ai",
                    "anthropic": "https://api.moonshot.ai/anthropic",
                },
            )
        api_key = self.get_env("OPENAI_API_KEY")
        if api_key is not None:
            self.register_endpoint(
                "openai", ["openai.com"],
                domain="openai.com",
                protocol="openai",
                api_url="https://api.openai.com",
                api_key=api_key,
            )
        api_key = self.get_env("TOGETHER_API_KEY")
        if api_key is not None:
            self.register_endpoint(
                "together", ["together.ai", "together.xyz"],
                domain="together.xyz",
                protocol="openai",
                api_url="https://api.together.xyz",
                api_key=api_key,
                throttle_rps=10,
            )
        api_key = self.get_env("XAI_API_KEY")
        if api_key is not None:
            self.register_endpoint(
                "x-ai", ["x.ai", "xai"],
                domain="x.ai",
                protocol="openai",
                api_url="https://api.x.ai",
                api_key=api_key,
            )
        api_key = self.get_env("ZAI_API_KEY")
        if api_key is not None:
            self.register_endpoint(
                "z-ai", ["z.ai", "zai"],
                domain="z.ai",
                protocol="openai",
                api_url="https://api.z.ai",
                api_key=api_key,
            )
        self.register_endpoint(
            "__local__",
            domain="localhost",
            protocol="openai",
            api_url="http://127.0.0.1:10000",
            api_key=None,
        )

    def _register_anthropic_models(self):
        self.register_model(
            "anthropic",
            "anthropic/claude-4-sonnet-thinking-off",
            endpoint_model_path="claude-sonnet-4-20250514",
            endpoint_extra_params={
                "thinking": {
                    "type": "disabled",
                },
            },
        )
        self.register_model(
            "anthropic",
            "anthropic/claude-4-sonnet-thinking-32k",
            ["anthropic/claude-4-sonnet-thinking-on-32k"],
            endpoint_model_path="claude-sonnet-4-20250514",
            endpoint_extra_params={
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": 32000,
                },
            },
        )
        self.register_model(
            "anthropic",
            "anthropic/claude-4-opus-thinking-off",
            endpoint_model_path="claude-opus-4-20250514",
            endpoint_extra_params={
                "thinking": {
                    "type": "disabled",
                },
            },
        )
        self.register_model(
            "anthropic",
            "anthropic/claude-4-opus-thinking-32k",
            ["anthropic/claude-4-opus-thinking-on-32k"],
            endpoint_model_path="claude-opus-4-20250514",
            endpoint_extra_params={
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": 32000,
                },
            },
        )
        self.register_model(
            "anthropic",
            "anthropic/claude-4.5-sonnet-thinking-off",
            endpoint_model_path="claude-sonnet-4-5-20250929",
            endpoint_extra_params={
                "thinking": {
                    "type": "disabled",
                },
            },
        )
        self.register_model(
            "anthropic",
            "anthropic/claude-4.5-sonnet-thinking-32k",
            ["anthropic/claude-4.5-sonnet-thinking-on-32k"],
            endpoint_model_path="claude-sonnet-4-5-20250929",
            endpoint_extra_params={
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": 32000,
                },
            },
        )
        self.register_model(
            "anthropic",
            "anthropic/claude-4.5-opus-thinking-off",
            endpoint_model_path="claude-opus-4-5-20251101",
            endpoint_extra_params={
                "thinking": {
                    "type": "disabled",
                },
            },
        )
        self.register_model(
            "anthropic",
            "anthropic/claude-4.5-opus-thinking-32k",
            ["anthropic/claude-4.5-opus-thinking-on-32k"],
            endpoint_model_path="claude-opus-4-5-20251101",
            endpoint_extra_params={
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": 32000,
                },
            },
        )

    def _register_deepinfra_models(self):
        self.register_model(
            "deepinfra",
            "deepinfra/deepseek-ai/deepseek-v3-fp4",
            endpoint_model_path="deepseek-ai/DeepSeek-V3",
        )
        self.register_model(
            "deepinfra",
            "deepinfra/deepseek-ai/deepseek-v3-0324-fp4",
            endpoint_model_path="deepseek-ai/DeepSeek-V3-0324",
        )
        self.register_model(
            "deepinfra",
            "deepinfra/deepseek-ai/deepseek-r1-fp4",
            endpoint_model_path="deepseek-ai/DeepSeek-R1",
        )
        self.register_model(
            "deepinfra",
            "deepinfra/deepseek-ai/deepseek-r1-0508-fp4",
            endpoint_model_path="deepseek-ai/DeepSeek-R1-0508-Turbo",
        )
        if False:
            self.register_model(
                "deepinfra",
                "deepinfra/deepseek-ai/deepseek-v3.1-fp4",
                endpoint_model_path="deepseek-ai/DeepSeek-V3.1",
            )
        self.register_model(
            "deepinfra",
            "deepinfra/google/gemini-2.5-flash",
            endpoint_model_path="google/gemini-2.5-flash",
        )
        self.register_model(
            "deepinfra",
            "deepinfra/google/gemini-2.5-pro",
            endpoint_model_path="google/gemini-2.5-pro",
        )
        self.register_model(
            "deepinfra",
            "deepinfra/moonshotai/kimi-k2-instruct-fp4",
            endpoint_model_path="moonshotai/Kimi-K2-Instruct",
        )
        self.register_model(
            "deepinfra",
            "deepinfra/moonshotai/kimi-k2-instruct-0905-fp4",
            endpoint_model_path="moonshotai/Kimi-K2-Instruct-0905",
        )
        self.register_model(
            "deepinfra",
            "deepinfra/openai/gpt-oss-20b-fp4",
            endpoint_model_path="openai/gpt-oss-20b",
        )
        self.register_model(
            "deepinfra",
            "deepinfra/openai/gpt-oss-20b-fp4-low",
            endpoint_model_path="openai/gpt-oss-20b",
            endpoint_extra_params={
                "reasoning_effort": "low",
            },
        )
        self.register_model(
            "deepinfra",
            "deepinfra/openai/gpt-oss-20b-fp4-high",
            endpoint_model_path="openai/gpt-oss-20b",
            endpoint_extra_params={
                "reasoning_effort": "high",
            },
        )
        self.register_model(
            "deepinfra",
            "deepinfra/openai/gpt-oss-120b-fp4",
            endpoint_model_path="openai/gpt-oss-120b",
        )
        self.register_model(
            "deepinfra",
            "deepinfra/qwen/qwen3-235b-a22b-instruct-2507-fp8",
            endpoint_model_path="Qwen/Qwen3-235B-A22B-Instruct-2507",
        )
        self.register_model(
            "deepinfra",
            "deepinfra/qwen/qwen3-235b-a22b-thinking-2507-fp8",
            endpoint_model_path="Qwen/Qwen3-235B-A22B-Thinking-2507",
        )
        self.register_model(
            "deepinfra",
            "deepinfra/qwen/qwen3-coder-480b-a35b-instruct-fp8",
            endpoint_model_path="Qwen/Qwen3-Coder-480B-A35B-Instruct",
        )
        self.register_model(
            "deepinfra",
            "deepinfra/qwen/qwen3-next-80b-a3b-instruct",
            endpoint_model_path="Qwen/Qwen3-Next-80B-A3B-Instruct",
        )
        self.register_model(
            "deepinfra",
            "deepinfra/qwen/qwen3-next-80b-a3b-thinking",
            endpoint_model_path="Qwen/Qwen3-Next-80B-A3B-Thinking",
        )
        self.register_model(
            "deepinfra",
            "deepinfra/zai-org/glm-4.5-fp8",
            endpoint_model_path="zai-org/GLM-4.5",
        )

    def _register_deepseek_models_deprecated(self):
        self.register_model(
            "deepseek",
            "deepseek-ai/deepseek-v3-0324",
            endpoint_model_path="deepseek-chat",
        )
        self.register_model(
            "deepseek",
            "deepseek-ai/deepseek-r1-0528",
            endpoint_model_path="deepseek-reasoner",
        )
        self.register_model(
            "deepseek",
            "deepseek-ai/deepseek-v3.1-thinking-off",
            endpoint_model_path="deepseek-chat",
        )
        self.register_model(
            "deepseek",
            "deepseek-ai/deepseek-v3.1-thinking",
            ["deepseek-ai/deepseek-v3.1-thinking-on"],
            endpoint_model_path="deepseek-reasoner",
            endpoint_extra_params={
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": -1,
                },
            },
        )
        self.register_model(
            "deepseek",
            "deepseek-ai/deepseek-v3.1-terminus-thinking-off",
            endpoint_model_path="deepseek-chat",
        )
        self.register_model(
            "deepseek",
            "deepseek-ai/deepseek-v3.1-terminus-thinking",
            ["deepseek-ai/deepseek-v3.1-terminus-thinking-on"],
            endpoint_model_path="deepseek-reasoner",
        )
        if False:
            self.register_model(
                "deepseek",
                "deepseek-ai/deepseek-v3.1-thinking-off-messages",
                endpoint_api_url="https://api.deepseek.com/anthropic",
                endpoint_protocol="anthropic",
                endpoint_model_path="deepseek-chat",
            )
            self.register_model(
                "deepseek",
                "deepseek-ai/deepseek-v3.1-thinking-on-messages",
                endpoint_api_url="https://api.deepseek.com/anthropic",
                endpoint_protocol="anthropic",
                endpoint_model_path="deepseek-reasoner",
            )
        self.register_model(
            "deepseek",
            "deepseek-ai/deepseek-v3.2-exp-thinking-off",
            endpoint_model_path="deepseek-chat",
        )
        self.register_model(
            "deepseek",
            "deepseek-ai/deepseek-v3.2-exp-thinking",
            ["deepseek-ai/deepseek-v3.2-exp-thinking-on"],
            endpoint_model_path="deepseek-reasoner",
            default_thinking=True,
        )

    def _register_deepseek_models(self):
        self.register_model(
            "deepseek",
            "deepseek-ai/deepseek-v3.2-thinking-off",
            endpoint_model_path="deepseek-chat",
        )
        self.register_model(
            "deepseek",
            "deepseek-ai/deepseek-v3.2-thinking",
            ["deepseek-ai/deepseek-v3.2-thinking-on"],
            endpoint_model_path="deepseek-reasoner",
            default_thinking=True,
        )
        self.register_model(
            "deepseek",
            "deepseek-ai/deepseek-v3.2-speciale-thinking",
            endpoint_protocol="deepseek",
            endpoint_api_url="https://api.deepseek.com/v3.2_speciale_expires_on_20251215",
            endpoint_model_path="deepseek-reasoner",
            default_thinking=True,
        )

    def _register_gemini_models(self):
        self.register_model(
            "gemini",
            "google/gemini-2.5-flash-thinking-off",
            endpoint_model_path="models/gemini-2.5-flash",
            endpoint_extra_params={
                "generationConfig": {
                    "thinkingConfig": {
                        "thinkingBudget": 0,
                    }
                },
            },
        )
        self.register_model(
            "gemini",
            "google/gemini-2.5-flash-thinking",
            ["google/gemini-2.5-flash-thinking-on"],
            endpoint_model_path="models/gemini-2.5-flash",
            endpoint_extra_params={
                "generationConfig": {
                    "thinkingConfig": {
                        "includeThoughts": True,
                        "thinkingBudget": -1,
                    }
                },
            },
        )
        self.register_model(
            "gemini",
            "google/gemini-2.5-pro-thinking-off",
            endpoint_model_path="models/gemini-2.5-pro",
            endpoint_extra_params={
                "generationConfig": {
                    "thinkingConfig": {
                        "thinkingBudget": 0,
                    }
                },
            },
        )
        self.register_model(
            "gemini",
            "google/gemini-2.5-pro-thinking",
            ["google/gemini-2.5-pro-thinking-on"],
            endpoint_model_path="models/gemini-2.5-pro",
            endpoint_extra_params={
                "generationConfig": {
                    "thinkingConfig": {
                        "includeThoughts": True,
                        "thinkingBudget": -1,
                    }
                },
            },
        )

    def _register_moonshot_models(self):
        if False:
            self.register_model(
                "moonshot",
                "moonshotai/kimi-k2-turbo-thinking-off",
                endpoint_model_path="kimi-k2-turbo-preview",
            )
            self.register_model(
                "moonshot",
                "moonshotai/kimi-k2-turbo-thinking",
                ["moonshotai/kimi-k2-turbo-thinking-on"],
                endpoint_model_path="kimi-k2-thinking-turbo",
            )
        self.register_model(
            "moonshot",
            "moonshotai/kimi-k2-instruct",
            endpoint_model_path="kimi-k2-0711-preview",
            throttle_rps=80,
            throttle_concurrency=198,
        )
        self.register_model(
            "moonshot",
            "moonshotai/kimi-k2-instruct-0905",
            endpoint_model_path="kimi-k2-0905-preview",
            throttle_rps=80,
            throttle_concurrency=198,
        )
        self.register_model(
            "moonshot",
            "moonshotai/kimi-k2-thinking",
            endpoint_model_path="kimi-k2-thinking",
            default_thinking=True,
            throttle_rps=80,
            throttle_concurrency=198,
        )
        self.register_model(
            "moonshot",
            "moonshotai/kimi-k2.5-thinking-off",
            endpoint_model_path="kimi-k2.5",
            endpoint_extra_params={
                "thinking": {"type": "disabled"},
            },
            throttle_rps=80,
            throttle_concurrency=198,
        )
        self.register_model(
            "moonshot",
            "moonshotai/kimi-k2.5-thinking",
            endpoint_model_path="kimi-k2.5",
            default_thinking=True,
            throttle_rps=80,
            throttle_concurrency=198,
        )

    def _register_openai_models(self):
        self.register_model(
            "openai",
            "openai/gpt-4o-mini",
            endpoint_model_path="gpt-4o-mini-2024-07-18",
        )
        self.register_model(
            "openai",
            "openai/gpt-4o-20240806",
            endpoint_model_path="gpt-4o-2024-08-06",
        )
        self.register_model(
            "openai",
            "openai/gpt-4.1",
            endpoint_model_path="gpt-4.1-2025-04-14",
        )
        self.register_model(
            "openai",
            "openai/o3-high",
            endpoint_model_path="o3-2025-04-16",
            endpoint_extra_params={
                "reasoning_effort": "high",
            },
        )
        self.register_model(
            "openai",
            "openai/o4-mini-high",
            endpoint_model_path="o4-mini-2025-04-16",
            endpoint_extra_params={
                "reasoning_effort": "high",
            },
        )
        self.register_model(
            "openai",
            "openai/gpt-5-high",
            endpoint_model_path="gpt-5-2025-08-07",
            endpoint_extra_params={
                "reasoning_effort": "high",
            },
        )
        self.register_model(
            "openai",
            "openai/gpt-5.1-high",
            endpoint_model_path="gpt-5.1-2025-11-13",
            endpoint_extra_params={
                "reasoning_effort": "high",
            },
        )
        self.register_model(
            "openai",
            "openai/gpt-5.2-high",
            endpoint_model_path="gpt-5.2-2025-12-11",
            endpoint_extra_params={
                "reasoning_effort": "high",
            },
        )

    def _register_together_models(self):
        self.register_model(
            "together",
            "togetherai/qwen/qwen3-235b-a22b-fp8",
            endpoint_model_path="Qwen/Qwen3-235B-A22B-fp8-tput",
        )
        self.register_model(
            "together",
            "togetherai/qwen/qwen3-235b-a22b-instruct-2507-fp8",
            endpoint_model_path="Qwen/Qwen3-235B-A22B-Instruct-2507-tput",
        )
        self.register_model(
            "together",
            "togetherai/qwen/qwen3-235b-a22b-thinking-2507-fp8",
            endpoint_model_path="Qwen/Qwen3-235B-A22B-Thinking-2507",
        )
        self.register_model(
            "together",
            "togetherai/moonshotai/kimi-k2-instruct",
            endpoint_model_path="moonshotai/Kimi-K2-Instruct",
        )

    def _register_xai_models(self):
        self.register_model(
            "x-ai",
            "x-ai/grok-3-mini",
            endpoint_model_path="grok-3-mini",
            throttle_rps=8,
        )
        self.register_model(
            "x-ai",
            "x-ai/grok-4",
            endpoint_model_path="grok-4-0709",
            throttle_rps=8,
        )
        self.register_model(
            "x-ai",
            "x-ai/grok-4-fast-thinking-off",
            endpoint_model_path="grok-4-fast-non-reasoning",
            throttle_rps=8,
        )
        self.register_model(
            "x-ai",
            "x-ai/grok-4-fast-thinking",
            ["x-ai/grok-4-fast-thinking-on"],
            endpoint_model_path="grok-4-fast-reasoning",
            throttle_rps=8,
        )
        self.register_model(
            "x-ai",
            "x-ai/grok-code-fast",
            endpoint_model_path="grok-code-fast-1",
            throttle_rps=8,
        )
        self.register_model(
            "x-ai",
            "x-ai/grok-4.1-fast-thinking-off",
            endpoint_model_path="grok-4-1-fast-non-reasoning",
            throttle_rps=8,
        )
        self.register_model(
            "x-ai",
            "x-ai/grok-4.1-fast-thinking",
            ["x-ai/grok-4.1-fast-thinking-on"],
            endpoint_model_path="grok-4-1-fast-reasoning",
            throttle_rps=8,
        )

    def _register_zai_models(self):
        self.register_model(
            "z-ai",
            "z-ai/glm-4.7-thinking",
            endpoint_model_path="glm-4.7",
            endpoint_extra_params={
                "thinking": "enabled",
            },
            default_thinking=True,
            throttle_concurrency=5,
        )

    def _register_local_models(self):
        pass
        # self.register_model(
        #     "__local__",
        #     "__local__/openai/gpt-oss-20b",
        #     endpoint_model_path="openai/gpt-oss-20b",
        # )
        self.register_model(
            "__local__",
            "__local__/qwen/qwen3-4b-instruct-2507-fp8",
            endpoint_model_path="Qwen/Qwen3-4B-Instruct-2507-FP8",
        )
        self.register_model(
            "__local__",
            "__local__/qwen/qwen3-4b-thinking-2507-fp8",
            endpoint_model_path="Qwen/Qwen3-4B-Thinking-2507-FP8",
        )
        self.register_model(
            "__local__",
            "__local__/qwen/qwen3-30b-a3b-instruct-2507-fp8",
            endpoint_model_path="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
        )
        self.register_model(
            "__local__",
            "__local__/qwen/qwen3-30b-a3b-thinking-2507-fp8",
            endpoint_model_path="Qwen/Qwen3-30B-A3B-Thinking-2507-FP8",
        )
