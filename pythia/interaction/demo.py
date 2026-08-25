from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional
from typing import Sequence

from .chat_completions import ChatCompletionsEndpoint
from .chat_completions import ChatCompletionsModel
from .context import ModelContext
from .default_environment import DefaultEnvironment
from .environment import Environment
from .items import Message
from .model import Model
from .model import SamplingOptions
from .user import UserInteraction


DEFAULT_PROMPT = "Summarize the repository in the current working directory."

_SYSTEM_MESSAGE = (
    "You are a repository analyst. Inspect the repository with tools as "
    "needed. Do not modify files when the user only asks for analysis."
)


def run_repository_summary(
    model: Model,
    environment: Environment,
    *,
    prompt: str = DEFAULT_PROMPT,
    max_samples: int = 100,
    options: Optional[SamplingOptions] = None,
) -> str:
    if not hasattr(model, "sample") or not callable(model.sample):
        raise TypeError("model must provide sample(...)")
    if not isinstance(environment, Environment):
        raise TypeError("environment must be Environment")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must not be empty")
    if (
        isinstance(max_samples, bool)
        or not isinstance(max_samples, int)
        or max_samples <= 0
    ):
        raise ValueError("max_samples must be a positive integer")
    if options is not None and not isinstance(options, SamplingOptions):
        raise TypeError("options must be SamplingOptions or None")

    context = ModelContext(
        (
            Message(role="system", text=_SYSTEM_MESSAGE),
        )
    )
    user_interaction = UserInteraction(
        items=(Message(role="user", text=prompt),),
    )
    context.extend(user_interaction.context_items())
    for display_item in user_interaction.display_items():
        print(display_item)

    for _ in range(max_samples):
        sample = model.sample(
            context,
            tools=environment.tool_specs,
            options=options,
        )
        context.extend(sample.context_items())
        for display_item in sample.display_items():
            print(display_item)
        if not sample.tool_calls:
            final_text = sample.last_assistant_text
            if final_text is None or not final_text.strip():
                raise RuntimeError("model returned no final assistant text")
            return final_text

        result = environment.execute_tool_calls(sample.tool_calls)
        context.extend(result.context_items())
        for display_item in result.display_items(source_calls=sample.tool_calls):
            print(display_item)

    raise RuntimeError(
        f"model did not produce a final answer within {max_samples} samples"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Ask a local Chat Completions model to summarize a repository "
            "using Pythia's default local tools."
        ),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model")
    parser.add_argument("--scheme", default="http")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--cwd", default=".")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=60.0,
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    cwd = Path(args.cwd).expanduser().resolve()
    endpoint = ChatCompletionsEndpoint(
        host=args.host,
        port=args.port,
        model=args.model,
        scheme=args.scheme,
        request_timeout_seconds=args.request_timeout_seconds,
        api_key=args.api_key,
    )
    model = ChatCompletionsModel(endpoint)
    options = (
        SamplingOptions(max_tokens=args.max_tokens)
        if args.max_tokens is not None
        else None
    )

    print(
        "Warning: exec_command runs without a sandbox; use only with a "
        "trusted local model and workspace.",
        file=sys.stderr,
    )
    try:
        with DefaultEnvironment(cwd=cwd) as environment:
            run_repository_summary(
                model,
                environment,
                prompt=args.prompt,
                max_samples=args.max_samples,
                options=options,
            )
    except Exception as exc:
        print(f"demo failed: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
