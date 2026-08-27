from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional
from typing import Sequence
from typing import Union

from .chat_completions import ChatCompletionsEndpoint
from .chat_completions import ChatCompletionsModel
from .context import ModelContext
from .default_environment import DefaultEnvironment
from .display import render_interaction_items
from .environment import Environment
from .items import Message
from .items import ModelSampleBoundary
from .items import TurnMetadata
from .items import UserInteractionBoundary
from .model import Model
from .model import SamplingOptions
from .session import load_interaction_session
from .session import save_interaction_session
from .user import UserInteraction


_SYSTEM_MESSAGE = (
    "You are a repository analyst. Inspect the repository with tools as "
    "needed. Do not modify files when the user only asks for analysis."
)

# DEFAULT_PROMPT = "Summarize the repository in the current working directory."
# DEFAULT_PROMPT = "Here is the log for a recent run of the pythia/interaction demo. Let's investigate why there appears to be no interleaved assistant reasoning/response text, but only tool calls."
# DEFAULT_PROMPT = "Here is the log (demo.log.1) for a recent run of the pythia/interaction demo. Let's review the investigation of why there appears to be no interleaved assistant reasoning/response text but only tool calls, and implement a narrow fix for reasoning (no system message change). (Note that we should support both \"reasoning_content\" and \"reasoning\", as the former is still returned by some inference engines (although we might prioritize the latter if it exists and is non-null/non-empty)."
# DEFAULT_PROMPT = "Here is the log (demo.log.1) for a recent run of the pythia/interaction demo. Notice that the blocks (`[user] ...`, `[assistant] ...`, etc.) are not left-indented/delimited like in the existing autopythia/contradex implementation. Let's investigate and plan to re-add the same left-indentation/decoration to pythia.interaction as part of the display item impl."
DEFAULT_PROMPT = "In pythia.interaction is the model context (list of interaction items) sufficient state for saving/resuming sessions? (Pending/interrupted tool calls/results might pose an issue, but we can ignore those for now so long as those interrupted calls can be swept over on resume.) Assuming sufficiency, let's implement initial support for saving the current session (in interaction.jsonl), and optionally resuming from it by passing --resume to the demo (let's also keep the working tree changes to the demo)."
DEFAULT_SESSION_PATH = Path("interaction.jsonl")


def run_repository_summary(
    model: Model,
    environment: Environment,
    *,
    prompt: Optional[str] = DEFAULT_PROMPT,
    max_samples: int = 100,
    options: Optional[SamplingOptions] = None,
    session_path: Optional[Union[str, Path]] = None,
    resume: bool = False,
) -> str:
    if not hasattr(model, "sample") or not callable(model.sample):
        raise TypeError("model must provide sample(...)")
    if not isinstance(environment, Environment):
        raise TypeError("environment must be Environment")
    if not isinstance(resume, bool):
        raise TypeError("resume must be a bool")
    if prompt is not None and (
        not isinstance(prompt, str) or not prompt.strip()
    ):
        raise ValueError("prompt must be a non-empty string or None")
    if resume and session_path is None:
        raise ValueError("resume requires session_path")
    if (
        isinstance(max_samples, bool)
        or not isinstance(max_samples, int)
        or max_samples <= 0
    ):
        raise ValueError("max_samples must be a positive integer")
    if options is not None and not isinstance(options, SamplingOptions):
        raise TypeError("options must be SamplingOptions or None")

    if resume:
        context = load_interaction_session(session_path)

        # Restore the human-visible transcript as well as the model state.
        for display_item in render_interaction_items(context.items):
            print(display_item)
    else:
        if prompt is None:
            raise ValueError("prompt must not be None without resume")
        context = ModelContext(
            (
                # Message(role="system", text=_SYSTEM_MESSAGE),
            )
        )

    if session_path is not None:
        save_interaction_session(session_path, context)

    def _persist() -> None:
        if session_path is not None:
            save_interaction_session(session_path, context)

    pending_calls = context.pending_tool_calls()
    if pending_calls:
        # A process interruption may leave the durable context after model
        # output but before its tool results.  Complete that batch before
        # asking the model for a new sample.
        result = environment.execute_tool_calls(pending_calls)
        context.extend(result.context_items())
        _persist()
        for display_item in result.display_items(source_calls=pending_calls):
            print(display_item)
    elif resume and prompt is None:
        final_text = _final_assistant_text(context)
        if final_text is None or not final_text.strip():
            raise RuntimeError("resumed session has no final assistant text")
        return final_text

    if prompt is not None:
        # For a resumed session, add the follow-up only after any pending
        # tool batch has been made valid again.
        user_interaction = UserInteraction(
            items=(Message(role="user", text=prompt),),
        )
        context.extend(user_interaction.context_items())
        _persist()
        for display_item in user_interaction.display_items():
            print(display_item)

    for _ in range(max_samples):
        sample = model.sample(
            context,
            tools=environment.tool_specs,
            options=options,
        )
        context.extend(sample.context_items())
        _persist()
        for display_item in sample.display_items():
            print(display_item)
        if not sample.tool_calls:
            final_text = sample.last_assistant_text
            if final_text is None or not final_text.strip():
                raise RuntimeError("model returned no final assistant text")
            return final_text

        result = environment.execute_tool_calls(sample.tool_calls)
        context.extend(result.context_items())
        _persist()
        for display_item in result.display_items(source_calls=sample.tool_calls):
            print(display_item)

    raise RuntimeError(
        f"model did not produce a final answer within {max_samples} samples"
    )


def _final_assistant_text(context: ModelContext) -> Optional[str]:
    """Return final assistant text if the effective context ends with it."""
    for item in reversed(context.model_items()):
        if isinstance(
            item,
            (ModelSampleBoundary, TurnMetadata, UserInteractionBoundary),
        ):
            continue
        if isinstance(item, Message) and item.role == "assistant":
            return item.text
        return None
    return None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Ask a local Chat Completions model to summarize a repository "
            "using Pythia's default local tools."
        ),
    )
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--cwd", default=".")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=60.0,
    )
    parser.add_argument("--prompt")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume interaction.jsonl instead of starting a new session",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    cwd = Path(args.cwd).expanduser().resolve()
    prompt = args.prompt
    if prompt is None and not args.resume:
        prompt = DEFAULT_PROMPT
    endpoint = ChatCompletionsEndpoint(
        api_url=args.api_url,
        model=args.model,
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
                prompt=prompt,
                max_samples=args.max_samples,
                options=options,
                session_path=DEFAULT_SESSION_PATH,
                resume=args.resume,
            )
    except Exception as exc:
        print(f"demo failed: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
