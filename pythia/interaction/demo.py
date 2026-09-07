from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional
from typing import Sequence
from typing import Union

from .context import ModelContext
from .default_environment import DefaultEnvironment
from .display import render_interaction_items
from .environment import Environment
from .items import Init
from .items import Instructions
from .items import Message
from .items import ModelSampleBoundary
from .items import TurnMetadata
from .items import TurnSummary
from .items import UserInteractionBoundary
from .items import summarize_turn_usage
from .model import Model
from .model import SamplingOptions
from .model_config import DEFAULT_SAVE_PATH as DEFAULT_SAVE_PATH
from .model_config import build_model
from .model_config import build_parser
from .model_config import initial_model_name
from .model_config import resolve_save_path
from .save import load_interaction_save
from .save import save_interaction_save
from .user import UserInteraction


_SYSTEM_MESSAGE = (
    "You are a repository analyst. Inspect the repository with tools as "
    "needed. Do not modify files when the user only asks for analysis."
)

DEFAULT_PROMPT = "Summarize the repository in the current working directory."
# DEFAULT_PROMPT = "Here is the log for a recent run of the pythia/interaction demo. Let's investigate why there appears to be no interleaved assistant reasoning/response text, but only tool calls."
# DEFAULT_PROMPT = "Here is the log (demo.log.1) for a recent run of the pythia/interaction demo. Let's review the investigation of why there appears to be no interleaved assistant reasoning/response text but only tool calls, and implement a narrow fix for reasoning (no system message change). (Note that we should support both \"reasoning_content\" and \"reasoning\", as the former is still returned by some inference engines (although we might prioritize the latter if it exists and is non-null/non-empty)."
# DEFAULT_PROMPT = "Here is the log (demo.log.1) for a recent run of the pythia/interaction demo. Notice that the blocks (`[user] ...`, `[assistant] ...`, etc.) are not left-indented/delimited like in the existing autopythia/contradex implementation. Let's investigate and plan to re-add the same left-indentation/decoration to pythia.interaction as part of the display item impl."
# DEFAULT_PROMPT = "In pythia.interaction is the model context (list of interaction items) sufficient state for saving/resuming sessions? (Pending/interrupted tool calls/results might pose an issue, but we can ignore those for now so long as those interrupted calls can be swept over on resume.) Assuming sufficiency, let's implement initial support for saving the current session (in interaction.jsonl), and optionally resuming from it by passing --resume to the demo (let's also keep the working tree changes to the demo)."


def run(
    model: Model,
    environment: Environment,
    *,
    prompt: Optional[str] = DEFAULT_PROMPT,
    instructions: Optional[Union[str, Instructions]] = None,
    max_samples: Optional[int] = None,
    options: Optional[SamplingOptions] = None,
    save_path: Optional[Union[str, Path]] = None,
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
    if instructions is not None and not isinstance(
        instructions, (str, Instructions)
    ):
        raise TypeError("instructions must be a string, Instructions, or None")
    # Empty/whitespace-only strings are supported; only absence (None)
    # means "no instructions". Later items override earlier ones.
    instructions_item: Optional[Instructions] = None
    if isinstance(instructions, str):
        instructions_item = Instructions(text=instructions)
    elif isinstance(instructions, Instructions):
        instructions_item = instructions
    if resume and save_path is None:
        raise ValueError("resume requires save_path")
    if max_samples is not None and (
        isinstance(max_samples, bool)
        or not isinstance(max_samples, int)
        or max_samples <= 0
    ):
        raise ValueError("max_samples must be a positive integer or None")
    if options is not None and not isinstance(options, SamplingOptions):
        raise TypeError("options must be SamplingOptions or None")

    resumed_existing_save = False
    if resume and Path(save_path).exists():
        context = load_interaction_save(save_path)
        resumed_existing_save = True

        # Restore the human-visible transcript as well as the model state.
        for display_item in render_interaction_items(context.items):
            print(display_item)
    else:
        if resume:
            print(
                f"Warning: no existing {Path(save_path).name} was found; "
                "a fresh one was created.",
                file=sys.stderr,
            )
            if prompt is None:
                prompt = DEFAULT_PROMPT
        if prompt is None:
            raise ValueError("prompt must not be None without resume")
        initial: list = [Init(model=initial_model_name(model))]
        if instructions_item is not None:
            initial.append(instructions_item)
        context = ModelContext(tuple(initial))

    if save_path is not None:
        save_interaction_save(save_path, context)

    def _persist() -> None:
        if save_path is not None:
            save_interaction_save(save_path, context)

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
    elif resumed_existing_save and prompt is None and instructions_item is None:
        final_text = _final_assistant_text(context)
        if final_text is None or not final_text.strip():
            raise RuntimeError("resumed save has no final assistant text")
        return final_text

    if instructions_item is not None and resumed_existing_save:
        # Append override after pending batch is valid again. Strict
        # tool-sequence validation requires no pending calls here.
        context.extend((instructions_item,))
        _persist()
        for display_item in render_interaction_items((instructions_item,)):
            print(display_item)

    if prompt is not None:
        # For a resumed save, add the follow-up only after any pending
        # tool batch has been made valid again.
        user_interaction = UserInteraction(
            items=(Message(role="user", text=prompt),),
        )
        context.extend(user_interaction.context_items())
        _persist()
        for display_item in user_interaction.display_items():
            print(display_item)

    sample_count = 0
    while max_samples is None or sample_count < max_samples:
        sample_count += 1
        sample = model.sample(
            context,
            tools=environment.tool_specs,
            options=options,
        )
        context.extend(sample.context_items())
        _persist()
        for display_item in sample.display_items():
            print(display_item)
        if sample.stop_reason == "compaction":
            # A paused Messages server compaction contains a durable
            # compaction block but no final assistant text. Replay it
            # immediately so the provider can continue the turn.
            continue
        if not sample.tool_calls:
            final_text = sample.last_assistant_text
            if final_text is None or not final_text.strip():
                raise RuntimeError("model returned no final assistant text")
            # Derive the cumulative end-of-turn usage and make it visible.
            # ``summarize_turn_usage`` skips existing ``TurnSummary`` items
            # so re-entering this path never double-counts, and
            # ``TurnSummary`` is encoder-transparent and durable.
            turn_summary = summarize_turn_usage(context.items)
            context.extend((turn_summary,))
            _persist()
            for display_item in render_interaction_items((turn_summary,)):
                print(display_item)
            return final_text

        result = environment.execute_tool_calls(sample.tool_calls)
        context.extend(result.context_items())
        _persist()
        for display_item in result.display_items(source_calls=sample.tool_calls):
            print(display_item)

    raise RuntimeError(
        f"model did not produce a final answer within {max_samples} samples"
    )


# Preserve the original public demo helper name for existing callers.
run_repository_summary = run


def _final_assistant_text(context: ModelContext) -> Optional[str]:
    """Return final assistant text if the effective context ends with it."""
    for item in reversed(context.model_items()):
        if isinstance(
            item,
            (
                ModelSampleBoundary,
                TurnMetadata,
                TurnSummary,
                UserInteractionBoundary,
            ),
        ):
            continue
        if isinstance(item, Message) and item.role == "assistant":
            return item.text
        return None
    return None


def _build_model(args: argparse.Namespace) -> Model:
    return build_model(args)


def _build_parser() -> argparse.ArgumentParser:
    return build_parser(
        "Ask a Chat Completions, Messages, or Codex Responses model to "
        "summarize a repository using Pythia's default local tools."
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    cwd = Path(args.cwd).expanduser().resolve()
    prompt = args.prompt
    if prompt is None and not args.resume:
        prompt = DEFAULT_PROMPT
    options = (
        SamplingOptions(max_tokens=args.max_tokens)
        if args.max_tokens is not None
        else None
    )

    print(
        "Warning: exec_command runs without a sandbox; use only with a "
        "trusted model and workspace.",
        file=sys.stderr,
    )
    try:
        save_path = resolve_save_path(args.save_path)
        model = _build_model(args)
        with DefaultEnvironment(cwd=cwd) as environment:
            run(
                model,
                environment,
                prompt=prompt,
                instructions=args.instructions,
                max_samples=args.max_samples,
                options=options,
                save_path=save_path,
                resume=args.resume,
            )
    except Exception as exc:
        print(f"demo failed: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
