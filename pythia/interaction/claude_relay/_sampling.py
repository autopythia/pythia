"""Resolve only verified relay controls, using Pythia's ordinary extra-map precedence.

Catalog spelling follows Messages: {"output_config": {"effort": "max"}}.
No API request body is passed to Claude Code, and no native limit metadata is
rewritten. Explicit output budgets remain unsupported until their transport and
native enforcement have been tested.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from ..model import ModelConfigurationError, SampleParams

# Advertised by the wrapped CLI 2.1.289 help; per-model behavior is a live-test gate.
EFFORT_LEVELS = frozenset(('low', 'medium', 'high', 'xhigh', 'max'))


@dataclass(frozen=True)
class ResolvedSampling:
    # TODO(output-budgets): add the verified native cap here so a changed budget
    # retires a parked continuation, just like effort. Keep model capacity facts
    # separate; no implicit cap, silent clamping, or unverified env forwarding.
    effort: str | None = None

    def cli_args(self) -> list[str]:
        return [] if self.effort is None else ['--effort', self.effort]


def resolve_extra(extra: Mapping) -> ResolvedSampling:
    if not isinstance(extra, Mapping) or set(extra) - {'output_config'}:
        raise ModelConfigurationError('Claude Relay supports only extra_sample_params.output_config.effort')
    output = extra.get('output_config', {})
    if not isinstance(output, Mapping) or set(output) - {'effort'}:
        raise ModelConfigurationError('Claude Relay output_config must be an object containing only effort')
    if 'effort' not in output:
        return ResolvedSampling()
    effort = output['effort']
    if not isinstance(effort, str) or effort not in EFFORT_LEVELS:
        raise ModelConfigurationError('Claude Relay effort must be low, medium, high, xhigh, or max')
    # Null is a literal extra value, not a clearing operator. Clear with a per-call
    # extra={} or with output_config={} in a catalog/launch override instead.
    return ResolvedSampling(effort)


def resolve_sampling(binding, params: SampleParams | None) -> ResolvedSampling:
    if params is not None and not isinstance(params, SampleParams):
        raise ModelConfigurationError('Claude Relay requires SampleParams or None')
    params = params if params is not None else SampleParams()
    if params.max_output_tokens is not None:
        # TODO(output-budgets): map generation AND Pi-summary budgets only after
        # native enforcement/range/stop semantics and transport capabilities are
        # verified (claude-relay/SAMPLING.md). Until then fail before launching.
        raise ModelConfigurationError('Claude Relay explicit output budgets await verified native enforcement')
    if any(getattr(params, key) is not None for key in ('temperature', 'top_p', 'seed')) or params.stop:
        raise ModelConfigurationError('Claude Relay does not support temperature, top_p, seed, or stop overrides')
    if params.request_timeout_seconds is not None:
        # Relay sends no HTTP model request; its own deadlines are endpoint settings.
        raise ModelConfigurationError('Claude Relay does not use request_timeout_seconds; '
                                      'configure timeouts.generation_seconds or --claude-relay-generation-timeout')
    extra = binding.extra_sample_params if params.extra is None else params.extra
    return resolve_extra(extra)
