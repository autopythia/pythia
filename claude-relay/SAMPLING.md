# Claude Relay: effort, output budgets, and Pi compaction follow-up

Status: **native text/MCP/Pi paths live-checked on 2.1.289; output budgets remain TODO**.
The adapter now resolves Messages-style `output_config.effort`, forwards native
`--effort`, includes effective effort in continuation identity, and preserves or
rejects native stop reasons rather than relabeling incomplete text as normal end.
The v2 native codec handles per-block assistant snapshots and narrow system
telemetry. Fixture and opt-in live tests exercise real Pi compaction and parked
continuation retirement; truncated-summary refusal remains fixture-tested.
Explicit generation/summary budgets still fail before
launch until their native transport/enforcement is verified. No broker settings,
model default, native capacity metadata or running service were changed.

## 1. Decisions that do not change

- All A-side integration code remains Python stdlib-only. The broker remains a
  thin stdlib process/stdio service under pre-provisioned UID B, not an SDK worker.
- The dependency constraint covers the Python implementation, not model-requested
  command payloads. Normal `exec_command`/`write_stdin` host tools are available
  under A, outside B's Claude sandbox, without adding integration dependencies.
- Pythia's catalog, command-line/model binding, per-role selection, and per-call
  parameters choose the model/settings. **No relay model fallback, Opus default,
  effort default, extra environment-based model selector, or automatic downgrade.**
- A user can explicitly select Opus 5.5 with max effort. No new built-in preset is
  required to support a user catalog, and existing Messages presets must not be
  silently reinterpreted as CLI settings.
- Model capacity metadata is the native Claude model's data, shared across
  transports—not smaller relay-specific ceilings or the CLI's current default
  request budget. This patch leaves those values unchanged.
- Pythia owns compaction through `PiCompactor`. Claude-native compaction remains
  requested off and unexpected internal compaction must invalidate the run.
- Native source acquisition, login, account management, package installation and
  privilege escalation are out of scope. Test through the already-running relay;
  do not invoke the native binary on A or bypass the sandbox.

## 2. Current live evidence and remaining unknowns

The current process environment had no `CLAUDE_RELAY_*` settings exported.
The non-secret entries in the existing `claude-relay/.env` identify the endpoint;
the file was read as data, not sourced/executed or modified.

| Checked through the existing relay | Observation |
| --- | --- |
| Socket and owner | `/tmp/claude-relay-1003/control.sock`, UID 1003 (`claude-dev`), matching the supplied connection settings |
| `--relay-check` | Successful authenticated health query; protocol version 1, profile `default` |
| Wrapped `--version` | `2.1.289 (Claude Code)`, matching the configured native version pin |
| Wrapped `--help`: effort | `--effort <level>` advertises `low`, `medium`, `high`, `xhigh`, `max` |
| Wrapped `--help`: model | Native latest-model aliases such as `fable`, `opus`, `sonnet`, or a full model name |
| Wrapped `--help`: compaction | `--autocompact <auto\|tokens>` sets a 100k–1M window; it is not documented there as an off switch |
| Wrapped `--help`: output budget | No `--max-output-tokens` flag or `CLAUDE_CODE_MAX_OUTPUT_TOKENS` description in help |
| Wrapped `--help`: bare mode | `--bare` skips OAuth/keychain reads and requires API-key/apiKeyHelper or third-party-provider auth; do not use it to simplify this home-login deployment |

The initial checks above were bounded health/version/help queries only. They **did execute the
existing CLI under B for version/help**, following authorization to use the live
relay, but made no model-generation request. No Claude download, explicit login,
broker restart or live configuration change was performed.

Help establishes a parser surface, not effective model behavior. Subsequent
generation evidence is recorded below. Native output-limit mechanism/range,
error/limit records, correct adapter token accounting, suppression of independent
compaction, and complete pending MCP round trips still require verification.
Earlier fixture tests are not evidence for these native properties.

### Subsequent low-risk implementation check

After implementing the effort resolver/argv mapping and stop handling, one bounded
live request used the existing relay with explicit `claude-opus-5-5`,
`output_config.effort=max`, no tools, and a request to reply only `OK`. It failed
closed in about 2.7 seconds at the **existing** unexpected-system-event guard after
two system records; no assistant result was accepted and the continuation was
retired. Native stderr was empty. No broker restart/configuration change, fallback,
binary download, or output-budget request was made.

That initial failure did not verify effective effort, native capacity enforcement,
or generation/MCP compatibility. The startup mismatch was addressed in the
subsequent adapter follow-up below. Effort precedence, forwarded argv,
continuation retirement and real `PiCompactor` behavior are verified with synthetic
fixtures, including failure to install truncated summaries.

### Native thinking-default probes (CLI 2.1.289)

Later bounded probes used a stdlib-only observational stream reader through the
same relay and sandbox, **not the production Pythia adapter**. Initialization
reported the `opus` entry as **Opus 5.5**, with `supportsAdaptiveThinking=true`,
`supportsEffort=true`, and effort levels `low/medium/high/xhigh/max`.

Two small arithmetic requests explicitly selected `claude-opus-5-5`: one omitted
both effort and thinking overrides; the other requested `--effort max` and still
omitted thinking. Both returned the correct answer, `end_turn`, a successful final
result, exit code 0 and empty stderr. Both emitted thinking blocks and native
`system/thinking_tokens` telemetry. The max-effort request's final streamed usage
reported 546 output tokens for a three-digit answer. Its thinking block had a
signature but **empty visible thinking text**; no reasoning text or signatures
are recorded here.

These observations establish **thinking on without an explicit thinking setting**
in this deployment, including when requesting max effort. The model capability
report establishes adaptive support, not that adaptive is the only possible mode
or the exact outbound API `thinking.type`. Likewise, accepting `--effort max`
and generating successfully is not an independent echo of effective API effort.
Public help did not advertise a main-model thinking-display setting, and these
probes did not demonstrate `display=summarized` support.

For the commented user-catalog Opus/max entry, it is appropriate to omit the
unsupported `thinking` object and adopt the native defaults:

```ini
extra_sample_params = {"output_config":{"effort":"max"}}
```

This removes an unsupported explicit requirement; it does **not** enforce a
thinking type/display policy across future CLI versions or deployments. Keep the
native capacity metadata unchanged. The user catalog was not edited or enabled
by these probes. Explicit `thinking` remains rejected rather than silently ignored.

Those probes identified the following **production-adapter issues, now fixed**:

1. Recognize the observed benign `system/status` value `requesting` and
   `system/thinking_tokens` telemetry without accepting arbitrary system events,
   hooks or native compaction.
2. Handle native `assistant` records that contain individual blocks, share one
   message ID, have null stop reasons and retain provisional usage. The max probe
   emitted separate thinking-only and text-only records for one streamed message;
   the initial assembler incorrectly expected complete-message snapshots.
3. Use completed stream boundaries and final message-delta usage for authoritative
   message completion/accounting. In that probe, assistant records reported four
   output tokens while the final stream reported 546. Do not treat provisional
   records as completed outputs, count them twice, or discard genuine conflicts.

Those observational probes left the production guards and assembler unchanged;
raw-relay success alone was **not an end-to-end Pythia/MCP compatibility claim**.
The subsequent implementation and adapter-path tests are recorded below.

All A-side probes ran under `/usr/bin/python3 -I -S`, with no SDK/dependencies.
Generation used print/stream-json with partial messages, empty built-in tools,
strict empty MCP configuration, `dontAsk`, empty settings sources, disabled slash
commands and no session persistence. Existing compaction-disable settings were
requested; these short prompts do not verify suppression at large windows.
No model tool calls, downloads, authentication changes, broker restarts, runtime
code changes or deployment changes were made.

### Production-adapter follow-up: implemented and live-checked

The `claude-stream-json-v2` codec now reconciles bounded shadow block snapshots
against their completed stream. Null-stop snapshots cannot emit a message,
complete a tool batch, or replace final usage. Explicit complete-record conflicts
still fail. Only the observed `status=requesting` and well-formed
`thinking_tokens` telemetry were added to the system-event allowlist; compacting,
compact boundaries, hooks and unknown events still retire the run.

The first live MCP attempt exposed an additional startup negotiation: Claude
sent `server/discover` with protocol `2026-07-28`, then fell back to standard
initialization. The server now rejects **that optional pre-initialization probe**
with JSON-RPC `-32601` without retiring the mailbox. It does not advertise or
implement the newer protocol. The observed fallback negotiated **2025-11-25**.

All four opt-in tests in `pythia_test/test_interaction_claude_relay_live.py` passed
through the actual `ClaudeRelayModel`, existing UID-1003 broker, sandbox and
unmodified CLI **2.1.289**, explicitly selecting **claude-opus-5-5**:

| Live check | Verified observation |
| --- | --- |
| Text, no effort override and inherited max | Correct arithmetic answers, `end_turn`, positive per-message usage, normal retirement |
| MCP tool and warm continuation | Bearer env expansion authenticated; native `_meta["claudecode/toolUseId"]` correlated; callback waited while the host executed and saved its receipt; equivalent effective effort reused the same process |
| Changed effort, max to high | Old process retired before results were released there; new process cold-imported the saved result and finished without executing the host handler again |
| Pi with a low host threshold | Actual tool-free summary generation inherited max effort, retired the parked tool process, produced a valid prefix, retained the receipt, and finished through a fresh post-compaction process without effect replay |

The live tool was an in-memory receipt generator, executed by Pythia's ordinary
`Environment`, not by the MCP callback. Saves were private temporary files on A.
No shell/filesystem tools, downloads, logins, broker restarts or configuration
changes were used. The tests ran under `-I -S` with no third-party imports.
Only comments changed in the standalone broker/wrapper; deployment is unchanged.
At that adapter-follow-up stage: **1004 tests, 9 skipped** (including the four
opt-in live tests); all four native tests also passed separately on that revision. Both runs
used isolated system Python and imported no site/dist-packages modules.

The subsequent host-tool policy clarification restores the ordinary CLI/auto
host command tools without changing the native protocol or B-side profile.
Scripted MCP tests now exercise an actual A-side `exec_command`/`write_stdin`
session through both frontends, plus mixed-provider auto catalogs and the normal
tool opt-out/role boundaries. Its isolated-stdlib regression run passed **1011
tests, 9 skipped**, with no third-party imports. These additional checks used
synthetic native fixtures, not new live Claude requests.

**Evidence limits:** short live runs request `DISABLE_AUTO_COMPACT=1` and observed
no native compaction. They do not establish its behavior near the native context
threshold or independently echo the effective setting. Compaction rejection and
truncated-summary refusal are covered offline. Native capacity ceilings, explicit
output-budget enforcement and real auth/context/length-error variants were not
stress-tested or deliberately induced. Unknown records/errors still fail closed;
do not infer these properties from successful short requests.

Output-budget work remains explicitly marked `TODO(output-budgets)` in sampling,
runtime, frontend validation, broker and wrapper code. Requested generation and
summary caps still fail before launch; catalog capacity metadata is unaffected.
See [SETUP.md](SETUP.md#opt-in-live-regression-checks) to rerun the live tests.

## 3. Catalog and request semantics

### Effort: one narrowly supported extra

Use the existing profile-scoped catalog machinery; no catalog-format bump:

```ini
[catalog]
version = 4

# Supported effort configuration; explicit output-budget forwarding is still deferred.
[model.opus55-relay-max]
endpoint.api = claude-relay
endpoint.model = claude-opus-5-5
extra_sample_params.output_config = {"effort": "max"}
```

Select this entry through Pythia (`--model opus55-relay-max`, with an explicit
`--endpoint-api claude-relay` if a name is ambiguous), or the existing auto
per-role/catalog-selection facilities. The wire ID above is the one used in the
repository's native model metadata and succeeded in the adapter tests above;
access and compatibility in other versions/deployments still require live checks.
Use the same native capacity facts in the catalog, not separate lower relay
limits. `limits.auto_compact_context_tokens` is Pythia policy, not model capacity.

Use the **same object shape as Messages**, not a new flat `effort` convention:
`{"output_config": {"effort": "max"}}`. In INI, the line above supplies the
`output_config` JSON object. A literal key `extra_sample_params.output_config.effort`
is not a nested-field shortcut in the existing catalog parser and is rejected.
The initial relay schema accepts only `output_config` and only its `effort` member:

- Exact strings `low`, `medium`, `high`, `xhigh`, `max` are candidates supported by
  this CLI's advertised syntax. Validate model/version applicability in the live
  matrix; help alone does not prove all combinations.
- Missing effort (including `output_config={}`) means no explicit CLI flag. Clear
  a launch overlay with `{"output_config":{}}`, or clear the entire per-call map
  with `SampleParams(extra={})`. This does not create a Pythia effort default.
- JSON `null` remains a literal value, consistent with Messages extras, not an
  inheritance/clearing operator. Null `output_config` or `effort` is rejected by
  this narrow adapter. `SampleParams.extra=None`, in contrast, means inherit.
- Reject other keys/types, spelling/case variants, arbitrary argv/env payloads,
  and still-unmapped `thinking` / Responses `reasoning` controls. Matching the
  effort convention is not blanket API-body forwarding: explicit thinking mode,
  display or thinking-token budgets need their own verified native controls.
- Do not add independent `CLAUDE_RELAY_EFFORT`/model override authority in the
  broker. Effort belongs to the resolved Pythia sampling policy.

Keep existing precedence exactly:

1. Catalog `extra_sample_params` supplies defaults.
2. Launch `--extra-sample-params` overlays the binding's map as it does today.
3. `SampleParams.extra is None` inherits that resolved binding. A supplied map
   **replaces** it; `{}` clears extras rather than re-inheriting catalog effort.
4. Resolve and validate to an immutable effective policy before native launch or
   releasing any pending result. Equivalent effective settings should compare
   equal regardless of which input layer supplied them.

Do not merge per-call extras with catalog extras, lower an unsupported effort,
rewrite a Messages preset into a relay preset, or switch the selected model.

### Output limits: requested budget versus capacity

| Field | Meaning |
| --- | --- |
| Catalog `limits.max_output_tokens` | Declared model capacity metadata; not a default request budget |
| `SampleParams.max_output_tokens` | Typed positive integer/None request for each model generation |
| Frontend `--max-output-tokens` / runtime configuration | Existing Pythia source of that per-call request |
| `--compaction-max-output-tokens` | Override for Pi summary calls; omission inherits the turn budget through `PiCompactor._summary_params` |

`None` sends no explicit native output cap. Never silently substitute a catalog
ceiling, hardcoded 128k, the max-effort setting, a timeout, a dollar budget, a turn
limit, or a stdout byte limit. The requested native cap must apply with its verified token
semantics (including reasoning if the native API counts it), not just visible text.

The leading candidate is the process environment variable
`CLAUDE_CODE_MAX_OUTPUT_TOKENS`, but help did not verify it. Confirm support,
accepted range, enforcement and failure/clamping behavior for the pinned CLI and
selected model before documenting it as supported. Keep the native model's
capacity metadata unchanged. If a CLI version, account or managed policy cannot
honor the requested native-model limit, report an explicit compatibility problem
rather than redefining the model's capacity or silently clamping the request.
Transport record/body byte limits are implementation safeguards, not token limits.

A small budget plus max effort may have special constraints. Test that combination
explicitly; do not secretly reduce effort or increase the budget to make it work.
Temperature, top-p, seed, custom stop sequences and other extras remain unsupported
in this iteration.

## 4. Adapter and transport changes

Keep new adapter implementation inside `pythia/interaction/claude_relay/`:

| Component | Change |
| --- | --- |
| `_sampling.py` | Implemented immutable effort resolution using `output_config.effort`; typed output-budget mapping/capability checks remain deferred |
| `_model.py` | Implemented narrow effort resolver, effective-policy continuation matching and true stop reasons; explicit budgets still rejected |
| `_runtime.py` | Implemented immutable effort policy and `--effort VALUE`; absent effort adds no flag; output-cap transport remains deferred |
| `_cli_protocol.py` | Preserve and reconcile authoritative model stop reasons/usage across partial/completed records; classify budget exhaustion and reject incomplete tool proposals |
| Existing `model_config.py` glue | Remove explicit output/summary-budget rejection only after the mapping is implemented; keep HTTP-only timeout/URL/auth restrictions |
| Catalog/setup examples | Add an explicitly selected relay max-effort example; retain API-scoped presets and no backend fallback |

Effort uses already-forwarded argv, so this low-risk patch needs no new broker
or wrapper environment key and no restart of the running broker. The later
output-cap environment change needs the separate deployment/capability work below.
No second effort/budget default on the public endpoint or broker is needed. Resolve
from the binding and `SampleParams` once per call, then pass the effective object
to Runtime. Do not just remove checks and continue launching the old argv/env.

Use `--effort` for effort on this CLI: argv is already transported unchanged.
For an environment-based output limit, update **both** standalone scripts' narrow
allowlists and validate canonical positive-integer strings at the relay boundary.
Do not accept arbitrary environment overrides or inherit a B-side ambient output
cap as a hidden default. The value is per invocation, not a broker-wide setting.

### Prevent silent loss through older deployed copies

The current client filters forwarded env and the wrapper filters sandbox env. An
updated adapter alone can therefore appear to set a limit that an old copy drops.
Add explicit transport capability negotiation for policy-bearing env settings:

1. Add a read-only `--sandbox-capabilities` management operation to the fixed
   wrapper. It reports a schema/revision and forwardable setting names, not
   credentials, home contents or native model capabilities. It never runs Claude.
2. The broker queries its operator-selected wrapper using a fixed management
   invocation, validates the bounded reply, and advertises the effective
   intersection of broker/wrapper settings in HELLO/`--relay-check`.
3. The A driver/client checks required capabilities before START whenever a new
   control such as the output-cap env is requested. Missing/legacy capability
   metadata must cause a clear refusal for that control, not silent filtering.
   Existing no-budget invocations may remain compatible with the legacy relay.
4. Pin compatible A/B script revisions. The running broker caches Python code;
   updating a file is not a hot reload. Coordinate updating B's broker AND wrapper
   and restarting B's service before the new env-based feature's live test. Do not
   restart the currently running service as a side effect of a sample or this plan.

Capability advertisement proves the transport path, **not native enforcement**.
Retain a live native test for the actual setting. Add the new metadata without
polluting CLI stdout or granting clients arbitrary wrapper management commands;
normal requested argv stays behind the broker-inserted `--`.

If a different native output-control mechanism proves necessary, specify and test
that mapping explicitly instead. Do not smuggle it through an unverified settings
blob merely to avoid updating the allowlists.

## 5. Stop reasons and output-limited completions: fix first

Implemented: the adapter no longer replaces every stop with `tool_use` or
`end_turn`. It preserves supported native reasons, normalizes `length` to
`max_tokens`, requires a known reason, and rejects incomplete/contradictory tool
batches before exposing calls. Duplicate/streamed/completed stop or usage conflicts
are errors. Pi also rejects the `length` and `pause_turn` aliases generically.
This is independently useful even with native default budgets.

- Preserve/normalize verified native reasons: normal end, tool handoff, output
  limit, refusal/other failure. Native `max_tokens`/`length` must not become
  `end_turn`. Reconcile partial/completed-record reason and usage consistently;
  do not deduplicate away the only authoritative completion information.
- A missing/unrecognized reason is not automatic evidence of normal completion.
  Define any fallback only from the observed pinned protocol and terminal result;
  otherwise fail closed when completion safety is uncertain.
- Do not emit executable `ToolCall`s from malformed/unfinished tool JSON or a
  length-limited/incomplete tool batch, even if an earlier block parsed. Preserve
  safe completed text/reasoning diagnostically if appropriate, not invented tool
  outcomes or success.
- For a complete text-only limited response, expose the actual incomplete reason
  through `ModelSample`, or raise a structured `ModelError` where the native run
  cannot be represented safely. Do not truncate captured stdout to simulate a
  token limit; doing so corrupts framing and cannot count hidden reasoning.
- Check whether Claude automatically retries/continues with a higher budget on a
  length stop. Do not permit that to bypass Pythia's requested per-generation
  policy. If needed, retire at the observed limit boundary and report explicit
  incomplete output; this intentional abort is not a fabricated successful run.
- Pi must reject every truncated/failed summary and install no `ContextPrefix` in
  that case. Preserve the old effective history and audit log for explicit recovery.

Version the native compatibility profile if record interpretation changes. Record
recognized structured context-window/authentication errors without guessing from
arbitrary model text; a missing final-result record remains a transport error
unless the adapter explicitly terminated an already-identified limited completion.

## 6. Pythia-managed compaction and continuation identity

Keep `auto_compaction_owner = "host"`, default `--compaction-mode pi`, and rejection
of provider compaction. Forward the existing `DISABLE_AUTO_COMPACT=1` and verify its
effect for this native version. `--autocompact auto` or a large native window is
**not** a replacement for disabling independent native compaction. Do not use
`--bare`: the observed help says that would change/break this OAuth login flow.

The automatic trigger remains the existing Pythia policy:

- Enabled flag AND an explicit/catalog threshold are required. A
  `limits.max_context_tokens` ceiling alone does not currently derive a trigger.
- `limits.auto_compact_context_tokens` seeds the threshold; CLI/runtime
  `auto_compact_tokens` overrides it according to existing config precedence.
- Use verified per-model-message usage as the estimate anchor, then estimate newly
  appended content. Never add cumulative CLI run totals to each sample.
- Select a conservative threshold below the **effective native** context window
  minus output reservation and overhead margin. Cold-import JSON, system text,
  MCP schemas, retained reasoning and native overhead matter. Preserve the native
  capacity facts; any additional deployment restriction is a separately reported
  compatibility constraint, not a lower model-capacity value invented by the relay.

Include `ResolvedSampling` in the continuation signature along with endpoint /
model / pinned profile, projected instructions, and the current catalog. A change
in effective effort or budget requires retiring the old process before releasing
any result, then cold-importing the authoritative updated history. A process env
or startup flag cannot be changed by pretending an old continuation was updated.

### What "restart on changed settings" means

The state being replaced is a native process, not Pythia's saved conversation.
There is currently no verified in-place way to change its startup effort/budget
or replace its hidden context. Compare **resolved** settings, not the syntax or
source of an override:

```text
sample(H, effort=max) -> complete tool batch T; CLI #1 waits for MCP results
Pythia executes/checkpoints T's actual results R

sample(H + T + R, effort=max)  -> release R to CLI #1; retain its continuation
sample(H + T + R, effort=high) -> stop CLI #1 BEFORE releasing R there;
                                cold-launch CLI #2 with H + T + R and effort=high
```

Catalog max inherited via `extra=None` and the same max supplied explicitly are
equivalent and keep the process. Clearing to a missing effort is a real setting
change and restarts it. A changed model/profile, instructions, tool catalog or
projected history likewise retires the old continuation. Metadata-only audit
changes do not. Output-budget changes will use the same rule once supported.

Cold import includes completed tool outcomes as history, not proposals to execute
again. If historical calls still lack outcomes, a cold launch is refused; the
harness must finish/checkpoint them or explicitly record unavailable/not-executed
outcomes. Retirement does not cancel an already-running host side effect. There
is no adapter-driven replay; as with any model, later *new* proposals remain
subject to the host's ordinary tool policy and are not an exactly-once guarantee.

Pi already calls the same model with a summary-only `ContextPrefix`, `tools=()`,
and `enable_auto_compaction=False`. Preserve that integration:

1. Pythia checkpoints all actual tool outcomes before the next sampling decision.
2. If Pi runs while the native process is waiting on MCP results, retire/revoke
   that parked process; do not feed it results after the context has changed.
3. Cold-launch the tool-free summarizer with inherited effort and Pi's explicit
   summary budget, or the turn budget if no summary override exists.
4. Validate the summary's completion before installing summary + retained tail.
5. Cold-launch the next normal continuation from that new projected context with
   the normal tool catalog and request policy. The adapter does not replay old effects.

Existing structured context-window errors may trigger the harness's bounded
compact-and-retry path. Verify native error classification; do not catch every
model/transport error and label it context overflow. Document the additional cold
launches/usage during compaction, rather than retaining hidden pre-compaction CLI
state. The original durable history remains unchanged by projection.

## 7. Implementation order and acceptance tests

### A. Offline correctness and policy first

1. Fix stop-reason/truncation propagation and add synthetic complete-text, capped
   text, partial-tool JSON, capped-tool-batch, refusal, and late-error fixtures.
   Assert Pi never installs a truncated summary or executes incomplete proposals.
2. Implement `_sampling.py`; table-test catalog + launch overlay + per-call
   replacement, empty-map clearing/null rejection, positive-int budgets, invalid types/levels,
   unknown extras, unsupported `thinking`/other `output_config` fields, and unchanged native capacities.
3. Add model/runtime effort mapping and narrow output-cap transport support with
   capability checks. Test old A client, old B broker, old wrapper, unavailable
   capability metadata, malformed settings and missing fields: no silent drop.
4. Test warm reuse for identical effective settings and cold retirement for each
   changed setting, including during a parked handoff. All already-executed results
   remain in Pythia history; no callback/launch/effect replay or late result leakage.
5. Exercise real `PiCompactor` on fixture CLI responses, not merely a generic
   changed-context test: threshold trigger, explicit summary budget, inherited
   effort, tool-free summary calls, retained tail, failed summary rollback,
   output-limit failure and recognized context-overflow recovery.
6. Cover user-catalog API inference/selection, explicit per-role choices, CLI/auto
   config and runtime budget changes. Keep HTTP backends unaffected; no model
   fallback/built-in preset or meaning change for catalog capacity metadata.
7. Run the suite under `-I -S` with only the trusted repo added to the path; audit
   transitive imports and helper processes. No SDK/MCP dependency or binary fetch.

### B. Live tests through the already-running relay

Use its configured endpoint and expected UID, not a local native executable.
Start with no-tools, harmless prompts and bounded startup/generation/stop waits.
Avoid workspace mutation, shell tools, external side effects or large-context load
until the small tests pass. Record CLI/script/profile versions and safe scalar
observations, not credentials, full prompts, or raw sensitive diagnostics.

1. Recheck health/version and the B-side profile. The current planning baseline is
   protocol 1 / UID 1003 / CLI 2.1.289; do not silently change that pin or assume the
   service has been restarted after source edits.
2. Establish a baseline text-only invocation through the existing adapter before
   testing new controls. Native flags, init/inventory, JSON record boundaries and
   successful final-result/EOF handling must actually work.
3. Select Opus 5.5 explicitly and request max effort. Verify model access and the
   effective setting where the native protocol exposes it. An accepted flag is
   not proof of effective effort; do not infer it from latency or answer length.
   Test an unsupported model/effort request rejects instead of silently downgrading.
4. After coordinated B-side deployment, verify capability advertisement and test
   two small supported output budgets on a controlled bounded-output prompt.
   Observe native stop reasons and per-message counts, including thinking if
   applicable. A naturally short answer alone does not prove the cap was applied.
   Determine range/clamping/auto-retry behavior without an unbounded stress prompt.
5. Exercise max effort together with an output budget, plus the unsupported/too-
   small/too-large cases. Keep settings fixed for a short two-call tool chain;
   separately test changing them at a host handoff without replaying results.
6. Use only safe stdlib tools (e.g. echo/update_plan) to prove the actual native-ID
   MCP round trip, bearer expansion, and no continuation while results are withheld.
7. Force one inexpensive Pi compaction with a deliberately low host threshold and
   a small disposable text history. Prove an actual summary request, valid prefix
   installation, a fresh post-compaction continuation, and correct usage accounting.
   Do not drive the live model toward its maximum context window just to trigger it.
   This small test does not by itself prove native auto-compaction is disabled at
   larger windows; verify supported configuration/diagnostics and retain fail-closed
   detection, with any larger validation explicitly budgeted separately.
8. Repeat a summary with an intentionally insufficient supported budget and confirm
   no truncated summary is installed. Exercise teardown on timeout/interrupt.

Stop on any incompatible native behavior. If configuration observability is absent,
record what was merely requested versus what was demonstrated; do not label a
candidate setting supported just because the fixture or help text accepted it.

## 8. Completion criteria and deferred scope

Done means an explicitly selected user-catalog Opus/max entry reaches the verified
native control, requested generation and summary budgets have faithful semantics,
Pi owns compaction with safe stop handling, policy changes cannot resume stale
CLI state, and incompatible deployments/settings fail visibly. No new model
fallback, global broker sampling policy, SDK dependency, replay or filesystem grant.

Deferred: raw thinking-token budgets/display controls, temperature/top-p/seed/stop,
other API extras, automatic context-window discovery, generic filesystem staging,
new MCP features, cross-turn native-session reuse, and changes to unrelated model
profiles. If any deferred feature is required by the observed native interface,
stop and revise its contract rather than disguising it as effort/budget support.
