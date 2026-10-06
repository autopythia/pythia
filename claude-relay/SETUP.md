# Set up and use Claude Relay

**Text, MCP handoff and Pi paths live-checked on CLI 2.1.289 / Opus 5.5.**
Ordinary tests use synthetic Python programs, not Claude. Opt-in tests and commands
marked **live** below are for your subsequent manual test: they execute the
already-installed, unmodified native CLI inside B's sandbox. Nothing here
requires the relay to download Claude, create accounts, or invoke sudo.

For upcoming effort/output-budget/Pi changes, see [SAMPLING.md](SAMPLING.md).
User catalog entries now support the Messages-style `output_config.effort`
object, including `max`; explicit generation/summary budgets remain blocked.
There is no independent model/effort default in the relay. Checks through the
running UID-1003 relay reached CLI 2.1.289. Later raw-relay Opus 5.5 generations
confirmed thinking without a thinking override, including with `--effort max`.
Omit the unsupported catalog `thinking` object to use those native defaults;
explicit type/display controls remain unmapped. Production-adapter text, MCP,
effort-change retirement and Pi tests now pass after native-protocol fixes.
Large-context compaction suppression and explicit output limits remain unverified.
The sampled deployment is not a new default; recheck native upgrades.

## 1. Deployment layout and prerequisites

| Account | Needs | Runs |
| --- | --- | --- |
| **A: harness user** | A trusted Pythia checkout/package and its copy of `claude_relay.py`; system Python 3.9+ and the shell/commands selected for host tools | Stdlib-only Pythia, native-protocol driver, MCP server/mailbox, relay client; ordinary host tool commands |
| **B: Claude user** | Same-version `claude_relay.py` and `one_click_claude.py` in one trusted directory; existing native Claude executable | Persistent broker, sandbox wrapper, Bubblewrap and Claude |

Both are **already-provisioned non-root accounts on the same Linux host**.
Start B's broker from B's login/session or an existing supervisor. An A-side
command cannot become B merely by selecting B's UID or putting it in a pathname.
The broker UID is always the UID that started the broker.

B's checked Debian/Ubuntu baseline is `python3 bubblewrap ca-certificates bash
coreutils git`, plus libraries required by the installed native binary. The OS
must permit unprivileged user/PID/mount namespaces (including applicable
AppArmor/SELinux/container policy). The wrapper checks and fails closed; it never
installs packages or changes host policy. No SDK, pip MCP package, uidmap, socat,
Docker, or extra network proxy is required by this implementation.

Arrange readable copies through your existing deployment mechanism. A need not
read B's real home or source binary; B need not import Pythia. Do not relax an
account's private-home permissions just to share the scripts. Root-owned or
account-owned trusted files are supported, with non-group/other-writable parents.
Keep scripts, Pythia code and control files **outside** B's writable Claude home
and outside workspaces exposed to model-driven file edits.

Repository layout:

```text
claude-relay/                       # standalone deployment executables
    claude_relay.py                 # client/broker/child guard; one file
    one_click_claude.py             # local sandbox launcher; one file
pythia/interaction/claude_relay/     # self-contained adapter package under A
    __init__.py                    # public ClaudeRelayEndpoint, ClaudeRelayModel
    _model.py
    _runtime.py
    _cli_protocol.py
    _mcp.py
    _context.py
    _sampling.py                   # narrow Messages-style effort policy
```

The adapter uses the harness's shared model/item/catalog types. Its internal
implementation is contained in that package; shared configuration and lifecycle
registration remain in the existing frontend modules.

## 2. Record A's UID, then prepare B's sandbox

In **A's existing session**, obtain the numeric UID to authorize:

```sh
/usr/bin/python3 -I -S -c 'import os; print(os.getuid())'
```

In **B's existing session**, set these to your actual deployment values:

```sh
export A_UID=1002                     # replace with the UID reported above
export RELAY_B_DIR="$HOME/claude-relay" # contains B's two trusted script copies
export B_UID="$(/usr/bin/python3 -I -S -c 'import os; print(os.getuid())')"
export RELAY_SOCKET="/tmp/claude-relay-$B_UID/control.sock"
```

B's defaults are:

- Native source: `$HOME/.local/bin/claude` (installation symlinks are supported).
- Writable Claude home: `$HOME/.local/share/one-click-claude/home`.
- Native working directory: that private home's `work` subdirectory.

Optionally set `ONE_CLICK_CLAUDE_BIN` and `ONE_CLICK_CLAUDE_HOME` **in B's session**
before all steps below. For repeatable testing, select a stable installed release
path rather than changing the source version during a broker session. Both paths
must satisfy the wrapper's checks; the source and launcher cannot live within
Claude's writable home. The entire account home is not mounted.

```sh
# No Claude binary required and no Claude execution:
/usr/bin/python3 -I -S "$RELAY_B_DIR/one_click_claude.py" --sandbox-doctor
/usr/bin/python3 -I -S "$RELAY_B_DIR/one_click_claude.py" --sandbox-print-home

# Requires the existing binary; inspects it and probes without executing it:
/usr/bin/python3 -I -S "$RELAY_B_DIR/one_click_claude.py" --sandbox-setup-only

# LIVE: run Claude inside the sandbox and complete its native login if needed:
/usr/bin/python3 -I -S "$RELAY_B_DIR/one_click_claude.py"
```

Authentication belongs to the CLI in this private home. Existing account-home
`.claude*` files/API keys are not automatically imported. Open any login URL
manually in your browser. The relay is headless; it is not the interactive-login
transport. Exit that interactive session before proceeding.

## 3. Start the broker as B

Still in B's session, with the same `ONE_CLICK_CLAUDE_*` selection:

```sh
/usr/bin/python3 -I -S "$RELAY_B_DIR/claude_relay.py" --relay-serve \
  --relay-socket "$RELAY_SOCKET" --relay-allow-uid "$A_UID"
```

Leave this foreground process running, or use your existing B-side supervisor.
Record **B_UID** and **RELAY_SOCKET** for A. Startup itself does not execute Claude.
The default wrapper is the sibling `one_click_claude.py`; `--relay-wrapper` is a
B-side operator override, not a client-controlled remote command.

The broker creates its control directory as B, normally mode 0711, and a mode-0666
socket. Mandatory `SO_PEERCRED` checks allow only the configured client UIDs;
mode 0666 is **not** public authorization. This location lets A traverse to the
socket without opening B's mode-0700 private home. Do not put the control endpoint
inside the sandbox-visible home/runtime. Existing unsafe directories/symlinks or
live listeners are refused; do not blindly delete or chmod someone else's socket.

Useful B-side flags:

- `--relay-max-active 8`: concurrent/reserved invocations, including version queries.
- `--relay-stop-grace 3`: seconds between termination and forced kill.
- `--relay-profile NAME`: non-secret deployment label shown by `--relay-check`.
- Repeat `--relay-allow-uid` only when intentionally granting another trusted user
  the same Claude profile's authority; this is not multi-tenant isolation.

## 4. Configure and check the client as A

Selecting a relay model in the catalog does **not** configure the relay connection.
Pythia reads explicit launch flags or the launching process's exported environment;
it does not automatically load `claude-relay/.env`, expand literal `$VARIABLE`
strings in settings, or restore connection settings from a saved conversation.
Export these values **before starting the CLI/auto process**, including when you
intend to select the relay model later with an interactive model switch. Exporting
in another terminal or after Pythia has started does not update that process.
The launcher is A's Python relay client, **not** the Claude binary or B's wrapper.

In **A's session**, use A's readable copy of the client and the values from B:

```sh
export PYTHIA_REPO=/absolute/path/to/A/trusted/pythia
export CLAUDE_RELAY_LAUNCHER="$PYTHIA_REPO/claude-relay/claude_relay.py"
export CLAUDE_RELAY_SOCKET=/tmp/claude-relay-1003/control.sock # replace with B's path
export CLAUDE_RELAY_SERVER_UID=1003                         # replace with B_UID
unset CLAUDE_CONFIG_DIR  # A-side config directories are intentionally unsupported

# Broker identity/protocol check only; DOES NOT execute Claude:
/usr/bin/python3 -I -S "$CLAUDE_RELAY_LAUNCHER" --relay-check

# LIVE: forwarded to the sandboxed native CLI under B, not a relay version:
/usr/bin/python3 -I -S "$CLAUDE_RELAY_LAUNCHER" --version
```

Set the exact leading version token just reported, for example:

```sh
export CLAUDE_RELAY_CLI_VERSION=2.1.289  # checked version; use your deliberately verified pin
export CLAUDE_RELAY_MODEL=sonnet        # example native CLI selector you can access
```

`CLAUDE_RELAY_MODEL` above is only a shell/example-API convenience: the relay and
adapter do not read it as a model-selection override. Pythia's `--model`, catalog,
and per-role selections remain authoritative; use a unique relay catalog selector
there if desired. The low-level Python endpoint example instead takes a native
model ID directly. No model is chosen by the broker and no Opus fallback is added.

This pin is a compatibility guard, **not** a statement that the version has been
verified. Repin deliberately after testing upgrades; do not auto-accept a changing
version on every launch. A does not set B's `ONE_CLICK_CLAUDE_*` paths. Those are
fixed by the already-running broker, and A's cwd is not forwarded to Claude.

## 5. Run Pythia text and tool smoke tests as A

The following shell helper starts the frontend with isolated system Python,
no site packages, and one explicitly trusted repository path. It avoids relying
on an editable pip installation, ambient `PYTHONPATH`, or the working directory:

```sh
pythia_run() {
  /usr/bin/python3 -I -S -c \
    'import runpy, sys; root, module = sys.argv[1:3]; del sys.argv[1:3]; sys.path.insert(0, root); runpy.run_module(module, run_name="__main__", alter_sys=True)' \
    "$PYTHIA_REPO" "$@"
}

export RELAY_WORKDIR="$HOME/claude-relay-smoke"
/usr/bin/python3 -I -S -c \
  'import os; from pathlib import Path; Path(os.environ["RELAY_WORKDIR"]).mkdir(mode=0o700, parents=True, exist_ok=True)'
```

These next commands are **live native tests**. Use fresh save paths, outside the
trusted code tree. `--resume False` explicitly replaces an existing CLI save;
auto refuses an existing save directory unless you request `--resume`.

```sh
# First: text only, no tools.
pythia_run pythia.interaction.cli --endpoint-api claude-relay \
  --model "$CLAUDE_RELAY_MODEL" --no-user-model-catalog --headless \
  --enable-default-tools False --prompt 'Reply with a short greeting.' \
  --cwd "$RELAY_WORKDIR" --resume False --save "$RELAY_WORKDIR/text-smoke.jsonl"

# Then: a safe, stdlib tool round trip; no supervising model yet.
pythia_run pythia.interaction.auto --endpoint-api claude-relay \
  --model "$CLAUDE_RELAY_MODEL" --no-user-model-catalog --headless \
  --watcher-max-resumes 0 --cwd "$RELAY_WORKDIR" \
  --prompt 'Use update_plan to record one completed smoke-test step, then reply.' \
  --save "$RELAY_WORKDIR/auto-smoke"
```

Headless mode checkpoints the conversation but is not a transcript renderer; inspect
those saves to verify the assistant reply and tool call/result records. After both
smokes pass, test supervising watcher use by omitting `--watcher-max-resumes 0` and
using another fresh save. Each role gets its own continuation and relay connection.

This backend uses the ordinary default tools: `exec_command`, `write_stdin`,
`update_plan`, `apply_patch` (plus bound auto-role tools). CLI's
`--enable-default-tools False` still disables them; auto's watcher still receives
only its watcher tools. Selecting Claude Relay does not narrow another role's
catalog. **Commands execute as A, outside B's Claude sandbox**, with the same
host authority as for other backends. `enable_workspace` restricts command cwd
and patch paths, not the filesystem authority of a shell command.

The harness, relay, MCP implementation and Python handlers remain stdlib-only;
this does not prohibit host command payloads from running ordinary external
programs. Custom Python callbacks/plugins must still satisfy that dependency
constraint. Claude's built-in tools remain disabled: the MCP mailbox returns
Pythia's real host results, and the whole harness is not sandboxed.

### Opt-in live regression checks

With the connection/version settings above, run these **billable native tests**
through the production Python adapter. They make several bounded short requests
with native output defaults, use an in-memory receipt tool and private temporary
saves, and test warm handoff, changed effort and Pi compaction. No shell tools,
downloads, account changes or broker restart occur. Use a model supporting the
tested `max` and `high` effort levels:

```sh
(
  export PYTHIA_TEST_CLAUDE_RELAY_LIVE=1
  export CLAUDE_RELAY_TEST_MODEL=claude-opus-5-5
  pythia_run unittest -v pythia_test.test_interaction_claude_relay_live
)
```

`CLAUDE_RELAY_TEST_MODEL` selects only this test's explicit binding; it is not a
backend model fallback. Without the opt-in flag, ordinary test discovery skips
all live tests. The heartbeat check deliberately withholds a harmless result
past the native heartbeat interval. These tests do not exercise native context ceilings
or prove auto-compaction suppression near them; see [SAMPLING.md](SAMPLING.md).

### Launch options

| CLI option | Environment fallback / default |
| --- | --- |
| `--claude-relay-launcher` | `CLAUDE_RELAY_LAUNCHER`; absolute trusted client path required |
| `--claude-relay-socket` | `CLAUDE_RELAY_SOCKET`; required |
| `--claude-relay-server-uid` | `CLAUDE_RELAY_SERVER_UID`; non-root numeric UID required |
| `--claude-relay-cli-version` | `CLAUDE_RELAY_CLI_VERSION`; explicit native version pin required |
| `--claude-relay-tool-id-pointer` | `/params/_meta/claudecode~1toolUseId` |
| `--claude-relay-generation-timeout` | 1200 seconds per generation wait for new endpoints |
| `--claude-relay-parked-timeout` | 1800 seconds waiting for a host tool result |
| `--claude-relay-startup-timeout` | 30 seconds for startup/version/input work |
| `--claude-relay-stop-timeout` | 5 seconds per process-stop wait |

These are launch-only settings, not native-state restoration instructions. Repeat
them (or retain the A-side environment) when resuming; auto does not restore them
from config.json. Use `--compaction-mode pi` if specifying compaction explicitly.
HTTP endpoint URLs/API keys/provider compaction are not supported. Effort is the
one supported native sampling override; explicit output-token budgets and other
sampling controls remain blocked pending their native mapping/verification.

### Select model and effort through Pythia's catalog

Use a unique entry in `~/.pythia/model-catalog.ini` (or an explicit
`--model-catalog FILE`), following the Messages API's object spelling:

```ini
[catalog]
version = 4

[model.opus55-relay-max]
endpoint.api = claude-relay
endpoint.model = claude-opus-5-5
limits.max_context_tokens = 1000000
limits.max_output_tokens = 128000
extra_sample_params.output_config = {"effort": "max"}
```

The capacity numbers above are the repository's native Opus 5.5 facts, not relay
ceilings or requested budgets. An automatic Pi threshold is separate host policy:
set `limits.auto_compact_context_tokens` or `--auto-compact-tokens` with headroom.
Choose the preset with `--model opus55-relay-max`; do **not** use the smoke command's
`--no-user-model-catalog` when loading your user catalog. No broker model override
or fallback is involved. Account/model support for an advertised native effort
level remains a live-test gate.

Use `output_config` as a JSON object; a literal INI key ending
`output_config.effort` is not a nested-field shortcut. The optional levels are
`low`, `medium`, `high`, `xhigh`, `max`. Other fields, including explicit API
`thinking` configurations, remain unsupported rather than silently discarded.

A launch `--extra-sample-params '{"output_config":{"effort":"high"}}'` overlays
the catalog. A per-call `SampleParams(extra={"output_config":{"effort":"high"}})`
replaces the extras map; `extra=None` inherits and `extra={}` clears it. Use an
empty `output_config` object to clear just that catalog/launch field. JSON `null`
is a literal invalid effort/object value here, not a special reset operation.

Equivalent effective effort reuses a waiting tool-chain process. A different
effort—or a new model, instructions, catalog or projected history—retires that
process before delivering results there, then starts from the authoritative
history. Pi uses this to run tool-free summaries and then a fresh compacted
continuation. See [SAMPLING.md](SAMPLING.md#what-restart-on-changed-settings-means)
for the precise handoff sequence and remaining output-budget work.

## 6. Python API

The public imports are unchanged by the package reorganization:

```python
from pythia.interaction.claude_relay import ClaudeRelayEndpoint, ClaudeRelayModel
# Also exported by pythia.interaction.
```

A minimal **live** library test, using the environment from step 4:

```sh
/usr/bin/python3 -I -S - <<'PY'
import os
import sys
sys.path.insert(0, os.environ['PYTHIA_REPO'])
from pythia.interaction import InteractionContext, Message
from pythia.interaction.claude_relay import ClaudeRelayEndpoint, ClaudeRelayModel

endpoint = ClaudeRelayEndpoint(
    model=os.environ['CLAUDE_RELAY_MODEL'],
    launcher=os.environ['CLAUDE_RELAY_LAUNCHER'],
    socket_path=os.environ['CLAUDE_RELAY_SOCKET'],
    server_uid=int(os.environ['CLAUDE_RELAY_SERVER_UID']),
    expected_version=os.environ['CLAUDE_RELAY_CLI_VERSION'],
)
model = ClaudeRelayModel(endpoint)
try:
    context = InteractionContext((Message('user', 'Reply with a short greeting.'),))
    print(model.sample(context, tools=()).last_assistant_text)
finally:
    model.close()
PY
```

For tools: pass the explicit `Environment.tool_specs` on each `sample()`, append
and checkpoint `sample.context_items()`, execute the proposed calls in the host
`Environment`, append/checkpoint its real `outcome.context_items()`, then sample
again. Never execute tools in the MCP callback or add a synthetic “continue” user
message. Keep the same model object through the chain; call `retire()` when
abandoning it and `close()` in `finally` on final shutdown.

## 7. Troubleshooting and stopping

| Symptom | Check / action |
| --- | --- |
| Launcher not configured / must be an absolute path | Set `--claude-relay-launcher` or export `CLAUDE_RELAY_LAUNCHER` to A's absolute `claude_relay.py` path, then start Pythia from that shell. The same error in older code also meant the value was missing. `.env` is not automatically loaded. |
| Permission denied connecting | A needs directory traversal to the socket; the broker must have authorized A's actual UID. Prefer the B-owned `/tmp` control directory, not B's private home. Do not weaken the Claude home permissions. |
| Unexpected socket owner/server UID | Confirm the broker really runs as B and A configured that numeric UID. A path label does not change identity. Never bypass the peer check. |
| No broker / stale socket / capacity rejection | Confirm the broker remains running and its active limit is adequate. No client auto-start/reconnect/replay occurs. Existing live listeners or unsafe socket paths are refused. |
| Missing source or unsafe source/home paths | Fix B's source/profile selection. The wrapper will not fetch Claude; code/native source must remain outside the writable home. |
| Namespace/mount failure | Run the doctor **as B**, using B's real service environment; package presence alone does not establish kernel/LSM/container permissions. |
| Authentication failure | Authenticate via the direct sandbox wrapper as B. Pythia `/login` and `/quota` are Codex services, not Claude login. |
| Version mismatch | Review the reported native version and compatibility before deliberately changing the pin. Freeze the source version during a broker session. |
| Native flags/records/inventory rejected | The initial profile may differ from the installed CLI or managed policy. Stop and inspect the difference; do not bypass the sandbox or required policy. |
| MCP 401/initialization failure | Verify native `${PYTHIA_CLAUDE_MCP_TOKEN}` header expansion and the supported HTTP/MCP subset. No literal secret goes in argv and authentication is never disabled. |
| Missing native tool ID | Verify the raw native metadata path. The default was observed with CLI 2.1.289. Never substitute JSON-RPC IDs or match by arguments/order. |
| File-backed config/system prompt required | Remote staging is not implemented. A cannot write B's 0700 home; do not share A's cwd or `/tmp` to work around it. |

### Diagnose a native stream failure

Safe failure records now include nested event identities (for example
`stream_event/error`), a bounded event tail, the true attempt event count,
structured error code and elapsed time. A timeout message includes the configured
generation budget and recent stdout-byte/record activity. The generation timeout
still waits for a completed model message; activity does not reset it.

For the next launch, add **`--debug-trace`** to CLI or auto. CLI writes
`SAVE_STEM.trace.events.jsonl`; auto writes
`SAVE_DIR/contexts/1.trace.events.jsonl` for main and the analogous sidecars for
other model-running roles. HTTP-backed roles also use `.trace.req.jsonl` /
`.trace.res.jsonl`. The flag must be repeated on resume and requires no broker
restart. It cannot recover the event missing from an older saved failure.

**These are sensitive, opt-in traces, not safe transcripts:** native
stdin/stdout/stderr and MCP JSON bodies are captured incrementally before parsing.
HTTP traces remain verbatim, including credentials. MCP Authorization headers
and environment variables are not deliberately dumped, but native/tool text can
still contain secrets. Logs are private (0600), append-only and can grow large;
unsafe existing targets are refused. Trace write failures warn without replaying
tools or model requests. No trace is used as conversation or resume state.

This observes A's relay pipes and MCP endpoint, not Claude's upstream HTTPS.
The library's bounded `model.stderr_tail` remains private; do not print it
automatically. See the [diagnostics/tracing design](../doc/interaction-diagnostics-tracing-plan.md)
for event correlation, lifecycle and verification details.

The traced heartbeat/subagent false positive is addressed by ignoring exact
`tool_progress` / boolean `heartbeat=true` records as telemetry. Parent/name/ID/
elapsed validation remains a TODO; this does not authorize tool execution or
result delivery. Ordinary progress, genuine subagent, native-error and terminal
guards remain in place. See the [compatibility notes](../doc/claude-relay-heartbeat-plan.md).
Relaunch the A-side frontend to load this fix; no broker restart is required.

If a native `tool_result` reports a timeout before Pythia releases its real reply,
the adapter invalidates that continuation instead of accepting the synthetic
outcome. For the recognized native timeout form, once Pythia has saved every real
host outcome and confirmed cleanup, the shared loop records the incident and
allows **one cold inference recovery per turn**. It does not rerun tool handlers
or discard A-side command-session handles. A genuine host tool timeout remains
its ordinary tool result. Unknown/ambiguous cases, missing outcomes, cleanup
failure, cancellation and repeated expiry are not given automatic retry permission.

This native MCP/tool deadline is separate from generation/parked deadlines;
raising the generation timeout does not necessarily address it. Native timeout
configuration is still unverified and unchanged. See the
[continuation recovery contract](../doc/claude-relay-continuation-recovery-plan.md).
Direct library callers not using `run_turn` receive `ModelContinuationExpired`
(a `ModelTransportError` subclass) and decide whether to call `sample()` again
with the same authoritative history. Relaunch the A-side frontend to load the
change; the broker needs no restart.

Ctrl-C/termination of a client retires its invocation; loss of A's driver or B's
broker also triggers child cleanup. To shut down the service, stop B's broker
normally (e.g. Ctrl-C in its session). This terminates active invocations but does
not delete the private Claude home/authentication or A's saved conversations.
Restart the broker under B explicitly; recovery starts cold and must not replay
uncertain host tool effects. Do not change profiles while a live chain is running.
