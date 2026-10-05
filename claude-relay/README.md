# Claude sandbox launcher and thin cross-UID relay

## Status: text, MCP handoff and Pi live-checked on CLI 2.1.289

Both standalone files are implemented and use only Python stdlib:

- [`one_click_claude.py`](one_click_claude.py): existing rootless Bubblewrap wrapper.
- [`claude_relay.py`](claude_relay.py): client/broker, mutual peer-UID checks,
  descriptor passing, bounded stdio pumps, and parent-death/signal cleanup.

The adapter is contained in the
[`pythia.interaction.claude_relay` package](../pythia/interaction/claude_relay/),
with a small public `__init__.py` and private implementation modules. The
non-HTTP `claude-relay` binding, CLI/auto configuration and model cleanup are also
implemented. During initial implementation there was deliberately no available
Claude binary; tests used Python fixtures, synthetic ELF data and harmless system
programs, without downloading or executing Claude. A user-provided live relay is
now available. After fixing native system telemetry, per-block assistant records
and optional MCP discovery fallback, bounded **production-adapter** tests passed
through UID B=1003 / CLI **2.1.289** / **claude-opus-5-5**: text, authenticated MCP
handoff, effort-change retirement, and Pi summary/fresh continuation. See
[SAMPLING.md](SAMPLING.md) for exact evidence and remaining gates.

Thinking occurred without an override, including with requested max effort;
initialization advertised adaptive support, not its exclusivity. Omit the
unsupported catalog `thinking` object to use native defaults. Explicit output
budgets remain TODO/rejected; summarized thinking display, native capacity
enforcement and large-context auto-compaction suppression are not established
by these short tests.

### Implemented scope and limits

- Linux same-host headless transport, one operator-fixed profile per broker,
  mandatory non-root UIDs, no account switching, package installs or fallback.
- The broker uses bounded per-connection control/reaper workers (default 8), not
  the originally proposed single selector loop. Each worker owns one `Popen`;
  stdio data is pumped only by disposable clients. No thread-unsafe `preexec_fn`.
- The A driver launches the relay with fixed `/usr/bin/python3 -I -S`, supplies its
  parent PID, and the client installs a death guard before connecting. B's child
  guard independently covers broker death before the wrapper execs Bubblewrap.
- An **explicit expected native CLI version** is required. The experimental
  `claude-stream-json-v2` profile uses print/stream-json input/output, partial
  events, strict MCP configuration, empty built-in tools/settings sources,
  `dontAsk` permissions, and no native session persistence. Unexpected control
  operations, built-in tools, compaction, or inventory differences fail closed.
- The MCP endpoint implements stateless JSON-response Streamable HTTP with
  initialization, tools/list, tools/call and ping, using the explicitly supported
  protocol dates 2025-03-26, 2025-06-18 and 2025-11-25. GET/SSE and DELETE return
  405; no resources/prompts/sampling/tasks or generic JSON Schema validator.
  The observed optional pre-initialization `server/discover` probe receives
  `-32601`, allowing native fallback to a supported initialization protocol.
- Correlation defaults to JSON pointer
  `/params/_meta/claudecode~1toolUseId`, now observed in the live 2.1.289 callback.
  `--claude-relay-tool-id-pointer` can select another verified field
  under params._meta. Never substitute JSON-RPC IDs or guessed argument matching.
- Bearer delivery uses native MCP header environment expansion, live-checked on
  2.1.289: `${PYTHIA_CLAUDE_MCP_TOKEN}`. That one capability and
  `DISABLE_AUTO_COMPACT` were explicitly added to wrapper/relay allowlists. The
  literal bearer never goes in argv or saves. Failure to expand it must fail
  authentication; there is no unauthenticated fallback. Compaction suppression
  also requires native verification; observable compaction is rejected.
- Text-only cold import with visible reasoning text (no signatures/opaque state).
  Messages-style `extra_sample_params.output_config = {"effort":"max"}` now maps
  to native `--effort`; other output_config/thinking/Responses settings are not
  implicitly translated. Missing effort sends no flag. Explicit output budgets,
  temperature/top-p/seed/stop remain rejected. HTTP timeout is not a model deadline.
- Known native completion reasons are preserved (`length` becomes `max_tokens`).
  Missing/conflicting reasons and incomplete tool batches fail closed; Pi rejects
  truncated/refused/paused summaries. Native capacity metadata is unchanged—there
  are no smaller relay-specific context/output ceilings or new model defaults.
- No remote staging or client-side `CLAUDE_CONFIG_DIR`. Inline limits remain real
  OS argv limits; conversation input streams on stdin. If native compatibility
  requires files, implement the staging extension before enabling those options.
- CLI and auto main/worker use the ordinary host-tool catalog: `exec_command`,
  `write_stdin`, `update_plan`, `apply_patch`, plus any bound role tools. CLI's
  default-tools opt-out and auto's watcher-only tool boundary remain unchanged;
  choosing Claude Relay does not restrict other auto roles. Handlers use stdlib,
  but command payloads can run ordinary external programs **under A, outside
  Claude's sandbox**. Claude's own built-in tools remain disabled.

## Sampling and compaction follow-up

[See SAMPLING.md](SAMPLING.md) for the implementation plan covering explicit
catalog effort, requested output/summary budgets, true length-stop propagation,
and Pythia-owned Pi compaction. Effort and safe stop/continuation handling are now implemented and fixture-tested;
explicit output budgets still await native verification. The running CLI advertises `--effort` levels
`low/medium/high/xhigh/max`; the output-budget mechanism and effective behavior
still need live tests. Pythia remains the selection authority: no broker/relay
model fallback or Opus/effort default is being introduced.

The plan also elaborates reuse versus cold restart: identical effective policy
keeps the waiting CLI; changed settings retire it before publishing results and
start fresh from checkpointed history. Pi temporarily replaces the context/tools
for summarization, then the normal continuation cold-imports the compacted view.
The framework does not replay already-recorded effects.

The remaining budget plan includes transport capability negotiation so an older client/broker/
wrapper cannot silently drop a new output-cap setting. Coordinate B-side script
updates/restart before that feature's live test; the planning probes did not
restart or modify the running service.

## Setup and use

Both CLI and auto now support `--debug-trace` for relay runs. Private event
sidecars capture native/MCP traffic before decoding; normal failure records also
retain safe nested event identities, counts and timing without tracing. HTTP
provider traces remain separate and verbatim. See
[diagnostics setup](SETUP.md#diagnose-a-native-stream-failure) and the
[design/verification notes](../doc/interaction-diagnostics-tracing-plan.md).

Follow [SETUP.md](SETUP.md) for the complete A/B deployment procedure:

1. Identify A's UID and place trusted, same-version script copies where A/B can
   read their own copies; keep code outside writable Claude/tool workspaces.
2. As B, check prerequisites and inspect the existing native source with
   `--sandbox-doctor` / `--sandbox-setup-only`, then authenticate through the
   wrapped CLI when ready for a live test.
3. Start B's broker with a traversable control socket path and explicit A UID
   allowlist. No UID switch, sudo, package installation or broker auto-start.
4. As A, configure the relay path/socket/expected B UID, run `--relay-check`, then
   explicitly run the live native version query and select that version pin.
5. Run isolated-Python text and safe-tool smoke tests, then use CLI/auto or the
   stable `ClaudeRelayEndpoint` / `ClaudeRelayModel` Python API.

The guide includes the exact environment variables/options, source/private-home
layout, socket permissions, a runnable Python API example, shutdown/recovery, and
troubleshooting. It marks commands that execute Claude, including opt-in native
regression tests. Repeat those checks for new native versions/deployments.

### Tests

`pythia_test/test_claude_relay.py` covers transport/lifetime with a fixed Python
wrapper fixture. `test_interaction_claude_relay.py` covers the model, mailbox,
codec, configuration and full headless CLI/auto handoff through that broker.
`fixtures/claude_relay_cli.py` is explicitly **not Claude**. Wrapper tests remain
in `test_one_click_claude.py`. Use an isolated HOME to avoid unrelated user catalog
settings, and run A-side suites with site packages unavailable for the dependency
audit. `test_interaction_claude_relay_protocol.py` covers sanitized native record
shapes; `test_interaction_claude_relay_live.py` is skipped unless explicitly opted
in (see SETUP.md). The detailed design below records the contract and future gates.

## 1. Smallest useful architecture

**The A-side Python implementation is strictly stdlib-only.** Its Pythia adapter, direct native-CLI protocol
driver, relay client, HTTP MCP server, callbacks and helpers use only repo-owned
Python plus stdlib. No Agent SDK, Python `mcp`/FastMCP package, ASGI/HTTP framework,
third-party validator, optional dependency path, or dependency sidecar is allowed.
Do not move an SDK worker to B as a workaround: B's broker remains a stdlib process
launcher. See the integration plan for the direct CLI codec and minimal stdlib MCP
wire contract; this relay must stay independent of those protocol semantics.

This is a dependency constraint on the harness/relay implementation, **not a
restriction on model-requested command payloads**. Normal host shell tools may
launch external programs as A, with A's permissions, outside B's Claude sandbox.
That exception does not permit an SDK/MCP integration sidecar disguised as a
tool/helper; the protocol implementation and Python handlers remain stdlib-only.

| Role | Runtime dependencies |
| --- | --- |
| A: Pythia, protocol driver, MCP server, relay client, selected handlers/helpers | Repo-owned Python and Python stdlib only; no Claude binary, SDK, third-party MCP runtime, or helper sidecar |
| A: host command payloads | Ordinary shell/external programs selected by the tool caller; unsandboxed host execution, not integration dependencies |
| B: broker and sandbox wrapper | Python stdlib only |
| B: sandbox construction and payload | Preinstalled Bubblewrap, the unmodified native Claude binary, and its system runtime; never an A-side fallback |

```text
UID A: Pythia + stdlib CLI protocol driver + stdlib HTTP MCP mailbox
                    |
          stdlib subprocess -> claude_relay.py (client mode)
                    |
          authenticated pathname UNIX control socket
          + exactly three anonymous pipe ends via SCM_RIGHTS
                    |
UID B: persistent claude_relay.py broker (already started as B)
                    |
          one guarded child per client invocation
                    |
          one_click_claude.py -- <unchanged Claude argv>
                    |
          Bubblewrap -> unchanged native Claude executable
                    |
          host networking -> A's authenticated loopback MCP listener
```

The broker persists across CLI invocations; a native process persists only for
its owning invocation/tool chain. Version probes get separate short invocations.
Concurrent main/watcher/worker driver instances get separate children and streams,
not one shared native session. Do not add cross-turn session reuse or reconnect.

Pythia, the direct CLI driver, and the stdlib MCP endpoint remain under A. The
broker under B owns **process launch and cleanup only**: no SDK dependency,
model-message parsing, tool catalog, call-ID matching, mailbox, tool execution,
conversation state, or inference HTTP API.
The existing provider/handoff design remains on A's side.

B is already provisioned and its broker is started in B's session or by an
existing supervisor arrangement. Creating a UID alone does not authorize A to
start B processes. Neither file calls sudo, su, runuser, setuid, a package manager,
or an account-management tool. No privileged worker is introduced.

V1 is Linux, same host, **headless pipes**, one fixed Claude profile per broker.
Interactive login remains a direct invocation of `one_click_claude.py` as B.
No remote hosts, TCP relay, abstract control sockets, PTY transport, generic
command runner, per-request profiles, or automatic broker startup in v1.

## 2. Command/configuration contract

The following is the implemented headless command contract; staging remains deferred.

### Broker: start already as B

```sh
/usr/bin/python3 -I -S /trusted/claude-relay/claude_relay.py \
  --relay-serve \
  --relay-socket /tmp/claude-relay-B_UID/control.sock \
  --relay-allow-uid A_UID
```

The default wrapper is the sibling `one_click_claude.py`; an operator-only
`--relay-wrapper /absolute/path` override can select another trusted installation.
Require both scripts, their interpreters, control files, and socket directory to
be outside B's writable sandbox home. Control endpoints and secret control files
must also be outside its mounted runtime trees; the system interpreter/libraries
remain intentional read-only runtime input. Validate source ownership and parent
permissions without requiring caller-owned code to be root-owned.

Freeze these server-side settings at startup:

- Broker UID B and explicit non-root client UID allowlist (normally just A).
- Fixed wrapper path and system `/usr/bin/python3 -I -S` launch command.
- B's account `HOME` and the wrapper's `ONE_CLICK_CLAUDE_BIN` /
  `ONE_CLICK_CLAUDE_HOME` selection. Resolve the private home with the wrapper's
  nonexecuting `--sandbox-print-home` command. Never derive B's HOME from A's env.
- An operator profile label, concurrency limit, and handshake/stop/drain limits.
- Operator-selected proxy/runtime environment, if needed. Pin the native source
  to a stable local version; do not let version probes and launches silently use
  different auto-updated releases.

The existing wrapper is still the authority for native validation, mounting,
private-home setup, managed policy, and fail-closed namespace creation. The broker
never replaces it with a raw native invocation. B provisions the native binary
and authenticates in the private Claude home beforehand. Broker startup/health
checks must not run native `--version` or trigger authentication.

### Client: subprocess executable under A

```text
launcher_executable = /trusted-A/claude-relay/claude_relay.py
CLAUDE_RELAY_SOCKET = /tmp/claude-relay-B_UID/control.sock
CLAUDE_RELAY_SERVER_UID = B_UID
```

Client mode is the default; native `-v`, `--version`, and run argv need no extra
wrapper flags. The stdlib driver constructs an explicit per-subprocess environment
with the endpoint/expected UID for both version queries and main launches. There
is no SDK `cli_path` API, discovery step, or per-query-env quirk to depend on.
These are trusted deployment settings, not conversation/save data. Copies under
A and B may live at different paths.

Reserve leading `--relay-serve`, `--relay-check`, and `--relay-help` management
options; stop relay option parsing at the first CLI argument. Preserve later
prompt strings even if they look like relay options. `-v` is never a relay version
shortcut. Missing configuration, rejected peers, unavailable broker, or protocol
mismatch fail locally with stderr diagnostics and nonzero status: **no local
Claude fallback, broker autostart, prompting, reconnect, or replay**.

`--relay-check` verifies endpoint identity/protocol/profile information without
starting a native process. It must not imply native authentication or CLI/MCP
compatibility. Only that explicit management operation may print relay metadata
to stdout; normal client stdout consists solely of child stdout bytes.

## 3. Authentication and authority boundary

Use a **pathname** `AF_UNIX/SOCK_STREAM` socket, outside B's sandbox-visible
filesystem. The sandbox shares host networking, so an abstract socket would
remove that filesystem barrier. Do not expose a generic localhost TCP service.

Authenticate both directions with Linux `SO_PEERCRED`:

- B checks the connecting process UID against its configured allowlist before
  reading a launch request or accepting descriptors. Do not trust UID fields in
  JSON, caller environment, or claimed process names.
- A checks that its peer really is the configured UID B before sending arguments,
  environment values, or pipe descriptors. Validate the expected socket path and
  ownership as an additional deployment check, not a substitute for peer creds.

A usable rootless cross-UID permission arrangement is a B-owned mode-0711 control
directory under a traversable base, with a mode-0666 socket and **mandatory**
peer-UID authorization. A pre-provisioned restrictive shared group/ACL is also
possible, but not required. Do not relax B's mode-0700 Claude home. Mode-0666 is
not public authorization: all non-allowlisted UIDs are immediately rejected.

For `/tmp` deployment, validate existing directory ownership/components and reject
symlinks or someone else's preclaimed directory. Serialize startup with a B-owned
lock inode that is never replaced. Do not blindly unlink an existing socket:
refuse a live listener, and remove only a verified stale, B-owned socket while
holding the lock. Cleanup must compare the created socket inode so an old process
cannot unlink a replacement broker's endpoint.

Authorized clients are trusted to exercise the **whole selected Claude profile's**
authority, including native CLI arguments and home-backed authentication. This is
not per-tool authorization or tenant isolation. It delegates sandboxed execution,
not arbitrary host execution as B. Multiple unrelated tenants need separate
profiles/brokers, not a larger allowlist on one credential-sharing instance.

## 4. Pipe transport: keep the broker out of model-protocol data

Prefer descriptor passing over framing stdout/stdin in the control protocol.
Linux and Python stdlib already provide `socket.sendmsg`/`recvmsg`, `SCM_RIGHTS`,
`array`, and `fcntl`. No extra system package is needed beyond the existing
Python/Bubblewrap runtime prerequisites.

For each invocation the client creates three `O_CLOEXEC` anonymous pipes:

| Pipe | A retains | B receives for the child's stdio |
| --- | --- | --- |
| stdin | Write end | Read end -> fd 0 |
| stdout | Read end | Write end -> fd 1 |
| stderr | Read end | Write end -> fd 2 |

Do **not** send A's original standard descriptors: those might be terminals,
regular host files, sockets, or directories. B accepts exactly the specified
anonymous pipe ends, checks type/access direction, and rejects truncation,
extra/missing rights, unexpected ancillary messages, and inappropriate FD flags.
Apply close-on-exec atomically where supported (`MSG_CMSG_CLOEXEC`), and close
*every* received FD on every rejection/exception path. The child gets only fd
0/1/2; listener, control, lock, other clients' pipes, and broker files must not leak.

Client-side pumps copy **bytes**, without decoding, line splitting, buffering a
whole response, or interpreting CLI control messages. The A-side stdlib driver
owns the native streaming/control codec; its separate stdlib HTTP MCP endpoint
owns MCP envelopes and raw tool-call metadata. There is no SDK in-process MCP
layer. Opaque pipe transport does not prove the CLI's event/handoff compatibility.

Use bounded stdio pump threads in the disposable client, with a main control/
signal loop and a small wakeup channel. Each pump holds at most a fixed chunk
(e.g. 64 KiB), then relies on kernel pipe backpressure. Use raw `os.read`/`os.write`
with partial-write handling, not buffered Python text streams. Never let a blocked
driver stdout/stderr consumer block signal handling or socket-disconnect detection.
Handle `/dev/null` input and inherited stderr terminals as local endpoints; B
still receives only pipes. Do not change a parent's shared stdio status flags.
No interactive TTY/raw-mode/job-control support is promised by this client.

Descriptor ownership and EOF are part of the protocol:

- A closes its copies of the three remote pipe ends immediately after successful
  transfer; B closes its received copies immediately after child creation or error.
- A's stdin EOF closes its retained stdin write end. It is **not** cancellation:
  keep the control connection and drain both output pipes.
- Child stdin closure/EPIPE stops input pumping and exposes the corresponding
  closed-input behavior to the driver. Do not turn it into fabricated model output.
- Child exit status can arrive while pipe data is buffered. A must drain both
  output streams to EOF before reporting a normal exit. Do not wait for a stdin
  reader blocked on further driver input after the child has exited.
- Output errors, lost consumers, or a post-exit drain deadline cause cancellation
  or an explicit transport failure, never successful truncation. Client shutdown
  must not hang joining blocked pump threads; explicitly close/control its own
  resources and terminate the disposable client after bounded drain/cleanup.

This avoids binary/base64 framing, stream-credit protocols, and unbounded data
queues in the persistent broker. Control stays independent of data backpressure.
There is no global ordering promise between stdout and stderr; preserve byte order
within each stream exactly.

## 5. Small versioned control protocol

One connection owns at most one launch; no multiplexed job IDs or reattachment.
Use a fixed magic/version followed by bounded length-prefixed UTF-8 JSON control
records (for example a 32-bit network-order length, maximum 1 MiB). Validate JSON
objects/types, duplicate keys, nonfinite numbers, NULs, argv/env counts and sizes,
and legal state transitions before allocating/spawning. Actual OS argv/env limits
still apply; report `E2BIG` clearly, do not split/rewrite prompts automatically.

| Phase/record | Contract |
| --- | --- |
| `HELLO` B -> A | Protocol version, broker-instance nonce, public profile label, server limits. No secrets or native execution. Peer UID was already verified. |
| `START` A -> B | An argv vector and explicitly allowed native-protocol/harness environment values; no executable, UID, filesystem mounts, source path, arbitrary cwd, or process PID. A health-only operation ends without starting a child. |
| `READY_FOR_FDS` B -> A | B has validated metadata and reserved capacity for this connection. |
| FD transfer A -> B | One fixed marker byte in a `sendmsg` with exactly three `SCM_RIGHTS` descriptors. Receive with `recvmsg` in this explicit protocol state; reject unexpected ancillary data in every other phase. Do not accidentally discard rights through ordinary `recv`. |
| `STARTED` B -> A | The connection's invocation ID, after successful `Popen`. This is not proof that native initialization/authentication succeeded. |
| `SIGNAL` A -> B | Only a small symbolic allowlist, e.g. INT/TERM/HUP/KILL, targeting this connection's owned child. No client-selected PID or broker-global cancellation. |
| `EXIT` B -> A | Exactly one exit-code or terminating-signal result after reaping. Preserve the observed distinction; do not guess that an exit code `128+n` is a signal. Output EOF/draining is independently observed on the pipes. |
| `ERROR` B -> A | Bounded, sanitized configuration/protocol/capacity/launch failure. Close FDs and stop/reap any child before relinquishing ownership. Never write this to CLI protocol stdout. |

Treat unexpected connection EOF, including a partial record, as cancellation
unless an `EXIT` result has already completed the invocation. There is no socket
half-close convention for stdin; actual pipe EOF supplies that operation.

START env is a deliberately narrow, versioned allowlist of native protocol/
tracing variables plus supplied `CLAUDE_CODE_MCP_AUTO_BACKGROUND_MS`,
`ENABLE_TOOL_SEARCH`, `DISABLE_AUTO_COMPACT`, and the one generation capability
`PYTHIA_CLAUDE_MCP_TOKEN`. MCP bearer expansion was live-checked on 2.1.289;
large-window native compaction suppression still needs verification. Do not invent
those values. B constructs the remaining
sanitized environment from its own profile. Never forward A's HOME, cwd, PATH,
loader/Python injection, credentials, `ONE_CLICK_*`, or `CLAUDE_RELAY_*` values
into B's launch policy. Pin A/B file versions and test their env contract against
the wrapper's allowlist; fail on incompatible protocol fields instead of silently
broadening the environment. Legacy wrapper fields containing `SDK` in their names
are wire data, not a package dependency; the direct driver must not manufacture
an official SDK version/identity. The single MCP capability variable above is an explicit reviewed addition,
not permission to forward arbitrary credentials/environment.

V1 remote `CLAUDE_CONFIG_DIR`/session-store materialization is unsupported:
reject it explicitly rather than ignoring it or mounting A's files. Any operator-
configured B-side config path must already be inside the private home and accepted
by the wrapper. The transport does not rewrite arbitrary CLI file-path arguments.

Broker launch is always an argv-list invocation, conceptually:

```text
/usr/bin/python3 -I -S FIXED_ONE_CLICK_SCRIPT -- [EXACT_REQUESTED_CLAUDE_ARGV...]
```

The inserted `--` prevents remote `--sandbox-*` strings from becoming wrapper
management options. Run the wrapper from a trusted fixed cwd (e.g. `/`), not from
an agent-writable symlink or A's cwd. Its own policy starts Claude in B's private
home/work. Do not introduce a generic remote shell or rely on shell quoting.

## 6. Lifecycle and resource limits

The initial broker uses a bounded set of per-connection control/reaper workers,
each with its own registered `Popen`. Control reads poll cancellation and child
status; no model-event reader or stdio data-pump threads run in the broker.
Track an explicit state machine such as authenticated -> metadata -> FDs ->
starting -> running -> stopping -> reaped. Every error/timeout/disconnect converges
on the same idempotent close/terminate/reap path.

Initial configurable defaults can be: 8 active/reserved invocations, a bounded
pending-handshake count, a 10-second handshake deadline, a 3-second TERM grace,
and a 5-second post-exit client drain deadline. Reject capacity excess rather
than accumulating an unbounded launch queue. Include version probes in capacity.
Do not impose a short no-output/idle timeout: a valid tool chain may be parked on
a withheld MCP result. Generation/parked-call deadlines remain harness-owned;
any broker wall-clock cap is an explicit operator policy.

Local client signals must reach the owned remote invocation over control; native CLI
protocol interrupts pass unchanged through stdin. SIGKILL cannot be trapped, so
socket closure must independently terminate the child. A normal client exit is
reported only after receiving remote status and draining output; missing terminal
status is transport failure (e.g. local status 125), never success. For a genuine
remote signal, re-raise that signal locally after cleanup where practical so ordinary
subprocess semantics are retained.

On disconnect, broker shutdown, launch failure, or stop deadline: close input,
signal the owned child/process group, escalate to KILL if needed, and reap. Never
signal a supplied PID, an already-reaped/reused PID, another invocation, or the
broker's own process group. A client failure must not stop unrelated jobs.

**Cover the pre-Bubblewrap startup window.** The wrapper does static inspection
and copying before execing bwrap. Its existing `--die-with-parent` protects the
sandbox after bwrap starts, but is not sufficient if the broker dies earlier.
The A driver supplies `CLAUDE_RELAY_PARENT_PID` so client death follows its
owner's death even during interpreter startup (this local setting is never
forwarded to B). Independently use a fixed B-side child-start guard in this file: install Linux
`PR_SET_PDEATHSIG=SIGKILL` via stdlib `ctypes`, verify the expected parent PID before
and after installation, and only then exec the fixed wrapper with the sanitized
profile. If the parent disappeared before the guard ran, abort; do not launch an
orphan CLI. Start each child in its own session/group. Test broker death both
before bwrap initialization and after native startup. No privileged syscall or
setuid executable is needed; the non-setuid wrapper/bwrap contract remains.

Do not reconnect or retry a lost START, even if STARTED was never received: the
child may already have run. The harness retires the continuation, preserves
uncertain outcomes, and applies its existing cold-recovery policy. Logs contain
peer/invocation IDs, sizes, states and statuses—not argv, prompt/protocol bytes, MCP
bearer capabilities, authentication files, or the complete environment.

## 7. File staging: explicit initial limit, optional later extension

A cannot directly stage data under B's mode-0700 private home. The minimum useful
release therefore uses inline system prompts and inline MCP/settings where the
pinned native CLI supports them. Reject unsupported adapter options early; keep
conversation streaming on stdin. Do not use A's `/tmp`, automatically copy an
argv-referenced file, relax B's home permissions, or make a blanket shared mount.
The driver's subprocess `cwd` on A is only the local relay's launch cwd; it must
be A-accessible, not a B-private pathname. It is never forwarded to B.

Inline is not a license to leak credentials through process arguments. In the
pinned native CLI/stdlib-driver spike, check how an HTTP MCP bearer capability
reaches Claude; do not assume host process listings hide argv from unrelated users. Use
a verified protocol/stdin mechanism, explicitly supported secret environment
substitution (with reviewed relay/wrapper allowlists), or a protected B-side
staged configuration. If files are necessary, staging is a prerequisite for that
MCP mode. Do not drop listener authentication to keep the inline release smaller.

If real prompt sizes or native CLI behavior require files, add a separately
reviewed broker-managed staging API before enabling those options:

- Explicit prepare/upload/release operations, per-client opaque leases, bounded
  byte/file counts and expiry; bind leases to authenticated UID and broker instance.
- Server-chosen filenames and paths beneath B's private home; no arbitrary
  client-selected host paths. Return canonical B-side paths for the CLI arguments.
- Claim a lease for exactly the intended main invocation, not the driver's version
  probe; retire/cleanup it after teardown. Orphan/expired leases are not reusable
  continuation handles.
- Treat the home as attacker-writable. Use anchored directory FDs, no-follow
  component traversal, exclusive creation and safe unlink/rmdir; never follow
  symlinks, overwrite hardlink aliases, or recursively delete arbitrary targets.
  A fresh random filename alone is not sufficient protection for mutable parents.
- These are writable sandbox data, not trusted broker/driver imports or immutable
  host control state. Staging does not expose Claude authentication files to a
  generic file-read RPC.

Do not implement a generic filesystem service or import/vendor an SDK merely
to avoid specifying this contract. File-backed/remote session-store support is a
separate gate, not something the thin relay can transparently infer from argv.

## 8. Implementation sequence and tests

1. Implement pure control codec, UID/config validation, descriptor/state ownership,
   and client argv/env selection. Use only fixtures; no native discovery/execution.
2. Implement peer-authenticated UNIX listener/client and strict FD handshake. Test
   fragment/coalescence, wrong peer IDs, oversized/malformed messages, wrong/extra
   rights, CLOEXEC, EOF, startup locks, stale sockets, and descriptor-leak paths.
3. Implement guarded fixed-wrapper spawning and reaping. Use a **fixed harmless
   wrapper fixture** in the broker operator configuration, never a remotely
   selected executable. Cover missing child, exec failure, `E2BIG`, concurrency,
   broker death at every startup stage, cancellation, and no wrong-UID fallback.
4. Implement byte pumps and signal/exit propagation with synthetic programs. Cover
   binary data, partial writes, large simultaneous stdout/stderr, blocked consumers,
   closed stdin while output continues, early child input closure, version-probe
   behavior, stdin still open at exit, missing terminal status, and no replay.
5. Exercise the existing Bubblewrap wrapper with an explicitly selected **system
   Python** substitute and temporary private home: A-side argv that resembles
   wrapper options cannot change B's profile; host cwd stays absent; verified
   CLI/MCP env, localhost HTTP, FD isolation and descendant/signal cleanup hold.
   Same-UID local tests validate mechanics, not the cross-UID security boundary.
6. On already-provisioned A/B accounts, separately run cross-UID tests with harmless
   payloads: real peer credentials, socket traversal permissions, B's 0700 home,
   correct process UIDs, unauthorized third UID rejection, and no account changes
   or sudo. Gate staging support on its own hostile-symlink/hardlink tests.
7. Only after separate authorization, perform the pinned native Claude + stdlib
   CLI driver/MCP compatibility spike in the harness plan. No Claude download or
   execution belongs in the implementation test suite.

Acceptance requires both driver-generated native version/main invocations and
complete parked-continuation teardown, not merely a successful socket connect.
There must be no privilege escalation, unmanaged live descendants, protocol stdout
pollution, filesystem widening, automatic retry, or tool execution by the broker.

The A-side driver and MCP server are independent additional protocol work, not
features to fold into this byte relay. Audit imports and helpers transitively and
run A-side fixture tests with third-party site packages unavailable. Do not run a
third-party SDK/MCP implementation as a test oracle under A. Same-UID all-Python
fixtures can test relay mechanics; real native/Bubblewrap launches belong to B
in the strict deployment. Audit Python tool handlers/plugins as integration code;
model-requested shell/program payloads are intentionally outside that dependency
constraint and execute as A. No generic command executor is added to B's relay.
