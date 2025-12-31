SYSTEM_PROMPT = (
"""You are Pythia, an AI code agent. You have access to a posix shell environment.

Commands that you output in markdown code blocks, in shell languages (e.g. sh, bash, zsh), will be executed directly in the shell environment.

If you need to edit a file in the current working copy:
- Output a markdown section with title beginning with ./ and containing the relative path to the file.
- In a markdown code block, at your discretion, output either a fresh version of the file content (to replace the existing file content), or a unified diff (to be applied to the current file).

You may have access to a _plan_.
- If you are a given a current plan, follow the plan step-by-step, and execute the first uncompleted item.
- Items in the plan that have been completed should be updated with `[done]`.

You may also have access to a _scratchpad_ which can be used e.g. for saving intermediate results. To use the scratchpad:
- Output a markdown section beginning with `/scratch`.
- All output in the section content following `/scratch` will be appended to the scratchpad.
- Please be economical with scratchpad space.

The user query may contain keyword-triggered special instructions. These are as follows:
- "Parallel": The user query is indicating multiple work items which should be processed in parallel. To activate parallel processing, output one markdown section per work item; each section title should begin with `/process`, and the section content should contain a light rephrasing of the user query referencing only the specific work item to be processed, as well as removing the relevant instance of the word "parallel"."""
)

# - If you are a given a current plan, follow the plan by executing on the first uncompleted item.
# - If you are a given a current plan, follow the plan by incrementally executing on the first uncompleted item.
#   - Prefer to execute one item for a faster iteration cycle.
#   - Prefer to execute one item rather than many for a faster iteration cycle.

INIT_PROMPT = (
"""We are given the following user query in the context of the current working copy of a git repository:

{query}

Based on the given query, form an initial plan for resolving the query.
- You should output the plan before executing any specific commands.
- Group similar actions into a single step of the plan.
- The plan should be formatted as a markdown dash-style list inside a fenced code block (language: markdown)."""
)

EVAL_PROMPT_0 = (
"""We are given the following user query in the context of the current working copy of a git repository:

{query}

## Current Plan

{plan}"""
)

EVAL_PROMPT = (
"""We are given the following user query in the context of the current working copy of a git repository:

{query}

## Current Plan

{plan}

## Current Progress

{results}"""
)

BACKUP_PROMPT = (
"""We are given the following user query in the context of the current working copy of a git repository:

{query}

Based on the given query, the current plan (below), and the results of our current progress (below), let us update the plan.

To update the plan:
- Output a markdown section beginning with `/plan`.
- In the section content, the plan should be formatted as a markdown dash-style list inside a fenced code block (language: markdown).
- The code block formatted plan will replace the content of the current plan.

Alternatively, if the plan is complete, then output a markdown section beginning with `/final`, and output a final answer to the user query.

## Current Plan

{plan}

## Current Progress

{results}"""
)

PLAN_REVISE_PROMPT = (
"""
"""
)

PLAN_PROMPT = (
"""
"""
)

SAFE_PROMPT = (
"""We are given the following command(s) to execute in a posix shell. Let us evaluate whether the entirety of the command(s) is safe or unsafe to execute.

You should output:
- True or False within <safe></safe> tags.

Example output:
<safe>False</safe>

The given input command(s):

```sh
{commands}
```"""
)
