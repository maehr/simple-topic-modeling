# Agents Tooling Specification — Python

Use this specification for a Python agent codebase. Copy it into the repository as `AGENTS.md`.

## 1. Orchestration

Context is the scarce resource. Manage it.

**Model tier.** A frontier model holds the plan, the decisions, and the shared context. A small model does a bounded subtask. Pick the tier before you spawn the agent.

**Delegate on evidence.** A subagent starts cold and derives the context again. Delegate work that reads far more than it reports: broad search, log triage, or fan-out over many files. Do a small local edit inline.

**Contracts, not conversations.** Give a subagent one task, the context it cannot infer, the output shape, and the stop condition. A subagent returns a conclusion, never a file dump. Verify a report before you act on it.

**Parallel only when independent.** Run agents at the same time only when no result feeds another. Use three at most.

**Context ladder.** At 25% of the window, name the source of the pressure. At 50%, reduce it: write state to disk, delegate the reading, or narrow the re-reads. At 75%, stop and reduce before further work.

**State on disk.** Write plans, findings, and decisions to files. The transcript dies at the next compaction.

**Read narrow.** Read the slice, not the file. After you write a file, verify the diff or the section you changed. Do not read the whole file again.

## 2. Tooling

Pin versions in `pyproject.toml`. Commit `uv.lock`. Install with `uv sync --locked`. Never ship a development tool to runtime.

**Development.** [uv](https://docs.astral.sh/uv/) dependencies, environments, and the lockfile · [ruff](https://docs.astral.sh/ruff/) lint and format · [ty](https://docs.astral.sh/ty/) types, pinned exactly while it stays in beta · [pytest](https://docs.pytest.org/) with [pytest-cov](https://pytest-cov.readthedocs.io/) tests, doctests, and coverage · [prek](https://prek.j178.dev/) hooks, with `prek.toml` · [commitizen](https://commitizen-tools.github.io/commitizen/) (`cz`) commits and SemVer bumps · [git-cliff](https://git-cliff.org/docs/) changelog.

**Runtime.** [Pydantic](https://pydantic.dev/) v2 validation · [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) configuration · [FastAPI](https://fastapi.tiangolo.com/) async APIs, server only · [HTTPX2](https://github.com/pydantic/httpx2) HTTP client, imported as `httpx2` · [sqlite3](https://docs.python.org/3/library/sqlite3.html) stdlib database, no pin · [pandas](https://pandas.pydata.org/docs/) tables · [Typer](https://typer.tiangolo.com/) CLIs · [marimo](https://docs.marimo.io/) reactive notebooks · [structlog](https://www.structlog.org/) logging · [Altair](https://altair-viz.github.io/) and [Matplotlib](https://matplotlib.org/stable/) charts.

## 3. Standards

* [SemVer 2.0.0](https://semver.org/) for versions.
* [Conventional Commits 1.0.0](https://www.conventionalcommits.org/en/v1.0.0/) for commit messages.
* [Contributor Covenant 3.0](https://www.contributor-covenant.org/version/3/0/code_of_conduct/) as `CODE_OF_CONDUCT.md`.
* [AGPL-3.0](https://www.gnu.org/licenses/agpl-3.0.en.html) as `LICENSE`. SPDX identifier `AGPL-3.0-only`.

## 4. Code

* Use a type hint everywhere.
* Use Pydantic at an I/O boundary: an API, a tool, configuration, or serialized state.
* Use a dataclass for an internal carrier. Keep core logic in pure functions.
* Put a runnable example in each docstring.
* Require 100% coverage on core logic. Keep each exclusion narrow and explicit.
* Run the gate below through `prek`, with a `cz` `commit-msg` hook. Let `git-cliff` write the changelog.

```bash
ruff check . && ruff format --check .
ty check
pytest --doctest-modules --cov --cov-fail-under=100
```

## 5. GitHub Workflow

Use a fork and a pull request. Never push to upstream.

```bash
gh repo fork OWNER/REPO --clone
git switch -c feat/thing
gh pr create --repo OWNER/REPO
```

Allow maintainer edits. Resync with `gh repo sync`. Put one logical change in one PR. Title the PR with a Conventional Commit.

**Stacked PRs.** Each PR targets the branch below it. The bottom PR targets the trunk. Merge from the bottom up. GitHub retargets the rest. Take the requirements from the trunk only. Use the `gh stack` extension. Keep a stack in one repository, never across forks.

**Trunk protection.** Both layers are idempotent, so a second run converges.

```bash
gh repo edit --enable-squash-merge --enable-merge-commit=false --enable-rebase-merge=false \
  --delete-branch-on-merge --allow-update-branch \
  --enable-secret-scanning --enable-secret-scanning-push-protection
gh api -X PUT repos/OWNER/REPO/branches/main/protection --input protection.json
```

The `PUT` replaces the whole configuration. Take the body shape from the [API reference](https://docs.github.com/en/rest/branches/branch-protection), not from a stale snippet.

For a repository with more than one maintainer, enforce a PR before a merge, at least one approval, dismissal of a stale approval on push, code-owner approval, last-push approval, conversation resolution, strict status checks, linear history, `enforce_admins: true`, no force-push, and no deletion. A solo repository differs. See the bullets below.

* Without `enforce_admins`, the rule is advisory for whoever can bypass it.
* A solo repository needs three settings together: 0 required approvals, no required code-owner review, and no required last-push approval. You cannot approve your own pull request, so either of the last two deadlocks the merge even at 0 approvals.
* The free plan covers branch protection on a public repository. Branch protection on a private repository needs Pro.
* Plan limits differ per feature. Secret scanning, push protection, CodeQL, and dependency review are free on a public repository. On a private repository each one needs a paid GitHub security plan. Check the plan before you enable one in a workflow or in `gh repo edit`.
* Rulesets are the successor at organization scale. `gh ruleset` only reads, and creation uses `POST`, so a second run duplicates the ruleset. Prefer the `PUT` for one repository.

## 6. CI/CD Security

Follow [secure use of Actions](https://docs.github.com/en/actions/reference/security/secure-use).

* Set `permissions: contents: read` at the top level. Widen it per job only where a job needs more.
* Default the repository token to read: `gh api -X PUT repos/OWNER/REPO/actions/permissions/workflow -f default_workflow_permissions=read`.
* Pin an action to a full commit SHA. Verify the SHA against the upstream repository, not a fork. Let Dependabot bump it.
* Never check out fork code under `pull_request_target`. Prefer `workflow_run`, and treat its artifacts as untrusted.
* Never interpolate `github.event.*` into `run:`. Pass the value through `env:` and quote `"$VAR"`.
* Use OIDC and a short-lived cloud role. Do not store a long-lived secret. Keep a secret a scalar, never a JSON blob. Rotate it.
* Gate a deploy on an Environment with required reviewers. Prefer an environment secret over a repository secret.
* Do not use a self-hosted runner on a public repository.
* Cover `.github/workflows/**` in `CODEOWNERS`.
* Require the section 4 gate, CodeQL, and `dependency-review-action`. Run the same gate locally, so CI gives no surprise. On a private repository, confirm the plan covers CodeQL and dependency review first.

## 7. Browser and WASM

These rules apply only to Python that runs in a browser. A server target has no such limit.

* Ship pure Python, or a verified Pyodide or PyPI WASM wheel. Never ship a development tool.
* Every Runtime entry except FastAPI and HTTPX2 is a candidate. Check the version first.
* Do not use an OS socket, a subprocess, or a thread. Do not run FastAPI in a browser.
* Use `pyodide.http.pyfetch` or `pyxhr` for HTTP. Use HTTPX2 only with a tested custom transport.

## 8. Project notes — Simple Topic Modeling

These rules come from measured behaviour of `marimo` 0.24.2. Do not change them without a new test.

### Package layout

Keep `simple_topic_modeling/` at the repository root. Do not use a `src/` layout.

`marimo export html-wasm` builds a wheel from each local module that the notebook imports. It resolves
the module name against the notebook directory first. A `src/` layout therefore produces a wheel named
`src`, which the browser cannot import as `simple_topic_modeling`. The marimo setting `runtime.pythonpath` does
not change this order.

### Build the wheel before each export

marimo's own wheel builder copies `.py` files only. It drops package data, so the stop-word lists never
reach the browser. Build the wheel with `hatchling` instead:

```bash
uv build --wheel -o public/wheels
```

`app.py` names that wheel in its PEP 723 block:

```text
simple-topic-modeling @ public/wheels/simple_topic_modeling-0.1.0-py3-none-any.whl
```

The export rewrites the path to `../public/wheels/...` and copies `public/` into `dist/`. marimo skips
its own wheel for any package that the PEP 723 block already names.

The wheel file name carries the version. Update the path in `app.py` when you bump the version in
`pyproject.toml`.

`public/wheels/` holds build output. Git ignores it. Always build before you export.

### Keep the dependency bounds below the Pyodide versions

Pyodide ships its own build of every runtime package. micropip refuses to reinstall one, so a lower
bound above the shipped version stops the whole install. The app then fails with
`ModuleNotFoundError: No module named 'simple_topic_modeling'`.

Write each bound as the oldest API the code uses. Never raise a bound to the newest release. The
Pyodide version that marimo pins lags the Pyodide documentation.

Keep `marimo` out of `[project].dependencies`. Only `app.py` imports marimo. The package does not.

### Install the wheel again from an absolute URL

marimo hands micropip the PEP 723 wheel path as a relative URL. micropip does not resolve it against
the page. The first cell therefore installs the wheel again, under a `sys.platform == "emscripten"`
guard, using `mo.notebook_location()` to build an absolute URL. Keep both paths. The PEP 723 entry
also stops marimo from building its own wheel.

### Check the notebook with `app.run()`

`marimo check` does not catch a name that two cells both define. `app.run()` does. Run this after you
edit `app.py`:

```bash
uv run python -c "
import importlib.util, sys
spec = importlib.util.spec_from_file_location('notebook_app', 'app.py')
module = importlib.util.module_from_spec(spec); sys.modules['notebook_app'] = module
spec.loader.exec_module(module); module.app.run(); print('cells ok')
"
```

Prefix every cell-local variable with `_`. marimo requires a unique name across cells.

### Two targets, two checks

`marimo edit app.py` runs local CPython. The exported `dist/` runs Pyodide in the browser. A change
can pass one and fail the other. Always check both before you call the work done.

### Fit only on the run button, and keep the last good result

`SPECS.md` section 4 needs an explicit fit. `SPECS.md` section 8 needs the last good result to
survive a failed rerun. marimo reruns a cell when any referenced value changes, so these two rules
need care.

Use this pattern. Do not replace it without a browser test.

* Hold the result, the failure, and the name overrides in `mo.state`.
* Let the fit cell reference `run_button.value` and the settings.
* marimo resets `run_button.value` to False after the dependent cells run. A settings change
  therefore reruns the fit cell without a fit, and the state keeps the previous result.
* Catch the failure in the fit cell. Store the message. Never clear the result.
* Compare `result.config` against the live config to raise the *Configuration changed* banner.

`mo.ui.form` does not work here. A form copies the element it wraps, so the app cannot read the live
value to compare it against the submitted value.

### The demo corpus ships inside the wheel

`SPECS.md` section 9 puts the demo corpus in `assets/`. That path needs a network fetch, which is
fragile in Pyodide. Keep the file at `simple_topic_modeling/data/demo_corpus.csv` instead. `importlib.resources`
reads it, exactly as it reads the stop-word lists.

### The type checker cannot follow Altair

Altair builds each `mark_*` method at runtime, so `ty` cannot infer that the method returns a chart.
`pyproject.toml` turns `unresolved-attribute` off for `simple_topic_modeling/plots.py` only. Keep the rule
on for every other file.

### Write the changelog with git-cliff

Write each commit message as a Conventional Commit. `cliff.toml` maps the commit type to a changelog
group. Run this command after you commit:

```bash
git-cliff --tag v2.0.0 -o CHANGELOG.md
```

Change the tag when you bump the version in `pyproject.toml`.

### The repository runs on GitHub

The remote is `github.com/maehr/simple-topic-modeling`. The published app is
`https://maehr.github.io/simple-topic-modeling/`.

Three workflows run:

* `ci.yml` runs the section 4 gate in the `Gate` job. The `Notebook` job runs `marimo check` and
  the `app.run()` check. The `dependency-review` job runs on a pull request only.
* `pages.yml` builds the wheel, exports the app, and publishes `dist/`. It fails when the wheel
  file name does not match the project version.
* `codeql.yml` scans the Python code.

Run the same gate locally before each commit. CI must give no surprise.

This project has one maintainer. Branch protection therefore uses 0 required approvals, no
code-owner review, and no last-push approval. Any of the last two would deadlock a merge, because
you cannot approve your own pull request.

These parts of sections 2, 3, 5, and 6 still wait:

* `commitizen` hooks. Write the Conventional Commit message by hand.
* The fork-based pull request flow. A solo maintainer pushes a branch in this repository.

### Version 1 stays where it is

Version 1 was a Streamlit app that ran through stlite. Version 2 replaced it on `main` with an
unrelated history. The version 1 commits stay at the tag `v1.0.0` and the branch
`legacy/streamlit`. Never merge that branch into `main`. Never revive its dependencies; the
pyLDAvis and numpy caps belong to that tree only.

### Bump the version in three places

The wheel file name carries the version, and the browser installs the app from that file. A bump
that misses one place passes every test and then fails in the browser with
`ModuleNotFoundError: No module named 'simple_topic_modeling'`.

1. `pyproject.toml`, the `version` field.
2. `app.py`, the PEP 723 block.
3. `app.py`, the `mo.notebook_location()` path in the first cell.

Then run `uv lock`, rebuild the wheel, and regenerate the changelog with the new tag.
