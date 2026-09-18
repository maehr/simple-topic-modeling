# AGENTS.md

This repository holds a Streamlit application that finds topics in a text corpus with Latent Dirichlet Allocation (LDA).

This file follows the shared Python specification in `~/.claude/agents-specs/10-AGENTS.python.md`. Sections 1, 3, 5, and 6 are the shared text. Sections 2, 4, and 7 are narrowed to this repository. The specification assumes uv, ty, prek, and Pydantic. This repository uses none of them, so those parts are removed.

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

Pin versions in `pyproject.toml`. Commit `poetry.lock`. Install with `poetry install`.

**Python.** [Poetry](https://python-poetry.org/docs/) dependencies, environments, and the lockfile · [ruff](https://docs.astral.sh/ruff/) lint and format · [pytest](https://docs.pytest.org/) with [pytest-cov](https://pytest-cov.readthedocs.io/) tests.

**JavaScript.** [pnpm](https://pnpm.io/) dependencies, pinned by the `packageManager` field · [prettier](https://prettier.io/) format for Markdown, YAML, HTML, and JSON · [husky](https://typicode.github.io/husky/) the pre-commit hook · [commitizen](https://commitizen-tools.github.io/commitizen/) (`cz`) commits · [git-cliff](https://git-cliff.org/docs/) changelog, configured in `cliff.toml`.

**Runtime.** [Streamlit](https://docs.streamlit.io/) the user interface · [pandas](https://pandas.pydata.org/docs/) tables · [scikit-learn](https://scikit-learn.org/stable/) the LDA model and the vectorizer · [pyLDAvis](https://github.com/bmabey/pyLDAvis) the topic figure.

There is no type checker. Do not add one without the owner's agreement.

## 3. Standards

- [SemVer 2.0.0](https://semver.org/) for versions.
- [Conventional Commits 1.0.0](https://www.conventionalcommits.org/en/v1.0.0/) for commit messages.
- [Contributor Covenant 3.0](https://www.contributor-covenant.org/version/3/0/code_of_conduct/) as `CODE_OF_CONDUCT.md`.
- [AGPL-3.0](https://www.gnu.org/licenses/agpl-3.0.en.html) as `LICENSE.md`. SPDX identifier `AGPL-3.0-only`.

## 4. Code

- Keep the model logic in `src/topic_model.py`. Keep `src/app.py` for Streamlit calls only.
- Write pure functions in `src/topic_model.py`. The tests import them directly.
- Add a test in `tests/` for each defect you fix.
- Do not use `unsafe_allow_html`. Use `st.download_button` for a file the user downloads.
- Keep the results in `st.session_state`. A widget click reruns the script, and `st.button` is then false.
- There is no coverage gate. Do not add one without the owner's agreement.

Run this gate before a handoff:

```bash
poetry run ruff check . && poetry run ruff format --check .
poetry run pytest
pnpm check
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

- Without `enforce_admins`, the rule is advisory for whoever can bypass it.
- A solo repository needs three settings together: 0 required approvals, no required code-owner review, and no required last-push approval. You cannot approve your own pull request, so either of the last two deadlocks the merge even at 0 approvals.
- The free plan covers branch protection on a public repository. Branch protection on a private repository needs Pro.
- Plan limits differ per feature. Secret scanning, push protection, CodeQL, and dependency review are free on a public repository. On a private repository each one needs a paid GitHub security plan. Check the plan before you enable one in a workflow or in `gh repo edit`.
- Rulesets are the successor at organization scale. `gh ruleset` only reads, and creation uses `POST`, so a second run duplicates the ruleset. Prefer the `PUT` for one repository.

## 6. CI/CD Security

Follow [secure use of Actions](https://docs.github.com/en/actions/reference/security/secure-use).

- Set `permissions: contents: read` at the top level. Widen it per job only where a job needs more.
- Default the repository token to read: `gh api -X PUT repos/OWNER/REPO/actions/permissions/workflow -f default_workflow_permissions=read`.
- Pin an action to a full commit SHA. Verify the SHA against the upstream repository, not a fork. Let Dependabot bump it.
- Never check out fork code under `pull_request_target`. Prefer `workflow_run`, and treat its artifacts as untrusted.
- Never interpolate `github.event.*` into `run:`. Pass the value through `env:` and quote `"$VAR"`.
- Use OIDC and a short-lived cloud role. Do not store a long-lived secret. Keep a secret a scalar, never a JSON blob. Rotate it.
- Gate a deploy on an Environment with required reviewers. Prefer an environment secret over a repository secret.
- Do not use a self-hosted runner on a public repository.
- Cover `.github/workflows/**` in `CODEOWNERS`.
- Require the section 4 gate, CodeQL, and `dependency-review-action`. Run the same gate locally, so CI gives no surprise. On a private repository, confirm the plan covers CodeQL and dependency review first.

## 7. Browser and WASM

This repository ships the same `src/` to two runtimes. The browser runtime decides its own package versions, so it sets the limit for both. Read this section before you change a dependency or a library call.

| Package      | Browser | Local  |
| ------------ | ------- | ------ |
| Python       | 3.13.2  | 3.13   |
| streamlit    | 1.62.0  | 1.64.0 |
| pandas       | 2.3.3   | 2.3.3  |
| scikit-learn | 1.7.0   | 1.9.1  |
| numpy        | 2.2.5   | 2.2.5  |

- Use no API that is newer than the browser column. A local test cannot find that mistake.
- Ship pure Python, or a verified Pyodide or PyPI WASM wheel. Never ship a development tool.
- Do not use an OS socket, a subprocess, or a thread.
- `index.html` mounts three files: `app.py`, `topic_model.py`, and `utils/stopwords.py`. It mounts them flat. Import `topic_model`, never `src.topic_model`.
- Keep `numpy` below 2.3. pyLDAvis 3.4.1 cannot serialize the values that a newer numpy returns. The application then fails at the figure step.

### Scripts

Run `python3 scripts/fetch_stlite.py` to get the browser runtime. The script verifies the download against `scripts/stlite.lock.json`. `assets/stlite/` is not committed.

Run `python3 scripts/build_pyldavis_wheel.py` to rebuild `assets/dist/pyLDAvis-3.4.1-py3-none-any.whl`. The script verifies the upstream wheel against a pinned SHA-256. The output is reproducible. `assets/dist/` is committed.

Caution: do not bypass either integrity check. A mismatch means the upstream artifact changed. Investigate the change first.

### Local browser test

1. Run `python3 scripts/fetch_stlite.py`.
2. Run `python3 -m http.server 8000`.
3. Open `http://localhost:8000/` in a browser.

Open the directory URL. Do not open `http://localhost:8000/index.html`. stlite builds its asset URLs from the page URL, and the file form breaks them.
