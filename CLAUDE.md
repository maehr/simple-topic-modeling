@AGENTS.md

## Claude Code

Use `AGENTS.md` as the canonical machine contract. The lines below add what is specific to Claude Code.

- `poetry run ruff check . && poetry run pytest` after a normal change batch
- `poetry run ruff check . && poetry run ruff format --check . && poetry run pytest && pnpm check` before a handoff

Run Poetry from `~/.local/bin`. It is installed as a `uv` tool, not through Homebrew.

A change to `src/` reaches the browser as well as the local app. Section 7 of `AGENTS.md` holds the version limits. Test the browser build before a handoff that changes `src/`, `index.html`, or `assets/dist/`.
