@AGENTS.md

## Claude Code

Use `AGENTS.md` as the canonical machine contract. The lines below add what is specific to Claude Code.

- `uv run ruff check . && uv run pytest` after a normal change batch
- `uv run ruff check . && uv run ruff format --check . && uv run pytest && pnpm check` before a handoff

A change to `src/` reaches the browser as well as the local app. Section 7 of `AGENTS.md` holds the version limits. Test the browser build before a handoff that changes `src/`, `index.html`, or `assets/dist/`.
