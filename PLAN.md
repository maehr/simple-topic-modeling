# Plan — UX review (issue #31), items 3–5

Part of #31. Items 1 and 2 are in PR #33-era branches; items 6–10 are out of scope.

## Stack (each PR targets the branch below it)

| PR | Branch | Base | Item |
|----|--------|------|------|
| 1 | `fix/reject-ambiguous-uploads` | `main` | 1 — reject ambiguous uploads |
| 2 | `feat/run-demo` | `fix/reject-ambiguous-uploads` | 2 — Run demo / Use my own data |
| 3 (#35) | `feat/shared-topic-selection` | `feat/run-demo` | 3 — one shared selected topic |
| 4 (#36) | `feat/rerun-from-notice` | `feat/shared-topic-selection` | 4 — Rerun in the banner, sidebar Run control |
| 5 | `feat/single-document-modes` | `feat/rerun-from-notice` | 5 — rename modes by intent |

Merge from the bottom up.

## Status
- Items 3–5: implemented, gate green, browser-tested on local `marimo run`.
- Not verified: exported Pyodide `dist/` (the proxy blocks the Pyodide CDN in the build sandbox). Check it after merge.
- Open: marimo-internal `handleFillUpdated/Unmount` console messages seen once after a slider change on PR 4. Cause unknown.
