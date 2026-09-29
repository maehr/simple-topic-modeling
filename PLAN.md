# Plan — UX review (issue #31)

Part of #31. The numbers below follow the "Suggested implementation order" in the issue.
Two findings are still open:

- 6 — make the topic map directly readable. #35 makes the bubbles and bars clickable. Labels on
  the bubbles are still missing.
- 10 — review responsive behaviour and accessibility.

## Stack (each PR targets the branch below it)

| PR | Branch | Base | Finding |
|----|--------|------|---------|
| #33 | `fix/reject-ambiguous-uploads` | `main` | 1 — reject ambiguous uploads |
| #34 | `feat/run-demo` | `fix/reject-ambiguous-uploads` | 2 — Run demo / Use my own data |
| #35 | `feat/shared-topic-selection` | `feat/run-demo` | 3 — one shared selected topic |
| #36 | `feat/rerun-from-notice` | `feat/shared-topic-selection` | 4 — Rerun in the banner, sidebar Run control |
| #37 | `feat/single-document-modes` | `feat/rerun-from-notice` | 5 — rename modes by intent |
| #38 | `feat/rename-in-topic-view` | `feat/single-document-modes` | 7 — rename a topic from its header |
| #39 | `feat/diagnostics-explanations` | `feat/rename-in-topic-view` | 9 — explain the diagnostics, compare runs |
| #40 | `feat/task-oriented-export` | `feat/diagnostics-explanations` | 8 — research package as the primary export |

Merge from the bottom up.

## Status
- Items 3–5: implemented, gate green, and browser-tested on local `marimo run`.
- Copilot review, 2026-09-29: all six open threads fixed.
  - #34: Run demo keeps the reader's settings.
  - #35: rename restored, card buttons labelled, heatmap matched by topic id.
  - #37: PDF segments and pasted text named correctly.
  - Browser testing found that two equal topic names crashed the result view. That bug is also on `main`.
    Fixed in #35: a taken name gets a number.
- The exported Pyodide `dist/` of #37 passed a smoke test on 2026-09-29.
- Findings 7, 8, and 9: implemented, gate green, and browser-tested on local `marimo run`.
  - 7: **Rename** sits beside the topic heading in the Topics tab. It opens a form with
    **Save name** and **Cancel**. A selection change closes the form.
  - 8: **Download complete research package (.zip)** comes first. A closed **Individual files**
    section holds the single files with task labels.
  - 9: each measure has a one-line explanation. A table compares up to five runs of the session
    on the same corpus.
- Lesson: `app.run()` runs only the path without a result. A deleted cell therefore passed the gate
  and broke the result view. Check the cell parameters against the cell returns after a large edit.
- Open: marimo-internal `handleFillUpdated/Unmount` console messages seen once after a slider change on PR 4. Cause unknown.
