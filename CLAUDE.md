# Rules for Claude in this repo

This repo holds code and an experiment registry (`exps/`) only. No plans, question lists, literature notes, drafts or interpretation.

## Role

- You execute experiment cards from Erfan. Don't propose research directions, next experiments or interpretations unless asked.
- Never commit, merge, push or rebase. Leave changes unstaged for Erfan to review.

## When a card arrives

1. Save it as `exps/E###-<slug>.md` exactly as given.
2. Add a row to `exps/INDEX.md` (newest first) with status `queued`.
3. Check what data, checkpoints and outputs already exist. Report anything missing or ambiguous before running. Don't guess.

## Code

- Experiment-specific scripts go in `scripts/exp/E###_<name>.py`.
- Changes to shared code (`agent/`, `retrieval/`, `baselines/`, shared scripts) happen in a worktree: `git worktree add ../sar-<slug> -b feat/<slug>`. Wait for Erfan to merge before the run.

## Runs

- Follow the card. Where the card specifies something (paths, names, settings), it overrides these defaults.
- All data, results and logs go under `$DATA_ROOT`, never in the repo. If a card names a repo path for data or results (e.g. `data/…`), put it at the matching place under `$DATA_ROOT` and note the path in the card's Execution section.
- Default run ID: `E###-<arm>-s<seed>`. Default outputs: `$DATA_ROOT/logs/E###/<run id>/`.
- Record the git commit and dirty state.
- Never delete a checkpoint or output that a card references. If disk space is needed, list candidates and ask Erfan first.
- In each card's Execution section, record the checkpoint paths used and confirm they still exist after the run.
- Record any deviation from the card (a substituted checkpoint, a moved path, an implementation choice the card left open) in the card's Execution section.

## After a run

- Fill in the card's Execution and Result sections: commit, run IDs, paths, wall time, result tables, and a short factual summary of the numbers (with n and standard errors where relevant). No interpretation, no recommendations.
- Update the card's row in `exps/INDEX.md`.
- Report back in at most one screen: status, the key table, anything that went wrong.
