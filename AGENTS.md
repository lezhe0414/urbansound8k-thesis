# AI Agent Collaboration Guide

This repository supports a thesis project that includes both academic writing and code.
The user has permission from their professor to use AI agents, so assistance is allowed.

## Working Language

- Use Traditional Chinese by default when communicating with the user.
- Keep operational guidance concrete: exact commands, file paths, and next steps are preferred.
- When writing academic thesis content, use formal Traditional Chinese unless the target paper requires English.

## Repository Purpose

The project should help the user:

- define and refine the thesis topic and research questions;
- organize literature notes and references;
- write reproducible code for experiments, analysis, or prototypes;
- record experiments, results, and decisions;
- draft, revise, and format thesis sections.

## Expected Structure

- `docs/`: thesis planning, outline, research logs, meeting notes, and writing drafts.
- `docs/chapters/`: chapter-level manuscript drafts.
- `src/`: reusable source code.
- `notebooks/`: exploratory analysis or experiment notebooks.
- `data/raw/`: original data. Do not overwrite raw data.
- `data/processed/`: cleaned or derived data.
- `results/`: experiment outputs, metrics, tables, and logs.
- `figures/`: charts and images used in the thesis.
- `references/`: papers, citation exports, and literature notes.

## Collaboration Rules

- Inspect current files before changing them.
- Read `docs/dashboard.md` first when resuming broad thesis support work.
- Read `docs/current_status.md` first when resuming broad thesis support work.
- Use `docs/first_week_plan.md` when the project still lacks thesis direction, coding task, professor request, or data source.
- Use `docs/professor_questions.md` to prepare questions for the professor when requirements are unclear.
- Use `docs/professor_update_template.md` when drafting a message or meeting summary for the professor.
- Use `docs/next_input_template.md` when the user needs a concise way to provide thesis direction, coding needs, or professor requests.
- Use `docs/weekly_review.md` for recurring progress reviews.
- Record substantial AI assistance in `docs/ai_usage_log.md`.
- Treat `docs/ai_disclosure_draft.md` as a draft only; professor and school requirements decide final disclosure wording.
- Do not delete or rewrite user work unless explicitly asked.
- Prefer small, traceable changes with clear purpose.
- Keep code reproducible: document inputs, outputs, parameters, and commands.
- When adding analysis or experiment code, include a minimal way to run or verify it.
- If a claim depends on literature, data, or experiment output, identify the evidence source.
- Separate speculation from verified results.
- Follow `docs/writing_style_guide.md` and `docs/glossary.md` when drafting or revising thesis prose.
- Record major thesis, methodology, data, tooling, or professor-driven decisions in `docs/decision_log.md`.
- Put unclassified tasks in `docs/task_inbox.md` before sorting them into milestones or specific files.
- Track important deliverables in `docs/artifact_index.md`.
- Track risks and blockers in `docs/risk_register.md`.
- Before professor handoff or staged delivery, check `docs/submission_checklist.md`.

## Code Standards

- Keep source code in `src/` unless there is a project-specific reason.
- Keep notebooks for exploration, not as the only source of important logic.
- Prefer deterministic scripts for final experiments.
- Store generated outputs in `results/` or `figures/`, not mixed into source folders.
- Avoid committing large raw datasets unless the user explicitly wants that.
- Record runtime setup, package managers, environment variables, and reproduction commands in `docs/environment.md`.
- Define coding tasks in `docs/code_task_spec.md` before implementing thesis-related code.
- Do not commit real secrets. Use `.env.example` for public environment variable names and `.env` for local private values.

## Thesis Workflow

For each substantial task, aim to leave behind at least one durable artifact:

- a draft section in `docs/`;
- a chapter draft in `docs/chapters/`;
- an experiment script or notebook;
- a result table, chart, or log;
- a literature summary;
- an artifact index update in `docs/artifact_index.md`;
- or a decision record in `docs/research_log.md`.
- or a formal decision in `docs/decision_log.md`.

When uncertain about the thesis topic or research method, first improve the project plan
and list concrete questions for the user or professor.
