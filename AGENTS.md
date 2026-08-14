# MailAgent Framework Architecture Rules

This repository is the public, vertical-agnostic MailAgent framework.

## Mandatory boundary

- `src/mailagent/core/` and `src/mailagent/classification/` own only generic
  contracts, orchestration, audit, fallback, validation, and versioning.
- Core must not contain a company's names, domain vocabulary, entity fields,
  taxonomy labels, example keywords, sender domains, subject patterns, prompts,
  routing assumptions, or vertical-specific conditionals.
- Executable business behavior belongs in an installable vertical package
  registered through the `mailagent.verticals` entry-point group.
- Reviewed declarative business knowledge belongs in the vertical package's
  external profile: manifest, taxonomy, rules, signals, schemas, patterns,
  retrieval policy, and RAG declarations.
- External profiles cannot choose arbitrary Python imports. Installed entry
  points provide code; the selected profile must match plugin ID and namespace.

## How to extend the system

When a private vertical needs a capability that current contracts cannot
express, extend a generic validated Protocol/schema/loader here, prove it with
neutral tests, then implement and configure the business behavior in the
private vertical repository. Never patch a private example into Core.

The bundled `example_triage` plugin is intentionally synthetic and minimal. It
exists only to test and demonstrate the public extension surface.

## Required checks

Run before every merge:

```bash
pytest -q
ruff check .
mypy src/mailagent
mailagent vertical validate --json
```

`tests/test_architecture_boundary.py` is a release gate. Update it only when a
generic public contract intentionally changes; do not weaken it to accommodate
a business requirement.
