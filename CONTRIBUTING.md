# Contributing

MailAgent welcomes changes to its generic mail-understanding runtime and plugin
contracts. Business-specific behavior is developed in separate vertical
packages.

## Choose the correct repository

Contribute here when a change applies across multiple unrelated verticals:

- classifier and orchestration contracts;
- validation schemas and asset loaders;
- audit, review, persistence, gateway, and evaluation infrastructure;
- generic preprocessing and retrieval mechanisms;
- plugin discovery and runtime composition.

Contribute to a private or third-party vertical when a change contains business
entities, taxonomy labels, rules, prompts, extraction logic, company data, or
domain examples.

If a vertical exposes a missing generic capability, propose the smallest
vertical-neutral contract here first. Add neutral positive and negative tests,
then consume that contract from the vertical package.

## Development

```bash
python3.11 -m venv .venv
.venv/bin/pip install -e '.[dev]'
cp config.example.yml config.yml
.venv/bin/pytest -q
.venv/bin/ruff check .
.venv/bin/mypy src/mailagent
MAILAGENT_VERTICAL__ID=example-triage .venv/bin/mailagent vertical validate --json
```

Do not commit `config.yml`, credentials, authorised mail, customer identifiers,
or production retrieval samples.
