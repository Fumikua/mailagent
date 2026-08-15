# MailAgent

A public, vertical-agnostic mail classification and automation runtime. Core owns
generic classification, fusion, audit, and a mail-understanding pipeline; each
**vertical** remains an independently buildable plugin that contributes executable
enrichers plus an externally editable business profile (taxonomy, rules, signals,
schemas).

> The shipped `example_triage` vertical is a minimal sample (3-label taxonomy) for
> demonstration. Build your own vertical package for real business domains.

## Current Capabilities

- **Pluggable classification**: feature-flagged `Rules → Vector RAG → LLM` fusion is
  built from the selected vertical's manifest-owned assets; when disabled, the
  generic LLM cascade remains available for backward compatibility.
- **Mail-understanding pipeline**: `ClassificationOrchestrator` owns source
  selection, fallback, audit, and review semantics. `MailUnderstandingPipeline`
  then runs the selected vertical's enrichers and returns one versioned
  `ClassificationResponse`.
- **Versioned vertical data**: business output lives in
  `ClassificationResponse.data.<namespace>` and is checked against the selected
  vertical's JSON Schema before it is persisted. A bad enrichment is isolated,
  recorded, and held for review.
- **Vertical plugin architecture**: executable vertical behavior is resolved from
  installed Python plugins; externally editable business profiles own taxonomy,
  rules, signals, RAG declarations, patterns, and schemas. Run
  `mailagent vertical validate` to check the installed plugin/profile match plus
  taxonomy, rules, signals, schemas, patterns, target profiles, RAG declarations,
  and retrieval policy before restarting a Worker.
- **Async processing** (arq + Redis): `POST /api/v1/runs` returns 202 + `pending`
  immediately; the Worker resolves installed code by vertical ID, validates the
  matching external business profile, and persists the complete classification
  envelope (including orchestration audit). The P0 baseline sets
  `waiting_approval`; it does not automatically accept a classification.
- **Anchor-mapped confidence calibration**: raw LLM confidence → mid-of-bucket
  (0.95→0.975 / 0.80→0.87 / 0.60→0.70 / <0.60→0.30); `calibration_log` preserves
  raw + calibrated + anchor for replay evaluation.
- **Auditable high-risk actions**: send / forward / delete are always proposed,
  never executed. Blocked by the policy engine; require human approval.
- **Three-path fusion** (`vector-similarity-path-b`, feature-flagged):
  `RuleClassifier` + `VectorClassifier` + `LLMClassifier` are three independent
  implementations of one `Classifier` Protocol, orchestrated by
  `FusionOrchestrator`. Five fusion strategies (`rule_only` /
  `rule_vector_confirmed` / `vector_only` / `llm_fallback` / `all_low_review`);
  `fusion.enabled=false` falls back to the single Cascade path. New classifiers
  only need to implement the Protocol.
- **Inbound mail gateway** (feature-flagged): polls one TLS IMAP or POP3 mailbox on
  a cron schedule and creates normal `PENDING` classification runs. Disabled by
  default. IMAP defaults to `from_now` (imports no backlog, records the current
  highest UID); POP3 uses an incremental baseline. Atomic dedup via
  `INSERT ... ON CONFLICT DO NOTHING`; `BODY.PEEK[]` never marks mail as read.
- **Bootstrap pipeline**: Stage 1 seed annotation and Stage 2 tiered import are
  runnable via CLI (`mailagent bootstrap seed/import/review/confirm`), with
  markdown/JSON reports, 12-month archival, periodic HDBSCAN clustering, and rule
  auto-learning. Bootstrap generates and reviews retrieval text and sample quality
  before calling embeddings; it does not train embedding models.
- **Mail preprocessing**: generic subject normalization (Re/Fwd/【外部邮件】 prefix
  stripping + NBSP folding) + thread parsing (7 quote-block patterns + 0.6/0.4
  recency-biased embedding fusion) + auditable `RetrievalDocument` builder (HTML →
  text, signature/disclaimer/quote-history cleaning, length caps, eligibility) + a
  vertical extension point for domain-specific field extraction.
- **Vector RAG sample governance**: `SampleQualityAssessment` admission checks
  (content eligibility / duplicate fingerprint / flat-taxonomy category validation
  / disposition reason) + provenance fields on the sample table + reviewer
  retrieval-text rewrite and re-embedding + an `ambiguous_candidates` no-match
  branch that retains an LLM/human fallback.

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python ≥ 3.11 |
| API | FastAPI + uvicorn |
| Async queue | arq + Redis |
| Persistence | SQLAlchemy 2 (async) + SQLite (dev) / PostgreSQL (prod) |
| Migrations | Alembic |
| LLM | OpenAI-compatible API (DeepSeek / Qwen / OpenAI / Ollama) via `httpx.AsyncClient` + tenacity retry |
| Orchestration | Async Worker pipeline through arq |
| Validation | pydantic v2 / pydantic-settings |
| Lint / type | ruff, mypy |
| Tests | pytest + pytest-asyncio |
| Container | Docker + Compose (api / worker / postgres / redis) |

## Project Structure

```
mailagent/
├── src/mailagent/
│   ├── domain/           # Pydantic models + policy + calibration
│   ├── classification/   # Generic classification foundation: contracts, taxonomy, Rules/Vector/LLM
│   ├── core/             # Fusion, audit, version coordination, mail-understanding pipeline
│   ├── preprocessing/    # Subject normalization, thread parsing, retrieval document
│   ├── verticals/        # Plugin contract/registry/selection + example_triage implementation
│   ├── llm/              # LLM + embedding provider clients
│   ├── infra/            # Config, store, vector store, bootstrap, clustering, rule learner, CLI, queue, worker
│   ├── gateway/          # Inbound IMAP/POP3 polling (feature-flagged)
│   └── api/              # HTTP entrypoints and run lifecycle service
├── migrations/           # Alembic migrations
├── verticals/
│   └── example_triage/   # Sample vertical: manifest, taxonomy, rules, data-schema
├── examples/
│   └── vertical-plugin-template/ # Copyable, independently buildable public plugin example
├── config.example.yml
├── compose.yaml
└── pyproject.toml
```

To create a separately installable vertical, copy
[`examples/vertical-plugin-template`](examples/vertical-plugin-template). It
demonstrates the Python entry point, external profile layout, tests, and an
independent wheel without adding the example package to the Core workspace or
Core wheel.

## Quick Start

### 1. Install dependencies

```bash
cp config.example.yml config.yml
uv sync --all-packages --all-extras
```

### 2. Start Redis (required by the Worker)

```bash
docker run -d -p 6379:6379 --name redis redis:7-alpine
```

### 3. Start the API

```bash
.venv/bin/python -m uvicorn mailagent.api.main:app --reload
```

Open `http://127.0.0.1:8000/docs`.

### 4. Start the Worker (processes classification jobs)

```bash
.venv/bin/python -m mailagent.infra.worker
# or
.venv/bin/arq mailagent.infra.worker.WorkerSettings
```

### 5. Submit an email and poll for the classification

```bash
# Submit → returns 202 + pending + run_id
curl -X POST http://127.0.0.1:8000/api/v1/runs \
  -H 'content-type: application/json' \
  -d '{"email":{"message_id":"req-1","sender":"manager@example.com","subject":"Please approve the Q3 budget by Friday","body":"Hi, I need your approval on the attached Q3 budget before Friday's meeting."}}'

# Poll GET /runs/{run_id}
curl http://127.0.0.1:8000/api/v1/runs/<run_id>
```

### 6. Bootstrap Quick Start (optional, three-path fusion)

Enable three-path fusion in `config.yml` (`fusion.enabled: true`) and ensure a TEI
embedding endpoint (or OpenAI-compatible embedding API) is reachable. Then build
the sample library:

```bash
# Stage 1: seed manually-selected .eml files (50-200 representative emails)
mailagent bootstrap seed --dir ./seed_emails/

# Stage 2: import historical emails in weekly batches with tiered labeling
mailagent bootstrap import --dir ./week_2026w29/ --batch-size 50

# Review one generated report tier, then bulk-confirm eligible Tier 1 samples
mailagent bootstrap review --report-id <report_id> --tier 2
mailagent bootstrap confirm --report-id <report_id> --tier 1 --all
```

### 7. Enable inbound polling (optional)

Copy `config.example.yml` to `config.yml`, add an enabled entry under the mail
gateway section, and export the password under the env var named by `password_env`.

```bash
# .env (never commit)
MAILAGENT_IMAP_PASSWORD=your-imap-password
```

```yaml
# config.yml — minimum to enable IMAP polling
mail_gateway:
  enabled: true
  adapter: imap
  mailbox_id: primary
  host: imap.your-provider.com
  port: 993
  username: ops@your-domain.com
  password_env: MAILAGENT_IMAP_PASSWORD   # name, not the value
  mailbox: INBOX
  initial_sync_mode: from_now             # safe default — imports no backlog
```

Restart the Worker. The first poll logs `initialized: from_now` and records the
current highest UID; subsequent polls create one `PENDING` run per new message. For
public mailboxes that only support POP3, set `adapter: pop3` and
`initial_sync_mode: incremental`.

## Configuration

Copy `config.example.yml` to `config.yml` and fill in the model endpoint + name.
Never commit `config.yml` or real secrets.

### config.yml highlights

```yaml
classification:
  taxonomy_path: ./taxonomy.yaml      # resolved from the vertical manifest at runtime
  autonomy_level_default: L0
  confidence_threshold: 0.8
  auto_accept_enabled: false          # P0: every classification stays a reviewer suggestion

classification_feedback:
  mode: disabled                      # fail closed; trusted_internal only behind an identity-injecting proxy

model:
  base_url: https://api.openai.com/v1
  model_name: gpt-4.1-mini
  api_key_env: OPENAI_API_KEY
  enabled: true

vertical:
  id: example-triage                  # select exactly one installed vertical plugin
  verticals_path: ./verticals

vertical_overrides: {}                # vertical-owned runtime knobs; each vertical reads its own section
```

### Environment variables (preferred for secrets)

- `OPENAI_API_KEY` — LLM API key
- `MAILAGENT_IMAP_PASSWORD` / `MAILAGENT_POP3_PASSWORD` — mailbox passwords
- `MAILAGENT_MODEL__BASE_URL` / `MAILAGENT_MODEL__MODEL_NAME` — override model settings
- `MAILAGENT_DATABASE_URL` / `MAILAGENT_REDIS__URL` — override database / Redis
- `MAILAGENT_CLASSIFICATION_FEEDBACK__MODE` — leave unset/`disabled` for P0
- `MAILAGENT_MAIL_GATEWAY__*` — override any `mail_gateway.*` field at deploy time

### P0 trusted classification baseline

P0 requires `classification.auto_accept_enabled: false`. After all enrichers run,
the Pipeline reapplies the acceptance boundary, preserves useful labels and
immutable version/audit data, forces `meta.needs_human_review=true`, records
`p0_auto_accept_disabled`, and the Worker persists the run as `waiting_approval`.
This applies even if an enricher mutates review metadata or Rules/Vector/LLM
evidence is confident. Production auto-acceptance is not enabled in P0.

Reviewers append corrections without rewriting the original classification:

```text
POST /api/v1/runs/{run_id}/classification-feedback
GET  /api/v1/runs/{run_id}/classification-feedback
```

Both endpoints return `403` by default. Set `classification_feedback.mode:
trusted_internal` only for an admin route behind a trusted reverse proxy that
removes any client-supplied reviewer header and injects the authenticated
reviewer's opaque ID. Revisions are immutable and always ineligible for sample
proposal until a separate trusted promotion workflow exists.

## Verticals

A vertical has two halves:

1. **Executable plugin** (installable Python package under
   `src/mailagent/verticals/<id>/`): runtime code — enrichers, extractors,
   registry, anything with behavior.
2. **External business profile** (under `verticals/<id>/`): reviewed declarative
   knowledge — `manifest.yaml`, `taxonomy.yaml`, `rules/`, `signals.yaml`,
   `data-schema.json`, `target_profiles.yaml`, RAG declarations, patterns. This is
   editable without touching code.

The `example_triage` vertical ships a flat 3-label taxonomy
(`action_required` / `notification` / `noise`) and four rule YAML files. Replace
these assets with your own; Core contains no vertical examples or label-specific
conditionals.

### Add a business field without changing Core

1. Version the selected vertical's `data-schema.json` and bump `data_schema_version`
   in its `manifest.yaml`.
2. Make that vertical's enricher emit the field inside its declared namespace.
3. The pipeline validates the patch before writing
   `ClassificationResponse.data.<namespace>`; invalid output is omitted and sent to
   review.

For example, a `customer-service` vertical can own `data.customer_service.ticket_id`
without adding that field to `ClassificationResponse` or Core.

## Architecture

```
API (FastAPI)              Worker (arq)                Store (SQLite/PG)
     │                          │                          │
     │ POST /runs                │                          │
     │ → create PENDING run      │                          │
     │ → enqueue_classify ──────►│ classify_job             │
     │ ← 202 + run_id            │   ├─ get_run             │
     │                           │   ├─ MailUnderstandingPipeline
     │                           │   │   ├─ ClassificationOrchestrator
     │                           │   │   │   └─ Rules → Vector RAG → LLMClassifier
     │                           │   │   └─ selected vertical enrichers
     │                           │   └─ update_classification
     │                           │     (P0: WAITING_APPROVAL) ─►│
     │                           │                          │
     │ GET /runs/{id} ◄─────────────────────────────────────│
     │ ← {status:waiting_approval, classification:{labels,meta,
     │     data.<vertical namespace>, ...}}                   │
```

## Docker Compose

```bash
cp config.example.yml config.yml
docker compose up --build
```

Compose starts `api`, `worker`, `postgres`, and `redis`. The Worker automatically
pulls `classify_job` tasks from Redis and processes them.

## Tests

The Core suite uses only the synthetic `example_triage` plugin and includes an
architecture regression gate that rejects private-vertical vocabulary and
business fields in Core. Independently installed plugins run their own suites:

```bash
uv run pytest -q
uv run ruff check .
uv run mypy src/mailagent
MAILAGENT_VERTICAL__ID=example-triage uv run mailagent vertical validate --json
```

See [CONTRIBUTING.md](CONTRIBUTING.md) before adding a classifier, field, asset
schema, or vertical capability.

## Hard Constraints

- Core passes Ruff, mypy, and pytest; plugin packages validate against the same
  public contracts in their own repositories.
- All high-risk actions (send / forward / delete) remain proposed only — never
  executed without explicit human approval.
- Untrusted email bodies never gain instruction priority; prompt-injection
  detection routes to human review.
- Every run preserves email identifier, skill version, decision, proposed actions,
  and full trace for audit.
- Async API follows the state machine:
  `PENDING → PROCESSING → COMPLETED | FAILED | WAITING_APPROVAL | REJECTED`.
- Reviewed business configuration such as taxonomy and classification rules is
  managed through validated external profiles, not embedded in Core.
