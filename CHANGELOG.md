# Changelog

All notable changes to MailAgent are documented here. The project follows
[Semantic Versioning](https://semver.org/) from the first stable release; while
the package is pre-1.0, minor releases may contain contract changes.

## Unreleased

### Added

- Durable database outbox and idempotent background-job dispatch.
- API-key roles, trusted audit actors, truthful readiness checks, Worker
  heartbeat, and request correlation IDs.
- Versioned vertical plugin ABI validation and selected-plugin lazy loading.
- Wheel-installed migration assets and container/package smoke checks.

### Changed

- Exclusive-label behavior and precision overrides now come from vertical
  configuration instead of Core-owned label names.
- Bootstrap HTTP operations without executable implementations are no longer
  advertised; supported bootstrap workflows remain available through the CLI.

### Security

- Production configuration fails closed when API authentication is disabled.
- Run state transitions use compare-and-set guards to prevent stale concurrent
  workers or reviewers from overwriting terminal state.
