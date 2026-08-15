# MailAgent Vertical Plugin Template

This directory is a copyable example of an independently installable MailAgent
vertical plugin. It deliberately uses a business-neutral support-triage profile.

The Python package provides executable behavior through the
`mailagent.verticals` entry-point group. The YAML and JSON files under
`verticals/example_plugin/` remain external, editable deployment assets and are
not bundled into either the MailAgent Core wheel or this plugin wheel.

## Create a plugin

1. Copy this directory into a new repository.
2. Rename `example-plugin`, `example_plugin`, `mailagent-example-plugin`, and
   `mailagent_example_plugin` consistently.
3. Replace the neutral taxonomy, rules, and schema with reviewed business
   assets.
4. Add executable enrichers in `src/` only when the profile cannot express the
   required behavior.
5. Keep credentials, real mail, private registries, and evaluation corpora out
   of Git.

## Develop and validate

From the MailAgent repository root:

```bash
uv sync --all-packages --all-extras
PYTHONPATH=examples/vertical-plugin-template/src \
  uv run pytest -q examples/vertical-plugin-template/tests
uv build --wheel --project examples/vertical-plugin-template
```

After installing the plugin, select it and point Core at its external profile:

```yaml
vertical:
  id: example-plugin
  verticals_path: /absolute/path/to/vertical-plugin-template/verticals
```

Then validate the installed code/profile pairing:

```bash
MAILAGENT_VERTICAL__ID=example-plugin \
MAILAGENT_VERTICAL__VERTICALS_PATH=/absolute/path/to/vertical-plugin-template/verticals \
mailagent vertical validate --json
```
