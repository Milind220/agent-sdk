# Releasing

## One-time setup

1. Create a PyPI account and verify email.
2. Create a pending project for `agent-sdk` (or publish first release manually once).
3. In PyPI project settings, add a trusted publisher:
   - Owner: `Milind220`
   - Repository: `agent-sdk`
   - Workflow: `publish-pypi.yml`
   - Environment: `pypi`
4. In GitHub repo settings, create environment `pypi` (no secrets needed).

## Release flow

1. Bump `version` in `pyproject.toml`.
2. Commit + push to `main`.
3. Tag release: `git tag vX.Y.Z && git push origin vX.Y.Z`.
4. GitHub Action `.github/workflows/publish-pypi.yml` builds + publishes to PyPI.

## Verify

1. Confirm workflow run is green.
2. Confirm package page: `https://pypi.org/project/agent-sdk/`.
