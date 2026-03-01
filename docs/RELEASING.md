# Releasing

## One-time setup

1. Create/verify PyPI account.
2. In PyPI, add trusted publisher for `the-agent-sdk`:
   - Owner: `Milind220`
   - Repository: `agent-sdk`
   - Workflow: `publish-pypi.yml`
   - Environment: `pypi`
3. In GitHub repo settings, create environment `pypi`.

## Automated release behavior

- Workflow: `.github/workflows/publish-pypi.yml`
- Triggers: pushes to `main` and `alpha`
- Versioning engine: Python Semantic Release (Conventional Commits)

### Version bump rules

- `feat:` -> minor
- `fix:` / `perf:` -> patch
- `feat!:` / `fix!:` / `BREAKING CHANGE:` -> major (for 1.x+)
- `docs:` / `chore:` / `ci:` / `style:` / `refactor:` / `test:` -> no release

### Prerelease behavior

- Branch `main`: stable releases
- Branch `alpha`: prereleases with `-alpha.N` suffix

## First publish target

- Current calculated next release is `v0.1.0`.
- No manual tag/release required.

## Day-to-day flow

1. Merge Conventional Commit PRs to `main` (or `alpha` for prereleases).
2. Workflow auto-detects next version, creates tag + GitHub release, builds package.
3. If a releasable commit exists, package auto-publishes to PyPI via trusted publisher.

## Verify

1. Check GitHub Actions run for `publish-pypi.yml`.
2. Check GitHub Releases for new tag.
3. Check PyPI: `https://pypi.org/project/the-agent-sdk/`.

## One-time manual publish (for current 0.1.0)

If you need to publish the current `pyproject.toml` version without creating a new semantic-release tag:

1. Run workflow `Release and Publish`.
2. Set input `force_publish_current=true`.
3. This publishes current package version as-is (currently `0.1.0`).
