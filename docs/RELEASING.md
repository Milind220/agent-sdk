# Releasing

## One-time setup

1. Create/verify PyPI account.
2. In PyPI, add trusted publisher for `agent-sdk`:
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
3. Check PyPI: `https://pypi.org/project/agent-sdk/`.
