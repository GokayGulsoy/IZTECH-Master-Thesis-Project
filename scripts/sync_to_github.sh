#!/usr/bin/env bash
set -euo pipefail

# Sync critical experiment outputs to GitHub for ephemeral cloud instances.
# Usage:
#   scripts/sync_to_github.sh
#   COMMIT_MSG="checkpoint sync" scripts/sync_to_github.sh
#
# Optional env vars:
#   REPO_DIR: repository root (default: script parent parent)
#   INCLUDE_PATHS: space-separated paths to add (default: results checkpoints logs/matrix)

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
INCLUDE_PATHS="${INCLUDE_PATHS:-results checkpoints logs/matrix}"
COMMIT_MSG="${COMMIT_MSG:-auto-sync: checkpoints/results $(date -Iseconds)}"
LOCK_FILE="${REPO_DIR}/.git/.sync_to_github.lock"

if [[ ! -d "${REPO_DIR}/.git" ]]; then
  echo "[sync] Not a git repository: ${REPO_DIR}" >&2
  exit 1
fi

# Prevent overlapping sync runs.
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "[sync] Another sync is already running. Exiting."
  exit 0
fi

cd "${REPO_DIR}"

# Ensure git identity is configured.
if [[ -z "$(git config --get user.name || true)" ]] || [[ -z "$(git config --get user.email || true)" ]]; then
  echo "[sync] git user.name / user.email is not configured." >&2
  echo "[sync] Run:" >&2
  echo "       git config user.name \"Your Name\"" >&2
  echo "       git config user.email \"you@example.com\"" >&2
  exit 1
fi

# Stage only existing include paths.
added_any=0
for p in ${INCLUDE_PATHS}; do
  if [[ -e "${p}" ]]; then
    git add -A -- "${p}"
    added_any=1
  fi
done

if [[ "${added_any}" -eq 0 ]]; then
  echo "[sync] No configured paths exist yet (${INCLUDE_PATHS})."
  exit 0
fi

# Nothing to commit.
if git diff --cached --quiet; then
  echo "[sync] No changes to commit."
  exit 0
fi

# Commit and push current branch.
git commit -m "${COMMIT_MSG}"
current_branch="$(git rev-parse --abbrev-ref HEAD)"
git push origin "${current_branch}"

echo "[sync] Pushed to origin/${current_branch}"
