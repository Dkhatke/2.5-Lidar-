#!/usr/bin/env bash
# ============================================================
# commit.sh — Team commit helper for SIH26
#
# Usage:
#   ./commit.sh "Phase 1 complete — geometry + allocation + dashboard"
#   ./commit.sh "fix: S3 tile scoring edge case with empty tiles"
#   ./commit.sh "feat: S4 sparse semantic backend + geometry fallback"
#
# Convention (use these prefixes in messages):
#   feat:  new feature / stage implemented
#   fix:   bug fix
#   refac: refactoring (no behaviour change)
#   docs:  documentation / comments only
#   test:  adding or fixing tests
#   chore: config, deps, CI
# ============================================================

set -e   # stop on error

MSG="${1}"

if [ -z "$MSG" ]; then
  echo "❌  Usage: ./commit.sh \"your commit message\""
  exit 1
fi

echo "📦  Staging all changes..."
git add .

echo "📝  Committing: $MSG"
git commit -m "$MSG"

echo "🚀  Pushing to origin/main..."
git push origin main

echo ""
echo "✅  Done! Commit pushed to https://github.com/Dkhatke/2.5-Lidar-"
echo "    $(git log -1 --oneline)"
