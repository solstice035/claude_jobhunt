#!/usr/bin/env bash
# reset-test-data.sh — Clears all agent-generated test data without touching the Adzuna job DB
# Safe to run anytime. Does NOT delete the master CV or core config.
#
# Usage: ./scripts/reset-test-data.sh [--full]
#   --full: also resets the SQLite DB (re-scrapes needed after)

set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
AGENT_DATA="$PROJECT_DIR/agent-data"

echo "🧹 Job Hunt Test Data Reset"
echo "==========================="

# 1. Agent-generated files (leads, CVs, cover letters, briefs)
echo ""
echo "Clearing agent-generated files..."
for dir in pipeline/leads pipeline/active pipeline/applied pipeline/rejected \
           pipeline/materials cv/tailored cover-letters research briefs \
           interview-prep company-research outreach reports reports/weekly \
           networking/drafts networking/sent scripts templates; do
    target="$AGENT_DATA/$dir"
    if [ -d "$target" ]; then
        count=$(find "$target" -type f ! -name '.gitkeep' | wc -l | tr -d ' ')
        if [ "$count" -gt 0 ]; then
            find "$target" -type f ! -name '.gitkeep' -delete
            echo "  ✅ $dir — $count files removed"
        else
            echo "  ⏭  $dir — already clean"
        fi
    fi
done

# 2. Keep master-cv.md (it's the input, not output)
echo ""
if [ -f "$AGENT_DATA/master-cv.md" ]; then
    echo "📄 master-cv.md preserved (not test data)"
fi

# 3. Keep knowledge/target-profile.md
if [ -f "$AGENT_DATA/knowledge/target-profile.md" ]; then
    echo "📄 knowledge/target-profile.md preserved"
fi

# 4. Slack message history — nothing to clear (Slack retains its own)

# 5. Full reset: nuke the SQLite DB
if [ "${1:-}" = "--full" ]; then
    echo ""
    echo "⚠️  Full reset: removing SQLite database..."
    if [ -f "$PROJECT_DIR/data/jobs.db" ]; then
        rm "$PROJECT_DIR/data/jobs.db"
        echo "  ✅ jobs.db removed — restart Docker to recreate, then trigger /jobs/refresh"
    fi
else
    echo ""
    echo "ℹ️  SQLite DB (717 jobs) preserved. Use --full to also reset the DB."
fi

# 6. Agent memory/logs in workspaces
echo ""
echo "Clearing agent workspace memory files..."
for ws in workspace-coordinator workspace-market-intel workspace-app-tracker \
          workspace-cv-tailor workspace-cover-letter; do
    mem_dir="$HOME/.openclaw/$ws/memory"
    if [ -d "$mem_dir" ]; then
        count=$(find "$mem_dir" -type f | wc -l | tr -d ' ')
        if [ "$count" -gt 0 ]; then
            find "$mem_dir" -type f -delete
            echo "  ✅ $ws/memory — $count files cleared"
        else
            echo "  ⏭  $ws/memory — already clean"
        fi
    fi
done

echo ""
echo "✅ Done. Live Adzuna data intact. Agent outputs cleared."
echo "   To re-scrape jobs: curl -X POST http://localhost:8000/jobs/refresh"
