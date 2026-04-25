#!/bin/bash
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CV_PATH="$REPO_ROOT/agent-data/master-cv.md"
DRY_RUN=false

# Parse arguments
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
fi

echo "============================================================"
echo "Re-match All Jobs"
echo "============================================================"

if [[ "$DRY_RUN" == "true" ]]; then
    echo "🔍 DRY RUN MODE - No changes will be made"
    echo ""
fi

# Check if master CV exists
if [[ ! -f "$CV_PATH" ]]; then
    echo "❌ Master CV not found: $CV_PATH"
    exit 1
fi

echo "📖 Reading master CV from: $CV_PATH"
CV_LENGTH=$(wc -c < "$CV_PATH" | xargs)
echo "   ✅ Loaded $CV_LENGTH characters"
echo ""

# Get current profile stats
echo "📄 Current Profile:"
/opt/homebrew/opt/postgresql@18/bin/psql -U jeeves -d jeeves -t -c "
    SELECT 
        '   Old: ' || LENGTH(cv_text) || ' characters' ||
        E'\n   Updated: ' || updated_at
    FROM jobhunt_profiles 
    WHERE id = 'default';
"

# Get current match score stats
echo ""
echo "🔄 Current Match Scores:"
/opt/homebrew/opt/postgresql@18/bin/psql -U jeeves -d jeeves -t -c "
    SELECT 
        '   Total jobs: ' || COUNT(*) ||
        E'\n   Currently scored: ' || COUNT(CASE WHEN match_score > 0 THEN 1 END) ||
        E'\n   Score range: ' || COALESCE(MIN(match_score)::text, 'N/A') || ' - ' || COALESCE(MAX(match_score)::text, 'N/A') ||
        E'\n   Average: ' || COALESCE(ROUND(AVG(match_score)::numeric, 2)::text, 'N/A')
    FROM jobhunt_jobs 
    WHERE status != 'duplicate';
"

if [[ "$DRY_RUN" == "true" ]]; then
    echo ""
    echo "   [DRY RUN] Would update profile with new CV"
    echo "   [DRY RUN] Would reset all match scores to 0"
    echo ""
    echo "============================================================"
    echo "✅ Dry run complete - no changes made"
    echo "============================================================"
    exit 0
fi

echo ""
echo "📝 Updating profile..."

# Create a temp file with the SQL command
# We need to escape single quotes in the CV text
TEMP_SQL=$(mktemp)
cat > "$TEMP_SQL" << 'EOSQL'
UPDATE jobhunt_profiles 
SET cv_text = :'cv_content',
    cv_embedding = NULL,
    updated_at = CURRENT_TIMESTAMP
WHERE id = 'default';
EOSQL

# Read CV and update profile
CV_CONTENT=$(cat "$CV_PATH")
/opt/homebrew/opt/postgresql@18/bin/psql -U jeeves -d jeeves \
    -v cv_content="$CV_CONTENT" \
    -f "$TEMP_SQL" \
    -q

rm "$TEMP_SQL"

echo "   ✅ Profile updated"

echo ""
echo "🧹 Resetting match scores..."

RESET_COUNT=$(/opt/homebrew/opt/postgresql@18/bin/psql -U jeeves -d jeeves -t -A -c "
    WITH reset AS (
        UPDATE jobhunt_jobs 
        SET match_score = 0,
            match_reasons = '[]'::jsonb,
            embedding = NULL,
            updated_at = CURRENT_TIMESTAMP
        WHERE status != 'duplicate'
        RETURNING 1
    )
    SELECT COUNT(*) FROM reset;
")

echo "   ✅ Reset $RESET_COUNT job scores to 0"

echo ""
echo "🚀 Triggering re-matching..."
echo "   Note: Scores will be recalculated on-demand when jobs are viewed"
echo "   Or you can trigger via API: POST /api/jobs/recalculate-scores"

echo ""
echo "============================================================"
echo "✅ Profile updated and match scores reset"
echo "   $RESET_COUNT jobs queued for re-scoring"
echo "============================================================"
