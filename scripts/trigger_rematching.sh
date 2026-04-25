#!/bin/bash
set -e

echo "============================================================"
echo "Trigger Job Re-matching"
echo "============================================================"
echo ""
echo "ℹ️  Note: The job matching system scores jobs on-demand when"
echo "   they are viewed in the frontend. You have two options:"
echo ""
echo "1. Let natural scoring happen (recommended)"
echo "   - Jobs will be scored as you browse them"
echo "   - No server load, efficient"
echo ""
echo "2. Force immediate re-scoring (optional)"
echo "   - Requires Celery worker to be running"
echo "   - Use if you need all scores updated now"
echo ""

# Check if Celery is running
CELERY_RUNNING=$(docker ps | grep celery | wc -l)

if [[ "$CELERY_RUNNING" -eq 0 ]]; then
    echo "⚠️  Celery worker is not running"
    echo "   Scores will be calculated on-demand when viewing jobs"
    echo ""
    echo "   To start Celery worker:"
    echo "   docker-compose up -d celery"
    echo ""
else
    echo "✅ Celery worker is running"
    echo "   You can trigger immediate re-scoring via the API"
    echo ""
    echo "   To trigger re-scoring:"
    echo "   curl -X POST http://localhost:8000/api/jobs/recalculate-scores"
    echo ""
fi

# Show current stats
echo "📊 Current Match Status:"
/opt/homebrew/opt/postgresql@18/bin/psql -U jeeves -d jeeves -t -c "
    SELECT 
        '   Total jobs: ' || COUNT(*) ||
        E'\n   Scored: ' || COUNT(CASE WHEN match_score > 0 THEN 1 END) ||
        E'\n   Unscored: ' || COUNT(CASE WHEN match_score = 0 THEN 1 END)
    FROM jobhunt_jobs 
    WHERE status != 'duplicate';
"

echo ""
echo "💡 Recommendation:"
echo "   Profile has been updated with comprehensive career details."
echo "   Browse jobs in the frontend - scores will update automatically."
echo "   High-scoring roles will surface naturally as you explore."
echo ""
echo "============================================================"
