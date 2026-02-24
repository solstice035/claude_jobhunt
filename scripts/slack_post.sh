#!/bin/bash
# Post a message to a Slack channel
# Usage: ./slack_post.sh <channel> <message>
# Channels: briefing, daily, general, overnight-sprints-channel, reading-list
# Or pass a channel ID directly (starts with C)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/../.env"

if [ -f "$ENV_FILE" ]; then
    SLACK_BOT_TOKEN=$(grep SLACK_BOT_TOKEN "$ENV_FILE" | cut -d= -f2)
fi

SLACK_BOT_TOKEN="${SLACK_BOT_TOKEN:?SLACK_BOT_TOKEN not set}"

# Channel name to ID mapping
declare -A CHANNELS=(
    [briefing]="C0ACJQZMP6H"
    [daily]="C0ADGFVC6QG"
    [general]="C07DDJTSHL4"
    [overnight-sprints-channel]="C0ACTBJRRMW"
    [reading-list]="C0ADXQ4M0GZ"
    [status]="C07DB1AN3JP"
)

CHANNEL_INPUT="${1:?Usage: slack_post.sh <channel> <message>}"
MESSAGE="${2:?Usage: slack_post.sh <channel> <message>}"

# Resolve channel name to ID
if [[ "$CHANNEL_INPUT" == C* ]]; then
    CHANNEL_ID="$CHANNEL_INPUT"
else
    CHANNEL_ID="${CHANNELS[$CHANNEL_INPUT]:?Unknown channel: $CHANNEL_INPUT}"
fi

# Post message
curl -s -X POST \
    -H "Authorization: Bearer $SLACK_BOT_TOKEN" \
    -H "Content-Type: application/json; charset=utf-8" \
    -d "$(python3 -c "import json; print(json.dumps({'channel': '$CHANNEL_ID', 'text': '''$MESSAGE'''}))")" \
    'https://slack.com/api/chat.postMessage' | python3 -c "
import json, sys
resp = json.load(sys.stdin)
if resp.get('ok'):
    print(f\"Posted to {resp['channel']} (ts={resp['ts']})\" )
else:
    print(f\"ERROR: {resp.get('error', 'unknown')}\", file=sys.stderr)
    sys.exit(1)
"
