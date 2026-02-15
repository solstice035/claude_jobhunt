#!/usr/bin/env python3
"""Post messages to Slack channels for the job hunt agent system.

Usage:
    python3 slack_post.py <channel> <message>
    python3 slack_post.py briefing "📋 Daily Briefing — 15 Feb 2026..."
    python3 slack_post.py daily "Market scan complete: 12 new leads"

Channels: briefing, daily, general, overnight-sprints-channel, reading-list
"""

import json
import os
import sys
import urllib.request
import urllib.error
from pathlib import Path

CHANNELS = {
    "briefing": "C0ACJQZMP6H",
    "daily": "C0ADGFVC6QG",
    "general": "C07DDJTSHL4",
    "overnight-sprints-channel": "C0ACTBJRRMW",
    "reading-list": "C0ADXQ4M0GZ",
    "status": "C07DB1AN3JP",
}

def get_token():
    token = os.environ.get("SLACK_BOT_TOKEN")
    if token:
        return token
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("SLACK_BOT_TOKEN="):
                return line.split("=", 1)[1].strip()
    raise RuntimeError("SLACK_BOT_TOKEN not found")

def post_message(channel: str, text: str, blocks: list = None) -> dict:
    """Post a message to Slack. Returns the API response dict."""
    token = get_token()
    channel_id = CHANNELS.get(channel, channel)
    
    payload = {"channel": channel_id, "text": text}
    if blocks:
        payload["blocks"] = blocks
    
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "https://slack.com/api/chat.postMessage",
        data=data,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        },
    )
    
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read())
    
    if not result.get("ok"):
        raise RuntimeError(f"Slack API error: {result.get('error', 'unknown')}")
    
    return result

def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    
    channel = sys.argv[1]
    message = sys.argv[2]
    
    result = post_message(channel, message)
    print(f"✅ Posted to #{channel} (ts={result['ts']})")

if __name__ == "__main__":
    main()
