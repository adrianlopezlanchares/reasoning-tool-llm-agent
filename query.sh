#!/bin/bash

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 \"PROMPT text\""
  exit 1
fi

PROMPT="$1"

curl -X POST "http://localhost:8009/phase2/tools" \
  -H "Content-Type: application/json" \
  -d "{\"prompt\": \"$PROMPT\"}"