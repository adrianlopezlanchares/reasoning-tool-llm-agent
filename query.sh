#!/bin/bash

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 \"PROMPT text\""
fi

PROMPT="$1"

# curl -X POST "https://affine-nonastronomically-evelynn.ngrok-free.dev/phase2/tools" \
curl -X POST "http://localhost:8009/phase2/tools" \
  -H "Content-Type: application/json" \
  -d "{\"prompt\": \"$PROMPT\"}" | jq
echo ""
