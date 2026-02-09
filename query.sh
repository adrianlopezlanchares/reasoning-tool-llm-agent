#!/bin/bash

# set prompt variable to the first argument passed to the script
PROMPT="$1"

curl -X POST "http://localhost:8009/phase2/reasoning" -H "Content-Type: application/json" -d "{\"prompt\": \"$PROMPT\"}"