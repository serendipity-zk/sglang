#!/bin/bash
# Script to set TPOT (Time Per Output Token) SLO on SGLang workers
# Usage: ./set_slo.sh [tpot_ms] [ports]
#   tpot_ms: TPOT value in milliseconds (default: 30)
#   ports: comma-separated list of worker ports (default: 31001,31002)

# Default values
TPOT_MS=${1:-30}
PORTS=${2:-"31001,31002"}
HOST=${HOST:-"localhost"}

# Convert comma-separated ports to array
IFS=',' read -ra PORT_ARRAY <<< "$PORTS"

echo "Setting TPOT to ${TPOT_MS}ms on workers..."
echo ""

# Track success/failure
SUCCESS_COUNT=0
FAIL_COUNT=0

# Loop through each port and send the curl request
for PORT in "${PORT_ARRAY[@]}"; do
    PORT=$(echo "$PORT" | xargs) # trim whitespace
    URL="http://${HOST}:${PORT}/set_tpot"

    echo -n "Worker at port ${PORT}: "

    # Send the request and capture response
    RESPONSE=$(curl -s -X POST "$URL" \
        -H "Content-Type: application/json" \
        -d "{\"tpot\": ${TPOT_MS}}" \
        -w "\n%{http_code}" 2>&1)

    # Extract status code (last line) and body (everything else)
    HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
    BODY=$(echo "$RESPONSE" | head -n-1)

    if [ "$HTTP_CODE" = "200" ]; then
        echo "✓ SUCCESS"
        ((SUCCESS_COUNT++))
    else
        echo "✗ FAILED (HTTP $HTTP_CODE)"
        if [ -n "$BODY" ]; then
            echo "  Response: $BODY"
        fi
        ((FAIL_COUNT++))
    fi
done

echo ""
echo "Summary: $SUCCESS_COUNT succeeded, $FAIL_COUNT failed"

# Exit with error if any failed
[ $FAIL_COUNT -eq 0 ] && exit 0 || exit 1
