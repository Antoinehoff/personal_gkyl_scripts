#!/bin/bash
# Continuously refreshing squeue monitor
# Usage: gkyl_check_queue.sh [refresh_interval_in_seconds]

INTERVAL="${1:-5}"
USER="ah1032"
FORMAT="%.9i %.10P %.48j %.2t %.9M %.6D %.16R"

trap 'tput cnorm; clear; exit 0' INT TERM

tput civis  # hide cursor

while true; do
    OUTPUT=$(squeue -u "$USER" --format="$FORMAT" --sort=i 2>&1)
    tput cup 0 0  # move cursor to top-left
    tput ed        # clear from cursor to end of screen
    echo "=== Queue for $USER === (refreshing every ${INTERVAL}s, Ctrl+C to quit)"
    echo ""
    echo "$OUTPUT"
    sleep "$INTERVAL"
done
