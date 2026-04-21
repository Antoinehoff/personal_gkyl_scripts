#!/bin/bash
# Continuously refreshing squeue monitor
# Usage: perlmutter_queue.sh [-r SECONDS] [-h]

INTERVAL=5
USER="ah1032"
FORMAT="%.9i %.10P %.48j %.2t %.9M %.6D %.16R"

show_help() {
    echo "Usage: $0 [-r SECONDS] [-h]"
    echo ""
    echo "Continuously refreshing squeue monitor for user $USER."
    echo ""
    echo "Options:"
    echo "  -r SECONDS    Refresh interval in seconds (default: $INTERVAL)"
    echo "  -h            Show this help message and exit"
}

while getopts ":r:h" opt; do
    case $opt in
        r) INTERVAL="$OPTARG" ;;
        h) show_help; exit 0 ;;
        :) echo "Error: -$OPTARG requires an argument." >&2; exit 1 ;;
        \?) echo "Invalid option: -$OPTARG" >&2; exit 1 ;;
    esac
done

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
