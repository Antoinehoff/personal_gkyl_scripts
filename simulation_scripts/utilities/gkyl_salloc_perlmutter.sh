#!/bin/bash

# Default values
ACCOUNT="m4564"
NODES=1
GPUS=4
TASKS_PER_NODE=4
TIME="04:00:00"
NODE_TYPE="gpu"  # Default node type
QOS="interactive"
PRINT_ONLY=false  # Default: do not just print the command

# Help function
show_help() {
    echo "Usage: $0 [-a ACCOUNT] [-N NODES] [-g GPUS] [-t TIME] [-T TASKS] [-q QOS] [--gpu|--cpu] [-p]"
    echo "Run salloc with specified or default parameters."
    echo "  -a ACCOUNT      Account to use (default: $ACCOUNT)"
    echo "  -N NODES        Number of nodes (default: $NODES)"
    echo "  -g GPUS         Number of GPUs (default: $GPUS)"
    echo "  -T TASKS        Tasks per node (default: $TASKS_PER_NODE)"
    echo "  -t TIME         Wall time (default: $TIME)"
    echo "  --gpu           Use GPU nodes (default)"
    echo "  --cpu           Use CPU nodes"
    echo "  -q QOS          QOS to use (default: $QOS)"
    echo "  -p              Print the command line and exit without executing"
    echo "  -h              Display this help and exit"
}

# Parse options (GNU getopt for long option support)
PARSED=$(getopt -o a:N:g:T:t:q:ph --long gpu,cpu,account:,nodes:,gpus:,tasks:,time:,qos:,print,help -n "$0" -- "$@")
if [[ $? -ne 0 ]]; then
    exit 1
fi
eval set -- "$PARSED"

while true; do
    case "$1" in
        -a|--account)       ACCOUNT="$2";         shift 2 ;;
        -N|--nodes)         NODES="$2";           shift 2 ;;
        -g|--gpus)          GPUS="$2";            shift 2 ;;
        -T|--tasks)         TASKS_PER_NODE="$2";  shift 2 ;;
        -t|--time)          TIME="$2";            shift 2 ;;
        -q|--qos)           QOS="$2";             shift 2 ;;
        --gpu)              NODE_TYPE="gpu";       shift   ;;
        --cpu)              NODE_TYPE="cpu";       shift   ;;
        -p|--print)         PRINT_ONLY=true;       shift   ;;
        -h|--help)          show_help; exit 0      ;;
        --)                 shift; break           ;;
        *)                  echo "Invalid option: $1" >&2; exit 1 ;;
    esac
done

# Construct the salloc command
if [[ "$NODE_TYPE" == "gpu" ]]; then
    CONSTRAINT="gpu"
    COMMAND="salloc --account $ACCOUNT --nodes $NODES --gpus $GPUS --tasks-per-node=$TASKS_PER_NODE --time=$TIME --constraint=$CONSTRAINT --qos=$QOS"
else
    CONSTRAINT="cpu"
    COMMAND="salloc --account $ACCOUNT --nodes $NODES --time=$TIME --constraint=$CONSTRAINT --qos=$QOS"
fi

# Print the command
echo "$COMMAND"

# Stop if print only
if $PRINT_ONLY; then
    exit 0
fi

# Run the salloc command
$COMMAND
