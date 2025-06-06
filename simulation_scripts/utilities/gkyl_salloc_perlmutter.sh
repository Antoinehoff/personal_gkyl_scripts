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
    echo "Usage: $0 [-a ACCOUNT] [-N NODES] [-g GPUS] [-t TIME] [-T TASKS] [-n NODE_TYPE] [-q QOS] [-p]"
    echo "Run salloc with specified or default parameters."
    echo "  -a ACCOUNT      Account to use (default: $ACCOUNT)"
    echo "  -N NODES        Number of nodes (default: $NODES)"
    echo "  -g GPUS         Number of GPUs (default: $GPUS)"
    echo "  -T TASKS        Tasks per node (default: $TASKS_PER_NODE)"
    echo "  -t TIME         Wall time (default: $TIME)"
    echo "  -n NODE_TYPE    Node type: gpu or cpu (default: $NODE_TYPE)"
    echo "  -q QOS          QOS to use (default: $QOS)"
    echo "  -p              Print the command line and exit without executing"
    echo "  -h              Display this help and exit"
}

# Parse options
while getopts ":a:N:g:T:t:n:q:ph" opt; do
    case ${opt} in
        a )
            ACCOUNT=$OPTARG
            ;;
        N )
            NODES=$OPTARG
            ;;
        g )
            GPUS=$OPTARG
            ;;
        T )
            TASKS_PER_NODE=$OPTARG
            ;;
        t )
            TIME=$OPTARG
            ;;
        n )
            NODE_TYPE=$OPTARG
            if [[ "$NODE_TYPE" != "gpu" && "$NODE_TYPE" != "cpu" ]]; then
                echo "Invalid node type: $NODE_TYPE. Must be 'gpu' or 'cpu'." >&2
                exit 1
            fi
            ;;
        q )
            QOS=$OPTARG
            ;;
        p )
            PRINT_ONLY=true
            ;;
        h )
            show_help
            exit 0
            ;;
        \? )
            echo "Invalid option: -$OPTARG" >&2
            exit 1
            ;;
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

# Print the command if requested
if $PRINT_ONLY; then
    echo "$COMMAND"
    exit 0
fi

# Run the salloc command
$COMMAND
