#!/bin/bash

# Script to clean distribution function files, keeping only frames at 100-frame intervals
# Keeps frames: 0, 100, 200, 300, etc.
# Cleans for both "ion" and "elc" species

set -e

CLEAN_CMD="sh gkyl_clean_out.sh"

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Clean distribution function files, keeping only every 100th frame (0, 100, 200, 300, etc.)
for both ion and elc species.

Options:
    -d <dir>        Directory to clean (default: current directory)
    -i <interval>   Frame interval to keep (default: 10)
    -s <species>    Species to clean (default: "ion elc")
    -p <prefix>     File prefix pattern (default: auto-detect)
    -n              Dry run - show what would be deleted without deleting
    -v              Verbose - show file detection details
    -h, --help      Display this help message

Examples:
    $0                      # Clean current directory with default settings
    $0 -d /path/to/sim      # Clean specific directory
    $0 -i 50                # Keep every 50th frame instead of 10
    $0 -s "ion"             # Clean only ion files
    $0 -n                   # Preview what would be deleted
    $0 -v                   # Verbose mode to debug file detection

EOF
    exit 0
}

# Default values
WORK_DIR="."
INTERVAL=10
SPECIES="ion elc"
PREFIX=""
DRY_RUN=0
VERBOSE=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d)
            WORK_DIR="$2"
            shift 2
            ;;
        -i)
            INTERVAL="$2"
            shift 2
            ;;
        -s)
            SPECIES="$2"
            shift 2
            ;;
        -p)
            PREFIX="$2"
            shift 2
            ;;
        -n)
            DRY_RUN=1
            shift
            ;;
        -v)
            VERBOSE=1
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Verify directory exists
if [ ! -d "$WORK_DIR" ]; then
    echo "Error: Directory '$WORK_DIR' does not exist"
    exit 1
fi

cd "$WORK_DIR"

echo "========================================"
echo "Distribution Function Cleanup"
echo "========================================"
echo "Directory:     $PWD"
echo "Species:       $SPECIES"
echo "Keep Interval: Every ${INTERVAL}th frame"
echo "Dry Run:       $([ $DRY_RUN -eq 1 ] && echo "YES" || echo "NO")"
echo "Verbose:       $([ $VERBOSE -eq 1 ] && echo "YES" || echo "NO")"
echo "========================================"
echo ""

# Function to detect file prefix for a species
detect_prefix() {
    local species=$1
    # Look ONLY for files with pattern: *-species_XX.gkyl
    local sample_file=$(ls *-${species}_[0-9]*.gkyl 2>/dev/null | head -n 1)
    
    if [ -z "$sample_file" ]; then
        echo ""
        return
    fi
    
    # Extract prefix by removing -species_XX.gkyl pattern
    # Example: mysim-ion_13.gkyl -> mysim
    local prefix=$(echo "$sample_file" | sed -E "s/-${species}_[0-9]+\.gkyl$//")
    echo "$prefix"
}

# Function to find the maximum frame number for a given species
find_max_frame() {
    local species=$1
    local prefix=$2
    local max_frame=-1
    
    # Build search pattern - ONLY *-species_XX.gkyl
    if [ -n "$prefix" ]; then
        pattern="${prefix}-${species}_[0-9]*.gkyl"
    else
        pattern="*-${species}_[0-9]*.gkyl"
    fi
    
    if [ $VERBOSE -eq 1 ]; then
        echo "  [DEBUG] Searching for pattern: $pattern"
    fi
    
    # Find all matching files and extract frame numbers
    for file in $pattern; do
        if [ -f "$file" ]; then
            # Extract the frame number
            frame=$(echo "$file" | grep -oP "${species}_\K\d+(?=\.gkyl)" | head -n 1)
            
            if [ $VERBOSE -eq 1 ] && [ -n "$frame" ]; then
                echo "  [DEBUG] File: $file -> Frame: $frame"
            fi
            
            if [ -n "$frame" ] && [ "$frame" -gt "$max_frame" ]; then
                max_frame=$frame
            fi
        fi
    done
    
    echo $max_frame
}

# Process each species
for spec in $SPECIES; do
    echo "Processing species: $spec"
    
    # Auto-detect prefix if not provided
    if [ -z "$PREFIX" ]; then
        detected_prefix=$(detect_prefix "$spec")
        if [ $VERBOSE -eq 1 ]; then
            echo "  [DEBUG] Auto-detected prefix: '$detected_prefix'"
        fi
    else
        detected_prefix="$PREFIX"
    fi
    
    # Find maximum frame number
    max_frame=$(find_max_frame "$spec" "$detected_prefix")
    
    if [ "$max_frame" -eq -1 ]; then
        echo "  No files found for species '$spec'"
        if [ $VERBOSE -eq 1 ]; then
            echo "  [DEBUG] Looking for files matching: *${spec}_*.gkyl"
            echo "  [DEBUG] Files in directory:"
            ls -1 *${spec}*.gkyl 2>/dev/null | head -n 5 || echo "  [DEBUG] No matching files"
        fi
        echo ""
        continue
    fi
    
    echo "  Found frames 0 to $max_frame"
    
    # Determine the file prefix for gkyl_clean_out command
    # Always use hyphen separator (since we only look for *-species_XX.gkyl)
    if [ -n "$detected_prefix" ]; then
        clean_prefix="${detected_prefix}-${spec}"
    else
        # If no prefix detected, just use "-species" format
        clean_prefix="-${spec}"
    fi

    if [ $VERBOSE -eq 1 ]; then
        echo "  [DEBUG] Using prefix for deletion: '$clean_prefix'"
    fi
    
    # Calculate ranges to delete
    # Keep: 0, INTERVAL, 2*INTERVAL, 3*INTERVAL, ...
    # Delete everything between these frames
    
    frame=0
    total_deleted=0
    
    while [ $frame -lt $max_frame ]; do
        next_keep=$((frame + INTERVAL))
        
        # Delete frames between (frame+1) and (next_keep-1)
        start_delete=$((frame + 1))
        end_delete=$((next_keep - 1))
        
        if [ $start_delete -le $end_delete ] && [ $end_delete -le $max_frame ]; then
            if [ $DRY_RUN -eq 1 ]; then
                echo "  [DRY RUN] Would delete: ${clean_prefix} frames $start_delete to $end_delete"
                num_files=$((end_delete - start_delete + 1))
                total_deleted=$((total_deleted + num_files))
            else
                echo "  Deleting: ${clean_prefix} frames $start_delete to $end_delete"
                $CLEAN_CMD "$clean_prefix" $start_delete $end_delete
                num_files=$((end_delete - start_delete + 1))
                total_deleted=$((total_deleted + num_files))
            fi
        fi
        
        frame=$next_keep
    done
    
    # Keep frames to display
    kept_frames=""
    frame=0
    count=0
    while [ $frame -le $max_frame ] && [ $count -lt 10 ]; do
        kept_frames="$kept_frames $frame"
        frame=$((frame + INTERVAL))
        count=$((count + 1))
    done
    if [ $frame -le $max_frame ]; then
        kept_frames="$kept_frames ..."
    fi
    
    echo "  Keeping frames:$kept_frames"
    echo "  Deleted: ~$total_deleted files"
    echo ""
done

if [ $DRY_RUN -eq 1 ]; then
    echo "========================================"
    echo "DRY RUN COMPLETE - No files were deleted"
    echo "Run without -n flag to actually delete files"
    echo "========================================"
else
    echo "========================================"
    echo "Cleanup Complete"
    echo "========================================"
fi
