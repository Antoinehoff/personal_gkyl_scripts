import os
import itertools
import stat
import re
import glob
from datetime import datetime
import numpy as np

def parse_slurm_time_to_seconds(time_str):
    """
    Convert Slurm time format to seconds.
    Supports formats: HH:MM:SS, MM:SS, or D-HH:MM:SS
    """
    if '-' in time_str:
        # Format: D-HH:MM:SS
        days, time_part = time_str.split('-')
        parts = time_part.split(':')
        return int(days) * 86400 + int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    else:
        parts = time_str.split(':')
        if len(parts) == 3:
            # Format: HH:MM:SS
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        elif len(parts) == 2:
            # Format: MM:SS
            return int(parts[0]) * 60 + int(parts[1])
        else:
            raise ValueError(f"Unable to parse time format: {time_str}")

class ScanConfig:
    def __init__(self, job_name="scan", account="my_account", time="00:30:00", qos="regular", constraint="gpu", email="", email_type="FAIL,END", 
                 gpu_per_instance=1, num_jobs=1, execution_mode="ALL",scan_arrays=None, work_dir="scan_simulations",
                 total_gpus_per_node=4, total_cores_per_node=64, gpus_per_task=1,
                 gkyl_additional_options="", output_script="submit_scan.sh",
                 gkeyll_input_c="gkeyll.c", restart_from_last_frame=True,
                 last_frame_detector_script="", max_sim_time=1000):
        self.job_name = job_name
        self.account = account
        self.time = time
        self.qos = qos
        self.constraint = constraint
        self.email = email
        self.email_type = email_type
        self.gpu_per_instance = gpu_per_instance
        self.num_jobs = num_jobs
        self.execution_mode = execution_mode
        self.scan_arrays = scan_arrays if scan_arrays is not None else {}
        self.work_dir = work_dir
        self.total_gpus_per_node = total_gpus_per_node
        self.total_cores_per_node = total_cores_per_node
        self.gpus_per_task = gpus_per_task
        self.cpus_per_task = total_cores_per_node // total_gpus_per_node
        self.gkyl_additional_options = gkyl_additional_options
        self.output_script = output_script
        self.gkeyll_input_c = gkeyll_input_c
        self.restart_from_last_frame = restart_from_last_frame
        self.last_frame_detector_script = os.path.expanduser(last_frame_detector_script)
        self.gkeyll_exe = gkeyll_input_c.replace('.c', '')

        self.max_sim_time = max_sim_time # in microseconds

        self.table_content = ""
        self.case_lines = []
        self.combinations = []
        self.total_cases = 0
        self.sim_per_node = 0
        self.nodes_needed = 0
        self.subscans = []  # List of (start_idx, end_idx) tuples for each job
        self.params = []
        self._setup()
        
        self._validate()

        
    def _validate(self):
        if self.gpu_per_instance < 1:
            raise ValueError("GPU_PER_INSTANCE must be at least 1.")
        if self.gpu_per_instance > self.total_gpus_per_node:
            raise ValueError("GPU_PER_INSTANCE cannot exceed TOTAL_GPUS_PER_NODE.")
        if not self.scan_arrays:
            raise ValueError("SCAN_ARRAYS cannot be empty.")
        if self.restart_from_last_frame and not os.path.isfile(self.last_frame_detector_script):
            raise FileNotFoundError(f"Last frame detector script not found: {self.last_frame_detector_script}")
        if not os.path.isfile(self.gkeyll_input_c):
            raise FileNotFoundError(f"Gkeyll input file not found: {self.gkeyll_input_c}")
        if isinstance(self.time, str):
            self.max_run_time = parse_slurm_time_to_seconds(self.time)
        else:
            raise ValueError("Time input must be a string in Slurm format HH:MM:SS.")
        if self.total_gpus_per_node % self.gpu_per_instance != 0:
            print(f"WARNING: GPUS_PER_INSTANCE ({self.gpu_per_instance}) does not divide evenly into node GPUs ({self.total_gpus_per_node}). Resources may be wasted.")
        if self.num_jobs < 1:
            raise ValueError("NUM_JOBS must be at least 1.")
        
    def _setup(self):
        # Prepare Scan Combinations
        keys = list(self.scan_arrays.keys())
        values = list(self.scan_arrays.values())
        self.combinations = list(itertools.product(*values))
        
        # Generate the cases array and parameter table content
        self.table_content = "# Simulation Parameter Table\n"
        self.table_content += "# Format: ID | " + " | ".join(keys) + "\n"
        self.table_content += "#" + "-" * 80 + "\n"
        
        for i, combo in enumerate(self.combinations):
            # Create the parameter string: "key1=val1,key2=val2,..."
            param_str = ",".join([f"{k}={v}" for k, v in zip(keys, combo)])
            # Store in CASES array
            self.case_lines.append(f'CASES[{i}]="{i}|{param_str}"\n')
            # Add to table
            table_line = f"{i:05d} | " + " | ".join([str(v) for v in combo])
            self.table_content += table_line + "\n"
            # Store params values
            self.params.append(combo)
            
        # Calculate required nodes based on total number of cases
        self.total_cases = len(self.combinations)
        self.sim_per_node = int(self.total_gpus_per_node / self.gpu_per_instance)
        self.nodes_needed = (self.total_cases + self.sim_per_node - 1) // self.sim_per_node
        
        # Split cases into subscans
        self._create_subscans()
        
    def _create_subscans(self):
        """Divide total cases into num_jobs subscans"""
        cases_per_job = (self.total_cases + self.num_jobs - 1) // self.num_jobs  # Ceiling division
        
        for job_idx in range(self.num_jobs):
            start_idx = job_idx * cases_per_job
            end_idx = min((job_idx + 1) * cases_per_job, self.total_cases)
            
            if start_idx < self.total_cases:
                self.subscans.append((start_idx, end_idx))
        
    def generate_script(self):
        """Generate submission scripts for all subscans"""
        print(f"Generating scripts for {self.total_cases} total cases split into {self.num_jobs} job(s).")
        print(f"Each node will run {self.sim_per_node} simulations concurrently.")
        
        if self.num_jobs > 1:
            self._generate_subscan_scripts()
            self._generate_master_submit_script()
        else:
            self._generate_single_script()
    
    def _generate_single_script(self):
        """Generate a single submission script (num_jobs=1)"""
        script_content = self._build_script_content(0, self.total_cases)
        
        with open(self.output_script, "w") as f:
            f.write(script_content)
        
        st = os.stat(self.output_script)
        os.chmod(self.output_script, st.st_mode | stat.S_IEXEC)
        
        print(f"Successfully generated {self.output_script}")
        print(f"To submit: sh {self.output_script}")
    
    def _generate_subscan_scripts(self):
        """Generate individual submission scripts for each subscan"""
        for job_idx, (start_idx, end_idx) in enumerate(self.subscans):
            script_name = self.output_script.replace('.sh', f'_job_{job_idx:03d}.sh')
            script_content = self._build_script_content(start_idx, end_idx, job_idx)
            
            with open(script_name, "w") as f:
                f.write(script_content)
            
            st = os.stat(script_name)
            os.chmod(script_name, st.st_mode | stat.S_IEXEC)
            
            cases_in_job = end_idx - start_idx
            nodes_needed_job = (cases_in_job + self.sim_per_node - 1) // self.sim_per_node
            print(f"  Generated {script_name}: cases {start_idx}-{end_idx-1} ({cases_in_job} cases, {nodes_needed_job} node(s))")
    
    def _generate_master_submit_script(self):
        """Generate a master script that submits all subscans sequentially"""
        master_script = self.output_script.replace('.sh', '_master.sh')
        
        master_content = "#!/bin/bash\n"
        master_content += "# Master submission script - submits all subscans\n\n"
        
        for job_idx in range(self.num_jobs):
            script_name = self.output_script.replace('.sh', f'_job_{job_idx:03d}.sh')
            master_content += f"echo 'Submitting job {job_idx + 1}/{self.num_jobs}...'\n"
            master_content += f"sbatch {script_name}\n"
            master_content += "sleep 1\n\n"
        
        with open(master_script, "w") as f:
            f.write(master_content)
        
        st = os.stat(master_script)
        os.chmod(master_script, st.st_mode | stat.S_IEXEC)
        
        print(f"Successfully generated {master_script}")
        print(f"To submit all jobs: bash {master_script}")
    
    def _build_script_content(self, start_idx, end_idx, job_idx=0):
        """Build SLURM script content for a specific case range"""
        cases_in_job = end_idx - start_idx
        nodes_needed_job = (cases_in_job + self.sim_per_node - 1) // self.sim_per_node
        
        job_name_suffix = f"_job{job_idx:03d}" if self.num_jobs > 1 else ""
        current_script_name = self.output_script.replace('.sh', f'_job_{job_idx:03d}.sh') if self.num_jobs > 1 else self.output_script
        
        script_content = f"""#!/bin/bash

# =============================================================================
# AUTO-GENERATED SLURM SUBMISSION SCRIPT
# Generated by: slurm_auto_scripting.py
# Cases: {start_idx} to {end_idx-1} (Job {job_idx + 1}/{self.num_jobs})
# =============================================================================

# --- Parse Command Line Arguments ---
DEPENDENCY_JOB=""
SHOW_INFO=0

usage() {{
    cat << EOF
Usage: $0 [OPTIONS]

Submit SLURM job for parameter scan (Job {job_idx + 1}/{self.num_jobs})

Options:
    -j <job_id>     Create dependency on specified job ID
                    Job will start after the specified job completes
    --info          Display job information without submitting
    -h, --help      Display this help message

Job Configuration:
    Job Name:        {self.job_name}{job_name_suffix}
    Cases:           {start_idx} to {end_idx-1} ({cases_in_job} cases)
    Nodes:           {nodes_needed_job}
    Sims per Node:   {self.sim_per_node}
    GPUs per Sim:    {self.gpu_per_instance}
    Time Limit:      {self.time}
    QOS:             {self.qos}
    Account:         {self.account}

Examples:
    $0                  # Submit job normally
    $0 -j 12345         # Submit with dependency on job 12345
    $0 --info           # Show configuration without submitting

EOF
    exit 0
}}

while [[ $# -gt 0 ]]; do
    case $1 in
        -j)
            DEPENDENCY_JOB="$2"
            shift 2
            ;;
        --info)
            SHOW_INFO=1
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

# --- Display Info if Requested ---
if [ $SHOW_INFO -eq 1 ]; then
    echo "========================================"
    echo "Job Information (Job {job_idx + 1}/{self.num_jobs})"
    echo "========================================"
    echo "Job Name:            {self.job_name}{job_name_suffix}"
    echo "Cases:               {start_idx} to {end_idx-1} ({cases_in_job} total)"
    echo "Nodes Required:      {nodes_needed_job}"
    echo "Simulations/Node:    {self.sim_per_node}"
    echo "GPUs per Simulation: {self.gpu_per_instance}"
    echo "CPUs per Task:       {self.cpus_per_task}"
    echo "Time Limit:          {self.time}"
    echo "QOS:                 {self.qos}"
    echo "Account:             {self.account}"
    echo "Constraint:          {self.constraint}"
    echo "Work Directory:      {self.work_dir}"
    echo "Gkeyll Executable:   {self.gkeyll_exe}"
    echo "Restart Enabled:     {self.restart_from_last_frame}"
    echo "Max Sim Time:        {self.max_sim_time} µs"
    echo "Execution Mode:      {self.execution_mode}"
    echo "========================================"
    echo ""
    echo "Scan Parameters:"
"""
        
        # Add scan parameter info
        for key, values in self.scan_arrays.items():
            script_content += f'    echo "  {key}: {values}"\n'
        
        script_content += f"""
    echo ""
    echo "To submit this job:"
    echo "  sbatch {current_script_name}"
    echo ""
    echo "To submit with dependency:"
    echo "  {current_script_name} -j <job_id>"
    exit 0
fi

# --- Build SLURM Command ---
SBATCH_CMD="sbatch"

if [ -n "$DEPENDENCY_JOB" ]; then
    echo "Setting up dependency on job $DEPENDENCY_JOB"
    SBATCH_CMD="$SBATCH_CMD --dependency=afterany:$DEPENDENCY_JOB"
fi

# --- Create Temporary SLURM Script ---
TEMP_SCRIPT=$(mktemp /tmp/slurm_submit_XXXXXX.sh)

cat > "$TEMP_SCRIPT" << 'SLURM_SCRIPT_EOF'
#!/bin/bash
#SBATCH --job-name={self.job_name}{job_name_suffix}
#SBATCH --qos={self.qos}
#SBATCH --account={self.account}
#SBATCH --constraint={self.constraint}
#SBATCH --nodes={nodes_needed_job}
#SBATCH --ntasks-per-node={self.total_gpus_per_node}
#SBATCH --gpus-per-task={self.gpus_per_task}
#SBATCH --cpus-per-task={self.cpus_per_task}
#SBATCH --time={self.time}
#SBATCH --output={self.work_dir}/slurm_out/%x-%j.out
#SBATCH --error={self.work_dir}/slurm_out/%x-%j.err
"""

        if self.email:
            script_content += f"#SBATCH --mail-user={self.email}\n"
            script_content += f"#SBATCH --mail-type={self.email_type}\n"

        script_content += f"""

set -euo pipefail

# --- Environment Setup ---
module load PrgEnv-gnu/8.5.0 craype-accel-nvidia80 cray-mpich/8.1.28 cudatoolkit/12.4 nccl/2.18.3-cu12

export MPICH_GPU_SUPPORT_ENABLED=0
export DVS_MAXNODES=32_
export MPICH_MPIIO_DVS_MAXNODES=32
export NCCL_DEBUG=WARN
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export MPICH_OFI_NIC_POLICY=GPU

# --- Simulation Parameters ---
MAX_RUN_TIME={self.max_run_time}
EXECUTION_MODE="{self.execution_mode}"

# --- Job Control Configuration ---
# {self.total_gpus_per_node} GPUs per node / {self.gpu_per_instance} GPUs per sim
JOBS_PER_NODE={self.sim_per_node}
MAX_CONCURRENT_JOBS=$((SLURM_NNODES * JOBS_PER_NODE))

echo "========================================"
echo "Job Configuration (Subscan {job_idx + 1}/{self.num_jobs}):"
echo "  Cases:               {start_idx} to {end_idx-1}"
echo "  Nodes:               $SLURM_NNODES"
echo "  Simulations/Node:    $JOBS_PER_NODE"
echo "  Max Concurrent Jobs: $MAX_CONCURRENT_JOBS"
echo "========================================"

# --- Construct Case List ---
declare -a CASES=()
"""
        
        for i in range(start_idx, end_idx):
            script_content += self.case_lines[i]
        
        script_content += f"""
TOTAL_CASES=${{#CASES[@]}}
echo "Total Cases to Run in this Job: $TOTAL_CASES"

# --- Create Work Directory and Parameter Table ---
mkdir -p {self.work_dir}
mkdir -p {self.work_dir}/slurm_out

# Copy files with force overwrite to handle concurrent jobs
cp -f {current_script_name} {self.work_dir}/ 2>/dev/null || true
cp -f {self.gkeyll_input_c} {self.work_dir}/ 2>/dev/null || true

# Write parameter table (unique per job, safe to write)
cat > {self.work_dir}/parameter_table.txt << 'EOF'
{self.table_content}EOF

echo "Parameter table written to {self.work_dir}/parameter_table.txt"
# --- Execution Function ---
run_case() {{
    local case_info=$1
    
    # Split ID and Params
    local id=${{case_info%%|*}}
    local params=${{case_info#*|}}
    
    local sim_name="{self.job_name}_$(printf "%05d" $id)"
    local workdir="{self.work_dir}/${{sim_name}}"
    
    if [ "{self.restart_from_last_frame}" = "True" ]; then
        last_frame=$(sh {self.last_frame_detector_script} "$workdir")
        if [ -n "$last_frame" ]; then
            echo "Restarting from last detected frame: $last_frame"
            restart_opt="-r $last_frame"
        fi
    else
        echo "Starting fresh simulation (no restart)."
        restart_opt=""
    fi
    
    # Prepare directory
    mkdir -p "$workdir"
    cp -u {self.gkeyll_exe} "$workdir/."
    
    echo "[$(date '+%H:%M:%S')] Starting Case $id: $sim_name"
    echo "  Params: $params"
    
    cd "$workdir"
    
    # Write context JSON file
    ./{self.gkeyll_exe} --write-ctx -o "$params,max_run_time=$MAX_RUN_TIME,sim_name=$sim_name" 
    
    # Execute Simulation
    srun --nodes=1 \\
         --ntasks={self.gpu_per_instance} \\
         --gpus-per-task=1 \\
         --cpus-per-task={self.cpus_per_task} \\
         --cpu-bind=cores \\
         --exclusive \\
         --gpu-bind=none \\
         ./{self.gkeyll_exe} -g -M -e {self.gpu_per_instance} {self.gkyl_additional_options} $restart_opt \\
         -o "$params,max_run_time=$MAX_RUN_TIME,sim_name=$sim_name" \\
         2>> "../err-${{sim_name}}.log" | tee "../std-${{sim_name}}.log"
         
    local status=$?
    
    if [ $status -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] Finished Case $id: $sim_name (SUCCESS)"
    else
        echo "[$(date '+%H:%M:%S')] Finished Case $id: $sim_name (FAILED - see ${{workdir}}/${{sim_name}}-slurm.log)"
    fi

    # Handle Execution Mode Logic
    if [ "$EXECUTION_MODE" == "FIRST" ]; then
        echo "EXECUTION_MODE=FIRST: Case $id finished. Stopping all other jobs..."
        kill -TERM $$
    elif [ "$EXECUTION_MODE" == "FAIL" ] && [ $status -ne 0 ]; then
        echo "EXECUTION_MODE=FAIL: Case $id failed. Stopping all other jobs..."
        kill -TERM $$
    fi
}}

# Trap to ensure background jobs are killed if script is aborted
cleanup() {{
    echo "Script aborted. Killing background jobs..."
    pkill -P $$ || true
    exit 1
}}
trap cleanup SIGINT SIGTERM

# --- Main Loop ---
for case_info in "${{CASES[@]}}"; do
    # Manage Concurrency
    while [ $(jobs -r | wc -l) -ge $MAX_CONCURRENT_JOBS ]; do
        sleep 2
    done
    
    # Launch Job
    run_case "$case_info" &
    
    # Stagger starts
    sleep 2
done

wait
echo "All cases in this job completed."
SLURM_SCRIPT_EOF

# --- Submit Job ---
echo "Submitting job with: $SBATCH_CMD $TEMP_SCRIPT"
JOB_ID=$($SBATCH_CMD "$TEMP_SCRIPT" | grep -oP '\\d+')

if [ -n "$JOB_ID" ]; then
    echo "Job submitted successfully with ID: $JOB_ID"
else
    echo "Failed to submit job"
    rm -f "$TEMP_SCRIPT"
    exit 1
fi

rm -f "$TEMP_SCRIPT"
exit 0
    """
        
        return script_content
    
    def _display_progress(self, max_time):
        """Display current progress of all simulations"""
        # Clear screen (works on Unix-like systems)
        # print("\033[2J\033[H")
        
        print("=" * 80)
        print(f"Simulation Progress Monitor - {self.job_name}")
        print(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        print()
        
        # Collect progress data
        progress_data = []
        for i in range(self.total_cases):
            sim_name = f"{self.job_name}_{i:05d}"
            log_file = os.path.join(self.work_dir, f"std-{sim_name}.log")
            
            if os.path.exists(log_file):
                try:
                    sim_time, dt = self._extract_simulation_time(log_file)
                except:
                    sim_time = 0.0
                    dt = 0.0
                status = "dt={:.6f} mus".format(dt)
            else:
                sim_time = 0.0
                dt = 0.0
                status = "PENDING"
            
            # Check if completed
            if sim_time >= max_time * 0.99:  # Within 1% of max time
                status = "COMPLETE"
            
            progress_data.append({
                'id': i,
                'name': sim_name,
                'time': sim_time,
                'dt': dt,
                'status': status,
                'progress': min(sim_time / max_time * 100, 100) if max_time > 0 else 0,
                'params': self.params[i] if i < len(self.params) else "N/A"
            })
        
        # Display summary statistics
        completed = sum(1 for p in progress_data if p['status'] == 'COMPLETE')
        running = sum(1 for p in progress_data if p['status'] == 'RUNNING')
        pending = sum(1 for p in progress_data if p['status'] == 'PENDING')
        avg_progress = sum(p['progress'] for p in progress_data) / len(progress_data)
        
        print(f"Total Cases: {self.total_cases} | "
              f"Complete: {completed} | Running: {running} | Pending: {pending}")
        print(f"Overall Progress: {avg_progress:.1f}%")
        print()
        
        # Header
        print(f"{'ID':^5} | {'Progress':^32} | {'t curr (µs)':^9} | {'Params':^16} | {'last dt (µs)':<8}")
        
        # Display individual progress bars
        bar_width = 25
        for data in progress_data:
            filled = int(data['progress'] / 100 * bar_width)
            bar = '█' * filled + '░' * (bar_width - filled)
            
            # Color coding based on status
            # status_color = {
            #     'COMPLETE': '\033[92m',  # Green
            #     'RUNNING': '\033[93m',   # Yellow
            #     'PENDING': '\033[90m'    # Gray
            # }
            # reset_color = '\033[0m'
            # color = status_color.get(data['status'], '')
            reset_color = ''
            color = ''
            
            params_line = ''
            for key, val in zip(self.scan_arrays.keys(), data['params']):
                if np.abs(val) > 10:
                    params_line += f"{val:.1e} "
                elif val > 0:
                    params_line += f" {val:1.1f} "
                else:
                    params_line += f"{val:1.1f} "
            params_line = params_line.strip()
            
            if data['dt'] == 0:
                dtinfo = "PENDING"
            elif data['dt'] < 1e-6:
                dtinfo = "CRASHED"
            else:
                dtinfo = f"{data['dt']:1.1e}"
            
            print(f"{data['id']:05d} | {color}{bar}{reset_color} "
                  f"{data['progress']:5.1f}% | "
                  f"t={data['time']:5.0f}/{max_time:.0f} | "
                  f"{params_line} | {dtinfo}")
        print("=" * 80)
    
    def _extract_simulation_time(self, log_file):
        """
        Extract the latest simulation time from a log file.
        Looks for lines like: "Taking time-step at t = 70.7722 mus ... dt = 0.00196159 mus"
        
        Returns:
            float: Latest simulation time in microseconds, or 0.0 if not found
        """
        try:
            # Read file from end to find the last time-step line
            with open(log_file, 'rb') as f:
                # Get file size
                f.seek(0, 2)
                file_size = f.tell()
                
                # Start from end, read in chunks backwards
                block_size = 8192
                blocks_read = []
                
                # Read up to 10 blocks (80KB) from the end
                for i in range(10):
                    seek_pos = max(file_size - (i + 1) * block_size, 0)
                    f.seek(seek_pos)
                    block = f.read(min(block_size, file_size - seek_pos))
                    blocks_read.insert(0, block)
                    
                    if seek_pos == 0:
                        break
                
                # Combine blocks and decode
                content = b''.join(blocks_read).decode('utf-8', errors='ignore')
                lines = content.splitlines()
            
            # Search backwards for the time-step line
            for line in reversed(lines):
                if 'Taking time-step at t =' in line:
                    match = re.search(r'Taking time-step at t = ([\d.]+) mus ... dt = ([\d.]+) mus', line)
                    if match:
                        return float(match.group(1)), float(match.group(2))
            
            return 0.0
        except Exception as e:
            # If file doesn't exist or can't be read, return 0
            return 0.0
    
    def print_summary(self, max_time=None):
        """Print a quick summary of simulation progress without continuous monitoring"""
        max_time = max_time if max_time is not None else self.max_sim_time
        self._display_progress(max_time)
    
    def get_progress_data(self):
        """
        Get progress data for all simulations as a list of dictionaries.
        Useful for scripting or integration with other tools.
        
        Returns:
            list: List of dictionaries with keys: id, name, time, status, progress
        """
        max_time = self.max_sim_time
        progress_data = []
        
        for i in range(self.total_cases):
            sim_name = f"{self.job_name}_{i:05d}"
            log_file = os.path.join(self.work_dir, f"std-{sim_name}.log")
            
            if os.path.exists(log_file):
                sim_time = self._extract_simulation_time(log_file)
                status = "RUNNING"
            else:
                sim_time = 0.0
                status = "PENDING"
            
            if sim_time >= max_time * 0.99:
                status = "COMPLETE"
            
            progress_data.append({
                'id': i,
                'name': sim_name,
                'time': sim_time,
                'status': status,
                'progress': min(sim_time / max_time * 100, 100) if max_time > 0 else 0
            })
        
        return progress_data

