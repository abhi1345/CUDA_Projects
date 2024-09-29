import subprocess
import re

REPEAT = 3

GPU_arch = 'H100'
lower_bound_meps = None
lower_bound_occupancy = None
lower_bound_throughput = None
ideal_meps = None

if GPU_arch == 'L40S':
    lower_bound_meps = 400
    lower_bound_occupancy = 70
    lower_bound_throughput = 80

    ideal_meps = 650
elif GPU_arch == 'H100':
    lower_bound_meps = 350
    lower_bound_occupancy = 70
    lower_bound_throughput = 55

    ideal_meps = 520

def compile():
    compile_command = "nvcc -Xcompiler -rdynamic -lineinfo -x cu kernel.cu"

    compile_process = subprocess.Popen(compile_command, shell=True, stderr=subprocess.PIPE, universal_newlines=True)
    compile_process.communicate()  # Wait for the compilation to finish
    compile_return_code = compile_process.returncode

    if compile_return_code != 0:
        error_message = compile_process.stderr.strip()
        print(f"Compilation failed: {error_message}")
        return None

    return "Compilation successful"

def run(size):
    run_command = f"./a.out {size}"

    run_process = subprocess.Popen(run_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    run_output, run_error = run_process.communicate()
    if "FUNCTIONAL SUCCESS" in run_output:
        return run_output
    else:
        print(f"Functional test failed for size = {size}")

    return None

def run_ncu_metric(metric, size):
    ncu_command = f"ncu --metric {metric} --print-summary per-gpu ./a.out {size}"

    ncu_process = subprocess.Popen(ncu_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    ncu_output, _ = ncu_process.communicate()

    # Extract the last numbers from lines containing the specified metric
    last_numbers = [float(match.group(3)) for match in re.finditer(rf"{metric}.*?(\d+\.\d+).*?(\d+\.\d+).*?(\d+\.\d+)", ncu_output)]

    return last_numbers

def calculate_average_metric(metric_values):
    if metric_values:
        average = sum(metric_values) / len(metric_values)
        average = round(average,2)
        return average
    return None

def main():
    score = 0
    memory_throughput_score = 0
    occupancy_througput_score = 0
    mep_metric_score = 0
    correctness_score = 0

    # Task 2: Run and check for 'FUNCTIONAL SUCCESS'
    for size in [2000, 10000, 100000, 1000000, 10000000]:
        output = run(size)
        if output:
            correctness_score += 1

    # Task 3: Calculate average for sm__warps_active.avg.pct_of_peak_sustained_active
    metric_output = run_ncu_metric("sm__warps_active.avg.pct_of_peak_sustained_active", 10000000)
    average_metric = calculate_average_metric(metric_output)
    
    print(f"Achieved Occupancy: {average_metric}")
    if average_metric is not None and average_metric > lower_bound_occupancy:
        occupancy_througput_score = 1

    # Task 5: Run ./a.out 10000000 and check for speedup print
    best_meps = 0
    for _ in range(REPEAT):
        output = run(10000000)
        match = re.search(r"GPU Sort Time \(ms\) :.*\s(\d+\.\d+)", output)
        meps = 10e3 / float(match.group(1))
        if meps > best_meps:
            best_meps = meps
    if best_meps:
        print(f"Million elements per second: {best_meps}")
        if best_meps > lower_bound_meps:
            mep_metric_score = min(15, (best_meps/ideal_meps)*15)

    # # Task 4: Calculate average for gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed
    metric_output = run_ncu_metric("gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed", 10000000)
    average_metric = calculate_average_metric(metric_output)
    print(f"Memory Throughput: {average_metric}")

    if average_metric and average_metric > lower_bound_throughput:
        memory_throughput_score = 1

    if correctness_score == 5:
        score = round(correctness_score + mep_metric_score + memory_throughput_score + occupancy_througput_score)
    else:
        score = correctness_score

    # Display total score
    print(f"Total Score: {score} pts")

if __name__ == "__main__":
    if not compile():
        print("Exiting due to compilation error")
        exit(1)
    main()

