import os
import subprocess
import argparse

# Configuration settings
mpi_job_script = "./mpi/mpi.grace_job"
mpi_single_job_script = "./mpi/mpi_single.grace_job"
cuda_job_script = "./cuda/cuda.grace_job"
cali_dir = "./cali"
input_types = ["s", "r", "rs", "p"]
input_sizes = [16, 18, 20, 22, 24, 26, 28]
mpi_procs = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
cuda_threads = [64, 128, 256, 512, 1024]
procs_per_node = 32


def is_missing_or_empty(filename):
    return not os.path.exists(filename) or os.path.getsize(filename) == 0


def get_filename(job_type, size, parameter, input_type):
    return os.path.join(
        cali_dir,
        "sample-{}-{}{}-i{}.cali".format(job_type, parameter, size, input_type),
    )


def submit_job(job_type, size, parameter, input_type):
    if job_type == "mpi":
        nodes = (
            1
            if parameter <= procs_per_node
            else (parameter + procs_per_node - 1) // procs_per_node
        )
        memory = (size.bit_length() - 1) // 2
        script = (
            mpi_single_job_script if parameter <= procs_per_node else mpi_job_script
        )
        command = [
            "sbatch",
            "--nodes={}".format(nodes),
            "--mem={}G".format(memory),
            script,
            str(size),
            str(parameter),
            input_type,
            cali_dir,
        ]
    else:
        memory = (size.bit_length() - 1) // 2
        command = [
            "sbatch",
            "--mem={}G".format(memory),
            cuda_job_script,
            str(size),
            str(parameter),
            input_type,
            cali_dir,
        ]
    subprocess.call(command)


def process_jobs(operation, job_type):
    subprocess.call(["mkdir", "-p", cali_dir])
    missing_files = []

    job_types_to_process = ["mpi", "cuda"] if job_type == "both" else [job_type]

    for input_type in input_types:
        for size_power in input_sizes:
            size = 2**size_power
            if "mpi" in job_types_to_process:
                for procs in mpi_procs:
                    filename = get_filename("mpi", size, procs, input_type)
                    if operation == "run" or (
                        operation == "resubmit" and is_missing_or_empty(filename)
                    ):
                        submit_job("mpi", size, procs, input_type)
                    elif operation == "missing" and is_missing_or_empty(filename):
                        missing_files.append(filename)

            if "cuda" in job_types_to_process:
                for threads in cuda_threads:
                    filename = get_filename("cuda", size, threads, input_type)
                    if operation == "run" or (
                        operation == "resubmit" and is_missing_or_empty(filename)
                    ):
                        submit_job("cuda", size, threads, input_type)
                    elif operation == "missing" and is_missing_or_empty(filename):
                        missing_files.append(filename)

    if operation == "missing":
        print("Missing or Empty Files:")
        for file in missing_files:
            print(file)
        print("Total Count:", len(missing_files))


def main():
    parser = argparse.ArgumentParser(description="Manage MPI and CUDA job submissions.")
    parser.add_argument(
        "operation",
        choices=["missing", "resubmit", "run"],
        help="The operation to perform",
    )
    parser.add_argument(
        "job_type",
        choices=["mpi", "cuda", "both"],
        default="both",
        nargs="?",
        help="Type of job to process (mpi, cuda, or both)",
    )
    args = parser.parse_args()
    process_jobs(args.operation, args.job_type)


if __name__ == "__main__":
    main()
