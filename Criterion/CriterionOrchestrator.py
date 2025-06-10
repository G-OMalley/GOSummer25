import subprocess
import sys
import os
import time

def run_script(script_name):
    """
    Runs a Python script using the same Python interpreter that is running this script.
    Checks if the script exists before trying to run it.
    """
    # Get the directory of the current orchestrator script
    orchestrator_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(orchestrator_dir, script_name)

    if not os.path.exists(script_path):
        print(f"--- SKIPPING: Script not found at {script_path} ---")
        return

    print(f"\n{'='*20} Running: {script_name} {'='*20}")
    
    try:
        # sys.executable ensures we use the same python environment
        process = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            check=True,  # This will raise an exception for non-zero exit codes
            encoding='utf-8'
        )
        print(f"--- Output from {script_name}: ---")
        print(process.stdout)
        if process.stderr:
            print(f"--- Errors from {script_name} (if any): ---")
            print(process.stderr)
        print(f"{'='*20} Finished: {script_name} {'='*20}")

    except subprocess.CalledProcessError as e:
        print(f"\n{'!'*20} ERROR running {script_name} {'!'*20}")
        print(f"Return Code: {e.returncode}")
        print("\n--- STDOUT: ---")
        print(e.stdout)
        print("\n--- STDERR: ---")
        print(e.stderr)
        print(f"{'!'*20} Script {script_name} failed. Halting orchestration. {'!'*20}")
        # Stop the rest of the orchestration if one script fails
        sys.exit(1) 
    except FileNotFoundError:
        print(f"ERROR: Could not find the script '{script_name}' at the expected path: {script_path}")
    except Exception as e:
        print(f"An unexpected error occurred while trying to run {script_name}: {e}")
        sys.exit(1)


def main():
    """
    Main function to orchestrate the execution of all Criterion data update scripts.
    """
    start_time = time.time()
    print(">>> Starting Criterion Data Orchestration <<<")

    # Define the order of execution for the scripts
    # This list now only contains the scripts you want to run.
    scripts_to_run = [
        "UpdateCriterionStorage.py",
        "UpdateCriterionLNG.py",
        "UpdateAndForecastFundy.py",
    ]

    for script in scripts_to_run:
        run_script(script)

    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n>>> All scripts completed successfully in {total_time:.2f} seconds. <<<")


if __name__ == '__main__':
    main()
