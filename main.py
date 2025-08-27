import subprocess
import sys

def run_script(script_name):
    """Run a Python script and exit if it fails."""
    print(f"Running {script_name}...")
    result = subprocess.run([sys.executable, script_name])
    if result.returncode != 0:
        print(f"{script_name} failed with exit code {result.returncode}.")
        sys.exit(result.returncode)
    print(f"{script_name} completed successfully.\n")

if __name__ == "__main__":
    # 先执行癌症检测
    run_script("tumor_detection.py")
    
    # 再执行癌症亚型预测
    run_script("metastasis_prediction.py")
    
    print("All scripts executed successfully.")