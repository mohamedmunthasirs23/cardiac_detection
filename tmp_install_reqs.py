import subprocess
import sys

with open('requirements.txt', 'r') as f:
    reqs = [line.strip() for line in f if line.strip() and not line.startswith('#')]

for req in reqs:
    print(f"Installing {req}...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", req], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {req}: {e}")
        # Stop on the first major failure to see what killed it
        if e.returncode == -1073741510:
             print(f"CRITICAL ACCESS VIOLATION ON {req}")
             break
