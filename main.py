import os
import sys

# Wrap launch.py so users can run `python main.py`
if __name__ == "__main__":
    launch_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "launch.py")
    with open(launch_path, "r") as f:
        exec(f.read())
