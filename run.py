#!/usr/bin/env python
import os
import sys
import subprocess

# Clone or pull the latest repo
repo_dir = '/kaggle/working/repo'
if os.path.exists(repo_dir):
    subprocess.run(['git', '-C', repo_dir, 'pull', 'origin', 'main'], check=True)
else:
    subprocess.run(['git', 'clone', 'https://github.com/Kabir08/Jumper.git', repo_dir], check=True)

# Add to path and run main.py with passed args
sys.path.insert(0, repo_dir)
exec(open(os.path.join(repo_dir, 'main.py')).read())