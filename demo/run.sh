#!/bin/bash
set -e
mkdir -p /home/admin/logs/atec2026/robot/solution
cd /home/admin/appspace/atec2026/robot
source ./venv/bin/activate
python solution/server.py 2>&1 | tee /home/admin/logs/atec2026/robot/solution/server.stdout
