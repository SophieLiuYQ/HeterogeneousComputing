#!/usr/bin/env bash
sudo -u ec2-user /bin/bash << EOF
echo "executing as..."
whoami
source "/home/ec2-user/.bashrc"
python3 "/home/ec2-user/HeterogeneousComputing/CPUPullPrice.py"
EOF
