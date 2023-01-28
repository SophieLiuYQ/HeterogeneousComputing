#!/usr/bin/env bash
sudo -u ec2-user /bin/bash << EOF
echo "executing as..."
whoami
source "/home/ec2-user/.bashrc"
cd "/home/ec2-user/HeterogeneousComputing"
python3 "/home/ec2-user/HeterogeneousComputing/use_for_test_better_result.py" -a
python3 "/home/ec2-user/HeterogeneousComputing/use_for_test_better_result.py" -s
EOF
