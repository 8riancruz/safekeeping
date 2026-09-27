#!/bin/bash
# One-time setup on a fresh Amazon Linux 2023 EC2 instance.
# Usage: bash setup_ec2.sh your-bucket-name
set -e
BUCKET="$1"
if [ -z "$BUCKET" ]; then echo "Usage: bash setup_ec2.sh your-bucket-name"; exit 1; fi

echo "== Installing Python 3.11"
sudo dnf install -y python3.11 python3.11-pip

echo "== Downloading app code and model from S3"
mkdir -p ~/app && cd ~/app
aws s3 cp "s3://$BUCKET/elliptic/app/" . --recursive
aws s3 cp "s3://$BUCKET/elliptic/wallet_model/" . --recursive

echo "== Installing Python packages (pinned to the versions used for training)"
python3.11 -m venv venv
./venv/bin/pip install --upgrade pip -q
./venv/bin/pip install -r requirements_app.txt -q

sed -i "s/^BUCKET=.*/BUCKET=$BUCKET/" config.env
echo "== Done. Check config.env (nano config.env), then start the app with: bash run_app.sh"
