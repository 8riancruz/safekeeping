#!/bin/bash
# Start (or restart) the web app in the background. Usage: bash run_app.sh
cd ~/app
set -a; source config.env; set +a
export MODEL_PATH=wallet_model.joblib
export CACHE_S3_URI="s3://$BUCKET/elliptic/cache/"

if [ -f app.pid ]; then kill "$(cat app.pid)" 2>/dev/null || true; sleep 1; fi   # stop old copy
nohup ./venv/bin/streamlit run app.py --server.port 8501 --server.address 0.0.0.0 \
      --server.headless true > app.log 2>&1 &
echo $! > app.pid
sleep 5
TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 60")
IP=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/public-ipv4)
echo "App running. Open: http://$IP:8501"
echo "Logs: tail -f ~/app/app.log"
