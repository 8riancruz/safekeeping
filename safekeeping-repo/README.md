# SafeKeeping: Bitcoin wallet risk intelligence

Paste a Bitcoin address and get an anti-money-laundering risk score, the reasons behind it, and a plain-English summary. The score comes from a model trained on labeled wallets and computed from live blockchain data.

## What's in this repo

| Folder | What it is |
|---|---|
| `part1_research/` | Transaction-level study on the Elliptic dataset: GraphSAGE (PyTorch Geometric) vs XGBoost, Random Forest and a hybrid, with a strict temporal split, plus a per-week concept-drift analysis |
| `part2_wallet_model/` | Wallet-level model trained on Elliptic++ (265k labeled real wallets) and a parity check proving live features match the dataset |
| `app/` | Streamlit web app: OFAC sanctions screen → blockchain.com history → 50 wallet features per two-week window → risk score → reasons → Amazon Bedrock summary |

## Architecture (AWS)

```
User ─► EC2 (Streamlit app)
          ├─ OFAC sanctioned-address list
          ├─ blockchain.com API ──► cached in S3
          ├─ wallet model (trained in SageMaker, stored in S3)
          └─ Amazon Bedrock (Claude) plain-English summary
```

## Key results

- **Part 1:** Random Forest illicit F1 **0.81** (precision 0.95, recall 0.70) on the future test weeks (35–49), consistent with the original Elliptic paper. Every model fails after the dark-market shutdown at time step 43, which is concept drift.
- **Part 2:** features rebuilt from live blockchain.com data match the Elliptic++ values **100%** for the validated wallets. The first check matched 68%; the mismatches revealed three definition differences, which we fixed.

## Run it

1. Upload the Elliptic (Part 1) or Elliptic++ `wallets_features.csv` / `wallets_classes.csv` (Part 2) files to an S3 bucket.
2. Run `part2_wallet_model/wallet_model_ready.ipynb` in SageMaker. It trains the model and uploads the model and app files to S3.
3. On an Amazon Linux 2023 EC2 instance with an IAM role for S3 and Bedrock, and port 8501 open:
   ```bash
   aws s3 cp s3://YOUR-BUCKET/elliptic/app/setup_ec2.sh . && bash setup_ec2.sh YOUR-BUCKET
   nano ~/app/config.env      # add your blockchain.com key (optional)
   bash ~/app/run_app.sh
   ```

The blockchain.com API key is added on the server only and is not stored in this repo. The bucket and server were temporary AWS workshop resources used for the hackathon.

## Limitations

- Training data covers 2016–2017, and criminal behavior drifts over time.
- Only the most recent transactions are analyzed, because of the free API rate limit.
- A risk score is an indicator for review, not proof of wrongdoing.

## Data

- Elliptic Data Set (Weber et al., 2019)
- Elliptic++ (Elmougy & Liu, 2023): https://github.com/git-disl/EllipticPlusPlus
- OFAC sanctioned addresses list: https://github.com/0xB10C/ofac-sanctioned-digital-currency-addresses
