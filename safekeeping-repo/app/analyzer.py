"""
Wallet risk analyzer: address -> sanctions check -> blockchain.com history ->
Elliptic++ features -> model score -> reasons -> plain-English summary.

Settings come from environment variables (never hard-code keys):
  MODEL_S3_URI            s3://bucket/elliptic/wallet_model/wallet_model.joblib
  MODEL_PATH              local path (default: wallet_model.joblib)
  BLOCKCHAIN_API_CODE     your blockchain.com key (optional)
  BLOCKCHAIN_MIN_INTERVAL seconds between API calls (default 10)
  MAX_TXS                 most recent transactions to fetch (default 200)
  CACHE_DIR               local cache folder (default: cache)
  CACHE_S3_URI            optional s3://bucket/prefix for a shared cache
  BEDROCK_MODEL_ID        e.g. us.anthropic.claude-haiku-4-5-20251001-v1:0 (blank = rule-based only)
  AWS_REGION              e.g. us-east-1
"""
import os, re, json, time, math
import numpy as np
import pandas as pd
import requests
import joblib
import boto3

from features import windowed_features, label

CFG = {
    'model_s3_uri': os.getenv('MODEL_S3_URI', ''),
    'model_path':   os.getenv('MODEL_PATH', 'wallet_model.joblib'),
    'api_code':     os.getenv('BLOCKCHAIN_API_CODE', ''),
    'min_interval': float(os.getenv('BLOCKCHAIN_MIN_INTERVAL', '10')),
    'max_txs':      int(os.getenv('MAX_TXS', '200')),
    'cache_dir':    os.getenv('CACHE_DIR', 'cache'),
    'cache_s3_uri': os.getenv('CACHE_S3_URI', ''),
    'cache_ttl_h':  float(os.getenv('CACHE_TTL_HOURS', '24')),
    'bedrock_model': os.getenv('BEDROCK_MODEL_ID', ''),
    'region':       os.getenv('AWS_REGION', 'us-east-1'),
    'ofac_url':     os.getenv('OFAC_URL', 'https://raw.githubusercontent.com/0xB10C/'
                          'ofac-sanctioned-digital-currency-addresses/lists/sanctioned_addresses_XBT.txt'),
}
API = 'https://blockchain.info/rawaddr/{address}'
PAGE = 50


# ---------------------------------------------------------------- helpers
def _split_s3(uri):
    b, _, k = uri[5:].partition('/')
    return b, k


def is_valid_address(address):
    """Basic format check for legacy (1...), P2SH (3...) and bech32 (bc1...) addresses."""
    a = address.strip()
    return bool(re.fullmatch(r'[13][a-km-zA-HJ-NP-Z1-9]{25,34}', a) or
                re.fullmatch(r'bc1[02-9ac-hj-np-z]{11,87}', a.lower()))


def normalize(address):
    a = address.strip()
    return a.lower() if a.lower().startswith('bc1') else a


# ---------------------------------------------------------------- model
_bundle = None
def load_bundle():
    global _bundle
    if _bundle is None:
        path = CFG['model_path']
        if not os.path.exists(path) and CFG['model_s3_uri']:
            b, k = _split_s3(CFG['model_s3_uri'])
            boto3.client('s3', region_name=CFG['region']).download_file(b, k, path)
        _bundle = joblib.load(path)
    return _bundle


# ---------------------------------------------------------------- sanctions list
_ofac = None
def ofac_list():
    """OFAC-sanctioned Bitcoin addresses (None if the list couldn't be downloaded)."""
    global _ofac
    if _ofac is None:
        try:
            r = requests.get(CFG['ofac_url'], timeout=15); r.raise_for_status()
            _ofac = {line.strip() for line in r.text.splitlines() if line.strip()}
        except Exception:
            return None
    return _ofac


# ---------------------------------------------------------------- blockchain.com client
class BlockchainClient:
    """Rate-limited, cached client for blockchain.com's free address endpoint."""
    def __init__(self):
        self._last = 0.0
        os.makedirs(CFG['cache_dir'], exist_ok=True)
        self.s3 = boto3.client('s3', region_name=CFG['region']) if CFG['cache_s3_uri'] else None

    def _wait(self):
        gap = CFG['min_interval'] - (time.time() - self._last)
        if gap > 0:
            time.sleep(gap)
        self._last = time.time()

    def _get(self, address, offset):
        params = {'limit': PAGE, 'offset': offset}
        if CFG['api_code']:
            params['api_code'] = CFG['api_code']
        for attempt in range(4):
            self._wait()
            r = requests.get(API.format(address=address), params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(15 * (attempt + 1))      # back off when rate limited
                continue
            r.raise_for_status()
        raise RuntimeError('blockchain.com kept rate-limiting; try again in a minute')

    # cache: local file first, then optional S3
    def _cache_key(self, address):
        return f"{address}_{CFG['max_txs']}.json"

    def _cache_read(self, address):
        path = os.path.join(CFG['cache_dir'], self._cache_key(address))
        if not os.path.exists(path) and self.s3:
            b, p = _split_s3(CFG['cache_s3_uri'])
            try:
                self.s3.download_file(b, f"{p.rstrip('/')}/{self._cache_key(address)}", path)
            except Exception:
                return None
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            if time.time() - data.get('_fetched_at', 0) < CFG['cache_ttl_h'] * 3600:
                return data
        return None

    def _cache_write(self, address, data):
        path = os.path.join(CFG['cache_dir'], self._cache_key(address))
        with open(path, 'w') as f:
            json.dump(data, f)
        if self.s3:
            b, p = _split_s3(CFG['cache_s3_uri'])
            try:
                self.s3.upload_file(path, b, f"{p.rstrip('/')}/{self._cache_key(address)}")
            except Exception:
                pass

    def fetch(self, address, progress=None):
        cached = self._cache_read(address)
        if cached:
            if progress: progress(1.0, 'Loaded from cache')
            cached['_from_cache'] = True
            return cached
        first = self._get(address, 0)
        n_tx = first.get('n_tx', 0)
        target = min(n_tx, CFG['max_txs'])
        txs = list(first.get('txs', []))
        pages = max(1, math.ceil(target / PAGE))
        if progress: progress(1 / pages, f'Fetched page 1 of {pages}')
        while len(txs) < target:
            page = self._get(address, len(txs)).get('txs', [])
            if not page:
                break
            txs.extend(page)
            if progress: progress(min(1.0, len(txs) / target), f'Fetched {len(txs)} of {target} transactions')
        data = {k: first.get(k) for k in ('address', 'n_tx', 'total_received', 'total_sent', 'final_balance')}
        data['txs'] = txs[:target]
        data['_fetched_at'] = time.time()
        self._cache_write(address, data)
        data['_from_cache'] = False
        return data


# ---------------------------------------------------------------- scoring + reasons
def tier(p, thr):
    return 'high' if p >= thr else 'medium' if p >= thr / 2 else 'low'


def reasons(bundle, feats, n_risk=3, n_ok=2):
    """Compare the wallet to typical illicit vs licit wallets on the model's most important features."""
    ref = bundle['ref_stats']
    risk, ok = [], []
    for f in bundle['importance_order'][:20]:
        r = ref[f]
        v = math.log1p(max(feats[f], 0))
        li, il = math.log1p(max(r['licit_median'], 0)), math.log1p(max(r['illicit_median'], 0))
        if abs(li - il) < 0.05:           # feature doesn't separate the classes
            continue
        item = {'feature': f, 'label': label(f), 'value': feats[f],
                'illicit_median': r['illicit_median'], 'licit_median': r['licit_median']}
        (risk if abs(v - il) < abs(v - li) else ok).append(item)
    return risk[:n_risk], ok[:n_ok]


def out_of_range_share(bundle, feats):
    ref = bundle['ref_stats']
    outside = [f for f in bundle['features'] if not (ref[f]['p01'] <= feats[f] <= ref[f]['p99'])]
    return len(outside) / len(bundle['features'])


def _fmt(x):
    return f'{x:,.0f}' if abs(x) >= 100 else f'{x:,.4g}'


def rule_summary(rep):
    if rep['sanctioned']:
        return ('This address appears on the U.S. Treasury OFAC sanctions list. Transacting with it may be '
                'prohibited regardless of the model score.')
    if rep['status'] != 'scored':
        return rep['message']
    lines = [f"Risk score {rep['risk_score']:.0f}/100 ({rep['risk_tier']} risk), based on "
             f"{rep['n_txs_analyzed']} recent transactions across {len(rep['windows'])} two-week windows."]
    if rep['risk_reasons']:
        lines.append('Patterns that resemble illicit wallets: ' + '; '.join(
            f"{r['label']} = {_fmt(r['value'])} (typical illicit {_fmt(r['illicit_median'])}, "
            f"typical licit {_fmt(r['licit_median'])})" for r in rep['risk_reasons']) + '.')
    if rep['ok_reasons']:
        lines.append('Patterns that resemble licit wallets: ' + '; '.join(r['label'] for r in rep['ok_reasons']) + '.')
    if rep['out_of_range'] > 0.3:
        lines.append('Caution: this wallet behaves unlike most training wallets, so the score is less reliable.')
    lines.append('This is a risk indicator, not proof of wrongdoing.')
    return ' '.join(lines)


SYSTEM_PROMPT = (
    'You are an anti-money-laundering analyst assistant. Write a short, plain-English summary (max 120 words) '
    'of a Bitcoin wallet risk report for a non-expert. Use only the facts given. Never state or imply the owner '
    'is a criminal; describe risk indicators and patterns. Mention the main reasons, any sanctions match, and any '
    'reliability caveats. End with one suggested next step for the user.')


def bedrock_summary(rep):
    facts = {k: rep[k] for k in ('address', 'status', 'sanctioned', 'risk_score', 'risk_tier', 'n_txs_analyzed',
                                 'risk_reasons', 'ok_reasons', 'out_of_range', 'message') if k in rep}
    client = boto3.client('bedrock-runtime', region_name=CFG['region'])
    resp = client.converse(
        modelId=CFG['bedrock_model'],
        system=[{'text': SYSTEM_PROMPT}],
        messages=[{'role': 'user', 'content': [{'text': json.dumps(facts, default=float)}]}],
        inferenceConfig={'maxTokens': 400, 'temperature': 0.2})
    return resp['output']['message']['content'][0]['text'].strip()


# ---------------------------------------------------------------- main entry point
def analyze(address, client=None, progress=None):
    address = normalize(address)
    rep = {'address': address, 'status': 'scored', 'message': '', 'sanctioned': False,
           'sanctions_list_loaded': True, 'windows': [], 'risk_reasons': [], 'ok_reasons': [],
           'out_of_range': 0.0, 'summary_source': 'rules'}

    if not is_valid_address(address):
        rep.update(status='invalid', message='That does not look like a valid Bitcoin address.')
        rep['summary'] = rule_summary(rep); return rep

    ofac = ofac_list()
    rep['sanctions_list_loaded'] = ofac is not None
    rep['sanctioned'] = bool(ofac and address in ofac)

    bundle = load_bundle()
    data = (client or BlockchainClient()).fetch(address, progress)
    txs = [t for t in data['txs'] if t.get('block_height') is not None]
    rep.update(n_tx_total=data.get('n_tx', 0), n_txs_analyzed=len(txs), from_cache=data.get('_from_cache', False),
               final_balance_btc=(data.get('final_balance') or 0) / 1e8)

    if len(txs) < 2:
        rep.update(status='insufficient', message=f'Only {len(txs)} confirmed transaction(s) found, which is not '
                   'enough history to score this wallet reliably.')
    else:
        wins = windowed_features(txs, address, bundle['window_blocks'], bundle['cumulative'])
        X = pd.DataFrame([w['features'] for w in wins])[bundle['features']]
        probs = bundle['model'].predict_proba(X)[:, 1]
        worst = int(np.argmax(probs))
        thr = bundle['threshold']
        for w, p in zip(wins, probs):
            w['risk'] = float(p)
        rep['windows'] = [{k: w[k] for k in ('block_start', 'block_end', 'n_txs', 'risk')} for w in wins]
        rep['risk_prob'] = float(probs[worst])
        rep['risk_score'] = round(float(probs[worst]) * 100, 1)
        rep['risk_tier'] = tier(float(probs[worst]), thr)
        rep['worst_window'] = rep['windows'][worst]
        rep['worst_features'] = wins[worst]['features']
        rep['risk_reasons'], rep['ok_reasons'] = reasons(bundle, wins[worst]['features'])
        rep['out_of_range'] = out_of_range_share(bundle, wins[worst]['features'])
        if data.get('n_tx', 0) > len(data['txs']):
            rep['message'] = (f"Analyzed the most recent {len(data['txs'])} of {data['n_tx']:,} transactions.")

    if rep['sanctioned']:
        rep['risk_tier'] = 'sanctioned'

    rep['summary'] = rule_summary(rep)
    if CFG['bedrock_model']:
        try:
            rep['summary'] = bedrock_summary(rep)
            rep['summary_source'] = 'bedrock'
        except Exception as e:                  # no access / wrong model id -> keep rule-based text
            rep['bedrock_error'] = str(e)[:200]
    return rep
