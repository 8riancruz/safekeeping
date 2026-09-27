"""
Wallet features, computed the same way as the Elliptic++ "actors" dataset.

Each Elliptic++ row describes one wallet address during one time step (~2 weeks,
about 2016 Bitcoin blocks). We rebuild the same features from live blockchain.com
data so the model sees live wallets exactly the way it saw training wallets.
"""
import numpy as np

SATOSHI = 1e8
STATS = ['total', 'min', 'max', 'mean', 'median']

# per-transaction series whose total/min/max/mean/median become features
GROUPS = ['btc_transacted', 'btc_sent', 'btc_received', 'fees', 'fees_as_share',
          'blocks_btwn_txs', 'blocks_btwn_input_txs', 'blocks_btwn_output_txs',
          'transacted_w_address']

SINGLES = ['num_txs_as_sender', 'num_txs_as_receiver', 'total_txs',
           'lifetime_in_blocks', 'num_addr_transacted_multiple']

ALL_FEATURES = SINGLES + [f'{g}_{s}' for g in GROUPS for s in STATS]

# plain-English names for reports
LABELS = {
    'num_txs_as_sender': 'transactions sending funds',
    'num_txs_as_receiver': 'transactions receiving funds',
    'total_txs': 'total transactions',
    'lifetime_in_blocks': 'active lifetime (blocks)',
    'num_addr_transacted_multiple': 'counterparties dealt with more than once',
    'btc_transacted': 'BTC moved per transaction',
    'btc_sent': 'BTC sent per transaction',
    'btc_received': 'BTC received per transaction',
    'fees': 'fees paid per transaction (BTC)',
    'fees_as_share': 'wallet share of fees (BTC)',
    'blocks_btwn_txs': 'blocks between transactions',
    'blocks_btwn_input_txs': 'blocks between sends',
    'blocks_btwn_output_txs': 'blocks between receipts',
    'transacted_w_address': 'interactions per counterparty',
}


def label(feature):
    """'blocks_btwn_txs_median' -> 'median blocks between transactions'."""
    for g in GROUPS:
        if feature.startswith(g + '_'):
            stat = feature[len(g) + 1:]
            return f"{stat} {LABELS[g]}" if stat != 'total' else f"total {LABELS[g].replace(' per transaction', '')}"
    return LABELS.get(feature, feature)


def _stats(values, prefix):
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return {f'{prefix}_{s}': 0.0 for s in STATS}
    return {f'{prefix}_total': float(v.sum()), f'{prefix}_min': float(v.min()),
            f'{prefix}_max': float(v.max()), f'{prefix}_mean': float(v.mean()),
            f'{prefix}_median': float(np.median(v))}


def _gaps(heights):
    h = sorted(heights)
    return np.diff(h) if len(h) > 1 else []


def tx_view(tx, address):
    """What one transaction looks like from the wallet's point of view."""
    inputs, outs = tx.get('inputs', []), tx.get('out', [])
    in_addrs = {i.get('prev_out', {}).get('addr') for i in inputs}
    out_addrs = {o.get('addr') for o in outs}
    sent = sum(i.get('prev_out', {}).get('value', 0) for i in inputs
               if i.get('prev_out', {}).get('addr') == address) / SATOSHI
    received = sum(o.get('value', 0) for o in outs if o.get('addr') == address) / SATOSHI
    is_input, is_output = address in in_addrs, address in out_addrs
    # counterparties follow the money: when sending -> the recipients; when receiving -> the senders
    # (co-senders / co-recipients in the same transaction are NOT counterparties)
    others = (out_addrs if is_input else set()) | (in_addrs if is_output else set())
    others = {a for a in others if a and a != address}
    fee = tx.get('fee', 0) / SATOSHI
    tx_in_total = sum(i.get('prev_out', {}).get('value', 0) for i in inputs) / SATOSHI
    tx_total = tx_in_total or (sum(o.get('value', 0) for o in outs) / SATOSHI + fee)
    return {'height': tx['block_height'], 'sent': sent, 'received': received,
            'is_input': is_input, 'is_output': is_output, 'fee': fee, 'tx_total': tx_total,
            'others': others}


def compute_features(txs, address):
    """Elliptic++-style features for one wallet over a list of confirmed transactions."""
    views = [tx_view(t, address) for t in txs if t.get('block_height') is not None]
    if not views:
        return {f: 0.0 for f in ALL_FEATURES}

    heights = [v['height'] for v in views]
    moved = [v['sent'] + v['received'] for v in views]
    counterparty_counts = {}
    for v in views:
        for a in v['others']:
            counterparty_counts[a] = counterparty_counts.get(a, 0) + 1

    f = {
        'num_txs_as_sender': float(sum(v['is_input'] for v in views)),
        'num_txs_as_receiver': float(sum(v['is_output'] for v in views)),
        'total_txs': float(len(views)),
        'lifetime_in_blocks': float(max(heights) - min(heights)),
        'num_addr_transacted_multiple': float(sum(c > 1 for c in counterparty_counts.values())),
    }
    f.update(_stats(moved, 'btc_transacted'))
    # Elliptic++ averages sent/received over ALL the wallet's transactions (0 where it didn't send/receive)
    f.update(_stats([v['sent'] for v in views], 'btc_sent'))
    f.update(_stats([v['received'] for v in views], 'btc_received'))
    f.update(_stats([v['fee'] for v in views], 'fees'))
    # the wallet's share of each fee: fee x (wallet's amount / whole transaction value), in BTC
    f.update(_stats([v['fee'] * m / v['tx_total'] for v, m in zip(views, moved) if v['tx_total'] > 0],
                    'fees_as_share'))
    f.update(_stats(_gaps(heights), 'blocks_btwn_txs'))
    f.update(_stats(_gaps([v['height'] for v in views if v['is_input']]), 'blocks_btwn_input_txs'))
    f.update(_stats(_gaps([v['height'] for v in views if v['is_output']]), 'blocks_btwn_output_txs'))
    # how many times the wallet dealt with each counterparty (one value per counterparty)
    f.update(_stats(list(counterparty_counts.values()), 'transacted_w_address'))
    return f


def windowed_features(txs, address, window_blocks=2016, cumulative=False):
    """
    Split a wallet's history into ~2-week windows (like Elliptic time steps) and compute
    features per window. cumulative=True uses all history up to each window's end,
    matching the training data if Elliptic++ rows turn out to be cumulative.
    """
    confirmed = sorted([t for t in txs if t.get('block_height') is not None],
                       key=lambda t: t['block_height'])
    if not confirmed:
        return []
    start = confirmed[0]['block_height']
    buckets = {}
    for t in confirmed:
        buckets.setdefault((t['block_height'] - start) // window_blocks, []).append(t)
    out = []
    for k in sorted(buckets):
        w_start = start + k * window_blocks
        w_end = w_start + window_blocks - 1
        window_txs = [t for t in confirmed if t['block_height'] <= w_end] if cumulative else buckets[k]
        out.append({'block_start': w_start, 'block_end': w_end,
                    'n_txs': len(buckets[k]), 'features': compute_features(window_txs, address)})
    return out
