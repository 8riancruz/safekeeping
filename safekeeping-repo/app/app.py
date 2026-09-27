"""Streamlit web page: paste a Bitcoin address, get a risk report.  Run: streamlit run app.py"""
import os, math, html
import pandas as pd
import streamlit as st
import analyzer
from features import label

APP_NAME = 'SafeKeeping'                                   # <-- product name shown on the page
TAGLINE = 'Bitcoin wallet risk intelligence'

st.set_page_config(page_title=f'{APP_NAME} · Wallet Risk', layout='wide')

COLORS = {'low': '#10b981', 'medium': '#f59e0b', 'high': '#ef4444', 'sanctioned': '#dc2626'}
TIER_TEXT = {'low': 'LOW RISK', 'medium': 'MEDIUM RISK', 'high': 'HIGH RISK', 'sanctioned': 'SANCTIONED'}

st.markdown("""
<style>
  .block-container {padding-top: 2rem; max-width: 1150px;}
  .brand {font-size: 2.1rem; font-weight: 800; letter-spacing: -0.02em; margin-bottom: 0;}
  .brand span {color: #2dd4bf;}
  .tagline {color: #94a3b8; margin-top: -4px; margin-bottom: 1.2rem;}
  .card {background: #131c31; border: 1px solid #1e2a44; border-radius: 14px; padding: 18px 20px; margin-bottom: 12px;}
  .reason {background: #131c31; border: 1px solid #1e2a44; border-left: 4px solid; border-radius: 10px;
           padding: 10px 14px; margin-bottom: 8px;}
  .reason .t {font-weight: 600;} .reason .s {color: #94a3b8; font-size: 0.88rem;}
  .pill {display: inline-block; padding: 3px 12px; border-radius: 999px; font-weight: 700;
         font-size: 0.8rem; letter-spacing: 0.05em; color: #0b1120;}
  .step {background: #131c31; border: 1px solid #1e2a44; border-radius: 12px; padding: 14px; height: 100%;}
  .step .n {color: #2dd4bf; font-weight: 800; font-size: 0.8rem; letter-spacing: 0.08em;}
  .step .h {font-weight: 700; margin: 4px 0 6px;} .step .d {color: #94a3b8; font-size: 0.88rem;}
  .muted {color: #94a3b8; font-size: 0.88rem;}
  .addr {font-family: ui-monospace, monospace; word-break: break-all; color: #cbd5e1;}
</style>""", unsafe_allow_html=True)


# ---------------------------------------------------------------- cached resources
@st.cache_resource
def model():
    return analyzer.load_bundle()


@st.cache_resource
def client():
    return analyzer.BlockchainClient()


@st.cache_data
def demo_addresses():
    return pd.read_csv('demo_addresses.csv') if os.path.exists('demo_addresses.csv') else None


# ---------------------------------------------------------------- small UI helpers
def gauge(score, thr, tier):
    """Semicircle SVG gauge; zones follow the model's own threshold."""
    cx, cy, r = 150, 140, 105
    def pt(v, rad=r):
        a = math.pi * (1 - v / 100)
        return cx + rad * math.cos(a), cy - rad * math.sin(a)
    def arc(v1, v2, color):
        (x1, y1), (x2, y2) = pt(v1), pt(v2)
        return (f'<path d="M {x1:.1f} {y1:.1f} A {r} {r} 0 0 1 {x2:.1f} {y2:.1f}" '
                f'stroke="{color}" stroke-width="20" fill="none"/>')
    t = thr * 100
    nx, ny = pt(min(max(score, 0.5), 99.5), r - 28)
    color = COLORS[tier]
    return f"""
    <svg viewBox="0 0 300 205" width="100%" style="max-width:340px;display:block;margin:auto">
      {arc(0, t / 2, COLORS['low'])}{arc(t / 2, t, COLORS['medium'])}{arc(t, 100, COLORS['high'])}
      <line x1="{cx}" y1="{cy}" x2="{nx:.1f}" y2="{ny:.1f}" stroke="#e2e8f0" stroke-width="4" stroke-linecap="round"/>
      <circle cx="{cx}" cy="{cy}" r="8" fill="#e2e8f0"/>
      <text x="{cx}" y="{cy + 50}" text-anchor="middle" font-size="36" font-weight="800" fill="{color}">{score:.0f}<tspan font-size="14" font-weight="400" fill="#94a3b8"> / 100</tspan></text>
      <text x="40" y="{cy + 18}" text-anchor="middle" font-size="11" fill="#64748b">0</text>
      <text x="260" y="{cy + 18}" text-anchor="middle" font-size="11" fill="#64748b">100</text>
    </svg>"""


def pill(tier):
    return f'<span class="pill" style="background:{COLORS[tier]}">{TIER_TEXT[tier]}</span>'


def reason_card(r, risky):
    color = COLORS['high'] if risky else COLORS['low']
    return (f'<div class="reason" style="border-left-color:{color}"><div class="t">{html.escape(r["label"][:1].upper() + r["label"][1:])}: '
            f'{r["value"]:,.4g}</div><div class="s">typical illicit wallet {r["illicit_median"]:,.4g} · '
            f'typical licit wallet {r["licit_median"]:,.4g}</div></div>')


def pick(address):
    st.session_state['addr'] = address
    st.session_state['run'] = True


# ---------------------------------------------------------------- page
bundle = model()
st.session_state.setdefault('recent', [])

st.markdown('<div class="brand">Safe<span>Keeping</span></div>', unsafe_allow_html=True)
st.markdown(f'<div class="tagline">{TAGLINE} · machine-learning risk scoring on live blockchain data</div>',
            unsafe_allow_html=True)

with st.sidebar:
    st.markdown('### Model')
    m = bundle['metrics'][bundle['model_name']]
    st.markdown(f"**{bundle['model_name']}** · Elliptic++ wallets")
    c1, c2, c3 = st.columns(3)
    c1.metric('F1', f"{m['Illicit F1']:.2f}"); c2.metric('Prec.', f"{m['Illicit Precision']:.2f}")
    c3.metric('Recall', f"{m['Illicit Recall']:.2f}")
    demos = demo_addresses()
    if demos is not None:
        st.markdown('### Try an example')
        for _, r in demos.groupby('kind').head(3).iterrows():
            st.button(f"{'Illicit' if 'illicit' in r['kind'] else 'Licit'} · {r['address'][:16]}…", key='demo_' + r['address'],
                      on_click=pick, args=(r['address'],), width='stretch')
    st.markdown('### Recent searches')
    recent_box = st.container()          # filled at the end of the page so it includes the latest search

tab_check, tab_how = st.tabs(['Check a wallet', 'How it works'])

# ---------------------------------------------------------------- tab 1: check a wallet
with tab_check:
    col_in, col_btn = st.columns([5, 1], vertical_alignment='bottom')
    address = col_in.text_input('Bitcoin address', key='addr', placeholder='Paste an address: 1…, 3… or bc1…')
    go = col_btn.button('Analyze', type='primary', width='stretch') or st.session_state.pop('run', False)

    if go and address:
        if not analyzer.is_valid_address(address):
            st.error('That does not look like a valid Bitcoin address.')
            st.stop()
        bar = st.progress(0.0, text='Fetching transaction history from blockchain.com (free tier ≈ 10 s per page)…')
        try:
            rep = analyzer.analyze(address, client=client(),
                                   progress=lambda frac, msg: bar.progress(min(frac, 1.0), text=msg))
        except Exception as e:
            bar.empty(); st.error(f'Could not fetch data for this address: {e}'); st.stop()
        bar.empty()

        scored = rep['status'] == 'scored'
        tier = rep.get('risk_tier') if (scored or rep['sanctioned']) else None
        entry = (rep['address'], rep.get('risk_score'), tier)
        st.session_state['recent'] = [entry] + [x for x in st.session_state['recent'] if x[0] != entry[0]]

        st.markdown(f'<div class="addr">{html.escape(rep["address"])}</div>', unsafe_allow_html=True)
        if rep['sanctioned']:
            st.error('This address is on the U.S. Treasury OFAC sanctions list. Transacting with it may be prohibited.')
        elif not rep['sanctions_list_loaded']:
            st.info('Sanctions list could not be loaded right now, so no sanctions check was done.')

        left, right = st.columns([2, 3], gap='large')
        with left:
            if scored:
                st.markdown(gauge(rep['risk_score'], bundle['threshold'], rep['risk_tier']), unsafe_allow_html=True)
                st.markdown(f'<div style="text-align:center">{pill(rep["risk_tier"])}</div>', unsafe_allow_html=True)
            elif rep['sanctioned']:
                st.markdown(f'<div class="card" style="text-align:center">{pill("sanctioned")}</div>',
                            unsafe_allow_html=True)
            else:
                st.warning(rep['message'])
            if scored:
                a, b = st.columns(2)
                a.metric('Txs analyzed', rep['n_txs_analyzed'])
                b.metric('Balance (BTC)', f"{rep['final_balance_btc']:.4f}")
                if rep['message']:
                    st.caption(rep['message'])
        with right:
            source = 'Amazon Bedrock' if rep['summary_source'] == 'bedrock' else 'rule-based'
            st.markdown(f'<div class="card"><div class="muted">Summary · {source}</div>'
                        f'<div style="margin-top:6px;line-height:1.55">{html.escape(rep["summary"])}</div></div>',
                        unsafe_allow_html=True)
            if rep['risk_reasons'] or rep['ok_reasons']:
                st.markdown('**Why this score**')
                st.markdown(''.join(reason_card(r, True) for r in rep['risk_reasons']) +
                            ''.join(reason_card(r, False) for r in rep['ok_reasons']), unsafe_allow_html=True)

        if scored and rep['out_of_range'] > 0.3:
            st.warning('This wallet behaves unlike most wallets the model was trained on, so treat the score with caution.')

        if scored and len(rep['windows']) > 1:
            st.markdown('**Risk over time** · two-week windows (~2016 blocks each); the headline score is the riskiest window')
            w = pd.DataFrame(rep['windows'])
            w['risk score'] = w['risk'] * 100
            st.area_chart(w.set_index('block_start')['risk score'], color='#2dd4bf', height=220)

        if scored:
            with st.expander('All features for the riskiest window'):
                f = pd.Series(rep['worst_features'])[bundle['features']]
                st.dataframe(pd.DataFrame({'feature': [label(k) for k in f.index], 'value': f.values}),
                             hide_index=True, width='stretch')
        if rep.get('from_cache'):
            st.caption('Used cached blockchain data (refreshed every 24 h).')
    elif not address:
        st.markdown('<div class="card muted">Paste a Bitcoin address above, or pick an example from the sidebar. '
                    'A first-time lookup takes 10–40 seconds on the free blockchain.com API; repeat lookups are instant.'
                    '</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------- tab 2: how it works
with tab_how:
    steps = [
        ('01', 'Sanctions screen', 'Checks the address against the U.S. Treasury OFAC list of sanctioned Bitcoin addresses.'),
        ('02', 'Live history', 'Pulls recent transactions from blockchain.com, cached in Amazon S3 for instant repeat lookups.'),
        ('03', 'Wallet features', '50 behavioral features per two-week window: amounts, fees, timing between transactions, counterparties.'),
        ('04', 'ML risk score', f"{bundle['model_name']} trained on 265k labeled Elliptic++ wallets in Amazon SageMaker."),
        ('05', 'Explanation', 'Compares the wallet to typical illicit and licit wallets; Amazon Bedrock writes a plain-English summary.'),
    ]
    cols = st.columns(len(steps))
    for c, (n, h, d) in zip(cols, steps):
        c.markdown(f'<div class="step"><div class="n">STEP {n}</div><div class="h">{h}</div><div class="d">{d}</div></div>',
                   unsafe_allow_html=True)
    st.markdown('')
    a, b = st.columns(2, gap='large')
    with a:
        st.markdown('#### Validated against the research data')
        st.markdown('Features rebuilt from live blockchain data were compared with the Elliptic++ dataset for the same '
                    'wallets and time windows. After fixing three definition differences, they match **100%**, so live '
                    'scores come from the same inputs the model was trained on.')
        st.markdown('#### Built on AWS')
        st.markdown('**S3** data, model and cache · **SageMaker** training · **EC2** this web app · **Bedrock** summaries')
    with b:
        st.markdown('#### Limits')
        st.markdown('- Trained on wallets from 2016–2017. Criminal behavior shifts over time, so accuracy on today\'s '
                    'wallets will be lower than the test score.\n'
                    '- Only the most recent transactions are analyzed (free API rate limit).\n'
                    '- A risk score is an indicator for review, **not proof of wrongdoing**.')

# ---------------------------------------------------------------- recent searches (drawn last)
with recent_box:
    if not st.session_state['recent']:
        st.caption('Wallets you check will appear here.')
    for i, (a, score, tier) in enumerate(st.session_state['recent'][:8]):
        tag = f'{score:.0f}' if score is not None else '–'
        word = {'low': 'Low', 'medium': 'Med', 'high': 'High', 'sanctioned': 'OFAC'}.get(tier, 'n/a')
        st.button(f'{tag} {word} · {a[:16]}…', key=f'recent_{i}_{a}', on_click=pick, args=(a,),
                  width='stretch')
