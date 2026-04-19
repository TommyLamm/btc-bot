import polars as pl
import numpy as np
import random
import sys
sys.path.insert(0, "/root/btc-bot")

BASE_COLS = ["roc_5","roc_20","ma10_dev","ma30_dev","vwap_dev","price_impact","macd_hist"]

def roll_mean(x, w=10):
    x = np.array(x, dtype=np.float64)
    r = np.full_like(x, np.nan)
    cs = np.nancumsum(np.where(np.isnan(x), 0, x))
    r[w-1:] = (cs[w-1:] - np.concatenate([[0], cs[:-w]])) / w
    return r

def roll_std(x, w=20):
    x = np.array(x, dtype=np.float64)
    r = np.full_like(x, np.nan)
    for i in range(w-1, len(x)):
        r[i] = np.std(x[i-w+1:i+1])
    return r

def zscore(x, w=20):
    return (x - roll_mean(x,w)) / (roll_std(x,w) + 1e-10)

def delay(x, d=1):
    r = np.full(len(x), np.nan)
    r[d:] = x[:-d]
    return r

def diff_n(x, d=1):
    r = np.full(len(x), np.nan)
    r[d:] = x[d:] - x[:-d]
    return r

UNARY_OPS = {
    "log":    lambda x: np.log(np.abs(x)+1e-10),
    "sqrt":   lambda x: np.sqrt(np.abs(x)),
    "sq":     lambda x: np.sign(x)*x**2,
    "zs10":   lambda x: zscore(x,10),
    "zs20":   lambda x: zscore(x,20),
    "d1":     lambda x: delay(x,1),
    "d3":     lambda x: delay(x,3),
    "df1":    lambda x: diff_n(x,1),
    "df3":    lambda x: diff_n(x,3),
    "rm10":   lambda x: roll_mean(x,10),
    "rs20":   lambda x: roll_std(x,20),
    "abs":    lambda x: np.abs(x),
    "sign":   lambda x: np.sign(x),
}

BINARY_OPS = {
    "add": lambda a,b: a+b,
    "sub": lambda a,b: a-b,
    "mul": lambda a,b: a*b,
    "div": lambda a,b: a/(np.abs(b)+1e-10),
    "max": lambda a,b: np.maximum(a,b),
    "min": lambda a,b: np.minimum(a,b),
}

def safe(func, *args):
    try:
        with np.errstate(all="ignore"):
            r = np.array(func(*args), dtype=np.float64)
            r[np.isinf(r)] = np.nan
            if np.isnan(r).mean() > 0.3: return None
            p1,p99 = np.nanpercentile(r,[1,99])
            return np.clip(r,p1,p99)
    except: return None

def ic_full(f, r):
    v = ~(np.isnan(f)|np.isnan(r)|np.isinf(f))
    return float(np.corrcoef(f[v],r[v])[0,1]) if v.sum()>500 else 0.0

def ic_monthly(f, r, ts, min_m=3):
    ms = 30*24*60*60*1000
    t = ts[0]; ics=[]
    while t < ts[-1]:
        mask=(ts>=t)&(ts<t+ms)
        if mask.sum()>200:
            fv,rv=f[mask],r[mask]
            v=~(np.isnan(fv)|np.isnan(rv)|np.isinf(fv))
            if v.sum()>100:
                ic=np.corrcoef(fv[v],rv[v])[0,1]
                if not np.isnan(ic): ics.append(ic)
        t+=ms
    if len(ics)<min_m: return None
    return np.mean(ics), np.std(ics), len(ics)

def mine(df, ret, ts, n=1000, seed=42):
    random.seed(seed); np.random.seed(seed)
    base={c:df[c].to_numpy().astype(np.float64) for c in BASE_COLS if c in df.columns}
    avail=list(base.keys()); res=[]
    print(f"搜索 {n} 次，因子: {avail}\n数据量: {len(ret)}\n")

    for i in range(n):
        t=random.choice(["u","b","bu","tri"])
        if t=="u":
            f1=random.choice(avail); op=random.choice(list(UNARY_OPS))
            val=safe(UNARY_OPS[op],base[f1]); desc=f"{op}({f1})"
        elif t=="b":
            f1,f2=random.sample(avail,2); op=random.choice(list(BINARY_OPS))
            val=safe(BINARY_OPS[op],base[f1],base[f2]); desc=f"({f1} {op} {f2})"
        elif t=="bu":
            f1,f2=random.sample(avail,2)
            bop=random.choice(list(BINARY_OPS)); uop=random.choice(list(UNARY_OPS))
            tmp=safe(BINARY_OPS[bop],base[f1],base[f2])
            val=safe(UNARY_OPS[uop],tmp) if tmp is not None else None
            desc=f"{uop}({f1} {bop} {f2})"
        else:
            f1,f2,f3=random.sample(avail,3)
            b1=random.choice(list(BINARY_OPS)); b2=random.choice(list(BINARY_OPS))
            tmp=safe(BINARY_OPS[b1],base[f1],base[f2])
            val=safe(BINARY_OPS[b2],tmp,base[f3]) if tmp is not None else None
            desc=f"(({f1} {b1} {f2}) {b2} {f3})"

        if val is None: continue

        m=ic_monthly(val,ret,ts)
        if m: ic,std,nm=m; method=f"月度({nm}月)"
        else: ic=ic_full(val,ret); std=0.0; method="全样本"

        if abs(ic)>=0.012:
            g="⭐⭐⭐" if abs(ic)>0.025 else "⭐⭐" if abs(ic)>0.018 else "⭐"
            res.append((desc,ic,std,method,g,val.copy()))

        if (i+1)%250==0:
            print(f"  {i+1}/{n} 发现 {len(res)} 个候选")

    res.sort(key=lambda x:abs(x[1]),reverse=True)
    kept=[]; kv=[]
    for row in res:
        v=row[-1]
        dup=any((~(np.isnan(v)|np.isnan(p))).sum()>100 and
                abs(np.corrcoef(v[~(np.isnan(v)|np.isnan(p))],
                                p[~(np.isnan(v)|np.isnan(p))])[0,1])>0.80
                for p in kv)
        if not dup: kept.append(row[:-1]); kv.append(v)
    return kept

if __name__=="__main__":
    from factors.factor_engine import compute_factors
    print("载入数据...")
    df=pl.read_parquet("data/btc_5m.parquet")
    df=compute_factors(df).drop_nulls().fill_nan(0)
    df=df.with_columns([pl.col("close").pct_change().shift(-1).alias("future_ret")]).drop_nulls()
    print(f"数据: {len(df)} 行  {df['datetime'].min()} ~ {df['datetime'].max()}\n")
    ret=df["future_ret"].to_numpy(); ts=df["timestamp"].to_numpy()
    results=mine(df,ret,ts,n=1000)
    print(f"\n{'─'*70}")
    print(f"{'表达式':<36} {'IC':>8} {'std':>7} {'方法':>12} {'评级':>8}")
    print(f"{'─'*70}")
    for desc,ic,std,method,g in results[:30]:
        print(f"{desc:<36} {ic:>+8.4f} {std:>7.4f} {method:>12} {g:>8}")
    print(f"\n共发现 {len(results)} 个有效因子")
    if results:
        print("\n⭐⭐ 以上：")
        for r in results:
            if "⭐⭐" in r[4]: print(f"  {r[0]}  IC={r[1]:+.4f}")
