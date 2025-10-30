#!/usr/bin/env python3
import csv, sys, re, math, argparse

def gen_regex(csv_path: str, topn: int) -> str:
    rows = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        rdr = csv.DictReader(f)
        if not rdr.fieldnames:
            return ""
        time_cols = [c for c in rdr.fieldnames if 'Total Time' in c or 'Time (ns)' in c or (c and c.lower().startswith('time'))]
        for r in rdr:
            name = r.get('Name') or r.get('Kernel Name') or r.get('Kernel')
            if not name:
                for k in r:
                    if 'Name' in k:
                        name = r[k]; break
            if not name:
                continue
            tval = None
            for tc in time_cols:
                v = r.get(tc)
                if v is None: continue
                v2 = v.replace(',', '').strip()
                try:
                    tval = float(v2)
                    break
                except Exception:
                    pass
            if tval is None:
                for k, v in r.items():
                    if k == 'Name': continue
                    if v is None: continue
                    v2 = str(v).replace(',', '').strip()
                    try:
                        tval = float(v2); break
                    except Exception:
                        pass
            if tval is None or math.isnan(tval):
                tval = 0.0
            rows.append((tval, name))
    rows.sort(key=lambda x: x[0], reverse=True)
    top = [r[1] for r in rows[:topn]]
    parts = [re.escape(x) for x in top if x]
    return "|".join(parts) if parts else ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv', help='nsys cuda_gpu_kern_sum.csv path')
    ap.add_argument('topn', type=int, help='Top-N kernels to include')
    ap.add_argument('out', help='Output regex file')
    args = ap.parse_args()
    regex = gen_regex(args.csv, args.topn)
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write(regex)
    print(regex)

if __name__ == '__main__':
    main()

