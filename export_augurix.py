"""Export the Augurix dashboard as a static site for the portfolio.

Run from the FinProj folder (where dashboard.py lives), with the same Python you use for the dashboard:

    python export_augurix.py              # exports the official run, gen-v1
    python export_augurix.py gen-v1 baselines

It reads the attempt log read-only (the dashboard can stay closed), copies dashboard.html, and saves
every API answer the page asks for as a JSON file. A small script added to the page sends its /api/...
requests to those files instead of a server. Output: augurix-static/ and augurix-static.zip.

Positions are taken only from the replay cache in runs/positions/: nothing is run in the sandbox here.
Discovery-period data only, exactly what the dashboard already shows; no validation or holdout bars.
"""
import json
import shutil
import sys
from pathlib import Path

import dashboard  # FinProj's own module: same queries the dashboard serves

OUT = Path("augurix-static")
RUNS = sys.argv[1:] or ["gen-v1"]

SHIM = """<script>
// Static export: answer the dashboard's /api requests from saved JSON files.
(() => {
  const real = window.fetch.bind(window);
  const file = (u) => {
    const url = new URL(u, location.href);
    if (!url.pathname.includes('/api/')) return null;
    const p = url.pathname.slice(url.pathname.indexOf('/api/') + 5);
    const run = url.searchParams.get('run') || '_all';
    if (p === 'state') return `data/state/${encodeURIComponent(run)}.json`;
    if (p === 'live') return `data/live/${encodeURIComponent(run)}.json`;
    if (p === 'bars') return 'data/bars.json';
    const m = /^(positions|attempt)\\/(\\d+)$/.exec(p);
    return m ? `data/${m[1]}/${m[2]}.json` : null;
  };
  window.fetch = (u, opts) => {
    const f = typeof u === 'string' ? file(u) : null;
    if (!f) return real(u, opts);
    return real(f).then((r) => r.ok ? r : new Response(JSON.stringify(
      f.includes('/positions/') ? { error: 'This attempt was not replayed for the online demo.' } : null),
      { headers: { 'Content-Type': 'application/json' } }));
  };
  if (!new URLSearchParams(location.search).has('run')) history.replaceState(null, '', '?run=' + %s);
})();
</script>
""" % json.dumps(RUNS[0])


def dump(rel, obj):
    path = OUT / "data" / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, separators=(",", ":")), encoding="utf-8")
    return path.stat().st_size


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    total = 0

    everything = dashboard.state(None)
    keep = [r for r in everything["runs"] if r["run_id"] in RUNS]
    missing = set(RUNS) - {r["run_id"] for r in keep}
    if missing:
        sys.exit(f"No such run(s): {', '.join(sorted(missing))}. Runs in the log: {[r['run_id'] for r in everything['runs']]}")

    ids = set()
    for run in RUNS:
        s = dashboard.state(run)
        s["runs"] = keep                      # the run picker offers only the exported runs
        total += dump(f"state/{run}.json", s)
        ids.update(a["id"] for a in s["attempts"])
        total += dump(f"live/{run}.json", dashboard.live(run))
    everything["runs"] = keep
    everything["attempts"] = [a for a in everything["attempts"] if a["id"] in ids]
    total += dump("state/_all.json", everything)

    total += dump("bars.json", dashboard.bars())
    for i in sorted(ids):
        total += dump(f"attempt/{i}.json", dashboard.attempt(i))
    cached = 0
    for f in sorted(dashboard.POSITIONS.glob("*.json")):
        if f.stem.isdigit() and int(f.stem) in ids:
            total += dump(f"positions/{f.name}", json.loads(f.read_text()))
            cached += 1

    html = (dashboard.ROOT / "dashboard.html").read_text(encoding="utf-8")
    html = html.replace("<head>", "<head>\n" + SHIM, 1)
    (OUT / "index.html").write_text(html, encoding="utf-8")

    shutil.make_archive("augurix-static", "zip", OUT)
    print(f"Exported runs {RUNS}: {len(ids)} attempts, {cached} cached replays, {total / 1e6:.1f} MB of data.")
    print(f"Upload {Path('augurix-static.zip').resolve()} to the chat.")


if __name__ == "__main__":
    main()
