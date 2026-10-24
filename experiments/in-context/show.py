import json, sys
keys = sys.argv[1].split(",") if len(sys.argv) > 1 else ["T", "BURST", "D", "L", "steps", "lr", "seed"]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
for r in [json.loads(l) for l in open("runs.jsonl")][-n:]:
    print(" ".join(f"{k}={r.get(k)}" for k in keys), "| recog", r["recog"], "copy", r["copy"], "rep", r["rep"], "loss", r["loss"], "conf ctx/mem", r.get("conf_ctx"), r.get("conf_mem"), f"{r['sec']}s")
