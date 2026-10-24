"""run configs 2 at a time (waits while 8+ heavy python jobs run on the Mac).
usage: python3 par.py "T=4 BURST=8 :: 0 300 0.02" "T=16 BURST=8 :: 0 300 0.02" ...   (env :: icl.py args)"""
import os, sys, subprocess, time
def heavy():
    out = subprocess.run(["ps", "-Ao", "pcpu,comm"], capture_output=True, text=True).stdout.splitlines()[1:]
    return sum(1 for l in out if "python" in l.split(None, 1)[-1].lower() and float(l.split()[0]) > 50)
jobs, run = list(sys.argv[1:]), []
script = os.environ.get("SCRIPT", "icl.py")
while jobs or run:
    run = [p for p in run if p.poll() is None]
    if jobs and len(run) < int(os.environ.get("NPAR", 2)) and heavy() < 8:
        env_s, args = jobs.pop(0).split("::")
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", **dict(kv.split("=") for kv in env_s.split()))
        run.append(subprocess.Popen(["nice", "-n", "20", "python3", script] + args.split(), env=env, stdout=subprocess.DEVNULL))
        time.sleep(0.5)
    else: time.sleep(0.3)
