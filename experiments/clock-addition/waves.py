"""Are there sine waves in the trained weights? Compares a trained model with the same model before training.
usage: P=47 python3 waves.py runs/X.pt
1. embedding: share of each number-embedding's variance carried by the top 5 frequencies (of P//2)
2. output: share of the answer scores explained by a function of (a + b - c) alone, and that function's top frequencies
3. MLP neurons: share of each neuron's (a, b) activation map in its strongest frequency k (terms k on a, b, a+b, a-b)
"""
import sys, torch
import mod
P = mod.P


def emb_share(E):
    E = E[:P] - E[:P].mean(0); pw = (torch.fft.rfft(E, dim=0).abs() ** 2).sum(1)[1:]; pw = pw / pw.sum()
    return pw.sort(descending=True).values[:5].sum().item(), (pw.argsort(descending=True)[:5] + 1).tolist()


@torch.no_grad()
def check(m, name):
    x, y, _, _ = mod.data(0); L = m(x).view(P, P, P)                      # L[a, b, c] = score of answer c
    hid = m.blk.hid[:, -1].view(P, P, -1)                                 # MLP neurons at the '=' position
    a, b, c = torch.meshgrid(torch.arange(P), torch.arange(P), torch.arange(P), indexing="ij")
    Lc = L - L.mean(2, keepdim=True)                                      # only differences between answers matter
    g = torch.zeros(P).index_add_(0, ((a + b - c) % P).flatten(), Lc.flatten()) / (P * P)   # best function of (a+b-c)
    out_share = 1 - ((Lc - g[(a + b - c) % P]) ** 2).sum().item() / (Lc ** 2).sum().item()
    gp = torch.fft.rfft(g).abs()[1:] ** 2; gtop = (gp.argsort(descending=True)[:5] + 1).tolist(); gs = (gp.sort(descending=True).values[:5].sum() / gp.sum()).item()
    H2 = torch.fft.fft2(hid - hid.mean((0, 1)), dim=(0, 1)).abs() ** 2   # 2D frequencies (ka, kb) per neuron
    tot = H2.sum((0, 1)).clamp_min(1e-12); best = torch.zeros(hid.shape[-1])
    for k in range(1, P // 2 + 1):                                        # power on a, b, a+b, a-b at frequency k (both signs)
        idx = [(k, 0), (P - k, 0), (0, k), (0, P - k), (k, k), (P - k, P - k), (k, P - k), (P - k, k)]
        best = torch.maximum(best, sum(H2[i, j] for i, j in idx) / tot)
    es, etop = emb_share(m.emb.weight)
    print(f"{name:10} embedding top-5 share {es:.2f} {etop} | output explained by (a+b-c) {out_share:.2f}, its top-5 share {gs:.2f} {gtop} "
          f"| neurons: median share in one frequency {best.median():.2f}, >0.8: {(best > 0.8).float().mean():.2f}")


torch.manual_seed(0); check(mod.Model(), "untrained")
m = mod.Model(); m.load_state_dict(torch.load(sys.argv[1])); check(m, "trained")
