"""Check that a Muon implementation orthogonalizes each MoE expert alone.
Give every expert the same random gradient: the right layout returns, for every expert, Newton-Schulz of that one gradient.
Replace `muon_update` with a call into your optimizer's orthogonalization (same tensor layout as your expert weights)."""
import torch


def ns(G):
    a, b, c = 3.4445, -4.7750, 2.0315; X = G / (G.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    tr = X.shape[-2] > X.shape[-1]
    if tr: X = X.transpose(-2, -1)
    for _ in range(5): A = X @ X.transpose(-2, -1); X = a * X + (b * A + c * A @ A) @ X
    return X.transpose(-2, -1) if tr else X


LAYOUTS = {   # how frameworks have handled an [E, out, in] expert gradient
    "each expert alone": lambda G: ns(G),
    "fused [E*out, in]": lambda G: ns(G.reshape(-1, G.shape[-1])).view(G.shape),
    "flattened [E, out*in]": lambda G: ns(G.reshape(G.shape[0], -1)).view(G.shape),
    "expert axis as reduction axis": lambda G: ns(G.permute(2, 0, 1)).permute(1, 2, 0),
}

if __name__ == "__main__":
    E, o, i = 8, 44, 32; g = torch.randn(o, i); ref = ns(g)
    for name, muon_update in LAYOUTS.items():
        U = muon_update(g.expand(E, o, i).clone())
        print(f"{name:32s} expert-0 update size / correct size: {U[0].norm() / ref.norm():.2f}   error: {(U[0] - ref).norm() / ref.norm():.2f}")
