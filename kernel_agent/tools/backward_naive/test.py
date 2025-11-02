# test_aot_capture.py
import torch
from aot_capture import capture_reference_backward

def attn_ref(Q, K, V):
    # Q,K,V: [B,H,N,D] -> O: [B,H,N,D]
    B, H, N, D = Q.shape
    q = Q.reshape(B * H, N, D)
    k = K.reshape(B * H, N, D)
    v = V.reshape(B * H, N, D)
    S = torch.bmm(q, k.transpose(1, 2)) / (D ** 0.5)
    P = torch.softmax(S, dim=-1)
    O = torch.bmm(P, v)
    return O.reshape(B, H, N, D)

def main():
    torch.manual_seed(0)
    B, H, N, D = 2, 2, 8, 4
    Q = torch.randn(B, H, N, D, requires_grad=True)
    K = torch.randn(B, H, N, D, requires_grad=True)
    V = torch.randn(B, H, N, D, requires_grad=True)

    stats: Dict[str, Any] = {}
    ref_fn, capture = capture_reference_backward(
        attn_ref, [Q, K, V], {}, stats=stats, verbose=True
    )

    # Optional: see what was captured
    print("Captured keys:", list(capture.keys()))
    print("Stats keys:", list(stats.keys()))

if __name__ == "__main__":
    main()
