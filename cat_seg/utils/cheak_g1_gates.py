# inspect_g1_gates.py
import os
import re
import torch

CKPT_PATH = "/home/host/mounted2/phm/OVRS/outputs/train_iSAID_RALA+TTT+G1/model_final.pth"

def _unwrap_state_dict(ckpt_obj):
    """
    兼容 Detectron2 / 纯 torch.save(state_dict) / 其他包装形式。
    """
    if isinstance(ckpt_obj, dict):
        # detectron2 常见：{"model": state_dict, ...}
        if "model" in ckpt_obj and isinstance(ckpt_obj["model"], dict):
            return ckpt_obj["model"]
        # lightning 常见：{"state_dict": ...}
        if "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
            return ckpt_obj["state_dict"]
        # 也可能直接就是 state_dict
        if all(isinstance(k, str) for k in ckpt_obj.keys()):
            return ckpt_obj
    raise ValueError("无法从 checkpoint 中解析出 state_dict，请打印 ckpt.keys() 看看包装格式。")

def _strip_prefix(k: str):
    # DDP/DP 常见前缀
    for p in ("module.", "model.", "student.", "ema."):
        if k.startswith(p):
            k = k[len(p):]
    return k

def _collect_gate_keys(state):
    """
    只抓取 CLIP 内部的 gate_proj（避免误匹配到其他模块）。
    text:  clip_model.transformer.resblocks.<i>.attn.gate_proj.(weight|bias)
    vision: clip_model.visual.transformer.resblocks.<i>.attn.gate_proj.(weight|bias)
    """
    patterns = {
        "text": re.compile(r"(?:^|.*)clip_model\.transformer\.resblocks\.(\d+)\.attn\.gate_proj\.(weight|bias)$"),
        "vision": re.compile(r"(?:^|.*)clip_model\.visual\.transformer\.resblocks\.(\d+)\.attn\.gate_proj\.(weight|bias)$"),
    }

    out = {"text": {}, "vision": {}}
    for k in list(state.keys()):
        kk = _strip_prefix(k)
        for branch, pat in patterns.items():
            m = pat.match(kk)
            if m:
                layer = int(m.group(1))
                wb = m.group(2)  # weight or bias
                out[branch].setdefault(layer, {})[wb] = kk
    return out

def _tensor_stats(t):
    t = t.detach().float().cpu()
    return {
        "shape": tuple(t.shape),
        "mean": float(t.mean()),
        "std":  float(t.std()),
        "min":  float(t.min()),
        "max":  float(t.max()),
    }

def _print_headwise_bias_info(bias, topk=5):
    """
    bias: [H] 时，sigmoid(bias) 是“每个 head 的 gate 开合概率/系数”（0~1）。
    """
    b = bias.detach().float().cpu()
    g = torch.sigmoid(b)
    # 最关 / 最开
    lowv, lowi = torch.topk(g, k=min(topk, g.numel()), largest=False)
    hiv, hii   = torch.topk(g, k=min(topk, g.numel()), largest=True)

    def fmt(idxs, vals):
        return ", ".join([f"h{int(i)}:{float(v):.4f}" for i, v in zip(idxs, vals)])

    print(f"            sigmoid(bias) lowest : {fmt(lowi, lowv)}")
    print(f"            sigmoid(bias) highest: {fmt(hii, hiv)}")

def inspect_gates(ckpt_path: str):
    assert os.path.exists(ckpt_path), f"找不到 checkpoint：{ckpt_path}"

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = _unwrap_state_dict(ckpt)

    # 统一去掉常见前缀后的“可查 key 映射”
    # 这样既能用 strip 后的 key 做匹配，也能找到原始 key 的张量
    stripped_to_orig = {}
    for k in state.keys():
        stripped_to_orig[_strip_prefix(k)] = k

    gate_map = _collect_gate_keys(state)

    for branch in ("text", "vision"):
        layers = sorted(gate_map[branch].keys())
        print("\n" + "=" * 90)
        print(f"[{branch.upper()}] Found layers with gate_proj: {len(layers)}")
        if len(layers) == 0:
            print(f"  没找到 {branch} 的 gate_proj；请确认你确实在该分支注入了 gate_proj，并且保存到了 ckpt。")
            continue

        for li in layers:
            keys = gate_map[branch][li]
            wkey = keys.get("weight", None)
            bkey = keys.get("bias", None)

            print(f"\n  - layer {li}")
            if wkey is not None:
                w = state[stripped_to_orig[wkey]]
                ws = _tensor_stats(w)
                print(f"        gate_proj.weight: shape={ws['shape']} mean={ws['mean']:.6g} std={ws['std']:.6g} "
                      f"min={ws['min']:.6g} max={ws['max']:.6g}")
            else:
                print("        gate_proj.weight: NOT FOUND")

            if wkey is not None:
                w = state[stripped_to_orig[wkey]]
                ws = _tensor_stats(w)
                print(f"        gate_proj.weight: shape={ws['shape']} mean={ws['mean']:.6g} std={ws['std']:.6g} "
                      f"min={ws['min']:.6g} max={ws['max']:.6g}")
            else:
                print("        gate_proj.weight: NOT FOUND")

            if bkey is not None:
                b = state[stripped_to_orig[bkey]]
                bs = _tensor_stats(b)
                print(f"        gate_proj.bias  : shape={bs['shape']} mean={bs['mean']:.6g} std={bs['std']:.6g} "
                      f"min={bs['min']:.6g} max={bs['max']:.6g}")

                # 检查attn类型
                attn_prefix = wkey.rsplit(".gate_proj", 1)[0] if wkey else (bkey.rsplit(".gate_proj", 1)[0] if bkey else None)
                if attn_prefix is not None:
                    in_proj_key = attn_prefix + ".in_proj_weight"
                    q_proj_key = attn_prefix + ".q_proj_weight"
                    k_proj_key = attn_prefix + ".k_proj_weight"
                    v_proj_key = attn_prefix + ".v_proj_weight"
                    found = []
                    if in_proj_key in stripped_to_orig:
                        found.append("in_proj_weight (MultiheadAttention原版)")
                    if q_proj_key in stripped_to_orig:
                        found.append("q_proj_weight")
                    if k_proj_key in stripped_to_orig:
                        found.append("k_proj_weight")
                    if v_proj_key in stripped_to_orig:
                        found.append("v_proj_weight")
                    if found:
                        print(f"        [ATTN类型] 当前attn权重包含: {', '.join(found)}")
                    else:
                        print(f"        [ATTN类型] 未检测到 in_proj_weight 或 q/k/v_proj_weight")
            if bkey is not None:
                b = state[stripped_to_orig[bkey]]
                bs = _tensor_stats(b)
                print(f"        gate_proj.bias  : shape={bs['shape']} mean={bs['mean']:.6g} std={bs['std']:.6g} "
                      f"min={bs['min']:.6g} max={bs['max']:.6g}")

                # headwise gate：bias 维度通常是 num_heads（比如 vision 12 / text 8）
                if len(bs["shape"]) == 1 and bs["shape"][0] <= 64:
                    _print_headwise_bias_info(b, topk=5)
                else:
                    # elementwise gate bias 很长：给一些分位数
                    g = torch.sigmoid(b.detach().float().cpu().flatten())
                    qs = torch.quantile(g, torch.tensor([0.01, 0.1, 0.5, 0.9, 0.99]))
                    print(f"            sigmoid(bias) quantiles: "
                          f"p1={float(qs[0]):.4f}, p10={float(qs[1]):.4f}, "
                          f"p50={float(qs[2]):.4f}, p90={float(qs[3]):.4f}, p99={float(qs[4]):.4f}")
            else:
                print("        gate_proj.bias  : NOT FOUND")

            # 检查当前 gate_proj 所在 attn 是哪种实现
            attn_prefix = wkey.rsplit(".gate_proj", 1)[0] if wkey else (bkey.rsplit(".gate_proj", 1)[0] if bkey else None)
            if attn_prefix is not None:
                # MultiheadAttention 原版
                in_proj_key = attn_prefix + ".in_proj_weight"
                # 拆分QKV
                q_proj_key = attn_prefix + ".q_proj_weight"
                k_proj_key = attn_prefix + ".k_proj_weight"
                v_proj_key = attn_prefix + ".v_proj_weight"
                found = []
                if in_proj_key in stripped_to_orig:
                    found.append("in_proj_weight (MultiheadAttention原版)")
                if q_proj_key in stripped_to_orig:
                    found.append("q_proj_weight")
                if k_proj_key in stripped_to_orig:
                    found.append("k_proj_weight")
                if v_proj_key in stripped_to_orig:
                    found.append("v_proj_weight")
                if found:
                    print(f"        [ATTN类型] 当前attn权重包含: {', '.join(found)}")
                else:
                    print(f"        [ATTN类型] 未检测到 in_proj_weight 或 q/k/v_proj_weight")

if __name__ == "__main__":
    inspect_gates(CKPT_PATH)
