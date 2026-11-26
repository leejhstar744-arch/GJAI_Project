"""Pack ultralytics ONNX exports for the browser, losslessly and about half the size.

The .pt weights are stored in fp16; export fuses Conv+BN in fp32, so a fused weight row is fl32(s_i * w16_ij) with one
fp32 scale per output channel. We ship w16 (from the .pt) + s (fp32) and the browser rebuilds the exact fp32 bytes.
Tensors where that does not hold bit-exactly are shipped as raw fp32.

out/<stem>.graph.onnx   graph, every initializer as external data "<stem>.weights" (fp32 layout)
out/<stem>.pack.json    {"weights_bytes", "parts": [file names], "tensors": [{o, n, kind, p, axis, rows, cols}]}
out/<stem>.pack.NN      packed bytes split in parts (<= PART bytes)

usage: python pack_models.py out_dir model.onnx:model.pt ...
"""
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import numpy_helper
from onnx.external_data_helper import set_external_data

PART = 15 * 1024 * 1024


def f32(x):
    return np.asarray(x, np.float32)


def try_scale(W, w16, axis):
    """Per-slice fp32 scale s with fl32(s * w16) == W exactly along axis; None if impossible."""
    Wm = np.moveaxis(W, axis, 0).reshape(W.shape[axis], -1)
    wm = np.moveaxis(w16.astype(np.float32), axis, 0).reshape(W.shape[axis], -1)
    s = np.zeros(Wm.shape[0], np.float32)
    for i in range(Wm.shape[0]):
        nz = np.nonzero(wm[i])[0]
        if len(nz) == 0:
            if np.any(Wm[i] != 0):
                return None
            continue
        k = nz[np.argmax(np.abs(wm[i][nz]))]
        c = np.float32(Wm[i, k] / np.float64(wm[i, k]))
        ok = None
        for cand in [c, np.nextafter(c, np.float32(np.inf)), np.nextafter(c, np.float32(-np.inf))]:
            if np.array_equal((wm[i].astype(np.float64) * np.float64(cand)).astype(np.float32), Wm[i]):
                ok = cand
                break
        if ok is None:
            return None
        s[i] = ok
    return s


def pack(onnx_path, pt_path, out):
    stem = Path(onnx_path).stem
    m = onnx.load(onnx_path)
    ck = torch.load(pt_path, weights_only=False)
    sd = {k: v.cpu().numpy() for k, v in (ck.get("ema") or ck["model"]).state_dict().items()}
    blob, tensors, off = bytearray(), [], 0
    for t in m.graph.initializer:
        a = numpy_helper.to_array(t)
        raw = a.tobytes()
        if len(raw) < 4096:  # small tensors (shapes, split sizes, biases) stay inside the graph
            continue
        rec = dict(o=off, n=len(raw))
        w16 = sd.get(t.name)
        done = False
        if a.dtype == np.float32 and a.size >= 4096 and w16 is not None and w16.dtype == np.float16 and w16.shape == a.shape:
            for axis in (0, 1):
                s = try_scale(a, w16, axis)
                if s is not None:
                    rows = a.shape[axis]
                    cols = a.size // rows
                    rec.update(kind=1, p=len(blob), axis=axis, rows=rows, cols=cols, shape=list(a.shape))
                    blob += s.tobytes() + np.moveaxis(w16, axis, 0).reshape(rows, -1).astype(np.float16).tobytes()
                    done = True
                    break
        if not done:
            rec.update(kind=0, p=len(blob))
            blob += raw
        while len(blob) % 4:
            blob += b"\0"
        tensors.append(rec)
        set_external_data(t, location=f"{stem}.weights", offset=off, length=len(raw))
        t.ClearField("raw_data")
        t.data_location = onnx.TensorProto.EXTERNAL
        off += len(raw)
        off += (-off) % 16
    out.mkdir(parents=True, exist_ok=True)
    onnx.save(m, out / f"{stem}.graph.onnx")
    parts = []
    for i in range(0, len(blob), PART):
        name = f"{stem}.pack.{i // PART:02d}"
        (out / name).write_bytes(blob[i:i + PART])
        parts.append(name)
    meta = dict(weights_bytes=off, parts=parts, packed_bytes=len(blob), tensors=tensors)
    (out / f"{stem}.pack.json").write_text(json.dumps(meta, separators=(",", ":")))
    k1 = sum(r["n"] for r in tensors if r["kind"] == 1)
    print(f"{stem}: fp32 {off / 1e6:.1f} MB -> packed {len(blob) / 1e6:.1f} MB ({k1 / max(off, 1):.0%} of bytes as fp16+scale), "
          f"graph {(out / f'{stem}.graph.onnx').stat().st_size / 1e3:.0f} kB, {len(parts)} parts")
    return meta


def unpack(out, stem):
    """Python twin of the browser unpacker, for verification."""
    meta = json.loads((out / f"{stem}.pack.json").read_text())
    blob = b"".join((out / p).read_bytes() for p in meta["parts"])
    W = bytearray(meta["weights_bytes"])
    for r in meta["tensors"]:
        if r["kind"] == 0:
            W[r["o"]:r["o"] + r["n"]] = blob[r["p"]:r["p"] + r["n"]]
        else:
            rows, cols = r["rows"], r["cols"]
            s = np.frombuffer(blob, np.float32, rows, r["p"])
            w = np.frombuffer(blob, np.float16, rows * cols, r["p"] + 4 * rows).reshape(rows, cols)
            v = (w.astype(np.float64) * s[:, None].astype(np.float64)).astype(np.float32)
            shp = r["shape"]
            moved = [shp[r["axis"]]] + [d for i, d in enumerate(shp) if i != r["axis"]]
            v = np.moveaxis(v.reshape(moved), 0, r["axis"])
            W[r["o"]:r["o"] + r["n"]] = np.ascontiguousarray(v).tobytes()
    return bytes(W)


if __name__ == "__main__":
    out = Path(sys.argv[1])
    for spec in sys.argv[2:]:
        o, p = spec.split(":")
        pack(o, p, out)
