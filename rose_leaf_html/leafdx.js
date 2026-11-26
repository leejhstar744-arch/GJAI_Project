/* leafdx.js: browser port of package/leafdx (stage 1 leaf search + stage 2 leaf state + VLM views).
 *
 * Follows the Python code step by step so results match it leaf by leaf:
 *   stage1.py   LeafSegmenter      (ultralytics YOLO-seg predict: LetterBox, NMS, retina masks, tiling, dedup)
 *   stage2.py   LeafClassifier     (gray_world, to_canvas with Pillow's resampling, YOLO-cls, thresholds)
 *   pipeline.py LeafPipeline.run   (reading order, polygon, solidity, sharpness, dark share)
 *   vlm.py      views / select / verdict / prompt
 * Needs the globals `ort` (onnxruntime-web) and `cv` (OpenCV.js). Images are {data: Uint8ClampedArray RGB, width, height}.
 */
(function (global) {
  "use strict";

  // ------------------------------------------------------------------ small helpers
  const f32 = Math.fround;

  /** Python round(x) for floats: half to even. */
  function pyRound(x) {
    const r = Math.round(x);
    return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
  }

  /** Python round(x, n) for display values. */
  function roundN(x, n) {
    return Number(Number(x).toFixed(n));
  }

  const sleep0 = () => new Promise((r) => setTimeout(r, 0));

  // ------------------------------------------------------------------ fp16 -> fp32
  let F16 = null;
  function f16Table() {
    if (F16) return F16;
    F16 = new Float32Array(65536);
    for (let h = 0; h < 65536; h++) {
      const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
      F16[h] = e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15);
    }
    return F16;
  }

  /** Rebuild the exact fp32 external weights from a pack (see tools/pack_models.py). */
  function unpackWeights(meta, blob) {
    const T = f16Table();
    const out = new Uint8Array(meta.weights_bytes);
    const outF = new Float32Array(out.buffer);
    for (const r of meta.tensors) {
      if (r.kind === 0) {
        out.set(blob.subarray(r.p, r.p + r.n), r.o);
        continue;
      }
      const rows = r.rows, cols = r.cols;
      const s = new Float32Array(blob.buffer, blob.byteOffset + r.p, rows);
      const w = new Uint16Array(blob.buffer, blob.byteOffset + r.p + 4 * rows, rows * cols);
      const base = r.o / 4;
      if (r.axis === 0) {
        for (let i = 0; i < rows; i++) {
          const si = s[i], o = base + i * cols, wi = i * cols;
          for (let j = 0; j < cols; j++) outF[o + j] = si * T[w[wi + j]];
        }
      } else { // axis 1: shape [d0, d1, rest...] was stored as [d1, d0 * rest]
        const d0 = r.shape[0], d1 = r.shape[1], rest = cols / d0;
        for (let i = 0; i < rows; i++) {
          const si = s[i];
          for (let a = 0; a < d0; a++) {
            const o = base + (a * d1 + i) * rest, wi = i * cols + a * rest;
            for (let k = 0; k < rest; k++) outF[o + k] = si * T[w[wi + k]];
          }
        }
      }
    }
    return out;
  }

  // ------------------------------------------------------------------ model loading
  /** fetchBytes(name) -> Promise<Uint8Array>; onProgress(bytesDone). Returns an ort.InferenceSession. */
  async function loadModel(stem, fetchBytes, sessionOptions, onProgress) {
    const meta = JSON.parse(new TextDecoder().decode(await fetchBytes(`${stem}.pack.json`)));
    const graph = await fetchBytes(`${stem}.graph.onnx`);
    const blob = new Uint8Array(meta.packed_bytes);
    let off = 0;
    for (const p of meta.parts) {
      const b = await fetchBytes(p);
      blob.set(b, off);
      off += b.length;
      if (onProgress) onProgress(b.length);
    }
    const weights = unpackWeights(meta, blob);
    const opts = Object.assign({}, sessionOptions, { externalData: [{ path: `${stem}.weights`, data: weights }] });
    return ort.InferenceSession.create(graph, opts);
  }

  // ------------------------------------------------------------------ image ops
  function cvResize(img, w, h) {
    if (img.width === w && img.height === h) return img;
    const src = cv.matFromArray(img.height, img.width, cv.CV_8UC3, img.data);
    const dst = new cv.Mat();
    cv.resize(src, dst, new cv.Size(w, h), 0, 0, cv.INTER_LINEAR);
    const out = { data: new Uint8ClampedArray(dst.data), width: w, height: h };
    src.delete(); dst.delete();
    return out;
  }

  function crop(img, x0, y0, x1, y1) {
    const w = x1 - x0, h = y1 - y0, out = new Uint8ClampedArray(w * h * 3);
    for (let y = 0; y < h; y++) out.set(img.data.subarray(((y0 + y) * img.width + x0) * 3, ((y0 + y) * img.width + x1) * 3), y * w * 3);
    return { data: out, width: w, height: h };
  }

  /** ultralytics LetterBox (scaleup, center, pad 114) -> {tensor data CHW float32 /255, H, W}. */
  function letterbox(img, size, auto, stride = 32) {
    const h = img.height, w = img.width;
    const r = Math.min(size / h, size / w);
    const nw = Math.max(pyRound(w * r), 1), nh = Math.max(pyRound(h * r), 1);
    let dw = size - nw, dh = size - nh;
    if (auto) { dw %= stride; dh %= stride; }
    dw /= 2; dh /= 2;
    const top = pyRound(dh - 0.1), bottom = pyRound(dh + 0.1), left = pyRound(dw - 0.1), right = pyRound(dw + 0.1);
    const rs = cvResize(img, nw, nh);
    const H = nh + top + bottom, W = nw + left + right, plane = H * W;
    const t = new Float32Array(3 * plane).fill(114 / 255);
    const d = rs.data;
    for (let y = 0; y < nh; y++) {
      let o = (y + top) * W + left, i = y * nw * 3;
      for (let x = 0; x < nw; x++, o++, i += 3) {
        t[o] = d[i] / 255; t[plane + o] = d[i + 1] / 255; t[2 * plane + o] = d[i + 2] / 255;
      }
    }
    return { data: t, H, W };
  }

  /** ultralytics scale_boxes (xyxy, letterbox padding) + clip, in place on a Float32Array of 4 values. */
  function scaleBox(b, H1, W1, h0, w0) {
    const gain = Math.min(H1 / h0, W1 / w0);
    const nh = Math.max(pyRound(h0 * gain), 1), nw = Math.max(pyRound(w0 * gain), 1);
    const gy = f32(nh / h0), gx = f32(nw / w0);
    const px = pyRound((W1 - nw) / 2 - 0.1), py = pyRound((H1 - nh) / 2 - 0.1);
    b[0] = Math.min(Math.max(f32(f32(b[0] - px) / gx), 0), w0);
    b[1] = Math.min(Math.max(f32(f32(b[1] - py) / gy), 0), h0);
    b[2] = Math.min(Math.max(f32(f32(b[2] - px) / gx), 0), w0);
    b[3] = Math.min(Math.max(f32(f32(b[3] - py) / gy), 0), h0);
    return b;
  }

  // ------------------------------------------------------------------ YOLO-seg
  /** ultralytics non_max_suppression for one image, nc = 1. Returns [{box: Float32Array(4) xyxy, conf, k (anchor index)}]. */
  function nms(out0, nAnch, conf, iou = 0.7, maxDet = 300) {
    const C = out0.length / nAnch, cand = [];
    for (let a = 0; a < nAnch; a++) {
      const s = out0[4 * nAnch + a];
      if (s > conf) {
        const cx = out0[a], cy = out0[nAnch + a], w = out0[2 * nAnch + a], h = out0[3 * nAnch + a];
        const dw = f32(w / 2), dh = f32(h / 2);
        cand.push({ box: new Float32Array([f32(cx - dw), f32(cy - dh), f32(cx + dw), f32(cy + dh)]), conf: s, k: a });
      }
    }
    if (C < 5) return [];
    cand.sort((p, q) => q.conf - p.conf);
    const keep = [], areas = cand.map((c) => f32(f32(c.box[2] - c.box[0]) * f32(c.box[3] - c.box[1])));
    const dead = new Uint8Array(cand.length);
    for (let i = 0; i < cand.length; i++) {
      if (dead[i]) continue;
      keep.push(cand[i]);
      if (keep.length >= maxDet) break;
      const a = cand[i].box;
      for (let j = i + 1; j < cand.length; j++) {
        if (dead[j]) continue;
        const b = cand[j].box;
        const w = Math.max(0, f32(Math.min(a[2], b[2]) - Math.max(a[0], b[0])));
        const h = Math.max(0, f32(Math.min(a[3], b[3]) - Math.max(a[1], b[1])));
        const inter = f32(w * h);
        if (f32(inter / f32(f32(areas[i] + areas[j]) - inter)) > iou) dead[j] = 1;
      }
    }
    return keep;
  }

  /** PyTorch bilinear (align_corners=False) source coordinate. */
  function srcIndex(scale, dst, inSize) {
    let s = f32(f32(scale * f32(dst + 0.5)) - 0.5);
    if (s < 0) s = 0;
    const i0 = Math.floor(s), i1 = i0 < inSize - 1 ? i0 + 1 : i0;
    const l1 = f32(s - i0);
    return [i0, i1, f32(1 - l1), l1];
  }

  /** process_mask_native for one detection, then threshold and crop to the box: returns Uint8Array(h*w) or null. */
  function retinaMask(proto, mh, mw, coef, box, h, w) {
    const gain = Math.min(4 * mh / h, 4 * mw / w);
    const pad = (4 * mh - Math.max(pyRound(h * gain), 1)) | (4 * mw - Math.max(pyRound(w * gain), 1));
    const gcd = (a, b) => (b ? gcd(b, a % b) : a);
    const f = 4 / gcd(4, pad | (pad >> 1));
    const plane = mh * mw;
    // logits at prototype resolution (computed lazily per pixel)
    const logit = new Float32Array(plane), have = new Uint8Array(plane);
    const L = (i) => {
      if (!have[i]) {
        let s = 0;
        for (let c = 0; c < 32; c++) s += coef[c] * proto[c * plane + i];
        logit[i] = s; have[i] = 1;
      }
      return logit[i];
    };
    let get, sh = mh, sw = mw;
    if (f > 1) { // pre-upsample by f (scale_factor path)
      sh = mh * f; sw = mw * f;
      const up = new Float32Array(sh * sw);
      const sc = f32(1 / f);
      const ys = [], xs = [];
      for (let y = 0; y < sh; y++) ys.push(srcIndex(sc, y, mh));
      for (let x = 0; x < sw; x++) xs.push(srcIndex(sc, x, mw));
      for (let y = 0; y < sh; y++) {
        const [y0, y1, a0, a1] = ys[y];
        for (let x = 0; x < sw; x++) {
          const [x0, x1, b0, b1] = xs[x];
          up[y * sw + x] = a0 * (b0 * L(y0 * mw + x0) + b1 * L(y0 * mw + x1)) + a1 * (b0 * L(y1 * mw + x0) + b1 * L(y1 * mw + x1));
        }
      }
      get = (y, x) => up[y * sw + x];
    } else {
      get = (y, x) => L(y * mw + x);
    }
    // scale_masks: crop the letterbox padding, resize to (h, w)
    const g = Math.min(sh / h, sw / w);
    const ch = Math.max(pyRound(h * g), 1), cw = Math.max(pyRound(w * g), 1);
    const top = pyRound((sh - ch) / 2 - 0.1), left = pyRound((sw - cw) / 2 - 0.1);
    const scy = f32(ch / h), scx = f32(cw / w);
    // crop_mask keeps columns x1 <= c < x2 and rows y1 <= r < y2
    const cx0 = Math.max(0, Math.ceil(box[0])), cx1 = Math.min(w, Math.ceil(box[2]));
    const cy0 = Math.max(0, Math.ceil(box[1])), cy1 = Math.min(h, Math.ceil(box[3]));
    if (cx0 >= cx1 || cy0 >= cy1) return null;
    const m = new Uint8Array(h * w);
    const xs = [];
    for (let x = cx0; x < cx1; x++) xs.push(srcIndex(scx, x, cw));
    let any = false;
    for (let y = cy0; y < cy1; y++) {
      const [y0, y1, a0, a1] = srcIndex(scy, y, ch);
      const r0 = top + y0, r1 = top + y1;
      for (let i = 0, x = cx0; x < cx1; x++, i++) {
        const [x0, x1, b0, b1] = xs[i];
        const v = a0 * (b0 * get(r0, left + x0) + b1 * get(r0, left + x1)) + a1 * (b0 * get(r1, left + x0) + b1 * get(r1, left + x1));
        if (f32(v) > 0) { m[y * w + x] = 1; any = true; }
      }
    }
    return any ? m : null;
  }

  /** process_mask(upsample=True) for one detection: is any pixel of the input-size mask inside the box > 0? */
  function anyMaskUpsample(proto, mh, mw, coef, box, H1, W1) {
    const plane = mh * mw, scy = f32(mh / H1), scx = f32(mw / W1);
    const cx0 = Math.max(0, Math.ceil(box[0])), cx1 = Math.min(W1, Math.ceil(box[2]));
    const cy0 = Math.max(0, Math.ceil(box[1])), cy1 = Math.min(H1, Math.ceil(box[3]));
    const cache = new Map();
    const L = (i) => {
      let v = cache.get(i);
      if (v === undefined) { v = 0; for (let c = 0; c < 32; c++) v += coef[c] * proto[c * plane + i]; cache.set(i, v); }
      return v;
    };
    for (let y = cy0; y < cy1; y++) {
      const [y0, y1, a0, a1] = srcIndex(scy, y, mh);
      for (let x = cx0; x < cx1; x++) {
        const [x0, x1, b0, b1] = srcIndex(scx, x, mw);
        if (f32(a0 * (b0 * L(y0 * mw + x0) + b1 * L(y0 * mw + x1)) + a1 * (b0 * L(y1 * mw + x0) + b1 * L(y1 * mw + x1))) > 0) return true;
      }
    }
    return false;
  }

  class LeafSegmenter {
    constructor(session, cfg) {
      this.s = session;
      this.conf = cfg.conf; this.tileMult = cfg.tile_mult; this.imgsz = cfg.imgsz; this.estImgsz = cfg.est_imgsz;
      this.minArea = cfg.min_area; this.dedupIou = cfg.dedup_iou;
    }

    async _run(img, size, retina) {
      const lb = letterbox(img, size, true);
      const feeds = { images: new ort.Tensor("float32", lb.data, [1, 3, lb.H, lb.W]) };
      const r = await this.s.run(feeds);
      const o0 = r.output0, o1 = r.output1;
      const nA = o0.dims[2];
      const out0 = o0.location === "cpu" || !o0.getData ? o0.data : await o0.getData();
      let dets = nms(out0, nA, this.conf);
      const res = { H1: lb.H, W1: lb.W };
      res.proto = o1.location === "cpu" || !o1.getData ? o1.data : await o1.getData();
      res.mh = o1.dims[2]; res.mw = o1.dims[3];
      const C = o0.dims[1];
      for (const d of dets) {
        d.coef = new Float32Array(32);
        for (let c = 0; c < 32; c++) d.coef[c] = out0[(C - 32 + c) * nA + d.k];
      }
      // predictor keeps only detections whose mask is not empty (non-retina: mask at input size, cropped to the box)
      if (!retina) dets = dets.filter((d) => anyMaskUpsample(res.proto, res.mh, res.mw, d.coef, d.box, lb.H, lb.W));
      res.dets = dets;
      o0.dispose && o0.dispose(); o1.dispose && o1.dispose();
      return res;
    }

    async estimateLeafLength(img) {
      const r = await this._run(img, this.estImgsz, false);
      if (r.dets.length < 5) return 90.0;
      const ls = r.dets.map((d) => {
        const b = scaleBox(d.box, r.H1, r.W1, img.height, img.width);
        return f32(Math.max(f32(b[2] - b[0]), f32(b[3] - b[1])));
      }).sort((a, b) => a - b);
      const n = ls.length;
      return n % 2 ? ls[(n - 1) / 2] : f32((ls[n / 2 - 1] + ls[n / 2]) / 2);
    }

    /** -> {leaves: [{conf, box:[x0,y0,x1,y1] exclusive, mask: Uint8Array(bw*bh)}], L} */
    async run(img, L, onTile) {
      const H = img.height, W = img.width;
      L = L ? Number(L) : await this.estimateLeafLength(img);
      const R = Math.trunc(Math.min(Math.max(H, W), Math.max(320, this.tileMult * L))), st = Math.max(1, Math.trunc(R * 0.75));
      const steps = (n) => {
        const v = new Set();
        for (let i = 0; i < Math.max(1, n - R); i += st) v.add(i);
        v.add(Math.max(0, n - R));
        return [...v].sort((a, b) => a - b);
      };
      const xs = steps(W), ys = steps(H);
      const preds = [];
      let t = 0;
      for (const ty of ys) {
        for (const tx of xs) {
          if (onTile) await onTile(t++, xs.length * ys.length);
          const x1 = Math.min(W, tx + R), y1 = Math.min(H, ty + R), tw = x1 - tx, th = y1 - ty;
          const r = await this._run(crop(img, tx, ty, x1, y1), this.imgsz, true);
          for (const d of r.dets) {
            const b = scaleBox(d.box, r.H1, r.W1, th, tw);
            const m = retinaMask(r.proto, r.mh, r.mw, d.coef, b, th, tw);
            if (!m) continue;
            let n = 0, mnx = tw, mny = th, mxx = -1, mxy = -1;
            for (let y = 0; y < th; y++) {
              const row = y * tw;
              for (let x = 0; x < tw; x++) {
                if (m[row + x]) { n++; if (x < mnx) mnx = x; if (x > mxx) mxx = x; if (y < mny) mny = y; if (y > mxy) mxy = y; }
              }
            }
            if (n < 20) continue;
            const e = 2;
            if ((mnx <= e && tx > 0) || (mny <= e && ty > 0) || (mxx >= tw - 1 - e && x1 < W) || (mxy >= th - 1 - e && y1 < H)) continue;
            const bx = [tx + mnx, ty + mny, tx + mxx + 1, ty + mxy + 1];
            const bw = mxx + 1 - mnx, bh = mxy + 1 - mny, mm = new Uint8Array(bw * bh);
            let area = 0;
            for (let y = 0; y < bh; y++) for (let x = 0; x < bw; x++) { const v = m[(mny + y) * tw + mnx + x]; mm[y * bw + x] = v; area += v; }
            preds.push({ conf: d.conf, box: bx, mask: mm, area });
          }
        }
      }
      preds.sort((p, q) => q.conf - p.conf); // stable, like Python's sort with key=-conf
      const keep = [];
      for (const p of preds) {
        const b = p.box;
        let dup = false;
        for (const q of keep) {
          const b2 = q.box;
          const ix0 = Math.max(b[0], b2[0]), iy0 = Math.max(b[1], b2[1]), ix1 = Math.min(b[2], b2[2]), iy1 = Math.min(b[3], b2[3]);
          if (ix0 >= ix1 || iy0 >= iy1) continue;
          const w1 = b[2] - b[0], w2 = b2[2] - b2[0];
          let inter = 0;
          for (let y = iy0; y < iy1; y++) {
            const r1 = (y - b[1]) * w1 - b[0], r2 = (y - b2[1]) * w2 - b2[0];
            for (let x = ix0; x < ix1; x++) inter += p.mask[r1 + x] & q.mask[r2 + x];
          }
          if (inter && inter / (p.area + q.area - inter) > this.dedupIou) { dup = true; break; }
        }
        if (!dup) keep.push(p);
      }
      return { leaves: keep.filter((p) => p.area >= this.minArea), L };
    }
  }

  // ------------------------------------------------------------------ stage 2
  /** stage2.gray_world (float64 maths like numpy, truncation to uint8). */
  function grayWorld(img) {
    const d = img.data, n = img.width * img.height;
    let s0 = 0, s1 = 0, s2 = 0, cnt = 0;
    const lim = 60 / 255;
    for (let i = 0, j = 0; i < n; i++, j += 3) {
      const r = d[j], g = d[j + 1], b = d[j + 2];
      const mx = r > g ? (r > b ? r : b) : (g > b ? g : b), mn = r < g ? (r < b ? r : b) : (g < b ? g : b);
      if ((mx - mn) / Math.max(mx, 1) < lim && mx > 60 && mx < 240) { s0 += r; s1 += g; s2 += b; cnt++; }
    }
    if (cnt < 1000) {
      s0 = s1 = s2 = 0;
      for (let j = 0; j < 3 * n; j += 3) { s0 += d[j]; s1 += d[j + 1]; s2 += d[j + 2]; }
      cnt = n;
    }
    const m = [s0 / cnt, s1 / cnt, s2 / cnt], mm = (m[0] + m[1] + m[2]) / 3, k = m.map((v) => mm / v);
    const out = new Uint8ClampedArray(3 * n);
    for (let j = 0; j < 3 * n; j += 3) {
      for (let c = 0; c < 3; c++) { const v = d[j + c] * k[c]; out[j + c] = v >= 255 ? 255 : v <= 0 ? 0 : Math.trunc(v); }
    }
    return { data: out, width: img.width, height: img.height };
  }

  /** pipeline._simple_gray_world (float32 maths). */
  function simpleGrayWorld(img) {
    const d = img.data, n = img.width * img.height;
    const s = [0, 0, 0];
    for (let j = 0; j < 3 * n; j += 3) { s[0] += d[j]; s[1] += d[j + 1]; s[2] += d[j + 2]; }
    const m = s.map((v) => f32(v / n)), mm = f32(f32(f32(m[0] + m[1]) + m[2]) / 3), k = m.map((v) => f32(mm / Math.max(v, 1)));
    const out = new Uint8ClampedArray(3 * n);
    for (let j = 0; j < 3 * n; j += 3) for (let c = 0; c < 3; c++) {
      const v = f32(d[j + c] * k[c]);
      out[j + c] = v >= 255 ? 255 : v <= 0 ? 0 : Math.trunc(v);
    }
    return { data: out, width: img.width, height: img.height };
  }

  // Pillow Resample.c (8 bpc), filters BILINEAR and LANCZOS
  const PREC = 22;
  const FILTERS = {
    bilinear: { support: 1, fn: (x) => { x = Math.abs(x); return x < 1 ? 1 - x : 0; } },
    lanczos: {
      support: 3,
      fn: (x) => {
        const sinc = (t) => { if (t === 0) return 1; t *= Math.PI; return Math.sin(t) / t; };
        return -3 <= x && x < 3 ? sinc(x) * sinc(x / 3) : 0;
      },
    },
  };

  function pilCoeffs(inSize, outSize, filter) {
    const scale = inSize / outSize;
    const fscale = Math.max(scale, 1), support = filter.support * fscale, ksize = Math.ceil(support) * 2 + 1;
    const bounds = new Int32Array(outSize * 2), kk = new Int32Array(outSize * ksize), inv = 1 / fscale;
    const k = new Float64Array(ksize);
    for (let xx = 0; xx < outSize; xx++) {
      const center = (xx + 0.5) * scale;
      let ww = 0;
      let xmin = Math.trunc(center - support + 0.5); if (xmin < 0) xmin = 0;
      let xmax = Math.trunc(center + support + 0.5); if (xmax > inSize) xmax = inSize;
      xmax -= xmin;
      for (let x = 0; x < xmax; x++) { const w = filter.fn((x + xmin - center + 0.5) * inv); k[x] = w; ww += w; }
      for (let x = 0; x < xmax; x++) {
        const v = ww !== 0 ? k[x] / ww : k[x];
        kk[xx * ksize + x] = v < 0 ? Math.trunc(-0.5 + v * (1 << PREC)) : Math.trunc(0.5 + v * (1 << PREC));
      }
      bounds[2 * xx] = xmin; bounds[2 * xx + 1] = xmax;
    }
    return { ksize, bounds, kk };
  }

  const clip8 = (v) => { v >>= PREC; return v < 0 ? 0 : v > 255 ? 255 : v; };

  /** Image.resize for an RGBA image (4 bytes / pixel), including Pillow's RGBa premultiplication. */
  function pilResizeRGBA(src, sw, sh, w, h, filterName) {
    if (sw === w && sh === h) return src.slice();
    const filter = FILTERS[filterName];
    // RGBA -> RGBa
    const pm = new Uint8Array(src.length);
    for (let i = 0; i < src.length; i += 4) {
      const a = src[i + 3];
      for (let c = 0; c < 3; c++) { const t = src[i + c] * a + 128; pm[i + c] = ((t >> 8) + t) >> 8; }
      pm[i + 3] = a;
    }
    const V = pilCoeffs(sh, h, filter);
    const yfirst = V.bounds[0], ylast = V.bounds[2 * h - 2] + V.bounds[2 * h - 1];
    let cur = pm, cw = sw, chh = sh, yoff = 0;
    const needH = w !== sw, needV = h !== sh;
    if (needH) {
      const Hc = pilCoeffs(sw, w, filter), rows = ylast - yfirst, tmp = new Uint8Array(w * rows * 4);
      for (let yy = 0; yy < rows; yy++) {
        const li = (yy + yfirst) * sw * 4, lo = yy * w * 4;
        for (let xx = 0; xx < w; xx++) {
          const xmin = Hc.bounds[2 * xx], xmax = Hc.bounds[2 * xx + 1], kb = xx * Hc.ksize;
          let s0 = 1 << (PREC - 1), s1 = s0, s2 = s0, s3 = s0;
          for (let x = 0; x < xmax; x++) {
            const k = Hc.kk[kb + x], p = li + (x + xmin) * 4;
            s0 += cur[p] * k; s1 += cur[p + 1] * k; s2 += cur[p + 2] * k; s3 += cur[p + 3] * k;
          }
          const o = lo + xx * 4;
          tmp[o] = clip8(s0); tmp[o + 1] = clip8(s1); tmp[o + 2] = clip8(s2); tmp[o + 3] = clip8(s3);
        }
      }
      cur = tmp; cw = w; chh = rows; yoff = yfirst;
    }
    let out = cur;
    if (needV) {
      out = new Uint8Array(cw * h * 4);
      for (let yy = 0; yy < h; yy++) {
        const ymin = V.bounds[2 * yy] - (needH ? yoff : 0), ymax = V.bounds[2 * yy + 1], kb = yy * V.ksize;
        for (let xx = 0; xx < cw; xx++) {
          let s0 = 1 << (PREC - 1), s1 = s0, s2 = s0, s3 = s0;
          for (let y = 0; y < ymax; y++) {
            const k = V.kk[kb + y], p = ((y + ymin) * cw + xx) * 4;
            s0 += cur[p] * k; s1 += cur[p + 1] * k; s2 += cur[p + 2] * k; s3 += cur[p + 3] * k;
          }
          const o = (yy * cw + xx) * 4;
          out[o] = clip8(s0); out[o + 1] = clip8(s1); out[o + 2] = clip8(s2); out[o + 3] = clip8(s3);
        }
      }
    } else if (needH) {
      out = cur; // rows == h when no vertical pass
    }
    // RGBa -> RGBA
    const res = new Uint8Array(out.length);
    for (let i = 0; i < out.length; i += 4) {
      const a = out[i + 3];
      for (let c = 0; c < 3; c++) {
        if (a === 255 || a === 0) res[i + c] = out[i + c];
        else { const v = Math.trunc((255 * out[i + c]) / a); res[i + c] = v > 255 ? 255 : v; }
      }
      res[i + 3] = a;
    }
    return res;
  }

  /** stage2.to_canvas -> Uint8Array RGB size*size. */
  function toCanvas(img, leaf, size = 224, ratio = 0.6) {
    const [x0, y0, x1, y1] = leaf.box, bw = x1 - x0, bh = y1 - y0;
    const rgba = new Uint8Array(bw * bh * 4);
    for (let y = 0; y < bh; y++) for (let x = 0; x < bw; x++) {
      const i = ((y0 + y) * img.width + x0 + x) * 3, o = (y * bw + x) * 4;
      rgba[o] = img.data[i]; rgba[o + 1] = img.data[i + 1]; rgba[o + 2] = img.data[i + 2]; rgba[o + 3] = leaf._mask[y * bw + x] * 255;
    }
    const s = (size * ratio) / Math.max(bw, bh);
    const w = Math.max(1, pyRound(bw * s)), h = Math.max(1, pyRound(bh * s));
    const cut = pilResizeRGBA(rgba, bw, bh, w, h, s < 1 ? "lanczos" : "bilinear");
    const out = new Uint8Array(size * size * 3).fill(255);
    const ox = Math.floor((size - w) / 2), oy = Math.floor((size - h) / 2);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 4, a = cut[i + 3], o = ((oy + y) * size + ox + x) * 3;
      for (let c = 0; c < 3; c++) { const t = out[o + c] * (255 - a) + cut[i + c] * a + 128; out[o + c] = ((t >> 8) + t) >> 8; }
    }
    return out;
  }

  class LeafClassifier {
    constructor(session, states, cfg) {
      this.s = session;
      this.classes = ["black_spot", "downy_mildew", "healthy", "insects_infected"];
      this.healthy = this.classes.indexOf("healthy");
      this.anom = [0, 1, 3];
      this.th = states.thresholds;
      this.ratio = cfg.canvas_ratio; this.minSolidity = cfg.min_solidity; this.batch = cfg.cls_batch || 16;
    }

    async probs(img, leaves, onBatch) {
      const out = [];
      const S = 224, plane = S * S;
      for (let i = 0; i < leaves.length; i += this.batch) {
        if (onBatch) await onBatch(i, leaves.length);
        const part = leaves.slice(i, i + this.batch), n = part.length;
        const t = new Float32Array(n * 3 * plane);
        part.forEach((l, b) => {
          const c = toCanvas(img, l, S, this.ratio), base = b * 3 * plane;
          for (let p = 0, j = 0; p < plane; p++, j += 3) {
            t[base + p] = c[j] / 255; t[base + plane + p] = c[j + 1] / 255; t[base + 2 * plane + p] = c[j + 2] / 255;
          }
        });
        const r = await this.s.run({ images: new ort.Tensor("float32", t, [n, 3, S, S]) });
        const o = r.output0, d = o.location === "cpu" || !o.getData ? o.data : await o.getData();
        for (let b = 0; b < n; b++) out.push(Array.from(d.subarray(b * 4, b * 4 + 4)));
        o.dispose && o.dispose();
      }
      return out;
    }

    state(leaf, p) {
      const th = this.th;
      if (leaf.solidity < this.minSolidity || leaf.touches_border) return ["HOLD_SHAPE", null];
      if (leaf.sharpness < th.s_min || leaf.dark > th.d_max || leaf.L < th.L_min) return ["HOLD", null];
      let k = this.anom[0];
      for (const j of this.anom) if (p[j] > p[k]) k = j;
      if (p[k] >= th.t_anom) return ["ANOMALY", this.classes[k]];
      if (p[this.healthy] >= th.t_fresh) return ["FRESH", null];
      return ["UNKNOWN", null];
    }
  }

  // ------------------------------------------------------------------ pipeline
  function gray(img) {
    const m = cv.matFromArray(img.height, img.width, cv.CV_8UC3, img.data);
    const g = new cv.Mat(), lap = new cv.Mat();
    cv.cvtColor(m, g, cv.COLOR_RGB2GRAY);
    cv.Laplacian(g, lap, cv.CV_32F);
    const out = new Float32Array(lap.data32F);
    m.delete(); g.delete(); lap.delete();
    return out;
  }

  function contourInfo(mask, bw, bh) {
    const m = cv.matFromArray(bh, bw, cv.CV_8UC1, mask);
    const cs = new cv.MatVector(), hier = new cv.Mat();
    cv.findContours(m, cs, hier, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
    let best = 0, bestA = -1;
    for (let i = 0; i < cs.size(); i++) {
      const a = cv.contourArea(cs.get(i));
      if (a > bestA) { bestA = a; best = i; }
    }
    const c = cs.get(best), ap = new cv.Mat(), hull = new cv.Mat();
    cv.approxPolyDP(c, ap, 1.0, true);
    cv.convexHull(c, hull, false, true);
    const hullArea = cv.contourArea(hull);
    const poly = [];
    for (let i = 0; i < ap.rows; i++) poly.push([ap.data32S[2 * i], ap.data32S[2 * i + 1]]);
    m.delete(); cs.delete(); hier.delete(); ap.delete(); hull.delete();
    return { poly, hullArea };
  }

  class LeafPipeline {
    constructor(seg, cls, states, cfg) {
      this.cfg = cfg;
      this.seg = new LeafSegmenter(seg, cfg);
      this.cls = new LeafClassifier(cls, states, cfg);
    }

    /** progress(stage, done, total) */
    async run(img, progress = () => {}, L = null) {
      const t0 = performance.now();
      const H = img.height, W = img.width;
      await progress("leaf length", 0, 1);
      const r1 = await this.seg.run(img, L, async (i, n) => { await progress("search", i, n); await sleep0(); });
      const found = r1.leaves;
      L = r1.L;
      const t1 = performance.now();
      const key = (f) => [Math.floor(Math.floor((f.box[1] + f.box[3]) / 2) / 150), Math.floor((f.box[0] + f.box[2]) / 2)];
      found.sort((a, b) => { const ka = key(a), kb = key(b); return ka[0] - kb[0] || ka[1] - kb[1]; });
      const im = simpleGrayWorld(img);
      const lap = gray(im);
      const leaves = found.map((f, idx) => {
        const [x0, y0, x1, y1] = f.box, bw = x1 - x0, bh = y1 - y0, m = f.mask;
        const { poly, hullArea } = contourInfo(m, bw, bh);
        let n = 0, s = 0, s2 = 0, dark = 0;
        const vals = [];
        for (let y = 0; y < bh; y++) for (let x = 0; x < bw; x++) {
          if (!m[y * bw + x]) continue;
          const v = lap[(y0 + y) * W + x0 + x];
          vals.push(v); s += v; n++;
          const j = ((y0 + y) * W + x0 + x) * 3;
          const mx = Math.max(im.data[j], im.data[j + 1], im.data[j + 2]);
          if (f32(mx / 255) < 0.2) dark++;
        }
        const mean = s / n;
        for (const v of vals) s2 += (v - mean) * (v - mean);
        return {
          id: "L" + String(idx + 1).padStart(4, "0"), box: [x0, y0, x1, y1], polygon: poly.map(([x, y]) => [x + x0, y + y0]),
          area: f.area, conf: roundN(f.conf, 4), solidity: roundN(f.area / Math.max(1, hullArea), 3),
          touches_border: x0 <= 2 || y0 <= 2 || x1 >= W - 2 || y1 >= H - 2,
          sharpness: roundN(s2 / n, 1), dark: roundN(dark / n, 3), L: Math.max(bw, bh), _mask: m,
        };
      });
      const gw = this.cfg.white_balance ? grayWorld(img) : img;
      const P = await this.cls.probs(gw, leaves, async (i, n) => { await progress("classify", i, n); await sleep0(); });
      const counts = {};
      leaves.forEach((l, i) => {
        const p = P[i];
        const [st, ac] = this.cls.state(l, p);
        l.state = st; l.anomaly_class = ac;
        l.probs = {};
        this.cls.classes.forEach((c, j) => { l.probs[c] = roundN(p[j], 4); });
        const k = st !== "ANOMALY" ? st : `ANOMALY:${ac}`;
        counts[k] = (counts[k] || 0) + 1;
      });
      await progress("done", 1, 1);
      return {
        L: roundN(L, 1), size: [W, H], counts, leaves,
        timing: { stage1_s: roundN((t1 - t0) / 1000, 2), stage2_s: roundN((performance.now() - t1) / 1000, 2) },
      };
    }

    static toJSON(res) {
      return Object.assign({}, res, { leaves: res.leaves.map((l) => { const o = {}; for (const k in l) if (!k.startsWith("_")) o[k] = l[k]; return o; }) });
    }
  }

  // ------------------------------------------------------------------ VLM helpers (vlm.py)
  const VLM = {
    LABELS: ["healthy", "black_spot", "downy_mildew", "insects_infected", "not_a_leaf", "unclear"],
    SYSTEM: `You review single rose leaves for a plant-health monitor. The photos come from a growth chamber: miniature roses seen from above, lit by pink/purple LED grow lights, with reflective silver (mirror-like) walls.

You get two images of the same leaf, outlined in cyan. Image 1 is a close-up in which everything outside the outline is darkened. Image 2 shows the outlined leaf inside its surroundings. Judge only the outlined leaf. Answer with exactly one label:

- healthy: uniform green leaf. These are normal and NOT disease: serrated edges, reddish or bronze margins and young red-bronze leaves, pale yellow-green new growth, glare or specular highlights, the pink colour cast of the LED light, shadows from other leaves.
- black_spot: round dark brown to black spots, often with feathery or fringed edges, sometimes with yellowing around them.
- downy_mildew: angular purplish-red, brown or yellow blotches limited by the leaf veins; grey fuzzy growth on the underside.
- insects_infected: chewed holes, ragged or eaten edges, leaf-miner trails, fine stippling, visible insects, webbing or eggs.
- not_a_leaf: the outlined region is not one real leaf, e.g. a reflection in the mirror wall, a stem, the pot, background, or several leaves merged into one outline.
- unclear: the leaf is too small, blurred, dark or hidden to judge.

Pick a disease label only when you can point to the symptom in the image. Write "symptoms" and "reason" in Korean, one short sentence each.`,
    USER: "Image 1: close-up of the outlined leaf. Image 2: the same leaf in its surroundings. Classify the outlined leaf.",
    PRICE: { "claude-opus-5-5": [4.0, 20.0], "claude-sonnet-5-5": [2.0, 10.0], "claude-haiku-5-5": [0.1, 0.5] },
  };
  VLM.SCHEMA = {
    type: "object",
    properties: {
      label: { type: "string", enum: VLM.LABELS },
      confidence: { type: "string", enum: ["low", "medium", "high"] },
      symptoms: { type: "string" },
      reason: { type: "string" },
    },
    required: ["label", "confidence", "symptoms", "reason"],
    additionalProperties: false,
  };
  VLM.JSON_HINT = '\n\nReply with only one JSON object, no other text: {"label": "<one of ' + VLM.LABELS.join(", ") +
    '>", "confidence": "<low|medium|high>", "symptoms": "<Korean>", "reason": "<Korean>"}';

  VLM.parseAnswer = function (text) {
    const m = /\{[\s\S]*\}/.exec(text || "");
    let d = {};
    try { d = m ? JSON.parse(m[0]) : {}; } catch (e) { d = {}; }
    if (!VLM.LABELS.includes(d.label)) return { label: "unclear", confidence: "low", symptoms: "", reason: "", error: "unparsed answer: " + String(text).slice(0, 200) };
    return { label: d.label, confidence: String(d.confidence || "low"), symptoms: String(d.symptoms || ""), reason: String(d.reason || "") };
  };
  VLM.modelLabel = (l) => (l.state === "ANOMALY" ? l.anomaly_class : l.state === "FRESH" ? "healthy" : null);
  VLM.verdict = function (leaf, label) {
    if (label === "unclear" || label === "not_a_leaf") return label.toUpperCase();
    const m = VLM.modelLabel(leaf);
    if (m === null) return label === "healthy" ? "VLM_HEALTHY" : "VLM_ANOMALY";
    if (label === m) return "AGREE";
    if (m === "healthy") return "MISSED";
    if (label === "healthy") return "FALSE_ALARM";
    return "OTHER_CLASS";
  };
  /** vlm.select; the FRESH sample uses its own seeded shuffle (not Python's random), so the spot-check leaves can differ. */
  VLM.select = function (leaves, states = ["ANOMALY", "UNKNOWN"], nFresh = 5, maxN = 40, seed = 0) {
    const score = (l) => Math.max(...Object.entries(l.probs).filter(([k]) => k !== "healthy").map(([, v]) => v));
    const flagged = leaves.filter((l) => states.includes(l.state)).sort((a, b) => score(b) - score(a));
    const fresh = leaves.filter((l) => l.state === "FRESH" && !states.includes("FRESH"));
    let s = (seed + 0x9e3779b9) >>> 0;
    const rnd = () => { s = (s + 0x6d2b79f5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
    const pool = fresh.slice();
    for (let i = pool.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [pool[i], pool[j]] = [pool[j], pool[i]]; }
    return flagged.concat(pool.slice(0, Math.min(nFresh, pool.length))).slice(0, maxN);
  };

  /** vlm._outlined with canvas: square crop around the leaf, outside darkened, cyan outline, long side `side`. */
  function outlined(srcCanvas, leaf, scale, side, dim) {
    const W = srcCanvas.width, H = srcCanvas.height;
    const [x0, y0, x1, y1] = leaf.box;
    const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2, s = (Math.max(x1 - x0, y1 - y0) * scale) / 2;
    const a = Math.trunc(Math.max(0, cx - s)), b = Math.trunc(Math.max(0, cy - s)), c = Math.trunc(Math.min(W, cx + s)), d = Math.trunc(Math.min(H, cy + s));
    const k = side / Math.max(c - a, d - b);
    const w = Math.max(1, Math.round((c - a) * k)), h = Math.max(1, Math.round((d - b) * k));
    const cv2d = document.createElement("canvas");
    cv2d.width = w; cv2d.height = h;
    const g = cv2d.getContext("2d");
    g.imageSmoothingEnabled = true; g.imageSmoothingQuality = "high";
    g.drawImage(srcCanvas, a, b, c - a, d - b, 0, 0, w, h);
    const path = new Path2D();
    leaf.polygon.forEach(([x, y], i) => (i ? path.lineTo : path.moveTo).call(path, (x - a) * k, (y - b) * k));
    path.closePath();
    if (dim < 1) {
      g.save();
      g.fillStyle = `rgba(0,0,0,${1 - dim})`;
      const outside = new Path2D();
      outside.rect(0, 0, w, h);
      outside.addPath(path);
      g.fill(outside, "evenodd");
      g.restore();
    }
    g.strokeStyle = "rgb(0,255,255)";
    g.lineWidth = Math.max(2, Math.floor(side / 256));
    g.lineJoin = "round";
    g.stroke(path);
    return cv2d;
  }
  VLM.views = (srcCanvas, leaf) => [outlined(srcCanvas, leaf, 1.4, 512, 0.35), outlined(srcCanvas, leaf, 3.0, 768, 1.0)];

  // ------------------------------------------------------------------ drawing (draw.py)
  const COLORS = { FRESH: [60, 230, 60], black_spot: [255, 40, 40], insects_infected: [40, 140, 255], downy_mildew: [255, 220, 0],
    UNKNOWN: [255, 0, 255], HOLD: [150, 150, 150], HOLD_SHAPE: [90, 90, 90] };

  function drawResult(srcCanvas, res, title) {
    const W = srcCanvas.width, H = srcCanvas.height;
    const c = document.createElement("canvas");
    c.width = W; c.height = H;
    const g = c.getContext("2d");
    g.drawImage(srcCanvas, 0, 0);
    g.fillStyle = "rgba(0,0,0,0.2)";
    g.fillRect(0, 0, W, H);
    g.lineJoin = "round";
    for (const l of res.leaves) {
      const key = l.state === "ANOMALY" ? l.anomaly_class : l.state;
      g.strokeStyle = `rgb(${COLORS[key]})`;
      g.lineWidth = l.state === "ANOMALY" || l.state === "UNKNOWN" ? 5 : 2;
      g.beginPath();
      l.polygon.forEach(([x, y], i) => (i ? g.lineTo(x + 0.5, y + 0.5) : g.moveTo(x + 0.5, y + 0.5)));
      g.closePath();
      g.stroke();
    }
    g.font = "bold 22px sans-serif";
    g.textBaseline = "top";
    g.lineWidth = 4;
    for (const l of res.leaves) {
      if (l.state !== "ANOMALY" && l.state !== "UNKNOWN") continue;
      const t = `${l.id} ${l.anomaly_class || "UNKNOWN"}`, x = l.box[0], y = Math.max(0, l.box[1] - 26);
      g.strokeStyle = "#000"; g.strokeText(t, x, y);
      g.fillStyle = `rgb(${COLORS[l.anomaly_class || "UNKNOWN"]})`; g.fillText(t, x, y);
    }
    const items = Object.entries(res.counts).sort();
    g.fillStyle = "#000";
    g.fillRect(8, 8, 632, 48 + 32 * items.length);
    g.fillStyle = "#fff";
    g.font = "bold 28px sans-serif";
    g.fillText(`${title} leaves ${res.leaves.length} | L ${Math.round(res.L)}px`, 18, 14);
    g.font = "22px sans-serif";
    items.forEach(([k, v], i) => {
      g.strokeStyle = `rgb(${COLORS[k.split(":").pop()]})`; g.lineWidth = 4;
      g.strokeRect(18, 58 + 32 * i, 30, 20);
      g.fillStyle = "#fff"; g.fillText(`${k} ${v}`, 60, 56 + 32 * i);
    });
    return c;
  }

  global.leafdx = {
    LeafPipeline, LeafSegmenter, LeafClassifier, loadModel, unpackWeights, grayWorld, toCanvas, pilResizeRGBA, letterbox,
    nms, retinaMask, VLM, drawResult, COLORS, pyRound,
  };
})(typeof window !== "undefined" ? window : globalThis);
