/* Rose leaf anomaly check, browser UI. Logic lives in leafdx.js; this file loads libraries and models, runs photos
 * one after another, renders results, runs the optional Claude check and builds the results zip. */
{
  const $ = (s) => document.querySelector(s);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  // ---------------------------------------------------------------- settings (package/config.json, models/states.json)
  const CFG = { conf: 0.35, tile_mult: 8, imgsz: 640, est_imgsz: 1280, min_area: 400, dedup_iou: 0.5, canvas_ratio: 0.6,
    white_balance: true, min_solidity: 0.8, cls_batch: 16 };
  const STATES = { thresholds: { s_min: 0, d_max: 1.01, L_min: 0, t_anom: 0.5, t_fresh: 0.7 } };
  const SEG = "leaf_seg_v8r2";
  const CLS = [
    { stem: "leaf_cls_yolov8m", name: "YOLOv8m-cls (첫 번째 모델, 기본값)", acc: 0.957, mb: 31.6 },
    { stem: "leaf_cls_yolov8m_du", name: "YOLOv8m-cls (du)", acc: 0.95, mb: 31.6 },
    { stem: "leaf_cls_yolo26m_du", name: "YOLO26m-cls (du)", acc: 0.948, mb: 20.7 },
  ];
  const SEG_MB = 20.8;
  const MODEL_BASE = "models/";
  const SAMPLE = "samples/fix_eafda190.jpg";
  const LIBS = {
    ort: "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/",
    cv: "https://cdn.jsdelivr.net/npm/@techstark/opencv-js@5.0.0-release.1/dist/opencv.js",
    zip: "https://cdn.jsdelivr.net/npm/jszip@3.10.2/dist/jszip.min.js",
  };
  const COLORS = { FRESH: "var(--fresh)", black_spot: "var(--black_spot)", insects_infected: "var(--insects_infected)", downy_mildew: "var(--downy_mildew)",
    UNKNOWN: "var(--unknown)", HOLD: "var(--hold)", HOLD_SHAPE: "var(--hold_shape)", healthy: "var(--fresh)" };
  const VERDICT_KO = { AGREE: "일치", FALSE_ALARM: "오경보 가능성", MISSED: "놓친 이상 가능성", OTHER_CLASS: "이상 종류 다름",
    VLM_HEALTHY: "의심 → VLM 건강", VLM_ANOMALY: "의심 → VLM 이상", NOT_A_LEAF: "잎이 아님", UNCLEAR: "판단 불가" };
  const isFile = location.protocol === "file:";

  // ---------------------------------------------------------------- state
  let files = [];
  let photos = [];          // {name, stem, canvas, rgb, res, json, visUrl, vlm}
  let current = 0;
  let stopFlag = false;
  let picked = null;        // Map(name -> File) when the models folder was picked by hand
  const bytesCache = new Map();
  const sessions = new Map(); // key `${stem}|${engine}` -> session
  let engineUsed = "";
  let sampleFn = null, downloadsCap = null;
  let vlmRun = null;        // {backend, model, effort, tokens, cost, counts, errors}

  // ---------------------------------------------------------------- status helpers
  function status(el, text, frac, err) {
    const box = $(el);
    box.classList.add("show");
    box.classList.toggle("err", !!err);
    box.querySelector("span").textContent = text;
    const bar = box.querySelector(".bar");
    bar.classList.toggle("hidden", frac === null || !!err);
    bar.classList.toggle("indet", frac === undefined);
    bar.querySelector("i").style.width = frac === undefined || frac === null ? "" : Math.round(frac * 100) + "%";
  }
  const tick = () => new Promise((r) => setTimeout(r, 0));

  // ---------------------------------------------------------------- libraries
  function loadScript(src) {
    return new Promise((res, rej) => {
      const s = document.createElement("script");
      s.src = src; s.async = false;
      s.onload = () => res(src);
      s.onerror = () => { s.remove(); rej(new Error("load failed: " + src)); };
      document.head.append(s);
    });
  }
  async function loadOne(localPath, cdn, ready) {
    if (!isFile) { try { await loadScript(localPath); if (ready()) return "local"; } catch (e) { /* fall through to CDN */ } }
    await loadScript(cdn);
    if (!ready()) throw new Error("라이브러리를 불러오지 못했습니다: " + cdn);
    return "cdn";
  }
  let libsPromise = null;
  function loadLibs() {
    if (libsPromise) return libsPromise;
    libsPromise = (async () => {
      const where = await loadOne("lib/ort.min.js", LIBS.ort + "ort.min.js", () => !!window.ort);
      ort.env.wasm.wasmPaths = where === "local" ? new URL("lib/", location.href).href : LIBS.ort;
      ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
      await loadOne("lib/opencv.js", LIBS.cv, () => !!window.cv);
      let c = window.cv;
      if (c instanceof Promise) c = await c;
      else if (!c.Mat) await new Promise((r) => { c.onRuntimeInitialized = r; });
      window.cv = c;
      loadOne("lib/jszip.min.js", LIBS.zip, () => !!window.JSZip).catch(() => {});
    })();
    libsPromise.catch(() => { libsPromise = null; });
    return libsPromise;
  }

  // ---------------------------------------------------------------- models
  async function fetchBytes(name) {
    if (bytesCache.has(name)) return bytesCache.get(name);
    let b;
    if (picked) {
      const f = picked.get(name);
      if (!f) throw new Error(`고른 폴더에 ${name} 파일이 없습니다`);
      b = new Uint8Array(await f.arrayBuffer());
    } else {
      const r = await fetch(MODEL_BASE + name);
      if (!r.ok) throw new Error(`모델 파일을 받지 못했습니다: ${name} (${r.status})`);
      b = new Uint8Array(await r.arrayBuffer());
    }
    bytesCache.set(name, b);
    return b;
  }

  function engineChoice() { return document.querySelector("input[name=engine]:checked").value; }

  async function getSession(stem, mb, onMB) {
    const want = engineChoice();
    const tryOrder = want === "wasm" ? ["wasm"] : want === "webgpu" ? ["webgpu"] : navigator.gpu ? ["webgpu", "wasm"] : ["wasm"];
    let lastErr = null;
    for (const ep of tryOrder) {
      const key = `${stem}|${ep}`;
      if (sessions.has(key)) return { s: sessions.get(key), ep };
      try {
        const s = await leafdx.loadModel(stem, fetchBytes, { executionProviders: [ep], graphOptimizationLevel: "all" }, onMB);
        sessions.set(key, s);
        return { s, ep };
      } catch (e) {
        lastErr = e;
        if (/모델 파일|폴더/.test(String(e.message))) throw e;
      }
    }
    throw lastErr;
  }

  async function prepareModels(clsStem) {
    await loadLibs();
    const cls = CLS.find((c) => c.stem === clsStem);
    const total = SEG_MB + cls.mb;
    let done = 0;
    const show = () => { $("#loadInfo").textContent = `모델 받는 중 ${done.toFixed(0)} / ${total.toFixed(0)} MB (처음 한 번만, 이후에는 브라우저가 기억합니다)`; };
    const onMB = (n) => { done += n / 1048576; show(); };
    show();
    const seg = await getSession(SEG, SEG_MB, onMB);
    const c = await getSession(clsStem, cls.mb, onMB);
    engineUsed = seg.ep === "webgpu" ? "GPU (WebGPU)" : "CPU (WASM)";
    $("#loadInfo").textContent = `모델 준비됨 · 계산 장치: ${engineUsed}` + (seg.ep === "wasm" && !self.crossOriginIsolated ? " · CPU 단일 스레드라 사진 1장에 1~2분 걸립니다" : "");
    return new leafdx.LeafPipeline(seg.s, c.s, STATES, CFG);
  }

  $("#pickBtn").onclick = () => $("#pickInput").click();
  $("#pickInput").onchange = () => {
    const m = new Map();
    for (const f of $("#pickInput").files) m.set(f.name, f);
    if (!m.has(`${SEG}.pack.json`)) { $("#loadInfo").textContent = "고른 폴더에 모델 파일(leaf_seg_v8r2.pack.json 등)이 없습니다."; return; }
    picked = m;
    bytesCache.clear();
    $("#pickModels").classList.add("hidden");
    $("#loadInfo").textContent = `모델 폴더를 골랐습니다 (${m.size}개 파일). 분석을 시작하세요.`;
  };

  // ---------------------------------------------------------------- photos in
  const drop = $("#drop"), input = $("#file");
  drop.onclick = () => input.click();
  drop.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } };
  drop.ondragover = (e) => { e.preventDefault(); drop.classList.add("over"); };
  drop.ondragleave = () => drop.classList.remove("over");
  drop.ondrop = (e) => { e.preventDefault(); drop.classList.remove("over"); addFiles(e.dataTransfer.files); };
  input.onchange = () => { addFiles(input.files); input.value = ""; };
  function addFiles(list) {
    for (const f of list) if (/\.(jpe?g|png|bmp|webp|jfif)$/i.test(f.name) && !files.some((x) => x.name === f.name && x.size === f.size)) files.push(f);
    renderThumbs();
  }
  function renderThumbs() {
    $("#thumbs").innerHTML = files.map((f, i) => `<figure><img src="${URL.createObjectURL(f)}" alt=""><figcaption>${esc(f.name)}</figcaption><button data-i="${i}" title="빼기" aria-label="빼기">×</button></figure>`).join("");
    $("#thumbs").querySelectorAll("button").forEach((b) => { b.onclick = () => { files.splice(+b.dataset.i, 1); renderThumbs(); }; });
    $("#run").disabled = !files.length;
  }

  async function decode(blob) {
    const bmp = await createImageBitmap(blob, { imageOrientation: "from-image", colorSpaceConversion: "none", premultiplyAlpha: "none" });
    const c = document.createElement("canvas");
    c.width = bmp.width; c.height = bmp.height;
    const g = c.getContext("2d", { willReadFrequently: true });
    g.drawImage(bmp, 0, 0);
    bmp.close && bmp.close();
    const d = g.getImageData(0, 0, c.width, c.height).data, n = c.width * c.height, rgb = new Uint8ClampedArray(3 * n);
    for (let i = 0, j = 0; i < n; i++, j += 3) { rgb[j] = d[4 * i]; rgb[j + 1] = d[4 * i + 1]; rgb[j + 2] = d[4 * i + 2]; }
    return { canvas: c, rgb: { data: rgb, width: c.width, height: c.height } };
  }

  function safeStem(name, used) {
    let stem = name.replace(/\.[^.]+$/, "").replace(/[^\p{L}\p{N}_.-]/gu, "_").slice(0, 60) || "photo";
    let s = stem, i = 1;
    while (used.has(s.toLowerCase())) s = `${stem}_${++i}`;
    used.add(s.toLowerCase());
    return s;
  }

  // ---------------------------------------------------------------- analysis
  $("#run").onclick = () => start(files.slice());
  $("#sample").onclick = async () => {
    try {
      const r = await fetch(SAMPLE);
      if (!r.ok) throw new Error();
      start([new File([await r.blob()], "fix_eafda190.jpg", { type: "image/jpeg" })]);
    } catch (e) { status("#status", "샘플 사진을 불러오지 못했습니다.", null, true); }
  };
  $("#stop").onclick = () => { stopFlag = true; $("#stop").disabled = true; };

  function busy(b) {
    $("#run").disabled = b || !files.length; $("#sample").disabled = b;
    $("#stop").classList.toggle("hidden", !b); $("#stop").disabled = false;
    document.querySelectorAll("input[name=cls], input[name=engine]").forEach((x) => { x.disabled = b; });
  }

  async function start(list) {
    if (!list.length) return;
    stopFlag = false;
    busy(true);
    const clsStem = document.querySelector("input[name=cls]:checked").value;
    photos = []; vlmRun = null;
    $("#results").classList.add("hidden"); $("#vlmCard").classList.add("hidden"); $("#vlmView").innerHTML = ""; $("#plan").innerHTML = "";
    $("#vstatus").classList.remove("show");
    let pl;
    try {
      status("#status", "라이브러리와 모델을 준비하는 중", undefined);
      pl = await prepareModels(clsStem);
    } catch (e) {
      busy(false);
      status("#status", "모델을 준비하지 못했습니다: " + e.message, null, true);
      if (!picked) $("#pickModels").classList.remove("hidden");
      return;
    }
    const used = new Set();
    try {
      for (let i = 0; i < list.length; i++) {
        const f = list[i], stem = safeStem(f.name, used);
        const label = list.length > 1 ? `사진 ${i + 1}/${list.length} (${f.name})` : f.name;
        status("#status", `${label}: 사진 읽는 중`, i / list.length);
        await tick();
        let dec;
        try { dec = await decode(f); } catch (e) { throw new Error(`${f.name}: 이미지를 열 수 없습니다`); }
        const prog = async (stage, done, total) => {
          if (stopFlag) throw new Error("STOP");
          const txt = { "leaf length": "잎 크기 추정", search: `잎 찾는 중 (타일 ${done + 1}/${total})`, classify: `잎 상태 판정 중 (${done}/${total})`, done: "정리 중" }[stage];
          const within = stage === "search" ? 0.1 + 0.6 * (done / total) : stage === "classify" ? 0.7 + 0.3 * (done / total) : stage === "done" ? 1 : 0.05;
          status("#status", `${label}: ${txt}`, (i + within) / list.length);
        };
        const res = await pl.run(dec.rgb, prog);
        const json = Object.assign({ photo: f.name, cls_model: clsStem + ".pt" }, leafdx.LeafPipeline.toJSON(res));
        const vis = leafdx.drawResult(dec.canvas, res, `${stem} [${clsStem}]`);
        const visBlob = await new Promise((r) => vis.toBlob(r, "image/jpeg", 0.9));
        photos.push({ name: f.name, stem, canvas: dec.canvas, res, json, visBlob, visUrl: URL.createObjectURL(visBlob), vlm: null, engine: engineUsed });
      }
      const tot = photos.reduce((a, p) => a + p.res.leaves.length, 0);
      status("#status", `분석 완료: 사진 ${photos.length}장, 잎 ${tot}개 (${clsStem}, ${engineUsed})`, null);
    } catch (e) {
      if (e.message === "STOP") status("#status", `멈췄습니다. 끝난 사진 ${photos.length}장의 결과만 보여 줍니다.`, null, !photos.length);
      else status("#status", "분석 실패: " + e.message, null, true);
    }
    busy(false);
    if (photos.length) showResults();
  }

  // ---------------------------------------------------------------- results
  function chips(counts) {
    return Object.entries(counts).sort().map(([k, n]) => `<span class="chip"><i style="background:${COLORS[k.split(":").pop()] || "#999"}"></i>${esc(k)} <b>${n}</b></span>`).join("");
  }
  const leafLabel = (l) => (l.state === "ANOMALY" ? l.anomaly_class : l.state);
  const maxAnom = (l) => Math.max(...Object.entries(l.probs).filter(([k]) => k !== "healthy").map(([, v]) => v));
  function probs(p) {
    return `<div class="probs">${Object.entries(p).map(([k, v]) => `<div><span>${k}</span><s><u style="width:${v * 100}%;background:${COLORS[k]}"></u></s><em>${v.toFixed(2)}</em></div>`).join("")}</div>`;
  }

  const viewCache = new Map();
  function leafViews(pi, leaf) {
    const key = `${pi}|${leaf.id}`;
    if (!viewCache.has(key)) {
      const [close, ctx] = leafdx.VLM.views(photos[pi].canvas, leaf);
      viewCache.set(key, { close, ctx, closeUrl: close.toDataURL("image/jpeg", 0.88), ctxUrl: null });
    }
    const v = viewCache.get(key);
    if (!v.ctxUrl) v.ctxUrl = v.ctx.toDataURL("image/jpeg", 0.88);
    return v;
  }
  function thumb(pi, leaf) {
    const c = document.createElement("canvas"), src = photos[pi].canvas;
    const [x0, y0, x1, y1] = leaf.box, s = Math.max(x1 - x0, y1 - y0) * 0.7, cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
    c.width = c.height = 96;
    const g = c.getContext("2d");
    g.fillStyle = "#222"; g.fillRect(0, 0, 96, 96);
    const a = Math.max(0, cx - s), b = Math.max(0, cy - s), w = Math.min(src.width, cx + s) - a, h = Math.min(src.height, cy + s) - b, k = 96 / (2 * s);
    g.drawImage(src, a, b, w, h, (a - (cx - s)) * k, (b - (cy - s)) * k, w * k, h * k);
    g.strokeStyle = "cyan"; g.lineWidth = 1.5; g.beginPath();
    leaf.polygon.forEach(([x, y], i) => { const px = (x - (cx - s)) * k, py = (y - (cy - s)) * k; i ? g.lineTo(px, py) : g.moveTo(px, py); });
    g.closePath(); g.stroke();
    return c.toDataURL("image/jpeg", 0.85);
  }

  function showResults() {
    $("#results").classList.remove("hidden"); $("#vlmCard").classList.remove("hidden");
    $("#tabs").innerHTML = photos.map((p, i) => `<button data-i="${i}">${esc(p.name)}</button>`).join("");
    $("#tabs").querySelectorAll("button").forEach((b) => { b.onclick = () => { current = +b.dataset.i; renderPhoto(); }; });
    $("#tabs").classList.toggle("hidden", photos.length < 2);
    current = 0; viewCache.clear();
    renderPhoto();
  }

  function renderPhoto() {
    $("#tabs").querySelectorAll("button").forEach((b, i) => b.classList.toggle("on", i === current));
    const p = photos[current], r = p.res, i = current;
    const flagged = r.leaves.filter((l) => l.state === "ANOMALY" || l.state === "UNKNOWN").sort((a, b) => maxAnom(b) - maxAnom(a));
    const states = [...new Set(r.leaves.map(leafLabel))].sort();
    $("#photoView").innerHTML = `
      <div class="chips">${chips(r.counts)}</div>
      <div class="meta">잎 ${r.leaves.length}개 · 추정 잎 길이 ${Math.round(r.L)}px · ${r.size[0]}×${r.size[1]} · 1단계 ${r.timing.stage1_s}s / 2단계 ${r.timing.stage2_s}s · ${esc(p.engine)}</div>
      <img class="vis" src="${p.visUrl}" alt="결과 이미지" data-cap="${esc(p.name)} 결과">
      <h3 style="font-size:16px;margin:20px 0 10px">이상 또는 의심으로 판정된 잎 <span class="hint">${flagged.length}개</span></h3>
      ${flagged.length ? `<div class="leafgrid">${flagged.map((l) => `<div class="leaf"><img src="${leafViews(i, l).closeUrl}" alt="${l.id}" data-leaf="${l.id}" data-cap="${l.id} ${leafLabel(l)}">
        <div class="b"><div class="t"><span>${l.id}</span><span style="color:${COLORS[leafLabel(l)]}">${leafLabel(l)}</span></div>${probs(l.probs)}</div></div>`).join("")}</div>` : `<p class="hint">이상 또는 의심 잎 없음</p>`}
      <details class="more" id="allD"><summary>모든 잎 보기 (${r.leaves.length}개)</summary>
        <div class="row" style="margin:10px 0"><select id="filt"><option value="">전체 상태</option>${states.map((s) => `<option>${s}</option>`).join("")}</select></div>
        <div class="tablewrap"><table><thead><tr><th>잎</th><th>ID</th><th>상태</th><th>확률</th><th>면적</th><th>볼록도</th><th>박스</th></tr></thead><tbody id="allLeaves"></tbody></table></div>
      </details>`;
    const thumbs = new Map();
    const fill = () => {
      const f = $("#filt").value;
      $("#allLeaves").innerHTML = r.leaves.filter((l) => !f || leafLabel(l) === f).map((l) => {
        if (!thumbs.has(l.id)) thumbs.set(l.id, thumb(i, l));
        return `<tr><td><img src="${thumbs.get(l.id)}" data-leaf="${l.id}" data-cap="${l.id} ${leafLabel(l)}" alt=""></td>
        <td>${l.id}</td><td><span class="chip"><i style="background:${COLORS[leafLabel(l)]}"></i>${leafLabel(l)}</span></td>
        <td style="min-width:200px">${probs(l.probs)}</td><td>${l.area}</td><td>${l.solidity}${l.touches_border ? " · 가장자리" : ""}</td>
        <td style="font-family:var(--mono);font-size:12px">${l.box.join(", ")}</td></tr>`;
      }).join("");
    };
    $("#filt").onchange = fill;
    $("#allD").addEventListener("toggle", function () { if (this.open && !$("#allLeaves").innerHTML) fill(); });
    renderVlm();
  }

  // lightbox: leaf -> close-up, click again -> context, click again -> close
  document.addEventListener("click", (e) => {
    const img = e.target.closest("img.vis, img[data-leaf]");
    if (img) {
      if (img.dataset.leaf) {
        const leaf = photos[current].res.leaves.find((l) => l.id === img.dataset.leaf);
        const v = leafViews(current, leaf);
        openLb(v.closeUrl, img.dataset.cap, v.ctxUrl);
      } else openLb(img.src, img.dataset.cap || "", "");
      return;
    }
    if (e.target.closest("#lb")) {
      const lb = $("#lb");
      if (lb.dataset.next) { $("#lbImg").src = lb.dataset.next; $("#lbCap").textContent = lb.dataset.cap + " · 주변 사진 (누르면 닫힘)"; lb.dataset.next = ""; }
      else lb.classList.remove("show");
    }
  });
  function openLb(src, cap, next) {
    const lb = $("#lb");
    $("#lbImg").src = src; lb.dataset.cap = cap; lb.dataset.next = next || "";
    $("#lbCap").textContent = cap + (next ? " · 확대 사진 (누르면 주변 사진)" : " (누르면 닫힘)");
    lb.classList.add("show");
  }
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") $("#lb").classList.remove("show"); });

  // ---------------------------------------------------------------- files out
  function csv(rows, cols) {
    const q = (v) => { const s = Array.isArray(v) ? JSON.stringify(v) : String(v ?? ""); return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s; };
    return "﻿" + [cols.join(","), ...rows.map((r) => cols.map((c) => q(r[c] ?? (c in r ? r[c] : 0))).join(","))].join("\r\n") + "\r\n";
  }
  function summaryCsv() {
    const states = [...new Set(photos.flatMap((p) => Object.keys(p.res.counts)))].sort();
    const cols = ["photo", "cls_model", "leaves", "L", ...states, "stage1_s", "stage2_s"];
    const rows = photos.map((p) => Object.assign({ photo: p.name, cls_model: p.json.cls_model, leaves: p.res.leaves.length, L: p.res.L },
      Object.fromEntries(states.map((s) => [s, p.res.counts[s] || 0])), p.res.timing));
    return csv(rows, cols);
  }
  function vlmCsv() {
    const cols = ["photo", "cls_model", "id", "state", "model", "model_p", "vlm_label", "confidence", "verdict", "symptoms", "reason", "error", "box"];
    return csv(photos.flatMap((p) => (p.vlm ? p.vlm.rows.map((r) => Object.assign({ photo: p.name, cls_model: p.json.cls_model }, r)) : [])), cols);
  }
  async function save(name, blob) {
    if (downloadsCap) {
      try { await downloadsCap.save({ filename: name, data: blob }); }
      catch (e) { if (e && e.code !== "declined") status("#status", "파일을 저장하지 못했습니다: " + (e.message || e.code), null, true); }
      return;
    }
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob); a.download = name;
    document.body.append(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 4000);
  }
  $("#csv").onclick = () => save("summary.csv", new Blob([summaryCsv()], { type: "text/csv" }));
  $("#zip").onclick = async () => {
    if (!window.JSZip) { try { await loadOne("lib/jszip.min.js", LIBS.zip, () => !!window.JSZip); } catch (e) { status("#status", "zip 라이브러리를 불러오지 못했습니다", null, true); return; } }
    const z = new JSZip();
    for (const p of photos) {
      z.file(`${p.stem}.json`, JSON.stringify(p.json));
      z.file(`${p.stem}_vis.jpg`, p.visBlob);
      if (p.vlm) {
        z.file(`${p.stem}_vlm.json`, JSON.stringify({ photo: p.name, backend: vlmRun.backend, rows: p.vlm.rows }, null, 1));
        if (p.vlm.sheet) z.file(`${p.stem}_vlm.jpg`, p.vlm.sheet);
      }
    }
    z.file("summary.csv", summaryCsv());
    if (photos.some((p) => p.vlm)) z.file("vlm_review.csv", vlmCsv());
    save("results.zip", await z.generateAsync({ type: "blob" }));
  };

  // ---------------------------------------------------------------- VLM check
  function backends() {
    const o = [];
    if (sampleFn) o.push(["claudeai", "Claude (지금 로그인한 claude.ai 계정, API 키 필요 없음)"]);
    if (!window.claude) o.push(["api", "Claude API 키 (Anthropic에 직접 요청, 사용량만큼 요금)"]);
    $("#backend").innerHTML = o.map(([v, t]) => `<option value="${v}">${t}</option>`).join("");
    if (!o.length) $("#backend").innerHTML = `<option value="">이 화면에서는 VLM 검증을 쓸 수 없습니다</option>`;
    syncBackend();
  }
  function syncBackend() {
    const b = $("#backend").value;
    document.querySelectorAll(".keyrow").forEach((x) => x.classList.toggle("hidden", b !== "api"));
    $("#claudeModel").innerHTML = (b === "claudeai" ? ["자동 (기본 모델)"] : []).concat(["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"]).map((m) => `<option>${m}</option>`).join("");
    $("#vlmNote").textContent = b === "claudeai"
      ? "사진은 이 페이지를 보고 있는 사람의 Claude 계정으로 보내지고, 그 계정의 사용량을 씁니다. 처음 한 번 허용을 묻습니다."
      : b === "api" ? "잎 하나에 요청 한 번이 Anthropic API로 바로 나가고, 쓴 만큼 요금이 나갑니다. 키는 이 페이지 메모리에만 있습니다."
      : "";
    $("#verify").disabled = true; $("#plan").innerHTML = "";
    $("#dry").disabled = !b;
  }
  $("#backend").onchange = syncBackend;
  ["#apikey", "#fresh", "#max", "#claudeModel"].forEach((s) => $(s).addEventListener("input", () => { $("#verify").disabled = true; }));

  $("#keyFileBtn").onclick = (e) => { e.preventDefault(); $("#keyFileInput").click(); };
  $("#keyFileInput").onchange = async () => {
    const f = $("#keyFileInput").files[0];
    $("#keyFileInput").value = "";
    if (!f) return;
    const text = (await f.text()).replace(/^﻿/, "");
    let key = "";
    for (let line of text.split(/\r?\n/)) {
      line = line.trim();
      if (!line || line.startsWith("#")) continue;
      if (line.includes("=")) line = line.split("=").slice(1).join("=").trim();
      key = line.replace(/^['"]+|['"]+$/g, "");
      break;
    }
    if (!key) { $("#keyinfo").textContent = "키 파일이 비어 있습니다: " + f.name; return; }
    $("#apikey").value = key;
    $("#keyinfo").textContent = `키 파일을 읽었습니다: ${f.name} (${key.slice(0, 7)}...${key.slice(-4)}, ${key.length}자)` + (key.startsWith("sk-ant-") ? "" : " · 주의: Anthropic API 키는 보통 sk-ant- 로 시작합니다");
  };

  function plan() {
    const fresh = Math.max(0, Math.min(50, +$("#fresh").value || 0)), max = Math.max(1, Math.min(200, +$("#max").value || 40));
    return photos.map((p, i) => ({ i, leaves: leafdx.VLM.select(p.res.leaves, ["ANOMALY", "UNKNOWN"], fresh, max, 0) }));
  }
  $("#dry").onclick = () => {
    const b = $("#backend").value;
    if (b === "api" && !$("#apikey").value.trim()) { $("#plan").innerHTML = `<div class="status show err">API 키를 넣어 주세요</div>`; return; }
    const pl = plan(), total = pl.reduce((a, x) => a + x.leaves.length, 0);
    const key = $("#apikey").value.trim();
    $("#plan").innerHTML = `<div class="note"><b>검증할 잎 ${total}개</b> · ${b === "api" ? `API ${esc($("#claudeModel").value)} · 키 ${esc(key.slice(0, 7))}...${esc(key.slice(-4))}` : `claude.ai 계정 · ${esc($("#claudeModel").value)}`}<br>
      ${pl.map((x) => `${esc(photos[x.i].name)}: ${x.leaves.length}개`).join(" · ")}</div>`;
    $("#verify").disabled = !total;
  };

  const b64 = (blob) => new Promise((r) => { const fr = new FileReader(); fr.onload = () => r(String(fr.result).split(",")[1]); fr.readAsDataURL(blob); });
  const jpeg = (c) => new Promise((r) => c.toBlob(r, "image/jpeg", 0.9));

  class Fatal extends Error {}

  async function askApi(key, model, effort, close, ctx, signal) {
    const V = leafdx.VLM;
    const img = async (c) => ({ type: "image", source: { type: "base64", media_type: "image/jpeg", data: await b64(await jpeg(c)) } });
    let r;
    try {
      r = await fetch("https://api.anthropic.com/v1/messages", {
        method: "POST", signal,
        headers: { "x-api-key": key, "anthropic-version": "2023-06-01", "anthropic-beta": "server-side-fallback-2026-07-01",
          "anthropic-dangerous-direct-browser-access": "true", "content-type": "application/json" },
        body: JSON.stringify({ model, max_tokens: 8000, system: V.SYSTEM,
          messages: [{ role: "user", content: [await img(close), await img(ctx), { type: "text", text: V.USER }] }],
          output_config: { effort, format: { type: "json_schema", schema: V.SCHEMA } }, fallbacks: "default" }),
      });
    } catch (e) {
      if (e.name === "AbortError") throw e;
      throw new Fatal("Anthropic API에 연결하지 못했습니다. 이 페이지에서 외부 요청이 막혀 있거나 인터넷 연결 문제일 수 있습니다.");
    }
    const body = await r.json().catch(() => ({}));
    if (!r.ok) {
      const msg = body?.error?.message || r.statusText;
      if (r.status === 401) throw new Fatal("API 키가 올바르지 않습니다 (401)");
      if (r.status === 403) throw new Fatal("이 키로는 쓸 수 없습니다 (403): " + msg);
      if (r.status === 404) throw new Fatal("모델 이름을 확인하세요 (404): " + msg);
      if (/credit balance/i.test(msg)) throw new Fatal("크레딧이 부족합니다 (Console의 Billing 확인): " + msg);
      return { ans: { label: "unclear", confidence: "low", symptoms: "", reason: "", error: `${r.status}: ${msg}`.slice(0, 300) }, usage: [0, 0] };
    }
    const usage = [body.usage?.input_tokens || 0, body.usage?.output_tokens || 0];
    if (body.stop_reason === "refusal") return { ans: { label: "unclear", confidence: "low", symptoms: "", reason: "", error: "refusal" }, usage };
    if (body.stop_reason === "max_tokens") return { ans: { label: "unclear", confidence: "low", symptoms: "", reason: "", error: "max_tokens" }, usage };
    const t = (body.content || []).find((c) => c.type === "text");
    return { ans: V.parseAnswer(t ? t.text : ""), usage };
  }

  async function askClaudeAi(model, effort, close, ctx, signal) {
    const V = leafdx.VLM;
    const opts = { images: [await jpeg(close), await jpeg(ctx)], signal, effort };
    if (model && !model.startsWith("자동")) opts.model = model;
    try {
      const { text } = await sampleFn(V.SYSTEM + "\n\n" + V.USER + V.JSON_HINT, opts);
      return { ans: V.parseAnswer(text), usage: [0, 0] };
    } catch (e) {
      if (e && e.code === "cancelled") throw Object.assign(new Error("aborted"), { name: "AbortError" });
      if (e && ["not_granted", "sampling_disabled", "not_declared", "capability_disabled", "capability_removed", "images_unavailable", "session_expired"].includes(e.code))
        throw new Fatal(e.code === "not_granted" ? "Claude 사용을 허용하지 않아 멈췄습니다" : `Claude를 쓸 수 없습니다 (${e.code})`);
      if (e && e.code === "rate_limited") throw new Fatal("Claude 사용 한도에 걸렸습니다. 잠시 뒤 다시 시도하세요");
      return { ans: { label: "unclear", confidence: "low", symptoms: "", reason: "", error: String(e?.code || e?.message || e).slice(0, 300) }, usage: [0, 0] };
    }
  }

  let vctl = null;
  $("#vstop").onclick = () => vctl && vctl.abort();
  $("#verify").onclick = async () => {
    const V = leafdx.VLM, b = $("#backend").value, key = $("#apikey").value.trim(), model = $("#claudeModel").value, effort = $("#effort").value;
    const pl = plan(), total = pl.reduce((a, x) => a + x.leaves.length, 0);
    vctl = new AbortController();
    $("#verify").disabled = true; $("#dry").disabled = true; $("#vstop").classList.remove("hidden");
    photos.forEach((p) => { p.vlm = null; });
    $("#vlmView").innerHTML = "";
    vlmRun = { backend: b === "api" ? "claude" : "claude.ai", model, effort, tokens: [0, 0], cost: null, counts: {}, errors: 0 };
    let done = 0, fatal = null;
    status("#vstatus", `잎 0/${total} 검증 중`, 0);
    try {
      for (const { i, leaves } of pl) {
        const p = photos[i], rows = new Array(leaves.length), views = new Array(leaves.length);
        let next = 0;
        const worker = async () => {
          while (next < leaves.length && !fatal) {
            const k = next++, leaf = leaves[k];
            const [close, ctx] = V.views(p.canvas, leaf);
            views[k] = close;
            let out;
            try {
              out = b === "api" ? await askApi(key, model, effort, close, ctx, vctl.signal) : await askClaudeAi(model, effort, close, ctx, vctl.signal);
            } catch (e) {
              if (e instanceof Fatal) { fatal = e; vctl.abort(); return; }
              if (e.name === "AbortError") return;
              out = { ans: { label: "unclear", confidence: "low", symptoms: "", reason: "", error: String(e.message).slice(0, 300) }, usage: [0, 0] };
            }
            vlmRun.tokens[0] += out.usage[0]; vlmRun.tokens[1] += out.usage[1];
            const a = out.ans;
            rows[k] = { id: leaf.id, state: leaf.state, model: V.modelLabel(leaf) || leaf.state, model_p: Number(Math.max(...Object.values(leaf.probs)).toFixed(4)),
              vlm_label: a.label, confidence: a.confidence, verdict: V.verdict(leaf, a.label), symptoms: a.symptoms, reason: a.reason, error: a.error || "", box: leaf.box };
            done++;
            status("#vstatus", `잎 ${done}/${total} 검증 중`, done / total);
          }
        };
        await Promise.all(Array.from({ length: b === "api" ? 4 : 2 }, worker));
        if (fatal || vctl.signal.aborted) break;
        const ok = rows.filter(Boolean);
        p.vlm = { rows: ok, counts: count(ok), sheet: ok.length ? await jpeg(contactSheet(ok.map((r) => [views[rows.indexOf(r)], r]))) : null };
      }
    } finally {
      $("#vstop").classList.add("hidden"); $("#dry").disabled = false;
    }
    const all = photos.flatMap((p) => (p.vlm ? p.vlm.rows : []));
    vlmRun.counts = count(all);
    vlmRun.errors = all.filter((r) => r.error).length;
    const price = V.PRICE[model];
    if (b === "api" && price) vlmRun.cost = (vlmRun.tokens[0] * price[0] + vlmRun.tokens[1] * price[1]) / 1e6;
    if (fatal) status("#vstatus", "VLM 검증 실패: " + fatal.message, null, true);
    else if (vctl.signal.aborted) status("#vstatus", `멈췄습니다. 끝난 사진의 결과만 남겼습니다 (잎 ${done}/${total}).`, null);
    else status("#vstatus", `VLM 검증 완료 · ${vlmRun.backend} ${model}` + (b === "api" ? ` · 토큰 입력 ${vlmRun.tokens[0]} / 출력 ${vlmRun.tokens[1]}` : "") +
      (vlmRun.cost != null ? ` · 약 $${vlmRun.cost.toFixed(3)}` : "") + (vlmRun.errors ? ` · 오류 ${vlmRun.errors}건 (error 열 참고)` : ""), null);
    renderVlm();
  };
  const count = (rows) => rows.reduce((m, r) => { m[r.verdict] = (m[r.verdict] || 0) + 1; return m; }, {});

  function contactSheet(tiles, cols = 6, side = 256) {
    const VC = { AGREE: "rgb(60,200,60)", FALSE_ALARM: "rgb(255,140,0)", MISSED: "rgb(255,40,40)", OTHER_CLASS: "rgb(255,220,0)",
      VLM_HEALTHY: "rgb(60,200,60)", VLM_ANOMALY: "rgb(255,40,40)", NOT_A_LEAF: "rgb(150,150,150)", UNCLEAR: "rgb(150,150,150)" };
    const rows = Math.ceil(tiles.length / cols), c = document.createElement("canvas");
    c.width = cols * side; c.height = Math.max(1, rows) * (side + 66);
    const g = c.getContext("2d");
    g.fillStyle = "rgb(20,20,20)"; g.fillRect(0, 0, c.width, c.height);
    g.font = "15px sans-serif"; g.textBaseline = "top";
    tiles.forEach(([img, r], i) => {
      const x = (i % cols) * side, y = Math.floor(i / cols) * (side + 66), k = (side - 4) / Math.max(img.width, img.height);
      g.drawImage(img, x + 2, y + 2, img.width * k, img.height * k);
      g.fillStyle = "rgb(230,230,230)";
      g.fillText(`${r.id}  model: ${r.model}`, x + 4, y + side + 2);
      g.fillText(`VLM: ${r.vlm_label} (${r.confidence})`, x + 4, y + side + 21);
      g.fillStyle = VC[r.verdict]; g.fillText(r.verdict, x + 4, y + side + 40);
    });
    return c;
  }

  function renderVlm() {
    const p = photos[current];
    if (!vlmRun || !p) { $("#vlmView").innerHTML = ""; return; }
    const rows = p.vlm ? p.vlm.rows : [];
    const vchips = (c) => Object.entries(c || {}).map(([k, n]) => `<span class="v ${k}">${k} ${n}</span>`).join(" ");
    $("#vlmView").innerHTML = `
      <div style="margin-top:14px" class="row"><b>전체</b> ${vchips(vlmRun.counts)}</div>
      <h3 style="font-size:16px;margin:18px 0 8px">${esc(p.name)} <span class="hint">${rows.length}개 검증</span> ${vchips(p.vlm && p.vlm.counts)}</h3>
      ${rows.length ? `<div class="tablewrap"><table><thead><tr><th>잎</th><th>ID</th><th>분류 모델</th><th>VLM</th><th>비교</th><th>보인 증상 · 이유</th></tr></thead><tbody>
      ${rows.map((r) => { const leaf = p.res.leaves.find((l) => l.id === r.id); return `<tr><td><img src="${leafViews(current, leaf).closeUrl}" data-leaf="${r.id}" data-cap="${r.id} 모델 ${esc(r.model)} / VLM ${esc(r.vlm_label)}" alt=""></td>
        <td>${r.id}</td><td>${esc(r.model)}<div class="hint">${r.model_p.toFixed(2)}</div></td>
        <td>${esc(r.vlm_label)}<div class="hint">${esc(r.confidence)}</div></td>
        <td><span class="v ${r.verdict}">${r.verdict}</span><div class="hint">${VERDICT_KO[r.verdict] || ""}</div></td>
        <td>${esc(r.symptoms)}<div class="hint">${esc(r.reason)}</div>${r.error ? `<div style="color:var(--err)">${esc(r.error)}</div>` : ""}</td></tr>`; }).join("")}
      </tbody></table></div>
      <p class="row" style="margin-top:12px"><button class="btn ghost" id="sheetBtn">검증 이미지 (_vlm.jpg)</button><button class="btn ghost" id="vcsvBtn">vlm_review.csv</button></p>`
      : `<p class="hint">이 사진에서는 검증한 잎이 없습니다.</p>`}
      <details class="more"><summary>비교 결과(verdict) 설명</summary><div class="legend" style="margin-top:10px">
        ${Object.entries(VERDICT_KO).map(([k, t]) => `<span class="v ${k}">${k}</span><span>${t}</span>`).join("")}</div></details>`;
    if ($("#sheetBtn")) $("#sheetBtn").onclick = () => p.vlm.sheet && save(`${p.stem}_vlm.jpg`, p.vlm.sheet);
    if ($("#vcsvBtn")) $("#vcsvBtn").onclick = () => save("vlm_review.csv", new Blob([vlmCsv()], { type: "text/csv" }));
  }

  // ---------------------------------------------------------------- start up
  $("#models").innerHTML = CLS.map((m, i) => `<label><input type="radio" name="cls" value="${m.stem}" ${i ? "" : "checked"}>
    <span>${esc(m.name)}<code>${m.stem}.pt · ${m.mb} MB</code></span><span class="acc">val ${m.acc.toFixed(3)}</span></label>`).join("");
  $("#devinfo").textContent = navigator.gpu ? "이 브라우저는 GPU(WebGPU) 계산을 지원합니다" : "GPU(WebGPU)를 지원하지 않는 브라우저라 CPU로 계산합니다";
  if (isFile) {
    $("#sample").classList.add("hidden");
    $("#loadInfo").textContent = "파일로 직접 열었습니다. 라이브러리는 인터넷(jsdelivr)에서 받고, 모델은 분석을 시작할 때 models 폴더를 고르면 됩니다.";
    $("#pickModels").classList.remove("hidden");
  } else {
    fetch(SAMPLE, { method: "HEAD" }).then((r) => { if (!r.ok) $("#sample").classList.add("hidden"); }).catch(() => $("#sample").classList.add("hidden"));
  }
  backends();
  loadLibs().catch((e) => { $("#loadInfo").textContent = "라이브러리를 불러오지 못했습니다. 인터넷 연결을 확인하세요. (" + e.message + ")"; });
  if (window.claude && window.claude.use) {
    window.claude.use("sample").then((s) => { sampleFn = s; backends(); }).catch(() => {});
    window.claude.use("downloads").then((d) => { downloadsCap = d; }).catch(() => {});
  }
}
