/* Unified detection dashboard. */
(() => {
  if (!requireAuth()) return;

  const resultBody = document.getElementById("resultBody");
  const resultId = document.getElementById("resultId");
  const EMPTY_RESULT = `<div class="result-empty"><div class="big">🛡️</div>Run an analysis to see the verdict, confidence score and forensic signals here.</div>`;
  const clears = {};

  const SAMPLES = {
    real: {
      title: "State government announces new irrigation scheme",
      text: "The state government on Tuesday announced an irrigation scheme worth 2,400 crore rupees to expand drip irrigation across 1.5 lakh hectares, the agriculture minister said in a statement. According to officials, the scheme will be implemented over three years and farmers will receive a 55 percent subsidy on equipment. The department said applications will open next month through its online portal.",
    },
    fake: {
      title: "SHOCKING: Drinking this tea cures cancer overnight!!!",
      text: "You won't believe this! A secret herbal tea discovered by village elders CURES cancer overnight and doctors are FURIOUS. Big pharma has been hiding this miracle for decades. The mainstream media won't tell you the truth. Share this with everyone before it gets deleted!!!",
    },
  };

  // ---------- Tabs ----------
  document.querySelectorAll(".tab").forEach((tab) =>
    tab.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t === tab));
      document.querySelectorAll(".tab-panel").forEach((p) => p.classList.toggle("active", p.id === `panel-${tab.dataset.tab}`));
    })
  );

  // ---------- Result helpers ----------
  function showLoading(label) {
    resultId.textContent = "";
    resultBody.innerHTML = `<div class="loader"><div class="spinner"></div><div>${escapeHtml(label)}</div></div>`;
  }
  function showError(msg) {
    resultBody.innerHTML = `<div class="form-error">${escapeHtml(msg)}</div>`;
  }
  function resetResult() {
    resultId.textContent = "";
    resultBody.innerHTML = EMPTY_RESULT;
  }
  function showResult(r) {
    resultId.textContent = `#${r.id}`;
    renderResult(resultBody, r);
    loadStats();
    if (window.innerWidth < 1080) document.getElementById("resultCard").scrollIntoView({ behavior: "smooth" });
  }

  async function runDetection(form, label, fn) {
    const btn = form.querySelector("button[type=submit]");
    btn.disabled = true;
    showLoading(label);
    try {
      showResult(await fn());
    } catch (err) {
      showError(err.message);
      toast(err.message, true);
    } finally {
      const fileInput = form.querySelector('input[type="file"]');
      btn.disabled = fileInput ? fileInput.files.length === 0 : false;
    }
  }

  // ---------- Text ----------
  const textForm = document.getElementById("textForm");
  const textBody = document.getElementById("textBody");
  const textTitle = document.getElementById("textTitle");
  const charCount = document.getElementById("charCount");
  const updateCount = () => (charCount.textContent = `${textBody.value.length} / 20000`);
  textBody.addEventListener("input", updateCount);
  document.querySelectorAll("[data-sample]").forEach((b) =>
    b.addEventListener("click", () => {
      const s = SAMPLES[b.dataset.sample];
      textTitle.value = s.title;
      textBody.value = s.text;
      updateCount();
    })
  );
  clears.text = () => {
    textTitle.value = "";
    textBody.value = "";
    updateCount();
  };

  textForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const text = textBody.value.trim();
    if (text.length < 20) return toast("Please enter at least 20 characters of text.", true);
    runDetection(textForm, "Running NLP pipeline (NLTK → TF-IDF → Transformer)…", () =>
      API.request("/api/detect/text", { method: "POST", json: { text, title: textTitle.value.trim() || null } })
    );
  });

  // ---------- File uploads (image / video) ----------
  function setupUpload({ formId, dropId, inputId, previewId, maxMb, endpoint, label, mediaTag }) {
    const form = document.getElementById(formId);
    const drop = document.getElementById(dropId);
    const input = document.getElementById(inputId);
    const preview = document.getElementById(previewId);
    const media = preview.querySelector(mediaTag);
    const meta = preview.querySelector(".file-meta");
    const btn = form.querySelector("button[type=submit]");
    let file = null;
    let objectUrl = null;

    function choose(f) {
      if (!f) return;
      if (f.size > maxMb * 1024 * 1024) return toast(`File exceeds ${maxMb} MB limit.`, true);
      file = f;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
      objectUrl = URL.createObjectURL(f);
      media.src = objectUrl;
      meta.textContent = `${f.name} · ${formatBytes(f.size)}`;
      preview.classList.add("show");
      btn.disabled = false;
    }

    input.addEventListener("change", () => choose(input.files[0]));
    ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("drag"); }));
    ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("drag"); }));
    drop.addEventListener("drop", (e) => choose(e.dataTransfer.files[0]));

    function clear() {
      file = null;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
      objectUrl = null;
      media.removeAttribute("src");
      if (mediaTag === "video") media.load();
      meta.textContent = "";
      preview.classList.remove("show");
      input.value = "";
      btn.disabled = true;
    }

    form.addEventListener("submit", (e) => {
      e.preventDefault();
      if (!file) return;
      const fd = new FormData();
      fd.append("file", file);
      runDetection(form, label, () => API.request(endpoint, { method: "POST", form: fd }));
    });
    return clear;
  }

  clears.image = setupUpload({
    formId: "imageForm", dropId: "imageDrop", inputId: "imageInput", previewId: "imagePreview", maxMb: 10,
    endpoint: "/api/detect/image", label: "Running image forensics (OpenCV → CNN / XceptionNet → ELA & spectrum)…", mediaTag: "img",
  });
  clears.video = setupUpload({
    formId: "videoForm", dropId: "videoDrop", inputId: "videoInput", previewId: "videoPreview", maxMb: 100,
    endpoint: "/api/detect/video", label: "Extracting key frames and running CNN + LSTM temporal analysis…", mediaTag: "video",
  });

  document.getElementById("clearTab").addEventListener("click", () => {
    const active = document.querySelector(".tab.active")?.dataset.tab;
    clears[active]?.();
    resetResult();
    toast("Cleared");
  });

  // ---------- Stats / status / metrics ----------
  async function loadStats() {
    try {
      const s = await API.request("/api/history/stats");
      document.getElementById("statTotal").textContent = s.total;
      document.getElementById("statFake").textContent = s.fake;
      document.getElementById("statReal").textContent = s.real;
      document.getElementById("statText").textContent = s.by_modality.text || 0;
      document.getElementById("statImage").textContent = s.by_modality.image || 0;
      document.getElementById("statVideo").textContent = s.by_modality.video || 0;
    } catch { /* non-critical */ }
  }

  const dotClass = (v) => {
    const s = String(v);
    if (/^ready|weights|seed-trained/.test(s)) return "ok";
    if (/loading|not-loaded|no-weights|disabled/.test(s)) return "warn";
    return "err";
  };
  const STATUS_ROWS = [
    ["Text · TF-IDF baseline", (s) => s.text.baseline],
    ["Text · Transformer (BERT/RoBERTa)", (s) => s.text.transformer],
    ["Image · Forensic analysis", (s) => s.image.forensics],
    ["Image · EfficientNet-B0", (s) => s.image.efficientnet],
    ["Image · XceptionNet (faces)", (s) => s.image.xception],
    ["Video · CNN encoder", (s) => s.video.cnn_encoder],
    ["Video · CNN + LSTM", (s) => s.video.cnn_lstm],
  ];
  let statusTimer;
  async function loadStatus() {
    const list = document.getElementById("statusList");
    try {
      const s = await API.request("/api/system/status");
      list.innerHTML = STATUS_ROWS.map(([name, get]) => {
        const v = get(s);
        return `<div class="row"><span>${name}</span><span><span class="dot ${dotClass(v)}"></span>${escapeHtml(v)}</span></div>`;
      }).join("") + (s.warmup_complete ? "" : `<div class="muted" style="font-size:.8rem">Heavy models are still warming up in the background…</div>`);
      clearTimeout(statusTimer);
      if (!s.warmup_complete) statusTimer = setTimeout(loadStatus, 5000);
    } catch (err) {
      list.innerHTML = `<span class="muted">${escapeHtml(err.message)}</span>`;
    }
  }
  document.getElementById("refreshStatus").addEventListener("click", loadStatus);

  async function loadMetrics() {
    const body = document.getElementById("metricsBody");
    try {
      const rows = await API.request("/api/metrics");
      if (!rows.length) {
        body.innerHTML = `<tr><td colspan="7" class="muted">No evaluations recorded yet. Run the training scripts to populate metrics.</td></tr>`;
        return;
      }
      const pct = (v) => `${(v * 100).toFixed(1)}%`;
      body.innerHTML = rows.map((m) => `
        <tr>
          <td>${escapeHtml(m.model_name)}</td>
          <td><span class="pill mod">${escapeHtml(m.modality)}</span></td>
          <td class="muted" style="font-size:.82rem">${escapeHtml(m.dataset)} (n=${m.num_samples})</td>
          <td>${pct(m.accuracy)}</td><td>${pct(m.precision)}</td><td>${pct(m.recall)}</td><td><strong>${pct(m.f1_score)}</strong></td>
        </tr>`).join("");
    } catch (err) {
      body.innerHTML = `<tr><td colspan="7" class="muted">${escapeHtml(err.message)}</td></tr>`;
    }
  }

  loadStats();
  loadStatus();
  loadMetrics();
})();
