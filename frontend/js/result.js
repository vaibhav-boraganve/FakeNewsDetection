/* Renders a detection result (verdict badge, confidence gauge, signals, model breakdown). */
const MODALITY_LABEL = { text: "Newspaper / Text", image: "Image", video: "Video" };

function gaugeSvg(score, verdict) {
  const r = 54;
  const c = 2 * Math.PI * r;
  const color = verdict === "FAKE" ? "var(--fake)" : "var(--real)";
  return `
    <div class="gauge" aria-label="Confidence ${score}%">
      <svg width="132" height="132" viewBox="0 0 132 132">
        <circle class="track" cx="66" cy="66" r="${r}" fill="none" stroke-width="12"></circle>
        <circle class="arc" cx="66" cy="66" r="${r}" fill="none" stroke="${color}" stroke-width="12"
          stroke-dasharray="${c}" stroke-dashoffset="${c}" data-target="${c * (1 - score / 100)}"></circle>
      </svg>
      <div class="label"><div><b>${score.toFixed(1)}%</b><small>confidence</small></div></div>
    </div>`;
}

function componentBars(components) {
  return Object.entries(components || {})
    .map(([name, p]) => `
      <div class="bar-row">
        <div class="top"><span>${escapeHtml(name)}</span><span class="mono">${p.toFixed(1)}% fake</span></div>
        <div class="bar"><span style="width:${p}%;background-position:${p}% 0"></span></div>
      </div>`)
    .join("");
}

function extraDetails(r) {
  const d = r.details || {};
  let html = "";
  if (r.modality === "text" && d.top_terms) {
    const fake = (d.top_terms.fake_indicators || []).map((t) => `<span class="chip fake">${escapeHtml(t)}</span>`).join("");
    const real = (d.top_terms.real_indicators || []).map((t) => `<span class="chip real">${escapeHtml(t)}</span>`).join("");
    if (fake || real) {
      html += `<div class="section-label">Influential terms (TF-IDF)</div><div class="chips">${fake}${real}</div>`;
    }
  }
  if (r.modality === "image") {
    const m = (d.forensics && d.forensics.metadata) || {};
    html += `<div class="meta-grid" style="margin-top:1rem">
      <div class="meta-item"><div class="k">Resolution</div><div class="v">${escapeHtml(d.resolution || "-")}</div></div>
      <div class="meta-item"><div class="k">Faces detected</div><div class="v">${d.faces_detected ?? 0}</div></div>
      <div class="meta-item"><div class="k">Format</div><div class="v">${escapeHtml(m.format || "-")}</div></div>
      <div class="meta-item"><div class="k">Camera</div><div class="v">${escapeHtml(m.camera || "Unknown")}</div></div>
    </div>`;
  }
  if (r.modality === "video" && d.video) {
    const v = d.video;
    html += `<div class="meta-grid" style="margin-top:1rem">
      <div class="meta-item"><div class="k">Duration</div><div class="v">${v.duration_s}s @ ${v.fps} fps</div></div>
      <div class="meta-item"><div class="k">Resolution</div><div class="v">${escapeHtml(v.resolution)}</div></div>
      <div class="meta-item"><div class="k">Key frames</div><div class="v">${v.key_frames_analyzed} / ${v.total_frames}</div></div>
      <div class="meta-item"><div class="k">Face presence</div><div class="v">${Math.round((d.temporal?.face_presence_ratio || 0) * 100)}%</div></div>
    </div>`;
  }
  return html;
}

function renderResult(container, r) {
  const signals = (r.details?.signals || []).map((s) => `<li>${escapeHtml(s)}</li>`).join("");
  container.innerHTML = `
    <div class="verdict">
      ${gaugeSvg(r.confidence_score, r.prediction)}
      <div>
        <div class="verdict-badge ${r.prediction}">${r.prediction}</div>
        <p class="muted" style="margin:.6rem 0 0;font-size:.9rem">
          The content is classified as <strong>${r.prediction}</strong> with ${r.confidence_score.toFixed(1)}% confidence.
        </p>
      </div>
    </div>
    <div class="meta-grid">
      <div class="meta-item"><div class="k">Modality</div><div class="v">${MODALITY_LABEL[r.modality] || r.modality}</div></div>
      <div class="meta-item"><div class="k">Inference time</div><div class="v">${formatMs(r.inference_time_ms)}</div></div>
      <div class="meta-item"><div class="k">Fake probability</div><div class="v">${(r.details?.fake_probability ?? (r.prediction === "FAKE" ? r.confidence_score : 100 - r.confidence_score)).toFixed(1)}%</div></div>
      <div class="meta-item"><div class="k">Analysis mode</div><div class="v" style="font-size:.85rem">${escapeHtml(r.details?.mode || (r.details?.transformer_status === "ready" ? "transformer ensemble" : "baseline ensemble"))}</div></div>
    </div>
    <div class="section-label">Model breakdown</div>
    ${componentBars(r.details?.component_fake_probabilities)}
    ${signals ? `<div class="section-label">Signals</div><ul class="signals">${signals}</ul>` : ""}
    ${extraDetails(r)}
    <details class="raw"><summary>Raw JSON response</summary><pre>${escapeHtml(JSON.stringify(r, null, 2))}</pre></details>
    <p class="disclaimer">Predictions are produced by AI models and forensic heuristics and may be wrong for novel or highly
      sophisticated content. Use them to support, not replace, expert verification and trusted fact-checking.</p>`;

  requestAnimationFrame(() => {
    const arc = container.querySelector(".arc");
    if (arc) arc.style.strokeDashoffset = arc.dataset.target;
  });
}
