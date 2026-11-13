/* Detection history table with filters, pagination and detail modal. */
(() => {
  if (!requireAuth()) return;

  const PAGE = 15;
  let offset = 0;
  let total = 0;

  const body = document.getElementById("historyBody");
  const modality = document.getElementById("filterModality");
  const prediction = document.getElementById("filterPrediction");
  const info = document.getElementById("pagerInfo");
  const prev = document.getElementById("prevPage");
  const next = document.getElementById("nextPage");
  const modal = document.getElementById("detailModal");
  const detailBody = document.getElementById("detailBody");

  async function load() {
    const params = new URLSearchParams({ limit: PAGE, offset });
    if (modality.value) params.set("modality", modality.value);
    if (prediction.value) params.set("prediction", prediction.value);
    body.innerHTML = `<tr><td colspan="8" class="muted">Loading…</td></tr>`;
    try {
      const page = await API.request(`/api/history?${params}`);
      total = page.total;
      if (!page.items.length) {
        body.innerHTML = `<tr><td colspan="8" class="muted" style="text-align:center;padding:2rem">No detections yet. <a href="dashboard.html">Run your first analysis →</a></td></tr>`;
      } else {
        body.innerHTML = page.items.map((h) => `
          <tr data-id="${h.id}" style="cursor:pointer">
            <td class="muted">${h.id}</td>
            <td>${formatDate(h.created_at)}</td>
            <td><span class="pill mod">${escapeHtml(h.modality)}</span></td>
            <td class="summary-cell" title="${escapeHtml(h.input_summary)}">${escapeHtml(h.input_summary)}</td>
            <td><span class="pill ${h.prediction}">${h.prediction}</span></td>
            <td><div class="mini-bar"><div class="bar"><span style="width:${h.confidence_score}%;background:${h.prediction === "FAKE" ? "var(--fake)" : "var(--real)"}"></span></div><span class="mono">${h.confidence_score.toFixed(1)}%</span></div></td>
            <td class="mono muted">${formatMs(h.inference_time_ms)}</td>
            <td><button class="btn btn-danger btn-sm" data-del="${h.id}" title="Delete">✕</button></td>
          </tr>`).join("");
      }
      const end = Math.min(offset + PAGE, total);
      info.textContent = total ? `Showing ${offset + 1}–${end} of ${total}` : "";
      prev.disabled = offset === 0;
      next.disabled = end >= total;
    } catch (err) {
      body.innerHTML = `<tr><td colspan="8" class="muted">${escapeHtml(err.message)}</td></tr>`;
    }
  }

  body.addEventListener("click", async (e) => {
    const del = e.target.closest("[data-del]");
    if (del) {
      e.stopPropagation();
      if (!confirm("Delete this detection record?")) return;
      try {
        await API.request(`/api/history/${del.dataset.del}`, { method: "DELETE" });
        toast("Record deleted");
        if (offset > 0 && offset >= total - 1) offset -= PAGE;
        load();
      } catch (err) { toast(err.message, true); }
      return;
    }
    const row = e.target.closest("tr[data-id]");
    if (!row) return;
    try {
      const item = await API.request(`/api/history/${row.dataset.id}`);
      detailBody.innerHTML = `<p class="muted" style="margin-top:0">${escapeHtml(item.input_summary)}<br/><small>${formatDate(item.created_at)}</small></p><div id="detailResult"></div>`;
      renderResult(document.getElementById("detailResult"), item);
      modal.classList.add("show");
    } catch (err) { toast(err.message, true); }
  });

  document.getElementById("closeModal").addEventListener("click", () => modal.classList.remove("show"));
  modal.addEventListener("click", (e) => { if (e.target === modal) modal.classList.remove("show"); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") modal.classList.remove("show"); });

  [modality, prediction].forEach((el) => el.addEventListener("change", () => { offset = 0; load(); }));
  prev.addEventListener("click", () => { offset = Math.max(0, offset - PAGE); load(); });
  next.addEventListener("click", () => { offset += PAGE; load(); });

  load();
})();
