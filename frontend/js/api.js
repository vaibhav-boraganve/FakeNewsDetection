/* Shared API client, session handling and small UI helpers. */
const API = (() => {
  const TOKEN_KEY = "fg_token";
  const USER_KEY = "fg_user";

  const getToken = () => localStorage.getItem(TOKEN_KEY);
  const getUser = () => {
    try { return JSON.parse(localStorage.getItem(USER_KEY) || "null"); } catch { return null; }
  };
  const setSession = (token, user) => {
    localStorage.setItem(TOKEN_KEY, token);
    localStorage.setItem(USER_KEY, JSON.stringify(user));
  };
  const clearSession = () => {
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(USER_KEY);
  };

  function errorMessage(body, status) {
    if (!body) return `Request failed (${status})`;
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail)) {
      return body.detail.map((d) => `${(d.loc || []).slice(-1)[0] || "field"}: ${d.msg}`).join("; ");
    }
    return `Request failed (${status})`;
  }

  async function request(path, { method = "GET", json, form, auth = true } = {}) {
    const headers = {};
    let body;
    if (json !== undefined) { headers["Content-Type"] = "application/json"; body = JSON.stringify(json); }
    if (form) body = form;
    if (auth && getToken()) headers.Authorization = `Bearer ${getToken()}`;

    let res;
    try {
      res = await fetch(path, { method, headers, body });
    } catch {
      throw new Error("Cannot reach the server. Is the backend running?");
    }
    if (res.status === 401 && auth) {
      clearSession();
      if (!location.pathname.endsWith("index.html")) location.href = "index.html";
      throw new Error("Session expired, please sign in again");
    }
    if (res.status === 204) return null;
    const data = await res.json().catch(() => null);
    if (!res.ok) throw new Error(errorMessage(data, res.status));
    return data;
  }

  return { request, getToken, getUser, setSession, clearSession };
})();

function requireAuth() {
  if (!API.getToken()) { location.href = "index.html"; return null; }
  const user = API.getUser();
  const nameEl = document.getElementById("userName");
  const avEl = document.getElementById("userAvatar");
  if (user && nameEl) nameEl.textContent = user.full_name;
  if (user && avEl) avEl.textContent = user.full_name.trim().charAt(0).toUpperCase();
  const logout = document.getElementById("logoutBtn");
  if (logout) logout.addEventListener("click", () => { API.clearSession(); location.href = "index.html"; });
  return user;
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function formatDate(iso) {
  const d = new Date(iso.endsWith("Z") || iso.includes("+") ? iso : iso + "Z");
  return d.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

function formatMs(ms) {
  return ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${Math.round(ms)} ms`;
}

function formatBytes(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

let toastTimer;
function toast(msg, isError = false) {
  const el = document.getElementById("toast");
  if (!el) return alert(msg);
  el.textContent = msg;
  el.classList.toggle("error", isError);
  el.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove("show"), 3500);
}
