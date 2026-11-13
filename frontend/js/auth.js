/* Login & registration pages. */
(() => {
  if (API.getToken()) { location.href = "dashboard.html"; return; }

  const errorBox = document.getElementById("formError");
  const showError = (msg) => { errorBox.textContent = msg; errorBox.classList.remove("hidden"); };
  const hideError = () => errorBox.classList.add("hidden");

  async function submit(form, path, payload) {
    hideError();
    const btn = form.querySelector("button[type=submit]");
    const label = btn.textContent;
    btn.disabled = true;
    btn.textContent = "Please wait…";
    try {
      const data = await API.request(path, { method: "POST", json: payload, auth: false });
      API.setSession(data.access_token, data.user);
      location.href = "dashboard.html";
    } catch (err) {
      showError(err.message);
    } finally {
      btn.disabled = false;
      btn.textContent = label;
    }
  }

  const loginForm = document.getElementById("loginForm");
  if (loginForm) {
    loginForm.addEventListener("submit", (e) => {
      e.preventDefault();
      const email = loginForm.email.value.trim();
      const password = loginForm.password.value;
      if (!email || !password) return showError("Please enter your email and password.");
      submit(loginForm, "/api/auth/login", { email, password });
    });
  }

  const registerForm = document.getElementById("registerForm");
  if (registerForm) {
    registerForm.addEventListener("submit", (e) => {
      e.preventDefault();
      const f = registerForm;
      const full_name = f.full_name.value.trim();
      const email = f.email.value.trim();
      const password = f.password.value;
      if (full_name.length < 2) return showError("Please enter your full name.");
      if (!/^\S+@\S+\.\S+$/.test(email)) return showError("Please enter a valid email address.");
      if (password.length < 8) return showError("Password must be at least 8 characters.");
      if (password !== f.confirm.value) return showError("Passwords do not match.");
      submit(f, "/api/auth/register", { full_name, email, password });
    });
  }
})();
