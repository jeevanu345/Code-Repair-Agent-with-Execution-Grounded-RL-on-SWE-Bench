/* ============================================================
   SWE-RL MODEL CONNECTIONS
============================================================ */

(() => {
  "use strict";

  const $ = (selector, root = document) =>
    root.querySelector(selector);

  const $$ = (selector, root = document) =>
    Array.from(root.querySelectorAll(selector));

  const state = {
    providers: [],
    active: null,
  };

  const escapeHtml = (value) =>
    window.SWERL?.escapeHtml
      ? window.SWERL.escapeHtml(value)
      : String(value ?? "");

  /* ==========================================================
     HELPERS
  ========================================================== */

  async function request(url, options = {}) {
    return window.SWERLAPI.request(url, options);
  }

  function toast(title, message, type) {
    window.SWERL?.showToast(
      title,
      message,
      type
    );
  }

  function normalizeProviderList(payload) {
    if (Array.isArray(payload)) {
      return {
        providers: payload,
        active: null,
      };
    }

    return {
      providers:
        payload?.providers ||
        payload?.items ||
        payload?.supported_providers ||
        [],

      active:
        payload?.active ||
        payload?.active_connection ||
        payload?.connection ||
        payload?.settings ||
        null,
    };
  }

  function providerId(provider) {
    if (typeof provider === "string") {
      return provider;
    }

    return (
      provider?.id ||
      provider?.name ||
      provider?.provider ||
      ""
    );
  }

  function providerLabel(provider) {
    if (typeof provider === "string") {
      return provider;
    }

    return (
      provider?.label ||
      provider?.display_name ||
      provider?.name ||
      provider?.id ||
      "Provider"
    );
  }

  function getSelectedProvider() {
    return $("#providerSelect").value;
  }

  function getFormPayload() {
    return {
      provider:
        $("#providerSelect")
          .value
          .trim(),

      model:
        $("#modelIdInput")
          .value
          .trim(),

      base_url:
        $("#baseUrlInput")
          .value
          .trim(),

      api_key:
        $("#apiKeyInput")
          .value,
    };
  }

  /* ==========================================================
     PROVIDERS
  ========================================================== */

  async function loadProviders() {
    const select =
      $("#providerSelect");

    try {
      const payload =
        await request("/api/providers");

      const normalized =
        normalizeProviderList(payload);

      state.providers =
        normalized.providers;

      state.active =
        normalized.active;

      renderProviders();
      applyActiveConnection();
      providerChanged(true);

    } catch (error) {
      select.innerHTML = `
        <option value="">
          Unable to load providers
        </option>
      `;

      toast(
        "Provider list unavailable",
        error.message,
        "error"
      );
    }
  }

  function renderProviders() {
    const select =
      $("#providerSelect");

    if (!state.providers.length) {
      select.innerHTML = `
        <option value="">
          No providers returned
        </option>
      `;

      return;
    }

    select.innerHTML =
      state.providers
        .map((provider) => {
          const id =
            providerId(provider);

          const label =
            providerLabel(provider);

          return `
            <option value="${escapeHtml(id)}">
              ${escapeHtml(label)}
            </option>
          `;
        })
        .join("");
  }

  function applyActiveConnection() {
    const active =
      state.active;

    if (!active) {
      setConnectionPresentation(null);
      return;
    }

    const provider =
      active.provider ||
      active.name ||
      "";

    const model =
      active.model ||
      active.model_id ||
      "";

    const baseUrl =
      active.base_url ||
      active.api_base ||
      active.endpoint ||
      "";

    if (provider) {
      $("#providerSelect").value =
        provider;
    }

    $("#modelIdInput").value =
      model;

    $("#baseUrlInput").value =
      baseUrl;

    $("#apiKeyInput").value = "";

    setConnectionPresentation({
      provider,
      model,
    });
  }

  function providerChanged(useActive = false) {
    const provider = state.providers.find(p => providerId(p) === getSelectedProvider());
    if (!provider) return;
    $("#apiKeyInput").value = "";
    $("#apiKeyInput").type = "password";
    if (!useActive) { $("#modelIdInput").value = ""; $("#baseUrlInput").value = provider.base_url || "http://127.0.0.1:8000/v1"; }
    $("#baseUrlInput").readOnly = !["vllm", "custom", "qwen"].includes(provider.id);
    $("#apiKeyInput").placeholder = state.active?.provider === provider.id && state.active?.has_key ? "Key configured; leave blank to retain" : provider.requires_key ? "Enter provider API key" : "Optional key";
    const link = document.querySelector("#providerDocs");
    if (link) { link.hidden = !provider.docs; if (provider.docs && /^https:\/\//.test(provider.docs)) link.href = provider.docs; }
  }

  function setConnectionPresentation(connection) {
    const badge =
      $("#connectionStatusBadge");

    const name =
      $("#activeModelName");

    const provider =
      $("#activeModelProvider");

    const topbar =
      $("#activeModelTopbar");

    if (!connection?.model) {
      badge.className =
        "status-badge neutral";

      badge.textContent =
        "Not configured";

      name.textContent =
        "No active model";

      provider.textContent =
        "Configure a provider";

      topbar.textContent =
        "No model";

      return;
    }

    badge.className =
      "status-badge success";

    badge.textContent =
      "Configured";

    name.textContent =
      connection.model;

    provider.textContent =
      connection.provider ||
      "Model provider";

    topbar.textContent =
      connection.model;
  }

  /* ==========================================================
     SAVE
  ========================================================== */

  async function saveConnection(event) {
    event.preventDefault();

    const payload =
      getFormPayload();

    if (!payload.provider) {
      toast(
        "Provider required",
        "Choose an inference provider.",
        "error"
      );

      return;
    }

    if (!payload.model) {
      toast(
        "Model required",
        "Enter or select a model ID.",
        "error"
      );

      return;
    }

    const button =
      $("#saveConnectionButton");

    button.disabled = true;
    button.textContent = "Saving...";

    try {
      const result =
        await request(
          "/api/model/settings",
          {
            method: "POST",
            body: JSON.stringify(payload),
          }
        );

      $("#apiKeyInput").value = "";

      state.active =
        result?.settings ||
        result?.active ||
        (result?.provider ? result : null) ||
        {
          provider: payload.provider,
          model: payload.model,
          base_url: payload.base_url,
        };

      setConnectionPresentation({
        provider: payload.provider,
        model: payload.model,
      });

      toast(
        "Connection saved",
        `${payload.provider} · ${payload.model}`,
        "success"
      );

    } catch (error) {
      toast(
        "Unable to save connection",
        error.message,
        "error"
      );
    } finally {
      $("#apiKeyInput").value = "";
      $("#apiKeyInput").type = "password";
      button.disabled = false;
      button.textContent = "Save connection";
    }
  }

  /* ==========================================================
     REMOVE
  ========================================================== */

  async function removeConnection() {
    const button =
      $("#removeConnectionButton");

    button.disabled = true;

    try {
      await request(
        "/api/model/settings",
        {
          method: "DELETE",
        }
      );

      state.active = await request("/api/providers").then(r => r.active);

      $("#modelIdInput").value = "";
      $("#baseUrlInput").value = "";
      $("#apiKeyInput").value = "";

      applyActiveConnection();

      toast(
        "Connection removed",
        "The saved model profile was cleared.",
        "success"
      );

    } catch (error) {
      toast(
        "Unable to remove connection",
        error.message,
        "error"
      );
    } finally {
      $("#apiKeyInput").value = "";
      $("#apiKeyInput").type = "password";
      button.disabled = false;
    }
  }

  /* ==========================================================
     FETCH MODELS
  ========================================================== */

  async function fetchModels() {
    const payload =
      getFormPayload();

    if (!payload.provider) {
      toast(
        "Provider required",
        "Choose a provider first.",
        "error"
      );

      return;
    }

    const button =
      $("#fetchModelsButton");

    button.disabled = true;
    button.textContent = "Loading...";

    try {
      const result =
        await request(
          "/api/model/models",
          {
            method: "POST",
            body: JSON.stringify(payload),
          }
        );

      const models =
        normalizeModels(result);

      renderModels(models);

      toast(
        "Models loaded",
        `${models.length} model${models.length === 1 ? "" : "s"} returned.`,
        "success"
      );

    } catch (error) {
      toast(
        "Unable to fetch models",
        error.message,
        "error"
      );
    } finally {
      $("#apiKeyInput").value = "";
      $("#apiKeyInput").type = "password";
      button.disabled = false;
      button.textContent = "Fetch models";
    }
  }

  function normalizeModels(payload) {
    const source =
      Array.isArray(payload)
        ? payload
        : (
          payload?.models ||
          payload?.data ||
          payload?.items ||
          []
        );

    return source
      .map((item) => {
        if (typeof item === "string") {
          return item;
        }

        return (
          item?.id ||
          item?.name ||
          item?.model ||
          ""
        );
      })
      .filter(Boolean);
  }

  function renderModels(models) {
    const panel =
      $("#modelsResultPanel");

    const container =
      $("#availableModels");

    panel.classList.remove("hidden");

    if (!models.length) {
      container.innerHTML = `
        <div class="empty-state">
          No models were returned by this provider.
        </div>
      `;

      return;
    }

    container.innerHTML =
      models.map((model) => `
        <button
          type="button"
          class="model-option"
          data-model="${escapeHtml(model)}"
        >
          <div class="model-option-name">
            ${escapeHtml(model)}
          </div>

          <div class="model-option-hint">
            Use this model
          </div>
        </button>
      `).join("");

    $$(".model-option", container)
      .forEach((button) => {
        button.addEventListener(
          "click",
          () => {
            $("#modelIdInput").value =
              button.dataset.model;

            toast(
              "Model selected",
              button.dataset.model,
              "success"
            );
          }
        );
      });
  }

  /* ==========================================================
     TEST CONNECTION
  ========================================================== */

  async function testConnection() {
    const payload =
      getFormPayload();

    if (!payload.provider) {
      toast(
        "Provider required",
        "Choose a provider first.",
        "error"
      );

      return;
    }

    if (!payload.model) {
      toast(
        "Model required",
        "Enter a model ID first.",
        "error"
      );

      return;
    }

    const button =
      $("#testConnectionButton");

    button.disabled = true;
    button.textContent = "Testing...";

    const started =
      performance.now();

    try {
      const result =
        await request(
          "/api/model/test",
          {
            method: "POST",
            body: JSON.stringify(payload),
          }
        );

      $("#apiKeyInput").value = "";

      const elapsed =
        Math.round(
          performance.now() - started
        );

      const resultMessage =
        result?.message ||
        result?.response ||
        result?.status ||
        "Connection succeeded.";

      toast(
        "Connection successful",
        `${resultMessage} · ${elapsed} ms`,
        "success"
      );

    } catch (error) {
      toast(
        "Connection test failed",
        error.message,
        "error"
      );
    } finally {
      $("#apiKeyInput").value = "";
      $("#apiKeyInput").type = "password";
      button.disabled = false;
      button.textContent = "Test connection";
    }
  }

  /* ==========================================================
     SECRET VISIBILITY
  ========================================================== */

  function toggleSecret() {
    const input =
      $("#apiKeyInput");

    const button =
      $("#toggleApiKeyButton");

    const visible =
      input.type === "text";

    input.type =
      visible
        ? "password"
        : "text";

    button.textContent =
      visible
        ? "Show"
        : "Hide";

    button.setAttribute(
      "aria-label",
      visible
        ? "Show API key"
        : "Hide API key"
    );
  }

  /* ==========================================================
     INIT
  ========================================================== */

  function initialize() {
    $("#providerSelect").addEventListener("change", () => providerChanged());
    $("#connectionForm")
      ?.addEventListener(
        "submit",
        saveConnection
      );

    $("#removeConnectionButton")
      ?.addEventListener(
        "click",
        removeConnection
      );

    $("#fetchModelsButton")
      ?.addEventListener(
        "click",
        fetchModels
      );

    $("#testConnectionButton")
      ?.addEventListener(
        "click",
        testConnection
      );

    $("#toggleApiKeyButton")
      ?.addEventListener(
        "click",
        toggleSecret
      );

    loadProviders();
  }

  document.addEventListener(
    "DOMContentLoaded",
    initialize
  );
})();
