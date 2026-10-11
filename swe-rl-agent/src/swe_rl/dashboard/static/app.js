/* ============================================================
   SWE-RL ENTERPRISE CONSOLE
============================================================ */

(() => {
  "use strict";

  /* ==========================================================
     STATE
  ========================================================== */

  const state = {
    page: "overview",
    runs: [],
    filteredRuns: [],
    selectedRun: null,
    health: null,
    commandIndex: 0,
    assistantResponse: "",
  };

  /* ==========================================================
     DOM HELPERS
  ========================================================== */

  const $ = (selector, root = document) =>
    root.querySelector(selector);

  const $$ = (selector, root = document) =>
    Array.from(root.querySelectorAll(selector));

  const escapeHtml = (value) => {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  };

  const safeJson = (value) => {
    try {
      return JSON.stringify(value ?? {}, null, 2);
    } catch {
      return String(value ?? "");
    }
  };

  const firstDefined = (...values) =>
    values.find((value) => value !== undefined && value !== null);

  /* ==========================================================
     PUBLIC APP API
  ========================================================== */

  window.SWERL = {
    showToast,
    refreshAll,
    setPage,
    escapeHtml, fetchJson, selectRun, activateRunTab, getState: () => state, loadRuns,
  };

  /* ==========================================================
     FORMATTERS
  ========================================================== */

  function formatNumber(value) {
    const n = Number(value);

    if (!Number.isFinite(n)) {
      return "—";
    }

    if (n >= 1_000_000) {
      return `${(n / 1_000_000).toFixed(1)}M`;
    }

    if (n >= 1_000) {
      return `${(n / 1_000).toFixed(1)}K`;
    }

    return String(n);
  }

  function formatPercent(value) {
    const n = Number(value);

    if (!Number.isFinite(n)) {
      return "—";
    }

    return `${n.toFixed(1)}%`;
  }

  function formatDuration(milliseconds) {
    const ms = Number(milliseconds);

    if (!Number.isFinite(ms)) {
      return "—";
    }

    if (ms < 1000) {
      return `${Math.round(ms)} ms`;
    }

    return `${(ms / 1000).toFixed(2)} s`;
  }

  function getRunId(run) {
    return String(
      firstDefined(
        run?.run_id,
        run?.id,
        run?.trajectory_id,
        run?.metadata?.run_id,
        "unknown-run"
      )
    );
  }

  function getInstanceId(run) {
    return String(
      firstDefined(
        run?.instance_id,
        run?.instance?.instance_id,
        run?.instance?.id,
        run?.metadata?.instance_id,
        run?.id,
        "unknown-instance"
      )
    );
  }

  function getModel(run) {
    return String(
      firstDefined(
        run?.model,
        run?.model_name,
        run?.metadata?.model,
        run?.metadata?.model_name,
        run?.provider_model,
        "Unknown model"
      )
    );
  }

  function getStatus(run) {
    const raw = String(run?.status || "").toLowerCase();
    if (["queued", "preparing", "running", "testing", "evaluating", "cancelled", "failed", "paused"].includes(raw)) return raw;
    const resolved = run?.resolved ?? run?.reward_details?.exec?.resolved ?? run?.reward_details?.resolved;
    if (resolved === true) return "resolved";
    if (resolved === false && isCompleted(run)) return "unresolved";
    if (isCompleted(run) && run?.reward != null) return Number(run.reward) === 1 ? "resolved" : "unresolved";
    return "incomplete";
  }

  function isCompleted(run) {
    return ["completed", "resolved", "unresolved", "success", "passed"].includes(String(run?.status || "").toLowerCase()) || Boolean(run?.finished_at && run?.reward != null);
  }

  function getReward(run) {
    const value = Number(
      firstDefined(
        run?.reward,
        run?.execution_reward,
        run?.reward_details?.reward,
        run?.reward_details?.binary_reward,
        0
      )
    );

    return Number.isFinite(value) ? value : 0;
  }

  function getSteps(run) {
    return Number(
      firstDefined(
        run?.steps,
        run?.n_steps,
        run?.step_count,
        run?.metadata?.steps,
        Array.isArray(run?.tool_calls) ? run.tool_calls.length : 0
      )
    ) || 0;
  }

  function getToolCalls(run) {
    const value = firstDefined(
      run?.tool_calls,
      run?.trajectory?.tool_calls,
      run?.calls,
      []
    );

    return Array.isArray(value) ? value : [];
  }

  function getMessages(run) {
    const value = firstDefined(
      run?.messages,
      run?.trajectory?.messages,
      run?.conversation,
      []
    );

    return Array.isArray(value) ? value : [];
  }

  function getPatch(run) {
    return String(
      firstDefined(
        run?.final_patch,
        run?.patch,
        run?.trajectory?.final_patch,
        ""
      )
    );
  }

  function getIssue(run) {
    return String(
      firstDefined(
        run?.problem_statement,
        run?.issue,
        run?.instance?.problem_statement,
        run?.metadata?.problem_statement,
        run?.config?.problem_statement,
        "No issue statement recorded."
      )
    );
  }

  function getInputTokens(run) {
    return Number(
      firstDefined(
        run?.tokens_in,
        run?.input_tokens,
        run?.tokens?.input,
        run?.usage?.prompt_tokens,
        run?.usage?.input_tokens,
        0
      )
    ) || 0;
  }

  function getOutputTokens(run) {
    return Number(
      firstDefined(
        run?.tokens_out,
        run?.output_tokens,
        run?.tokens?.output,
        run?.usage?.completion_tokens,
        run?.usage?.output_tokens,
        0
      )
    ) || 0;
  }

  function getTotalTokens(run) {
    return getInputTokens(run) + getOutputTokens(run);
  }

  /* ==========================================================
     THEME
  ========================================================== */

  function initializeTheme() {
    const saved = localStorage.getItem("swe-rl-theme");

    const theme =
      saved ||
      (
        window.matchMedia("(prefers-color-scheme: dark)").matches
          ? "dark"
          : "light"
      );

    document.documentElement.dataset.theme = theme;
  }

  function toggleTheme() {
    const current =
      document.documentElement.dataset.theme || "light";

    const next = current === "dark" ? "light" : "dark";

    document.documentElement.dataset.theme = next;

    localStorage.setItem("swe-rl-theme", next);
  }

  /* ==========================================================
     NAVIGATION
  ========================================================== */

  const PAGE_NAMES = {
    overview: "Overview",
    runs: "Repair runs",
    infrastructure: "Infrastructure",
    connections: "Model connections",
    assistant: "Repair assistant",
    "new-run": "New repair run", experiments: "Experiments", compare: "Compare", benchmarks: "SWE-bench", evaluations: "Evaluations", reports: "Reports", training: "Training jobs", checkpoints: "Checkpoints", replay: "Replay buffer", reward: "Reward analysis", registry: "Model registry", prompts: "Prompt versions", workers: "Workers", containers: "Containers", metrics: "Metrics", artifacts: "Artifacts", datasets: "Datasets", settings: "Settings",
  };

  function setPage(page) {
    if (!PAGE_NAMES[page]) {
      page = "overview";
    }

    state.page = page;

    $$(".page").forEach((element) => {
      element.classList.toggle(
        "is-active",
        element.id === `page-${page}`
      );
    });

    $$(".nav-item[data-page]").forEach((item) => {
      item.classList.toggle("is-active", item.dataset.page === page);
      if(item.dataset.page === page) item.setAttribute("aria-current", "page"); else item.removeAttribute("aria-current");
    });

    $("#breadcrumbCurrent").textContent =
      PAGE_NAMES[page];

    history.replaceState(
      null,
      "",
      `#${page}`
    );

    closeMobileSidebar();
    document.dispatchEvent(new CustomEvent("swerl:page", {detail:page}));
  }

  function initializeNavigation() {
    $$(".nav-item[data-page]").forEach((item) => {
      item.addEventListener("click", () => {
        setPage(item.dataset.page);
      });
    });

    $$("[data-action='view-runs']").forEach((button) => {
      button.addEventListener("click", () => setPage("runs"));
    });

    $$("[data-action='open-connections']").forEach((button) => {
      button.addEventListener(
        "click",
        () => setPage("connections")
      );
    });

    $$("[data-action='open-infrastructure']").forEach((button) => {
      button.addEventListener(
        "click",
        () => setPage("infrastructure")
      );
    });

    const hashPage =
      window.location.hash.replace("#", "");

    if (PAGE_NAMES[hashPage]) {
      setPage(hashPage);
    }
  }

  /* ==========================================================
     MOBILE SIDEBAR
  ========================================================== */

  function openMobileSidebar() {
    document.body.classList.add("sidebar-open");
  }

  function closeMobileSidebar() {
    document.body.classList.remove("sidebar-open");
  }

  /* ==========================================================
     API
  ========================================================== */

  async function fetchJson(url, options = {}) {
    return window.SWERLAPI.request(url, options);
  }

  /* ==========================================================
     RUNS
  ========================================================== */

  function normalizeRuns(payload) {
    if (Array.isArray(payload)) {
      return payload;
    }

    if (Array.isArray(payload?.runs)) {
      return payload.runs;
    }

    if (Array.isArray(payload?.items)) {
      return payload.items;
    }

    if (Array.isArray(payload?.data)) {
      return payload.data;
    }

    return [];
  }

  async function loadRuns() {
    try {
      const payload = await fetchJson("/api/runs");
      state.runs = normalizeRuns(payload).filter(run => !run.kind || run.kind === "run");
      document.dispatchEvent(new CustomEvent("swerl:runs", {detail:state.runs}));
      if (state.selectedRun) { const updated = state.runs.find(r => getRunId(r) === getRunId(state.selectedRun)); if (updated) { state.selectedRun = updated; renderInspector(updated); } }

      applyRunFilters();
      renderOverview();

      $("#overviewNotice").classList.add("notice-hidden");
    } catch (error) {
      state.runs = [];
      state.filteredRuns = [];
      state.selectedRun = null;
      $("#runEmptyState").classList.remove("hidden");
      $("#runInspectorContent").classList.add("hidden");

      renderOverview();
      renderRunList();

      const notice = $("#overviewNotice");

      notice.textContent =
        `Unable to load recorded runs: ${error.message}`;

      notice.classList.remove("notice-hidden");

      showToast(
        "Runs unavailable",
        error.message,
        "error"
      );
    }
  }

  function applyRunFilters() {
    const query =
      ($("#runSearchInput")?.value || "")
        .trim()
        .toLowerCase();

    const filter =
      $("#runStatusFilter")?.value || "all";

    state.filteredRuns = state.runs.filter((run) => {
      const searchable = [
        getRunId(run),
        getInstanceId(run),
        getModel(run),
        run?.repository,
        run?.repo,
        safeJson(run?.messages), safeJson(run?.tool_calls), safeJson(run?.error),
      ]
        .filter(Boolean)
        .join(" ")
        .toLowerCase();

      const matchesQuery =
        !query || searchable.includes(query);

      const runStatus = getStatus(run);

      const matchesStatus =
        filter === "all" ||
        runStatus === filter;

      return matchesQuery && matchesStatus;
    });

    renderRunList();
  }

  function renderOverview() {
    const runs = state.runs;

    const completed =
      runs.filter(isCompleted);

    const resolved =
      completed.filter(
        (run) => getStatus(run) === "resolved"
      );

    const totalTokens =
      runs.reduce(
        (sum, run) => sum + getTotalTokens(run),
        0
      );

    $("#metricRuns").textContent =
      formatNumber(runs.length);

    $("#metricResolved").textContent =
      formatNumber(resolved.length);

    $("#metricRate").textContent =
      completed.length
        ? formatPercent(
            (resolved.length / completed.length) * 100
          )
        : "—";

    $("#metricTokens").textContent =
      formatNumber(totalTokens);

    $("#sidebarRunCount").textContent =
      formatNumber(runs.length);

    renderOverviewRuns();
  }

  function renderOverviewRuns() {
    const body = $("#overviewRunsBody");

    const recent = state.runs.slice(0, 6);

    if (!recent.length) {
      body.innerHTML = `
        <tr>
          <td colspan="5">
            <div class="table-loading">
              No recorded runs found.
            </div>
          </td>
        </tr>
      `;

      return;
    }

    body.innerHTML = recent.map((run) => {
      const status = getStatus(run);

      return `
        <tr
          class="clickable"
          data-overview-run="${escapeHtml(getRunId(run))}"
        >
          <td>
            <span class="table-primary mono">
              ${escapeHtml(getInstanceId(run))}
            </span>
          </td>

          <td>
            ${statusBadge(status)}
          </td>

          <td>
            ${escapeHtml(getModel(run))}
          </td>

          <td class="mono">
            ${escapeHtml(getReward(run).toFixed(2))}
          </td>

          <td class="mono">
            ${escapeHtml(getSteps(run))}
          </td>
        </tr>
      `;
    }).join("");

    $$("[data-overview-run]", body).forEach((row) => {
      row.addEventListener("click", () => {
        const run = state.runs.find(
          (candidate) =>
            getRunId(candidate) ===
            row.dataset.overviewRun
        );

        if (!run) {
          return;
        }

        setPage("runs");
        selectRun(run);
      });
    });
  }

  function statusBadge(status) {
    const classes = {
      resolved: "success",
      unresolved: "danger",
      incomplete: "warning", failed: "danger", cancelled: "neutral", queued: "neutral", preparing: "info", running: "info", testing: "info", evaluating: "info", paused: "warning",
    };

    return `
      <span class="status-badge ${classes[status] || "neutral"}">
        ${escapeHtml(status)}
      </span>
    `;
  }

  function renderRunList() {
    const container = $("#runList");

    if (!container) {
      return;
    }

    $("#runSourceLabel").textContent = "Repair runs";

    $("#runResultCount").textContent =
      `${state.filteredRuns.length} result${state.filteredRuns.length === 1 ? "" : "s"}`;

    if (!state.filteredRuns.length) {
      container.innerHTML = `
        <div class="empty-state compact-empty">
          No runs match this filter.
        </div>
      `;

      return;
    }

    container.innerHTML =
      state.filteredRuns.map((run) => {
        const selected =
          state.selectedRun &&
          getRunId(state.selectedRun) === getRunId(run);

        return `
          <button
            class="run-list-item ${selected ? "is-selected" : ""}"
            data-run-id="${escapeHtml(getRunId(run))}"
          >
            <div class="run-list-item-top">
              <span class="run-instance">
                ${escapeHtml(getInstanceId(run))}
              </span>

              ${statusBadge(getStatus(run))}
            </div>

            <div class="run-list-item-bottom">
              <span class="run-model">
                ${escapeHtml(getModel(run))}
              </span>

              <span class="run-reward">
                r=${escapeHtml(getReward(run).toFixed(2))}
              </span>
            </div>
          </button>
        `;
      }).join("");

    $$(".run-list-item", container).forEach((button) => {
      button.addEventListener("click", () => {
        const run = state.runs.find(
          (candidate) =>
            getRunId(candidate) === button.dataset.runId
        );

        if (run) {
          selectRun(run);
        }
      });
    });
  }

  function selectRun(run) {
    state.selectedRun = run;

    renderRunList();
    renderInspector(run);
    document.dispatchEvent(new CustomEvent("swerl:selected", {detail:run}));
  }

  function renderInspector(run) {
    $("#runEmptyState").classList.add("hidden");
    $("#runInspectorContent").classList.remove("hidden");

    $("#inspectorInstanceId").textContent =
      getInstanceId(run);

    $("#inspectorIssue").textContent =
      getIssue(run);

    $("#inspectorModel").textContent =
      getModel(run);

    $("#inspectorSteps").textContent =
      formatNumber(getSteps(run));

    $("#inspectorReward").textContent =
      getReward(run).toFixed(2);

    $("#inspectorToolCalls").textContent =
      formatNumber(getToolCalls(run).length);

    $("#inspectorTokens").textContent =
      formatNumber(getTotalTokens(run));

    $("#inspectorRunId").textContent =
      getRunId(run);

    const status = getStatus(run);
    const statusNode = $("#inspectorStatusBadge");

    statusNode.className =
      `status-badge ${
        status === "resolved"
          ? "success"
          : status === "unresolved"
            ? "danger"
            : "warning"
      }`;

    statusNode.textContent = status;

    renderPatch(run);
    renderTrace(run);
    renderTests(run);
    renderConversation(run);
    window.SWERLControl?.renderRunExtras(run);
  }

  /* ==========================================================
     PATCH
  ========================================================== */

  function renderPatch(run) {
    const patch = getPatch(run);
    const viewer = $("#patchViewer");

    if (!patch.trim()) {
      viewer.innerHTML = `
        <div class="empty-state">
          No patch was recorded for this run.
        </div>
      `;

      $("#patchStats").textContent = "No changes";
      return;
    }

    let added = 0;
    let removed = 0;

    const lines = patch.split("\n");

    const html = lines.map((line) => {
      let cls = "";

      if (
        line.startsWith("diff ") ||
        line.startsWith("index ") ||
        line.startsWith("--- ") ||
        line.startsWith("+++ ")
      ) {
        cls = "file";
      } else if (line.startsWith("@@")) {
        cls = "hunk";
      } else if (line.startsWith("+")) {
        cls = "added";
        added += 1;
      } else if (line.startsWith("-")) {
        cls = "removed";
        removed += 1;
      } else if (
        line.startsWith("\\ No newline")
      ) {
        cls = "meta";
      }

      return `
        <span class="diff-line ${cls}">
          ${escapeHtml(line || " ")}
        </span>
      `;
    }).join("");

    viewer.innerHTML = html;

    $("#patchStats").textContent =
      `+${added} −${removed}`;
  }

  /* ==========================================================
     TRACE
  ========================================================== */

  function renderTrace(run) {
    const calls = getToolCalls(run);
    const container = $("#traceViewer");

    if (!calls.length) {
      container.innerHTML = `
        <div class="empty-state">
          No tool calls recorded.
        </div>
      `;

      return;
    }

    container.innerHTML =
      calls.map((call, index) => {
        const tool =
          firstDefined(
            call?.tool,
            call?.name,
            call?.tool_name,
            "unknown_tool"
          );

        const status =
          String(
            firstDefined(
              call?.status,
              call?.success === false || call?.result?.ok === false || call?.result?.error ? "failed" : "success"
            )
          ).toLowerCase();

        const failed =
          status.includes("fail") ||
          status.includes("error");

        const duration =
          firstDefined(
            call?.duration_s != null ? call.duration_s * 1000 : undefined,
            call?.duration_ms,
            call?.duration,
            call?.elapsed_ms
          );

        const args =
          firstDefined(
            call?.arguments,
            call?.args,
            {}
          );

        const result =
          firstDefined(
            call?.result,
            call?.output,
            call?.observation,
            ""
          );

        return `
          <article class="trace-item" id="trace-step-${index + 1}">

            <button class="trace-summary" aria-expanded="false">
              <span class="trace-step">
                ${String(index + 1).padStart(2, "0")}
              </span>

              <span class="trace-tool">
                ${escapeHtml(tool)}
              </span>

              <span class="trace-duration">
                ${
                  duration !== undefined
                    ? escapeHtml(formatDuration(duration))
                    : ""
                }
              </span>

              <span class="trace-status ${failed ? "failed" : ""}"></span>
            </button>

            <div class="trace-body">

              <div class="trace-section">
                <div class="trace-section-title">
                  Arguments
                </div>

                <pre class="trace-code">${escapeHtml(safeJson(args))}</pre>
              </div>

              <div class="trace-section">
                <div class="trace-section-title">
                  Result
                </div>

                <pre class="trace-code">${escapeHtml(
                  typeof result === "string"
                    ? result
                    : safeJson(result)
                )}</pre>
              </div>

            </div>
          </article>
        `;
      }).join("");

    $$(".trace-summary", container).forEach((button) => {
      button.addEventListener("click", () => {
        const open = button.closest(".trace-item").classList.toggle("is-open");
        button.setAttribute("aria-expanded", String(open));
      });
    });
  }

  /* ==========================================================
     TESTS
  ========================================================== */

  function normalizeTestGroup(group) {
    if (!group) {
      return {
        passed: [],
        failed: [],
      };
    }

    if (Array.isArray(group)) {
      return {
        passed: group,
        failed: [],
      };
    }

    return {
      passed:
        Array.isArray(group.passed)
          ? group.passed
          : Array.isArray(group.success)
            ? group.success
            : [],

      failed:
        Array.isArray(group.failed)
          ? group.failed
          : Array.isArray(group.failures)
            ? group.failures
            : [],
    };
  }

  function getTests(run) {
    const f2p = normalizeTestGroup(
      firstDefined(
        run?.fail_to_pass,
        run?.FAIL_TO_PASS,
        run?.tests?.fail_to_pass,
        run?.reward_details?.fail_to_pass,
        {passed: run?.reward_details?.exec?.f2p_passed || run?.reward_details?.f2p_passed || [], failed: run?.reward_details?.exec?.f2p_failed || run?.reward_details?.f2p_failed || []}
      )
    );

    const p2p = normalizeTestGroup(
      firstDefined(
        run?.pass_to_pass,
        run?.PASS_TO_PASS,
        run?.tests?.pass_to_pass,
        run?.reward_details?.pass_to_pass,
        {passed: run?.reward_details?.exec?.p2p_passed || run?.reward_details?.p2p_passed || [], failed: run?.reward_details?.exec?.p2p_failed || run?.reward_details?.p2p_failed || []}
      )
    );

    return {
      f2p,
      p2p,
    };
  }

  function renderTests(run) {
    const { f2p, p2p } = getTests(run);

    $("#f2pSummary").textContent =
      `${f2p.passed.length} / ${
        f2p.passed.length + f2p.failed.length
      } passed`;

    $("#p2pSummary").textContent =
      `${p2p.passed.length} / ${
        p2p.passed.length + p2p.failed.length
      } passed`;

    const viewer = $("#testViewer");

    const groups = [
      ["FAIL_TO_PASS", f2p],
      ["PASS_TO_PASS", p2p],
    ];

    const hasAny =
      groups.some(
        ([, group]) =>
          group.passed.length ||
          group.failed.length
      );

    if (!hasAny) {
      viewer.innerHTML = `
        <div class="empty-state">
          No test evidence was recorded.
        </div>
      `;

      return;
    }

    viewer.innerHTML =
      groups.map(([title, group]) => {
        const rows = [
          ...group.passed.map((name) => ({
            name,
            passed: true,
          })),
          ...group.failed.map((name) => ({
            name,
            passed: false,
          })),
        ];

        return `
          <section class="test-group">

            <div class="test-group-title">
              ${escapeHtml(title)}
            </div>

            ${
              rows.length
                ? rows.map((item) => `
                    <div class="test-row">
                      <span class="test-result-icon ${item.passed ? "pass" : "fail"}">
                        ${item.passed ? "✓" : "×"}
                      </span>

                      <span>
                        ${escapeHtml(
                          typeof item.name === "string"
                            ? item.name
                            : safeJson(item.name)
                        )}
                      </span>
                    </div>
                  `).join("")
                : `
                  <div class="empty-state compact-empty">
                    No tests recorded in this group.
                  </div>
                `
            }

          </section>
        `;
      }).join("");
  }

  /* ==========================================================
     CONVERSATION
  ========================================================== */

  function renderConversation(run) {
    const messages = getMessages(run);
    const container = $("#conversationViewer");

    if (!messages.length) {
      container.innerHTML = `
        <div class="empty-state">
          No conversation was recorded.
        </div>
      `;

      return;
    }

    container.innerHTML =
      messages.map((message) => {
        const role =
          String(
            firstDefined(
              message?.role,
              message?.type,
              "message"
            )
          ).toLowerCase();

        const content =
          firstDefined(
            message?.content,
            message?.text,
            message?.message,
            ""
          );

        return `
          <article class="message ${escapeHtml(role)}">
            <div class="message-role">
              ${escapeHtml(role)}
            </div>

            <div class="message-content">
              ${escapeHtml(
                typeof content === "string"
                  ? content
                  : safeJson(content)
              )}
            </div>
          </article>
        `;
      }).join("");
  }

  /* ==========================================================
     HEALTH
  ========================================================== */

  function normalizeHealth(payload) {
    if (!payload || typeof payload !== "object") {
      return {};
    }

    if (
      payload.services &&
      typeof payload.services === "object"
    ) {
      return payload.services;
    }

    return Object.fromEntries(Object.entries(payload).filter(([key]) => !["runs", "jobs", "queue", "capabilities"].includes(key)));
  }

  function normalizeServiceState(value) {
    if (
      value &&
      typeof value === "object"
    ) {
      const raw = String(
        firstDefined(
          value.status,
          value.state,
          value.healthy === true ? "healthy" : undefined,
          value.connected === true ? "connected" : undefined,
          value.ok === true ? "healthy" : undefined,
          "unknown"
        )
      ).toLowerCase();

      return {
        raw,
        detail:
          firstDefined(
            value.detail,
            value.message,
            value.error,
            value.latency_ms
              ? `${value.latency_ms} ms`
              : ""
          ),
        object: value,
      };
    }

    const raw = String(value ?? "unknown")
      .toLowerCase();

    return {
      raw,
      detail: "",
      object: value,
    };
  }

  function serviceIsHealthy(service) {
    return [
      "ok",
      "healthy",
      "connected",
      "ready",
      "running",
      "true",
      "up",
      "available",
    ].includes(service.raw);
  }

  async function loadHealth() {
    try {
      const payload =
        await fetchJson("/api/health");

      state.health =
        normalizeHealth(payload);

      renderHealth();
    } catch (error) {
      state.health = null;

      renderHealth();

      showToast(
        "Infrastructure unavailable",
        error.message,
        "error"
      );
    }
  }

  function renderHealth() {
    const mini = $("#healthMiniList");
    const grid = $("#infrastructureGrid");

    if (!state.health) {
      mini.innerHTML = `
        <div class="empty-state compact-empty">
          Health endpoint unavailable.
        </div>
      `;

      grid.innerHTML = `
        <div class="service-card">
          <div class="service-name">Health API</div>
          <div class="service-state">Unavailable</div>
          <div class="service-detail">
            Unable to retrieve infrastructure state.
          </div>
        </div>
      `;

      setGlobalHealth(false);

      return;
    }

    const entries =
      Object.entries(state.health)
        .filter(
          ([key]) =>
            ![
              "timestamp",
              "time",
              "checked_at",
              "version"
            ].includes(key)
        );

    const normalized =
      entries.map(([name, value]) => {
        return {
          name,
          ...normalizeServiceState(value),
        };
      });

    const healthyCount =
      normalized.filter(serviceIsHealthy).length;

    const allHealthy =
      normalized.length > 0 &&
      healthyCount === normalized.length;

    setGlobalHealth(allHealthy);

    mini.innerHTML =
      normalized
        .slice(0, 6)
        .map((service) => {
          const healthy =
            serviceIsHealthy(service);

          return `
            <div class="health-mini-item">
              <span class="health-dot ${healthy ? "success" : "danger"}"></span>

              <span class="health-mini-name">
                ${escapeHtml(prettyServiceName(service.name))}
              </span>

              <span class="health-mini-state">
                ${escapeHtml(service.raw)}
              </span>
            </div>
          `;
        })
        .join("");

    grid.innerHTML =
      normalized
        .map((service) => {
          const healthy =
            serviceIsHealthy(service);

          return `
            <article class="service-card">

              <div class="service-card-top">
                <span class="service-name">
                  ${escapeHtml(prettyServiceName(service.name))}
                </span>

                <span class="service-dot ${healthy ? "success" : "danger"}"></span>
              </div>

              <div class="service-state">
                ${healthy ? "Reachable" : "Attention required"}
              </div>

              <div class="service-detail">
                ${
                  service.detail
                    ? escapeHtml(String(service.detail))
                    : escapeHtml(service.raw)
                }
              </div>

            </article>
          `;
        })
        .join("");

    $("#healthTimestamp").textContent =
      new Date().toLocaleTimeString();
  }

  function prettyServiceName(name) {
    return String(name)
      .replaceAll("_", " ")
      .replace(/\b\w/g, (char) => char.toUpperCase());
  }

  function setGlobalHealth(healthy) {
    const dot = $("#sidebarHealthDot");
    const navDot = $("#infraNavDot");
    const text = $("#sidebarHealthText");

    dot.className =
      `environment-dot ${healthy ? "success" : "danger"}`;

    navDot.className =
      `nav-status-dot ${healthy ? "success" : "danger"}`;

    text.textContent =
      healthy
        ? "Services reachable"
        : "Check services";
  }

  /* ==========================================================
     RUN TAB CONTROLS
  ========================================================== */

  function initializeRunTabs() {
    const tabs = $$("[data-run-tab]");

    tabs.forEach((tab, index) => {
      tab.addEventListener("click", () => {
        activateRunTab(tab.dataset.runTab);
      });

      tab.addEventListener("keydown", (event) => {
        if (
          ![
            "ArrowLeft",
            "ArrowRight",
            "Home",
            "End"
          ].includes(event.key)
        ) {
          return;
        }

        event.preventDefault();

        let nextIndex = index;

        if (event.key === "ArrowLeft") {
          nextIndex =
            (index - 1 + tabs.length) % tabs.length;
        }

        if (event.key === "ArrowRight") {
          nextIndex =
            (index + 1) % tabs.length;
        }

        if (event.key === "Home") {
          nextIndex = 0;
        }

        if (event.key === "End") {
          nextIndex = tabs.length - 1;
        }

        tabs[nextIndex].focus();

        activateRunTab(
          tabs[nextIndex].dataset.runTab
        );
      });
    });
  }

  function activateRunTab(name) {
    $$("[data-run-tab]").forEach((tab) => {
      const active =
        tab.dataset.runTab === name;

      tab.classList.toggle("is-active", active);
      tab.setAttribute("aria-selected", String(active));
      tab.tabIndex = active ? 0 : -1;
    });

    $$("[data-run-panel]").forEach((panel) => {
      panel.classList.toggle(
        "is-active",
        panel.dataset.runPanel === name
      );
    });
  }

  /* ==========================================================
     EXPORT / COPY
  ========================================================== */

  function downloadJson(data, filename) {
    const blob = new Blob(
      [JSON.stringify(data, null, 2)],
      {
        type: "application/json",
      }
    );

    const url =
      URL.createObjectURL(blob);

    const anchor =
      document.createElement("a");

    anchor.href = url;
    anchor.download = filename;

    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();

    URL.revokeObjectURL(url);
  }

  async function copyText(text, label = "Copied") {
    try {
      await navigator.clipboard.writeText(text);

      showToast(
        label,
        "Copied to clipboard.",
        "success"
      );
    } catch {
      showToast(
        "Copy failed",
        "Clipboard permission was unavailable.",
        "error"
      );
    }
  }

  /* ==========================================================
     ASSISTANT
  ========================================================== */

  async function runAssistant() {
    const prompt =
      $("#assistantPrompt").value.trim();

    if (!prompt) {
      showToast(
        "Prompt required",
        "Enter an issue description or code snippet.",
        "error"
      );

      return;
    }

    const button =
      $("#runAssistantButton");

    button.disabled = true;
    button.textContent = "Running...";

    $("#assistantResponse").innerHTML = `
      <div class="empty-state">
        <div class="spinner"></div>
        Requesting model response...
      </div>
    `;

    try {
      const started = performance.now();

      const payload =
        await fetchJson(
          "/api/model/assist",
          {
            method: "POST",
            body: JSON.stringify({
              prompt,
            }),
          }
        );

      const response =
        firstDefined(
          payload?.response,
          payload?.text,
          payload?.content,
          payload?.message,
          typeof payload === "string"
            ? payload
            : safeJson(payload)
        );

      state.assistantResponse =
        String(response ?? "");

      $("#assistantResponse").textContent =
        state.assistantResponse;

      $("#assistantResponseMeta").textContent =
        `Completed in ${(
          (performance.now() - started) /
          1000
        ).toFixed(1)} s`;

      $("#copyAssistantButton").disabled =
        !state.assistantResponse;

    } catch (error) {
      state.assistantResponse = "";

      $("#assistantResponse").innerHTML = `
        <div class="empty-state">
          ${escapeHtml(error.message)}
        </div>
      `;

      $("#assistantResponseMeta").textContent =
        "Request failed.";

      showToast(
        "Assistant request failed",
        error.message,
        "error"
      );
    } finally {
      button.disabled = false;
      button.textContent =
        "Ask repair assistant";
    }
  }

  /* ==========================================================
     COMMAND PALETTE
  ========================================================== */

  function baseCommands() {
    return [
      {
        type: "Navigation",
        title: "Open Overview",
        subtitle: "Workspace summary",
        action: () => setPage("overview"),
      },
      {
        type: "Navigation",
        title: "Open Repair Runs",
        subtitle: "Inspect trajectories and patches",
        action: () => setPage("runs"),
      },
      {
        type: "Navigation",
        title: "Open Infrastructure",
        subtitle: "View local service health",
        action: () => setPage("infrastructure"),
      },
      {
        type: "Navigation",
        title: "Open Model Connections",
        subtitle: "Configure inference provider",
        action: () => setPage("connections"),
      },
      {
        type: "Navigation",
        title: "Open Repair Assistant",
        subtitle: "Ask the active model for suggestions",
        action: () => setPage("assistant"),
      },
      {
        type: "Action",
        title: "Refresh all data",
        subtitle: "Reload runs and infrastructure status",
        action: refreshAll,
      },
      {
        type: "Action",
        title: "Toggle theme",
        subtitle: "Switch light and dark mode",
        action: toggleTheme,
      },
    ];
  }

  function getCommandItems(query = "") {
    const commands =
      baseCommands().map((command) => ({
        ...command,
        searchable:
          `${command.title} ${command.subtitle}`.toLowerCase(),
      }));

    const runCommands =
      state.runs.map((run) => ({
        type: "Run",
        title: getInstanceId(run),
        subtitle:
          `${getModel(run)} · ${getStatus(run)} · ${getRunId(run)}`,
        searchable:
          `${getInstanceId(run)} ${getModel(run)} ${getRunId(run)} ${safeJson(run.tool_calls)} ${safeJson(run.messages)}`.toLowerCase(),
        action: () => {
          setPage("runs");
          selectRun(run);
        },
      }));

    const all = [
      ...commands,
      ...runCommands,
      ...Object.entries(PAGE_NAMES).filter(([key]) => !["overview", "runs", "infrastructure", "connections", "assistant"].includes(key)).map(([key,title]) => ({type:"Navigation",title:`Open ${title}`,subtitle:"Control center",searchable:title.toLowerCase(),action:() => setPage(key)})),
      ...(window.SWERLControl?.commands() || []),
    ];

    if (!query) {
      return all.slice(0, 14);
    }

    const normalized =
      query.toLowerCase();

    return all
      .filter(
        (item) =>
          item.searchable.includes(normalized)
      )
      .slice(0, 30);
  }

  function openCommandPalette() {
    const palette =
      $("#commandPalette");

    palette.classList.remove("hidden");

    $("#commandInput").value = "";
    state.commandIndex = 0;

    renderCommandResults();

    requestAnimationFrame(() => {
      $("#commandInput").focus();
    });
  }

  function closeCommandPalette() {
    $("#commandPalette")
      .classList.add("hidden");
    $("#commandTrigger").focus();
  }

  function renderCommandResults() {
    const query =
      $("#commandInput").value.trim();

    const items =
      getCommandItems(query);

    const container =
      $("#commandResults");

    if (!items.length) {
      container.innerHTML = `
        <div class="empty-state compact-empty">
          No matching commands or runs.
        </div>
      `;

      return;
    }

    if (state.commandIndex >= items.length) {
      state.commandIndex = 0;
    }

    container.innerHTML =
      items.map((item, index) => `
        <button
          class="command-result ${
            index === state.commandIndex
              ? "is-active"
              : ""
          }"
          data-command-index="${index}"
        >
          <span class="command-result-type">
            ${escapeHtml(item.type)}
          </span>

          <span class="command-result-main">
            <span class="command-result-title">
              ${escapeHtml(item.title)}
            </span>

            <span class="command-result-subtitle">
              ${escapeHtml(item.subtitle)}
            </span>
          </span>
        </button>
      `).join("");

    $$("[data-command-index]", container)
      .forEach((button) => {
        button.addEventListener(
          "mouseenter",
          () => {
            state.commandIndex =
              Number(button.dataset.commandIndex);

            renderCommandResults();
          }
        );

        button.addEventListener(
          "click",
          () => {
            const target =
              items[
                Number(button.dataset.commandIndex)
              ];

            target?.action?.();
            closeCommandPalette();
          }
        );
      });
  }

  function executeActiveCommand() {
    const items =
      getCommandItems(
        $("#commandInput").value.trim()
      );

    const target =
      items[state.commandIndex];

    if (target) {
      target.action();
      closeCommandPalette();
    }
  }

  /* ==========================================================
     TOAST
  ========================================================== */

  function showToast(
    title,
    message = "",
    type = "info"
  ) {
    const region =
      $("#toastRegion");

    const toast =
      document.createElement("div");

    toast.className =
      `toast ${type}`;

    toast.innerHTML = `
      <span class="toast-indicator"></span>

      <div class="toast-content">
        <div class="toast-title">
          ${escapeHtml(title)}
        </div>

        ${
          message
            ? `
              <div class="toast-message">
                ${escapeHtml(message)}
              </div>
            `
            : ""
        }
      </div>
    `;

    region.appendChild(toast);

    window.setTimeout(
      () => toast.remove(),
      4200
    );
  }

  /* ==========================================================
     REFRESH
  ========================================================== */

  async function refreshAll() {
    const refresh =
      $("#refreshButton");

    refresh.disabled = true;

    await Promise.allSettled([
      loadRuns(),
      loadHealth(),
      window.SWERLControl?.refresh(),
    ]);

    refresh.disabled = false;

    showToast(
      "Workspace refreshed",
      "Run and infrastructure data were reloaded.",
      "success"
    );
  }

  /* ==========================================================
     EVENTS
  ========================================================== */

  function initializeEvents() {
    $("#themeToggle")
      .addEventListener("click", toggleTheme);

    $("#refreshButton")
      .addEventListener("click", refreshAll);

    $("#runsRefreshButton")
      .addEventListener("click", loadRuns);

    $("#infraRefreshButton")
      .addEventListener("click", loadHealth);

    $("#runSearchInput")
      .addEventListener(
        "input",
        applyRunFilters
      );

    $("#runStatusFilter")
      .addEventListener(
        "change",
        applyRunFilters
      );

    $("#exportRunButton")
      .addEventListener("click", () => {
        if (!state.selectedRun) {
          return;
        }

        const source = "recorded";

        downloadJson(
          {
            source,
            run: state.selectedRun,
          },
          `${getRunId(state.selectedRun)}.json`
        );
      });

    $("#copyPatchButton")
      .addEventListener("click", () => {
        if (!state.selectedRun) {
          return;
        }

        copyText(
          getPatch(state.selectedRun),
          "Patch copied"
        );
      });

    $("#assistantPrompt")
      .addEventListener("input", () => {
        const count =
          $("#assistantPrompt").value.length;

        $("#assistantCharacterCount").textContent =
          `${count.toLocaleString()} / 12,000`;
      });

    $("#runAssistantButton")
      .addEventListener("click", runAssistant);

    $("#clearAssistantButton")
      .addEventListener("click", () => {
        $("#assistantPrompt").value = "";
        $("#assistantPrompt")
          .dispatchEvent(new Event("input"));

        state.assistantResponse = "";

        $("#assistantResponse").innerHTML = `
          <div class="empty-state">
            Enter a problem description to request model suggestions.
          </div>
        `;

        $("#assistantResponseMeta").textContent =
          "Waiting for input.";

        $("#copyAssistantButton").disabled = true;
      });

    $("#copyAssistantButton")
      .addEventListener("click", () => {
        copyText(
          state.assistantResponse,
          "Response copied"
        );
      });

    $("#commandTrigger")
      .addEventListener(
        "click",
        openCommandPalette
      );

    $("#commandInput")
      .addEventListener(
        "input",
        () => {
          state.commandIndex = 0;
          renderCommandResults();
        }
      );

    $("#commandPalette")
      .addEventListener("mousedown", (event) => {
        if (
          event.target ===
          $("#commandPalette")
        ) {
          closeCommandPalette();
        }
      });

    $("#mobileSidebarOpen")
      .addEventListener(
        "click",
        openMobileSidebar
      );

    $("#mobileSidebarClose")
      .addEventListener(
        "click",
        closeMobileSidebar
      );

    $("#sidebarBackdrop")
      .addEventListener(
        "click",
        closeMobileSidebar
      );

    document.addEventListener(
      "keydown",
      (event) => {
        const isCommandShortcut =
          (event.metaKey || event.ctrlKey) &&
          event.key.toLowerCase() === "k";

        if (isCommandShortcut) {
          event.preventDefault();

          if (
            $("#commandPalette")
              .classList
              .contains("hidden")
          ) {
            openCommandPalette();
          } else {
            closeCommandPalette();
          }

          return;
        }

        if (
          !$("#commandPalette")
            .classList
            .contains("hidden")
        ) {
          const items =
            getCommandItems(
              $("#commandInput")
                .value
                .trim()
            );

          if (event.key === "Escape") {
            closeCommandPalette();
          }

          if (event.key === "ArrowDown") {
            event.preventDefault();

            state.commandIndex =
              Math.min(
                state.commandIndex + 1,
                Math.max(0, items.length - 1)
              );

            renderCommandResults();
          }

          if (event.key === "ArrowUp") {
            event.preventDefault();

            state.commandIndex =
              Math.max(
                0,
                state.commandIndex - 1
              );

            renderCommandResults();
          }

          if (event.key === "Enter") {
            event.preventDefault();
            executeActiveCommand();
          }
        }
      }
    );
  }

  /* ==========================================================
     INITIALIZE
  ========================================================== */

  async function initialize() {
    initializeTheme();
    initializeNavigation();
    initializeRunTabs();
    initializeEvents();

    await Promise.allSettled([
      loadRuns(),
      loadHealth(),
    ]);
  }

  document.addEventListener(
    "DOMContentLoaded",
    initialize
  );
})();
