"use strict";
const $ = (id) => document.getElementById(id);
const escapeHTML = (value) => String(value ?? "").replace(/[&<>"']/g, (char) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[char]));
const number = (value) => Number(value || 0).toLocaleString();
const outcome = (run) => run.status !== "completed" ? "incomplete" : run.reward_details?.exec?.resolved === true ? "resolved" : "unresolved";
const demoRuns = [
  {
    run_id:"demo-path-normalization", instance_id:"example/path-tools-42", model:"Illustrative repair agent",
    status:"completed", steps:5, tokens_in:4280, tokens_out:916, reward:1,
    metadata:{problem_statement:"Normalize relative paths before resolving repository files. Repeated separators and leading ./ segments should refer to the same file."},
    final_patch:"diff --git a/paths.py b/paths.py\n--- a/paths.py\n+++ b/paths.py\n@@ -1,4 +1,5 @@\n+from pathlib import PurePosixPath\n \n def normalize_path(path):\n-    return path.strip()\n+    return str(PurePosixPath(path.strip()))\n",
    tool_calls:[
      {tool:"repo_map",arguments:{path:"."},result:{output:"paths.py\ntests/test_paths.py",ok:true},duration_s:0.12},
      {tool:"file_edit",arguments:{path:"paths.py"},result:{output:"Read normalize_path. Leading ./ segments remain in returned paths.",ok:true},duration_s:0.08},
      {tool:"file_edit",arguments:{path:"paths.py"},result:{output:"Added PurePosixPath normalization.",ok:true},duration_s:0.1},
      {tool:"test_run",arguments:{tests:["tests/test_paths.py"]},result:{output:"3 passed in 0.24s",ok:true},duration_s:0.24},
      {tool:"finish",arguments:{},result:{output:"Patch ready for evaluation.",ok:true},duration_s:0.01}
    ],
    reward_details:{exec:{resolved:true,f2p_passed:["test_relative_path","test_repeated_separator"],f2p_failed:[],p2p_passed:["test_plain_path"],p2p_failed:[]}},
    messages:[{role:"user",content:"Normalize relative repository paths."},{role:"assistant",content:"Inspect the helper, apply a focused patch, and run the path tests."}]
  },
  {
    run_id:"demo-cache-expiry",instance_id:"example/cache-service-17",model:"Illustrative repair agent",
    status:"completed",steps:3,tokens_in:2104,tokens_out:502,reward:0,
    metadata:{problem_statement:"An expired cache entry is returned at the exact expiry boundary. The attempted repair fixes the boundary but breaks entries without a time-to-live."},
    final_patch:"diff --git a/cache.py b/cache.py\n--- a/cache.py\n+++ b/cache.py\n@@ -8,2 +8,2 @@\n-    if now > expires_at:\n+    if now >= expires_at:\n         return None\n",
    tool_calls:[{tool:"test_run",arguments:{},result:{ok:false,output:"test_expiry_boundary PASSED\ntest_no_ttl FAILED: TypeError comparing float and None"},duration_s:0.32}],
    reward_details:{exec:{resolved:false,f2p_passed:["test_expiry_boundary"],f2p_failed:[],p2p_passed:[],p2p_failed:["test_no_ttl"]}},
    messages:[{role:"assistant",content:"The boundary fix needs an explicit guard for entries without expiry."}]
  }
];
let recorded = [], mode = "live", selected = null, tab = "patch", requestId = 0;
const currentRuns = () => mode === "demo" ? demoRuns : recorded;
function setMode(next) {
  mode = next;
  $("live").setAttribute("aria-pressed", String(mode === "live"));
  $("demo").setAttribute("aria-pressed", String(mode === "demo"));
  selected = null;
  $("notice").textContent = mode === "demo"
    ? "Illustrative demo: synthetic examples for presentation. No model is running and no benchmark result is claimed."
    : "Recorded trajectories from your local outputs directory. Refresh to load new runs.";
  render();
}
function render() {
  const all = currentRuns();
  const completed = all.filter((run) => run.status === "completed");
  const resolved = all.filter((run) => outcome(run) === "resolved").length;
  const stats = [
    ["Repair runs", number(all.length), mode === "demo" ? "Illustrative examples" : "Local trajectory files"],
    ["Resolved", number(resolved), "All required tests passed"],
    ["Resolution rate", completed.length ? Math.round(resolved / completed.length * 100) + "%" : "—", "Among completed runs"],
    ["Tokens recorded", number(all.reduce((sum,run) => sum + Number(run.tokens_in || 0) + Number(run.tokens_out || 0),0)), "Input and output combined"]
  ];
  $("metrics").innerHTML = stats.map(([label,value,note]) => '<div class="metric"><span>'+label+'</span><strong>'+value+'</strong><small>'+note+'</small></div>').join("");
  const query = $("search").value.toLowerCase().trim();
  const filtered = all.filter((run) => (run.instance_id+" "+run.run_id+" "+run.model).toLowerCase().includes(query) && ($("filter").value === "all" || outcome(run) === $("filter").value));
  $("count").textContent = filtered.length + " shown";
  if (!filtered.includes(selected)) selected = filtered[0] || null;
  $("runs").replaceChildren();
  for (const run of filtered) {
    const button = document.createElement("button");
    button.className = "run" + (run === selected ? " selected" : "");
    button.setAttribute("aria-pressed", String(run === selected));
    button.innerHTML = '<div class="run-meta"><span>'+escapeHTML(mode === "demo" ? "Demo trajectory" : run.run_id.slice(0,10))+'</span><span class="badge '+outcome(run)+'">'+outcome(run)+'</span></div><strong>'+escapeHTML(run.instance_id)+'</strong><small>'+number(run.steps)+' steps · '+number(Number(run.tokens_in || 0)+Number(run.tokens_out || 0))+' tokens</small>';
    button.onclick = () => {selected = run; render();};
    $("runs").append(button);
  }
  if (!filtered.length) $("runs").innerHTML = '<p class="muted">No matching runs. Try another filter or explore the demo.</p>';
  renderDetail();
}
function renderDetail() {
  $("download").disabled = !selected;
  if (!selected) {
    $("instance").textContent = "Select a repair run";
    $("issue").textContent = "Choose a recorded trajectory or use Explore demo to present the workflow without an LLM.";
    $("outcome").textContent = "No selection";
    $("outcome").className = "badge";
    $("pipeline").replaceChildren();
    $("content").innerHTML = '<div class="empty"><h3>The repair, in full context.</h3><p>Recorded runs appear here after a rollout. Explore demo contains two illustrative examples.</p></div>';
    $("run-label").textContent = "Trajectory inspector";
    $("run-footer").textContent = "No model connection required to inspect saved runs.";
    return;
  }
  const run = selected, state = outcome(run);
  $("instance").textContent = run.instance_id;
  $("run-label").textContent = mode === "demo" ? "Demo trajectory / synthetic data" : "Recorded trajectory / " + run.run_id.slice(0,12);
  $("issue").textContent = run.metadata?.problem_statement || "No issue description was stored with this trajectory.";
  $("outcome").textContent = state;
  $("outcome").className = "badge " + state;
  $("pipeline").innerHTML = [run.metadata?.setup_failed ? "Setup failed" : "Trajectory recorded", (run.tool_calls || []).length+" tool calls", run.final_patch ? "Patch captured" : "No patch", run.reward_details?.exec ? "Evaluation recorded" : "No evaluation"].map((text) => "<span>"+escapeHTML(text)+"</span>").join("");
  $("run-footer").textContent = run.model + "  /  " + number(run.steps) + " steps  /  Reward: " + (run.reward ?? "not recorded");
  renderContent();
}
function renderContent() {
  const run = selected;
  if (!run) return;
  if (tab === "patch") {
    $("content").innerHTML = '<pre class="code" aria-label="Unified patch">'+(run.final_patch || "No patch recorded.").split("\n").map((line) => '<span class="'+(line.startsWith("+")?"add":line.startsWith("-")?"remove":line.startsWith("@@")?"hunk":"")+'">'+escapeHTML(line)+'</span>').join("")+'</pre>';
  } else if (tab === "trace") {
    $("content").innerHTML = '<div class="trace">'+((run.tool_calls || []).map((call,index) => '<details'+(index === 0?" open":"")+'><summary>'+ (index+1)+'. '+escapeHTML(call.tool)+'<small>'+escapeHTML(call.result?.ok === false ? "Failed" : "Recorded")+' · '+Number(call.duration_s || 0).toFixed(2)+'s</small></summary><pre>'+escapeHTML(JSON.stringify(call.arguments,null,2))+'</pre><pre>'+escapeHTML(call.result?.output || JSON.stringify(call.result,null,2))+'</pre></details>').join("") || '<p>No tool calls recorded.</p>')+'</div>';
  } else if (tab === "tests") {
    const evidence = run.reward_details?.exec;
    $("content").innerHTML = '<div class="trace">'+(!evidence ? '<p>No test evidence recorded.</p>' : [["Fail to pass","f2p"],["Pass to pass","p2p"]].map(([title,key]) => '<h3>'+title+'</h3>'+["passed","failed"].map((status) => (evidence[key+"_"+status] || []).map((name) => '<div class="test-row"><span>'+escapeHTML(name)+'</span><span class="badge '+(status === "passed"?"resolved":"unresolved")+'">'+status+'</span></div>').join("")).join("")).join(""))+'</div>';
  } else {
    $("content").innerHTML = '<div class="trace">'+((run.messages || []).map((message) => '<details><summary>'+escapeHTML(message.role)+'</summary><pre>'+escapeHTML(message.content)+'</pre></details>').join("") || '<p>No conversation recorded.</p>')+'</div>';
  }
}
function activateTab(button) {
  tab = button.dataset.tab;
  document.querySelectorAll("[data-tab]").forEach((item) => {item.setAttribute("aria-selected",String(item === button)); item.tabIndex = item === button ? 0 : -1;});
  $("content").setAttribute("aria-labelledby", button.id);
  renderContent();
}
async function getJSON(url) {
  const response = await fetch(url, {signal:AbortSignal.timeout(10000)});
  if (!response.ok) throw new Error("HTTP " + response.status);
  return response.json();
}
async function load() {
  const id = ++requestId;
  $("refresh").disabled = true;
  const results = await Promise.allSettled([getJSON("/api/runs"),getJSON("/api/health")]);
  if (id !== requestId) return;
  const [runs, health] = results;
  if (runs.status === "fulfilled" && Array.isArray(runs.value)) {
    const previousId = selected?.run_id;
    recorded = runs.value;
    if (mode === "live") {
      selected = recorded.find((run) => run.run_id === previousId) || null;
      $("notice").textContent = recorded.length ? "Recorded trajectories loaded. Select a run to inspect its evidence." : "No recorded runs yet. Explore demo for a guided example without a model connection.";
    }
  } else if (mode === "live") $("notice").textContent = "Could not refresh recorded runs. Check the dashboard server, then retry. Previously loaded data is retained.";
  const names = {api:"Dashboard API",docker:"Docker",postgresql:"PostgreSQL",redis:"Redis",minio:"Object storage",model_endpoint:"Model endpoint"};
  $("services").innerHTML = Object.entries(names).map(([key,name]) => {
    const value = health.status === "fulfilled" ? health.value[key] : "unknown";
    return '<div class="service"><strong>'+name+'</strong><small><i class="'+(value === "available"?"ok":"")+'"></i>'+escapeHTML(value || "unknown")+'</small></div>';
  }).join("");
  $("checked").textContent = health.status === "fulfilled" ? "Checked "+new Date().toLocaleTimeString() : "Health check unavailable";
  $("refresh").disabled = false;
  render();
}
$("live").onclick = () => setMode("live");
$("demo").onclick = () => setMode("demo");
$("empty-demo").onclick = () => setMode("demo");
$("search").oninput = render;
$("filter").onchange = render;
$("refresh").onclick = load;
document.querySelectorAll("[data-tab]").forEach((button,index,buttons) => {
  button.onclick = () => activateTab(button);
  button.onkeydown = (event) => {
    if (!["ArrowLeft","ArrowRight","Home","End"].includes(event.key)) return;
    event.preventDefault();
    const next = event.key === "Home" ? 0 : event.key === "End" ? buttons.length-1 : (index+(event.key === "ArrowRight"?1:-1)+buttons.length)%buttons.length;
    buttons[next].focus(); activateTab(buttons[next]);
  };
});
function setTheme(dark) {
  document.body.classList.toggle("dark", dark);
  $("theme").textContent = dark ? "Light theme" : "Dark theme";
  $("theme").setAttribute("aria-label", dark ? "Use light theme" : "Use dark theme");
  try {localStorage.setItem("swe-rl-theme",dark?"dark":"light");} catch {}
}
try {setTheme(localStorage.getItem("swe-rl-theme") === "dark");} catch {}
$("theme").onclick = () => setTheme(!document.body.classList.contains("dark"));
$("download").onclick = () => {
  if (!selected) return;
  const blob = new Blob([JSON.stringify({data_source:mode === "demo" ? "synthetic_demo" : "recorded",...selected},null,2)],{type:"application/json"});
  const url = URL.createObjectURL(blob), link = document.createElement("a");
  link.href = url; link.download = (mode === "demo" ? "demo-" : "run-") + String(selected.run_id).replace(/[^a-zA-Z0-9_-]/g,"_") + ".json";
  link.click(); setTimeout(() => URL.revokeObjectURL(url),1000);
};
load();
