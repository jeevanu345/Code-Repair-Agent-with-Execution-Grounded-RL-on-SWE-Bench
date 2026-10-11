/* Operational screens extending the supplied enterprise console. */
(() => {
  "use strict";
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));
  const esc = value => String(value ?? "").replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;").replaceAll('"',"&quot;").replaceAll("'","&#39;");
  const json = value => JSON.stringify(value ?? {}, null, 2);
  const api = (...args) => window.SWERLAPI.request(...args);
  const state = {jobs:[],experiments:[],artifacts:[],checkpoints:[],runs:[],stream:null,selected:null,events:[],diffMode:"unified",diffFile:"",ignoreWhitespace:false,compactContext:false,loading:false,selectedJob:null};
  const terminal = status => ["completed","failed","cancelled","resolved","unresolved"].includes(String(status).toLowerCase());
  const idOf = row => row?.run_id || row?.id || row?.trajectory_id;
  const num = (value, digits=0) => value == null || !Number.isFinite(Number(value)) ? "—" : Number(value).toLocaleString(undefined,{maximumFractionDigits:digits});
  const money = value => value == null || !Number.isFinite(Number(value)) ? "—" : `$${Number(value).toFixed(4)}`;
  const runtime = row => row.duration_s ?? row.runtime_s ?? (row.started_at ? ((row.finished_at ? Date.parse(row.finished_at) : terminal(row.status) ? NaN : Date.now())-Date.parse(row.started_at))/1000 : null);
  const cost = row => (typeof row.cost === "number" ? row.cost : undefined) ?? row.cost?.usd ?? row.cost?.total_usd ?? row.cost_usd ?? row.metadata?.cost_usd ?? row.metadata?.cost?.usd ?? row.metadata?.cost?.total_usd;
  const resolved = row => row.resolved ?? row.reward_details?.exec?.resolved ?? row.reward_details?.resolved ?? (row.reward == null ? null : Number(row.reward) === 1);
  const avg = values => {const found=values.filter(v => v != null && Number.isFinite(Number(v))).map(Number);return found.length ? found.reduce((a,b)=>a+b,0)/found.length : null;};
  const badge = status => `<span class="status-badge ${["completed","resolved"].includes(String(status).toLowerCase()) ? "success" : ["failed","unresolved"].includes(String(status).toLowerCase()) ? "danger" : ["running","preparing","testing","evaluating"].includes(String(status).toLowerCase()) ? "info" : "neutral"}">${esc(status || "unknown")}</span>`;
  const empty = text => `<div class="empty-state">${esc(text)}</div>`;
  const button = (label, action, rowId, extra="") => `<button class="button button-secondary button-sm" data-job-action="${action}" data-job-id="${esc(rowId)}" ${extra}>${label}</button>`;
  const navGroups = [
    ["Workspace",[["overview","Overview"]]],
    ["Agent",[["new-run","New run"],["runs","Repair runs"],["experiments","Experiments"],["compare","Compare"]]],
    ["Benchmarks",[["benchmarks","SWE-bench"],["evaluations","Evaluations"],["reports","Reports"]]],
    ["Training",[["training","Training jobs"],["checkpoints","Checkpoints"],["replay","Replay buffer"],["reward","Reward analysis"]]],
    ["Models",[["connections","Connections"],["registry","Model registry"],["prompts","Prompt versions"],["assistant","Repair assistant"]]],
    ["Infrastructure",[["infrastructure","Services"],["workers","Workers"],["containers","Containers"],["metrics","Metrics"]]],
    ["Artifacts",[["artifacts","Trajectories, patches & logs"],["datasets","Datasets"]]],
    ["Workspace settings",[["settings","Settings"]]],
  ];
  function section(id, title, description, content="") {return `<section class="page" id="page-${id}"><div class="page-header compact-page-header"><div><h1>${title}</h1><p>${description}</p></div><button class="button button-secondary" data-refresh="${id}">Refresh</button></div><div class="control-content" id="content-${id}">${content || empty("Loading recorded data…")}</div></section>`;}
  function field(name,label,value="",type="text",attrs="") {return `<label class="form-field"><span class="form-label">${label}</span><input class="input-control" name="${name}" type="${type}" value="${esc(value)}" ${attrs}></label>`;}
  function select(name,label,options,value="") {return `<label class="form-field"><span class="form-label">${label}</span><select class="select-control" name="${name}">${options.map(([v,l])=>`<option value="${esc(v)}" ${v===value?"selected":""}>${esc(l)}</option>`).join("")}</select></label>`;}
  function datasets() {return [["verified","SWE-bench Verified"],["lite","SWE-bench Lite"],["full","SWE-bench Full"]];}
  function runForm() {
    return `<form id="newRunForm" class="panel launch-form"><div class="panel-header"><div><h2>Repair configuration</h2><p>Each candidate creates an independent trajectory in an isolated repository.</p></div></div><div class="form-grid control-form-grid">
      ${select("dataset","Dataset",datasets(),"verified")}
      <label class="form-field"><span class="form-label">Instance IDs</span><textarea class="input-control" name="instances" required rows="3" placeholder="django__django-15252\nOne instance per line or comma"></textarea></label>
      ${field("model","Model (blank uses active connection)")}${select("provider","Provider",[["","Active connection"]])}
      ${select("strategy","Agent strategy",[["standard","Standard ReAct"],["plan_execute","Plan → Execute"],["test_driven","Test driven"],["minimal_patch","Minimal patch"],["high_exploration","High exploration"],["low_cost","Low cost"],["explorer_reviewer","Explorer → Implementer → Reviewer"]])}
      ${field("prompt_version","Prompt version ID (optional)")}${field("checkpoint","Checkpoint reference (optional)")}${field("dataset_revision","Dataset revision","main")}${select("selection","Patch selection",[["execution-score","Execution score"],["minimal-patch","Minimal patch"],["regression-safe","Regression safe"],["majority-behavior","Majority behavior"],["critic-ranking","Critic ranking (review from Experiments)"]])}${select("candidates","Candidates per instance",[1,2,4,8,16].map(n=>[String(n),String(n)]),"1")}
      ${field("temperature","Temperature",0.2,"number",'min="0" max="2" step="0.05"')}${field("top_p","Top p",0.95,"number",'min="0" max="1" step="0.01"')}
      ${field("seed","Seed",42,"number",'min="0" step="1"')}${field("max_steps","Maximum steps",30,"number",'min="1" max="200" required')}
      ${field("test_budget","Test budget",5,"number",'min="1" max="50" required')}${field("max_cost","Maximum cost per candidate (USD)",1,"number",'min="0.001" step="0.01" required')}
    </div><fieldset class="control-fieldset"><legend>Sandbox</legend><div class="control-form-grid">${field("cpus","CPU cores",4,"number",'min="1" max="64"')}${field("memory","Memory (GB)",8,"number",'min="1" max="256"')}${field("timeout","Timeout (seconds)",1200,"number",'min="60" max="14400"')}</div><label class="check-control"><input name="network_disabled" type="checkbox" checked> Disable network after setup</label></fieldset>
    <fieldset class="control-fieldset"><legend>Reward weights</legend><div class="control-form-grid">${field("binary_weight","Execution",1,"number",'min="0" step="0.1"')}${field("shaped_weight","Shaped",0.2,"number",'min="0" step="0.1"')}${field("prm_weight","Process reward model weight",0,"number",'min="0" max="1" step="0.1"')}${field("prm_model","PRM model ID (when weight is positive)")}</div></fieldset>
    <label class="check-control minimize-option"><input name="minimize_patch" type="checkbox"> Minimize successful patches and rerun tests</label><div class="form-actions"><span id="newRunStatus" class="form-help" role="status">Configuration and its fingerprint are saved before execution.</span><button class="button button-primary" type="submit">Start repair</button></div></form>`;
  }
  function evaluationForm() {return `<form id="evaluationForm" class="panel launch-form"><div class="panel-header"><div><h2>New benchmark evaluation</h2><p>Generate candidates and evaluate repair evidence across a dataset slice.</p></div></div><div class="control-form-grid form-grid">${select("dataset","Dataset",[...datasets(),["multilingual","SWE-bench Multilingual (import predictions)"],["multimodal","SWE-bench Multimodal (import predictions)"]],"verified")}${field("model","Model / served checkpoint","","text",'id="evaluationModel"')}${field("predictions_path","Imported predictions JSONL (for harness-only grading)")}${field("max_instances","Instances",10,"number",'min="1" max="2294" required')}${field("parallelism","Workers",2,"number",'min="1" max="64" required')}${select("candidates","Candidates per issue",[1,2,4,8,16].map(n=>[String(n),String(n)]),"1")}</div><div class="form-actions"><label class="check-control"><input name="invoke_harness" type="checkbox" checked> Grade with official SWE-bench harness</label><button type="submit" class="button button-primary">Start evaluation</button></div><div class="form-help" id="evaluationStatus" role="status"></div></form><div id="benchmarkJobs"></div>`;}
  function trainingForm() {return `<form id="trainingForm" class="panel launch-form"><div class="panel-header"><div><h2>New training job</h2><p>Training runs on the configured machine. GPU capacity and compatible dependencies are required.</p></div></div><div class="control-form-grid form-grid">${select("algorithm","Algorithm",[["sft","SFT"],["dpo","DPO"],["grpo","GRPO"]],"grpo")}${field("model","Base model","Qwen/Qwen2.5-Coder-7B-Instruct")}${field("dataset_path","Dataset path (optional)")}${field("learning_rate","Learning rate",0.000001,"number",'min="0.000000001" max="0.1" step="any" required')}${field("max_instances","Maximum instances",100,"number",'min="1" max="2294" required')}</div><div class="form-actions"><label class="check-control"><input name="allow_heavy_training" type="checkbox" required> Start model training on this machine</label><button type="submit" class="button button-primary">Start training</button></div><div id="trainingStatus" class="form-help" role="status"></div></form><div id="trainingJobs"></div>`;}
  function buildScreens() {
    const nav=$(".sidebar-nav");
    nav.innerHTML=navGroups.map(([name,items])=>`<div class="nav-group"><div class="nav-group-label">${name}</div>${items.map(([id,title])=>`<button class="nav-item ${id==="overview"?"is-active":""}" data-page="${id}"><span class="nav-symbol" aria-hidden="true">${id==="runs"?"≡":id==="new-run"?"+":id==="overview"?"▦":"·"}</span><span>${title}</span>${id==="runs"?'<span class="nav-counter" id="sidebarRunCount">0</span>':id==="infrastructure"?'<span id="infraNavDot" class="nav-status-dot"></span>':""}</button>`).join("")}</div>`).join("")+`<a class="nav-item" href="/api/docs" target="_blank" rel="noopener noreferrer">API reference ↗</a>`;
    $(".brand-subtitle").textContent="Control center";
    $("main").insertAdjacentHTML("beforeend",[
      section("new-run","New repair run","Launch reproducible agent repairs and independent candidates.",runForm()),
      section("experiments","Experiments","Versioned configurations, fingerprints, candidates and outcomes."),
      section("compare","Compare experiments","Compare the same instances across models, prompts, strategies and reward configurations."),
      section("benchmarks","SWE-bench","Containerized benchmark grading with recorded progress and artifacts.",evaluationForm()),
      section("evaluations","Evaluations","Benchmark jobs and their completed results."),section("reports","Reports","Recorded benchmark reports and evaluation artifacts."),
      section("training","Training jobs","SFT, DPO and execution grounded GRPO.",trainingForm()),section("checkpoints","Checkpoints","Registered model checkpoints and their training provenance."),
      section("replay","Replay buffer","Trajectory evidence available for training and researcher review."),section("reward","Reward analysis","Execution reward, shaped components and their recorded relationship."),
      section("registry","Model registry","Model connections, capabilities and measured provider checks."),section("prompts","Prompt versions","Versioned agent instructions for reproducible experiments."),
      section("workers","Workers","Local scheduler and worker state."),section("containers","Containers","Observed Docker sandboxes and resource usage."),section("metrics","Metrics","Operational and agent metrics from the running service."),
      section("artifacts","Artifacts","Browse persisted trajectories, patches, logs and evaluation output."),section("datasets","Datasets","Supported benchmark variants and exported training data."),section("settings","Settings","Workspace preferences, saved views and local model settings."),
    ].join(""));
    $("#page-overview").insertAdjacentHTML("beforeend",'<div id="researchMetrics" class="metric-grid"></div><div id="researchCharts" class="research-charts"></div><section class="panel"><div class="panel-header"><div><h2>Worker queue</h2><p>Durable jobs and active execution.</p></div><button class="button button-primary button-sm" data-go="new-run">New run</button></div><div id="overviewQueue"></div></section>');
    $(".run-list-toolbar").insertAdjacentHTML("afterend",'<div class="saved-view-toolbar"><select id="savedViewSelect" class="select-control" aria-label="Saved run views"><option value="">Saved views</option></select><button id="saveViewButton" class="button button-ghost button-sm">Save view</button><button id="deleteViewButton" class="button button-ghost button-sm" aria-label="Delete selected saved view">×</button></div>');
    const tabs=$("#runInspector .tabs");
    [["costs","Cost & tokens"],["reward","Reward"],["failure","Failure analysis"],["context","Code context"]].forEach(([id,title])=>{
      tabs.insertAdjacentHTML("beforeend",`<button role="tab" class="tab-button" id="tab-${id}" aria-controls="run-panel-${id}" aria-selected="false" tabindex="-1" data-run-tab="${id}">${title}</button>`);
      tabs.parentElement.insertAdjacentHTML("beforeend",`<div class="run-tab-panel" role="tabpanel" id="run-panel-${id}" aria-labelledby="tab-${id}" data-run-panel="${id}"><div id="run-${id}" class="evidence-panel"></div></div>`);
    });
    $$("[data-run-tab]").forEach(tab=>{const name=tab.dataset.runTab;tab.id ||= `tab-${name}`;tab.setAttribute("aria-controls",`run-panel-${name}`);tab.tabIndex=tab.getAttribute("aria-selected")==="true"?0:-1;});
    $$("[data-run-panel]").forEach(panel=>{panel.id=`run-panel-${panel.dataset.runPanel}`;panel.setAttribute("role","tabpanel");panel.setAttribute("aria-labelledby",`tab-${panel.dataset.runPanel}`);});
    const content=$("#runInspectorContent"), metadata=$(".run-meta-grid",content);
    const grid=document.createElement("div");grid.className="inspector-grid";
    grid.innerHTML='<aside class="inspector-files"><h3>Changed files</h3><div id="inspectorFileTree"></div></aside><div class="inspector-center"></div><aside class="inspector-info"></aside>';
    const center=$(".inspector-center",grid),info=$(".inspector-info",grid);
    [$("#executionPipeline"), tabs, ...$$("[data-run-panel]",content)].forEach(el=>center.appendChild(el));
    info.appendChild(metadata);
    metadata.insertAdjacentHTML("beforeend",'<div class="run-meta-item"><span>Runtime</span><strong id="inspectorRuntime">—</strong></div><div class="run-meta-item"><span>Cost</span><strong id="inspectorCost">—</strong></div>');
    content.appendChild(grid);
    $("#inspectorIssue").insertAdjacentHTML("afterend",'<div id="runAnnotation" class="annotation-controls"></div>');
  }
  function table(headers, rows, noData="No records yet.") {return `<div class="table-wrapper"><table class="data-table"><thead><tr>${headers.map(h=>`<th scope="col">${h}</th>`).join("")}</tr></thead><tbody>${rows.length?rows.map(cells=>`<tr>${cells.map(c=>`<td>${c}</td>`).join("")}</tr>`).join(""):`<tr><td colspan="${headers.length}">${empty(noData)}</td></tr>`}</tbody></table></div>`;}
  function jobActions(row) {
    const id=idOf(row);const status=String(row.status).toLowerCase();
    if (terminal(status)) return `${button("Retry","retry",id)} ${button("Requeue","requeue",id)}`;
    if (status === "queued") return `${button("Cancel","cancel",id)} <label class="queue-priority">Priority <input type="number" value="${row.priority||0}" min="-100" max="100" data-priority="${esc(id)}" aria-label="Priority for ${esc(id)}"></label>`;
    return `${["preparing","running"].includes(status)?button(row.paused?"Resume":"Pause",row.paused?"resume":"pause",id):""} ${button("Cancel","cancel",id)} ${row.kind==="run" && ["preparing","running"].includes(status)?button("Stop & evaluate","stop-evaluate",id):""}`;
  }
  function jobsTable(rows) {return table(["Job / instance","Kind","State","Progress","Updated","Actions"],rows.map(row=>[
    `<button class="link-button mono" data-inspect="${esc(idOf(row))}">${esc(row.config?.instance_id || row.instance_id || idOf(row))}</button><small class="table-subtitle mono">${esc(idOf(row))}</small>`,esc(row.kind || "run"),badge(row.paused?"paused":row.status),progress(row),esc(new Date(row.updated_at||row.created_at).toLocaleString()),jobActions(row)
  ]),"The queue is empty. Launch a repair, evaluation or training job.");}
  function progress(row) {
    const result=row.result || {};const complete=result.completed ?? result.progress?.completed;const total=result.total ?? result.progress?.total;
    return complete!=null && total ? `<progress value="${complete}" max="${total}"></progress> ${num(complete)} / ${num(total)}` : esc(row.error || result.stage || "—");
  }
  async function refresh() {
    if(state.loading)return;state.loading=true;
    const targets=[["jobs","/api/jobs"],["experiments","/api/experiments"],["artifacts","/api/artifacts"],["checkpoints","/api/checkpoints"]];
    const results=await Promise.allSettled(targets.map(([,url])=>api(url)));
    results.forEach((result,index)=>{const key=targets[index][0];if(result.status==="fulfilled"){state[key]=Array.isArray(result.value)?result.value:result.value?.[key]||[];delete state[`${key}Error`];}else state[`${key}Error`]=result.reason.message;});
    state.loading=false;renderOverview();const page=window.SWERL?.getState().page;if(page)await renderPage(page);
    if(state.selectedJob && ["training","evaluations"].includes(page)) {const updated=state.jobs.find(j=>idOf(j)===idOf(state.selectedJob));if(updated){state.selectedJob=updated;await renderJobDetail(updated);}}
  }
  function metricsCard(label,value,note="") {return `<article class="metric-card"><span class="metric-label">${label}</span><div class="metric-value">${value}</div><div class="metric-footnote">${note}</div></article>`;}
  function barChart(title,values,unit="") {
    if(!values.length)return `<section class="panel research-chart"><div class="panel-header"><h2>${title}</h2></div>${empty("No completed evidence available.")}</section>`;
    const max=Math.max(...values.map(v=>v[1]),1);
    return `<section class="panel research-chart"><div class="panel-header"><h2>${title}</h2></div><div class="bar-chart" role="img" aria-label="${esc(title)}">${values.slice(0,10).map(([label,value])=>`<div class="bar-row"><span title="${esc(label)}">${esc(label)}</span><div class="bar-track"><div style="width:${Math.max(0,value/max*100)}%"></div></div><strong>${num(value,1)}${unit}</strong></div>`).join("")}</div></section>`;
  }
  function groups(rows,key) {const found=new Map();rows.forEach(r=>{const name=key(r);const list=found.get(name)||[];list.push(r);found.set(name,list);});return [...found.entries()];}
  function renderOverview() {
    const runs=state.runs.filter(r=>(!r.kind||r.kind==="run")&&terminal(r.status)&&resolved(r)!=null);const solved=runs.filter(r=>resolved(r));
    const candidateGroups=groups(runs,r=>`${r.experiment_id || r.metadata?.experiment_id || "legacy"}/${r.instance_id}`);
    const groups4=candidateGroups.filter(([,rs])=>rs.length>=4);
    const totalCost=runs.map(cost).filter(v=>v!=null).reduce((a,b)=>a+Number(b),0);
    $("#researchMetrics").innerHTML=metricsCard("Pass@4",groups4.length?`${num(groups4.reduce((sum,[,rs])=>sum+passAtK(rs.length,rs.filter(resolved).length,4),0)/groups4.length*100,1)}%`:"—",`${groups4.length} instance groups with ≥4 candidates`)+metricsCard("Average cost",money(avg(runs.map(cost))),"Recorded cost per completed candidate")+metricsCard("Average runtime",runtimeValue(avg(runs.map(runtime))),"Recorded start to finish")+metricsCard("Cost / resolved instance",solved.length&&runs.length&&runs.every(r=>cost(r)!=null)?money(totalCost/solved.length):"—","Requires cost coverage for every completed candidate")+metricsCard("Repository setup failures",num(runs.filter(r=>(r.failure_analysis?.category||r.metadata?.failure_analysis?.category)==="repository_setup").length),"Recorded failure classifications")+metricsCard("Agent failures",num(runs.filter(r=>["agent_error","agent_loop_exhausted"].includes(r.failure_analysis?.category||r.metadata?.failure_analysis?.category)).length),"Recorded agent errors and exhausted loops")+metricsCard("Evaluation failures",num(runs.filter(r=>["evaluation_setup","test_unavailable"].includes(r.failure_analysis?.category||r.metadata?.failure_analysis?.category)).length),"Recorded evidence failures")+metricsCard("Tokens / resolved instance",solved.length?num(runs.reduce((sum,r)=>sum+(r.tokens_in||r.input_tokens||0)+(r.tokens_out||r.output_tokens||0),0)/solved.length):"—","Includes all completed candidates");
    const modelGroups=groups(runs,r=>r.model||r.model_name||r.config?.model||"Unknown");
    const repoGroups=groups(runs,r=>r.repository||r.instance_id?.split("-")[0]||"Unknown");
    const failureGroups=groups(state.runs.filter(r=>terminal(r.status)&&(resolved(r)===false||r.status==="failed")),r=>r.failure_analysis?.category||r.metadata?.failure_analysis?.category||r.result?.failure_analysis?.category||"Uncategorized");
    const byDate=groups(runs,r=>(r.finished_at||r.updated_at||r.created_at||"").slice(0,10)||"Unknown").sort(([a],[b])=>a.localeCompare(b));
    $("#researchCharts").innerHTML=barChart("Resolution by model",modelGroups.map(([label,rs])=>[label,rs.filter(resolved).length/rs.length*100]),"%")+barChart("Resolution by repository",repoGroups.map(([label,rs])=>[label,rs.filter(resolved).length/rs.length*100]),"%")+barChart("Failures by category",failureGroups.map(([label,rs])=>[label,rs.length]))+barChart("Resolution over time",byDate.map(([label,rs])=>[label,rs.filter(resolved).length/rs.length*100]),"%")+scatterChart("Cost vs resolution",runs,cost,"USD")+scatterChart("Steps vs success",runs,r=>r.steps??r.n_steps,"steps")+scatterChart("Tokens vs success",runs,r=>(r.tokens_in??r.input_tokens??0)+(r.tokens_out??r.output_tokens??0),"tokens");
    $("#overviewQueue").innerHTML=state.jobsError?empty(state.jobsError):jobsTable(state.jobs.slice(0,12));
  }
  function passAtK(n,c,k) {if(n-c<k)return 1;let p=1;for(let i=0;i<k;i++)p*= (n-c-i)/(n-i);return 1-p;}
  function runtimeValue(v) {return v==null?"—":v<60?`${num(v,1)}s`:`${num(v/60,1)}m`;}
  function scatterChart(title,rows,metric,unit) {
    const values=rows.filter(r=>metric(r)!=null && Number.isFinite(Number(metric(r))));
    if(!values.length)return barChart(title,[]);
    const max=Math.max(1,...values.map(metric));
    return `<section class="panel research-chart"><div class="panel-header"><h2>${title}</h2></div><svg viewBox="0 0 440 150" role="img" aria-label="${esc(title)}"><line x1="42" y1="110" x2="420" y2="110" stroke="currentColor" opacity=".2"/><text x="5" y="45" fill="currentColor" font-size="11">Solved</text><text x="5" y="90" fill="currentColor" font-size="11">Failed</text>${values.map(r=>`<circle cx="${55+Number(metric(r))/max*350}" cy="${resolved(r)?40:85}" r="4" class="${resolved(r)?"chart-solved":"chart-failed"}"><title>${esc(r.instance_id)}: ${num(metric(r),4)} ${unit}</title></circle>`).join("")}<text x="42" y="135" fill="currentColor" font-size="11">0</text><text x="355" y="135" fill="currentColor" font-size="11">${num(max,2)} ${unit}</text></svg></section>`;
  }
  function experimentTable() {return table(["Experiment","Dataset","Configuration fingerprint","Candidates","Resolved","Inspect"],state.experiments.map(e=>{const id=e.experiment_id||e.id;const rows=state.runs.filter(r=>r.experiment_id===id||r.metadata?.experiment_id===id);return [esc(id),esc(e.config?.dataset||e.dataset||"—"),`<code title="${esc(e.fingerprint)}">${esc(e.fingerprint?.slice(0,18)||"—")}</code>`,num(e.run_ids?.length??rows.length),num(rows.filter(resolved).length),`<button class="link-button" data-experiment="${esc(id)}">View configuration</button>`];}),"No experiments yet. Start a repair to save a versioned configuration.");}
  function compareScreen() {
    const options=state.experiments.map(e=>[e.experiment_id||e.id,e.experiment_id||e.id]);
    return `<div class="panel"><form id="compareForm" class="compare-form">${select("a","Experiment A",options)}${select("b","Experiment B",options,options[1]?.[0])}<button class="button button-primary">Compare</button></form><div id="compareResult">${empty("Select two experiments to compare matched instances.")}</div></div>`;
  }
  function compareExperiments(a,b) {
    if(a===b)throw new Error("Choose two different experiments.");
    const selectRows=id=>state.runs.filter(r=>(r.experiment_id||r.metadata?.experiment_id)===id && terminal(r.status));
    const A=selectRows(a),B=selectRows(b);const map=rows=>new Map(groups(rows,r=>r.instance_id).map(([id,rs])=>[id,rs.some(resolved)]));const am=map(A),bm=map(B);const common=[...am.keys()].filter(k=>bm.has(k));
    const matchedA=A.filter(r=>common.includes(r.instance_id));const matchedB=B.filter(r=>common.includes(r.instance_id));
    const definitions=[["Resolved instances",rs=>{const m=map(rs);return m.size?`${num([...m.values()].filter(Boolean).length/m.size*100,1)}%`:"—";}],["Average steps",rs=>num(avg(rs.map(r=>r.steps??r.n_steps)),1)],["Average tokens",rs=>num(avg(rs.map(r=>(r.tokens_in||0)+(r.tokens_out||0))),1)],["Average runtime",rs=>runtimeValue(avg(rs.map(runtime)))],["Average cost",rs=>money(avg(rs.map(cost)))],["Setup failures",rs=>num(rs.filter(r=>(r.failure_analysis?.category||"").includes("setup")).length)],["Empty patches",rs=>num(rs.filter(r=>resolved(r)===false&&!r.final_patch?.trim()).length)]];
    const slices=[["Only A solved",id=>am.get(id)&&!bm.get(id)],["Only B solved",id=>!am.get(id)&&bm.get(id)],["Both solved",id=>am.get(id)&&bm.get(id)],["Neither solved",id=>!am.get(id)&&!bm.get(id)]];
    return `<div class="panel-header"><p>${common.length} matched instances; ${am.size-common.length} only in A and ${bm.size-common.length} only in B.</p></div>${table(["Metric",esc(a),esc(b)],definitions.map(([label,fn])=>[label,fn(matchedA),fn(matchedB)]))}<div class="comparison-slices">${slices.map(([label,predicate])=>`<details><summary>${label}: ${common.filter(predicate).length}</summary>${common.filter(predicate).map(id=>`<button class="link-button mono" data-inspect="${esc(idOf(A.find(r=>r.instance_id===id)))}">${esc(id)}</button>`).join("")||empty("No instances in this slice.")}</details>`).join("")}</div>`;
  }
  async function renderPage(page) {
    const target=$(`#content-${page}`);if(!target)return;
    try {
      if(page==="experiments")target.innerHTML=state.experimentsError?empty(state.experimentsError):experimentTable();
      else if(page==="compare")target.innerHTML=compareScreen();
      else if(page==="evaluations")target.innerHTML=jobsTable(state.jobs.filter(r=>r.kind==="evaluation"));
      else if(page==="benchmarks")$("#benchmarkJobs").innerHTML=jobsTable(state.jobs.filter(r=>r.kind==="evaluation"));
      else if(page==="training")$("#trainingJobs").innerHTML=jobsTable(state.jobs.filter(r=>r.kind==="training"));
      else if(page==="reports")target.innerHTML=artifactTable(state.artifacts.filter(a=>/report|evaluation|prediction/i.test(a.kind||a.path||a.name||"")));
      else if(page==="artifacts")target.innerHTML=`<div class="artifact-toolbar"><label class="form-field"><span class="form-label">Filter artifacts</span><input id="artifactSearch" class="input-control" type="search" placeholder="Path, run, experiment or SHA256"></label></div><div id="artifactTable">${state.artifactsError?empty(state.artifactsError):artifactTable(state.artifacts)}</div><pre id="artifactPreview" class="evidence-code hidden"></pre>`;
      else if(page==="checkpoints")target.innerHTML=checkpointScreen();
      else if(page==="reward")target.innerHTML=rewardOverview();
      else if(page==="replay")await renderReplay(target);
      else if(page==="prompts")await renderPrompts(target);
      else if(page==="registry")await renderRegistry(target);
      else if(["workers","containers","metrics"].includes(page))await renderInfrastructure(target,page);
      else if(page==="datasets") {
        const data=await api("/api/datasets");
        target.innerHTML=table(["Dataset","Source","State"],(Array.isArray(data)?data:data.datasets||[]).map(d=>[esc(d.name||d.id||d.variant),esc(d.path||d.dataset_name||d.source||"—"),esc(d.harness_only?"Import predictions for official grading":d.status||d.description||"Agent supported")]),"No dataset records available.")+`<form id="datasetSliceForm" class="panel registry-form">${select("dataset","Dataset",datasets(),"verified")}${field("repo","Repository contains")}${field("language","Language","python")}${select("difficulty","Issue length",[["","All lengths"],["short","Short"],["medium","Medium"],["long","Long"]])}${field("limit","Results",50,"number",'min="1" max="500"')}<button class="button button-primary">Browse instances</button></form><p class="form-help">Issue length is a heuristic and does not measure repair difficulty.</p><div id="datasetSliceResults"></div>`;
      }
      else if(page==="settings")target.innerHTML=settingsScreen();
    }catch(error){target.innerHTML=`<div class="notice">${esc(error.message)}</div>`;}
  }
  function artifactTable(artifacts) {return table(["Artifact","Kind","Run / experiment","SHA256","Actions"],artifacts.map(a=>[esc(a.path||a.name||a.id),esc(a.kind||"file"),`<code>${esc(a.run_id||a.experiment_id||"—")}</code>`,`<code title="${esc(a.sha256)}">${esc(a.sha256?.slice(0,16)||"—")}</code>`,`${/\.(bin|safetensors|parquet|pt|pth|pkl|zip|gz|png|jpg|pdf)$/i.test(a.path||a.name||"")?"":`<button class="link-button" data-artifact="${esc(a.id||a.artifact_id)}">Preview</button>`} <a class="link-button" href="/api/artifacts/${encodeURIComponent(a.id||a.artifact_id)}/download" download>Download</a>`]),"No artifacts recorded. Completed jobs publish their available output here.");}
  function checkpointScreen() {return `<div class="panel">${table(["Checkpoint","Base model / algorithm","State","Path","Actions"],state.checkpoints.map(c=>[esc(c.name||c.id),`${esc(c.base_model||c.model||"—")}<small class="table-subtitle">${esc(c.algorithm||"—")}</small>`,badge(c.archived?"archived":"available"),`<code>${esc(c.path)}</code>`,`${["evaluate","load",c.archived?"restore":"archive","delete"].map(action=>`<button class="button button-secondary button-sm" data-checkpoint="${esc(c.id||c.name)}" data-checkpoint-action="${action}">${action=== "load" ? "Select in vLLM" : action[0].toUpperCase()+action.slice(1)}</button>`).join(" ")}`]),"No completed training checkpoints recorded.")}<p class="form-help checkpoint-help">Selecting a checkpoint updates the model connection. Reload the vLLM serving process to serve its weights.</p></div>`;}
  function rewardDetails(row) {const details=row.reward_details||{};return {exec:details.exec||details.execution||details,shaped:details.shaped||{},combined:row.metadata?.training_reward ?? details.combined ?? row.reward};}
  function rewardOverview() {
    const runs=state.runs.filter(r=>(!r.kind||r.kind==="run")&&r.reward!=null);const solved=runs.filter(resolved),unsolved=runs.filter(r=>!resolved(r));
    const shaped=rows=>avg(rows.map(r=>rewardDetails(r).shaped.value));
    return `<div class="metric-grid">${metricsCard("Solved mean shaped reward",num(shaped(solved),4),`${solved.length} recorded candidates`)}${metricsCard("Unsolved mean shaped reward",num(shaped(unsolved),4),`${unsolved.length} recorded candidates`)}</div>${table(["Instance","Execution reward","Shaped reward","Training reward","Inspect"],runs.map(r=>[esc(r.instance_id),num(rewardDetails(r).exec.value??r.metadata?.binary_reward,3),num(rewardDetails(r).shaped.value,3),num(rewardDetails(r).combined,3),`<button class="link-button" data-inspect="${esc(idOf(r))}" data-inspect-tab="reward">Components</button>`]),"No reward evidence yet.")}`;
  }
  async function renderReplay(target) {
    let payload;try{payload=await api("/api/replay");}catch(error){target.innerHTML=empty(error.message);return;}
    const rows=Array.isArray(payload)?payload:payload.trajectories||payload.items||[];
    target.innerHTML=`<section class="panel"><div class="panel-header"><h2>Replay records</h2><span>${num(payload.count??rows.length)} trajectories</span></div>${table(["Trajectory","Instance","Reward","Review"],rows.map(r=>[`<code>${esc(idOf(r))}</code>`,esc(r.instance_id),num(r.reward,3),`<button class="link-button" data-inspect="${esc(idOf(r))}">Inspect & annotate</button>`]),"No replay records available.")}</section><form id="replayExportForm" class="panel registry-form">${select("format","Reviewed export format",[["sft","SFT messages"],["dpo","DPO preference pairs"]],"sft")}<button class="button button-primary">Export reviewed dataset</button></form>`;
  }
  async function renderPrompts(target) {
    const payload=await api("/api/prompts");const rows=Array.isArray(payload)?payload:payload.prompts||[];
    target.innerHTML=`<form id="promptForm" class="panel launch-form"><div class="control-form-grid">${field("name","Prompt name","","text","required")}${field("version","Version","1","text","required")}</div><label class="form-field"><span class="form-label">System prompt</span><textarea name="content" class="input-control" rows="10" required></textarea></label><button class="button button-primary">Save prompt version</button></form>${table(["Name","Version","Fingerprint","Content"],rows.map(r=>[esc(r.name||r.id),esc(r.version),`<code>${esc(r.sha256||r.fingerprint||"—")}</code>`,`<details><summary>Read prompt</summary><pre class="evidence-code">${esc(r.content||r.prompt)}</pre></details>`]),"No saved prompt versions.")}`;
  }
  async function renderRegistry(target) {
    const payload=await api("/api/model/registry");const models=Array.isArray(payload)?payload:payload.models||[];
    const fallback=await api("/api/model/fallback");
    target.innerHTML=`<form id="modelRegistryForm" class="panel launch-form"><div class="panel-header"><div><h2>Model capabilities & pricing</h2><p>Declared capabilities describe your selected model; latency and usage come from measured requests.</p></div></div><div class="control-form-grid">${field("provider","Provider ID","vllm","text","required")}${field("model","Model ID","","text","required")}${field("context_limit","Context limit (tokens)","","number",'min="1"')}${field("input_price_per_1k","Input price / 1K tokens (USD)","","number",'min="0" step="any"')}${field("output_price_per_1k","Output price / 1K tokens (USD)","","number",'min="0" step="any"')}</div><div class="capability-checks">${["tool_calling","json_mode","streaming"].map(k=>`<label class="check-control"><input name="${k}" type="checkbox"> ${k.replaceAll("_"," ")}</label>`).join(" ")}</div><button class="button button-primary">Save model record</button></form><div class="panel">${table(["Model","Provider","Context","Capabilities","Pricing / measured health"],models.map(m=>[esc(m.model||m.id||m.name),esc(m.provider),num(m.context_limit||m.context_window),esc([m.tool_calling?"Tool calling":null,m.json_mode?"JSON":null,m.streaming?"Streaming":null].filter(Boolean).join(", ")||"Not recorded"),`${esc(m.health||m.status||"Not measured")}<small class="table-subtitle">Input ${money(m.input_price_per_1k)} / output ${money(m.output_price_per_1k)} per 1K</small>`]),"Model capability information is not recorded yet. Configure and test a connection.")}<div class="panel-footer-link"><button class="link-button" data-go="connections">Configure / test model connection</button></div></div><div class="panel"><div class="panel-header"><h2>Measured provider requests</h2><span>Dashboard process</span></div>${table(["Provider / model","Requests","Average latency","Last request","Rate limited","Failures"],(payload.telemetry||[]).map(t=>[esc(`${t.provider} / ${t.model}`),num(t.requests),`${num(t.average_latency_ms,1)} ms`,t.last_request_at?esc(new Date(t.last_request_at*1000).toLocaleString()):"—",esc(t.rate_limited==null?"Unknown":t.rate_limited?"Yes":"No"),num(t.failures)]),"No measured requests in this process.")}</div><form id="fallbackForm" class="panel registry-form"><label class="check-control"><input name="enabled" type="checkbox"> Enable fallback</label>${field("provider","Fallback provider")}${field("model","Fallback model")}${field("base_url","Fallback endpoint")}${field("secret_ref","Fallback credential reference (configure the provider in Connections first)")}<button class="button button-primary">Save fallback</button></form>`;
    const fallbackForm=$("#fallbackForm");fallbackForm.elements.enabled.checked=Boolean(fallback.enabled);
    ["provider","model","base_url"].forEach(key=>fallbackForm.elements[key].value=fallback[key]||"");
  }
  async function renderInfrastructure(target,page) {
    const payload=await api(page==="metrics"?"/api/infrastructure/metrics":"/api/infrastructure");
    if(page==="workers") {
      const workers=payload.workers||[];const list=Array.isArray(workers)?workers:workers.items||[];
      target.innerHTML=`<div class="metric-grid">${Object.entries({queued:payload.queued,max_workers:payload.max_workers,scheduler:payload.scheduler}).filter(([,v])=>typeof v!="object").map(([k,v])=>metricsCard(esc(k.replaceAll("_"," ")),esc(v))).join("")}</div>${table(["Worker","State","Job","Observed metrics"],list.map(w=>[esc(w.id||w.name||w.pid),badge(w.status||w.state),esc(w.job_id||"—"),`<pre>${esc(json(w))}</pre>`]),"No workers observed. Start a job to activate local execution.")}${jobsTable(state.jobs)}`;
    }else if(page==="containers") {
      const containers=payload.containers||payload.docker?.containers||[];target.innerHTML=table(["Container","State","Image","CPU / memory"],(Array.isArray(containers)?containers:[]).map(c=>[esc(c.name||c.id),badge(c.status),esc(c.image),esc(c.cpu_percent!=null?`${num(c.cpu_percent,1)}% / ${num(c.memory_bytes)} bytes`:"Not measured")]),payload.docker?.error||"No running Docker containers recorded.");
    }else {
      const metrics=payload.metrics||payload;target.innerHTML=`<div class="panel"><div class="panel-header"><h2>Observed service metrics</h2><p>Missing metrics are unavailable, rather than estimated.</p></div>${table(["Source","Recorded measurement"],Object.entries(metrics).map(([k,v])=>[esc(k),`<pre class="metric-json">${esc(typeof v==="object"?json(v):v)}</pre>`]))}</div>`;
    }
  }
  function settingsScreen() {const views=readViews();return `<div class="panel"><div class="panel-header"><div><h2>Local workspace</h2><p>Theme and saved filters remain in this browser. Provider secrets are handled by the server credential store.</p></div></div><div class="settings-actions"><button class="button button-secondary" data-go="connections">Model connections</button><button class="button button-secondary" id="clearViewsButton">Clear saved views</button></div>${table(["Saved view","Search","Outcome"],views.map(v=>[esc(v.name),esc(v.query),esc(v.status)]),"No saved views.")}</div>`;}
  function readViews() {try{const rows=JSON.parse(localStorage.getItem("swe-rl-saved-views")||"[]");return Array.isArray(rows)?rows:[];}catch{return [];}}
  function updateSavedViews() {$("#savedViewSelect").innerHTML='<option value="">Saved views</option>'+readViews().map((v,i)=>`<option value="${i}">${esc(v.name)}</option>`).join("");}

  function patchFiles(patch) {
    const files=[];let current;
    String(patch||"").split("\n").forEach(line=>{
      if(line.startsWith("diff --git ")) {const match=line.match(/^diff --git (?:"?a\/.*"?) (?:"?b\/(.*?)"?)$/);current={name:match?.[1]||line.slice(11),lines:[line],added:0,removed:0};files.push(current);}
      else {if(!current){current={name:"Patch",lines:[],added:0,removed:0};files.push(current);}current.lines.push(line);if(line.startsWith("+++ b/"))current.name=line.slice(6);if(line.startsWith("+")&&!line.startsWith("+++"))current.added++;if(line.startsWith("-")&&!line.startsWith("---"))current.removed++;}
    });return files.filter(f=>f.lines.some(l=>l.trim()));
  }
  function renderDiff(run) {
    const files=patchFiles(run.final_patch||run.patch);const viewer=$("#patchViewer");
    if(!files.length){viewer.innerHTML=empty("No patch recorded.");$("#diffControls").innerHTML="";return;}
    $("#diffControls").innerHTML=`<div class="segmented-control" role="group" aria-label="Diff view"><button class="button button-sm ${state.diffMode==="unified"?"button-primary":"button-secondary"}" data-diff-mode="unified">Unified</button><button class="button button-sm ${state.diffMode==="split"?"button-primary":"button-secondary"}" data-diff-mode="split">Split</button></div><label class="check-control"><input type="checkbox" id="ignoreWhitespace" ${state.ignoreWhitespace?"checked":""}> Ignore whitespace</label><label class="check-control"><input type="checkbox" id="compactContext" ${state.compactContext?"checked":""}> Changes only</label><span class="form-help">Context is limited to the saved patch.</span>`;
    const selected=files.filter(f=>!state.diffFile||f.name===state.diffFile);if(!selected.length)state.diffFile="";
    viewer.innerHTML=`<div class="diff-workspace"><aside class="changed-files" aria-label="Changed files"><button class="file-tree-item ${!state.diffFile?"is-active":""}" data-diff-file="">${files.length} changed files</button>${files.map(f=>`<button class="file-tree-item ${state.diffFile===f.name?"is-active":""}" data-diff-file="${esc(f.name)}"><span>${esc(f.name)}</span><small>+${f.added} −${f.removed}</small></button>`).join("")}</aside><div class="diff-body">${(selected.length?selected:files).map(file=>renderDiffFile(file)).join("")}</div></div>`;
  }
  function renderDiffFile(file) {
    const rows=[];let left=0,right=0;const lines=file.lines;
    for(let i=0;i<lines.length;i++) {
      const line=lines[i];const hunk=line.match(/^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@/);
      if(hunk){left=Number(hunk[1]);right=Number(hunk[2]);rows.push(`<tr class="diff-hunk"><td colspan="4">${esc(line)}</td></tr>`);continue;}
      if(/^(diff |index |--- |\+\+\+ |\\|new file |deleted file |similarity |rename )/.test(line))continue;
      if(line.startsWith("-")) {
        const removed=[];while(i<lines.length && lines[i].startsWith("-")&&!lines[i].startsWith("---"))removed.push(lines[i++].slice(1));
        const added=[];while(i<lines.length && lines[i].startsWith("+")&&!lines[i].startsWith("+++"))added.push(lines[i++].slice(1));i--;
        const count=Math.max(removed.length,added.length);
        for(let n=0;n<count;n++) {
          const old=removed[n],next=added[n];const equal=state.ignoreWhitespace&&old!=null&&next!=null&&old.replace(/\s/g,"")===next.replace(/\s/g,"");
          if(state.diffMode==="split")rows.push(`<tr><td class="line-number">${old!=null?left++:""}</td><td class="${equal?"":"diff-removed"}">${esc(old??"")}</td><td class="line-number">${next!=null?right++:""}</td><td class="${equal?"":"diff-added"}">${esc(next??"")}</td></tr>`);
          else if(equal){rows.push(`<tr><td class="line-number">${left++}</td><td class="line-number">${right++}</td><td colspan="2"> ${esc(next)}</td></tr>`);}else{if(old!=null)rows.push(`<tr class="diff-removed"><td class="line-number">${left++}</td><td></td><td colspan="2">−${esc(old)}</td></tr>`);if(next!=null)rows.push(`<tr class="diff-added"><td></td><td class="line-number">${right++}</td><td colspan="2">+${esc(next)}</td></tr>`);}
        }continue;
      }
      if(line.startsWith("+")){rows.push(state.diffMode==="split"?`<tr><td></td><td></td><td class="line-number">${right++}</td><td class="diff-added">${esc(line.slice(1))}</td></tr>`:`<tr class="diff-added"><td></td><td class="line-number">${right++}</td><td colspan="2">${esc(line)}</td></tr>`);continue;}
      if(line.startsWith(" ")) {const a=left++,b=right++;if(!state.compactContext)rows.push(state.diffMode==="split"?`<tr><td class="line-number">${a}</td><td>${esc(line.slice(1))}</td><td class="line-number">${b}</td><td>${esc(line.slice(1))}</td></tr>`:`<tr><td class="line-number">${a}</td><td class="line-number">${b}</td><td colspan="2">${esc(line)}</td></tr>`);}
    }
    return `<section class="diff-file"><h3 class="diff-file-heading">${esc(file.name)}</h3><table class="diff-table ${state.diffMode}"><caption class="visually-hidden">${state.diffMode} diff for ${esc(file.name)}</caption>${state.diffMode==="split"?'<thead><tr><th colspan="2">Original</th><th colspan="2">Modified</th></tr></thead>':""}<tbody>${rows.join("")}</tbody></table></section>`;
  }
  function renderRunExtras(run) {
    state.selected=run;
    renderDiff(run);
    const calls=run.tool_calls||[];const details=rewardDetails(run);const runId=idOf(run);
    const status=String(run.status||"").toLowerCase();
    $("#liveRunControls").innerHTML=`<span class="mono">${esc(runId)}</span>${badge(run.paused?"paused":status)}${run.kind||run.config?jobActions(run):""}<span id="liveConnectionState" class="form-help">${terminal(status)?"Recorded evidence":state.stream?"Live event stream connected":"Select execution to follow live events"}</span>`;
    $("#executionPipeline").innerHTML=[
      ["Setup",run.metadata?.setup||calls.length>0||run.final_patch?"Recorded":status==="preparing"?"Active":"Unknown"],
      ["Agent",calls.length?`${calls.length} tool calls`:status==="running"?"Active":"Unknown"],
      ["Patch",run.final_patch?.trim()?"Recorded":"Not recorded"],
      ["Evaluation",run.reward!=null?"Recorded":status==="evaluating"?"Active":"Unknown"],
    ].map(([title,value])=>`<div class="pipeline-stage"><span class="pipeline-dot ${value==="Unknown"||value==="Not recorded"?"":"has-evidence"}"></span><span>${title}: ${value}</span></div>`).join('<div class="pipeline-line"></div>');
    const attributed=(run.metadata?.turn_usage||run.metadata?.step_usage||run.step_usage||[]).map(u=>({...u,tool:u.tool||calls.find(c=>c.step===u.step)?.tool||"model-only"}));
    $("#run-costs").innerHTML=`<div class="metric-grid">${metricsCard("Total cost",money(cost(run)),"Recorded USD usage")}${metricsCard("Input tokens",num(run.tokens_in??run.input_tokens))}${metricsCard("Output tokens",num(run.tokens_out??run.output_tokens))}${metricsCard("Runtime",runtimeValue(runtime(run)))}</div>${table(["Step","Action","Input tokens","Output tokens","Cost","Duration"],(attributed.length?attributed:calls).map((c,i)=>[num(c.step??i+1),esc(c.tool||c.action||"model"),num(c.tokens_in??c.input_tokens),num(c.tokens_out??c.output_tokens),money(c.cost_usd),runtimeValue(c.duration_s)]),"No step usage recorded.")}<p class="form-help">Missing per-step usage is not attributed or estimated.</p>`;
    $("#run-reward").innerHTML=`<div class="metric-grid">${metricsCard("Execution reward",num(details.exec.value??run.metadata?.binary_reward,3))}${metricsCard("Shaped reward",num(details.shaped.value,4))}${metricsCard("Configured reward",num(details.combined,4),"Recorded weighted reward")}</div>${table(["Component","Value"],Object.entries(details.shaped.components||{}).map(([k,v])=>[esc(k),num(v,5)]),"No shaped reward components recorded.")}<details><summary>Full reward evidence</summary><pre class="evidence-code">${esc(json(run.reward_details))}</pre></details>`;
    const failure=run.failure_analysis||run.metadata?.failure_analysis||run.result?.failure_analysis;
    $("#run-failure").innerHTML=failure?`<div class="notice">${esc(failure.category||failure.stage||"Recorded failure analysis")}</div><p class="form-help">${failure.model_generated?"Model generated explanation; inspect execution evidence before accepting.":"Rule based classification from recorded execution evidence."}</p><pre class="evidence-code">${esc(json(failure))}</pre>`:empty(resolved(run)===true?"This run resolved the recorded tests.":"No failure classification recorded.");
    const finalTests=run.metadata?.final_tests;
    if(finalTests)$("#testViewer").insertAdjacentHTML("beforeend",`<details class="test-logs"><summary>Final test stdout, stderr and execution details</summary><pre class="evidence-code">${esc(json(finalTests))}</pre></details>`);
    const lastTest=calls.findLastIndex(c=>c.tool==="test_run");
    const exec=details.exec;const failed=[...(exec.f2p_failed||[]),...(exec.p2p_failed||[])];
    if(lastTest>=0 && failed.length)$("#testViewer").insertAdjacentHTML("beforeend",`<button class="button button-secondary button-sm" data-jump-step="${lastTest+1}">Jump to agent test step ${lastTest+1}</button><pre class="evidence-code">${esc(json(calls[lastTest].result))}</pre>`);
    const testUsage=calls.filter(c=>c.tool==="test_run").length;
    $("#run-costs").insertAdjacentHTML("beforeend",`<p class="form-help">Tests used: ${run.test_budget_remaining!=null && run.config?.test_budget ? run.config.test_budget-run.test_budget_remaining : testUsage}${run.config?.test_budget?` / ${run.config.test_budget}`:""}; steps: ${num(run.steps??run.n_steps??calls.length)}${run.config?.max_steps?` / ${run.config.max_steps}`:""}</p>`);
    const graphCall=calls.findLast(c=>["repository_graph","repo_graph"].includes(c.tool));
    const graph=graphCall?.result?.data?.graph||run.metadata?.repository_graph;
    const suggested=graphCall?.result?.data?.recommended_tests||run.metadata?.recommended_tests;
    $("#run-context").innerHTML=graph ? `<h3>Repository graph</h3><pre class="evidence-code">${esc(json(graph))}</pre>${suggested?`<h3>Related tests</h3><pre class="evidence-code">${esc(json(suggested))}</pre>`:""}` : empty("No repository graph recorded. New runs can request repository_graph through agent tools.");
    if(run.metadata?.patch_minimization)$("#run-context").insertAdjacentHTML("beforeend",`<h3>Patch minimization evidence</h3><pre class="evidence-code">${esc(json(run.metadata.patch_minimization))}</pre>`);
    $("#runAnnotation").innerHTML=run.config?`<button class="button button-ghost button-sm" data-annotate="approved">Approve training example</button><button class="button button-ghost button-sm" data-annotate="rejected">Reject training example</button><label class="visually-hidden" for="annotationNote">Review note</label><input class="input-control annotation-note" id="annotationNote" placeholder="Research review note">`:"";
    if($("#inspectorRuntime"))$("#inspectorRuntime").textContent=runtimeValue(runtime(run));
    if($("#inspectorCost"))$("#inspectorCost").textContent=money(cost(run));
    if($("#inspectorFileTree"))$("#inspectorFileTree").innerHTML=patchFiles(run.final_patch).map(f=>`<button class="file-tree-item" data-diff-file="${esc(f.name)}"><span>${esc(f.name)}</span><small>+${f.added} −${f.removed}</small></button>`).join("")||empty("No changed files.");
    $$(".trace-body").forEach((body,i)=>{const usage=attributed.find(u=>u.step===calls[i].step)||calls[i];body.insertAdjacentHTML("afterbegin",`<p class="form-help">Input ${num(usage.tokens_in)} · Output ${num(usage.tokens_out)} · Cost ${money(usage.cost_usd)}</p>`);});
  }
  async function inspect(runId,tab) {
    let run=state.runs.find(r=>idOf(r)===runId);
    if(!run)run=await api(`/api/runs/${encodeURIComponent(runId)}`);
    if(run.kind && run.kind!=="run") {
      state.selectedJob=run;
      window.SWERL.setPage(run.kind==="evaluation"?"evaluations":"training");
      await renderJobDetail(run);return;
    }
    state.selectedJob=null;

    window.SWERL.setPage("runs");window.SWERL.selectRun(run);if(tab)window.SWERL.activateRunTab(tab);
  }
  async function renderJobDetail(run) {
    const page=run.kind==="evaluation"?"evaluations":"training";
    const target=$(`#content-${page}`);if(!target)return;
    $("#jobDetail")?.remove();
    target.insertAdjacentHTML("afterbegin",`<section class="panel job-detail" id="jobDetail"><div class="panel-header"><h2>${esc(idOf(run))}</h2>${badge(run.status)}</div><div class="job-controls">${jobActions(run)}</div><pre class="evidence-code">${esc(json({config:run.config,result:run.result,error:run.error}))}</pre><div id="trainingMetricHistory"></div></section>`);
    if(run.kind==="training") {
      const metrics=await api(`/api/training/jobs/${encodeURIComponent(idOf(run))}/metrics`);
      const history=metrics.history||[];const last=history.at(-1)||{};
      $("#trainingMetricHistory").innerHTML=`<div class="metric-grid">${metricsCard("Step",num(last.step))}${metricsCard("Reward",num(last.reward??last.rewards??last["rewards/mean"],4))}${metricsCard("KL",num(last.kl,4))}${metricsCard("GPU telemetry",metrics.gpu_metrics?"Measured":"Unavailable")}</div>`+table(["Step","Loss","Reward","KL","Learning rate"],history.map(m=>[num(m.step),num(m.loss,5),num(m.reward??m.rewards??m["rewards/mean"],5),num(m.kl,5),num(m.learning_rate,8)]),"No trainer metrics recorded yet.");
    }
  }
  function connectEvents(run) {
    state.stream?.close();state.stream=null;state.events=[];$("#liveEvents").innerHTML="";
    if(terminal(run.status)||!run.config)return;
    const runId=idOf(run);const stream=new EventSource(`/api/runs/${encodeURIComponent(runId)}/events`);state.stream=stream;
    stream.onopen=()=>{if($("#liveConnectionState"))$("#liveConnectionState").textContent="Live event stream connected";};
    const receive=event=>{
      if(state.selected && idOf(state.selected)!==runId)return;
      let payload;try{payload=JSON.parse(event.data);}catch{return;}
      state.events.push(payload);if(state.events.length>100)state.events.shift();
      $("#liveEvents").innerHTML=state.events.map(e=>`<details class="live-event"><summary><span class="mono">${esc(e.created_at||e.timestamp||"")}</span> ${esc(e.type||e.event||e.status||"Update")}</summary><pre>${esc(json(e.data||e.payload||e))}</pre></details>`).join("");
      if(payload.type==="step"||payload.type==="state"||payload.type==="status"||payload.type==="finished"||payload.type==="completed"||payload.status||payload.data?.status) {
        void window.SWERL.loadRuns();void refresh();
        if(terminal(payload.data?.status||payload.status||payload.type)){stream.close();$("#liveConnectionState").textContent="Execution finished; recorded evidence available";}
      }
    };
    stream.onmessage=receive;["status","step","tool","message","reward","completed","failed","finished","snapshot","update"].forEach(type=>stream.addEventListener(type,receive));
    stream.onerror=()=>{if($("#liveConnectionState"))$("#liveConnectionState").textContent="Live stream reconnecting…";};
  }
  async function jobAction(action,id) {
    const row=state.jobs.find(j=>idOf(j)===id)||state.selected;
    const endpoint=`/api/runs/${encodeURIComponent(id)}/${action}`;
    await api(endpoint,{method:"POST"});await refresh();await window.SWERL.loadRuns();
    window.SWERL.showToast("Job updated",`${action} requested.`,"success");
  }
  function formValues(form) {const data=new FormData(form);return Object.fromEntries(data.entries());}
  async function submitForm(form) {
    const values=formValues(form);const control=$("button[type=submit]",form)||$("button",form);if(control)control.disabled=true;
    try {
      if(form.id==="newRunForm") {
        const ids=[...new Set(values.instances.split(/[\s,]+/).map(s=>s.trim()).filter(Boolean))];
        const payload={instance_ids:ids,instance_id:ids[0],dataset:values.dataset,model:values.model,provider:values.provider||undefined,strategy:values.strategy,dataset_revision:values.dataset_revision,prompt_version:values.prompt_version||undefined,checkpoint:values.checkpoint||undefined,selection:values.selection,prm_model:values.prm_model||undefined,minimize_patch:form.elements.minimize_patch.checked,temperature:Number(values.temperature),top_p:Number(values.top_p),seed:Number(values.seed),max_steps:Number(values.max_steps),test_budget:Number(values.test_budget),max_cost:Number(values.max_cost),candidates:Number(values.candidates),sandbox:{cpus:Number(values.cpus),memory:Number(values.memory),timeout:Number(values.timeout),network_disabled:form.elements.network_disabled.checked},reward:{binary_weight:Number(values.binary_weight),shaped_weight:Number(values.shaped_weight),prm_weight:Number(values.prm_weight)}};
        const result=await api("/api/runs",{method:"POST",body:JSON.stringify(payload)});$("#newRunStatus").textContent=`Experiment ${result.experiment_id}; fingerprint ${result.fingerprint}. ${result.runs.length} candidates queued.`;
        await refresh();await window.SWERL.loadRuns();if(result.runs[0])await inspect(idOf(result.runs[0]),"trace");
      }else if(form.id==="evaluationForm") {
        await api("/api/evaluations",{method:"POST",body:JSON.stringify({dataset:values.dataset,model:values.model||"",predictions_path:values.predictions_path||undefined,max_instances:Number(values.max_instances),parallelism:Number(values.parallelism),candidates:Number(values.candidates),invoke_harness:form.elements.invoke_harness.checked})});$("#evaluationStatus").textContent="Evaluation queued.";await refresh();
      }else if(form.id==="trainingForm") {
        await api("/api/training/jobs",{method:"POST",body:JSON.stringify({algorithm:values.algorithm,model:values.model,dataset_path:values.dataset_path||undefined,learning_rate:Number(values.learning_rate),max_instances:Number(values.max_instances),allow_heavy_training:form.elements.allow_heavy_training.checked})});$("#trainingStatus").textContent="Training queued.";await refresh();
      }else if(form.id==="compareForm")$("#compareResult").innerHTML=compareExperiments(values.a,values.b);
      else if(form.id==="datasetSliceForm") {
        const query=new URLSearchParams({repo:values.repo,language:values.language,limit:values.limit,offset:"0"});if(values.difficulty)query.set("difficulty",values.difficulty);
        const result=await api(`/api/datasets/${encodeURIComponent(values.dataset)}/instances?${query}`);const rows=Array.isArray(result)?result:result.instances||result.items||[];
        $("#datasetSliceResults").innerHTML=table(["Instance","Repository","Issue","Repair"],rows.map(r=>[esc(r.instance_id),esc(r.repo),`<details><summary>${esc(r.problem_statement?.slice(0,140)||"Issue")}</summary><pre class="evidence-code">${esc(r.problem_statement)}</pre></details>`,`<button class="link-button" data-new-instance="${esc(r.instance_id)}" data-new-dataset="${esc(values.dataset)}">Configure run</button>`]),"No instances match this slice.");
      }
      else if(form.id==="promptForm"){await api("/api/prompts",{method:"POST",body:JSON.stringify(values)});await renderPage("prompts");}
      else if(form.id==="replayExportForm"){const result=await api("/api/replay/export",{method:"POST",body:JSON.stringify(values)});window.SWERL.showToast("Replay exported",json(result),"success");}
      else if(form.id==="modelRegistryForm"){const body={provider:values.provider,model:values.model};["context_limit","input_price_per_1k","output_price_per_1k"].forEach(key=>{if(values[key]!=="")body[key]=Number(values[key]);});["tool_calling","json_mode","streaming"].forEach(key=>body[key]=form.elements[key].checked);await api("/api/model/registry",{method:"POST",body:JSON.stringify(body)});await renderPage("registry");}
      else if(form.id==="fallbackForm"){await api("/api/model/fallback",{method:"POST",body:JSON.stringify({...values,enabled:form.elements.enabled.checked})});window.SWERL.showToast("Fallback saved",form.elements.enabled.checked?"The fallback model is available when the primary provider fails.":"Fallback disabled.","success");}
    }catch(error){window.SWERL.showToast("Action failed",error.message,"error");const status=$("[role=status]",form);if(status)status.textContent=error.message;}
    finally{if(control)control.disabled=false;}
  }
  function commands() {return [...state.experiments.map(e=>({type:"Experiment",title:e.experiment_id||e.id,subtitle:`${e.config?.dataset||""} ${e.fingerprint||""}`,searchable:json(e).toLowerCase(),action:()=>{window.SWERL.setPage("experiments");}})),...state.artifacts.map(a=>({type:"Artifact",title:a.path||a.name||a.id,subtitle:a.kind||"Artifact",searchable:json(a).toLowerCase(),action:()=>{window.SWERL.setPage("artifacts");}}))];}
  function initialize() {
    buildScreens();updateSavedViews();
    document.addEventListener("swerl:runs",event=>{state.runs=event.detail;renderOverview();});
    document.addEventListener("swerl:page",event=>{void renderPage(event.detail);});
    document.addEventListener("swerl:selected",event=>{state.diffFile="";renderDiff(event.detail);connectEvents(event.detail);});
    document.addEventListener("submit",event=>{if(["newRunForm","evaluationForm","trainingForm","compareForm","promptForm","replayExportForm","fallbackForm","modelRegistryForm","datasetSliceForm"].includes(event.target.id)){event.preventDefault();void submitForm(event.target);}});
    document.addEventListener("click",async event=>{
      const control=event.target.closest("button");if(!control)return;
      try {
        if(control.dataset.go)window.SWERL.setPage(control.dataset.go);
        else if(control.dataset.refresh)await refresh();
        else if(control.dataset.newInstance){window.SWERL.setPage("new-run");$("#newRunForm").elements.instances.value=control.dataset.newInstance;$("#newRunForm").elements.dataset.value=control.dataset.newDataset;}
        else if(control.dataset.inspect)await inspect(control.dataset.inspect,control.dataset.inspectTab);
        else if(control.dataset.jobAction){control.disabled=true;await jobAction(control.dataset.jobAction,control.dataset.jobId);}
        else if(control.dataset.diffMode){state.diffMode=control.dataset.diffMode;renderDiff(state.selected);}
        else if(control.hasAttribute("data-diff-file")){state.diffFile=control.dataset.diffFile;window.SWERL.activateRunTab("patch");renderDiff(state.selected);}
        else if(control.dataset.jumpStep){window.SWERL.activateRunTab("trace");const step=$(`#trace-step-${control.dataset.jumpStep}`);step?.classList.add("is-open");$(".trace-summary",step)?.setAttribute("aria-expanded","true");step?.scrollIntoView({block:"center"});}
        else if(control.dataset.experiment){const row=state.experiments.find(e=>(e.experiment_id||e.id)===control.dataset.experiment);$("#content-experiments").insertAdjacentHTML("afterbegin",`<details class="panel config-detail" open><summary>Experiment ${esc(control.dataset.experiment)}</summary><pre class="evidence-code">${esc(json(row))}</pre><div class="selection-toolbar">${select("selectionStrategy","Candidate selection",[["execution-score","Execution score"],["minimal-patch","Minimal patch"],["regression-safe","Regression safe"],["majority-behavior","Majority behavior"]])}<button class="button button-secondary" data-candidate-selection="${esc(control.dataset.experiment)}">Inspect selected candidates</button><button class="button button-secondary" data-critic-selection="${esc(control.dataset.experiment)}">Run critic review (model request)</button></div><div id="candidateSelectionResult"></div></details>`);}
        else if(control.dataset.candidateSelection||control.dataset.criticSelection){
          const id=control.dataset.candidateSelection||control.dataset.criticSelection;
          const strategy=control.dataset.criticSelection?"critic-ranking":$("select[name=selectionStrategy]",control.closest("details")).value;
          control.disabled=true;
          const result=await api(`/api/experiments/${encodeURIComponent(id)}/selection${control.dataset.criticSelection?"":`?strategy=${encodeURIComponent(strategy)}`}`,control.dataset.criticSelection?{method:"POST",body:JSON.stringify({strategy,allow_model_call:true})}:{});
          $("#candidateSelectionResult",control.closest("details")).innerHTML=table(["Instance","Selected candidate","Resolved","Selection evidence"],(result.selections||[]).map(r=>[esc(r.instance_id),`<button class="link-button" data-inspect="${esc(r.selected_run_id)}">${esc(r.selected_run_id)}</button>`,esc(r.resolved),esc(r.selection_source)]),"No completed candidates available to select.");
        }
        else if(control.dataset.artifact){const result=await api(`/api/artifacts/${encodeURIComponent(control.dataset.artifact)}/content`);let preview=$("#artifactPreview");if(!preview){$(`#content-${window.SWERL.getState().page}`).insertAdjacentHTML("beforeend",'<pre id="artifactPreview" class="evidence-code"></pre>');preview=$("#artifactPreview");}preview.classList.remove("hidden");preview.textContent=typeof result==="string"?result:result.content||json(result);}
        else if(control.dataset.checkpointAction){
          const action=control.dataset.checkpointAction;
          if(action==="evaluate") {
            const checkpoint=state.checkpoints.find(c=>c.id===control.dataset.checkpoint);
            window.SWERL.setPage("benchmarks");$("#evaluationModel").value=checkpoint.path;
            $("#evaluationStatus").textContent="Checkpoint selected. The configured endpoint must serve this checkpoint before evaluation.";
          }else{
            if(action==="delete"&&!window.confirm("Delete this checkpoint directory permanently? Training metadata will remain."))return;
            const result=await api(`/api/checkpoints/${encodeURIComponent(control.dataset.checkpoint)}/${action}`,{method:"POST"});await refresh();
            window.SWERL.showToast("Checkpoint updated",result.serving_reload_required?"Model selected. Reload vLLM with this checkpoint before running inference.":`${action} complete.`,"success");
          }
        }
        else if(control.dataset.annotate){await api(`/api/runs/${encodeURIComponent(idOf(state.selected))}/annotation`,{method:"POST",body:JSON.stringify({decision:control.dataset.annotate,notes:$("#annotationNote").value})});window.SWERL.showToast("Review saved",control.dataset.annotate,"success");}

        else if(control.id==="saveViewButton") {const name=window.prompt("Name this run view:");if(!name?.trim())return;const views=readViews();views.push({name:name.trim(),query:$("#runSearchInput").value,status:$("#runStatusFilter").value});localStorage.setItem("swe-rl-saved-views",JSON.stringify(views));updateSavedViews();}
        else if(control.id==="deleteViewButton"){const index=$("#savedViewSelect").value;if(index!==""){const views=readViews();views.splice(Number(index),1);localStorage.setItem("swe-rl-saved-views",JSON.stringify(views));updateSavedViews();}}
        else if(control.id==="clearViewsButton"){localStorage.removeItem("swe-rl-saved-views");updateSavedViews();await renderPage("settings");}
      }catch(error){window.SWERL.showToast("Action failed",error.message,"error");}
      finally{if(control.dataset.jobAction||control.dataset.candidateSelection||control.dataset.criticSelection)control.disabled=false;}
    });
    document.addEventListener("change",event=>{
      if(event.target.dataset.priority){void api(`/api/jobs/${encodeURIComponent(event.target.dataset.priority)}/priority`,{method:"POST",body:JSON.stringify({priority:Number(event.target.value)})}).then(()=>refresh()).catch(error=>window.SWERL.showToast("Priority update failed",error.message,"error"));}
      else if(event.target.id==="ignoreWhitespace"){state.ignoreWhitespace=event.target.checked;renderDiff(state.selected);}
      else if(event.target.id==="compactContext"){state.compactContext=event.target.checked;renderDiff(state.selected);}
      else if(event.target.id==="savedViewSelect"&&event.target.value!==""){const view=readViews()[Number(event.target.value)];$("#runSearchInput").value=view.query;$("#runStatusFilter").value=view.status;$("#runSearchInput").dispatchEvent(new Event("input"));}
    });
    document.addEventListener("input",event=>{if(event.target.id==="artifactSearch"){const query=event.target.value.toLowerCase();$("#artifactTable").innerHTML=artifactTable(state.artifacts.filter(a=>json(a).toLowerCase().includes(query)));}});
    document.addEventListener("keydown",event=>{
      if(event.target.closest?.("input,textarea,select,[contenteditable=true]")||!$("#commandPalette").classList.contains("hidden"))return;
      const key=event.key.toLowerCase();const active=window.SWERL.getState();
      if(active.page!=="runs")return;
      if(["j","k"].includes(key)){event.preventDefault();const list=active.filteredRuns;let index=list.findIndex(r=>idOf(r)===idOf(active.selectedRun));index=Math.max(0,Math.min(list.length-1,index+(key==="j"?1:-1)));if(list[index])window.SWERL.selectRun(list[index]);}
      else if({p:"patch",t:"tests",c:"conversation",e:"trace"}[key]){event.preventDefault();window.SWERL.activateRunTab({p:"patch",t:"tests",c:"conversation",e:"trace"}[key]);}
      else if((event.metaKey||event.ctrlKey)&&event.key==="Enter"&&state.selected){event.preventDefault();void jobAction("retry",idOf(state.selected)).catch(error=>window.SWERL.showToast("Retry failed",error.message,"error"));}
    });
    window.SWERLAPI.providers().then(payload=>{
      const provider=$("#newRunForm select[name=provider]");payload.providers.forEach(p=>provider.insertAdjacentHTML("beforeend",`<option value="${esc(p.id)}">${esc(p.name)}</option>`));
    }).catch(()=>{});
    void refresh();
    const monitor=window.setInterval(()=>{
      const page=window.SWERL?.getState().page;
      if(document.visibilityState!=="hidden"&&["overview","runs","training","evaluations","benchmarks","workers"].includes(page)&&state.jobs.some(j=>!terminal(j.status)))void refresh().catch(error=>window.SWERL.showToast("Refresh failed",error.message,"error"));
    },5000);
    window.addEventListener("pagehide",()=>window.clearInterval(monitor));
    window.addEventListener("hashchange",()=>window.SWERL.setPage(location.hash.replace("#","")));
    document.addEventListener("keydown",event=>{
      if(event.key!=="Tab"||$("#commandPalette").classList.contains("hidden"))return;
      const elements=$$("input,button",$("#commandPalette"));const first=elements[0],last=elements.at(-1);
      if(event.shiftKey&&document.activeElement===first){event.preventDefault();last?.focus();}
      else if(!event.shiftKey&&document.activeElement===last){event.preventDefault();first?.focus();}
    });
    window.addEventListener("pagehide",()=>state.stream?.close());
  }
  window.SWERLControl={refresh,renderRunExtras,commands,passAtK,patchFiles,compareExperiments};
  document.addEventListener("DOMContentLoaded",initialize);
})();
