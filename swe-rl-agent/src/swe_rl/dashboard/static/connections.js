"use strict";
(() => {
  const $ = (id) => document.getElementById(id);
  let catalog = [], active = null, token = "", busy = false;
  const status = (text) => {$("model-status").textContent = text;};
  function lock(value) {
    busy = value;
    ["save-model","discover-models","test-model","reset-model","ask-model"].forEach((id) => {$(id).disabled=value;});
  }
  async function request(url, method = "GET", body) {
    const response = await fetch(url,{method,headers:{"Content-Type":"application/json","X-Settings-Token":token},
      ...(body ? {body:JSON.stringify(body)} : {}),signal:AbortSignal.timeout(70000)});
    const data = await response.json();
    if(!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Request failed");
    return data;
  }
  function fields() {
    return {provider:$("provider").value,model:$("model-id").value.trim(),
      base_url:$("model-url").value.trim(),api_key:$("model-key").value};
  }
  function update(useActive = false) {
    const provider = catalog.find((item) => item.id === $("provider").value);
    if(!provider) return;
    $("model-key").value = "";
    $("model-options").replaceChildren();
    $("model-id").value = useActive ? active.model : "";
    $("model-url").value = useActive ? active.base_url : provider.base_url || (provider.id === "vllm" ? "http://127.0.0.1:8000/v1" : "");
    $("model-url").readOnly = !["vllm","custom","qwen"].includes(provider.id);
    $("provider-docs").href = provider.docs || "#model-panel";
    $("provider-docs").hidden = !provider.docs;
    $("model-key").placeholder = active?.provider === provider.id && active.has_key
      ? "Key configured; leave blank to retain" : provider.requires_key ? "Enter your provider API key" : "Optional key";
    $("endpoint-note").textContent = provider.id === "qwen"
      ? "Use the Singapore, Beijing, or US endpoint matching your Alibaba Cloud key."
      : "Enter a model available to your account. Fetch models contacts the provider.";
  }
  function badge() {
    $("active-model").textContent = active ? (catalog.find((p) => p.id === active.provider)?.name || active.provider)+" / "+(active.model || "Model not selected") : "No connection";
  }
  async function initialize() {
    try {
      const data=await request("/api/providers");catalog=data.providers;active=data.active;token=data.settings_token;
      $("provider").replaceChildren();
      catalog.forEach((p) => {const option=document.createElement("option");option.value=p.id;option.textContent=p.name;$("provider").append(option);});
      $("provider").value=active.provider;update(true);badge();
    } catch {status("Model settings could not load. Refresh to retry.");}
  }
  $("provider").onchange = () => {update();status("Enter your model and key, then save the connection.");};
  async function action(path, success) {
    if(busy) return;
    lock(true);status("Connecting…");
    try {success(await request(path,"POST",fields()));}
    catch(error) {status(error.message);}
    finally {$("model-key").value="";lock(false);}
  }
  $("model-form").onsubmit = (event) => {
    event.preventDefault();
    action("/api/model/settings",(data) => {active=data;badge();update(true);status("Saved. New agent runs and the repair assistant use this connection.");});
  };
  $("discover-models").onclick = () => action("/api/model/models",(data) => {
    $("model-options").replaceChildren();
    data.models.forEach((id) => {const option=document.createElement("option");option.value=id;$("model-options").append(option);});
    status(data.models.length+" models fetched. Choose a text model. Re-enter unsaved keys before saving.");
  });
  $("test-model").onclick = () => action("/api/model/test",(data) => {
    status("Completion verified: "+data.tokens_in+" input and "+data.tokens_out+" output tokens. Re-enter unsaved keys before saving.");
  });
  $("reset-model").onclick = async () => {
    if(busy) return;lock(true);
    try {active=await request("/api/model/settings","DELETE");$("provider").value=active.provider;update(true);badge();status("Saved connection and key removed; environment defaults restored.");}
    catch(error) {status(error.message);}
    finally {lock(false);}
  };
  $("assist-form").onsubmit = async (event) => {
    event.preventDefault();if(busy) return;lock(true);$("assist-result").textContent="Waiting for the selected model…";
    try {
      const data=await request("/api/model/assist","POST",{prompt:$("assist-prompt").value});
      $("assist-result").textContent=data.text || "No text returned. Try another model.";
      status("Response from "+data.provider+" / "+data.model+"; "+data.tokens_in+" input and "+data.tokens_out+" output tokens.");
    } catch(error) {$("assist-result").textContent=error.message;}
    finally {lock(false);}
  };
  initialize();
})();
