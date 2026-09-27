// Run with NODE_PATH pointing to an installed jsdom package. No browser needed.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { JSDOM } = require("jsdom");
const root = path.join(__dirname, "../src/swe_rl/dashboard/static");
const dom = new JSDOM(fs.readFileSync(path.join(root,"index.html"),"utf8"), {
  url:"http://localhost", runScripts:"outside-only"
});
const { window } = dom;
window.fetch = async (url) => ({ok:true,json:async () => url.endsWith("runs") ? [] : {api:"available",docker:"unavailable"}});
window.AbortSignal.timeout = () => undefined;
window.eval(fs.readFileSync(path.join(root,"app.js"),"utf8"));
const $ = (id) => window.document.getElementById(id);
(async () => {
  await new Promise((resolve) => setImmediate(resolve));
  assert.match($("notice").textContent,/No recorded runs/);
  $("demo").click();
  assert.equal(window.document.querySelectorAll(".run").length,2);
  assert.match($("notice").textContent,/synthetic/);
  assert.match($("content").textContent,/PurePosixPath/);
  $("tab-trace").click();
  assert.equal($("content").querySelectorAll("details").length,5);
  $("tab-tests").click();
  assert.match($("content").textContent,/test_relative_path/);
  $("filter").value = "unresolved";
  $("filter").dispatchEvent(new window.Event("change"));
  assert.match($("instance").textContent,/cache-service/);
  assert.match($("content").textContent,/test_no_ttl/);
  $("search").value = "missing";
  $("search").dispatchEvent(new window.Event("input"));
  assert.equal($("download").disabled,true);
  $("theme").click();
  assert.equal(window.localStorage.getItem("swe-rl-theme"),"dark");
  $("search").value = "";
  $("filter").value = "all";
  const malicious = '<img src=x onerror=alert(1)>';
  window.fetch = async (url) => ({ok:true,json:async () => url.endsWith("runs") ? [{
    run_id:"x",instance_id:malicious,model:"test",status:"completed",reward:0,
    final_patch:malicious,messages:[],tool_calls:[]
  }] : {api:"available"}});
  $("live").click();
  $("refresh").click();
  await new Promise((resolve) => setImmediate(resolve));
  $("tab-patch").click();
  assert.equal($("content").querySelector("img"),null);
  assert.equal($("instance").textContent,malicious);
  window.fetch = async () => {throw new Error("offline");};
  $("refresh").click();
  await new Promise((resolve) => setImmediate(resolve));
  assert.match($("notice").textContent,/Could not refresh/);
  assert.equal($("refresh").disabled,false);
  dom.window.close();
  console.log("PASS: demo, filters, tabs, theme, empty state, XSS escaping, API failure.");
})().catch((error) => {console.error(error);process.exitCode=1;dom.window.close();});
