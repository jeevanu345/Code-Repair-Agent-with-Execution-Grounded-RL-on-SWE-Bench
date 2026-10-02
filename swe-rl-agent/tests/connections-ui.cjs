const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const {JSDOM} = require("jsdom");
const root = path.join(__dirname,"../src/swe_rl/dashboard/static");
const dom = new JSDOM(fs.readFileSync(path.join(root,"index.html"),"utf8"),{
  url:"http://localhost",runScripts:"outside-only"
});
const w = dom.window, $ = (id) => w.document.getElementById(id);
const catalog = [
  {id:"vllm",name:"Local vLLM",base_url:"",requires_key:false},
  {id:"openai",name:"OpenAI",base_url:"https://api.openai.com/v1",requires_key:true,docs:"https://platform.openai.com/docs"}
];
const calls = [];
w.AbortSignal.timeout=()=>undefined;
w.fetch = async (url,options) => {
  calls.push({url,options});
  let data;
  if(url==="/api/providers") data={providers:catalog,active:{provider:"vllm",model:"local-model",base_url:"http://127.0.0.1:8000/v1",has_key:false},settings_token:"test-token"};
  else {
    assert.equal(options.headers["X-Settings-Token"],"test-token");
    if(url.endsWith("/settings")) data={provider:"openai",model:"account-model",base_url:"https://api.openai.com/v1",has_key:true};
    if(url.endsWith("/models")) data={models:["account-model"]};
    if(url.endsWith("/test")) data={ok:true,tokens_in:1,tokens_out:2};
    if(url.endsWith("/assist")) data={text:"<script>example</script>",provider:"openai",model:"account-model",tokens_in:10,tokens_out:4};
  }
  return {ok:true,json:async()=>data};
};
w.eval(fs.readFileSync(path.join(root,"connections.js"),"utf8"));
const tick = () => new Promise((resolve)=>setImmediate(resolve));
(async()=>{
  await tick();
  assert.equal($("provider").options.length,2);
  $("provider").value="openai";$("provider").dispatchEvent(new w.Event("change"));
  assert.equal($("model-url").value,"https://api.openai.com/v1");
  assert.equal($("model-url").readOnly,true);
  $("model-id").value="account-model";$("model-key").value="synthetic-test-key";
  $("model-form").dispatchEvent(new w.Event("submit",{cancelable:true}));await tick();
  assert.equal(JSON.parse(calls.at(-1).options.body).api_key,"synthetic-test-key");
  assert.equal($("model-key").value,"");
  assert.match($("active-model").textContent,/OpenAI/);
  assert.equal(w.localStorage.length,0);
  $("discover-models").click();await tick();
  assert.equal($("model-options").children.length,1);
  $("test-model").click();await tick();
  assert.match($("model-status").textContent,/verified/);
  $("assist-prompt").value="Fix code";
  $("assist-form").dispatchEvent(new w.Event("submit",{cancelable:true}));await tick();
  assert.equal($("assist-result").textContent,"<script>example</script>");
  assert.equal($("assist-result").querySelector("script"),null);
  w.fetch = async()=>({ok:false,json:async()=>({detail:"Invalid API key"})});
  $("test-model").click();await tick();
  assert.match($("model-status").textContent,/Invalid API key/);
  assert.equal($("test-model").disabled,false);
  dom.window.close();
  console.log("PASS: provider selection, key submission/redaction, model discovery, testing, assistant, errors.");
})().catch((error)=>{console.error(error);process.exitCode=1;dom.window.close();});
