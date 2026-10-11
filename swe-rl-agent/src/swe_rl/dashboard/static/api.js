/* Same-origin API access. The authorization token stays in memory. */
(() => {
  "use strict";
  let tokenPromise;
  const providers = () => tokenPromise ||= fetch("/api/providers", {signal:AbortSignal.timeout(10000),credentials:"same-origin"}).then(async response => {
    if (!response.ok) throw new Error("Model settings could not load. Refresh the page.");
    return response.json();
  }).catch(error => {tokenPromise = null; throw error;});
  async function request(url, options = {}) {
    const method = (options.method || "GET").toUpperCase();
    const headers = {Accept:"application/json", ...(options.body ? {"Content-Type":"application/json"} : {}), ...options.headers};
    if (!["GET", "HEAD"].includes(method)) headers["X-Settings-Token"] = (await providers()).settings_token;
    const response = await fetch(url, {...options, method, headers, credentials:"same-origin", signal:options.signal || AbortSignal.timeout(90000)});
    const contentType = response.headers?.get?.("content-type") || "application/json";
    const data = response.status === 204 ? null : /json/.test(contentType) ? await response.json() : await response.text();
    if (!response.ok) throw new Error(typeof data?.detail === "string" ? data.detail : `Request failed (${response.status}).`);
    return data;
  }
  window.SWERLAPI = {request,providers};
})();
