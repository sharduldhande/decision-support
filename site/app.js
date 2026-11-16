const API_URL = "https://pgwxbs38tj.execute-api.us-east-2.amazonaws.com";
// Cloudflare's always-pass test key when running locally; the real widget elsewhere.
const TURNSTILE_SITE_KEY = location.hostname === "localhost" ? "1x00000000000000000000AA" : "0x4AAAAAAGCykjXr2d6Y9vbX";
const MAX_PDF_BYTES = 20 * 1024 * 1024;
const POLL_MS = 2000;
const GIVE_UP_MS = 180000;
const SAMPLE_URL = "sample_angiogram.pdf";

const STAGES = {
  waiting: "Starting the analysis…",
  reading: "Reading the report…",
  searching: "Searching the ACC/AHA guideline…",
  writing: "Writing the recommendation…",
};

const $ = (id) => document.getElementById(id);
const form = $("upload-form"), fileInput = $("file"), drop = $("drop"), submit = $("submit"), sample = $("sample");
const statusBox = $("status"), statusText = $("status-text"), spinner = $("spinner"), elapsed = $("elapsed");
const resultBox = $("result");

let widgetId = null, token = null, timer = null, busy = false;

function updateButtons() {
  submit.disabled = busy || !(fileInput.files.length && token);
  sample.disabled = busy || !token;
}

window.onTurnstileLoad = () => {
  widgetId = turnstile.render("#turnstile", {
    sitekey: TURNSTILE_SITE_KEY,
    action: "analyze",
    size: "flexible",
    callback: (t) => { token = t; updateButtons(); },
    "expired-callback": () => { token = null; updateButtons(); },
    "error-callback": () => { token = null; updateButtons(); },
  });
};

fileInput.addEventListener("change", () => {
  $("file-label").textContent = fileInput.files[0]?.name || "Choose a PDF";
  updateButtons();
});
drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (e) => {
  e.preventDefault();
  drop.classList.remove("over");
  if (e.dataTransfer.files.length) {
    fileInput.files = e.dataTransfer.files;
    fileInput.dispatchEvent(new Event("change"));
  }
});

function setStatus(text, { busy = true, error = false } = {}) {
  statusBox.hidden = false;
  statusBox.classList.toggle("error", error);
  spinner.hidden = !busy;
  statusText.textContent = text;
}

function startTimer() {
  const start = Date.now();
  elapsed.textContent = "0s";
  timer = setInterval(() => { elapsed.textContent = `${Math.round((Date.now() - start) / 1000)}s`; }, 1000);
}

function stopTimer() {
  clearInterval(timer);
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  analyze(() => fileInput.files[0]);
});

sample.addEventListener("click", () => {
  analyze(async () => {
    const res = await fetch(SAMPLE_URL);
    if (!res.ok) throw new Error("Could not load the sample report. Try downloading it and uploading it instead.");
    return new File([await res.blob()], SAMPLE_URL, { type: "application/pdf" });
  });
});

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// An error reply from our API, keeping the HTTP status so callers can tell
// temporary problems from final ones.
class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

// fetch with a timeout, turning API errors into readable messages.
async function request(url, options = {}) {
  const res = await fetch(url, { ...options, signal: AbortSignal.timeout(20000) });
  if (res.ok) return res;
  if (res.status === 429) throw new ApiError("The demo is busy right now. Try again in a minute.", 429);
  const error = (await res.json().catch(() => ({}))).error;
  throw new ApiError(error || `The server returned an error (${res.status}). Try again in a minute.`, res.status);
}

// Worth asking again: the connection dropped (fetch throws a TypeError), the request
// timed out, the rate limit turned it away, or AWS had a one-off server error.
function isTemporary(err) {
  return err.name === "TypeError" || err.name === "TimeoutError" || err.status === 429 || err.status >= 500;
}

async function analyze(getFile) {
  if (busy || !token) return;
  busy = true;
  updateButtons();
  resultBox.replaceChildren();
  setStatus("Uploading…");
  startTimer();

  try {
    const file = await getFile();
    if (!file) throw new Error("Choose a PDF first.");
    if (file.size > MAX_PDF_BYTES) throw new Error("That PDF is larger than 20 MB. Try a smaller scan.");

    // 1. Pass the Turnstile check and get a one-time upload for a new job.
    const headers = { "x-turnstile-token": token };
    token = null;
    const { job_id, upload } = await (await request(`${API_URL}/uploads`, { method: "POST", headers })).json();

    // 2. Upload the PDF straight to S3. The presigned fields must come before the file.
    const body = new FormData();
    for (const [name, value] of Object.entries(upload.fields)) body.append(name, value);
    body.append("file", file);
    const stored = await fetch(upload.url, { method: "POST", body });
    if (!stored.ok) throw new Error("The upload failed. Try again.");

    // 3. The upload starts the analysis. Poll the job until it finishes.
    await waitForJob(job_id);
  } catch (err) {
    const message = err.name === "TimeoutError"
      ? "The server did not respond. Check your connection and try again."
      : err.message || "Could not reach the server. Check your connection and try again.";
    setStatus(message, { busy: false, error: true });
  } finally {
    stopTimer();
    // Turnstile tokens are single-use: get a fresh one for the next attempt.
    token = null;
    turnstile.reset(widgetId);
    busy = false;
    updateButtons();
  }
}

async function waitForJob(jobId) {
  const deadline = Date.now() + GIVE_UP_MS;
  while (Date.now() < deadline) {
    let job;
    try {
      job = await (await request(`${API_URL}/jobs/${jobId}`)).json();
    } catch (err) {
      // A missed poll doesn't stop the job, which keeps running on AWS. Ask again
      // after the usual pause; only final errors (like 404) end the wait.
      if (!isTemporary(err)) throw err;
      await sleep(POLL_MS);
      continue;
    }
    if (job.status === "done") {
      setStatus(`Done in ${elapsed.textContent}.`, { busy: false });
      renderResult(job.result);
      resultBox.scrollIntoView({ block: "start" });
      return;
    }
    if (job.status === "failed") throw new Error(job.error);
    setStatus(STAGES[job.status] || "Working…");
    await sleep(POLL_MS);
  }
  throw new Error("This is taking longer than expected. Try again.");
}

// ---- rendering (textContent only: model output is never inserted as HTML) ----

const SVG_NS = "http://www.w3.org/2000/svg";

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) node[k] = v;
  node.append(...children.filter((c) => c !== null && c !== undefined && c !== ""));
  return node;
}

function warningIcon() {
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("fill", "none");
  svg.setAttribute("stroke", "var(--red)");
  svg.setAttribute("stroke-width", "2.2");
  svg.setAttribute("aria-hidden", "true");
  for (const d of ["M12 3 2 20h20L12 3Z", "M12 10v4.5M12 17v.5"]) {
    const path = document.createElementNS(SVG_NS, "path");
    path.setAttribute("d", d);
    path.setAttribute("stroke-linejoin", "round");
    path.setAttribute("stroke-linecap", "round");
    svg.append(path);
  }
  return svg;
}

function list(items) {
  return items?.length ? el("ul", {}, ...items.map((i) => el("li", {}, i))) : el("p", { className: "none" }, "None noted.");
}

function renderResult({ findings, decision, message, note }) {
  if (note) resultBox.append(el("div", { className: "panel flag" }, warningIcon(), el("p", {}, el("strong", {}, "Note: "), note)));
  if (message) resultBox.append(el("div", { className: "panel" }, el("p", { className: "verdict" }, message)));
  if (decision) renderDecision(decision);
  if (findings) renderFindings(findings);
  resultBox.append(el("p", { className: "fine" },
    el("strong", {}, "Demo only. "), "Not a medical device; does not replace clinical judgement or a Heart Team discussion."));
}

function renderDecision(d) {
  resultBox.append(el("div", { className: "panel" },
    el("p", { className: "verdict" }, d.recommendation, el("span", { className: "tag" }, `${d.confidence} confidence`)),
    el("p", {}, d.primary_rationale),
    d.urgent_issues && el("div", { className: "flag" }, warningIcon(), el("p", {}, el("strong", {}, "Urgent: "), d.urgent_issues)),
    d.next_steps?.length ? el("div", { className: "next" },
      el("h3", {}, "Next steps"),
      el("ol", {}, ...d.next_steps.map((s) => el("li", {}, s))),
    ) : null,
  ));

  resultBox.append(el("div", { className: "panel factors" },
    el("div", {}, el("h3", {}, "Favours PCI"), list(d.conditions_favouring_pci)),
    el("div", {}, el("h3", {}, "Favours CABG"), list(d.conditions_favouring_cabg)),
    el("div", {}, el("h3", {}, "Favours medical therapy"), list(d.conditions_favouring_medical_management)),
  ));

  if (d.supporting_citations?.length) {
    resultBox.append(el("div", { className: "panel" }, el("h3", {}, "Guideline citations"),
      ...d.supporting_citations.map((c) => el("div", { className: "cite" },
        el("div", { className: "head" },
          c.class_of_recommendation && el("span", { className: "tag" }, `COR ${c.class_of_recommendation}`),
          c.level_of_evidence && el("span", { className: "tag" }, `LOE ${c.level_of_evidence}`),
          el("span", {}, c.section_title)),
        el("blockquote", {}, c.guideline_text),
        el("p", { className: "why" }, el("strong", {}, "Why it applies: "), c.patient_relevance),
      ))));
  }
}

function renderFindings(f) {
  const ctx = f.clinical_context, dx = f.angiographic_diagnosis;
  const facts = [
    ["Diagnosis", ctx.cardiac_diagnosis],
    ["Comorbidities", ctx.comorbidities.join(", ")],
    ["Valves", ctx.valvular_disease],
    ["LV function", ctx.lv_function],
    ["Dominance", f.coronary_anatomy.dominance],
    ["Report's diagnosis line", dx.summary],
    ["Significant vessels (computed)", f.significant_vessels.join(", ") || "none"],
  ].filter(([, v]) => v);

  const rows = f.lesions.map((l) => el("tr", {},
    el("td", {}, `${l.vessel} ${l.segment}`.trim()),
    el("td", {}, l.stenosis_pct ? `${l.stenosis_pct.join("–")}%` : l.stenosis_qualifier.replace("_", " ")),
    el("td", {}, l.severity)));

  resultBox.append(el("div", { className: "panel" }, el("h3", {}, "Extracted findings"),
    el("table", {}, ...facts.map(([k, v]) => el("tr", {}, el("th", {}, k), el("td", {}, v)))),
    rows.length ? el("table", { className: "lesions" },
      el("tr", {}, el("th", {}, "Lesion"), el("th", {}, "Stenosis"), el("th", {}, "Severity")), ...rows)
      : el("p", { className: "none" }, "No significant or intermediate lesions found."),
  ));
}
