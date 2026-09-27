"use strict";

const $ = (id) => document.getElementById(id);
const video = $("video");
const distributionToggle = $("show-video-distribution");
distributionToggle.checked = false;
let bundle = null, trace = null, selectedModel = null, loadGeneration = 0;
let lastWindow = -2, lastTruth = null, lastPrompt = null;
const colors = ["#64e3b0", "#7fabff", "#d1a6ff", "#f4c074", "#74d6de", "#ee9cac", "#99c887", "#b7b9fb"];

function floorIndex(items, time, getter) {
  let lo = 0, hi = items.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (getter(items[mid]) <= time) lo = mid + 1; else hi = mid;
  }
  return lo - 1;
}

function nameOf(label) {
  if (label === -1) return "Unconstrained / excluded";
  const i = bundle.labels.indexOf(label);
  return i >= 0 ? bundle.gesture_names[i] : "—";
}

function showError(error) {
  $("status").textContent = error.message || String(error);
  $("status").className = "error";
}

async function checkedFetch(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Could not load ${url} (${response.status})`);
  return response;
}

function populateProbabilityRows(container) {
  container.replaceChildren(...bundle.labels.map((label, i) => {
    const row = document.createElement("div"); row.className = "probability";
    const labelNode = document.createElement("span"); labelNode.textContent = bundle.gesture_names[i];
    const track = document.createElement("div"); track.className = "track";
    const fill = document.createElement("div"); fill.className = "fill"; track.append(fill);
    const number = document.createElement("span"); number.className = "number"; number.textContent = "—";
    row.append(labelNode, track, number); row.dataset.label = label;
    return row;
  }));
}

function clearVideoDistribution(message) {
  $("video-probabilities").hidden = false;
  $("video-probabilities").setAttribute("aria-label", `${message} Zero values are display placeholders.`);
  $("video-distribution-status").hidden = true;
  $("video-distribution-status").textContent = message;
  for (const row of $("video-probabilities").children) {
    row.classList.remove("selected");
    row.setAttribute("aria-label", `${row.firstElementChild.textContent}: 0.0%, unavailable`);
    row.querySelector(".fill").style.width = "0%";
    row.querySelector(".number").textContent = "0.0%";
  }
}

function updateVideoDistribution(probabilities) {
  const maximum = Math.max(...probabilities);
  const maximumIndex = probabilities.indexOf(maximum);
  $("video-probabilities").hidden = false;
  $("video-probabilities").setAttribute("aria-label", "Live class probabilities");
  $("video-distribution-status").hidden = true;
  [...$("video-probabilities").children].forEach((row, i) => {
    const percent = 100 * probabilities[i], isMaximum = i === maximumIndex;
    row.classList.toggle("selected", isMaximum);
    row.querySelector(".fill").style.width = `${percent}%`;
    row.querySelector(".number").textContent = `${percent.toFixed(1)}%`;
    row.setAttribute("aria-label", `${bundle.gesture_names[i]}: ${percent.toFixed(1)}%${isMaximum ? ", highest probability" : ""}`);
  });
}

function setVideoOverlayMode() {
  $("video-label-overlay").hidden = distributionToggle.checked;
  $("video-distribution-overlay").hidden = !distributionToggle.checked;
  lastWindow = -2; // Switching while paused must render the current window immediately.
  update();
}

async function loadRecording(stem) {
  const generation = ++loadGeneration;
  video.pause();
  bundle = null; trace = null; selectedModel = null;
  $("overlay-label").textContent = "Loading predictions…";
  $("overlay-confidence").textContent = "";
  clearVideoDistribution("Loading predictions…");
  $("status").className = "";
  $("status").textContent = "Loading synchronized traces and predictions…";
  const [metadata, binary] = await Promise.all([
    checkedFetch(`/api/replay/${stem}`).then((r) => r.json()),
    checkedFetch(`/api/traces/${stem}`).then((r) => r.arrayBuffer()),
  ]);
  if (generation !== loadGeneration) return;
  bundle = metadata; trace = new Float32Array(binary);
  if (trace.length !== bundle.trace.rows * bundle.trace.stride) throw new Error("Trace dimensions do not match the replay manifest");
  $("model").replaceChildren(...bundle.models.map((model) => {
    const option = document.createElement("option");
    option.value = model.id; option.textContent = model.label;
    return option;
  }));
  populateProbabilityRows($("probabilities"));
  populateProbabilityRows($("video-probabilities"));
  video.src = `/video/${stem}`;
  video.load();
  $("session").textContent = `${bundle.split_role === "training" ? "Training" : "Held out"} · Participant ${bundle.subject} · ${bundle.date} · ${bundle.trajectory}`;
  $("coverage").textContent = `${(100 * bundle.windows.filter((w) => w.mask >= 0).length / bundle.windows.length).toFixed(1)}%`;
  lastWindow = -2; lastTruth = null; lastPrompt = null;
  selectModel();
  $("status").textContent = bundle.split_role === "training"
    ? "Ready. Training replay: these samples were used during fitting; scores are in-sample."
    : "Ready. Held-out replay: this participant was excluded from fitting and calibration.";
}

function selectModel() {
  if (!bundle) return;
  selectedModel = bundle.models.find((m) => m.id === $("model").value) || bundle.models[0];
  document.querySelector(".video-distribution-heading").textContent = `${selectedModel.name} Prediction Distribution`;
  $("prediction-heading").textContent = `${selectedModel.name} EMG PREDICTION`;
  $("accuracy").textContent = `${(selectedModel.metrics.accuracy * 100).toFixed(1)}%`;
  $("f1").textContent = selectedModel.metrics.macro_f1.toFixed(3);
  $("recall").textContent = selectedModel.metrics.macro_recall.toFixed(3);
  const training = selectedModel.train_subjects;
  const inTraining = bundle.split_role === "training";
  $("replay-role-badge").textContent = inTraining ? "OFFLINE · TRAINING REPLAY" : "OFFLINE · HELD-OUT REPLAY";
  $("accuracy-scope").textContent = inTraining ? "Current training recording · in-sample" : "Current held-out recording";
  $("protocol-note").textContent = inTraining
    ? `One pooled model trained on ${training.length} participants · Participant ${bundle.subject} is in the training pool · In-sample replay`
    : `One pooled model trained on ${training.length} participants · Participant ${bundle.subject} is held out across both days · Zero calibration`;
  $("provenance").textContent = `${selectedModel.label}. This is ${inTraining ? "an in-sample training recording, excluded from the held-out comparison scores" : "a held-out recording"}. Training participants: ${training.join(", ")}. Held out: ${selectedModel.test_subjects.join(", ")}. Both days are included in each role. Observed timestamp rate: ${bundle.observed_sample_rate_hz.toFixed(3)} Hz. VIDEO_STAMP − sample time: ${bundle.video_minus_sample_seconds.toFixed(6)} s. ${bundle.alignment}. Shared fit: ${selectedModel.fit_id}.`;
  lastWindow = -2;
  update();
}

function updateLabels(time) {
  for (const [source, target] of [["ground_truth", "truth"], ["prompts", "prompt"]]) {
    const i = floorIndex(bundle[source], time, (s) => s.start);
    const segment = bundle[source][i];
    const label = segment && time < segment.end ? segment.label : null;
    if (target === "truth" && label !== lastTruth) { $(target).textContent = nameOf(label); lastTruth = label; }
    if (target === "prompt" && label !== lastPrompt) { $(target).textContent = nameOf(label); lastPrompt = label; }
  }
  const i = floorIndex(bundle.windows, time, (w) => w.time);
  if (i === lastWindow) return;
  lastWindow = i;
  const window = bundle.windows[i];
  const prediction = selectedModel.predictions[i];
  const probabilities = selectedModel.probabilities[i];
  const scored = Boolean(window && window.mask >= 0 && prediction != null);
  const title = scored ? nameOf(prediction) : (window ? "Unscored interval" : "Waiting for window");
  $("prediction").textContent = title; $("overlay-label").textContent = title;
  $("score-badge").textContent = scored ? "SCORED WINDOW" : "EXCLUDED";
  $("score-badge").className = `pill${scored ? " active" : ""}`;
  const confidence = scored && probabilities ? probabilities[bundle.labels.indexOf(prediction)] : null;
  $("confidence").textContent = confidence == null ? "—" : `${(confidence * 100).toFixed(1)}%`;
  $("overlay-confidence").textContent = confidence == null ? "" : `${(confidence * 100).toFixed(1)}%`;
  $("confidence-caption").textContent = scored && !probabilities ? "Original SVM: decision-only output" : "Predicted-class probability";
  [...$("probabilities").children].forEach((row, j) => {
    const value = scored && probabilities ? probabilities[j] : null;
    row.classList.toggle("selected", scored && bundle.labels[j] === prediction);
    row.querySelector(".fill").style.width = `${value == null ? 0 : 100 * value}%`;
    row.querySelector(".number").textContent = value == null ? "—" : `${Math.round(value * 100)}%`;
  });
  if (scored && probabilities) {
    updateVideoDistribution(probabilities);
  } else {
    clearVideoDistribution(scored
      ? `Probability distribution unavailable for ${selectedModel.name}. Prediction: ${nameOf(prediction)}.`
      : title);
  }
}

function drawTraces(time) {
  const start = Math.max(0, time - 5), end = start + 6;
  const activeWindow = bundle.windows[floorIndex(bundle.windows, time, (w) => w.time)];
  const stride = bundle.trace.stride, dpr = window.devicePixelRatio || 1;
  let lo = 0, hi = bundle.trace.rows;
  while (lo < hi) { const mid = (lo + hi) >>> 1; if (trace[mid * stride] < start) lo = mid + 1; else hi = mid; }

  for (const [id, firstChannel] of [["traces-elbow", 1], ["traces-middle", 9], ["traces-wrist", 17]]) {
    const canvas = $(id), ctx = canvas.getContext("2d");
    const width = canvas.clientWidth, height = canvas.clientHeight;
    if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
      canvas.width = Math.round(width * dpr); canvas.height = Math.round(height * dpr);
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, width, height);
    const left = 50, right = width - 18, top = 13, bottom = height - 28;
    const scale = (right - left) / (end - start), rowHeight = (bottom - top) / 8;
    const xOf = (t) => left + (t - start) * scale;
    ctx.save(); ctx.beginPath(); ctx.rect(left, top, right - left, bottom - top); ctx.clip();
    if (activeWindow) {
      ctx.fillStyle = activeWindow.mask >= 0 ? "#7fabff17" : "#ffffff08";
      ctx.fillRect(xOf(activeWindow.start), top, (activeWindow.time - activeWindow.start) * scale, bottom - top);
    }
    for (let c = 0; c < 8; c++) {
      const channel = firstChannel + c, y = top + rowHeight * (c + .5);
      ctx.strokeStyle = "#263444"; ctx.lineWidth = .7; ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(right, y); ctx.stroke();
      ctx.strokeStyle = colors[c]; ctx.lineWidth = .85; ctx.beginPath();
      const amplitude = rowHeight * .43 / bundle.trace.scales[channel - 1];
      for (let i = lo; i < bundle.trace.rows && trace[i * stride] <= end; i++) {
        const x = xOf(trace[i * stride]);
        const low = Math.max(-rowHeight * .47, Math.min(rowHeight * .47, trace[i * stride + channel * 2 - 1] * amplitude));
        const high = Math.max(-rowHeight * .47, Math.min(rowHeight * .47, trace[i * stride + channel * 2] * amplitude));
        ctx.moveTo(x, y - low); ctx.lineTo(x, y - high);
      }
      ctx.stroke();
    }
    ctx.strokeStyle = "#64e3b0"; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(xOf(time), top); ctx.lineTo(xOf(time), bottom); ctx.stroke();
    ctx.restore(); ctx.font = "10px monospace"; ctx.fillStyle = "#96a7ba"; ctx.textAlign = "left";
    for (let c = 0; c < 8; c++) ctx.fillText(`CH ${firstChannel + c}`, 10, top + rowHeight * (c + .5) + 3);
    ctx.textAlign = "center";
    const tickStep = right - left < 300 ? 2 : 1;
    for (let t = Math.ceil(start / tickStep) * tickStep; t <= end; t += tickStep) ctx.fillText(`${t}s`, xOf(t), height - 9);
  }
}

function update() {
  if (!bundle || !trace || !selectedModel) return;
  const time = video.currentTime || 0;
  $("clock").textContent = `${String(Math.floor(time / 60)).padStart(2, "0")}:${(time % 60).toFixed(2).padStart(5, "0")}`;
  updateLabels(time); drawTraces(time);
}

$("recording").addEventListener("change", () => loadRecording($("recording").value).catch(showError));
$("model").addEventListener("change", selectModel);
distributionToggle.addEventListener("change", setVideoOverlayMode);
$("speed").addEventListener("change", () => { video.playbackRate = Number($("speed").value); });
$("next-gesture").addEventListener("click", () => {
  const next = bundle?.ground_truth.find((s) => s.label > 0 && s.start > video.currentTime + .3);
  if (next) video.currentTime = Math.min(next.start + .7, (next.start + next.end) / 2);
});
for (const event of ["timeupdate", "seeked", "loadedmetadata", "pause"]) video.addEventListener(event, update);
video.addEventListener("loadedmetadata", () => { video.playbackRate = Number($("speed").value); });
window.addEventListener("resize", update);
if ("requestVideoFrameCallback" in video) {
  const frame = () => { update(); video.requestVideoFrameCallback(frame); };
  video.requestVideoFrameCallback(frame);
} else {
  const frame = () => { if (!video.paused) update(); requestAnimationFrame(frame); };
  requestAnimationFrame(frame);
}

(async () => {
  const recordings = await checkedFetch("/api/recordings").then((r) => r.json());
  if (!recordings.length) throw new Error("No replay bundles yet. Run emgbench snapshot restore, then emgbench demo.");
  const groups = [
    ["held_out", "Held-out participants · unseen"],
    ["training", "Training participants · in-sample"],
  ].map(([role, label]) => {
    const group = document.createElement("optgroup"); group.label = label;
    for (const recording of recordings.filter((item) => (item.split_role || "held_out") === role)) {
      const option = document.createElement("option"); option.value = recording.recording;
      option.textContent = `Participant ${recording.subject} · ${recording.date} · ${recording.trajectory}`;
      group.append(option);
    }
    return group;
  });
  $("recording").replaceChildren(...groups.filter((group) => group.children.length));
  await loadRecording(recordings[0].recording);
})().catch(showError);
