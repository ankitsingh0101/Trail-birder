import os
import tempfile
import traceback

import requests
from flask import Flask, Response, jsonify, request
from werkzeug.exceptions import HTTPException

import identify as tb

app = Flask(__name__)
ALLOWED = {".wav", ".mp3", ".ogg", ".flac"}

PAGE = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Trail Birder</title>
<style>
  :root {
    --bg: #f4f7f2; --card: #ffffff; --text: #1f2d1f; --muted: #5a6b5a;
    --accent: #2f7d32; --accent-soft: #e3f0e3; --border: #dfe7dc;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #121a12; --card: #1b261b; --text: #e8f0e8; --muted: #9db09d;
      --accent: #5fb363; --accent-soft: #243524; --border: #2c3d2c;
    }
  }
  * { box-sizing: border-box; }
  body { font-family: system-ui, sans-serif; background: var(--bg); color: var(--text);
         max-width: 600px; margin: 0 auto; padding: 1.5rem 1rem 3rem; }
  header h1 { margin: 0; font-size: 2rem; }
  header p { margin: .3rem 0 0; color: var(--muted); }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 14px;
          padding: 1.2rem; margin-top: 1rem; }
  .card h2 { margin: 0 0 .6rem; font-size: 1rem; color: var(--muted);
             text-transform: uppercase; letter-spacing: .05em; }
  .drop { display: block; border: 2px dashed var(--border); border-radius: 12px;
          padding: 1.4rem 1rem; text-align: center; cursor: pointer; color: var(--muted); }
  .drop:hover { border-color: var(--accent); color: var(--text); }
  .drop strong { color: var(--text); }
  input[type=file] { display: none; }
  audio { width: 100%; margin-top: 1rem; }
  button { background: var(--accent); color: #fff; border: 0; padding: .8rem 1.2rem;
           border-radius: 10px; font-size: 1rem; cursor: pointer; width: 100%; margin-top: 1rem; }
  button:disabled { opacity: .5; cursor: not-allowed; }
  #status { margin-top: .8rem; color: var(--muted); display: flex; align-items: center; gap: .6rem; }
  .spinner { width: 18px; height: 18px; border: 3px solid var(--border);
             border-top-color: var(--accent); border-radius: 50%;
             animation: spin 0.8s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .error { color: #c0392b; }
  #bird { font-size: 1.5rem; font-weight: 700; }
  #sci { font-style: italic; color: var(--muted); margin-bottom: .8rem; }
  .bar { height: 10px; background: var(--accent-soft); border-radius: 6px; overflow: hidden; }
  .bar div { height: 100%; background: var(--accent); width: 0; transition: width .6s; }
  #meta { color: var(--muted); font-size: .9rem; margin: .4rem 0 1rem; }
  #note { line-height: 1.6; white-space: pre-line; }
  .badge { display: inline-block; background: var(--accent-soft); color: var(--accent);
           padding: .25rem .7rem; border-radius: 99px; font-size: .85rem; margin-top: 1rem; }
  ul { list-style: none; padding: 0; margin: 0; }
  li { display: flex; justify-content: space-between; padding: .55rem 0;
       border-bottom: 1px solid var(--border); }
  li:last-child { border-bottom: 0; }
  li span:last-child { color: var(--muted); font-size: .85rem; }
  .empty { color: var(--muted); }
  footer { margin-top: 2rem; text-align: center; color: var(--muted); font-size: .8rem; }
</style>
</head>
<body>
<header>
  <h1>Trail Birder</h1>
  <p>Identify birds by sound, offline, with open-source AI.</p>
</header>

<div class="card">
  <label class="drop" for="file">
    <strong id="fname">Choose a bird recording</strong><br>
    <small>WAV, MP3, OGG or FLAC</small>
  </label>
  <input type="file" id="file" accept=".wav,.mp3,.ogg,.flac,audio/*">
  <audio id="player" controls hidden></audio>
  <button id="go">Identify bird</button>
  <div id="status"></div>
</div>

<div class="card" id="result" hidden>
  <div id="bird"></div>
  <div id="sci"></div>
  <div class="bar"><div id="conf"></div></div>
  <div id="meta"></div>
  <div id="note"></div>
  <span class="badge" id="life"></span>
</div>

<div class="card">
  <h2>My life list</h2>
  <ul id="list"></ul>
</div>

<footer>BirdNET for sound ID, Gemma through Ollama for field notes. Runs on your machine.</footer>

<script>
const $ = (id) => document.getElementById(id);
let audioUrl = null;

function setStatus(text, opts) {
  const s = $("status");
  s.textContent = "";
  if (opts && opts.spinner) {
    const sp = document.createElement("div");
    sp.className = "spinner";
    s.appendChild(sp);
  }
  const t = document.createElement("span");
  t.textContent = text;
  if (opts && opts.error) t.className = "error";
  s.appendChild(t);
}

$("file").onchange = () => {
  const f = $("file").files[0];
  if (!f) return;
  $("fname").textContent = f.name;
  if (audioUrl) URL.revokeObjectURL(audioUrl);
  audioUrl = URL.createObjectURL(f);
  $("player").src = audioUrl;
  $("player").hidden = false;
  $("result").hidden = true;
  setStatus("");
};

async function loadLifeList() {
  const r = await fetch("/life-list");
  const items = await r.json();
  const ul = $("list");
  ul.textContent = "";
  if (items.length === 0) {
    const li = document.createElement("li");
    li.className = "empty";
    li.textContent = "No birds yet. Identify your first one above.";
    ul.appendChild(li);
    return;
  }
  for (const b of items) {
    const li = document.createElement("li");
    const a = document.createElement("span");
    a.textContent = b.common;
    const c = document.createElement("span");
    c.textContent = "heard " + b.times_heard + "x, last " + b.last_heard;
    li.appendChild(a);
    li.appendChild(c);
    ul.appendChild(li);
  }
}

$("go").onclick = async () => {
  const f = $("file").files[0];
  if (!f) { setStatus("Choose an audio file first.", { error: true }); return; }
  $("go").disabled = true;
  $("result").hidden = true;
  setStatus("Listening... this can take up to a minute.", { spinner: true });
  const form = new FormData();
  form.append("audio", f);
  try {
    const r = await fetch("/identify", { method: "POST", body: form });
    const d = await r.json();
    if (!r.ok) { setStatus(d.error, { error: true }); return; }
    setStatus("");
    $("result").hidden = false;
    $("bird").textContent = d.common;
    $("sci").textContent = d.scientific;
    $("conf").style.width = Math.round(d.confidence * 100) + "%";
    $("meta").textContent = "Confidence " + Math.round(d.confidence * 100) +
      "%, heard in " + d.detections + " clips";
    $("note").textContent = d.note;
    $("life").textContent = (d.is_new ? "New bird! " : "Already on your list. ") +
      d.total + " species so far";
    loadLifeList();
  } catch (e) {
    setStatus("Something went wrong: " + e, { error: true });
  } finally {
    $("go").disabled = false;
  }
};

loadLifeList();
</script>
</body>
</html>
"""


@app.errorhandler(Exception)
def handle_error(e):
    if isinstance(e, HTTPException):
        return e
    traceback.print_exc()
    return jsonify(error=f"Server error: {e}"), 500


@app.route("/")
def home():
    return Response(PAGE, mimetype="text/html")


@app.route("/life-list")
def life_list():
    data = tb.load_json(tb.LIFE_LIST_FILE)
    items = sorted(data.values(), key=lambda b: b.get("last_heard", ""), reverse=True)
    return jsonify(items)


@app.route("/identify", methods=["POST"])
def identify_route():
    upload = request.files.get("audio")
    if upload is None or upload.filename == "":
        return jsonify(error="No file uploaded."), 400

    ext = os.path.splitext(upload.filename)[1].lower()
    if ext not in ALLOWED:
        return jsonify(error="Please use a WAV, MP3, OGG or FLAC file."), 400

    fd, path = tempfile.mkstemp(suffix=ext)
    os.close(fd)
    try:
        upload.save(path)
        bird = tb.identify(path)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass

    if bird is None:
        return jsonify(error="No bird heard clearly. Try a closer or quieter recording."), 422

    is_new, total = tb.update_life_list(bird)
    try:
        note = tb.field_note(bird)
    except requests.exceptions.ConnectionError:
        note = "Ollama isn't running. Start it with `ollama serve` to get field notes."

    return jsonify(**bird, note=note, is_new=is_new, total=total)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)