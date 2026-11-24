// VisionAId — sound rendering of the guidance on the phone (Web Audio).
// The computer decides what to play ("sound" messages, see docs/protocol.md); this file only
// executes it, identically to the offline rendering (visionaid/audio/renderer.py):
// - path: repeated note (sound 2, or sound 4 while searching), placed by a stereo pan
//   (StereoPannerNode, constant power: pan −1 = fully left, nothing on the right);
// - obstacles: looping pad of sound 1, overlaid, volume according to distance;
// - arrival: sound 3, once;
// - sentences ("speech" messages): browser speech synthesis (speechSynthesis); during a
//   "duck" sentence (object found), the path layer is suspended, obstacles stay.
"use strict";

class GuideAudio {
  constructor() {
    this.ctx = null;
    this.cfg = null;
    this.buffers = {};
    this.ready = false;
    this.active = false; // guidance running (Start button)
    this.cmd = null; // last instruction received
    this.lastMsg = 0; // performance.now() of the last instruction
    this.prevMode = "silent";
    this.next = 0; // next path note (audio time)
    this.loop = null; // looping source of the obstacle pad
    this.timer = null;
    this.userDb = 0;
    this.onStatus = () => {};
    this.voiceLang = "fr-FR";
    this.voiceRate = 1;
    this.duckUntil = 0; // performance.now(): path layer suspended until this instant
    this.speaking = 0; // "duck" sentences in progress
    try { this.userDb = +(localStorage.getItem("visionaid.volume_db") || 0); } catch (e) { /* storage unavailable */ }
  }

  // To be called INSIDE the user gesture (button): iOS only allows sound this way.
  unlock() {
    if (!this.ctx) {
      const AC = window.AudioContext || window.webkitAudioContext;
      if (!AC) { this.onStatus("Sound unavailable in this browser", "err"); return; }
      this.ctx = new AC();
      // iOS ≥ 16.4: plays even when the silent switch is on
      try { if (navigator.audioSession) navigator.audioSession.type = "playback"; } catch (e) { /* ancien iOS */ }
      const c = this.ctx;
      this.master = c.createGain();
      this.master.connect(c.destination);
      // path layer: notes → stereo pan → output
      this.pan = c.createStereoPanner();
      this.pan.connect(this.master);
      // obstacle layer: looping pad → variable gain → output
      this.obGain = c.createGain();
      this.obGain.gain.value = 0;
      this.obGain.connect(this.master);
    }
    this.ctx.resume();
    // iOS: speech synthesis is also unlocked inside a user gesture
    try {
      if (!this.speechUnlocked && window.speechSynthesis) {
        const u = new SpeechSynthesisUtterance(" ");
        u.volume = 0;
        window.speechSynthesis.speak(u);
        this.speechUnlocked = true;
      }
    } catch (e) { /* no speech synthesis */ }
    // silent buffer: finishes unlocking the audio output on iOS
    const b = this.ctx.createBuffer(1, 1, this.ctx.sampleRate), s = this.ctx.createBufferSource();
    s.buffer = b; s.connect(this.ctx.destination); s.start();
  }

  // Settings from the computer ("welcome" message): loads the sounds.
  async configure(a) {
    if (!a) { this.onStatus("The computer did not send any sounds", "warn"); return; }
    if (!this.ctx) return;
    this.cfg = a;
    this._applyVolume();
    try {
      await Promise.all(Object.entries(a.sounds).map(async ([k, url]) => {
        if (this.buffers[k] && this.buffers[k].url === url) return;
        const r = await fetch(url);
        if (!r.ok) throw new Error(`${url} : ${r.status}`);
        const ab = await r.arrayBuffer();
        const buf = await new Promise((res, rej) => this.ctx.decodeAudioData(ab, res, rej));
        buf.url = url;
        this.buffers[k] = buf;
      }));
      this.ready = true;
      this.onStatus("Sound ready", "ok");
      if (!this.timer) this.timer = setInterval(() => this._tick(), 25);
    } catch (e) {
      this.onStatus("Sounds not found: " + (e.message || e), "err");
    }
  }

  start() {
    this.active = true; this.cmd = null; this.lastMsg = 0; this.prevMode = "silent";
    this._startLoop();
  }

  stop() {
    this.active = false; this.cmd = null; this.prevMode = "silent";
    if (this.obGain) this._setObstacle(-120, 0.05);
    if (this.loop) { try { this.loop.stop(this.ctx.currentTime + 0.3); } catch (e) { /* already stopped */ } this.loop = null; }
  }

  // Instruction from the computer (~20 per second).
  onSound(m) {
    this.cmd = m;
    this.lastMsg = performance.now();
    if (!this.ready) return;
    if (m.event === "arrived") this._play("arrived", this.master, this.ctx.currentTime + 0.02);
    if (m.mode === "guide" || m.mode === "search") this._setPan(m.pan || 0, false);
    this._setObstacle(m.obstacle_db ?? -120);
  }

  resume() { if (this.ctx) this.ctx.resume(); }

  // Sentence spoken in the headset. `duck`: path suspended until the end of the sentence.
  speak(text, duck) {
    const ss = window.speechSynthesis;
    if (!ss || !text) return;
    const u = new SpeechSynthesisUtterance(text);
    u.lang = this.voiceLang || "fr-FR";
    u.rate = this.voiceRate || 1;
    const voices = ss.getVoices ? ss.getVoices() : [];
    const lang = u.lang.toLowerCase();
    const v = voices.find((x) => x.lang && x.lang.toLowerCase() === lang && x.localService)
      || voices.find((x) => x.lang && x.lang.toLowerCase().startsWith(lang.slice(0, 2)));
    if (v) u.voice = v;
    if (duck) {
      this.speaking += 1;
      // safeguard: never more than the probable duration of the sentence (≈ 14 characters/s) + 3 s
      this.duckUntil = Math.max(this.duckUntil, performance.now() + (text.length / 14) * 1000 / u.rate + 3000);
      const end = () => {
        this.speaking = Math.max(0, this.speaking - 1);
        if (!this.speaking) this.duckUntil = performance.now() + 300;
      };
      u.onend = end; u.onerror = end;
    }
    this.resume();
    ss.speak(u); // after the sentences already in progress
  }

  cancelSpeech() {
    try { if (window.speechSynthesis) window.speechSynthesis.cancel(); } catch (e) { /* rien */ }
    this.speaking = 0; this.duckUntil = 0;
  }

  // Short beep: "speak now".
  beep() {
    if (!this.ctx) return;
    const c = this.ctx, o = c.createOscillator(), g = c.createGain(), t = c.currentTime + 0.01;
    o.frequency.value = 880;
    g.gain.setValueAtTime(0, t);
    g.gain.linearRampToValueAtTime(0.25, t + 0.01);
    g.gain.exponentialRampToValueAtTime(0.001, t + 0.18);
    o.connect(g); g.connect(this.master || c.destination);
    o.start(t); o.stop(t + 0.2);
    o.onended = () => { o.disconnect(); g.disconnect(); };
  }

  setVolume(db) {
    this.userDb = db;
    try { localStorage.setItem("visionaid.volume_db", String(db)); } catch (e) { /* storage unavailable */ }
    this._applyVolume();
  }

  // Headset check: left, right, front.
  test() {
    if (!this.ready) return false;
    const t = this.ctx.currentTime + 0.05;
    [[-1, 0], [1, 1.0], [0, 2.0]].forEach(([p, dt]) => {
      this._setPan(p, t + dt);
      this._play("guide", this.pan, t + dt + 0.01);
    });
    return true;
  }

  // ---------------------------------------------------------------- internal
  _applyVolume() {
    if (!this.master) return;
    const db = (this.cfg ? this.cfg.master_gain_db : 0) + this.userDb;
    this.master.gain.setTargetAtTime(Math.pow(10, db / 20), this.ctx.currentTime, 0.05);
  }

  // `at`: false = smoothed transition now, number = jump at this audio instant
  _setPan(p, at) {
    const c = this.ctx, tau = this.cfg ? this.cfg.smoothing_s : 0.05;
    const v = Math.max(-1, Math.min(1, p));
    if (typeof at === "number") this.pan.pan.setValueAtTime(v, at);
    else this.pan.pan.setTargetAtTime(v, c.currentTime, tau);
  }

  _setObstacle(db, tau) {
    const t = tau ?? (this.cfg ? this.cfg.obstacle_smoothing_s : 0.3);
    const g = db <= -120 ? 0 : Math.pow(10, db / 20);
    this.obGain.gain.setTargetAtTime(g, this.ctx.currentTime, t);
  }

  _startLoop() {
    if (!this.ready || this.loop || !this.buffers.obstacle) return;
    const c = this.ctx, src = c.createBufferSource(), env = c.createGain();
    src.buffer = this.buffers.obstacle;
    src.loop = true; // prepared file, seamlessly loopable
    env.gain.value = Math.pow(10, (this.cfg.gains_db.obstacle || 0) / 20);
    src.connect(env); env.connect(this.obGain);
    src.start(c.currentTime + 0.02);
    src.onended = () => { src.disconnect(); env.disconnect(); };
    this.loop = src;
  }

  _play(key, dest, when) {
    const buf = this.buffers[key];
    if (!buf) return;
    const c = this.ctx, src = c.createBufferSource(), env = c.createGain();
    src.buffer = buf;
    env.gain.value = Math.pow(10, (this.cfg.gains_db[key] || 0) / 20);
    src.connect(env); env.connect(dest);
    src.start(when);
    src.onended = () => { src.disconnect(); env.disconnect(); };
  }

  _tick() {
    if (!this.ready || !this.active) return;
    if (!this.loop) this._startLoop(); // sounds loaded after Start
    const now = this.ctx.currentTime;
    let mode = this.cmd ? this.cmd.mode : "silent";
    // no more instructions from the computer (Wi-Fi cut, computer stopped): "stop" rather than a frozen guide
    if (this.lastMsg > 0 && performance.now() - this.lastMsg > this.cfg.watchdog_s * 1000) {
      mode = "lost";
      if (this.prevMode !== "lost") this._setObstacle(this.cfg.lost_obstacle_db);
    }
    const ducked = performance.now() < this.duckUntil;
    if ((mode === "guide" || mode === "search") && !ducked) {
      // new rhythm as soon as search ↔ path switches
      if (mode !== this.prevMode || this.next < now) this.next = now + 0.02;
      const key = mode === "search" && this.buffers.search ? "search" : "guide";
      while (this.next < now + 0.1) {
        this._play(key, this.pan, this.next);
        this.next += Math.max(0.2, this.cmd.interval || 1);
      }
    }
    this.prevMode = ducked ? "silent" : mode; // end of sentence: the rhythm restarts immediately
  }
}

window.guideAudio = new GuideAudio();
