/* nano-vllm-omni drills · hints — steppers, theme, progress (no dependencies) */
(function () {
  "use strict";

  // localStorage can be missing or throw (file://, private mode) -> degrade silently.
  var store = {
    get: function (k, d) {
      try { var v = window.localStorage.getItem(k); return v === null ? d : JSON.parse(v); } catch (e) { return d; }
    },
    set: function (k, v) {
      try { window.localStorage.setItem(k, JSON.stringify(v)); } catch (e) { /* ignore */ }
    }
  };
  var DONE_KEY = "nvo-drills-done";

  // ---------------------------------------------------------------- theme
  function applyTheme(t) {
    if (t) document.documentElement.setAttribute("data-theme", t);
    else document.documentElement.removeAttribute("data-theme");
  }
  applyTheme(store.get("nvo-theme", null));
  function initTheme() {
    var btn = document.getElementById("theme-btn");
    if (!btn) return;
    btn.addEventListener("click", function () {
      var cur = document.documentElement.getAttribute("data-theme");
      var dark = cur ? cur === "dark" : window.matchMedia("(prefers-color-scheme: dark)").matches;
      var next = dark ? "light" : "dark";
      applyTheme(next);
      store.set("nvo-theme", next);
    });
  }

  // ---------------------------------------------------------------- steppers
  // <div class="stepper" data-diagram="id?"> .st-head / .step[data-hl="a,b"] ...
  function initStepper(root) {
    var steps = Array.prototype.slice.call(root.querySelectorAll(":scope > .st-body > .step"));
    if (!steps.length) return;
    root.classList.add("js");
    var count = root.querySelector(".st-count");
    var prev = root.querySelector(".st-prev");
    var next = root.querySelector(".st-next");
    var dotsBox = root.querySelector(".st-dots");
    var diagram = root.querySelector(".st-diagram");
    var seen = {};
    var dots = steps.map(function (s, i) {
      var b = document.createElement("button");
      b.className = "st-dot";
      b.type = "button";
      b.textContent = String(i + 1);
      b.title = (s.querySelector(".step-title") || {}).textContent || "";
      b.addEventListener("click", function () { show(i); });
      dotsBox.appendChild(b);
      return b;
    });
    var cur = 0;
    function show(i) {
      cur = Math.max(0, Math.min(steps.length - 1, i));
      seen[cur] = true;
      steps.forEach(function (s, j) { s.classList.toggle("active", j === cur); });
      dots.forEach(function (d, j) { d.classList.toggle("on", j === cur); d.classList.toggle("seen", !!seen[j]); });
      count.textContent = (cur + 1) + " / " + steps.length;
      prev.disabled = cur === 0;
      next.disabled = cur === steps.length - 1;
      if (diagram) {
        var hl = (steps[cur].getAttribute("data-hl") || "").split(",").map(function (x) { return x.trim(); }).filter(Boolean);
        Array.prototype.forEach.call(diagram.querySelectorAll("[data-node]"), function (el) {
          el.classList.toggle("hl", hl.indexOf(el.getAttribute("data-node")) >= 0);
        });
      }
    }
    prev.addEventListener("click", function () { show(cur - 1); });
    next.addEventListener("click", function () { show(cur + 1); });
    root.setAttribute("tabindex", "0");
    root.addEventListener("keydown", function (e) {
      if (e.target !== root) return;
      if (e.key === "ArrowRight") { show(cur + 1); e.preventDefault(); }
      if (e.key === "ArrowLeft") { show(cur - 1); e.preventDefault(); }
    });
    show(0);
  }

  // ---------------------------------------------------------------- progress
  function getDone() { return store.get(DONE_KEY, {}) || {}; }
  function setDone(id, v) {
    var d = getDone();
    if (v) d[id] = true; else delete d[id];
    store.set(DONE_KEY, d);
    return d;
  }
  function initDoneToggle() {
    var box = document.getElementById("done-box");
    if (!box) return;
    var id = box.getAttribute("data-id");
    box.checked = !!getDone()[id];
    box.addEventListener("change", function () { setDone(id, box.checked); });
  }
  function initIndex() {
    var cards = document.querySelectorAll(".card[data-id]");
    if (!cards.length) return;
    function refresh() {
      var d = getDone(), n = 0, mins = 0, total = 0;
      Array.prototype.forEach.call(cards, function (c) {
        var id = c.getAttribute("data-id"), m = +c.getAttribute("data-min") || 0;
        total += m;
        var on = !!d[id];
        c.classList.toggle("done", on);
        var cb = c.querySelector("input.chk");
        if (cb) cb.checked = on;
        if (on) { n += 1; mins += m; }
      });
      var bar = document.getElementById("prog-bar");
      var txt = document.getElementById("prog-text");
      if (bar) bar.style.width = (100 * n / cards.length).toFixed(1) + "%";
      if (txt) txt.textContent = n + " / " + cards.length + " 题 · " + mins + " / " + total + " 分钟";
    }
    Array.prototype.forEach.call(cards, function (c) {
      var cb = c.querySelector("input.chk");
      if (!cb) return;
      cb.addEventListener("click", function (e) { e.stopPropagation(); });
      cb.addEventListener("change", function () { setDone(c.getAttribute("data-id"), cb.checked); refresh(); });
    });
    var reset = document.getElementById("prog-reset");
    if (reset) reset.addEventListener("click", function () {
      if (window.confirm("清空所有完成标记？")) { store.set(DONE_KEY, {}); refresh(); }
    });
    refresh();
  }

  // ---------------------------------------------------------------- expand / collapse hints
  function initHintButtons() {
    var openAll = document.getElementById("open-l1");
    if (!openAll) return;
    openAll.addEventListener("click", function () {
      var hints = document.querySelectorAll("details.hint");
      var anyClosed = Array.prototype.some.call(hints, function (d) { return !d.open; });
      Array.prototype.forEach.call(hints, function (d) { d.open = anyClosed; });
      openAll.textContent = anyClosed ? "全部收起" : "全部展开";
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    initTheme();
    Array.prototype.forEach.call(document.querySelectorAll(".stepper"), initStepper);
    initDoneToggle();
    initIndex();
    initHintButtons();
  });
})();
