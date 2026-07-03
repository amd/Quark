/* Interactive "Quick Install Selector" widget on the install page.
   Kept in a static asset (loaded only by the Sphinx build) rather than inline in
   install.rst, because Git hosts such as GitHub render the text content of inline
   <script>/<style> blocks as visible junk. The widget builds its own markup into
   the empty #qk-selector mount, so hosts that don't run this script show only the
   static fallback table. */
(function () {
  var WIDGET_HTML = [
    '<div class="qk-row">',
    '  <div class="qk-label">Quark Wheel</div>',
    '  <div class="qk-options" data-group="wheel">',
    '    <button type="button" class="qk-opt qk-active" data-value="universal">Universal (PyPI)</button>',
    '    <button type="button" class="qk-opt" data-value="prebuilt">Pre-built (AMD index)</button>',
    '  </div>',
    '</div>',
    '<div class="qk-row">',
    '  <div class="qk-label">Your OS</div>',
    '  <div class="qk-options" data-group="os">',
    '    <button type="button" class="qk-opt qk-active" data-value="linux">Linux</button>',
    '    <button type="button" class="qk-opt" data-value="windows">Windows</button>',
    '  </div>',
    '</div>',
    '<div class="qk-row">',
    '  <div class="qk-label">PyTorch</div>',
    '  <div class="qk-options" data-group="torch">',
    '    <button type="button" class="qk-opt qk-active" data-value="2.2-2.9">2.2 \u2013 2.9</button>',
    '    <button type="button" class="qk-opt" data-value="2.10+">2.10+</button>',
    '  </div>',
    '</div>',
    '<div class="qk-row">',
    '  <div class="qk-label">Python</div>',
    '  <div class="qk-options" data-group="python">',
    '    <button type="button" class="qk-opt" data-value="3.11">3.11</button>',
    '    <button type="button" class="qk-opt" data-value="3.12">3.12</button>',
    '    <button type="button" class="qk-opt qk-active" data-value="3.13">3.13</button>',
    '  </div>',
    '</div>',
    '<div class="qk-row">',
    '  <div class="qk-label">Compute Platform</div>',
    '  <div class="qk-options" data-group="platform">',
    '    <button type="button" class="qk-opt qk-active" data-value="cpu">CPU</button>',
    '    <button type="button" class="qk-opt" data-value="cu128">CUDA 12.8</button>',
    '    <button type="button" class="qk-opt" data-value="rocm71">ROCm 7.1</button>',
    '    <button type="button" class="qk-opt" data-value="rocm72">ROCm 7.2</button>',
    '  </div>',
    '</div>',
    '<div class="qk-row qk-cmd-row">',
    '  <div class="qk-label">Run this Command</div>',
    '  <div class="qk-cmd">',
    '    <pre><code id="qk-cmd-out"></code></pre>',
    '  </div>',
    '</div>'
  ].join("");

  function selected(root, group) {
    var el = root.querySelector('[data-group="' + group + '"] .qk-active:not(.qk-disabled)');
    return el ? el.getAttribute("data-value") : null;
  }

  function setActive(btn) {
    var group = btn.parentNode;
    group.querySelectorAll(".qk-opt").forEach(function (b) { b.classList.remove("qk-active"); });
    btn.classList.add("qk-active");
  }

  var PREBUILT_PY = ["3.11", "3.12", "3.13"];

  function applyConstraints(root) {
    var os = selected(root, "os");
    var python = selected(root, "python");
    var platBtns = root.querySelectorAll('[data-group="platform"] .qk-opt');

    // ROCm builds are Linux-only.
    platBtns.forEach(function (b) {
      var v = b.getAttribute("data-value");
      var disabled = (os === "windows" && (v === "rocm71" || v === "rocm72"));
      b.classList.toggle("qk-disabled", disabled);
      if (disabled && b.classList.contains("qk-active")) {
        b.classList.remove("qk-active");
        root.querySelector('[data-group="platform"] [data-value="cpu"]').classList.add("qk-active");
      }
    });

    // Pre-built wheels need PyTorch 2.10+ and Python 3.11-3.13; otherwise fall back to universal.
    var torch = selected(root, "torch");
    var prebuiltBtn = root.querySelector('[data-group="wheel"] [data-value="prebuilt"]');
    var prebuiltDisabled = (torch !== "2.10+") || (PREBUILT_PY.indexOf(python) === -1);
    prebuiltBtn.classList.toggle("qk-disabled", prebuiltDisabled);
    if (prebuiltDisabled && prebuiltBtn.classList.contains("qk-active")) {
      prebuiltBtn.classList.remove("qk-active");
      root.querySelector('[data-group="wheel"] [data-value="universal"]').classList.add("qk-active");
    }
  }

  function render(root) {
    applyConstraints(root);
    var wheel = selected(root, "wheel");
    var platform = selected(root, "platform");
    var outEl = root.querySelector("#qk-cmd-out");

    if (wheel === "prebuilt") {
      outEl.textContent = "pip install amd-quark --extra-index-url https://pypi.amd.com/quark/" + platform + "/simple";
    } else {
      outEl.textContent = "pip install amd-quark";
    }
  }

  function init() {
    var root = document.getElementById("qk-selector");
    if (!root) return;
    root.innerHTML = WIDGET_HTML;
    root.querySelectorAll(".qk-opt").forEach(function (btn) {
      btn.addEventListener("click", function () {
        if (btn.classList.contains("qk-disabled")) return;
        setActive(btn);
        render(root);
      });
    });
    render(root);
    // Signal success so the CSS hides the static fallback table only now that the
    // interactive selector is in place; if this script fails, the table stays visible.
    document.documentElement.classList.add("qk-js-ready");
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
