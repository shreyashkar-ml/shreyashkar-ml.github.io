---
title: "Interactive ML plots"
date: 2026-01-25
draft: false
math: false
toc: true
---

This post is a self-contained demo of interactive ML visuals built with plain HTML, SVG, and JavaScript (no external libraries). It uses simple sliders and buttons to change the plots in real time.

![Convex loss landscape](loss-landscape.svg)

## Interactive 1: Polynomial regression (degree slider)

<div class="ml-interactive" data-ml="polyfit">
  <div class="ml-controls">
    <label>
      Degree
      <input type="range" min="1" max="7" value="3" step="1" data-role="degree" />
      <span class="ml-readout" data-role="degree-val">3</span>
    </label>
    <label>
      Show true function
      <input type="checkbox" checked data-role="show-true" />
    </label>
    <button type="button" class="ml-button" data-role="regen">Regenerate noise</button>
    <span class="ml-readout">MSE: <span data-role="mse">0.000</span></span>
  </div>
  <svg class="ml-plot" viewBox="0 0 640 360" role="img" aria-label="Polynomial regression fit">
    <rect x="0" y="0" width="640" height="360" fill="white"></rect>
    <g data-role="grid"></g>
    <g data-role="points"></g>
    <path data-role="true" fill="none" stroke="#2b8a3e" stroke-width="2" opacity="0.8"></path>
    <path data-role="fit" fill="none" stroke="#1f77b4" stroke-width="3"></path>
    <g data-role="axes"></g>
  </svg>
  <div class="ml-legend">
    <span class="ml-chip"><span class="ml-dot" style="background:#1f77b4"></span>fit</span>
    <span class="ml-chip"><span class="ml-dot" style="background:#2b8a3e"></span>true</span>
    <span class="ml-chip"><span class="ml-dot" style="background:#111"></span>data</span>
  </div>
</div>

![Bias and variance trends](bias-variance.svg)

## Interactive 2: Logistic curve (steepness + threshold)

<div class="ml-interactive" data-ml="sigmoid">
  <div class="ml-controls">
    <label>
      Steepness k
      <input type="range" min="0.5" max="8" value="2" step="0.1" data-role="k" />
      <span class="ml-readout" data-role="k-val">2.0</span>
    </label>
    <label>
      Threshold
      <input type="range" min="0.1" max="0.9" value="0.5" step="0.05" data-role="thresh" />
      <span class="ml-readout" data-role="thresh-val">0.50</span>
    </label>
  </div>
  <svg class="ml-plot" viewBox="0 0 640 360" role="img" aria-label="Logistic curve">
    <rect x="0" y="0" width="640" height="360" fill="white"></rect>
    <g data-role="grid"></g>
    <line data-role="thresh-line" x1="0" y1="0" x2="0" y2="0" stroke="#d9480f" stroke-width="2" stroke-dasharray="6 6"></line>
    <path data-role="curve" fill="none" stroke="#1f77b4" stroke-width="3"></path>
    <g data-role="samples"></g>
    <g data-role="axes"></g>
  </svg>
  <div class="ml-legend">
    <span class="ml-chip"><span class="ml-dot" style="background:#1f77b4"></span>sigmoid</span>
    <span class="ml-chip"><span class="ml-dot" style="background:#d9480f"></span>threshold</span>
  </div>
</div>

<style>
.ml-interactive {
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 16px;
  background: var(--entry);
  margin: 1rem 0 2rem;
}

.ml-controls {
  display: flex;
  flex-wrap: wrap;
  gap: 12px 18px;
  align-items: center;
  margin-bottom: 12px;
}

.ml-controls label {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  font-size: 0.95rem;
}

.ml-controls input[type="range"] {
  width: 180px;
}

.ml-readout {
  font-variant-numeric: tabular-nums;
  color: var(--secondary);
}

.ml-plot {
  width: 100%;
  height: auto;
  display: block;
  border-radius: 8px;
  border: 1px solid var(--border);
}

.ml-legend {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  margin-top: 8px;
  font-size: 0.9rem;
  color: var(--secondary);
}

.ml-chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
}

.ml-dot {
  width: 10px;
  height: 10px;
  border-radius: 999px;
  display: inline-block;
}

.ml-button {
  padding: 6px 10px;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: var(--code-bg);
  color: var(--primary);
}

.ml-button:hover {
  background: var(--tertiary);
}
</style>

<script>
(function () {
  function setupPolyfit(root) {
    var svg = root.querySelector("svg");
    var grid = root.querySelector("[data-role='grid']");
    var axes = root.querySelector("[data-role='axes']");
    var pointsGroup = root.querySelector("[data-role='points']");
    var truePath = root.querySelector("[data-role='true']");
    var fitPath = root.querySelector("[data-role='fit']");
    var degreeInput = root.querySelector("[data-role='degree']");
    var degreeVal = root.querySelector("[data-role='degree-val']");
    var showTrue = root.querySelector("[data-role='show-true']");
    var regen = root.querySelector("[data-role='regen']");
    var mseEl = root.querySelector("[data-role='mse']");

    var width = 640;
    var height = 360;
    var pad = 45;
    var xMin = -1;
    var xMax = 1;
    var yMin = -1.5;
    var yMax = 1.5;

    var seed = 42;
    function rng() {
      seed = (seed * 1664525 + 1013904223) % 4294967296;
      return seed / 4294967296;
    }

    var data = [];

    function sx(x) {
      return pad + ((x - xMin) / (xMax - xMin)) * (width - 2 * pad);
    }

    function sy(y) {
      return height - pad - ((y - yMin) / (yMax - yMin)) * (height - 2 * pad);
    }

    function linePath(fn) {
      var steps = 200;
      var d = "";
      for (var i = 0; i <= steps; i += 1) {
        var x = xMin + ((xMax - xMin) * i) / steps;
        var y = fn(x);
        var px = sx(x);
        var py = sy(y);
        d += (i === 0 ? "M" : "L") + px.toFixed(2) + " " + py.toFixed(2) + " ";
      }
      return d.trim();
    }

    function generateData() {
      data = [];
      for (var i = 0; i < 24; i += 1) {
        var x = -1 + (2 * i) / 23;
        var yTrue = Math.sin(Math.PI * x);
        var noise = (rng() - 0.5) * 0.35;
        data.push({ x: x, y: yTrue + noise, yTrue: yTrue });
      }
    }

    function polyfit(deg) {
      var n = data.length;
      var m = deg + 1;
      var A = [];
      var b = [];
      for (var i = 0; i < m; i += 1) {
        A.push(new Array(m).fill(0));
        b.push(0);
      }
      for (var r = 0; r < n; r += 1) {
        var row = [];
        var x = data[r].x;
        var pow = 1;
        for (var c = 0; c < m; c += 1) {
          row.push(pow);
          pow *= x;
        }
        for (var i2 = 0; i2 < m; i2 += 1) {
          b[i2] += row[i2] * data[r].y;
          for (var j2 = 0; j2 < m; j2 += 1) {
            A[i2][j2] += row[i2] * row[j2];
          }
        }
      }
      return solveLinear(A, b);
    }

    function solveLinear(A, b) {
      var n = b.length;
      var M = [];
      for (var i = 0; i < n; i += 1) {
        M.push(A[i].slice().concat([b[i]]));
      }
      for (var k = 0; k < n; k += 1) {
        var maxRow = k;
        var maxVal = Math.abs(M[k][k]);
        for (var r = k + 1; r < n; r += 1) {
          var val = Math.abs(M[r][k]);
          if (val > maxVal) {
            maxVal = val;
            maxRow = r;
          }
        }
        if (maxVal < 1e-12) {
          continue;
        }
        if (maxRow !== k) {
          var tmp = M[k];
          M[k] = M[maxRow];
          M[maxRow] = tmp;
        }
        var pivot = M[k][k];
        for (var j = k; j <= n; j += 1) {
          M[k][j] /= pivot;
        }
        for (var i3 = 0; i3 < n; i3 += 1) {
          if (i3 === k) continue;
          var factor = M[i3][k];
          for (var j2 = k; j2 <= n; j2 += 1) {
            M[i3][j2] -= factor * M[k][j2];
          }
        }
      }
      var x = [];
      for (var i4 = 0; i4 < n; i4 += 1) {
        x.push(M[i4][n]);
      }
      return x;
    }

    function evalPoly(coeffs, x) {
      var y = 0;
      var pow = 1;
      for (var i = 0; i < coeffs.length; i += 1) {
        y += coeffs[i] * pow;
        pow *= x;
      }
      return y;
    }

    function drawAxes() {
      var lines = [];
      for (var i = 0; i <= 10; i += 1) {
        var gx = pad + ((width - 2 * pad) * i) / 10;
        var gy = pad + ((height - 2 * pad) * i) / 10;
        lines.push(
          '<line x1="' + gx + '" y1="' + pad + '" x2="' + gx + '" y2="' + (height - pad) + '" stroke="#eee" />'
        );
        lines.push(
          '<line x1="' + pad + '" y1="' + gy + '" x2="' + (width - pad) + '" y2="' + gy + '" stroke="#eee" />'
        );
      }
      grid.innerHTML = lines.join("");

      var axis = [];
      axis.push('<line x1="' + pad + '" y1="' + sy(0) + '" x2="' + (width - pad) + '" y2="' + sy(0) + '" stroke="#333" />');
      axis.push('<line x1="' + sx(0) + '" y1="' + pad + '" x2="' + sx(0) + '" y2="' + (height - pad) + '" stroke="#333" />');
      axes.innerHTML = axis.join("");
    }

    function render() {
      var deg = parseInt(degreeInput.value, 10);
      degreeVal.textContent = String(deg);
      var coeffs = polyfit(deg);

      var mse = 0;
      for (var i = 0; i < data.length; i += 1) {
        var pred = evalPoly(coeffs, data[i].x);
        mse += Math.pow(pred - data[i].y, 2);
      }
      mse /= data.length;
      mseEl.textContent = mse.toFixed(3);

      fitPath.setAttribute("d", linePath(function (x) { return evalPoly(coeffs, x); }));
      truePath.setAttribute("d", linePath(function (x) { return Math.sin(Math.PI * x); }));
      truePath.style.display = showTrue.checked ? "block" : "none";

      var points = [];
      for (var i2 = 0; i2 < data.length; i2 += 1) {
        var px = sx(data[i2].x);
        var py = sy(data[i2].y);
        points.push('<circle cx="' + px.toFixed(2) + '" cy="' + py.toFixed(2) + '" r="3.5" fill="#111" />');
      }
      pointsGroup.innerHTML = points.join("");
    }

    degreeInput.addEventListener("input", render);
    showTrue.addEventListener("change", render);
    regen.addEventListener("click", function () {
      seed = Math.floor(Math.random() * 1000000) + 1;
      generateData();
      render();
    });

    drawAxes();
    generateData();
    render();
  }

  function setupSigmoid(root) {
    var svg = root.querySelector("svg");
    var grid = root.querySelector("[data-role='grid']");
    var axes = root.querySelector("[data-role='axes']");
    var curve = root.querySelector("[data-role='curve']");
    var threshLine = root.querySelector("[data-role='thresh-line']");
    var samples = root.querySelector("[data-role='samples']");
    var kInput = root.querySelector("[data-role='k']");
    var kVal = root.querySelector("[data-role='k-val']");
    var tInput = root.querySelector("[data-role='thresh']");
    var tVal = root.querySelector("[data-role='thresh-val']");

    var width = 640;
    var height = 360;
    var pad = 45;
    var xMin = -6;
    var xMax = 6;
    var yMin = 0;
    var yMax = 1;

    function sx(x) {
      return pad + ((x - xMin) / (xMax - xMin)) * (width - 2 * pad);
    }

    function sy(y) {
      return height - pad - ((y - yMin) / (yMax - yMin)) * (height - 2 * pad);
    }

    function sigmoid(x, k) {
      return 1 / (1 + Math.exp(-k * x));
    }

    function linePath(fn) {
      var steps = 200;
      var d = "";
      for (var i = 0; i <= steps; i += 1) {
        var x = xMin + ((xMax - xMin) * i) / steps;
        var y = fn(x);
        var px = sx(x);
        var py = sy(y);
        d += (i === 0 ? "M" : "L") + px.toFixed(2) + " " + py.toFixed(2) + " ";
      }
      return d.trim();
    }

    function drawAxes() {
      var lines = [];
      for (var i = 0; i <= 10; i += 1) {
        var gx = pad + ((width - 2 * pad) * i) / 10;
        var gy = pad + ((height - 2 * pad) * i) / 10;
        lines.push(
          '<line x1="' + gx + '" y1="' + pad + '" x2="' + gx + '" y2="' + (height - pad) + '" stroke="#eee" />'
        );
        lines.push(
          '<line x1="' + pad + '" y1="' + gy + '" x2="' + (width - pad) + '" y2="' + gy + '" stroke="#eee" />'
        );
      }
      grid.innerHTML = lines.join("");

      var axis = [];
      axis.push('<line x1="' + pad + '" y1="' + sy(0) + '" x2="' + (width - pad) + '" y2="' + sy(0) + '" stroke="#333" />');
      axis.push('<line x1="' + pad + '" y1="' + sy(1) + '" x2="' + (width - pad) + '" y2="' + sy(1) + '" stroke="#333" />');
      axis.push('<line x1="' + sx(0) + '" y1="' + pad + '" x2="' + sx(0) + '" y2="' + (height - pad) + '" stroke="#333" />');
      axes.innerHTML = axis.join("");
    }

    function render() {
      var k = parseFloat(kInput.value);
      var t = parseFloat(tInput.value);
      kVal.textContent = k.toFixed(1);
      tVal.textContent = t.toFixed(2);

      curve.setAttribute("d", linePath(function (x) { return sigmoid(x, k); }));
      threshLine.setAttribute("x1", pad);
      threshLine.setAttribute("x2", width - pad);
      threshLine.setAttribute("y1", sy(t));
      threshLine.setAttribute("y2", sy(t));

      var dots = [];
      for (var x = -5; x <= 5; x += 1) {
        var p = sigmoid(x, k);
        var color = p >= t ? "#2b8a3e" : "#d9480f";
        dots.push(
          '<circle cx="' + sx(x).toFixed(2) + '" cy="' + sy(0.05).toFixed(2) + '" r="5" fill="' + color + '" />'
        );
      }
      samples.innerHTML = dots.join("");
    }

    kInput.addEventListener("input", render);
    tInput.addEventListener("input", render);

    drawAxes();
    render();
  }

  document.addEventListener("DOMContentLoaded", function () {
    var poly = document.querySelector("[data-ml='polyfit']");
    if (poly) setupPolyfit(poly);

    var sig = document.querySelector("[data-ml='sigmoid']");
    if (sig) setupSigmoid(sig);
  });
})();
</script>
