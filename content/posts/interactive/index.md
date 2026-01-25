---
title: "Interactive: Optimization Intuition (SGD ? Momentum ? Muon)"
date: 2026-01-25
draft: false
math: false
toc: true
---

This page hosts interactive visuals that pair with the Muon optimizer post. Everything is plain HTML/SVG/JS (no external libraries), so you can tweak sliders and see how trajectories and operator norms behave immediately.

## Interactive 1: Traditional SGD on a quadratic bowl

This is a 2D quadratic loss (elliptical contours). The blue polyline is the sequence of SGD steps from the green start point to the orange end point. Increase the learning rate to see faster progress (and potential overshoot), raise the step count to lengthen the trajectory, and move the start point to sample different regions of curvature.

<div class="ml-interactive" data-ml="sgd-basic">
  <div class="ml-controls">
    <label>
      Learning rate
      <input type="range" min="0.01" max="0.25" value="0.08" step="0.01" data-role="lr" />
      <span class="ml-readout" data-role="lr-val">0.08</span>
    </label>
    <label>
      Steps
      <input type="range" min="5" max="60" value="25" step="1" data-role="steps" />
      <span class="ml-readout" data-role="steps-val">25</span>
    </label>
    <label>
      Start x
      <input type="range" min="-3.5" max="3.5" value="2.5" step="0.1" data-role="x0" />
      <span class="ml-readout" data-role="x0-val">2.5</span>
    </label>
    <label>
      Start y
      <input type="range" min="-3.5" max="3.5" value="2.0" step="0.1" data-role="y0" />
      <span class="ml-readout" data-role="y0-val">2.0</span>
    </label>
    <button type="button" class="ml-button" data-role="reset">Reset start</button>
  </div>
  <svg class="ml-plot" viewBox="0 0 640 360" role="img" aria-label="SGD trajectory on quadratic bowl">
    <rect x="0" y="0" width="640" height="360" fill="white"></rect>
    <g data-role="grid"></g>
    <g data-role="contours"></g>
    <path data-role="path" fill="none" stroke="#1f77b4" stroke-width="3"></path>
    <circle data-role="start" r="5" fill="#2b8a3e"></circle>
    <circle data-role="end" r="5" fill="#d9480f"></circle>
    <g data-role="axes"></g>
  </svg>
  <div class="ml-legend">
    <span class="ml-chip"><span class="ml-dot" style="background:#1f77b4"></span>SGD path</span>
    <span class="ml-chip"><span class="ml-dot" style="background:#2b8a3e"></span>start</span>
    <span class="ml-chip"><span class="ml-dot" style="background:#d9480f"></span>end</span>
  </div>
</div>

## Interactive 2: Ill-conditioning and gradient scaling

This bowl has very different curvature along each axis. The same global learning rate produces zig-zagging updates, while a simple per-coordinate rescaling stabilizes the path.

Use the curvature ratio to control how stretched the contours are (higher means worse conditioning). The orange path is plain SGD with a single learning rate; the green path divides each gradient coordinate by its curvature so it behaves more like a well-conditioned problem. Compare the two paths as you change the learning rate and step count.

<div class="ml-interactive" data-ml="sgd-scaling">
  <div class="ml-controls">
    <label>
      Curvature ratio (a:b)
      <input type="range" min="1" max="60" value="25" step="1" data-role="ratio" />
      <span class="ml-readout" data-role="ratio-val">25</span>
    </label>
    <label>
      Learning rate
      <input type="range" min="0.01" max="0.4" value="0.12" step="0.01" data-role="lr" />
      <span class="ml-readout" data-role="lr-val">0.12</span>
    </label>
    <label>
      Steps
      <input type="range" min="5" max="60" value="30" step="1" data-role="steps" />
      <span class="ml-readout" data-role="steps-val">30</span>
    </label>
  </div>
  <svg class="ml-plot" viewBox="0 0 640 360" role="img" aria-label="Gradient scaling comparison">
    <rect x="0" y="0" width="640" height="360" fill="white"></rect>
    <g data-role="grid"></g>
    <g data-role="contours"></g>
    <path data-role="path-sgd" fill="none" stroke="#d9480f" stroke-width="3"></path>
    <path data-role="path-pre" fill="none" stroke="#2b8a3e" stroke-width="3"></path>
    <g data-role="axes"></g>
  </svg>
  <div class="ml-legend">
    <span class="ml-chip"><span class="ml-dot" style="background:#d9480f"></span>SGD (single lr)</span>
    <span class="ml-chip"><span class="ml-dot" style="background:#2b8a3e"></span>coordinate-scaled</span>
  </div>
</div>

## Interactive 3: Momentum slider

Momentum averages gradients over time and can reduce zig-zagging on ill-conditioned bowls. Try increasing the momentum value and watch the path straighten.

Here, both paths use the same learning rate and number of steps. The gray line is vanilla SGD; the blue line adds momentum with coefficient beta. Higher beta means stronger smoothing of gradients, which typically reduces oscillation across the steep axis.

<div class="ml-interactive" data-ml="sgd-momentum">
  <div class="ml-controls">
    <label>
      Momentum (beta)
      <input type="range" min="0" max="0.95" value="0.8" step="0.01" data-role="beta" />
      <span class="ml-readout" data-role="beta-val">0.80</span>
    </label>
    <label>
      Learning rate
      <input type="range" min="0.01" max="0.3" value="0.12" step="0.01" data-role="lr" />
      <span class="ml-readout" data-role="lr-val">0.12</span>
    </label>
    <label>
      Steps
      <input type="range" min="5" max="70" value="35" step="1" data-role="steps" />
      <span class="ml-readout" data-role="steps-val">35</span>
    </label>
  </div>
  <svg class="ml-plot" viewBox="0 0 640 360" role="img" aria-label="Momentum path on ill-conditioned bowl">
    <rect x="0" y="0" width="640" height="360" fill="white"></rect>
    <g data-role="grid"></g>
    <g data-role="contours"></g>
    <path data-role="path-sgd" fill="none" stroke="#adb5bd" stroke-width="2"></path>
    <path data-role="path-mom" fill="none" stroke="#1f77b4" stroke-width="3"></path>
    <g data-role="axes"></g>
  </svg>
  <div class="ml-legend">
    <span class="ml-chip"><span class="ml-dot" style="background:#1f77b4"></span>momentum</span>
    <span class="ml-chip"><span class="ml-dot" style="background:#adb5bd"></span>plain SGD</span>
  </div>
</div>

## Interactive 4: Operator norm + polar factor (Muon intuition)

We visualize a 2×2 weight matrix acting on the unit circle. The spectral norm is the maximum stretch. The polar factor removes stretching while preserving rotation, echoing Muon’s “orthogonalized” update.

The sliders labeled a, b, c, d are the entries of the weight matrix
`W = [[a, b], [c, d]]`. The orange curve shows how W stretches the unit circle; the green curve shows the closest orthogonal (polar) factor that preserves rotation but removes stretching. The dashed ring is an RMS gain guide that scales with fan-in/out via `sigma_max * sqrt(n/m)`; as you change n and m, the ring (and scale) adjusts even if the matrix entries stay fixed.

<div class="ml-interactive" data-ml="operator-norm">
  <div class="ml-controls">
    <label>
      a
      <input type="range" min="-2" max="2" value="1.4" step="0.05" data-role="a" />
      <span class="ml-readout" data-role="a-val">1.40</span>
    </label>
    <label>
      b
      <input type="range" min="-2" max="2" value="0.4" step="0.05" data-role="b" />
      <span class="ml-readout" data-role="b-val">0.40</span>
    </label>
    <label>
      c
      <input type="range" min="-2" max="2" value="-0.6" step="0.05" data-role="c" />
      <span class="ml-readout" data-role="c-val">-0.60</span>
    </label>
    <label>
      d
      <input type="range" min="-2" max="2" value="1.1" step="0.05" data-role="d" />
      <span class="ml-readout" data-role="d-val">1.10</span>
    </label>
    <label>
      fan-in (n)
      <input type="range" min="8" max="256" value="64" step="8" data-role="n" />
      <span class="ml-readout" data-role="n-val">64</span>
    </label>
    <label>
      fan-out (m)
      <input type="range" min="8" max="256" value="64" step="8" data-role="m" />
      <span class="ml-readout" data-role="m-val">64</span>
    </label>
  </div>
  <svg class="ml-plot" viewBox="0 0 640 360" role="img" aria-label="Operator norm and polar factor visualization">
    <rect x="0" y="0" width="640" height="360" fill="white"></rect>
    <g data-role="grid"></g>
    <g data-role="axes"></g>
    <path data-role="unit" fill="none" stroke="#adb5bd" stroke-width="2"></path>
    <path data-role="mapped" fill="none" stroke="#d9480f" stroke-width="3"></path>
    <path data-role="polar" fill="none" stroke="#2b8a3e" stroke-width="3"></path>
    <path data-role="rms" fill="none" stroke="#845ef7" stroke-width="2" stroke-dasharray="6 6"></path>
  </svg>
  <div class="ml-legend">
    <span class="ml-chip"><span class="ml-dot" style="background:#adb5bd"></span>unit circle</span>
    <span class="ml-chip"><span class="ml-dot" style="background:#d9480f"></span>W · circle</span>
    <span class="ml-chip"><span class="ml-dot" style="background:#2b8a3e"></span>polar(W) · circle</span>
    <span class="ml-chip"><span class="ml-dot" style="background:#845ef7"></span>RMS gain ring</span>
    <span class="ml-chip">s_max: <span class="ml-readout" data-role="sigma">0.00</span></span>
    <span class="ml-chip">RMS gain: <span class="ml-readout" data-role="rms">0.00</span></span>
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
  function makeScaler(width, height, pad, xMin, xMax, yMin, yMax) {
    function sx(x) {
      return pad + ((x - xMin) / (xMax - xMin)) * (width - 2 * pad);
    }
    function sy(y) {
      return height - pad - ((y - yMin) / (yMax - yMin)) * (height - 2 * pad);
    }
    return { sx: sx, sy: sy };
  }

  function drawGrid(grid, width, height, pad) {
    var lines = [];
    for (var i = 0; i <= 10; i += 1) {
      var gx = pad + ((width - 2 * pad) * i) / 10;
      var gy = pad + ((height - 2 * pad) * i) / 10;
      lines.push('<line x1="' + gx + '" y1="' + pad + '" x2="' + gx + '" y2="' + (height - pad) + '" stroke="#eee" />');
      lines.push('<line x1="' + pad + '" y1="' + gy + '" x2="' + (width - pad) + '" y2="' + gy + '" stroke="#eee" />');
    }
    grid.innerHTML = lines.join("");
  }

  function drawAxes(axes, width, height, pad, sx, sy) {
    var axis = [];
    axis.push('<line x1="' + pad + '" y1="' + sy(0) + '" x2="' + (width - pad) + '" y2="' + sy(0) + '" stroke="#333" />');
    axis.push('<line x1="' + sx(0) + '" y1="' + pad + '" x2="' + sx(0) + '" y2="' + (height - pad) + '" stroke="#333" />');
    axes.innerHTML = axis.join("");
  }

  function contourPath(a, b, sx, sy, levels) {
    var paths = [];
    for (var i = 0; i < levels.length; i += 1) {
      var c = levels[i];
      var rx = Math.sqrt(c / a);
      var ry = Math.sqrt(c / b);
      if (!isFinite(rx) || !isFinite(ry)) continue;
      var steps = 120;
      var d = "";
      for (var t = 0; t <= steps; t += 1) {
        var ang = (Math.PI * 2 * t) / steps;
        var x = rx * Math.cos(ang);
        var y = ry * Math.sin(ang);
        var px = sx(x);
        var py = sy(y);
        d += (t === 0 ? "M" : "L") + px.toFixed(2) + " " + py.toFixed(2) + " ";
      }
      paths.push('<path d="' + d.trim() + '" fill="none" stroke="#ced4da" stroke-width="1.2" />');
    }
    return paths.join("");
  }

  function pathFromPoints(points, sx, sy) {
    var d = "";
    for (var i = 0; i < points.length; i += 1) {
      var px = sx(points[i].x);
      var py = sy(points[i].y);
      d += (i === 0 ? "M" : "L") + px.toFixed(2) + " " + py.toFixed(2) + " ";
    }
    return d.trim();
  }

  function computeSGDPath(x0, y0, a, b, lr, steps) {
    var pts = [{ x: x0, y: y0 }];
    var x = x0;
    var y = y0;
    for (var i = 0; i < steps; i += 1) {
      var gx = a * x;
      var gy = b * y;
      x -= lr * gx;
      y -= lr * gy;
      pts.push({ x: x, y: y });
    }
    return pts;
  }

  function computeScaledPath(x0, y0, a, b, lr, steps) {
    var pts = [{ x: x0, y: y0 }];
    var x = x0;
    var y = y0;
    for (var i = 0; i < steps; i += 1) {
      var gx = a * x;
      var gy = b * y;
      x -= lr * (gx / a);
      y -= lr * (gy / b);
      pts.push({ x: x, y: y });
    }
    return pts;
  }

  function computeMomentumPath(x0, y0, a, b, lr, steps, beta) {
    var pts = [{ x: x0, y: y0 }];
    var x = x0;
    var y = y0;
    var vx = 0;
    var vy = 0;
    for (var i = 0; i < steps; i += 1) {
      var gx = a * x;
      var gy = b * y;
      vx = beta * vx + gx;
      vy = beta * vy + gy;
      x -= lr * vx;
      y -= lr * vy;
      pts.push({ x: x, y: y });
    }
    return pts;
  }

  function setupSgdBasic(root) {
    var grid = root.querySelector("[data-role='grid']");
    var axes = root.querySelector("[data-role='axes']");
    var contours = root.querySelector("[data-role='contours']");
    var path = root.querySelector("[data-role='path']");
    var startDot = root.querySelector("[data-role='start']");
    var endDot = root.querySelector("[data-role='end']");
    var lrInput = root.querySelector("[data-role='lr']");
    var lrVal = root.querySelector("[data-role='lr-val']");
    var stepsInput = root.querySelector("[data-role='steps']");
    var stepsVal = root.querySelector("[data-role='steps-val']");
    var x0Input = root.querySelector("[data-role='x0']");
    var x0Val = root.querySelector("[data-role='x0-val']");
    var y0Input = root.querySelector("[data-role='y0']");
    var y0Val = root.querySelector("[data-role='y0-val']");
    var resetBtn = root.querySelector("[data-role='reset']");

    var width = 640;
    var height = 360;
    var pad = 45;
    var xMin = -4;
    var xMax = 4;
    var yMin = -3.2;
    var yMax = 3.2;
    var scale = makeScaler(width, height, pad, xMin, xMax, yMin, yMax);

    var a = 1.0;
    var b = 4.0;

    function render() {
      var lr = parseFloat(lrInput.value);
      var steps = parseInt(stepsInput.value, 10);
      var x0 = parseFloat(x0Input.value);
      var y0 = parseFloat(y0Input.value);

      lrVal.textContent = lr.toFixed(2);
      stepsVal.textContent = String(steps);
      x0Val.textContent = x0.toFixed(1);
      y0Val.textContent = y0.toFixed(1);

      var pts = computeSGDPath(x0, y0, a, b, lr, steps);
      path.setAttribute("d", pathFromPoints(pts, scale.sx, scale.sy));
      startDot.setAttribute("cx", scale.sx(pts[0].x));
      startDot.setAttribute("cy", scale.sy(pts[0].y));
      var last = pts[pts.length - 1];
      endDot.setAttribute("cx", scale.sx(last.x));
      endDot.setAttribute("cy", scale.sy(last.y));

      contours.innerHTML = contourPath(a, b, scale.sx, scale.sy, [0.5, 1, 2, 3, 4, 5]);
    }

    resetBtn.addEventListener("click", function () {
      x0Input.value = "2.5";
      y0Input.value = "2.0";
      render();
    });

    drawGrid(grid, width, height, pad);
    drawAxes(axes, width, height, pad, scale.sx, scale.sy);
    render();

    lrInput.addEventListener("input", render);
    stepsInput.addEventListener("input", render);
    x0Input.addEventListener("input", render);
    y0Input.addEventListener("input", render);
  }

  function setupScaling(root) {
    var grid = root.querySelector("[data-role='grid']");
    var axes = root.querySelector("[data-role='axes']");
    var contours = root.querySelector("[data-role='contours']");
    var pathSgd = root.querySelector("[data-role='path-sgd']");
    var pathPre = root.querySelector("[data-role='path-pre']");
    var ratioInput = root.querySelector("[data-role='ratio']");
    var ratioVal = root.querySelector("[data-role='ratio-val']");
    var lrInput = root.querySelector("[data-role='lr']");
    var lrVal = root.querySelector("[data-role='lr-val']");
    var stepsInput = root.querySelector("[data-role='steps']");
    var stepsVal = root.querySelector("[data-role='steps-val']");

    var width = 640;
    var height = 360;
    var pad = 45;
    var xMin = -4;
    var xMax = 4;
    var yMin = -3.2;
    var yMax = 3.2;
    var scale = makeScaler(width, height, pad, xMin, xMax, yMin, yMax);

    function render() {
      var ratio = parseFloat(ratioInput.value);
      var lr = parseFloat(lrInput.value);
      var steps = parseInt(stepsInput.value, 10);

      ratioVal.textContent = ratio.toFixed(0);
      lrVal.textContent = lr.toFixed(2);
      stepsVal.textContent = String(steps);

      var a = ratio;
      var b = 1.0;
      var x0 = 2.5;
      var y0 = 2.0;

      var ptsSgd = computeSGDPath(x0, y0, a, b, lr, steps);
      var ptsPre = computeScaledPath(x0, y0, a, b, lr, steps);

      pathSgd.setAttribute("d", pathFromPoints(ptsSgd, scale.sx, scale.sy));
      pathPre.setAttribute("d", pathFromPoints(ptsPre, scale.sx, scale.sy));

      contours.innerHTML = contourPath(a, b, scale.sx, scale.sy, [0.5, 1, 2, 3, 4, 6, 8]);
    }

    drawGrid(grid, width, height, pad);
    drawAxes(axes, width, height, pad, scale.sx, scale.sy);
    render();

    ratioInput.addEventListener("input", render);
    lrInput.addEventListener("input", render);
    stepsInput.addEventListener("input", render);
  }

  function setupMomentum(root) {
    var grid = root.querySelector("[data-role='grid']");
    var axes = root.querySelector("[data-role='axes']");
    var contours = root.querySelector("[data-role='contours']");
    var pathSgd = root.querySelector("[data-role='path-sgd']");
    var pathMom = root.querySelector("[data-role='path-mom']");
    var betaInput = root.querySelector("[data-role='beta']");
    var betaVal = root.querySelector("[data-role='beta-val']");
    var lrInput = root.querySelector("[data-role='lr']");
    var lrVal = root.querySelector("[data-role='lr-val']");
    var stepsInput = root.querySelector("[data-role='steps']");
    var stepsVal = root.querySelector("[data-role='steps-val']");

    var width = 640;
    var height = 360;
    var pad = 45;
    var xMin = -4;
    var xMax = 4;
    var yMin = -3.2;
    var yMax = 3.2;
    var scale = makeScaler(width, height, pad, xMin, xMax, yMin, yMax);

    function render() {
      var beta = parseFloat(betaInput.value);
      var lr = parseFloat(lrInput.value);
      var steps = parseInt(stepsInput.value, 10);

      betaVal.textContent = beta.toFixed(2);
      lrVal.textContent = lr.toFixed(2);
      stepsVal.textContent = String(steps);

      var a = 30.0;
      var b = 1.0;
      var x0 = 2.5;
      var y0 = 2.0;

      var ptsSgd = computeSGDPath(x0, y0, a, b, lr, steps);
      var ptsMom = computeMomentumPath(x0, y0, a, b, lr, steps, beta);

      pathSgd.setAttribute("d", pathFromPoints(ptsSgd, scale.sx, scale.sy));
      pathMom.setAttribute("d", pathFromPoints(ptsMom, scale.sx, scale.sy));

      contours.innerHTML = contourPath(a, b, scale.sx, scale.sy, [0.5, 1, 2, 3, 4, 6, 8]);
    }

    drawGrid(grid, width, height, pad);
    drawAxes(axes, width, height, pad, scale.sx, scale.sy);
    render();

    betaInput.addEventListener("input", render);
    lrInput.addEventListener("input", render);
    stepsInput.addEventListener("input", render);
  }

  function mulMatVec(M, v) {
    return {
      x: M[0][0] * v.x + M[0][1] * v.y,
      y: M[1][0] * v.x + M[1][1] * v.y
    };
  }

  function invSqrt2x2(p, q, r) {
    var trace = p + r;
    var det = p * r - q * q;
    var disc = Math.sqrt(Math.max(0, trace * trace - 4 * det));
    var l1 = 0.5 * (trace + disc);
    var l2 = 0.5 * (trace - disc);

    l1 = Math.max(l1, 1e-12);
    l2 = Math.max(l2, 1e-12);

    var v1x = q;
    var v1y = l1 - p;
    if (Math.abs(v1x) + Math.abs(v1y) < 1e-8) {
      v1x = 1;
      v1y = 0;
    }
    var n1 = Math.sqrt(v1x * v1x + v1y * v1y);
    v1x /= n1;
    v1y /= n1;
    var v2x = -v1y;
    var v2y = v1x;

    var inv1 = 1 / Math.sqrt(l1);
    var inv2 = 1 / Math.sqrt(l2);

    var m00 = inv1 * v1x * v1x + inv2 * v2x * v2x;
    var m01 = inv1 * v1x * v1y + inv2 * v2x * v2y;
    var m11 = inv1 * v1y * v1y + inv2 * v2y * v2y;

    return [
      [m00, m01],
      [m01, m11]
    ];
  }

  function computeSigmaMax(a, b, c, d) {
    var s1 = a * a + b * b + c * c + d * d;
    var det = (a * d - b * c);
    var disc = Math.sqrt(Math.max(0, s1 * s1 - 4 * det * det));
    var sigma2 = 0.5 * (s1 + disc);
    return Math.sqrt(Math.max(0, sigma2));
  }

  function setupOperatorNorm(root) {
    var grid = root.querySelector("[data-role='grid']");
    var axes = root.querySelector("[data-role='axes']");
    var unitPath = root.querySelector("[data-role='unit']");
    var mappedPath = root.querySelector("[data-role='mapped']");
    var polarPath = root.querySelector("[data-role='polar']");
    var rmsPath = root.querySelector("[data-role='rms']");
    var sigmaEl = root.querySelector("[data-role='sigma']");
    var rmsEl = root.querySelector("[data-role='rms']");

    var inputs = {
      a: root.querySelector("[data-role='a']"),
      b: root.querySelector("[data-role='b']"),
      c: root.querySelector("[data-role='c']"),
      d: root.querySelector("[data-role='d']"),
      n: root.querySelector("[data-role='n']"),
      m: root.querySelector("[data-role='m']")
    };
    var vals = {
      a: root.querySelector("[data-role='a-val']"),
      b: root.querySelector("[data-role='b-val']"),
      c: root.querySelector("[data-role='c-val']"),
      d: root.querySelector("[data-role='d-val']"),
      n: root.querySelector("[data-role='n-val']"),
      m: root.querySelector("[data-role='m-val']")
    };

    var width = 640;
    var height = 360;
    var pad = 45;
    var baseX = 3.5;
    var baseY = 2.6;
    var scale = makeScaler(width, height, pad, -baseX, baseX, -baseY, baseY);

    function circlePath(transform, colorScale) {
      var steps = 140;
      var d = "";
      for (var i = 0; i <= steps; i += 1) {
        var ang = (Math.PI * 2 * i) / steps;
        var v = { x: Math.cos(ang), y: Math.sin(ang) };
        if (transform) {
          v = transform(v);
        }
        var px = scale.sx(v.x * colorScale);
        var py = scale.sy(v.y * colorScale);
        d += (i === 0 ? "M" : "L") + px.toFixed(2) + " " + py.toFixed(2) + " ";
      }
      return d.trim();
    }

    function render() {
      var a = parseFloat(inputs.a.value);
      var b = parseFloat(inputs.b.value);
      var c = parseFloat(inputs.c.value);
      var d = parseFloat(inputs.d.value);
      var n = parseInt(inputs.n.value, 10);
      var m = parseInt(inputs.m.value, 10);

      vals.a.textContent = a.toFixed(2);
      vals.b.textContent = b.toFixed(2);
      vals.c.textContent = c.toFixed(2);
      vals.d.textContent = d.toFixed(2);
      vals.n.textContent = String(n);
      vals.m.textContent = String(m);

      var sigma = computeSigmaMax(a, b, c, d);
      var rms = sigma * Math.sqrt(n / m);
      sigmaEl.textContent = sigma.toFixed(3);
      rmsEl.textContent = rms.toFixed(3);

      var maxRadius = Math.max(baseX, sigma * 1.15, rms * 1.15);
      var yRadius = Math.max(baseY, maxRadius * (baseY / baseX));
      scale = makeScaler(width, height, pad, -maxRadius, maxRadius, -yRadius, yRadius);
      drawGrid(grid, width, height, pad);
      drawAxes(axes, width, height, pad, scale.sx, scale.sy);

      var A = [
        [a, b],
        [c, d]
      ];
      var p = a * a + c * c;
      var q = a * b + c * d;
      var r = b * b + d * d;
      var invSqrt = invSqrt2x2(p, q, r);
      var polar = [
        [A[0][0] * invSqrt[0][0] + A[0][1] * invSqrt[1][0], A[0][0] * invSqrt[0][1] + A[0][1] * invSqrt[1][1]],
        [A[1][0] * invSqrt[0][0] + A[1][1] * invSqrt[1][0], A[1][0] * invSqrt[0][1] + A[1][1] * invSqrt[1][1]]
      ];

      unitPath.setAttribute("d", circlePath(null, 1.0));
      mappedPath.setAttribute("d", circlePath(function (v) { return mulMatVec(A, v); }, 1.0));
      polarPath.setAttribute("d", circlePath(function (v) { return mulMatVec(polar, v); }, 1.0));
      rmsPath.setAttribute("d", circlePath(null, rms));
    }

    render();

    Object.keys(inputs).forEach(function (key) {
      inputs[key].addEventListener("input", render);
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    var sgd = document.querySelector("[data-ml='sgd-basic']");
    if (sgd) setupSgdBasic(sgd);

    var scaling = document.querySelector("[data-ml='sgd-scaling']");
    if (scaling) setupScaling(scaling);

    var momentum = document.querySelector("[data-ml='sgd-momentum']");
    if (momentum) setupMomentum(momentum);

    var opnorm = document.querySelector("[data-ml='operator-norm']");
    if (opnorm) setupOperatorNorm(opnorm);
  });
})();
</script>
