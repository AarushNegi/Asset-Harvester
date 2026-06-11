// ── Background particle + orb animation ──
(function () {
  const canvas = document.getElementById('bg-canvas');
  const ctx = canvas.getContext('2d');

  let W, H;
  function resize() {
    W = canvas.width  = window.innerWidth;
    H = canvas.height = window.innerHeight;
  }
  resize();
  window.addEventListener('resize', resize);

  // ── Floating orbs ──
  const ORBS = [
    { x: 0.15, y: 0.10, r: 380, hue: 260, speed: 0.00018 },
    { x: 0.80, y: 0.75, r: 320, hue: 240, speed: 0.00014 },
    { x: 0.50, y: 0.50, r: 260, hue: 280, speed: 0.00022 },
    { x: 0.85, y: 0.15, r: 200, hue: 250, speed: 0.00030 },
  ].map(o => ({ ...o, ox: o.x, oy: o.y, t: Math.random() * Math.PI * 2 }));

  // ── Particles ──
  const PARTICLE_COUNT = 90;
  const particles = Array.from({ length: PARTICLE_COUNT }, () => ({
    x: Math.random(),
    y: Math.random(),
    size: 0.6 + Math.random() * 1.4,
    speedX: (Math.random() - 0.5) * 0.00012,
    speedY: (Math.random() - 0.5) * 0.00012,
    alpha: 0.15 + Math.random() * 0.45,
    pulse: Math.random() * Math.PI * 2,
    pulseSpeed: 0.004 + Math.random() * 0.008,
  }));

  // ── Grid dots ──
  const GRID_COLS = 28;
  const GRID_ROWS = 16;
  const gridDots = [];
  for (let r = 0; r <= GRID_ROWS; r++) {
    for (let c = 0; c <= GRID_COLS; c++) {
      gridDots.push({
        cx: c / GRID_COLS,
        cy: r / GRID_ROWS,
        phase: (c + r) * 0.4,
      });
    }
  }

  let t = 0;

  function draw() {
    ctx.clearRect(0, 0, W, H);
    t += 1;

    // -- Orbs --
    ORBS.forEach(o => {
      o.t += o.speed * 1000 / 60;
      const px = (o.ox + Math.sin(o.t) * 0.12) * W;
      const py = (o.oy + Math.cos(o.t * 0.7) * 0.09) * H;
      const grad = ctx.createRadialGradient(px, py, 0, px, py, o.r);
      grad.addColorStop(0, `hsla(${o.hue},70%,65%,0.13)`);
      grad.addColorStop(0.5, `hsla(${o.hue},60%,55%,0.05)`);
      grad.addColorStop(1, `hsla(${o.hue},50%,50%,0)`);
      ctx.beginPath();
      ctx.arc(px, py, o.r, 0, Math.PI * 2);
      ctx.fillStyle = grad;
      ctx.fill();
    });

    // -- Grid dots --
    gridDots.forEach(d => {
      const wave = Math.sin(t * 0.018 + d.phase) * 0.5 + 0.5;
      const alpha = 0.06 + wave * 0.14;
      const px = d.cx * W;
      const py = d.cy * H;
      ctx.beginPath();
      ctx.arc(px, py, 1, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(180,160,255,${alpha})`;
      ctx.fill();
    });

    // -- Particles --
    particles.forEach(p => {
      p.x += p.speedX;
      p.y += p.speedY;
      if (p.x < 0) p.x = 1;
      if (p.x > 1) p.x = 0;
      if (p.y < 0) p.y = 1;
      if (p.y > 1) p.y = 0;
      p.pulse += p.pulseSpeed;
      const a = p.alpha * (0.6 + Math.sin(p.pulse) * 0.4);
      ctx.beginPath();
      ctx.arc(p.x * W, p.y * H, p.size, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(200,185,255,${a})`;
      ctx.fill();
    });

    requestAnimationFrame(draw);
  }

  draw();
})();

// ── Asset Harvester — 3D Gaussian Splat Viewer ──

function loadSplat() {
  document.getElementById('file-input').click();
}

function handleDrop(e) {
  e.preventDefault();
  e.currentTarget.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file && file.name.endsWith('.ply')) renderSplat(file);
}

function handleFile(e) {
  const file = e.target.files[0];
  if (file) renderSplat(file);
}

function renderSplat(file) {
  const overlay = document.getElementById('viewer-overlay');
  overlay.innerHTML = `
    <div style="font-size:2rem">⏳</div>
    <p style="color:var(--accent)">Loading ${file.name}...</p>
    <p>${(file.size / 1024 / 1024).toFixed(1)} MB</p>
  `;

  const script = document.createElement('script');
  script.src = 'https://cdn.jsdelivr.net/npm/three@0.157.0/build/three.min.js';
  script.onload = () => initViewer(file);
  script.onerror = () => {
    overlay.innerHTML = `
      <div style="font-size:2rem">⚠️</div>
      <p>Could not load 3D viewer.<br>Open <strong>supersplat.playcanvas.com</strong><br>and upload your .ply file there.</p>
      <a href="https://supersplat.playcanvas.com" target="_blank" class="load-btn" style="margin-top:8px;">Open SuperSplat →</a>
    `;
  };
  document.head.appendChild(script);
}

function initViewer(file) {
  const overlay = document.getElementById('viewer-overlay');
  const reader = new FileReader();

  reader.onload = (e) => {
    const data = e.target.result;
    const bytes = new Uint8Array(data);

    // Count gaussians from PLY header
    const header = new TextDecoder().decode(bytes.slice(0, 2000));
    const match = header.match(/element vertex (\d+)/);
    const count = match ? parseInt(match[1]) : 'unknown';

    overlay.innerHTML = `
      <div style="font-size:2rem">✅</div>
      <p style="color:var(--success)">File loaded successfully</p>
      <p><strong style="color:var(--accent); font-family:var(--mono)">${typeof count === 'number' ? count.toLocaleString() : count}</strong> Gaussians</p>
      <p style="font-size:11px; margin-top:4px;">For full 3D rendering, open in SuperSplat</p>
      <a href="https://supersplat.playcanvas.com" target="_blank" class="load-btn" style="margin-top:8px;">Open SuperSplat →</a>
    `;
  };

  reader.onerror = () => {
    document.getElementById('viewer-overlay').innerHTML =
      `<p style="color:red">Failed to read file</p>`;
  };

  reader.readAsArrayBuffer(file);
}