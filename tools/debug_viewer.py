#!/usr/bin/env python3

"""Browser/debug snapshot viewer for Isaac Lab manager-based environments.

The script has two modes:

* real Isaac mode: launch a task, step one/few envs, and expose snapshots
* mock mode: serve the same browser UI with fake data for local UI checks
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


DEFAULT_NEAR_OBJECT_DISTANCE = 0.08


HTML_PAGE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Isaac Lab Debug Viewer</title>
  <style>
    :root {
      --bg: #111318;
      --panel: #191c22;
      --panel-2: #20242c;
      --border: #343a46;
      --text: #edf0f5;
      --muted: #a6adba;
      --accent: #4aa3ff;
      --good: #45c486;
      --warn: #f2b84b;
      --bad: #ef6a6a;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--text);
      background: var(--bg);
      font: 14px/1.4 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 12px 16px;
      border-bottom: 1px solid var(--border);
      background: #0f1116;
    }
    h1 {
      margin: 0;
      font-size: 16px;
      font-weight: 650;
      letter-spacing: 0;
    }
    main {
      display: grid;
      grid-template-columns: minmax(420px, 1.35fr) minmax(360px, 0.9fr);
      min-height: calc(100vh - 51px);
    }
    section {
      border-right: 1px solid var(--border);
      min-width: 0;
    }
    aside {
      min-width: 0;
      overflow: auto;
      max-height: calc(100vh - 51px);
    }
    .toolbar {
      display: flex;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
    }
    button, select {
      color: var(--text);
      background: var(--panel-2);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 7px 10px;
      font: inherit;
    }
    button:hover, select:hover { border-color: var(--accent); }
    button:focus-visible, select:focus-visible {
      outline: 2px solid var(--accent);
      outline-offset: 2px;
    }
    .view-stack {
      display: grid;
      grid-template-rows: minmax(360px, 1fr) minmax(220px, 0.55fr);
      gap: 1px;
      background: var(--border);
      height: calc(100vh - 51px);
    }
    .viewport {
      position: relative;
      background: #151922;
      min-height: 0;
    }
    canvas {
      display: block;
      width: 100%;
      height: 100%;
    }
    .caption {
      position: absolute;
      top: 10px;
      left: 12px;
      color: var(--muted);
      background: rgba(17, 19, 24, 0.82);
      border: 1px solid rgba(52, 58, 70, 0.9);
      border-radius: 6px;
      padding: 5px 8px;
      font-size: 12px;
    }
    .panel {
      padding: 14px 16px;
      border-bottom: 1px solid var(--border);
    }
    .panel h2 {
      margin: 0 0 10px;
      font-size: 13px;
      font-weight: 650;
      color: var(--muted);
      text-transform: uppercase;
    }
    .mode-card {
      display: grid;
      gap: 6px;
      padding: 10px;
      border: 1px solid var(--border);
      border-radius: 6px;
      background: var(--panel-2);
    }
    .mode-title {
      font-weight: 650;
      font-size: 15px;
    }
    .mode-help {
      color: var(--muted);
      font-size: 12px;
    }
    .stage-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 8px;
    }
    .stage {
      min-height: 76px;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 8px;
      background: var(--panel-2);
    }
    .stage.good { border-color: rgba(69, 196, 134, 0.65); }
    .stage.bad { border-color: rgba(239, 106, 106, 0.65); }
    .stage.unknown { border-color: rgba(166, 173, 186, 0.4); }
    .stage-name {
      display: flex;
      justify-content: space-between;
      gap: 8px;
      font-weight: 620;
      font-size: 13px;
    }
    .stage-detail {
      margin-top: 6px;
      color: var(--muted);
      font-size: 12px;
    }
    .timeline {
      width: 100%;
      height: 130px;
      border: 1px solid var(--border);
      border-radius: 6px;
      background: #151922;
    }
    .legend {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      margin-top: 8px;
      color: var(--muted);
      font-size: 12px;
    }
    .legend span::before {
      content: "";
      display: inline-block;
      width: 9px;
      height: 9px;
      margin-right: 5px;
      border-radius: 50%;
      background: var(--legend-color);
    }
    .kv {
      display: grid;
      grid-template-columns: minmax(120px, 1fr) minmax(110px, auto);
      gap: 6px 12px;
      align-items: baseline;
    }
    .kv dt { color: var(--muted); }
    .kv dd {
      margin: 0;
      font-variant-numeric: tabular-nums;
      text-align: right;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-variant-numeric: tabular-nums;
    }
    th, td {
      padding: 7px 6px;
      border-bottom: 1px solid rgba(52, 58, 70, 0.75);
      text-align: left;
      vertical-align: top;
    }
    th {
      color: var(--muted);
      font-weight: 600;
      font-size: 12px;
    }
    td.num, th.num { text-align: right; }
    .name { color: var(--text); }
    .muted { color: var(--muted); }
    .pill {
      display: inline-block;
      min-width: 52px;
      padding: 2px 7px;
      border-radius: 999px;
      text-align: center;
      background: var(--panel-2);
      border: 1px solid var(--border);
    }
    .active { color: var(--good); }
    .inactive { color: var(--muted); }
    .error {
      color: #ffd7d7;
      background: rgba(239, 106, 106, 0.13);
      border: 1px solid rgba(239, 106, 106, 0.45);
      border-radius: 6px;
      padding: 10px;
    }
    @media (max-width: 900px) {
      main { grid-template-columns: 1fr; }
      aside { max-height: none; }
      section { border-right: 0; border-bottom: 1px solid var(--border); }
      .view-stack { height: 72vh; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Isaac Lab Debug Viewer</h1>
    <div class="toolbar" aria-label="Simulation controls">
      <button id="playPause" type="button">Pause</button>
      <button id="step" type="button">Step</button>
      <button id="reset" type="button">Reset</button>
      <select id="speed" aria-label="Playback speed">
        <option value="1">1 step/tick</option>
        <option value="2">2 steps/tick</option>
        <option value="5">5 steps/tick</option>
        <option value="10">10 steps/tick</option>
      </select>
    </div>
  </header>
  <main>
    <section aria-label="Spatial debug views">
      <div class="view-stack">
        <div class="viewport">
          <canvas id="topCanvas"></canvas>
          <div class="caption">Top view: X/Y positions, target radii, object markers</div>
        </div>
        <div class="viewport">
          <canvas id="heightCanvas"></canvas>
          <div class="caption">Height view: X/Z positions and height thresholds</div>
        </div>
      </div>
    </section>
    <aside>
      <div class="panel" id="modePanel"></div>
      <div class="panel" id="statusPanel"></div>
      <div class="panel" id="bottleneckPanel"></div>
      <div class="panel" id="timelinePanel"></div>
      <div class="panel" id="metricsPanel"></div>
      <div class="panel" id="rewardPanel"></div>
      <div class="panel" id="probePanel"></div>
      <div class="panel" id="terminationPanel"></div>
      <div class="panel" id="assetPanel"></div>
      <div class="panel" id="overlayPanel"></div>
      <div class="panel" id="ioPanel"></div>
      <div class="panel" id="errorPanel"></div>
    </aside>
  </main>
  <script>
    const topCanvas = document.getElementById("topCanvas");
    const heightCanvas = document.getElementById("heightCanvas");
    const modePanel = document.getElementById("modePanel");
    const statusPanel = document.getElementById("statusPanel");
    const bottleneckPanel = document.getElementById("bottleneckPanel");
    const timelinePanel = document.getElementById("timelinePanel");
    const metricsPanel = document.getElementById("metricsPanel");
    const rewardPanel = document.getElementById("rewardPanel");
    const probePanel = document.getElementById("probePanel");
    const terminationPanel = document.getElementById("terminationPanel");
    const assetPanel = document.getElementById("assetPanel");
    const overlayPanel = document.getElementById("overlayPanel");
    const ioPanel = document.getElementById("ioPanel");
    const errorPanel = document.getElementById("errorPanel");
    const playPause = document.getElementById("playPause");
    const stepButton = document.getElementById("step");
    const resetButton = document.getElementById("reset");
    const speedSelect = document.getElementById("speed");

    let latest = null;
    let paused = false;
    let history = [];

    function resizeCanvas(canvas) {
      const rect = canvas.getBoundingClientRect();
      const ratio = window.devicePixelRatio || 1;
      const width = Math.max(1, Math.floor(rect.width * ratio));
      const height = Math.max(1, Math.floor(rect.height * ratio));
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }
      return { width, height, ratio };
    }

    function fmt(value, digits = 3) {
      if (value === null || value === undefined) return "-";
      if (typeof value === "number") return Number.isFinite(value) ? value.toFixed(digits) : String(value);
      if (typeof value === "boolean") return value ? "true" : "false";
      return String(value);
    }

    function statePill(value) {
      if (value === true) return `<span class="pill active">yes</span>`;
      if (value === false) return `<span class="pill" style="color: var(--bad)">no</span>`;
      return `<span class="pill inactive">n/a</span>`;
    }

    function modeDescription(snapshot) {
      if ((snapshot.viewerMode || "setup") === "policy") {
        return {
          title: "Policy playback mode",
          help: "Runs a trained checkpoint and shows where that policy gets stuck.",
        };
      }
      return {
        title: "Setup mode",
        help: "Runs zero/random actions to check scene placement, reset behavior, rewards, and thresholds before training.",
      };
    }

    function updateHistory(snapshot) {
      const previous = history.at(-1);
      if (!previous || snapshot.stepCount < previous.step) history = [];
      const signals = snapshot.bottlenecks?.signals || {};
      history.push({
        step: snapshot.stepCount || 0,
        objectZ: Number(signals.objectZ ?? snapshot.metrics?.objectZ ?? 0),
        eeDistance: Number(signals.eeObjectDistance ?? 0),
        gripperCommand: Number(signals.gripperCommand ?? 0),
        reward: Number((snapshot.rewards || []).reduce((sum, row) => sum + (Number(row.weightedEstimate) || 0), 0)),
      });
      if (history.length > 160) history = history.slice(history.length - 160);
    }

    function drawMiniTimeline() {
      const canvas = document.getElementById("timelineCanvas");
      if (!canvas) return;
      const { width, height } = resizeCanvas(canvas);
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = "#151922";
      ctx.fillRect(0, 0, width, height);
      drawGrid(ctx, width, height);
      if (history.length < 2) return;
      const series = [
        { key: "objectZ", color: "#45c486", scale: 0.20 },
        { key: "eeDistance", color: "#f2b84b", scale: 0.25 },
        { key: "gripperCommand", color: "#4aa3ff", scale: 2.0, offset: 1.0 },
        { key: "reward", color: "#ef6a6a", scale: 20.0 },
      ];
      for (const item of series) {
        ctx.strokeStyle = item.color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        history.forEach((point, index) => {
          const x = (index / Math.max(1, history.length - 1)) * width;
          const normalized = Math.max(0, Math.min(1, ((point[item.key] || 0) + (item.offset || 0)) / item.scale));
          const y = height - normalized * height;
          if (index === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.stroke();
      }
    }

    function pos2(asset) {
      const p = asset.position;
      return Array.isArray(p) && p.length >= 3 ? p : null;
    }

    function allPositions(snapshot) {
      const positions = [];
      for (const asset of snapshot.scene?.assets || []) {
        const p = pos2(asset);
        if (p) positions.push({ label: asset.label || asset.name, kind: asset.kind, p });
      }
      for (const frame of snapshot.scene?.frames || []) {
        const p = pos2(frame);
        if (p) positions.push({ label: frame.label || frame.name, kind: "frame", p });
      }
      for (const overlay of snapshot.overlays || []) {
        if (overlay.position) positions.push({ label: overlay.label, kind: "overlay", p: overlay.position });
        if (overlay.z !== undefined) positions.push({ label: overlay.label, kind: "height", p: [0, 0, overlay.z] });
      }
      return positions;
    }

    function bounds(snapshot) {
      const positions = allPositions(snapshot);
      const xs = positions.map(item => item.p[0]);
      const ys = positions.map(item => item.p[1]);
      const zs = positions.map(item => item.p[2]);
      return {
        minX: Math.min(-0.05, ...xs) - 0.15,
        maxX: Math.max(0.9, ...xs) + 0.15,
        minY: Math.min(-0.35, ...ys) - 0.15,
        maxY: Math.max(0.35, ...ys) + 0.15,
        minZ: Math.min(0, ...zs) - 0.03,
        maxZ: Math.max(0.35, ...zs) + 0.08,
      };
    }

    function drawGrid(ctx, width, height) {
      ctx.strokeStyle = "rgba(166, 173, 186, 0.16)";
      ctx.lineWidth = 1;
      for (let x = 0; x <= width; x += 48) {
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, height); ctx.stroke();
      }
      for (let y = 0; y <= height; y += 48) {
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke();
      }
    }

    function drawTop(snapshot) {
      const { width, height } = resizeCanvas(topCanvas);
      const ctx = topCanvas.getContext("2d");
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = "#151922";
      ctx.fillRect(0, 0, width, height);
      drawGrid(ctx, width, height);
      if (!snapshot) return;
      const b = bounds(snapshot);
      const sx = width / (b.maxX - b.minX);
      const sy = height / (b.maxY - b.minY);
      const scale = Math.min(sx, sy);
      const ox = (width - (b.maxX - b.minX) * scale) / 2;
      const oy = (height - (b.maxY - b.minY) * scale) / 2;
      const map = (x, y) => [ox + (x - b.minX) * scale, height - (oy + (y - b.minY) * scale)];

      for (const overlay of snapshot.overlays || []) {
        if (overlay.type === "target_radius" && overlay.position && overlay.radius !== undefined) {
          const [cx, cy] = map(overlay.position[0], overlay.position[1]);
          ctx.strokeStyle = overlay.color || "#4aa3ff";
          ctx.fillStyle = "rgba(74, 163, 255, 0.08)";
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(cx, cy, overlay.radius * scale, 0, Math.PI * 2);
          ctx.fill();
          ctx.stroke();
          ctx.fillStyle = "#edf0f5";
          ctx.font = "12px system-ui";
          ctx.fillText(overlay.label || "target", cx + 8, cy - 8);
        }
      }

      const positions = allPositions(snapshot);
      for (const item of positions) {
        if (item.kind === "height") continue;
        const [x, y] = map(item.p[0], item.p[1]);
        const color = item.kind === "articulation" ? "#45c486" : item.kind === "frame" ? "#f2b84b" : item.kind === "rigid_object" ? "#ef6a6a" : item.kind === "static_asset" ? "#9b8cff" : "#4aa3ff";
        ctx.fillStyle = color;
        ctx.strokeStyle = "#0f1116";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(x, y, item.kind === "frame" ? 5 : 7, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = "#edf0f5";
        ctx.font = "12px system-ui";
        ctx.fillText(item.label, x + 9, y + 4);
      }
    }

    function drawHeight(snapshot) {
      const { width, height } = resizeCanvas(heightCanvas);
      const ctx = heightCanvas.getContext("2d");
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = "#151922";
      ctx.fillRect(0, 0, width, height);
      drawGrid(ctx, width, height);
      if (!snapshot) return;
      const b = bounds(snapshot);
      const map = (x, z) => [
        ((x - b.minX) / (b.maxX - b.minX)) * width,
        height - ((z - b.minZ) / (b.maxZ - b.minZ)) * height,
      ];
      const labelYs = [];
      const labelY = (y) => {
        let candidate = Math.min(height - 10, Math.max(16, y - 6));
        for (let attempt = 0; attempt < 20; attempt += 1) {
          if (!labelYs.some(existing => Math.abs(existing - candidate) < 16)) {
            labelYs.push(candidate);
            return candidate;
          }
          candidate -= 16;
          if (candidate < 16) candidate = Math.min(height - 10, y + 16);
        }
        labelYs.push(candidate);
        return candidate;
      };

      for (const overlay of snapshot.overlays || []) {
        if (overlay.type === "height_plane" && overlay.z !== undefined) {
          const [, y] = map(0, overlay.z);
          const text = `${overlay.label || "height"} z=${fmt(overlay.z)}`;
          const textY = labelY(y);
          ctx.strokeStyle = overlay.color || "#f2b84b";
          ctx.lineWidth = 2;
          ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke();
          ctx.font = "12px system-ui";
          ctx.fillStyle = "rgba(17, 19, 24, 0.86)";
          ctx.fillRect(8, textY - 13, ctx.measureText(text).width + 8, 16);
          ctx.fillStyle = "#edf0f5";
          ctx.fillText(text, 12, textY);
        }
      }

      for (const item of allPositions(snapshot)) {
        if (item.kind === "overlay" || item.kind === "height") continue;
        const [x, y] = map(item.p[0], item.p[2]);
        const color = item.kind === "articulation" ? "#45c486" : item.kind === "frame" ? "#f2b84b" : item.kind === "rigid_object" ? "#ef6a6a" : item.kind === "static_asset" ? "#9b8cff" : "#4aa3ff";
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(x, y, item.kind === "frame" ? 5 : 7, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = "#edf0f5";
        ctx.font = "12px system-ui";
        ctx.fillText(item.label, x + 9, y + 4);
      }
    }

    function table(headers, rows) {
      const head = `<thead><tr>${headers.map(h => `<th class="${h.num ? "num" : ""}">${h.label}</th>`).join("")}</tr></thead>`;
      const body = `<tbody>${rows.map(row => `<tr>${row.map((cell, i) => `<td class="${headers[i]?.num ? "num" : ""}">${cell}</td>`).join("")}</tr>`).join("")}</tbody>`;
      return `<table>${head}${body}</table>`;
    }

    function renderPanels(snapshot) {
      if (!snapshot) {
        statusPanel.innerHTML = `<h2>Status</h2><div class="error">Waiting for snapshot...</div>`;
        return;
      }
      const mode = modeDescription(snapshot);
      modePanel.innerHTML = `<h2>Mode</h2><div class="mode-card">
        <div class="mode-title">${mode.title}</div>
        <div class="mode-help">${mode.help}</div>
      </div>`;

      statusPanel.innerHTML = `<h2>Status</h2><dl class="kv">
        <dt>Task</dt><dd>${snapshot.task || "-"}</dd>
        <dt>Step</dt><dd>${snapshot.stepCount ?? "-"}</dd>
        <dt>Env index</dt><dd>${snapshot.envIndex ?? 0}</dd>
        <dt>Action source</dt><dd>${snapshot.mode || "-"}</dd>
        <dt>Checkpoint</dt><dd>${snapshot.checkpointPath ? snapshot.checkpointPath.split("/").slice(-2).join("/") : "-"}</dd>
        <dt>Updated</dt><dd>${new Date(snapshot.timestamp * 1000).toLocaleTimeString()}</dd>
      </dl>`;

      const stages = snapshot.bottlenecks?.stages || [];
      bottleneckPanel.innerHTML = `<h2>Bottlenecks</h2><div class="stage-grid">${stages.map(stage => {
        const stateClass = stage.active === true ? "good" : stage.active === false ? "bad" : "unknown";
        return `<div class="stage ${stateClass}">
          <div class="stage-name"><span>${stage.label}</span>${statePill(stage.active)}</div>
          <div class="stage-detail">${stage.detail || ""}</div>
        </div>`;
      }).join("") || `<div class="muted">No bottleneck signals available.</div>`}</div>
      <h2 style="margin-top:16px">Episode Signals</h2>
      <dl class="kv">${Object.entries(snapshot.bottlenecks?.episodeSignals || {}).map(([k, v]) => `<dt>${k}</dt><dd>${fmt(v)}</dd>`).join("") || "<dt>none</dt><dd>-</dd>"}</dl>`;

      timelinePanel.innerHTML = `<h2>Live Timeline</h2>
        <canvas id="timelineCanvas" class="timeline"></canvas>
        <div class="legend">
          <span style="--legend-color:#45c486">cube z</span>
          <span style="--legend-color:#f2b84b">hand-cube distance</span>
          <span style="--legend-color:#4aa3ff">gripper command</span>
          <span style="--legend-color:#ef6a6a">reward estimate</span>
        </div>`;

      const metrics = snapshot.metrics || {};
      metricsPanel.innerHTML = `<h2>Key Signals</h2><dl class="kv">${Object.entries(metrics).map(([k, v]) => `<dt>${k}</dt><dd>${fmt(v)}</dd>`).join("") || "<dt>none</dt><dd>-</dd>"}</dl>`;

      rewardPanel.innerHTML = `<h2>Rewards</h2>` + table(
        [
          { label: "Term" },
          { label: "Raw", num: true },
          { label: "Weight", num: true },
          { label: "Pre-dt est.", num: true },
          { label: "Firing", num: true },
        ],
        (snapshot.rewards || []).map(r => [
          `<span class="name">${r.name}</span>`,
          fmt(r.currentRewardValue),
          fmt(r.weight),
          fmt(r.weightedEstimate),
          statePill(Boolean(r.isFiring)),
        ])
      ) + `<div class="mode-help" style="margin-top:8px">Pre-dt estimate is raw value times configured weight. Isaac may still scale reward terms by the environment step time.</div>`;

      probePanel.innerHTML = `<h2>Reward Probes</h2>` + table(
        [{ label: "Probe" }, { label: "Value", num: true }, { label: "Would fire", num: true }, { label: "Meaning" }],
        (snapshot.rewardProbes || []).map(p => [
          `<span class="name">${p.label || p.name}</span>`,
          fmt(p.currentValue),
          statePill(p.wouldFire),
          `<span class="muted">${p.detail || ""}</span>`,
        ])
      );

      terminationPanel.innerHTML = `<h2>Terminations</h2>` + table(
        [{ label: "Term" }, { label: "Active", num: true }, { label: "Params" }],
        (snapshot.terminations || []).map(t => [
          `<span class="name">${t.name}</span>`,
          `<span class="pill ${t.active ? "active" : "inactive"}">${t.active ? "true" : "false"}</span>`,
          `<span class="muted">${t.paramSummary || ""}</span>`,
        ])
      );

      assetPanel.innerHTML = `<h2>Scene Assets</h2>` + table(
        [{ label: "Name" }, { label: "Kind" }, { label: "Position" }],
        (snapshot.scene?.assets || []).map(a => [
          `<span class="name">${a.label || a.name}</span>`,
          a.kind,
          `<span class="muted">[${(a.position || []).map(v => fmt(v)).join(", ")}]</span>`,
        ])
      );

      overlayPanel.innerHTML = `<h2>Overlays</h2>` + table(
        [{ label: "Label" }, { label: "Type" }, { label: "Provenance" }],
        (snapshot.overlays || []).map(o => [
          `<span class="name">${o.label || o.type}</span>`,
          o.type,
          `<span class="muted">${o.source || ""}</span>`,
        ])
      );

      const obsRows = (snapshot.observations || []).map(o => [`${o.group}.${o.name}`, o.shape || "-", o.summary || ""]);
      const actionRows = (snapshot.actions || []).map(a => [a.name, a.shape || "-", a.summary || ""]);
      ioPanel.innerHTML = `<h2>Observations</h2>${table([{ label: "Term" }, { label: "Shape" }, { label: "Value" }], obsRows)}
        <h2 style="margin-top:16px">Actions</h2>${table([{ label: "Term" }, { label: "Shape" }, { label: "Value" }], actionRows)}`;

      const errors = snapshot.diagnosticErrors || [];
      errorPanel.innerHTML = errors.length
        ? `<h2>Diagnostic Errors</h2>${table([{ label: "Collector" }, { label: "Error" }], errors.map(e => [`<span class="name">${e.source}</span>`, `<span class="muted">${e.error}</span>`]))}`
        : `<h2>Diagnostic Errors</h2><div class="muted">none</div>`;
    }

    async function control(command, params = {}) {
      const query = new URLSearchParams({ command, ...params });
      const response = await fetch(`/control?${query.toString()}`);
      if (!response.ok) throw new Error(`Control failed: ${response.status}`);
      const data = await response.json();
      paused = Boolean(data.paused);
      playPause.textContent = paused ? "Play" : "Pause";
    }

    async function loadSnapshot() {
      try {
        const response = await fetch("/snapshot");
        if (!response.ok) throw new Error(`Snapshot failed: ${response.status}`);
        latest = await response.json();
        paused = Boolean(latest.paused);
        playPause.textContent = paused ? "Play" : "Pause";
        updateHistory(latest);
        renderPanels(latest);
        drawTop(latest);
        drawHeight(latest);
        drawMiniTimeline();
      } catch (error) {
        statusPanel.innerHTML = `<h2>Status</h2><div class="error">${error.message}</div>`;
      }
    }

    playPause.addEventListener("click", () => control(paused ? "play" : "pause").then(loadSnapshot));
    stepButton.addEventListener("click", () => control("step").then(loadSnapshot));
    resetButton.addEventListener("click", () => control("reset").then(loadSnapshot));
    speedSelect.addEventListener("change", () => control("speed", { value: speedSelect.value }));
    window.addEventListener("resize", () => { drawTop(latest); drawHeight(latest); drawMiniTimeline(); });

    loadSnapshot();
    setInterval(loadSnapshot, 250);
  </script>
</body>
</html>
"""


class SnapshotEncoder(json.JSONEncoder):
    """JSON encoder that tolerates tensors and scalar-like objects."""

    def default(self, value: Any) -> Any:
        if hasattr(value, "detach"):
            return _to_jsonable(value)
        return super().default(value)


def _to_jsonable(value: Any, env_index: int | None = None, max_items: int = 8) -> Any:
    """Convert tensors and small objects into JSON-safe values."""
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if env_index is not None and getattr(value, "ndim", 0) > 0 and value.shape[0] > env_index:
            value = value[env_index]
        if getattr(value, "numel", lambda: 1)() == 1:
            return float(value.item())
        flat = value.flatten()
        return [float(item) for item in flat[:max_items].tolist()]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item, env_index, max_items) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item, env_index, max_items) for item in value[:max_items]]
    return str(value)


def _shape_summary(value: Any) -> str:
    shape = getattr(value, "shape", None)
    if shape is None:
        if isinstance(value, (list, tuple)):
            return str(len(value))
        return "-"
    return "x".join(str(item) for item in shape)


def _to_scalar(value: Any, env_index: int) -> float | bool | str | None:
    """Convert manager term values into a scalar when Isaac returns a one-item sequence."""
    json_value = _to_jsonable(value, env_index)
    if isinstance(json_value, list) and len(json_value) == 1:
        return json_value[0]
    if isinstance(json_value, (float, int, bool, str)) or json_value is None:
        return json_value
    return str(json_value)


def _number(value: Any) -> float | None:
    """Return a float for scalar-like values, otherwise None."""
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list) and len(value) == 1:
        return _number(value[0])
    return None


def _weighted_estimate(raw_value: Any, weight: Any) -> float | None:
    """Estimate a reward contribution from a scalar raw term and its configured weight."""
    raw_number = _number(raw_value)
    weight_number = _number(weight)
    if raw_number is None or weight_number is None:
        return None
    return raw_number * weight_number


def _position_from_data(data: Any, env_index: int) -> list[float] | None:
    pos = getattr(data, "root_pos_w", None)
    if pos is None:
        return None
    return _to_jsonable(pos, env_index, max_items=3)


def _quat_from_data(data: Any, env_index: int) -> list[float] | None:
    quat = getattr(data, "root_quat_w", None)
    if quat is None:
        return None
    return _to_jsonable(quat, env_index, max_items=4)


def _manager_term_cfgs(manager: Any) -> dict[str, Any]:
    names = list(getattr(manager, "_term_names", []) or [])
    cfgs = list(getattr(manager, "_term_cfgs", []) or [])
    return {name: cfg for name, cfg in zip(names, cfgs, strict=False)}


def _manager_current_terms(manager: Any, env_index: int) -> dict[str, Any]:
    try:
        terms = manager.get_active_iterable_terms(env_index)
    except Exception:
        return {}
    current: dict[str, Any] = {}
    for item in terms:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        current[str(item[0])] = item[1]
    return current


def _param_summary(params: dict[str, Any], env_index: int) -> str:
    if not params:
        return ""
    parts = []
    for key, value in params.items():
        parts.append(f"{key}={_to_jsonable(value, env_index, max_items=3)}")
    return ", ".join(parts[:4])


def _collect_assets(scene: Any, env_index: int) -> list[dict[str, Any]]:
    assets: list[dict[str, Any]] = []
    for kind, collection_name in (
        ("articulation", "articulations"),
        ("rigid_object", "rigid_objects"),
        ("deformable_object", "deformable_objects"),
    ):
        collection = getattr(scene, collection_name, {}) or {}
        for name, asset in collection.items():
            position = _position_from_data(getattr(asset, "data", None), env_index)
            if position is None:
                continue
            assets.append(
                {
                    "name": name,
                    "label": name,
                    "kind": kind,
                    "position": position,
                    "orientation": _quat_from_data(asset.data, env_index),
                    "source": f"env.scene.{collection_name}.{name}",
                }
            )
    return assets


def _collect_static_assets(env: Any, env_index: int) -> list[dict[str, Any]]:
    """Collect configured static scene assets such as the visual bowl and table."""
    scene_cfg = getattr(getattr(env, "cfg", None), "scene", None)
    env_origin = None
    try:
        env_origin = env.scene.env_origins[env_index].detach().cpu()
    except Exception:
        pass

    assets: list[dict[str, Any]] = []
    for name, cfg in vars(scene_cfg).items() if scene_cfg is not None else []:
        if name.startswith("_"):
            continue
        if name in getattr(env.scene, "articulations", {}) or name in getattr(env.scene, "rigid_objects", {}):
            continue
        init_state = getattr(cfg, "init_state", None)
        position = getattr(init_state, "pos", None)
        if position is None:
            continue
        local_position = list(position)
        if env_origin is not None and len(local_position) >= 3:
            local_position = [float(local_position[index] - env_origin[index]) for index in range(3)]
        assets.append(
            {
                "name": name,
                "label": name,
                "kind": "static_asset",
                "position": local_position[:3],
                "orientation": getattr(init_state, "rot", None),
                "source": f"env.cfg.scene.{name}.init_state",
            }
        )
    return assets


def _collect_frames(scene: Any, env_index: int) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    sensors = getattr(scene, "sensors", {}) or {}
    for name, sensor in sensors.items():
        data = getattr(sensor, "data", None)
        target_pos = getattr(data, "target_pos_w", None)
        target_quat = getattr(data, "target_quat_w", None)
        if target_pos is None:
            continue
        positions = target_pos.detach().cpu()
        quats = target_quat.detach().cpu() if target_quat is not None else None
        if positions.ndim == 2:
            positions = positions[:, None, :]
        for target_index, position in enumerate(positions[env_index]):
            frames.append(
                {
                    "name": f"{name}:{target_index}",
                    "label": name if target_index == 0 else f"{name}:{target_index}",
                    "kind": "frame",
                    "position": [float(item) for item in position[:3].tolist()],
                    "orientation": (
                        [float(item) for item in quats[env_index, target_index, :4].tolist()]
                        if quats is not None and quats.ndim >= 3
                        else None
                    ),
                    "source": f"env.scene.sensors.{name}",
                }
            )
    return frames


def _collect_rewards(env: Any, env_index: int) -> list[dict[str, Any]]:
    manager = getattr(env, "reward_manager", None)
    if manager is None:
        return []
    cfgs = _manager_term_cfgs(manager)
    current = _manager_current_terms(manager, env_index)
    rows = []
    for name, cfg in cfgs.items():
        weight = _to_jsonable(getattr(cfg, "weight", None), env_index)
        current_value = _to_scalar(current.get(name), env_index)
        rows.append(
            {
                "name": name,
                "weight": weight,
                "params": _to_jsonable(getattr(cfg, "params", {}) or {}, env_index),
                "currentRewardValue": current_value,
                "weightedEstimate": _weighted_estimate(current_value, weight),
                "isFiring": abs(_number(current_value) or 0.0) > 1.0e-6,
                "source": "reward_manager",
            }
        )
    return rows


def _collect_terminations(env: Any, env_index: int) -> list[dict[str, Any]]:
    manager = getattr(env, "termination_manager", None)
    if manager is None:
        return []
    cfgs = _manager_term_cfgs(manager)
    current = _manager_current_terms(manager, env_index)
    rows = []
    for name, cfg in cfgs.items():
        value = current.get(name)
        scalar_value = _to_scalar(value, env_index)
        active = bool(scalar_value) if scalar_value is not None else False
        params = getattr(cfg, "params", {}) or {}
        rows.append(
            {
                "name": name,
                "active": active,
                "currentValue": scalar_value,
                "params": _to_jsonable(params, env_index),
                "paramSummary": _param_summary(params, env_index),
                "source": "termination_manager",
            }
        )
    return rows


def _collect_observations(env: Any, env_index: int) -> list[dict[str, Any]]:
    manager = getattr(env, "observation_manager", None)
    if manager is None:
        return []
    rows = []
    current = _manager_current_terms(manager, env_index)
    if current:
        for key, value in current.items():
            group_name, separator, term_name = key.partition("-")
            value_summary = _to_jsonable(value, None, max_items=6)
            rows.append(
                {
                    "group": group_name if separator else "observation",
                    "name": term_name if separator else group_name,
                    "shape": _shape_summary(value),
                    "summary": str(value_summary),
                    "source": "observation_manager",
                }
            )
        return rows

    try:
        observations = manager.compute()
    except Exception:
        observations = {}
    if not isinstance(observations, dict):
        observations = {"policy": observations}
    for group_name, group_value in observations.items():
        if isinstance(group_value, dict):
            terms = group_value.items()
        else:
            terms = [("concatenated", group_value)]
        for name, value in terms:
            rows.append(
                {
                    "group": str(group_name),
                    "name": str(name),
                    "shape": _shape_summary(value),
                    "summary": str(_to_jsonable(value, env_index, max_items=6)),
                    "source": "observation_manager",
                }
            )
    return rows


def _collect_actions(env: Any, env_index: int) -> list[dict[str, Any]]:
    manager = getattr(env, "action_manager", None)
    if manager is None:
        return []
    names = list(getattr(manager, "_term_names", []) or [])
    rows = []
    for name in names:
        try:
            term = manager.get_term(name)
        except Exception:
            continue
        raw_actions = getattr(term, "raw_actions", None)
        rows.append(
            {
                "name": name,
                "shape": _shape_summary(raw_actions),
                "summary": str(_to_jsonable(raw_actions, env_index, max_items=6)),
                "source": "action_manager",
            }
        )
    return rows


def _collect_overlays(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    overlays: list[dict[str, Any]] = []
    term_groups = (
        ("reward", snapshot.get("rewards", [])),
        ("termination", snapshot.get("terminations", [])),
    )
    for group, terms in term_groups:
        for term in terms:
            params = term.get("params") or {}
            source = f"{group}.{term['name']}.params"
            target_position = params.get("target_position")
            radius = params.get("radius")
            if isinstance(target_position, list) and len(target_position) >= 3 and isinstance(radius, (int, float)):
                overlays.append(
                    {
                        "type": "target_radius",
                        "label": f"{term['name']} radius",
                        "position": target_position[:3],
                        "radius": radius,
                        "source": source,
                        "confidence": "medium",
                        "color": "#4aa3ff" if group == "reward" else "#45c486",
                    }
                )
            for key in ("minimal_height", "minimum_height", "min_height", "max_height", "target_height"):
                value = params.get(key)
                if isinstance(value, (int, float)):
                    overlays.append(
                        {
                            "type": "height_plane",
                            "label": f"{term['name']} {key}",
                            "z": value,
                            "source": f"{source}.{key}",
                            "confidence": "low" if key in ("min_height", "max_height", "minimum_height") else "medium",
                            "color": "#f2b84b" if group == "reward" else "#ef6a6a",
                        }
                    )
    return overlays


def _find_asset(snapshot: dict[str, Any], name: str) -> dict[str, Any] | None:
    for asset in snapshot.get("scene", {}).get("assets", []):
        if asset.get("name") == name:
            return asset
    return None


def _find_first_frame(snapshot: dict[str, Any]) -> dict[str, Any] | None:
    frames = snapshot.get("scene", {}).get("frames", [])
    return frames[0] if frames else None


def _distance(a: list[float] | None, b: list[float] | None) -> float | None:
    if not a or not b or len(a) < 3 or len(b) < 3:
        return None
    return math.sqrt(sum((a[index] - b[index]) ** 2 for index in range(3)))


def _collect_metrics(snapshot: dict[str, Any]) -> dict[str, Any]:
    object_asset = _find_asset(snapshot, "object")
    robot_asset = _find_asset(snapshot, "robot")
    frame = _find_first_frame(snapshot)
    target_overlay = next((item for item in snapshot.get("overlays", []) if item.get("type") == "target_radius"), None)
    object_position = object_asset.get("position") if object_asset else None
    return {
        "objectZ": object_position[2] if object_position else None,
        "eeObjectDistance": _distance(frame.get("position") if frame else None, object_position),
        "objectTargetDistance": _distance(object_position, target_overlay.get("position") if target_overlay else None),
        "robotObjectDistance": _distance(robot_asset.get("position") if robot_asset else None, object_position),
    }


def _reward_param(snapshot: dict[str, Any], reward_name: str, param_name: str) -> Any:
    for reward in snapshot.get("rewards", []):
        if reward.get("name") == reward_name:
            return (reward.get("params") or {}).get(param_name)
    return None


def _termination_param(snapshot: dict[str, Any], termination_name: str, param_name: str) -> Any:
    for termination in snapshot.get("terminations", []):
        if termination.get("name") == termination_name:
            return (termination.get("params") or {}).get(param_name)
    return None


def _xy_distance(a: list[float] | None, b: list[float] | None) -> float | None:
    if not a or not b or len(a) < 2 or len(b) < 2:
        return None
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _stage(name: str, label: str, active: bool | None, detail: str, value: Any = None) -> dict[str, Any]:
    return {
        "name": name,
        "label": label,
        "active": active,
        "detail": detail,
        "value": value,
    }


def _get_tensor_scalar(value: Any, env_index: int) -> float | None:
    return _number(_to_jsonable(value, env_index))


def _record_error(errors: list[dict[str, str]], source: str, exc: Exception) -> None:
    errors.append({"source": source, "error": f"{type(exc).__name__}: {exc}"})


def _collect_gripper_signals(env: Any, env_index: int, errors: list[dict[str, str]]) -> dict[str, Any]:
    signals: dict[str, Any] = {}
    try:
        robot = env.scene["robot"]
        finger_cfg = getattr(env, "_finger_robot_cfg", None)
        joint_ids = getattr(finger_cfg, "joint_ids", None)
        if joint_ids is not None:
            finger_joint_pos = robot.data.joint_pos[env_index, joint_ids]
            signals["fingerPositions"] = _to_jsonable(finger_joint_pos, None, max_items=4)
            signals["gripperOpening"] = _get_tensor_scalar(finger_joint_pos.sum(), 0)
            signals["minFingerPosition"] = _get_tensor_scalar(finger_joint_pos.min(), 0)
    except Exception as exc:
        _record_error(errors, "gripper_joint_signals", exc)

    try:
        raw_actions = env.action_manager.get_term("gripper_action").raw_actions
        action = _to_jsonable(raw_actions, env_index, max_items=1)
        if isinstance(action, list) and action:
            action = action[0]
        signals["gripperCommand"] = action
        signals["closeCommandActive"] = (_number(action) or 0.0) < 0.0
    except Exception as exc:
        _record_error(errors, "gripper_action_signals", exc)
    return signals


def _collect_object_motion(env: Any, env_index: int, errors: list[dict[str, str]]) -> dict[str, Any]:
    try:
        object_asset = env.scene["object"]
        linear_speed = object_asset.data.root_lin_vel_w[env_index, :3].norm()
        angular_speed = object_asset.data.root_ang_vel_w[env_index, :3].norm()
        return {
            "objectLinearSpeed": _get_tensor_scalar(linear_speed, 0),
            "objectAngularSpeed": _get_tensor_scalar(angular_speed, 0),
        }
    except Exception as exc:
        _record_error(errors, "object_motion_signals", exc)
        return {}


def _collect_episode_signals(env: Any, env_index: int) -> dict[str, Any]:
    signal_names = (
        "_episode_max_lift_progress",
        "_episode_max_object_z_delta",
        "_episode_min_ee_object_distance",
        "_episode_min_gripper_opening",
        "_episode_close_command_hit",
        "_episode_close_near_object_hit",
        "_episode_max_lift_progress_after_close_near_object",
    )
    signals = {}
    for name in signal_names:
        value = getattr(env, name, None)
        if value is not None:
            signals[name.removeprefix("_episode_")] = _to_jsonable(value, env_index)
    return signals


def _collect_bottlenecks(
    env: Any, snapshot: dict[str, Any], env_index: int, errors: list[dict[str, str]]
) -> dict[str, Any]:
    """Collect task-stage signals that explain where an attempt is getting stuck."""
    metrics = dict(snapshot.get("metrics", {}))
    object_asset = _find_asset(snapshot, "object")
    object_position = object_asset.get("position") if object_asset else None
    target_position = _termination_param(snapshot, "object_in_bowl", "target_position")
    lifted_height = _reward_param(snapshot, "lifting_object", "minimal_height")
    bowl_radius = _termination_param(snapshot, "object_in_bowl", "radius")
    bowl_min_height = _termination_param(snapshot, "object_in_bowl", "min_height")
    bowl_max_height = _termination_param(snapshot, "object_in_bowl", "max_height")
    max_speed = _termination_param(snapshot, "object_in_bowl", "max_speed")
    max_angular_speed = _termination_param(snapshot, "object_in_bowl", "max_angular_speed")
    min_gripper_open = _termination_param(snapshot, "object_in_bowl", "min_gripper_open")

    gripper = _collect_gripper_signals(env, env_index, errors)
    motion = _collect_object_motion(env, env_index, errors)
    metrics.update(gripper)
    metrics.update(motion)

    ee_distance = _number(metrics.get("eeObjectDistance"))
    object_z = _number(metrics.get("objectZ"))
    xy_distance = _xy_distance(object_position, target_position)
    linear_speed = _number(metrics.get("objectLinearSpeed"))
    angular_speed = _number(metrics.get("objectAngularSpeed"))
    min_finger_position = _number(metrics.get("minFingerPosition"))
    close_command_active = gripper.get("closeCommandActive")

    near_object = ee_distance is not None and ee_distance < DEFAULT_NEAR_OBJECT_DISTANCE
    lifted = object_z is not None and _number(lifted_height) is not None and object_z > float(lifted_height)
    over_bowl = xy_distance is not None and _number(bowl_radius) is not None and xy_distance < float(bowl_radius)
    inside_height = (
        object_z is not None
        and _number(bowl_min_height) is not None
        and _number(bowl_max_height) is not None
        and float(bowl_min_height) < object_z < float(bowl_max_height)
    )
    settled = (
        linear_speed is not None
        and angular_speed is not None
        and _number(max_speed) is not None
        and _number(max_angular_speed) is not None
        and linear_speed < float(max_speed)
        and angular_speed < float(max_angular_speed)
    )
    gripper_open_for_release = (
        min_finger_position is not None
        and _number(min_gripper_open) is not None
        and min_finger_position > float(min_gripper_open)
    )

    stages = [
        _stage(
            "near_object",
            "Hand near cube",
            near_object if ee_distance is not None else None,
            f"ee-object {fmt_python(ee_distance)}m, target < {DEFAULT_NEAR_OBJECT_DISTANCE:.3f}m",
            ee_distance,
        ),
        _stage(
            "close_command",
            "Close command",
            bool(close_command_active) if close_command_active is not None else None,
            f"gripper command {fmt_python(metrics.get('gripperCommand'))}",
            metrics.get("gripperCommand"),
        ),
        _stage(
            "close_near_object",
            "Close while near cube",
            (
                (bool(close_command_active) and near_object)
                if close_command_active is not None and ee_distance is not None
                else None
            ),
            "close command and hand-near-cube are both true",
        ),
        _stage(
            "object_lifted",
            "Cube lifted",
            lifted if object_z is not None and lifted_height is not None else None,
            f"cube z {fmt_python(object_z)}m, lift threshold {fmt_python(lifted_height)}m",
            object_z,
        ),
        _stage(
            "over_bowl",
            "Cube over bowl",
            over_bowl if xy_distance is not None and bowl_radius is not None else None,
            f"xy distance {fmt_python(xy_distance)}m, radius {fmt_python(bowl_radius)}m",
            xy_distance,
        ),
        _stage(
            "inside_bowl_height",
            "Cube at bowl height",
            (
                inside_height
                if object_z is not None and bowl_min_height is not None and bowl_max_height is not None
                else None
            ),
            f"cube z {fmt_python(object_z)}m, allowed {fmt_python(bowl_min_height)}-{fmt_python(bowl_max_height)}m",
            object_z,
        ),
        _stage(
            "settled",
            "Cube settled",
            settled if linear_speed is not None and angular_speed is not None else None,
            f"speed {fmt_python(linear_speed)}m/s, angular {fmt_python(angular_speed)}rad/s",
        ),
        _stage(
            "released",
            "Gripper released",
            gripper_open_for_release if min_finger_position is not None and min_gripper_open is not None else None,
            f"min finger {fmt_python(min_finger_position)}m, release threshold > {fmt_python(min_gripper_open)}m",
            min_finger_position,
        ),
    ]
    success = all(stage["active"] is True for stage in stages[4:8])
    stages.append(_stage("success_gate", "Success gate", success, "over bowl, correct height, settled, and released"))
    return {
        "stages": stages,
        "signals": metrics,
        "episodeSignals": _collect_episode_signals(env, env_index),
    }


def fmt_python(value: Any) -> str:
    number = _number(value)
    if number is None:
        return "-"
    return f"{number:.3f}"


def _probe_row(name: str, label: str, value: Any, detail: str, env_index: int) -> dict[str, Any]:
    scalar = _to_scalar(value, env_index)
    return {
        "name": name,
        "label": label,
        "currentValue": scalar,
        "wouldFire": (_number(scalar) or 0.0) > 1.0e-6,
        "detail": detail,
        "source": "viewer_probe",
    }


def _collect_reward_probes(env: Any, env_index: int, errors: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Evaluate explicit inactive reward/check helpers as zero-weight probes."""
    try:
        from object_in_bowl.tasks.manager_based.object_in_bowl import mdp
        from object_in_bowl.tasks.manager_based.object_in_bowl.object_in_bowl_env_cfg import (
            BOWL_LOWERING_RADIUS,
            BOWL_LOWERING_REWARD_MIN_HEIGHT,
            BOWL_LOWERING_TARGET_HEIGHT,
            BOWL_SUCCESS_MAX_ANGULAR_SPEED,
            BOWL_SUCCESS_MAX_HEIGHT,
            BOWL_SUCCESS_MAX_SPEED,
            BOWL_SUCCESS_MIN_GRIPPER_OPEN,
            BOWL_SUCCESS_MIN_HEIGHT,
            BOWL_SUCCESS_RADIUS,
            OBJECT_LIFTED_HEIGHT,
            OBJECT_START_POSITION,
            PLACEMENT_TARGET_POSITION,
        )
        from isaaclab.managers import SceneEntityCfg
    except Exception as exc:
        _record_error(errors, "reward_probe_imports", exc)
        return []

    robot_cfg = SceneEntityCfg("robot", joint_names=["panda_finger.*"])
    try:
        robot_cfg.resolve(env.scene)
    except Exception as exc:
        _record_error(errors, "reward_probe_robot_cfg", exc)

    probes = []
    probe_specs = (
        (
            "grasping_object",
            "Would grasp reward fire?",
            lambda: mdp.compute_grasping_object_reward(env, std=0.10, minimal_height=OBJECT_LIFTED_HEIGHT),
            "near cube + closing gripper before lift",
        ),
        (
            "height_progress",
            "Would height-progress reward fire?",
            lambda: mdp.compute_object_height_progress_reward(
                env, initial_height=OBJECT_START_POSITION[2], target_height=OBJECT_LIFTED_HEIGHT
            ),
            "smooth 0-to-1 progress from table height to lift threshold",
        ),
        (
            "object_to_bowl",
            "Would carry-to-bowl reward fire?",
            lambda: mdp.compute_object_to_target_reward(
                env,
                target_position=PLACEMENT_TARGET_POSITION,
                std=0.30,
                minimal_height=BOWL_LOWERING_REWARD_MIN_HEIGHT,
            ),
            "cube closer to bowl target after a low lift gate",
        ),
        (
            "lowering_into_bowl",
            "Would lowering reward fire?",
            lambda: mdp.compute_object_lowering_into_bowl_reward(
                env,
                target_position=PLACEMENT_TARGET_POSITION,
                radius=BOWL_LOWERING_RADIUS,
                target_height=BOWL_LOWERING_TARGET_HEIGHT,
                height_std=0.05,
                minimal_height=BOWL_LOWERING_REWARD_MIN_HEIGHT,
            ),
            "cube over bowl radius and near bowl height",
        ),
        (
            "object_in_bowl_success",
            "Would bowl success reward fire?",
            lambda: mdp.compute_object_in_bowl_success_reward(
                env,
                target_position=PLACEMENT_TARGET_POSITION,
                radius=BOWL_SUCCESS_RADIUS,
                min_height=BOWL_SUCCESS_MIN_HEIGHT,
                max_height=BOWL_SUCCESS_MAX_HEIGHT,
                max_speed=BOWL_SUCCESS_MAX_SPEED,
                max_angular_speed=BOWL_SUCCESS_MAX_ANGULAR_SPEED,
                min_gripper_open=BOWL_SUCCESS_MIN_GRIPPER_OPEN,
                robot_cfg=robot_cfg,
            ),
            "inside bowl radius, correct height, slow, not spinning, released",
        ),
    )
    for name, label, func, detail in probe_specs:
        try:
            probes.append(_probe_row(name, label, func(), detail, env_index))
        except Exception as exc:
            _record_error(errors, f"reward_probe.{name}", exc)
            probes.append(
                {
                    "name": name,
                    "label": label,
                    "currentValue": None,
                    "wouldFire": None,
                    "detail": f"{detail}; probe unavailable: {exc}",
                    "source": "viewer_probe",
                }
            )
    return probes


def build_snapshot(env: Any, task: str, env_index: int, step_count: int, mode: str, paused: bool) -> dict[str, Any]:
    """Build a normalized snapshot from a manager-based Isaac Lab environment."""
    unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
    diagnostic_errors: list[dict[str, str]] = []
    snapshot = {
        "schemaVersion": 1,
        "task": task,
        "mode": mode,
        "envIndex": env_index,
        "stepCount": step_count,
        "paused": paused,
        "timestamp": time.time(),
        "scene": {
            "assets": _collect_assets(unwrapped.scene, env_index) + _collect_static_assets(unwrapped, env_index),
            "frames": _collect_frames(unwrapped.scene, env_index),
        },
        "rewards": _collect_rewards(unwrapped, env_index),
        "terminations": _collect_terminations(unwrapped, env_index),
        "observations": _collect_observations(unwrapped, env_index),
        "actions": _collect_actions(unwrapped, env_index),
    }
    snapshot["overlays"] = _collect_overlays(snapshot)
    snapshot["metrics"] = _collect_metrics(snapshot)
    snapshot["bottlenecks"] = _collect_bottlenecks(unwrapped, snapshot, env_index, diagnostic_errors)
    snapshot["rewardProbes"] = _collect_reward_probes(unwrapped, env_index, diagnostic_errors)
    snapshot["diagnosticErrors"] = diagnostic_errors
    return snapshot


class DebugRuntime:
    """Small state holder for serving snapshots and stepping one env."""

    def __init__(
        self,
        env: Any | None,
        task: str,
        action_source: str,
        mock: bool = False,
        viewer_mode: str = "setup",
        checkpoint_path: str | None = None,
        policy: Any | None = None,
        policy_model: Any | None = None,
    ):
        self.env = env
        self.task = task
        self.action_source = action_source
        self.mock = mock
        self.viewer_mode = viewer_mode
        self.checkpoint_path = checkpoint_path
        self.policy = policy
        self.policy_model = policy_model
        self.obs = None
        self.paused = False
        self.step_count = 0
        self.steps_per_tick = 1
        if self.env is not None:
            self.env.reset()
            self.obs = self._get_observations()

    def _get_observations(self):
        if self.env is None:
            return None
        if hasattr(self.env, "get_observations"):
            return self.env.get_observations()
        try:
            return self.env.unwrapped.observation_manager.compute()
        except Exception:
            return None

    def make_action(self):
        import torch

        if self.env is None:
            return None
        device = self.env.unwrapped.device
        shape = self.env.action_space.shape
        if self.action_source == "random":
            return 2.0 * torch.rand(shape, device=device) - 1.0
        return torch.zeros(shape, device=device)

    def step(self, count: int = 1) -> None:
        if self.mock:
            self.step_count += count
            return
        if self.env is None:
            return
        import torch

        with torch.inference_mode():
            for _ in range(count):
                if self.policy is not None:
                    if self.obs is None:
                        self.obs = self._get_observations()
                    actions = self.policy(self.obs)
                    self.obs, _, dones, _ = self.env.step(actions)
                    if self.policy_model is not None and hasattr(self.policy_model, "reset"):
                        self.policy_model.reset(dones)
                else:
                    self.env.step(self.make_action())
                self.step_count += 1

    def reset(self) -> None:
        if self.mock:
            self.step_count = 0
            return
        if self.env is not None:
            self.env.reset()
            self.obs = self._get_observations()
        if self.policy_model is not None and hasattr(self.policy_model, "reset"):
            try:
                self.policy_model.reset()
            except TypeError:
                pass
        self.step_count = 0

    def snapshot(self) -> dict[str, Any]:
        if self.mock:
            return build_mock_snapshot(self.step_count, self.paused, self.viewer_mode)
        snapshot = build_snapshot(self.env, self.task, 0, self.step_count, self.action_source, self.paused)
        snapshot["viewerMode"] = self.viewer_mode
        snapshot["checkpointPath"] = self.checkpoint_path
        return snapshot


def build_mock_snapshot(step_count: int, paused: bool, viewer_mode: str = "setup") -> dict[str, Any]:
    """Build a fake snapshot for browser verification without Isaac Lab installed."""
    phase = step_count / 18.0
    cube = [0.5 + 0.03 * math.sin(phase), 0.02 * math.cos(phase), 0.055 + 0.015 * max(0, math.sin(phase))]
    ee = [cube[0] - 0.05 * math.cos(phase), cube[1] + 0.04 * math.sin(phase), cube[2] + 0.11]
    target = [0.7, 0.2, 0.049]
    mock_gripper_command = -0.72 if viewer_mode == "policy" else 0.0
    return {
        "schemaVersion": 1,
        "task": "Mock-Manager-Based-Task-v0",
        "mode": "mock",
        "viewerMode": viewer_mode,
        "checkpointPath": "/mock/logs/model_499.pt" if viewer_mode == "policy" else None,
        "envIndex": 0,
        "stepCount": step_count,
        "paused": paused,
        "timestamp": time.time(),
        "scene": {
            "assets": [
                {"name": "robot", "label": "robot", "kind": "articulation", "position": [0.0, 0.0, 0.0]},
                {"name": "object", "label": "object", "kind": "rigid_object", "position": cube},
                {"name": "bowl", "label": "bowl", "kind": "rigid_object", "position": [0.7, 0.2, 0.025]},
            ],
            "frames": [{"name": "ee_frame:0", "label": "ee_frame", "kind": "frame", "position": ee}],
        },
        "rewards": [
            {
                "name": "reaching_object",
                "weight": 1.0,
                "currentRewardValue": 0.6,
                "weightedEstimate": 0.6,
                "isFiring": True,
                "params": {"std": 0.1},
            },
            {
                "name": "lifting_object",
                "weight": 15.0,
                "currentRewardValue": 0.0,
                "weightedEstimate": 0.0,
                "isFiring": False,
                "params": {"minimal_height": 0.10500000000000001},
            },
        ],
        "terminations": [
            {
                "name": "object_in_bowl",
                "active": False,
                "params": {"target_position": target, "radius": 0.11, "min_height": 0.044, "max_height": 0.109},
                "paramSummary": "target_position=[0.7, 0.2, 0.049], radius=0.11",
            }
        ],
        "observations": [
            {"group": "policy", "name": "concatenated", "shape": "1x36", "summary": "[...mock values...]"}
        ],
        "actions": [{"name": "arm_action", "shape": "1x7", "summary": "[0, 0, 0, 0, 0, 0]"}],
        "overlays": [
            {
                "type": "height_plane",
                "label": "lifting_object minimal_height",
                "z": 0.10500000000000001,
                "source": "reward.lifting_object.params.minimal_height",
                "confidence": "medium",
                "color": "#f2b84b",
            },
            {
                "type": "target_radius",
                "label": "object_in_bowl radius",
                "position": target,
                "radius": 0.11,
                "source": "termination.object_in_bowl.params",
                "confidence": "medium",
                "color": "#45c486",
            },
        ],
        "bottlenecks": {
            "stages": [
                _stage("near_object", "Hand near cube", True, "ee-object 0.045m, target < 0.080m", 0.045),
                _stage(
                    "close_command",
                    "Close command",
                    viewer_mode == "policy",
                    f"gripper command {mock_gripper_command:.3f}",
                    mock_gripper_command,
                ),
                _stage(
                    "close_near_object",
                    "Close while near cube",
                    viewer_mode == "policy",
                    "close command and hand-near-cube are both true",
                ),
                _stage("object_lifted", "Cube lifted", cube[2] > 0.105, "cube z is above lift threshold", cube[2]),
                _stage("over_bowl", "Cube over bowl", False, "xy distance still outside bowl radius"),
                _stage("inside_bowl_height", "Cube at bowl height", False, "cube is not yet at bowl placement height"),
                _stage("settled", "Cube settled", True, "mock cube speed is low"),
                _stage("released", "Gripper released", False, "mock gripper is not released"),
                _stage("success_gate", "Success gate", False, "over bowl, correct height, settled, and released"),
            ],
            "signals": {
                "objectZ": cube[2],
                "eeObjectDistance": _distance(ee, cube),
                "objectTargetDistance": _distance(cube, target),
                "gripperCommand": mock_gripper_command,
                "gripperOpening": 0.025,
                "objectLinearSpeed": 0.04,
            },
            "episodeSignals": {
                "max_lift_progress": max(0.0, min(1.0, (cube[2] - 0.055) / 0.05)),
                "min_ee_object_distance": 0.045,
                "close_near_object_hit": 1.0 if viewer_mode == "policy" else 0.0,
            },
        },
        "rewardProbes": [
            {
                "name": "grasping_object",
                "label": "Would grasp reward fire?",
                "currentValue": 0.5 if viewer_mode == "policy" else 0.0,
                "wouldFire": viewer_mode == "policy",
                "detail": "near cube + closing gripper before lift",
                "source": "viewer_probe",
            },
            {
                "name": "height_progress",
                "label": "Would height-progress reward fire?",
                "currentValue": max(0.0, min(1.0, (cube[2] - 0.055) / 0.05)),
                "wouldFire": cube[2] > 0.055,
                "detail": "smooth 0-to-1 progress from table height to lift threshold",
                "source": "viewer_probe",
            },
        ],
        "diagnosticErrors": [],
        "metrics": {
            "objectZ": cube[2],
            "eeObjectDistance": _distance(ee, cube),
            "objectTargetDistance": _distance(cube, target),
        },
    }


def make_handler(runtime: DebugRuntime):
    class DebugViewerHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        def _send(self, content: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
            self.send_response(status.value)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)

        def _send_json(self, payload: dict[str, Any]) -> None:
            self._send(json.dumps(payload, cls=SnapshotEncoder).encode("utf-8"), "application/json")

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send(HTML_PAGE.encode("utf-8"), "text/html; charset=utf-8")
                return
            if parsed.path == "/favicon.ico":
                self._send(b"", "image/x-icon", HTTPStatus.NO_CONTENT)
                return
            if parsed.path == "/snapshot":
                if not runtime.paused:
                    runtime.step(runtime.steps_per_tick)
                self._send_json(runtime.snapshot())
                return
            if parsed.path == "/control":
                query = parse_qs(parsed.query)
                command = query.get("command", [""])[0]
                if command == "pause":
                    runtime.paused = True
                elif command == "play":
                    runtime.paused = False
                elif command == "step":
                    runtime.step(1)
                    runtime.paused = True
                elif command == "reset":
                    runtime.reset()
                    runtime.paused = True
                elif command == "speed":
                    value = int(query.get("value", ["1"])[0])
                    runtime.steps_per_tick = max(1, min(value, 25))
                self._send_json({"paused": runtime.paused, "stepsPerTick": runtime.steps_per_tick})
                return
            self._send(b"not found", "text/plain", HTTPStatus.NOT_FOUND)

    return DebugViewerHandler


def serve(runtime: DebugRuntime, host: str, port: int) -> None:
    server = HTTPServer((host, port), make_handler(runtime))
    print(f"[INFO] Debug viewer listening on http://{host}:{port}")
    try:
        server.serve_forever()
    finally:
        server.server_close()


def run_mock(args: argparse.Namespace) -> None:
    runtime = DebugRuntime(
        env=None,
        task="Mock-Manager-Based-Task-v0",
        action_source="mock",
        mock=True,
        viewer_mode=args.viewer_mode,
    )
    if args.dump_json:
        args.dump_json.write_text(json.dumps(runtime.snapshot(), indent=2, cls=SnapshotEncoder))
        print(f"[INFO] Wrote mock snapshot: {args.dump_json}")
        return
    serve(runtime, args.host, args.port)


def parse_mock_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the Isaac Lab debug viewer with mock data.")
    parser.add_argument("--mock", action="store_true", help="Run without Isaac Lab using fake data.")
    parser.add_argument("--serve", action="store_true", help="Serve the browser viewer.")
    parser.add_argument("--viewer-mode", choices=("setup", "policy"), default="setup", help="Mock viewer mode.")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP host.")
    parser.add_argument("--port", type=int, default=8080, help="HTTP port.")
    parser.add_argument("--dump-json", type=Path, default=None, help="Write one snapshot JSON file and exit.")
    return parser.parse_args()


def _latest_checkpoint_in(run_dir: Path) -> Path:
    checkpoints = sorted(run_dir.glob("model_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No model_*.pt checkpoints found in {run_dir}")

    def checkpoint_step(path: Path) -> int:
        stem = path.stem.removeprefix("model_")
        return int(stem) if stem.isdigit() else -1

    return max(checkpoints, key=checkpoint_step)


def _resolve_checkpoint(args: argparse.Namespace, agent_cfg: Any) -> str:
    if args.checkpoint:
        from isaaclab.utils.assets import retrieve_file_path

        return retrieve_file_path(args.checkpoint)
    if args.run_dir:
        return str(_latest_checkpoint_in(Path(args.run_dir)))

    from isaaclab_tasks.utils import get_checkpoint_path

    log_root_path = Path("logs") / "rsl_rl" / agent_cfg.experiment_name
    load_run = args.load_run if args.load_run is not None else agent_cfg.load_run
    load_checkpoint = args.load_checkpoint if args.load_checkpoint is not None else agent_cfg.load_checkpoint
    return get_checkpoint_path(str(log_root_path.absolute()), load_run, load_checkpoint)


def _load_rsl_rl_policy(env: Any, args: argparse.Namespace) -> tuple[Any, Any, Any, str]:
    from rsl_rl.runners import DistillationRunner, OnPolicyRunner

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    agent_cfg = load_cfg_from_registry(args.task, args.agent)
    if args.device is not None:
        agent_cfg.device = args.device
    resume_path = _resolve_checkpoint(args, agent_cfg)

    wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(wrapped_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(wrapped_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    print(f"[INFO] Loading RSL-RL checkpoint: {resume_path}")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=wrapped_env.unwrapped.device)
    try:
        policy_model = runner.alg.policy
    except AttributeError:
        policy_model = runner.alg.actor_critic
    return wrapped_env, policy, policy_model, str(resume_path)


def run_real() -> None:
    import argparse

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Generic browser/debug snapshot viewer for Isaac Lab tasks.")
    parser.add_argument("--task", type=str, required=True, help="Name of the Isaac Lab task.")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
    parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
    parser.add_argument(
        "--viewer-mode",
        choices=("setup", "policy"),
        default="setup",
        help="Setup mode uses zero/random actions; policy mode loads an RSL-RL checkpoint.",
    )
    parser.add_argument(
        "--action-source", choices=("zero", "random"), default="zero", help="Action source while serving."
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="rsl_rl_cfg_entry_point",
        help="RL agent config entry point used when --viewer-mode=policy.",
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Exact RSL-RL checkpoint for policy playback.")
    parser.add_argument("--run-dir", type=str, default=None, help="Run directory; viewer loads the latest model_*.pt.")
    parser.add_argument("--load-run", type=str, default=None, help="Run name for Isaac Lab checkpoint lookup.")
    parser.add_argument("--load-checkpoint", type=str, default=None, help="Checkpoint name for Isaac Lab lookup.")
    parser.add_argument("--warmup-steps", type=int, default=1, help="Steps to run before the first snapshot.")
    parser.add_argument("--dump-json", type=Path, default=None, help="Write one snapshot JSON file and exit.")
    parser.add_argument("--serve", action="store_true", help="Serve the browser viewer.")
    parser.add_argument("--host", default="0.0.0.0", help="HTTP host for --serve.")
    parser.add_argument("--port", type=int, default=8080, help="HTTP port for --serve.")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import gymnasium as gym

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    import object_in_bowl.tasks  # noqa: F401

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs, use_fabric=not args.disable_fabric)
    env = gym.make(args.task, cfg=env_cfg)
    runtime_env = env
    policy = None
    policy_model = None
    checkpoint_path = None
    action_source = args.action_source
    if args.viewer_mode == "policy":
        runtime_env, policy, policy_model, checkpoint_path = _load_rsl_rl_policy(env, args)
        action_source = "policy"

    runtime = DebugRuntime(
        env=runtime_env,
        task=args.task,
        action_source=action_source,
        viewer_mode=args.viewer_mode,
        checkpoint_path=checkpoint_path,
        policy=policy,
        policy_model=policy_model,
    )
    runtime.step(max(0, args.warmup_steps))

    try:
        if args.dump_json:
            args.dump_json.parent.mkdir(parents=True, exist_ok=True)
            args.dump_json.write_text(json.dumps(runtime.snapshot(), indent=2, cls=SnapshotEncoder))
            print(f"[INFO] Wrote snapshot: {args.dump_json}")
        elif args.serve:
            serve(runtime, args.host, args.port)
        else:
            print(json.dumps(runtime.snapshot(), indent=2, cls=SnapshotEncoder))
    finally:
        runtime_env.close()
        simulation_app.close()


def main() -> None:
    if "--mock" in sys.argv:
        args = parse_mock_args()
        if not args.serve and args.dump_json is None:
            args.serve = True
        run_mock(args)
        return
    run_real()


if __name__ == "__main__":
    main()
