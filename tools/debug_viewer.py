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

from debug_adapters import resolve_task_adapter


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
    .summary-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 8px;
    }
    .summary-card {
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 8px;
      background: var(--panel-2);
    }
    .summary-label {
      color: var(--muted);
      font-size: 12px;
    }
    .summary-value {
      margin-top: 3px;
      font-size: 18px;
      font-weight: 680;
      font-variant-numeric: tabular-nums;
    }
    .summary-source {
      margin-top: 2px;
      color: var(--muted);
      font-size: 11px;
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
    .stage-margin {
      margin-top: 6px;
      font-size: 12px;
      font-variant-numeric: tabular-nums;
    }
    .stage-margin.good { color: var(--good); }
    .stage-margin.bad { color: var(--bad); }
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
    .health.good { color: var(--good); }
    .health.warn { color: var(--warn); }
    .health.bad { color: var(--bad); }
    .subhead {
      margin: 12px 0 6px;
      font-size: 12px;
      font-weight: 650;
      color: var(--muted);
      text-transform: uppercase;
    }
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
      <div class="panel" id="configPanel"></div>
      <div class="panel" id="diagnosticsPanel"></div>
      <div class="panel" id="timelinePanel"></div>
      <div class="panel" id="metricsPanel"></div>
      <div class="panel" id="rewardPanel"></div>
      <div class="panel" id="commandPanel"></div>
      <div class="panel" id="terminationPanel"></div>
      <div class="panel" id="episodeLogPanel"></div>
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
    const configPanel = document.getElementById("configPanel");
    const diagnosticsPanel = document.getElementById("diagnosticsPanel");
    const timelinePanel = document.getElementById("timelinePanel");
    const metricsPanel = document.getElementById("metricsPanel");
    const rewardPanel = document.getElementById("rewardPanel");
    const commandPanel = document.getElementById("commandPanel");
    const terminationPanel = document.getElementById("terminationPanel");
    const episodeLogPanel = document.getElementById("episodeLogPanel");
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

    function healthPill(status) {
      const value = status || "unknown";
      const cls = value === "ok" ? "good" : value === "warn" ? "warn" : value === "bad" ? "bad" : "inactive";
      return `<span class="pill health ${cls}">${value}</span>`;
    }

    function summaryCard(label, value, source) {
      return `<div class="summary-card">
        <div class="summary-label">${label}</div>
        <div class="summary-value">${value}</div>
        ${source ? `<div class="summary-source">${source}</div>` : ""}
      </div>`;
    }

    function formatStats(stats) {
      if (!stats) return "-";
      const range = stats.min !== undefined && stats.max !== undefined ? `${fmt(stats.min)}..${fmt(stats.max)}` : "-";
      const mean = stats.mean !== undefined ? `mean ${fmt(stats.mean)}` : "";
      const flags = [];
      if (stats.nanCount) flags.push(`nan ${stats.nanCount}`);
      if (stats.infCount) flags.push(`inf ${stats.infCount}`);
      if (stats.nearLimitRate !== undefined) flags.push(`limit ${fmt(stats.nearLimitRate, 2)}`);
      return [range, mean, flags.join(", ")].filter(Boolean).join("<br>");
    }

    function formatSample(row) {
      if (row.sample !== undefined) return fmt(row.sample);
      return row.summary || "-";
    }

    function shortLabel(label) {
      return String(label || "")
        .replace(/^object_in_bowl /, "bowl ")
        .replace(/^lifting_object /, "lift ")
        .replace(/^object_dropping /, "drop ")
        .replace("minimum_height", "min")
        .replace("minimal_height", "min")
        .replace("maximum_height", "max");
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
      if (previous && snapshot.stepCount === previous.step) return;
      const signals = snapshot.taskDiagnostics?.signals || snapshot.bottlenecks?.signals || {};
      history.push({
        step: snapshot.stepCount || 0,
        objectZ: Number(signals.objectZ ?? snapshot.metrics?.objectZ ?? 0),
        eeDistance: Number(signals.eeObjectDistance ?? 0),
        gripperCommand: Number(signals.gripperCommand ?? 0),
        reward: Number(
          (snapshot.rewards || []).reduce(
            (sum, row) => sum + (Number(row.weightedPreDtValue) || 0),
            0,
          ),
        ),
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
        if (p) positions.push({ name: asset.name, label: asset.label || asset.name, kind: asset.kind, p });
      }
      for (const frame of snapshot.scene?.frames || []) {
        const p = pos2(frame);
        if (p) positions.push({ name: frame.name, label: frame.label || frame.name, kind: "frame", p });
      }
      for (const overlay of snapshot.overlays || []) {
        if (overlay.position) positions.push({ name: overlay.type, label: overlay.label, kind: "overlay", p: overlay.position });
        if (overlay.z !== undefined) positions.push({ name: overlay.type, label: overlay.label, kind: "height", p: [0, 0, overlay.z] });
      }
      return positions;
    }

    function isHelperAsset(item) {
      if (item.kind !== "static_asset") return false;
      return item.name === "ground" || item.name === "dome_light" || String(item.name || "").includes("collision");
    }

    function shouldDrawMarker(item) {
      return item.kind !== "overlay" && item.kind !== "height" && !isHelperAsset(item);
    }

    function canvasLabel(item) {
      if (item.name === "object") return "cube";
      if (String(item.name || "").startsWith("ee_frame")) return "ee";
      if (item.kind !== "static_asset") return item.label;
      if (item.name === "bowl" || item.name === "table") return item.name;
      return "";
    }

    function drawCanvasLabel(ctx, text, x, y, labelBoxes) {
      if (!text) return;
      ctx.font = "12px system-ui";
      const width = ctx.measureText(text).width + 8;
      const height = 16;
      const candidates = [
        [x + 9, y - 12],
        [x + 9, y + 8],
        [x - width - 9, y - 12],
        [x - width - 9, y + 8],
      ];
      let box = null;
      for (const [left, top] of candidates) {
        const candidate = { left, top, right: left + width, bottom: top + height };
        const overlaps = labelBoxes.some(existing =>
          candidate.left < existing.right && candidate.right > existing.left
          && candidate.top < existing.bottom && candidate.bottom > existing.top
        );
        if (!overlaps) {
          box = candidate;
          break;
        }
      }
      if (!box) return;
      labelBoxes.push(box);
      ctx.fillStyle = "rgba(17, 19, 24, 0.82)";
      ctx.fillRect(box.left, box.top, width, height);
      ctx.fillStyle = "#edf0f5";
      ctx.fillText(text, box.left + 4, box.top + 12);
    }

    function drawLegendBox(ctx, title, rows, x, y) {
      if (!rows.length) return;
      ctx.font = "12px system-ui";
      const width = Math.min(
        320,
        Math.max(ctx.measureText(title).width + 26, ...rows.map(row => ctx.measureText(row.label).width + 42)),
      );
      const height = 26 + rows.length * 18;
      ctx.fillStyle = "rgba(17, 19, 24, 0.88)";
      ctx.strokeStyle = "rgba(52, 58, 70, 0.9)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.roundRect(x, y, width, height, 6);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = "#a6adba";
      ctx.fillText(title, x + 10, y + 16);
      rows.forEach((row, index) => {
        const rowY = y + 34 + index * 18;
        ctx.fillStyle = row.color || "#edf0f5";
        ctx.fillRect(x + 10, rowY - 9, 9, 9);
        ctx.fillStyle = "#edf0f5";
        ctx.fillText(row.label, x + 26, rowY);
      });
    }

    function bounds(snapshot) {
      const positions = allPositions(snapshot).filter(item =>
        item.kind === "overlay" || item.kind === "height" || shouldDrawMarker(item)
      );
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

      const drawnTopOverlays = new Set();
      const topLabelBoxes = [];
      for (const overlay of snapshot.overlays || []) {
        if (overlay.type === "target_radius" && overlay.position && overlay.radius !== undefined) {
          const key = `${overlay.position.join(",")}:${overlay.radius}`;
          if (drawnTopOverlays.has(key)) continue;
          drawnTopOverlays.add(key);
          const [cx, cy] = map(overlay.position[0], overlay.position[1]);
          ctx.strokeStyle = overlay.color || "#4aa3ff";
          ctx.fillStyle = "rgba(74, 163, 255, 0.08)";
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(cx, cy, overlay.radius * scale, 0, Math.PI * 2);
          ctx.fill();
          ctx.stroke();
          ctx.fillStyle = overlay.color || "#4aa3ff";
          ctx.beginPath();
          ctx.arc(cx, cy, 5, 0, Math.PI * 2);
          ctx.fill();
          drawCanvasLabel(ctx, shortLabel(overlay.label || "target"), cx, cy, topLabelBoxes);
        }
      }

      const positions = allPositions(snapshot);
      for (const item of positions) {
        if (!shouldDrawMarker(item)) continue;
        const [x, y] = map(item.p[0], item.p[1]);
        const color = item.kind === "articulation" ? "#45c486" : item.kind === "frame" ? "#f2b84b" : item.kind === "rigid_object" ? "#ef6a6a" : item.kind === "static_asset" ? "#9b8cff" : "#4aa3ff";
        ctx.fillStyle = color;
        ctx.strokeStyle = "#0f1116";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(x, y, item.kind === "frame" ? 5 : 7, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
        drawCanvasLabel(ctx, canvasLabel(item), x, y, topLabelBoxes);
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

      const heightLegend = [];
      const drawnHeightLines = new Set();
      for (const overlay of snapshot.overlays || []) {
        if (overlay.type === "height_plane" && overlay.z !== undefined) {
          const key = `${overlay.label}:${overlay.z}`;
          if (drawnHeightLines.has(key)) continue;
          drawnHeightLines.add(key);
          const [, y] = map(0, overlay.z);
          ctx.strokeStyle = overlay.color || "#f2b84b";
          ctx.lineWidth = 2;
          ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke();
          heightLegend.push({
            label: `${shortLabel(overlay.label || "height")} z=${fmt(overlay.z)}`,
            color: overlay.color || "#f2b84b",
            z: Number(overlay.z),
          });
        }
      }
      heightLegend.sort((a, b) => b.z - a.z);
      const visibleHeightRows = heightLegend.slice(0, 6);
      if (heightLegend.length > visibleHeightRows.length) {
        visibleHeightRows.push({ label: `+${heightLegend.length - visibleHeightRows.length} more`, color: "#a6adba" });
      }
      drawLegendBox(ctx, "Height thresholds", visibleHeightRows, 12, 42);

      const heightLabelBoxes = [];
      for (const item of allPositions(snapshot)) {
        if (!shouldDrawMarker(item)) continue;
        const [x, y] = map(item.p[0], item.p[2]);
        const color = item.kind === "articulation" ? "#45c486" : item.kind === "frame" ? "#f2b84b" : item.kind === "rigid_object" ? "#ef6a6a" : item.kind === "static_asset" ? "#9b8cff" : "#4aa3ff";
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(x, y, item.kind === "frame" ? 5 : 7, 0, Math.PI * 2);
        ctx.fill();
        drawCanvasLabel(ctx, canvasLabel(item), x, y, heightLabelBoxes);
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

      const diagnostics = snapshot.taskDiagnostics || {};
      const stages = diagnostics.stages || snapshot.bottlenecks?.stages || [];
      const probes = diagnostics.rewardProbes || snapshot.rewardProbes || [];
      const adapterName = diagnostics.displayName || diagnostics.adapterName || "No adapter active";
      const adapterDescription = diagnostics.description || "No task-specific diagnostics are available for this task.";
      diagnosticsPanel.innerHTML = `<h2>Task Diagnostics</h2><div class="mode-card" style="margin-bottom:10px">
        <div class="mode-title">${adapterName}</div>
        <div class="mode-help">${adapterDescription}</div>
      </div><div class="stage-grid">${stages.map(stage => {
        const stateClass = stage.active === true ? "good" : stage.active === false ? "bad" : "unknown";
        const margin = Number(stage.margin);
        const marginClass = Number.isFinite(margin) && margin >= 0 ? "good" : "bad";
        const marginText = Number.isFinite(margin) ? `<div class="stage-margin ${marginClass}">margin ${margin >= 0 ? "+" : ""}${fmt(margin)}${stage.marginUnit || ""}</div>` : "";
        return `<div class="stage ${stateClass}">
          <div class="stage-name"><span>${stage.label}</span>${statePill(stage.active)}</div>
          <div class="stage-detail">${stage.detail || ""}</div>
          ${marginText}
        </div>`;
      }).join("") || `<div class="muted">No adapter stages available.</div>`}</div>
      <h2 style="margin-top:16px">Episode Signals</h2>
      <dl class="kv">${Object.entries(diagnostics.episodeSignals || snapshot.bottlenecks?.episodeSignals || {}).map(([k, v]) => `<dt>${k}</dt><dd>${fmt(v)}</dd>`).join("") || "<dt>none</dt><dd>-</dd>"}</dl>
      <h2 style="margin-top:16px">Candidate Reward Probes</h2>
      <div class="mode-help" style="margin:-4px 0 8px">Inactive diagnostics only. These are not active rewards unless they also appear in the Rewards table.</div>${table(
        [
          { label: "Candidate probe" },
          { label: "Current value", num: true },
          { label: "Would fire now", num: true },
          { label: "Meaning" },
        ],
        probes.map(p => [
          `<span class="name">${p.label || p.name}</span><br><span class="muted">${p.statusLabel || "candidate"}</span>`,
          fmt(p.currentValue),
          statePill(p.wouldFire),
          `<span class="muted">${p.detail || ""}</span>`,
        ])
      )}`;

      timelinePanel.innerHTML = `<h2>Live Timeline</h2>
        <canvas id="timelineCanvas" class="timeline"></canvas>
        <div class="legend">
          <span style="--legend-color:#45c486">cube z</span>
          <span style="--legend-color:#f2b84b">hand-cube distance</span>
          <span style="--legend-color:#4aa3ff">gripper command</span>
          <span style="--legend-color:#ef6a6a">reward pre-dt</span>
        </div>`;

      const metrics = snapshot.metrics || {};
      metricsPanel.innerHTML = `<h2>Key Signals</h2><dl class="kv">${Object.entries(metrics).map(([k, v]) => `<dt>${k}</dt><dd>${fmt(v)}</dd>`).join("") || "<dt>none</dt><dd>-</dd>"}</dl>`;

      const cfg = snapshot.taskConfig || {};
      configPanel.innerHTML = `<h2>Task Config</h2><div class="summary-grid">${
        [
          ["Step dt", cfg.stepDt !== undefined ? `${fmt(cfg.stepDt)}s` : "-", "sim.dt * decimation"],
          ["Sim dt", cfg.simDt !== undefined ? `${fmt(cfg.simDt)}s` : "-", "env.cfg.sim.dt"],
          ["Decimation", cfg.decimation ?? "-", "env.cfg.decimation"],
          ["Episode", cfg.episodeLengthS !== undefined ? `${fmt(cfg.episodeLengthS)}s` : "-", "env.cfg.episode_length_s"],
          ["Num envs", cfg.numEnvs ?? "-", "env.cfg.scene.num_envs"],
          ["Spacing", cfg.envSpacing !== undefined ? `${fmt(cfg.envSpacing)}m` : "-", "env.cfg.scene.env_spacing"],
        ].map(([label, value, source]) => summaryCard(label, value, source)).join("")
      }</div>`;

      const rewardSummary = snapshot.rewardSummary || {};
      rewardPanel.innerHTML = `<h2>Rewards</h2><div class="summary-grid">${
        summaryCard("Total pre-dt", fmt(rewardSummary.totalWeightedPreDt), "sum weighted terms")
        + summaryCard("Actual step", fmt(rewardSummary.totalStepContribution), `pre-dt * step dt ${fmt(rewardSummary.stepDt)}s`)
        + summaryCard("Firing terms", `${rewardSummary.firingCount ?? 0}/${rewardSummary.termCount ?? 0}`, "non-zero terms")
      }</div>` + table(
        [
          { label: "Term" },
          { label: "Weighted pre-dt", num: true },
          { label: "Per step", num: true },
          { label: "Weight", num: true },
          { label: "Firing", num: true },
        ],
        (snapshot.rewards || []).map(r => [
          `<span class="name">${r.name}</span>`,
          fmt(r.weightedPreDtValue),
          fmt(r.stepContribution),
          fmt(r.weight),
          statePill(Boolean(r.isFiring)),
        ])
      ) + `<div class="mode-help" style="margin-top:8px">Isaac's active reward value is already multiplied by the configured weight. The actual per-step contribution is this value times the environment step time.</div>`;

      commandPanel.innerHTML = `<h2>Commands</h2>` + table(
        [{ label: "Name" }, { label: "Shape" }, { label: "Current" }, { label: "Config" }],
        (snapshot.commands || []).map(c => [
          `<span class="name">${c.name}</span>`,
          c.shape || "-",
          `<span class="muted">${formatSample(c)}</span>`,
          `<span class="muted">${c.paramSummary || ""}</span>`,
        ])
      ) + `<div class="subhead">Reset Events</div>` + table(
        [{ label: "Name" }, { label: "Mode" }, { label: "Config" }],
        (snapshot.events || []).map(e => [
          `<span class="name">${e.name}</span>`,
          e.mode || "-",
          `<span class="muted">${e.paramSummary || ""}</span>`,
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

      const logGroups = snapshot.extrasLog?.groups || [];
      episodeLogPanel.innerHTML = `<h2>Episode Logs</h2>${logGroups.length ? logGroups.map(group => (
        `<div class="subhead">${group.name}</div>` + table(
          [{ label: "Metric" }, { label: "Value", num: true }],
          group.rows.map(row => [`<span class="name">${row.name}</span>`, fmt(row.value)])
        )
      )).join("") : `<div class="muted">No episode logs have been emitted yet.</div>`}`;

      const obsRows = (snapshot.observations || []).map(o => [`${o.group}.${o.name}`, o.shape || "-", healthPill(o.health?.status), `<span class="muted">${formatStats(o.stats)}</span>`]);
      const actionRows = (snapshot.actions || []).map(a => [a.name, a.shape || "-", healthPill(a.health?.status), `<span class="muted">${formatStats(a.stats)}</span>`]);
      ioPanel.innerHTML = `<h2>Observation Health</h2>${table([{ label: "Term" }, { label: "Shape" }, { label: "Health", num: true }, { label: "Stats" }], obsRows)}
        <h2 style="margin-top:16px">Action Health</h2>${table([{ label: "Term" }, { label: "Shape" }, { label: "Health", num: true }, { label: "Stats" }], actionRows)}`;

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
            return _finite_number(float(value.item()))
        flat = value.flatten()
        return [_finite_number(float(item)) for item in flat[:max_items].tolist()]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return _finite_number(value)
    if isinstance(value, str):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item, env_index, max_items) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item, env_index, max_items) for item in value[:max_items]]
    return str(value)


def _finite_number(value: int | float) -> int | float | None:
    """Return a JSON-safe number, or None for nan/inf."""
    if isinstance(value, bool):
        return value
    return value if math.isfinite(float(value)) else None


def _snapshot_json(payload: dict[str, Any], *, indent: int | None = None) -> str:
    """Dump snapshot JSON and fail if a non-finite value escaped sanitizing."""
    return json.dumps(payload, indent=indent, cls=SnapshotEncoder, allow_nan=False)


def _shape_summary(value: Any) -> str:
    shape = getattr(value, "shape", None)
    if shape is None:
        if isinstance(value, (list, tuple)):
            return str(len(value))
        return "-"
    return "x".join(str(item) for item in shape)


def _func_name(value: Any) -> str:
    """Return a compact function/class name for config summaries."""
    if value is None:
        return "-"
    return getattr(value, "__name__", value.__class__.__name__)


def _numeric_stats(value: Any, env_index: int | None = None) -> dict[str, Any] | None:
    """Summarize numeric tensors/arrays without dumping long vectors."""
    if not hasattr(value, "detach"):
        return None
    import torch

    tensor = value.detach().float().cpu()
    if tensor.numel() == 0:
        return {"min": None, "max": None, "mean": None, "nanCount": 0, "infCount": 0}
    if env_index is not None and tensor.ndim > 0 and tensor.shape[0] > env_index:
        sample_tensor = tensor[env_index]
    else:
        sample_tensor = tensor
    flat = sample_tensor.flatten()
    finite = flat[torch.isfinite(flat)]
    nan_count = int(torch.isnan(flat).sum().item())
    inf_count = int(torch.isinf(flat).sum().item())
    near_limit_count = int((flat.abs() >= 0.98).sum().item())
    stats: dict[str, Any] = {
        "nanCount": nan_count,
        "infCount": inf_count,
        "nearLimitRate": near_limit_count / max(1, int(flat.numel())),
        "sample": [float(item) for item in flat[:6].tolist()],
    }
    if finite.numel() == 0:
        stats.update({"min": None, "max": None, "mean": None})
    else:
        stats.update(
            {
                "min": float(finite.min().item()),
                "max": float(finite.max().item()),
                "mean": float(finite.mean().item()),
            }
        )
    return stats


def _health_from_stats(stats: dict[str, Any] | None, flag_limits: bool = False) -> dict[str, Any]:
    """Classify compact tensor stats for quick scan in the UI."""
    if stats is None:
        return {"status": "unknown", "detail": "no numeric stats"}
    if stats.get("nanCount") or stats.get("infCount"):
        return {"status": "bad", "detail": "contains nan/inf"}
    if flag_limits and (stats.get("nearLimitRate") or 0.0) > 0.8:
        return {"status": "warn", "detail": "many values near action limits"}
    return {"status": "ok", "detail": "finite"}


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


def _collect_task_config(env: Any) -> dict[str, Any]:
    """Collect generic environment timing and scene config."""
    cfg = getattr(env, "cfg", None)
    sim_cfg = getattr(cfg, "sim", None)
    scene_cfg = getattr(cfg, "scene", None)
    sim_dt = _number(getattr(sim_cfg, "dt", None))
    decimation = _number(getattr(cfg, "decimation", None))
    step_dt = sim_dt * decimation if sim_dt is not None and decimation is not None else None
    return {
        "simDt": sim_dt,
        "decimation": decimation,
        "stepDt": step_dt,
        "renderInterval": _to_jsonable(getattr(sim_cfg, "render_interval", None)),
        "episodeLengthS": _to_jsonable(getattr(cfg, "episode_length_s", None)),
        "numEnvs": _to_jsonable(getattr(scene_cfg, "num_envs", None)),
        "envSpacing": _to_jsonable(getattr(scene_cfg, "env_spacing", None)),
        "source": "env.cfg",
    }


def _collect_commands(env: Any, env_index: int) -> list[dict[str, Any]]:
    """Collect command-manager values and command config summaries."""
    manager = getattr(env, "command_manager", None)
    if manager is None:
        return []
    cfgs = _manager_term_cfgs(manager)
    names = list(getattr(manager, "_term_names", []) or cfgs.keys())
    rows = []
    for name in names:
        value = None
        try:
            value = manager.get_command(name)
        except Exception:
            pass
        cfg = cfgs.get(name)
        params = getattr(cfg, "params", {}) or {}
        ranges = getattr(cfg, "ranges", None)
        if ranges is not None:
            params = dict(params)
            params["ranges"] = vars(ranges)
        resampling = getattr(cfg, "resampling_time_range", None)
        if resampling is not None:
            params = dict(params)
            params["resampling"] = resampling
        stats = _numeric_stats(value, env_index)
        rows.append(
            {
                "name": name,
                "shape": _shape_summary(value),
                "summary": str(_to_jsonable(value, env_index, max_items=6)),
                "stats": stats,
                "health": _health_from_stats(stats),
                "params": _to_jsonable(params, env_index),
                "paramSummary": _param_summary(params, env_index),
                "source": "command_manager",
            }
        )
    return rows


def _collect_event_cfgs(env: Any, env_index: int) -> list[dict[str, Any]]:
    """Collect reset/event config so sampled starting conditions are visible."""
    events_cfg = getattr(getattr(env, "cfg", None), "events", None)
    rows = []
    for name, cfg in vars(events_cfg).items() if events_cfg is not None else []:
        if name.startswith("_"):
            continue
        params = getattr(cfg, "params", {}) or {}
        rows.append(
            {
                "name": name,
                "mode": getattr(cfg, "mode", None),
                "func": _func_name(getattr(cfg, "func", None)),
                "params": _to_jsonable(params, env_index),
                "paramSummary": _param_summary(params, env_index),
                "source": "env.cfg.events",
            }
        )
    return rows


def _collect_extras_log(env: Any, env_index: int) -> dict[str, Any]:
    """Collect latest episode log scalars grouped by prefix."""
    extras = getattr(env, "extras", {}) or {}
    log = extras.get("log", {}) if isinstance(extras, dict) else {}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for key, value in log.items():
        group, separator, metric = str(key).partition("/")
        if not separator:
            group = "log"
            metric = str(key)
        grouped.setdefault(group, []).append({"name": metric, "value": _to_scalar(value, env_index)})
    return {
        "groups": [
            {"name": name, "rows": sorted(rows, key=lambda row: row["name"])}
            for name, rows in sorted(grouped.items())
        ],
        "source": "env.extras.log",
    }


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


def _collect_rewards(env: Any, env_index: int, step_dt: float | None = None) -> list[dict[str, Any]]:
    manager = getattr(env, "reward_manager", None)
    if manager is None:
        return []
    cfgs = _manager_term_cfgs(manager)
    current = _manager_current_terms(manager, env_index)
    rows = []
    for name, cfg in cfgs.items():
        weight = _to_jsonable(getattr(cfg, "weight", None), env_index)
        weighted_pre_dt_value = _to_scalar(current.get(name), env_index)
        weighted_number = _number(weighted_pre_dt_value)
        weight_number = _number(weight)
        raw_approx = (
            weighted_number / weight_number
            if weighted_number is not None and weight_number not in (None, 0.0)
            else None
        )
        step_contribution = weighted_number * step_dt if weighted_number is not None and step_dt is not None else None
        rows.append(
            {
                "name": name,
                "weight": weight,
                "params": _to_jsonable(getattr(cfg, "params", {}) or {}, env_index),
                "weightedPreDtValue": weighted_pre_dt_value,
                "rawApprox": raw_approx,
                "stepContribution": step_contribution,
                "isFiring": abs(weighted_number or 0.0) > 1.0e-6,
                "source": "reward_manager",
            }
        )
    return rows


def _collect_reward_summary(rewards: list[dict[str, Any]], step_dt: float | None) -> dict[str, Any]:
    total_pre_dt = sum(_number(reward.get("weightedPreDtValue")) or 0.0 for reward in rewards)
    return {
        "totalWeightedPreDt": total_pre_dt,
        "totalStepContribution": total_pre_dt * step_dt if step_dt is not None else None,
        "stepDt": step_dt,
        "firingCount": sum(1 for reward in rewards if reward.get("isFiring")),
        "termCount": len(rewards),
    }


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
            stats = _numeric_stats(value, env_index)
            rows.append(
                {
                    "group": group_name if separator else "observation",
                    "name": term_name if separator else group_name,
                    "shape": _shape_summary(value),
                    "summary": str(value_summary),
                    "sample": _to_jsonable(value, env_index, max_items=6),
                    "stats": stats,
                    "health": _health_from_stats(stats),
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
            stats = _numeric_stats(value, env_index)
            rows.append(
                {
                    "group": str(group_name),
                    "name": str(name),
                    "shape": _shape_summary(value),
                    "summary": str(_to_jsonable(value, env_index, max_items=6)),
                    "sample": _to_jsonable(value, env_index, max_items=6),
                    "stats": stats,
                    "health": _health_from_stats(stats),
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
        stats = _numeric_stats(raw_actions, env_index)
        rows.append(
            {
                "name": name,
                "shape": _shape_summary(raw_actions),
                "summary": str(_to_jsonable(raw_actions, env_index, max_items=6)),
                "sample": _to_jsonable(raw_actions, env_index, max_items=6),
                "stats": stats,
                "health": _health_from_stats(stats, flag_limits=True),
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


def _record_error(errors: list[dict[str, str]], source: str, exc: Exception) -> None:
    errors.append({"source": source, "error": f"{type(exc).__name__}: {exc}"})


def _collect_task_diagnostics(
    task: str, env: Any, snapshot: dict[str, Any], env_index: int, errors: list[dict[str, str]]
) -> dict[str, Any]:
    """Collect optional task-specific diagnostics through an adapter."""
    adapter = resolve_task_adapter(task)
    if adapter is None:
        return {
            "adapterName": None,
            "displayName": "No adapter active",
            "description": "No task-specific diagnostics are available for this task.",
            "stages": [],
            "signals": {},
            "episodeSignals": {},
            "rewardProbes": [],
        }
    try:
        return adapter.collect(env, snapshot, env_index, errors)
    except Exception as exc:
        _record_error(errors, f"{getattr(adapter, 'ADAPTER_NAME', 'task_adapter')}.collect", exc)
        return {
            "adapterName": getattr(adapter, "ADAPTER_NAME", None),
            "displayName": getattr(adapter, "DISPLAY_NAME", "Task adapter failed"),
            "description": f"Adapter failed: {type(exc).__name__}: {exc}",
            "stages": [],
            "signals": {},
            "episodeSignals": {},
            "rewardProbes": [],
        }


def build_snapshot(env: Any, task: str, env_index: int, step_count: int, mode: str, paused: bool) -> dict[str, Any]:
    """Build a normalized snapshot from a manager-based Isaac Lab environment."""
    unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
    diagnostic_errors: list[dict[str, str]] = []
    task_config = _collect_task_config(unwrapped)
    step_dt = _number(task_config.get("stepDt"))
    rewards = _collect_rewards(unwrapped, env_index, step_dt)
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
        "taskConfig": task_config,
        "rewards": rewards,
        "rewardSummary": _collect_reward_summary(rewards, step_dt),
        "terminations": _collect_terminations(unwrapped, env_index),
        "commands": _collect_commands(unwrapped, env_index),
        "events": _collect_event_cfgs(unwrapped, env_index),
        "extrasLog": _collect_extras_log(unwrapped, env_index),
        "observations": _collect_observations(unwrapped, env_index),
        "actions": _collect_actions(unwrapped, env_index),
    }
    snapshot["overlays"] = _collect_overlays(snapshot)
    snapshot["metrics"] = _collect_metrics(snapshot)
    snapshot["taskDiagnostics"] = _collect_task_diagnostics(task, unwrapped, snapshot, env_index, diagnostic_errors)
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


class StaticSnapshotRuntime:
    """Serve a saved snapshot JSON file so browser rendering can be debugged without Isaac."""

    def __init__(self, snapshot: dict[str, Any]):
        self._snapshot = snapshot
        self.paused = True
        self.steps_per_tick = 1

    def step(self, count: int = 1) -> None:
        return

    def reset(self) -> None:
        return

    def snapshot(self) -> dict[str, Any]:
        snapshot = dict(self._snapshot)
        snapshot["paused"] = self.paused
        snapshot["timestamp"] = time.time()
        return snapshot


def build_mock_snapshot(step_count: int, paused: bool, viewer_mode: str = "setup") -> dict[str, Any]:
    """Build a fake snapshot for browser verification without Isaac Lab installed."""
    phase = step_count / 18.0
    cube = [0.5 + 0.03 * math.sin(phase), 0.02 * math.cos(phase), 0.055 + 0.015 * max(0, math.sin(phase))]
    ee = [cube[0] - 0.05 * math.cos(phase), cube[1] + 0.04 * math.sin(phase), cube[2] + 0.11]
    target = [0.7, 0.2, 0.049]
    mock_gripper_command = -0.72 if viewer_mode == "policy" else 0.0
    step_dt = 0.02
    reaching_value = 0.6
    ee_distance = _distance(ee, cube)
    lift_progress = max(0.0, min(1.0, (cube[2] - 0.055) / 0.05))
    lift_gate_active = ee_distance < 0.08 and mock_gripper_command < 0.0
    lifting_value = 20.0 * (
        0.25 * lift_progress * float(lift_gate_active) + 0.75 * float(cube[2] > 0.105)
    )
    reward_total = reaching_value + lifting_value
    object_target_distance = _distance(cube, target)
    mock_task = "Mock-Object-In-Bowl-Task-v0"
    mock_adapter = resolve_task_adapter(mock_task)
    task_diagnostics = (
        mock_adapter.build_mock_diagnostics(cube, ee, target, mock_gripper_command, viewer_mode)
        if mock_adapter is not None and hasattr(mock_adapter, "build_mock_diagnostics")
        else {
            "adapterName": None,
            "displayName": "No adapter active",
            "description": "No task-specific diagnostics are available for this task.",
            "stages": [],
            "signals": {},
            "episodeSignals": {},
            "rewardProbes": [],
        }
    )
    return {
        "schemaVersion": 1,
        "task": mock_task,
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
                {"name": "table", "label": "table", "kind": "static_asset", "position": [0.35, 0.0, -0.05]},
                {"name": "ground", "label": "ground", "kind": "static_asset", "position": [-0.5, -0.35, -1.05]},
                {"name": "bowl", "label": "bowl", "kind": "static_asset", "position": [0.7, 0.2, 0.025]},
                {"name": "bowl_collision_base", "label": "bowl_collision_base", "kind": "static_asset", "position": [0.7, 0.2, 0.01]},
                {"name": "bowl_collision_front", "label": "bowl_collision_front", "kind": "static_asset", "position": [0.7, 0.31, 0.07]},
                {"name": "bowl_collision_back", "label": "bowl_collision_back", "kind": "static_asset", "position": [0.7, 0.09, 0.07]},
                {"name": "bowl_collision_left", "label": "bowl_collision_left", "kind": "static_asset", "position": [0.59, 0.2, 0.07]},
                {"name": "bowl_collision_right", "label": "bowl_collision_right", "kind": "static_asset", "position": [0.81, 0.2, 0.07]},
            ],
            "frames": [{"name": "ee_frame:0", "label": "ee_frame", "kind": "frame", "position": ee}],
        },
        "taskConfig": {
            "simDt": 0.01,
            "decimation": 2,
            "stepDt": step_dt,
            "renderInterval": 2,
            "episodeLengthS": 5.0,
            "numEnvs": 4096,
            "envSpacing": 2.5,
            "source": "mock env.cfg",
        },
        "rewards": [
            {
                "name": "reaching_object",
                "weight": 1.0,
                "weightedPreDtValue": reaching_value,
                "rawApprox": reaching_value,
                "stepContribution": reaching_value * step_dt,
                "isFiring": True,
                "params": {"std": 0.1},
            },
            {
                "name": "object_lift_progress",
                "weight": 20.0,
                "weightedPreDtValue": lifting_value,
                "rawApprox": 0.0,
                "stepContribution": lifting_value * step_dt,
                "isFiring": False,
                "params": {
                    "initial_height": 0.055,
                    "target_height": 0.10500000000000001,
                    "near_distance": 0.08,
                    "progress_scale": 0.25,
                    "completion_scale": 0.75,
                },
            },
        ],
        "rewardSummary": {
            "totalWeightedPreDt": reward_total,
            "totalStepContribution": reward_total * step_dt,
            "stepDt": step_dt,
            "firingCount": 1,
            "termCount": 2,
        },
        "terminations": [
            {
                "name": "object_in_bowl",
                "active": False,
                "params": {"target_position": target, "radius": 0.11, "min_height": 0.044, "max_height": 0.109},
                "paramSummary": "target_position=[0.7, 0.2, 0.049], radius=0.11",
            }
        ],
        "commands": [
            {
                "name": "object_pose",
                "shape": "1x7",
                "summary": "[0.7, 0.2, 0.31, 0, 0, 0]",
                "sample": [0.7, 0.2, 0.31, 0, 0, 0],
                "stats": {"min": 0.0, "max": 0.7, "mean": 0.201, "nanCount": 0, "infCount": 0},
                "health": {"status": "ok", "detail": "finite"},
                "params": {
                    "ranges": {"pos_x": [0.62, 0.78], "pos_y": [0.12, 0.28], "pos_z": [0.25, 0.45]},
                    "resampling": [5.0, 5.0],
                },
                "paramSummary": "ranges={'pos_x': [0.62, 0.78], 'pos_y': [0.12, 0.28], 'pos_z': [0.25, 0.45]}",
            }
        ],
        "events": [
            {
                "name": "reset_object_position",
                "mode": "reset",
                "params": {"pose_range": {"x": [-0.08, 0.08], "y": [-0.10, 0.10], "z": [0.0, 0.0]}},
                "paramSummary": "pose_range={'x': [-0.08, 0.08], 'y': [-0.1, 0.1], 'z': [0.0, 0.0]}",
            }
        ],
        "extrasLog": {
            "source": "mock env.extras.log",
            "groups": [
                {
                    "name": "Episode_Bowl",
                    "rows": [
                        {"name": "mean_xy_distance", "value": object_target_distance},
                        {"name": "success_rate", "value": 0.04},
                    ],
                },
                {
                    "name": "Episode_Diagnostics",
                    "rows": [
                        {"name": "max_lift_progress", "value": max(0.0, min(1.0, (cube[2] - 0.055) / 0.05))},
                        {"name": "min_ee_object_distance", "value": 0.045},
                    ],
                },
            ],
        },
        "observations": [
            {
                "group": "policy",
                "name": "concatenated",
                "shape": "1x36",
                "summary": "[...mock values...]",
                "sample": [0.0, 0.1, -0.2, 0.3],
                "stats": {"min": -0.2, "max": 0.3, "mean": 0.05, "nanCount": 0, "infCount": 0},
                "health": {"status": "ok", "detail": "finite"},
            }
        ],
        "actions": [
            {
                "name": "arm_action",
                "shape": "1x7",
                "summary": "[0, 0, 0, 0, 0, 0]",
                "sample": [0, 0, 0, 0, 0, 0],
                "stats": {"min": 0.0, "max": 0.0, "mean": 0.0, "nanCount": 0, "infCount": 0, "nearLimitRate": 0.0},
                "health": {"status": "ok", "detail": "finite"},
            },
            {
                "name": "gripper_action",
                "shape": "1x1",
                "summary": str([mock_gripper_command]),
                "sample": [mock_gripper_command],
                "stats": {
                    "min": mock_gripper_command,
                    "max": mock_gripper_command,
                    "mean": mock_gripper_command,
                    "nanCount": 0,
                    "infCount": 0,
                    "nearLimitRate": 0.0,
                },
                "health": {"status": "ok", "detail": "finite"},
            },
        ],
        "overlays": [
            {
                "type": "height_plane",
                "label": "object_in_bowl max_height",
                "z": 0.109,
                "source": "termination.object_in_bowl.params.max_height",
                "confidence": "medium",
                "color": "#ef6a6a",
            },
            {
                "type": "height_plane",
                "label": "object_lift_progress target_height",
                "z": 0.10500000000000001,
                "source": "reward.object_lift_progress.params.target_height",
                "confidence": "medium",
                "color": "#f2b84b",
            },
            {
                "type": "height_plane",
                "label": "object_in_bowl min_height",
                "z": 0.044,
                "source": "termination.object_in_bowl.params.min_height",
                "confidence": "medium",
                "color": "#ef6a6a",
            },
            {
                "type": "height_plane",
                "label": "object_dropping minimum_height",
                "z": -0.05,
                "source": "termination.object_dropping.params.minimum_height",
                "confidence": "medium",
                "color": "#ef6a6a",
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
            {
                "type": "target_radius",
                "label": "object_in_bowl radius",
                "position": target,
                "radius": 0.11,
                "source": "reward.object_in_bowl_success.params",
                "confidence": "medium",
                "color": "#4aa3ff",
            },
        ],
        "taskDiagnostics": task_diagnostics,
        "diagnosticErrors": [],
        "metrics": {
            "objectZ": cube[2],
            "eeObjectDistance": ee_distance,
            "objectTargetDistance": object_target_distance,
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
            self._send(_snapshot_json(payload).encode("utf-8"), "application/json")

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
    if args.snapshot_json:
        snapshot = json.loads(args.snapshot_json.read_text())
        snapshot.setdefault("viewerMode", args.viewer_mode)
        runtime = StaticSnapshotRuntime(snapshot)
    else:
        runtime = DebugRuntime(
            env=None,
            task="Mock-Object-In-Bowl-Task-v0",
            action_source="mock",
            mock=True,
            viewer_mode=args.viewer_mode,
        )
    if args.dump_json:
        args.dump_json.write_text(_snapshot_json(runtime.snapshot(), indent=2))
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
    parser.add_argument("--snapshot-json", type=Path, default=None, help="Serve a saved snapshot JSON file.")
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

    import object_in_bowl  # noqa: F401

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
            args.dump_json.write_text(_snapshot_json(runtime.snapshot(), indent=2))
            print(f"[INFO] Wrote snapshot: {args.dump_json}")
        elif args.serve:
            serve(runtime, args.host, args.port)
        else:
            print(_snapshot_json(runtime.snapshot(), indent=2))
    finally:
        runtime_env.close()
        simulation_app.close()


def main() -> None:
    has_snapshot_json = any(arg == "--snapshot-json" or arg.startswith("--snapshot-json=") for arg in sys.argv)
    if "--mock" in sys.argv or has_snapshot_json:
        args = parse_mock_args()
        if not args.serve and args.dump_json is None:
            args.serve = True
        run_mock(args)
        return
    run_real()


if __name__ == "__main__":
    main()
