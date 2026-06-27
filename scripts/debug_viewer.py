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
      <div class="panel" id="statusPanel"></div>
      <div class="panel" id="metricsPanel"></div>
      <div class="panel" id="rewardPanel"></div>
      <div class="panel" id="terminationPanel"></div>
      <div class="panel" id="assetPanel"></div>
      <div class="panel" id="overlayPanel"></div>
      <div class="panel" id="ioPanel"></div>
    </aside>
  </main>
  <script>
    const topCanvas = document.getElementById("topCanvas");
    const heightCanvas = document.getElementById("heightCanvas");
    const statusPanel = document.getElementById("statusPanel");
    const metricsPanel = document.getElementById("metricsPanel");
    const rewardPanel = document.getElementById("rewardPanel");
    const terminationPanel = document.getElementById("terminationPanel");
    const assetPanel = document.getElementById("assetPanel");
    const overlayPanel = document.getElementById("overlayPanel");
    const ioPanel = document.getElementById("ioPanel");
    const playPause = document.getElementById("playPause");
    const stepButton = document.getElementById("step");
    const resetButton = document.getElementById("reset");
    const speedSelect = document.getElementById("speed");

    let latest = null;
    let paused = false;

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
        const color = item.kind === "articulation" ? "#45c486" : item.kind === "frame" ? "#f2b84b" : item.kind === "rigid_object" ? "#ef6a6a" : "#4aa3ff";
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
        const color = item.kind === "articulation" ? "#45c486" : item.kind === "frame" ? "#f2b84b" : item.kind === "rigid_object" ? "#ef6a6a" : "#4aa3ff";
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
      statusPanel.innerHTML = `<h2>Status</h2><dl class="kv">
        <dt>Task</dt><dd>${snapshot.task || "-"}</dd>
        <dt>Step</dt><dd>${snapshot.stepCount ?? "-"}</dd>
        <dt>Env index</dt><dd>${snapshot.envIndex ?? 0}</dd>
        <dt>Mode</dt><dd>${snapshot.mode || "-"}</dd>
        <dt>Updated</dt><dd>${new Date(snapshot.timestamp * 1000).toLocaleTimeString()}</dd>
      </dl>`;

      const metrics = snapshot.metrics || {};
      metricsPanel.innerHTML = `<h2>Key Signals</h2><dl class="kv">${Object.entries(metrics).map(([k, v]) => `<dt>${k}</dt><dd>${fmt(v)}</dd>`).join("") || "<dt>none</dt><dd>-</dd>"}</dl>`;

      rewardPanel.innerHTML = `<h2>Rewards</h2>` + table(
        [{ label: "Term" }, { label: "Weight", num: true }, { label: "Current", num: true }, { label: "Source" }],
        (snapshot.rewards || []).map(r => [
          `<span class="name">${r.name}</span>`,
          fmt(r.weight),
          fmt(r.currentRewardValue),
          `<span class="muted">${r.source || ""}</span>`,
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
        renderPanels(latest);
        drawTop(latest);
        drawHeight(latest);
      } catch (error) {
        statusPanel.innerHTML = `<h2>Status</h2><div class="error">${error.message}</div>`;
      }
    }

    playPause.addEventListener("click", () => control(paused ? "play" : "pause").then(loadSnapshot));
    stepButton.addEventListener("click", () => control("step").then(loadSnapshot));
    resetButton.addEventListener("click", () => control("reset").then(loadSnapshot));
    speedSelect.addEventListener("change", () => control("speed", { value: speedSelect.value }));
    window.addEventListener("resize", () => { drawTop(latest); drawHeight(latest); });

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
        rows.append(
            {
                "name": name,
                "weight": _to_jsonable(getattr(cfg, "weight", None), env_index),
                "params": _to_jsonable(getattr(cfg, "params", {}) or {}, env_index),
                "currentRewardValue": _to_scalar(current.get(name), env_index),
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


def build_snapshot(env: Any, task: str, env_index: int, step_count: int, mode: str, paused: bool) -> dict[str, Any]:
    """Build a normalized snapshot from a manager-based Isaac Lab environment."""
    unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
    snapshot = {
        "schemaVersion": 1,
        "task": task,
        "mode": mode,
        "envIndex": env_index,
        "stepCount": step_count,
        "paused": paused,
        "timestamp": time.time(),
        "scene": {
            "assets": _collect_assets(unwrapped.scene, env_index),
            "frames": _collect_frames(unwrapped.scene, env_index),
        },
        "rewards": _collect_rewards(unwrapped, env_index),
        "terminations": _collect_terminations(unwrapped, env_index),
        "observations": _collect_observations(unwrapped, env_index),
        "actions": _collect_actions(unwrapped, env_index),
    }
    snapshot["overlays"] = _collect_overlays(snapshot)
    snapshot["metrics"] = _collect_metrics(snapshot)
    return snapshot


class DebugRuntime:
    """Small state holder for serving snapshots and stepping one env."""

    def __init__(self, env: Any | None, task: str, action_source: str, mock: bool = False):
        self.env = env
        self.task = task
        self.action_source = action_source
        self.mock = mock
        self.paused = False
        self.step_count = 0
        self.steps_per_tick = 1
        if self.env is not None:
            self.env.reset()

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
                self.env.step(self.make_action())
                self.step_count += 1

    def reset(self) -> None:
        if self.mock:
            self.step_count = 0
            return
        if self.env is not None:
            self.env.reset()
        self.step_count = 0

    def snapshot(self) -> dict[str, Any]:
        if self.mock:
            return build_mock_snapshot(self.step_count, self.paused)
        return build_snapshot(self.env, self.task, 0, self.step_count, self.action_source, self.paused)


def build_mock_snapshot(step_count: int, paused: bool) -> dict[str, Any]:
    """Build a fake snapshot for browser verification without Isaac Lab installed."""
    phase = step_count / 18.0
    cube = [0.5 + 0.03 * math.sin(phase), 0.02 * math.cos(phase), 0.055 + 0.015 * max(0, math.sin(phase))]
    ee = [cube[0] - 0.05 * math.cos(phase), cube[1] + 0.04 * math.sin(phase), cube[2] + 0.11]
    target = [0.7, 0.2, 0.049]
    return {
        "schemaVersion": 1,
        "task": "Mock-Manager-Based-Task-v0",
        "mode": "mock",
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
            {"name": "reaching_object", "weight": 1.0, "currentRewardValue": 0.6, "params": {"std": 0.1}},
            {
                "name": "lifting_object",
                "weight": 15.0,
                "currentRewardValue": 0.0,
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
        "observations": [{"group": "policy", "name": "concatenated", "shape": "1x36", "summary": "[...mock values...]"}],
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
    } | {"metrics": {"objectZ": cube[2], "eeObjectDistance": _distance(ee, cube), "objectTargetDistance": _distance(cube, target)}}


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
    runtime = DebugRuntime(env=None, task="Mock-Manager-Based-Task-v0", action_source="mock", mock=True)
    if args.dump_json:
        args.dump_json.write_text(json.dumps(runtime.snapshot(), indent=2, cls=SnapshotEncoder))
        print(f"[INFO] Wrote mock snapshot: {args.dump_json}")
        return
    serve(runtime, args.host, args.port)


def parse_mock_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the Isaac Lab debug viewer with mock data.")
    parser.add_argument("--mock", action="store_true", help="Run without Isaac Lab using fake data.")
    parser.add_argument("--serve", action="store_true", help="Serve the browser viewer.")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP host.")
    parser.add_argument("--port", type=int, default=8080, help="HTTP port.")
    parser.add_argument("--dump-json", type=Path, default=None, help="Write one snapshot JSON file and exit.")
    return parser.parse_args()


def run_real() -> None:
    import argparse

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Generic browser/debug snapshot viewer for Isaac Lab tasks.")
    parser.add_argument("--task", type=str, required=True, help="Name of the Isaac Lab task.")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
    parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
    parser.add_argument("--action-source", choices=("zero", "random"), default="zero", help="Action source while serving.")
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
    runtime = DebugRuntime(env=env, task=args.task, action_source=args.action_source)
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
        env.close()
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
