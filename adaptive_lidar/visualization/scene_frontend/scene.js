/*
 * FOVEA scene renderer.
 *
 * This file draws what Python decided. It contains no perception, no
 * mapping and no classification: every position, size, colour and verdict
 * arrives in the render payload from `scene_data.build_scene_data`, and the
 * only thing computed here is what a camera and a mouse are for.
 *
 * AXES
 *   The project's world is x east, y north, z up. three.js is y up, so the
 *   mapping is (x, y, z) -> (x, z, -y). It is applied in exactly one place,
 *   `w2t`, and nothing else in this file touches the convention.
 *
 * CELLS
 *   One InstancedMesh for the whole grid. Tens of thousands of separate
 *   meshes would be tens of thousands of draw calls; one instanced box is
 *   one. Each instance is scaled to the cell's REAL footprint and to its
 *   real base-to-top extent, times the height scale Python chose.
 *
 * THE GRID GAP
 *   With "show adaptive grid" on, each cell is drawn at 88% of its true
 *   footprint, so the dark seam between neighbours reads as a boundary and
 *   the size difference between a 5 cm and an 80 cm cell is obvious. It is
 *   a drawing device; the footprint the inspector reports is the true one.
 *
 * SELECTION
 *   A click raycasts objects first and cells second, the same priority the
 *   Python hit test uses. What goes back to Streamlit is a world point and
 *   an optional track id — never a cell index — so the authoritative lookup
 *   still happens in `selection.py` against the real cached frame.
 */
/* global THREE, Streamlit */
"use strict";

// ─────────────────────────────────────────────────────────────
// state that must survive a re-render
// ─────────────────────────────────────────────────────────────
var S = {
  renderer: null, scene: null, camera: null,
  // The run's geometry, sent once and kept here. Streamlit re-sends an
  // element's args on every rerun, so the payload arrives only when it
  // changed and is null otherwise — this is the copy that is drawn from.
  run: null, view: null,
  flatMesh: null, flatCapacity: 0, flatIdx: null,
  boxMesh: null, boxCapacity: 0, boxIdx: null,
  cellXY: null, pointCapacity: 0,
  objectGroup: null, egoGroup: null, ringGroup: null, pointCloud: null,
  markerGroup: null,
  objects: [], labels: [], ringLabels: [],
  camMode: "orbit",
  // Orbit state, in world units. Kept here so a frame advance does not
  // move the camera: the payload changes every displayed frame and the
  // view must not jump with it.
  orbit: { yaw: -2.35, pitch: 0.62, dist: 68, target: [0, 0, 0] },
  drag: null, moved: 0, width: 0, height: 520,
  needsRender: true, hoverName: null,
};

var HOST = document.getElementById("wrap");
var LABELS = document.getElementById("labels");
var CAMS = document.getElementById("cams");
var HUD = document.getElementById("hud");
var HOVER = document.getElementById("hover");
var EMPTY = document.getElementById("empty");

var CAM_MODES = [
  ["orbit", "Orbit"], ["top", "Top"], ["chase", "Chase"],
  ["sensor", "Sensor"], ["reset", "Reset"],
];

/* Object geometry, cached by shape and size rounded to a centimetre.
 *
 * Objects are rebuilt every frame and EdgesGeometry is not cheap — it
 * walks the faces to find the silhouette. A tracked car is the same box
 * from frame to frame; only its matrix changes.
 */
var GEO_CACHE = {};

var STATE_RGB = [0x5b6470, 0xd9483f, 0xc98a1e];   // STATIC, MOVING, MOVABLE
var SELECT_RGB = 0xffd60a;

// ─────────────────────────────────────────────────────────────
// helpers
// ─────────────────────────────────────────────────────────────
function b64(s, Type) {
  var bin = atob(s), n = bin.length, buf = new Uint8Array(n);
  for (var i = 0; i < n; i++) buf[i] = bin.charCodeAt(i);
  return new Type(buf.buffer);
}

/** world (x east, y north, z up) -> three (y up). The only place. */
function w2t(x, y, z) { return new THREE.Vector3(x, z, -y); }

function col(rgb01) {
  return new THREE.Color(rgb01[0], rgb01[1], rgb01[2]);
}

// ─────────────────────────────────────────────────────────────
// one-time setup
// ─────────────────────────────────────────────────────────────
function init(width, height) {
  S.renderer = new THREE.WebGLRenderer({
    antialias: true, alpha: false,
    // An integrated GPU is the common case for a demo laptop, and this
    // scene is fill-light and instance-heavy: the discrete card buys
    // nothing and costs power.
    powerPreference: "low-power" });
  // Capped at 1: at 18,000 instances a 2x pixel ratio quadruples the
  // fragment work for detail nobody reads in a technical diagram.
  S.renderer.setPixelRatio(1);
  S.renderer.setSize(width, height);
  S.renderer.setClearColor(0x0f1116, 1);
  HOST.appendChild(S.renderer.domElement);

  S.scene = new THREE.Scene();
  S.scene.fog = new THREE.Fog(0x0f1116, 90, 190);

  S.camera = new THREE.PerspectiveCamera(48, width / height, 0.3, 900);

  // Flat, even lighting. Shadows and speculars would imply a surface
  // finish the data says nothing about.
  S.scene.add(new THREE.HemisphereLight(0xcfd8e6, 0x1a1d24, 1.05));
  var key = new THREE.DirectionalLight(0xffffff, 0.34);
  key.position.set(0.4, 1, 0.25);
  S.scene.add(key);

  S.objectGroup = new THREE.Group();
  S.egoGroup = new THREE.Group();
  S.ringGroup = new THREE.Group();
  S.markerGroup = new THREE.Group();
  S.scene.add(S.objectGroup, S.egoGroup, S.ringGroup, S.markerGroup);

  buildCamButtons();
  bindPointer();
  observeResize();
  guardContext();
  animate();
}

/* A lost context used to leave a white canvas and nothing else.
 *
 * The cause is fixed — the renderer no longer churns GPU buffers — but a
 * driver can still drop the context when the machine is under load, and
 * recovering is a dozen lines. Without this the tab is simply dead until
 * a reload.
 */
function guardContext() {
  var el = S.renderer.domElement;
  el.addEventListener("webglcontextlost", function (e) {
    e.preventDefault();
    EMPTY.textContent = "the graphics context was lost — restoring…";
    EMPTY.style.display = "flex";
  }, false);
  el.addEventListener("webglcontextrestored", function () {
    S.flatMesh = null; S.flatCapacity = 0;
    S.boxMesh = null; S.boxCapacity = 0;
    S.pointCloud = null; S.pointCapacity = 0;
    GEO_CACHE = {};
    EMPTY.style.display = "none";
    if (S.view) drawView(S.view);
    S.needsRender = true;
  }, false);
}

/** Follow the column width.
 *
 * The render payload only arrives when Streamlit reruns, so without this
 * the canvas keeps whatever size it had when the page last rendered: widen
 * or narrow the browser and the scene sits stretched in a corner of a
 * stale viewport.
 */
function observeResize() {
  var apply = function () {
    var w = HOST.clientWidth || S.width;
    var h = S.height;
    if (!w || (w === S.width)) return;
    S.width = w;
    S.renderer.setSize(w, h);
    S.camera.aspect = w / h;
    S.camera.updateProjectionMatrix();
    S.needsRender = true;
  };
  if (typeof ResizeObserver !== "undefined") {
    new ResizeObserver(apply).observe(HOST);
  }
  window.addEventListener("resize", apply);
}

function buildCamButtons() {
  CAM_MODES.forEach(function (m) {
    var b = document.createElement("button");
    b.textContent = m[1];
    b.dataset.mode = m[0];
    b.onclick = function () {
      if (m[0] === "reset") {
        S.orbit.yaw = -2.35; S.orbit.pitch = 0.62; S.orbit.dist = 68;
        S.camMode = "orbit";
      } else {
        S.camMode = m[0];
      }
      syncCamButtons();
      S.needsRender = true;
    };
    CAMS.appendChild(b);
  });
  syncCamButtons();
}

function syncCamButtons() {
  Array.prototype.forEach.call(CAMS.children, function (b) {
    b.className = b.dataset.mode === S.camMode ? "on" : "";
  });
}

// ─────────────────────────────────────────────────────────────
// camera
// ─────────────────────────────────────────────────────────────
function placeCamera(args) {
  var ego = args.ego, ex = ego.xy[0], ey = ego.xy[1], h = ego.heading;
  var t = S.orbit.target;

  if (S.camMode === "top") {
    // Straight down, world-north up. Not a true orthographic projection —
    // a long lens from high up, which reads the same at this scale.
    S.camera.position.set(t[0], S.orbit.dist * 1.25, -t[1] + 0.01);
    S.camera.up.set(0, 0, -1);
    S.camera.lookAt(w2t(t[0], t[1], 0));
    return;
  }
  S.camera.up.set(0, 1, 0);

  if (S.camMode === "chase") {
    var back = 16, up = 8;
    S.camera.position.copy(
      w2t(ex - back * Math.cos(h), ey - back * Math.sin(h), up));
    S.camera.lookAt(w2t(ex + 14 * Math.cos(h), ey + 14 * Math.sin(h), 0.6));
    return;
  }
  if (S.camMode === "sensor") {
    S.camera.position.copy(w2t(ex, ey, ego.sensorZ));
    S.camera.lookAt(
      w2t(ex + 30 * Math.cos(h), ey + 30 * Math.sin(h), ego.sensorZ - 1.2));
    return;
  }
  // orbit
  var o = S.orbit, cp = Math.cos(o.pitch);
  S.camera.position.set(
    t[0] + o.dist * cp * Math.cos(o.yaw),
    o.dist * Math.sin(o.pitch),
    -t[1] + o.dist * cp * Math.sin(o.yaw));
  S.camera.lookAt(w2t(t[0], t[1], 0));
}

// ─────────────────────────────────────────────────────────────
// pointer: orbit / pan / zoom / select
// ─────────────────────────────────────────────────────────────
function bindPointer() {
  var el = S.renderer.domElement;
  el.addEventListener("contextmenu", function (e) { e.preventDefault(); });

  el.addEventListener("pointerdown", function (e) {
    el.setPointerCapture(e.pointerId);
    S.drag = { x: e.clientX, y: e.clientY,
               pan: e.button === 1 || e.button === 2 || e.shiftKey };
    S.moved = 0;
  });

  el.addEventListener("pointermove", function (e) {
    if (!S.drag) { hoverTest(e); return; }
    var dx = e.clientX - S.drag.x, dy = e.clientY - S.drag.y;
    S.drag.x = e.clientX; S.drag.y = e.clientY;
    S.moved += Math.abs(dx) + Math.abs(dy);
    if (S.camMode !== "orbit" && S.camMode !== "top") {
      // Chase and sensor are anchored to the vehicle; dragging them would
      // mean two different ideas of where the camera is.
      return;
    }
    if (S.drag.pan) {
      // Pan in the ground plane, scaled so the grab point tracks the cursor.
      var k = S.orbit.dist * 0.0016;
      var yaw = S.camMode === "top" ? -Math.PI / 2 : S.orbit.yaw;
      S.orbit.target[0] -= (dx * Math.sin(yaw) + dy * Math.cos(yaw)) * k;
      S.orbit.target[1] -= (dx * -Math.cos(yaw) + dy * Math.sin(yaw)) * k;
    } else if (S.camMode === "orbit") {
      S.orbit.yaw -= dx * 0.006;
      S.orbit.pitch = Math.max(0.06, Math.min(1.45,
        S.orbit.pitch + dy * 0.005));
    }
    S.needsRender = true;
  });

  el.addEventListener("pointerup", function (e) {
    var wasDrag = S.moved > 5;
    S.drag = null;
    if (!wasDrag && e.button === 0) select(e);
  });

  el.addEventListener("wheel", function (e) {
    e.preventDefault();
    S.orbit.dist = Math.max(6, Math.min(320,
      S.orbit.dist * (1 + Math.sign(e.deltaY) * 0.11)));
    S.needsRender = true;
  }, { passive: false });
}

function ndc(e) {
  var r = S.renderer.domElement.getBoundingClientRect();
  return new THREE.Vector2(
    ((e.clientX - r.left) / r.width) * 2 - 1,
    -((e.clientY - r.top) / r.height) * 2 + 1);
}

function pick(e) {
  var ray = new THREE.Raycaster();
  ray.setFromCamera(ndc(e), S.camera);
  // Objects first, then cells — the same priority the Python hit test uses.
  var hitObj = ray.intersectObjects(S.objectGroup.children, true)[0];
  if (hitObj) {
    var o = hitObj.object;
    while (o && o.userData.objectId === undefined) o = o.parent;
    if (o) return { kind: "object", id: o.userData.objectId,
                    name: o.userData.label };
  }
  var c = S.cellXY;
  if (c) {
    var meshes = [];
    if (S.flatMesh && S.flatMesh.count) meshes.push([S.flatMesh, S.flatIdx]);
    if (S.boxMesh && S.boxMesh.count) meshes.push([S.boxMesh, S.boxIdx]);
    var best = null, bestMap = null;
    for (var k = 0; k < meshes.length; k++) {
      var hit = ray.intersectObject(meshes[k][0], false)[0];
      if (hit && hit.instanceId !== undefined &&
          (!best || hit.distance < best.distance)) {
        best = hit; bestMap = meshes[k][1];
      }
    }
    if (best) {
      var j = bestMap[best.instanceId];
      return { kind: "cell",
               x: c.ox + c.xy[2 * j] * 0.01,
               y: c.oy + c.xy[2 * j + 1] * 0.01 };
    }
  }
  return { kind: "empty" };
}

function select(e) {
  var p = pick(e);
  p.t = Date.now();
  if (p.kind === "empty") {
    // Still send it: "I clicked nothing" is a real answer, and the panel
    // says so rather than leaving the previous selection looking current.
    var g = groundPoint(e);
    if (g) { p.x = g[0]; p.y = g[1]; }
  }
  Streamlit.setComponentValue(p);
}

/** Where the cursor ray meets z = 0, for a click that hit no geometry. */
function groundPoint(e) {
  var ray = new THREE.Raycaster();
  ray.setFromCamera(ndc(e), S.camera);
  var hit = new THREE.Vector3();
  var plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
  if (!ray.ray.intersectPlane(plane, hit)) return null;
  return [hit.x, -hit.z];
}

function hoverTest(e) {
  var p = pick(e);
  var text = null;
  if (p.kind === "object") text = p.name;
  else if (p.kind === "cell") text = p.x.toFixed(2) + ", " + p.y.toFixed(2) + " m";
  if (text) {
    HOVER.style.display = "block";
    HOVER.textContent = text;
    var r = S.renderer.domElement.getBoundingClientRect();
    HOVER.style.left = (e.clientX - r.left) + "px";
    HOVER.style.top = (e.clientY - r.top) + "px";
  } else {
    HOVER.style.display = "none";
  }
}

// ─────────────────────────────────────────────────────────────
// cells
// ─────────────────────────────────────────────────────────────
/*
 * ONE mesh, reused for the life of the page.
 *
 * This used to dispose and rebuild the InstancedMesh every frame. At
 * 18,000 cells that is 1.15 MB of instance matrices plus 216 kB of
 * instance colours allocated, uploaded and thrown away several times a
 * second — enough GPU churn to stall an integrated driver and eventually
 * lose the WebGL context, which is what turned the canvas white.
 *
 * Now the mesh is allocated once at a capacity that only ever grows, and a
 * frame just overwrites the buffers it already owns and sets `count`.
 */
/*
 * TWO reused meshes, never rebuilt.
 *
 * Most map cells are ground: base and top within a couple of centimetres
 * of each other. Drawing those as boxes costs twelve triangles each to
 * show a surface that two would show, so they go into a flat instanced
 * quad and only the cells with real vertical extent get a box.
 *
 * Both are allocated once at a capacity that only grows. This used to
 * dispose and rebuild the mesh every frame: at 18,000 cells that is
 * 1.15 MB of instance matrices plus 216 kB of colours allocated,
 * uploaded and thrown away several times a second — enough GPU churn to
 * stall an integrated driver and lose the WebGL context, which is what
 * turned the canvas white.
 */
var FLAT_MAX_H = 0.06;      // metres of extent below which a cell is flat

function makeCellMesh(geo, capacity) {
  var mesh = new THREE.InstancedMesh(
    geo, new THREE.MeshLambertMaterial({}), capacity);
  mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  mesh.setColorAt(0, new THREE.Color(1, 1, 1));
  mesh.instanceColor.setUsage(THREE.DynamicDrawUsage);
  // Culled against a bounding sphere we maintain ourselves: three cannot
  // derive one from instance matrices, and leaving it unculled means the
  // whole grid is submitted even when the camera faces away from it.
  mesh.frustumCulled = true;
  mesh.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1e-3);
  return mesh;
}

function ensureCellMeshes(nFlat, nBox) {
  if (!S.flatMesh || S.flatCapacity < nFlat) {
    if (S.flatMesh) {
      S.scene.remove(S.flatMesh);
      S.flatMesh.geometry.dispose();
      S.flatMesh.material.dispose();
    }
    S.flatCapacity = Math.max(1024, Math.ceil(nFlat * 1.35));
    // A plane lying in the ground plane: two triangles instead of twelve.
    var quad = new THREE.PlaneBufferGeometry(1, 1);
    quad.rotateX(-Math.PI / 2);
    S.flatMesh = makeCellMesh(quad, S.flatCapacity);
    S.scene.add(S.flatMesh);
  }
  if (!S.boxMesh || S.boxCapacity < nBox) {
    if (S.boxMesh) {
      S.scene.remove(S.boxMesh);
      S.boxMesh.geometry.dispose();
      S.boxMesh.material.dispose();
    }
    S.boxCapacity = Math.max(512, Math.ceil(nBox * 1.35));
    S.boxMesh = makeCellMesh(
      new THREE.BoxBufferGeometry(1, 1, 1), S.boxCapacity);
    S.scene.add(S.boxMesh);
  }
}

/** Decode one frame's packed cell arrays, once, and keep them. */
function decodeCells(c) {
  if (!c || !c.n) return null;
  if (c._d) return c._d;
  c._d = {
    n: c.n,
    xy: b64(c.xy, Int16Array),
    zcm: b64(c.zcm, Int16Array),
    lvl: b64(c.level, Uint8Array),
    cls: b64(c.cls, Uint8Array),
    trav: b64(c.trav, Uint8Array),
    ret: b64(c.returns, Uint8Array),
    span: b64(c.span, Uint16Array),
    ox: c.origin[0], oy: c.origin[1],
    sizes: c.sizes, zRange: c.zRange,
  };
  return c._d;
}

function buildCells(view) {
  var c = decodeCells(view.cells);
  if (!c) {
    S.cellXY = null;
    if (S.flatMesh) S.flatMesh.count = 0;
    if (S.boxMesh) S.boxMesh.count = 0;
    return;
  }

  var n = c.n, xy = c.xy, zcm = c.zcm, lvl = c.lvl;
  var hs = view.heightScale;
  var gap = view.showGrid ? 0.88 : 1.0;

  // Split once so each mesh knows how many instances it needs.
  var nBox = 0;
  for (var i = 0; i < n; i++) {
    if ((zcm[2 * i + 1] - zcm[2 * i]) * 0.01 * hs > FLAT_MAX_H) nBox++;
  }
  ensureCellMeshes(n - nBox, nBox);
  if (!S.flatIdx || S.flatIdx.length < S.flatCapacity) {
    S.flatIdx = new Int32Array(S.flatCapacity);
  }
  if (!S.boxIdx || S.boxIdx.length < S.boxCapacity) {
    S.boxIdx = new Int32Array(S.boxCapacity);
  }

  var FM = S.flatMesh.instanceMatrix.array, FC = S.flatMesh.instanceColor.array;
  var BM = S.boxMesh.instanceMatrix.array, BC = S.boxMesh.instanceColor.array;
  var pal = S.run.palettes, mode = view.colourBy;
  var pSem = pal.semantic, pTrav = pal.traversability, pLvl = pal.level;
  var zlo = c.zRange[0], zhi = Math.max(c.zRange[1], zlo + 0.001);
  var cls = c.cls, trav = c.trav, ret = c.ret, span = c.span;
  var tmp = new THREE.Color();
  var fi = 0, bi = 0;
  var minX = Infinity, maxX = -Infinity, minZ = Infinity, maxZ = -Infinity;
  var minY = Infinity, maxY = -Infinity;

  for (var j = 0; j < n; j++) {
    // Drawn footprint: the cell's own resolution, or the bucket it
    // stands for where the display thinned its neighbours away.
    var size = span[j] * 0.01 * gap;
    var base = zcm[2 * j] * 0.01;
    var h = (zcm[2 * j + 1] * 0.01 - base) * hs;
    var flat = h <= FLAT_MAX_H;
    var x = c.ox + xy[2 * j] * 0.01;
    var z = -(c.oy + xy[2 * j + 1] * 0.01);
    var y = flat ? base + 0.01 : base + h / 2;

    // An axis-aligned scale and translation, written straight into the
    // buffer: compose() through a Quaternion and three Vector3s costs
    // more than the rest of the loop at this instance count.
    var M = flat ? FM : BM;
    var o = (flat ? fi : bi) * 16;
    M[o] = size;  M[o + 1] = 0; M[o + 2] = 0;    M[o + 3] = 0;
    M[o + 4] = 0; M[o + 5] = flat ? 1 : h;       M[o + 6] = 0;
    M[o + 7] = 0; M[o + 8] = 0; M[o + 9] = 0;    M[o + 10] = size;
    M[o + 11] = 0; M[o + 12] = x; M[o + 13] = y; M[o + 14] = z;
    M[o + 15] = 1;

    var rgb;
    if (mode === "traversability") rgb = pTrav[trav[j]];
    else if (mode === "resolution level") rgb = pLvl[lvl[j]];
    else if (mode === "height") {
      var t = (zcm[2 * j + 1] * 0.01 - zlo) / (zhi - zlo);
      if (t < 0) t = 0; else if (t > 1) t = 1;
      tmp.setHSL(0.62 - 0.62 * t, 0.62, 0.30 + 0.28 * t);
      rgb = [tmp.r, tmp.g, tmp.b];
    } else rgb = pSem[cls[j]];

    // A cell backed by one return is dimmer than one backed by forty: the
    // same "observed vs barely observed" distinction the inspector spells
    // out, without a separate layer.
    var k = ret[j] / 12;
    var conf = 0.62 + 0.38 * (k > 1 ? 1 : k);
    var C = flat ? FC : BC;
    var p = (flat ? fi : bi) * 3;
    C[p] = rgb[0] * conf; C[p + 1] = rgb[1] * conf; C[p + 2] = rgb[2] * conf;

    // Instance order is not cell order once the two meshes split them,
    // so a raycast hit needs the way back.
    if (flat) { S.flatIdx[fi] = j; fi++; } else { S.boxIdx[bi] = j; bi++; }
    if (x < minX) minX = x; if (x > maxX) maxX = x;
    if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
    if (y < minY) minY = y; if (y + h > maxY) maxY = y + h;
  }

  S.flatMesh.count = fi;
  S.boxMesh.count = bi;
  S.flatMesh.instanceMatrix.needsUpdate = true;
  S.flatMesh.instanceColor.needsUpdate = true;
  S.boxMesh.instanceMatrix.needsUpdate = true;
  S.boxMesh.instanceColor.needsUpdate = true;

  // One sphere around everything drawn, so culling is correct rather than
  // merely disabled.
  var cxm = (minX + maxX) / 2, cym = (minY + maxY) / 2;
  var czm = (minZ + maxZ) / 2;
  var rad = 0.5 * Math.sqrt((maxX - minX) * (maxX - minX) +
                            (maxY - minY) * (maxY - minY) +
                            (maxZ - minZ) * (maxZ - minZ)) + 1;
  [S.flatMesh, S.boxMesh].forEach(function (m) {
    m.boundingSphere.center.set(cxm, cym, czm);
    m.boundingSphere.radius = rad;
  });

  // Kept for the click path: world xy of every drawn cell, in order.
  S.cellXY = c;
}


// ─────────────────────────────────────────────────────────────
// objects
// ─────────────────────────────────────────────────────────────
function cachedGeometry(key, make) {
  var g = GEO_CACHE[key];
  if (!g) { g = make(); GEO_CACHE[key] = g; }
  return g;
}

function primitiveGeometry(o) {
  var s = o.size;
  var k = o.primitive + "|" + s.map(function (v) {
    return Math.round(v * 100);
  }).join(",");
  if (o.primitive === "pole") {
    return cachedGeometry(k, function () {
      var r = Math.max(s[0], s[1]) / 2;
      return new THREE.CylinderBufferGeometry(r, r, s[2], 10);
    });
  }
  if (o.primitive === "person") {
    // A narrow capsule-ish column. r128 has no CapsuleGeometry, and a
    // cylinder with a domed top reads as a person at this scale.
    return cachedGeometry(k, function () {
      return new THREE.CylinderBufferGeometry(
        Math.max(s[0], s[1]) / 2, Math.max(s[0], s[1]) / 2.4, s[2], 8);
    });
  }
  return cachedGeometry(k, function () {
    return new THREE.BoxBufferGeometry(s[0], s[1], s[2]);
  });
}

function cachedEdges(geo, key) {
  return cachedGeometry("E|" + key, function () {
    return new THREE.EdgesGeometry(geo);
  });
}

function buildObjects(args) {
  // Only the wrappers are dropped; the geometry they point at is cached
  // and shared, so disposing here would throw away work every frame.
  while (S.objectGroup.children.length) {
    S.objectGroup.remove(S.objectGroup.children[0]);
  }
  S.objects = args.objects || [];
  if (!args.showObjects) return;

  S.objects.forEach(function (o) {
    var g = new THREE.Group();
    g.userData.objectId = o.id;
    g.userData.label = o.label + "  ·  " + o.stateName;

    var geo = primitiveGeometry(o);
    var gkey = o.primitive + "|" + o.size.map(function (v) {
      return Math.round(v * 100);
    }).join(",");
    var rgb = STATE_RGB[o.state];
    var solid = new THREE.Mesh(geo, new THREE.MeshLambertMaterial({
      color: rgb, transparent: true,
      // A padded box is drawn fainter than a measured one, so a viewer can
      // see at a glance which outlines the sensor actually established.
      opacity: o.geomSource === "padded" ? 0.20 : 0.34,
      depthWrite: false,
    }));
    var wire = new THREE.LineSegments(
      cachedEdges(geo, gkey),
      new THREE.LineBasicMaterial({ color: rgb }));
    g.add(solid, wire);

    g.position.copy(w2t(o.centre[0], o.centre[1], o.centre[2]));
    if (o.primitive === "box" || o.primitive === "vehicle") {
      g.rotation.y = -o.yaw;
    }
    S.objectGroup.add(g);

    if (args.showVelocity && o.stateName === "MOVING" && o.speed > 0.4) {
      S.objectGroup.add(arrow(o));
    }
  });
}

/** Velocity as an arrow, 1 m per m/s — the 2D overlay's own convention. */
function arrow(o) {
  var from = w2t(o.centre[0], o.centre[1], o.centre[2]);
  var dir = w2t(o.vel[0], o.vel[1], 0).normalize();
  var len = Math.min(Math.max(o.speed, 1.0), 14);
  var a = new THREE.ArrowHelper(dir, from, len, STATE_RGB[1], 1.1, 0.6);
  a.userData.objectId = o.id;
  a.userData.label = o.label + "  ·  " + o.speed.toFixed(1) + " m/s";
  return a;
}

// ─────────────────────────────────────────────────────────────
// ego, rings, points, selection marker
// ─────────────────────────────────────────────────────────────
function buildEgo(args) {
  while (S.egoGroup.children.length) S.egoGroup.remove(S.egoGroup.children[0]);
  var e = args.ego, s = e.size;

  var body = new THREE.Mesh(
    new THREE.BoxBufferGeometry(s[0], s[1], s[2]),
    new THREE.MeshLambertMaterial({ color: 0x2f7fd1, transparent: true,
                                    opacity: 0.42, depthWrite: false }));
  var wire = new THREE.LineSegments(
    new THREE.EdgesGeometry(new THREE.BoxBufferGeometry(s[0], s[1], s[2])),
    new THREE.LineBasicMaterial({ color: 0x8fc7ff }));
  var g = new THREE.Group();
  g.add(body, wire);

  // Forward indicator, so "which way is the vehicle pointing" never has to
  // be inferred from the box.
  var nose = new THREE.Mesh(
    new THREE.ConeBufferGeometry(0.45, 1.4, 12),
    new THREE.MeshLambertMaterial({ color: 0xffd60a }));
  nose.rotation.z = -Math.PI / 2;
  nose.position.set(s[0] / 2 + 0.7, 0, 0);
  g.add(nose);

  // The sensor, at its configured mount height.
  var sensor = new THREE.Mesh(
    new THREE.SphereBufferGeometry(0.22, 12, 10),
    new THREE.MeshBasicMaterial({ color: 0xffffff }));
  sensor.position.set(0, 0, e.sensorZ - s[2] / 2);
  g.add(sensor);
  var mast = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(0, 0, 0),
      new THREE.Vector3(0, 0, e.sensorZ - s[2] / 2)]),
    new THREE.LineBasicMaterial({ color: 0x6f7784 }));
  g.add(mast);

  // Build in world axes, then rotate the whole rig into three's frame.
  g.rotation.x = -Math.PI / 2;
  var outer = new THREE.Group();
  outer.add(g);
  outer.rotation.y = -e.heading;
  outer.position.copy(w2t(e.xy[0], e.xy[1], s[2] / 2));
  S.egoGroup.add(outer);
}

function buildRings(args) {
  while (S.ringGroup.children.length) {
    S.ringGroup.remove(S.ringGroup.children[0]);
  }
  S.ringLabels = [];
  if (!args.showRings) return;
  var e = args.ego;
  args.rings.forEach(function (r) {
    var pts = [];
    for (var a = 0; a <= 96; a++) {
      var th = (a / 96) * Math.PI * 2;
      pts.push(w2t(e.xy[0] + r * Math.cos(th), e.xy[1] + r * Math.sin(th),
                   0.02));
    }
    var line = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(pts),
      new THREE.LineBasicMaterial({ color: 0x39404d }));
    S.ringGroup.add(line);
    S.ringLabels.push({
      r: r, text: r + " m",
      // Labelled ahead of the vehicle, where the scene is.
      p: w2t(e.xy[0] + r * Math.cos(e.heading),
             e.xy[1] + r * Math.sin(e.heading), 0.05),
    });
  });
}

function buildPoints(args) {
  var p = args.points;
  var n = (p && p.n) ? p.n : 0;
  if (!n) {
    if (S.pointCloud) S.pointCloud.geometry.setDrawRange(0, 0);
    return;
  }
  // Same story as the cells: one buffer, grown when it has to be, and
  // drawn up to `n` rather than reallocated every frame.
  if (!S.pointCloud || S.pointCapacity < n) {
    if (S.pointCloud) {
      S.scene.remove(S.pointCloud);
      S.pointCloud.geometry.dispose();
      S.pointCloud.material.dispose();
    }
    S.pointCapacity = Math.max(4096, Math.ceil(n * 1.35));
    var geo = new THREE.BufferGeometry();
    var attr = new THREE.BufferAttribute(
      new Float32Array(S.pointCapacity * 3), 3);
    attr.setUsage(THREE.DynamicDrawUsage);
    geo.setAttribute("position", attr);
    S.pointCloud = new THREE.Points(geo, new THREE.PointsMaterial({
      color: 0xeb2b46, size: 0.16, sizeAttenuation: true }));
    S.pointCloud.frustumCulled = false;
    S.scene.add(S.pointCloud);
  }
  var xyz = b64(p.xyz, Float32Array);
  var dst = S.pointCloud.geometry.attributes.position.array;
  for (var i = 0; i < n; i++) {
    dst[3 * i] = xyz[3 * i];
    dst[3 * i + 1] = xyz[3 * i + 2];
    dst[3 * i + 2] = -xyz[3 * i + 1];
  }
  S.pointCloud.geometry.attributes.position.needsUpdate = true;
  S.pointCloud.geometry.setDrawRange(0, n);
}

function buildMarker(args) {
  while (S.markerGroup.children.length) {
    S.markerGroup.remove(S.markerGroup.children[0]);
  }
  var sel = args.selected;
  if (!sel || !sel.kind) return;

  if (sel.kind === "cell") {
    var size = Math.max(sel.size, 0.35);   // findable even at 5 cm
    var h = Math.max((sel.z[1] - sel.z[0]) * args.heightScale, 0.05) + 0.3;
    var box = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxBufferGeometry(size, h, size)),
      new THREE.LineBasicMaterial({ color: SELECT_RGB }));
    box.position.copy(w2t(sel.xy[0], sel.xy[1], sel.z[0] + h / 2));
    S.markerGroup.add(box);
    S.markerGroup.add(beacon(sel.xy[0], sel.xy[1], sel.z[1] + 0.4));
  } else if (sel.kind === "object") {
    var o = null;
    S.objects.forEach(function (x) { if (x.id === sel.id) o = x; });
    if (!o) return;
    var halo = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxBufferGeometry(
        o.size[0] + 0.7, o.size[2] + 0.7, o.size[1] + 0.7)),
      new THREE.LineBasicMaterial({ color: SELECT_RGB }));
    halo.position.copy(w2t(o.centre[0], o.centre[1], o.centre[2]));
    S.markerGroup.add(halo);
    S.markerGroup.add(beacon(o.centre[0], o.centre[1],
                             o.centre[2] + o.size[2] / 2 + 0.5));
  }
}

/** A short vertical stalk, so a selected 5 cm cell is findable at 50 m. */
function beacon(x, y, z) {
  return new THREE.Line(
    new THREE.BufferGeometry().setFromPoints([
      w2t(x, y, z), w2t(x, y, z + 2.2)]),
    new THREE.LineBasicMaterial({ color: SELECT_RGB }));
}

// ─────────────────────────────────────────────────────────────
// HTML labels, projected each frame
// ─────────────────────────────────────────────────────────────
function refreshLabels(args) {
  LABELS.innerHTML = "";
  var w = S.width, h = S.height;

  var placed = [];
  function place(text, v3, cls, spaced) {
    var p = v3.clone().project(S.camera);
    if (p.z > 1 || p.x < -1.05 || p.x > 1.05 || p.y < -1.05 || p.y > 1.05) {
      return;
    }
    var sx = (p.x + 1) / 2 * w, sy = (1 - p.y) / 2 * h;
    if (spaced) {
      // Overlapping labels are worse than missing ones: two stacked boxes
      // are unreadable, one is not.
      for (var i = 0; i < placed.length; i++) {
        if (Math.abs(placed[i][0] - sx) < 96 &&
            Math.abs(placed[i][1] - sy) < 15) return;
      }
      placed.push([sx, sy]);
    }
    var d = document.createElement("div");
    d.className = cls;
    d.textContent = text;
    d.style.left = sx + "px";
    d.style.top = sy + "px";
    LABELS.appendChild(d);
  }

  if (args.showLabels) {
    // Nearest first, capped: a label on every one of forty tracks is a
    // wall of text, not information.
    var sorted = S.objects.slice().sort(function (a, b) {
      // Movers first, then the non-ground classes, then nearest. A label
      // on a cluster the classifier called road is real output but it is
      // not what anyone is reading the scene for.
      var ka = (a.stateName === "MOVING" ? 0 : 1) * 10 + (a.cls < 2 ? 5 : 0);
      var kb = (b.stateName === "MOVING" ? 0 : 1) * 10 + (b.cls < 2 ? 5 : 0);
      return ka !== kb ? ka - kb : a.range - b.range;
    }).slice(0, 12);
    sorted.forEach(function (o) {
      place(o.label,
            w2t(o.centre[0], o.centre[1], o.centre[2] + o.size[2] / 2 + 0.35),
            "lbl" + (o.stateName === "MOVING" ? " moving" : ""), true);
    });
  }
  S.ringLabels.forEach(function (r) { place(r.text, r.p, "ring-lbl"); });
}

// ─────────────────────────────────────────────────────────────
// render loop
// ─────────────────────────────────────────────────────────────
function animate() {
  requestAnimationFrame(animate);
  if (!S.needsRender || !S.view) return;
  S.needsRender = false;
  placeCamera(S.view);
  S.renderer.render(S.scene, S.camera);
  refreshLabels(S.view);
}

// ─────────────────────────────────────────────────────────────
// Streamlit
// ─────────────────────────────────────────────────────────────
/** Draw one composed view. Everything below reads from the cached run. */
function drawView(view) {
  var t0 = performance.now();
  buildCells(view);
  buildObjects(view);
  buildEgo(view);
  buildRings(view);
  buildPoints(view);
  buildMarker(view);

  var buildMs = performance.now() - t0;
  S.buildMs = S.buildMs ? S.buildMs * 0.7 + buildMs * 0.3 : buildMs;
  S.view = view;
  S.needsRender = true;
}

/** Frame data plus the display options, in one object the builders read. */
function composeView(frame, state, args) {
  var view = {
    cells: frame.cells, objects: frame.objects, ego: frame.ego,
    points: frame.points, frame: frame.frame,
    rings: S.run.rings, palettes: S.run.palettes,
    colourBy: state.colourBy, heightScale: state.heightScale,
    selected: state.selected || {},
    showGrid: args.showGrid, showObjects: args.showObjects,
    showLabels: args.showLabels, showRings: args.showRings,
    showVelocity: args.showVelocity,
  };
  return view;
}

function onRender(event) {
  var args = event.detail.args;

  // The run arrives once, when it changed. On every other rerun it is
  // null and we draw from the copy already here — that is the whole
  // point of the split, and it is why playback no longer ships the map.
  if (args.run) S.run = args.run;
  if (!S.run) {
    // A page reload drops this copy while the Streamlit session still
    // believes it was sent. Ask for it rather than sitting blank.
    EMPTY.textContent = "loading scene…";
    EMPTY.style.display = "flex";
    Streamlit.setComponentValue({ kind: "need_run", t: Date.now() });
    return;
  }

  var state = args.state || {};
  var idx = Math.max(0, Math.min(state.frame | 0, S.run.nFrames - 1));
  var frame = S.run.frames[idx];
  if (!frame) return;

  var width = HOST.clientWidth || 900;
  var height = args.height || 520;
  if (!S.renderer) { S.width = width; S.height = height; init(width, height); }
  if (width !== S.width || height !== S.height) {
    S.width = width; S.height = height;
    S.renderer.setSize(width, height);
    S.camera.aspect = width / height;
    S.camera.updateProjectionMatrix();
  }

  // The orbit target follows the vehicle so the camera stays useful as the
  // run plays, but the world itself is never transformed: every position
  // is world-anchored and drawn exactly where Python put it.
  S.orbit.target = [frame.ego.xy[0], frame.ego.xy[1], 0];

  EMPTY.style.display = frame.cells && frame.cells.n ? "none" : "flex";
  drawView(composeView(frame, state, args));

  HUD.innerHTML =
    "frame " + (idx + 1) + " / " + (state.nFrames || S.run.nFrames) +
    "<br>" + frame.cells.n.toLocaleString() + " cells · " +
    frame.objects.length + " objects" +
    "<br>" + S.buildMs.toFixed(0) + " ms to build in the browser" +
    "<br>drag orbit · shift-drag pan · wheel zoom · click to inspect";

  // setFrameHeight is NOT called here. The height only changes when the
  // column does, and announcing it on every frame makes Streamlit
  // re-render the component host — roughly doubling the reruns during
  // playback for a number that did not change.
}

Streamlit.events.addEventListener(Streamlit.RENDER_EVENT, onRender);
Streamlit.setComponentReady();
Streamlit.setFrameHeight(520);
