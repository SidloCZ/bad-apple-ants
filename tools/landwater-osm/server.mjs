// Local map server for the land/water + OpenStreetMap edition.
//
// * /tiles/{z}/{x}/{y}.mvt — reads single tiles from a remote Protomaps planet
//   build (PMTiles v3) with HTTP range requests and caches them on disk, so each
//   tile is fetched once. Protomaps documents remote partial reads of its builds
//   (`pmtiles extract` "may be local or remote"); this reads far less than the
//   full extract. Remote concurrency is capped.
// * /assets/*, /lib/* — fonts/sprites (protomaps basemaps-assets) and the
//   MapLibre / basemaps scripts, cached on disk after the first request.
// * /render.html, /views.json — the renderer page and its work list.
// * POST /result/{name}.png — rendered maps written to the output directory.
//
// No npm dependencies. Usage:
//   node server.mjs --port 8130 --cache C:/.../osm --pmtiles https://build.protomaps.com/20261109.pmtiles
//                   --views views.json --out renders/
import http from 'node:http';
import fs from 'node:fs';
import fsp from 'node:fs/promises';
import path from 'node:path';
import zlib from 'node:zlib';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const args = Object.fromEntries(process.argv.slice(2).reduce((acc, v, i, a) => (v.startsWith('--') ? [...acc, [v.slice(2), a[i + 1]]] : acc), []));
const PORT = Number(args.port || 8130);
const CACHE = path.resolve(args.cache);
const PMTILES = args.pmtiles;
const VIEWS = args.views ? path.resolve(args.views) : null;
const OUT = args.out ? path.resolve(args.out) : null;
const REMOTE_LIMIT = Number(args.concurrency || 8);
const UPSTREAM = {
  '/assets/': 'https://protomaps.github.io/basemaps-assets/',
  '/lib/maplibre-gl/': 'https://unpkg.com/maplibre-gl@5.24.0/dist/',
  '/lib/basemaps/': 'https://unpkg.com/@protomaps/basemaps@5.7.2/dist/',
};
const stats = { remoteRanges: 0, remoteBytes: 0, tileHits: 0, tileMisses: 0, emptyTiles: 0, results: 0, errors: 0, shardsDone: new Set() };

// ---------- bounded remote fetch with retry ----------
let active = 0; const waiting = [];
async function slot(fn) {
  if (active >= REMOTE_LIMIT) await new Promise((r) => waiting.push(r));
  active++;
  try { return await fn(); } finally { active--; waiting.shift()?.(); }
}
async function fetchRetry(url, init = {}) {
  let delay = 500;
  for (let attempt = 0; ; attempt++) {
    try {
      const res = await slot(() => fetch(url, init));
      if (res.status === 429 || res.status >= 500) throw new Error(`HTTP ${res.status}`);
      return res;
    } catch (e) {
      if (attempt >= 6) throw e;
      await new Promise((r) => setTimeout(r, delay)); delay *= 2;
    }
  }
}
async function range(offset, length) {
  const res = await fetchRetry(PMTILES, { headers: { Range: `bytes=${offset}-${offset + length - 1}` } });
  if (res.status !== 206) throw new Error(`Range request returned ${res.status}`);
  const buf = Buffer.from(await res.arrayBuffer());
  if (buf.length !== length) throw new Error('Short range response');
  stats.remoteRanges++; stats.remoteBytes += length;
  return buf;
}

// ---------- PMTiles v3 ----------
function u64(b, o) { return Number(b.readBigUInt64LE(o)); }
let header = null;
async function getHeader() {
  if (header) return header;
  const file = path.join(CACHE, 'header.bin');
  let b;
  try { b = await fsp.readFile(file); } catch { b = await range(0, 127); await fsp.mkdir(CACHE, { recursive: true }); await fsp.writeFile(file, b); }
  if (b.toString('latin1', 0, 7) !== 'PMTiles' || b[7] !== 3) throw new Error('Not a PMTiles v3 archive');
  header = {
    rootOffset: u64(b, 8), rootLength: u64(b, 16), metadataOffset: u64(b, 24), metadataLength: u64(b, 32),
    leafOffset: u64(b, 40), leafLength: u64(b, 48), dataOffset: u64(b, 56), dataLength: u64(b, 64),
    internalCompression: b[97], tileCompression: b[98], tileType: b[99], minZoom: b[100], maxZoom: b[101],
  };
  return header;
}
function decompress(buf, kind) {
  if (kind === 1 || kind === 0) return buf;
  if (kind === 2) return zlib.gunzipSync(buf);
  throw new Error(`Unsupported PMTiles compression ${kind}`);
}
function readVarint(buf, pos) {
  let result = 0, shift = 0, b;
  do { b = buf[pos.i++]; result += (b & 0x7f) * 2 ** shift; shift += 7; } while (b & 0x80);
  return result;
}
function parseDirectory(raw) {
  const pos = { i: 0 }; const n = readVarint(raw, pos); const entries = new Array(n);
  let last = 0;
  for (let i = 0; i < n; i++) { last += readVarint(raw, pos); entries[i] = { tileId: last, runLength: 0, length: 0, offset: 0 }; }
  for (let i = 0; i < n; i++) entries[i].runLength = readVarint(raw, pos);
  for (let i = 0; i < n; i++) entries[i].length = readVarint(raw, pos);
  for (let i = 0; i < n; i++) {
    const v = readVarint(raw, pos);
    entries[i].offset = v === 0 && i > 0 ? entries[i - 1].offset + entries[i - 1].length : v - 1;
  }
  return entries;
}
const directories = new Map(); const pendingDirs = new Map();
async function directory(offset, length) {
  const key = `${offset}-${length}`;
  if (directories.has(key)) return directories.get(key);
  if (pendingDirs.has(key)) return pendingDirs.get(key);
  const job = (async () => {
    const file = path.join(CACHE, 'dirs', `${key}.bin`);
    let raw;
    try { raw = await fsp.readFile(file); } catch {
      raw = await range(offset, length);
      await fsp.mkdir(path.dirname(file), { recursive: true }); await fsp.writeFile(file, raw);
    }
    const entries = parseDirectory(decompress(raw, (await getHeader()).internalCompression));
    directories.set(key, entries); pendingDirs.delete(key);
    return entries;
  })();
  pendingDirs.set(key, job);
  return job;
}
function zxyToTileId(z, x, y) {
  let acc = 0;
  for (let t = 0; t < z; t++) acc += 4 ** t;
  let tx = x, ty = y, d = 0;
  for (let s = 2 ** (z - 1); s >= 1; s /= 2) {
    const rx = (tx & s) > 0 ? 1 : 0, ry = (ty & s) > 0 ? 1 : 0;
    d += s * s * ((3 * rx) ^ ry);
    if (ry === 0) { if (rx === 1) { tx = s - 1 - tx; ty = s - 1 - ty; } [tx, ty] = [ty, tx]; }
  }
  return acc + d;
}
function find(entries, id) {
  let lo = 0, hi = entries.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (entries[mid].tileId < id) lo = mid + 1; else if (entries[mid].tileId > id) hi = mid - 1; else return entries[mid];
  }
  if (hi >= 0) { const e = entries[hi]; if (e.runLength === 0) return e; if (id - e.tileId < e.runLength) return e; }
  return null;
}
async function tileBytes(z, x, y) {
  const h = await getHeader();
  if (z < h.minZoom || z > h.maxZoom) return null;
  const id = zxyToTileId(z, x, y);
  let dirOffset = h.rootOffset, dirLength = h.rootLength;
  for (let depth = 0; depth < 4; depth++) {
    const e = find(await directory(dirOffset, dirLength), id);
    if (!e) return null;
    if (e.runLength > 0) return range(h.dataOffset + e.offset, e.length);
    dirOffset = h.leafOffset + e.offset; dirLength = e.length;
  }
  throw new Error('PMTiles directory too deep');
}
const pendingTiles = new Map();
async function tile(z, x, y) {
  const file = path.join(CACHE, 'tiles', String(z), String(x), `${y}.mvt`);
  try { const b = await fsp.readFile(file); stats.tileHits++; return b; } catch {}
  const key = `${z}/${x}/${y}`;
  if (pendingTiles.has(key)) return pendingTiles.get(key);
  const job = (async () => {
    stats.tileMisses++;
    const b = (await tileBytes(z, x, y)) || Buffer.alloc(0);
    await fsp.mkdir(path.dirname(file), { recursive: true });
    await fsp.writeFile(file + '.part', b); await fsp.rename(file + '.part', file);
    pendingTiles.delete(key);
    return b;
  })();
  pendingTiles.set(key, job);
  return job;
}

// ---------- cached upstream proxy ----------
const pendingAssets = new Map();
async function upstream(prefix, rest) {
  const file = path.join(CACHE, 'upstream', prefix.replace(/\//g, '_'), decodeURIComponent(rest));
  try { return await fsp.readFile(file); } catch {}
  if (pendingAssets.has(file)) return pendingAssets.get(file);
  const job = (async () => {
    const res = await fetchRetry(UPSTREAM[prefix] + rest);
    let b;
    if (res.status === 404 && prefix === '/assets/' && rest.startsWith('fonts/')) {
      // The font set has no glyphs in this Unicode range (e.g. emoji): an empty
      // glyph PBF is valid and simply renders nothing for those characters.
      stats.emptyGlyphRanges = (stats.emptyGlyphRanges || 0) + 1; b = Buffer.alloc(0);
    } else if (res.status !== 200) { pendingAssets.delete(file); const e = new Error(`upstream ${res.status}`); e.status = res.status; throw e; }
    else b = Buffer.from(await res.arrayBuffer());
    await fsp.mkdir(path.dirname(file), { recursive: true }); await fsp.writeFile(file, b);
    pendingAssets.delete(file);
    return b;
  })();
  pendingAssets.set(file, job);
  return job;
}

const TYPES = { '.js': 'text/javascript', '.css': 'text/css', '.json': 'application/json', '.png': 'image/png', '.pbf': 'application/x-protobuf', '.html': 'text/html; charset=utf-8' };
function send(res, status, body, type, extra = {}) {
  res.writeHead(status, { 'Content-Type': type || 'application/octet-stream', 'Access-Control-Allow-Origin': '*', 'Cache-Control': 'no-store', ...extra });
  res.end(body);
}
const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, `http://${req.headers.host}`);
  const p = url.pathname;
  try {
    let m;
    if ((m = p.match(/^\/tiles\/(\d+)\/(\d+)\/(\d+)\.mvt$/))) {
      const b = await tile(+m[1], +m[2], +m[3]);
      if (!b.length) { stats.emptyTiles++; return send(res, 204, ''); }
      const gz = (await getHeader()).tileCompression === 2;
      return send(res, 200, b, 'application/x-protobuf', gz ? { 'Content-Encoding': 'gzip' } : {});
    }
    for (const prefix of Object.keys(UPSTREAM)) {
      if (p.startsWith(prefix)) {
        const rest = p.slice(prefix.length);
        return send(res, 200, await upstream(prefix, rest), TYPES[path.extname(rest)]);
      }
    }
    if (p === '/' || p === '/render.html') return send(res, 200, await fsp.readFile(path.join(here, 'render.html')), TYPES['.html']);
    if (p === '/style.js') return send(res, 200, await fsp.readFile(path.join(here, 'style.js')), TYPES['.js']);
    if (p === '/views.json' && VIEWS) return send(res, 200, await fsp.readFile(VIEWS), TYPES['.json']);
    if (req.method === 'POST' && (m = p.match(/^\/result\/([\w.-]+\.png)$/)) && OUT) {
      const chunks = []; for await (const c of req) chunks.push(c);
      await fsp.mkdir(OUT, { recursive: true });
      const file = path.join(OUT, m[1]); await fsp.writeFile(file + '.part', Buffer.concat(chunks)); await fsp.rename(file + '.part', file);
      stats.results++; return send(res, 200, 'ok', 'text/plain');
    }
    if (req.method === 'POST' && (m = p.match(/^\/log$/))) {
      const chunks = []; for await (const c of req) chunks.push(c);
      const line = Buffer.concat(chunks).toString(); if (/error/i.test(line)) stats.errors++;
      console.log(`[page] ${line}`); return send(res, 200, 'ok', 'text/plain');
    }
    if ((m = p.match(/^\/done\/(\d+)$/))) { stats.shardsDone.add(+m[1]); return send(res, 200, 'ok', 'text/plain'); }
    if (p === '/status') return send(res, 200, JSON.stringify({ ...stats, shardsDone: [...stats.shardsDone], active, header }), TYPES['.json']);
    return send(res, 404, 'not found', 'text/plain');
  } catch (e) {
    stats.errors++; console.error(`${p}: ${e.stack || e}`);
    return send(res, e.status || 502, String(e.message || e), 'text/plain');
  }
});
server.listen(PORT, '127.0.0.1', () => console.log(`landwater-osm server on http://127.0.0.1:${PORT} (pmtiles ${PMTILES})`));
