/* Dependency-graph panel for the chaos-ai dashboard.
 *
 * Pure functions (layout, blast radius, health mapping, SVG building) live here so they can be tested with
 * Node; the page only calls renderGraph(). No external libraries, so it works offline.
 *
 * Reading the picture: an arrow A -> B means "A depends on B" (A calls it). Services are laid out left to
 * right along the user's critical path (vote -> redis -> worker -> db -> result), so for this app every
 * arrow connects neighbours. Node border colour is the live container health; a dashed pink ring marks the
 * critical path; while a fault runs the faulted service is pulsing red and everything that depends on it
 * (its blast radius) is amber; after the experiment, services that really restarted keep a badge.
 */
(function (root) {
  'use strict';

  const VW = 900, NODE_H = 84;

  function esc(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }

  /* critical path first (in order), then any other service */
  function graphOrder(g) {
    const names = Object.keys((g && g.services) || {});
    const path = ((g && g.critical_path) || []).filter(function (n) { return names.indexOf(n) >= 0; });
    return path.concat(names.filter(function (n) { return path.indexOf(n) < 0; }));
  }

  /* who depends on `target`, directly or transitively (the services at risk when it fails) */
  function blastRadius(g, target) {
    const rev = {};
    Object.keys((g && g.services) || {}).forEach(function (n) {
      (g.services[n].depends_on || []).forEach(function (d) { (rev[d] = rev[d] || []).push(n); });
    });
    const seen = new Set(), stack = [target];
    while (stack.length) {
      const x = stack.pop();
      (rev[x] || []).forEach(function (p) {
        if (p !== target && !seen.has(p)) { seen.add(p); stack.push(p); }
      });
    }
    return seen;
  }

  /* container -> one of: ok | warn | down | plain (running, no healthcheck) | unknown */
  function nodeHealth(c) {
    if (!c) return 'unknown';
    if (c.state !== 'running') return 'down';
    if (c.health === 'unhealthy') return 'down';
    if (c.health === 'starting') return 'warn';
    if (c.health === 'healthy') return 'ok';
    return 'plain';
  }

  function layout(g) {
    const names = graphOrder(g), n = names.length;
    // headroom is only needed when some arrow must arc over/under the row, so a plain left-to-right chain stays compact
    const hasArc = names.some(function (name, ia) {
      return ((g.services[name].depends_on) || []).some(function (dep) {
        const ib = names.indexOf(dep);
        return ib >= 0 && Math.abs(ia - ib) !== 1;
      });
    });
    const CY = hasArc ? 150 : 104, VH = hasArc ? 300 : 208;
    const step = n > 1 ? (VW - 160) / (n - 1) : 0;
    const w = n > 1 ? Math.max(90, Math.min(150, step - 34)) : 150;
    const pos = {};
    names.forEach(function (name, i) {
      pos[name] = { x: n > 1 ? 80 + i * step : VW / 2, y: CY, w: w, h: NODE_H };
    });
    const edges = [];
    let arc = 0;
    names.forEach(function (name, ia) {
      ((g.services[name].depends_on) || []).forEach(function (dep) {
        const b = pos[dep];
        if (!b) return;
        const a = pos[name], ib = names.indexOf(dep);
        let d;
        if (Math.abs(ia - ib) === 1) {                       // neighbours: a straight arrow between the facing sides
          const s = Math.sign(b.x - a.x);
          d = 'M' + (a.x + s * a.w / 2) + ',' + CY + ' L' + (b.x - s * b.w / 2 - 2) + ',' + CY;
        } else {                                              // otherwise an arc over or under the row, alternating
          const up = (arc++ % 2 === 0);
          const sy = up ? CY - NODE_H / 2 : CY + NODE_H / 2;
          const lift = 36 + 14 * Math.abs(ia - ib);
          d = 'M' + a.x + ',' + sy + ' Q' + ((a.x + b.x) / 2) + ',' + (up ? sy - lift : sy + lift) + ' ' + b.x + ',' + (sy + (up ? -2 : 2));
        }
        edges.push({ from: name, to: dep, d: d });
      });
    });
    return { names: names, pos: pos, edges: edges, width: VW, height: VH, cy: CY };
  }

  function tagLine(svc) {
    const t = ['×' + svc.replicas];
    if (svc.stateful) t.push('stateful');
    if (svc.entrypoint) t.push('user-facing');
    return t.join(' · ');
  }

  function warnings(svc) {                        // one short line each, so they fit inside the node
    const w = [];
    if (svc.replicas < 2) w.push('single replica');
    if (!svc.healthcheck) w.push('no healthcheck');
    return w;
  }

  function badge(text, cls, y) {
    const w = 8 + text.length * 6.4;
    return '<g class="badge ' + cls + '" transform="translate(0,' + y + ')"><rect x="' + (-w / 2) + '" y="-9" width="' + w +
      '" height="18" rx="9"/><text text-anchor="middle" y="4">' + esc(text) + '</text></g>';
  }

  /* st = { containers: {svc: {state, health, restarts}}, experiment: {...dashboard state}, errors: {vote: bool, result: bool} } */
  function buildSvg(g, st) {
    if (!g || !g.services || !Object.keys(g.services).length) return '';
    st = st || {};
    const L = layout(g), ex = st.experiment || {}, cont = st.containers || {}, errs = st.errors || {};
    const active = !!ex.running && !!ex.target && ex.phase !== 'baseline';
    const risk = active ? blastRadius(g, ex.target) : new Set();
    const observed = (!ex.running && ex.result && ex.result.restarts) ? ex.result.restarts : {};
    const lastTarget = (!ex.running && ex.result) ? ex.target : null;

    let out = '<defs>' +
      '<marker id="garr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" style="fill:var(--muted)"/></marker>' +
      '<marker id="garrhot" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" style="fill:var(--bad)"/></marker>' +
      '</defs>';

    L.edges.forEach(function (e) {
      const hot = active && (e.to === ex.target || risk.has(e.to)) && risk.has(e.from);
      out += '<path class="gedge' + (hot ? ' hot' : '') + '" d="' + e.d + '" marker-end="url(#' + (hot ? 'garrhot' : 'garr') + ')"/>';
    });

    L.names.forEach(function (name) {
      const s = g.services[name], p = L.pos[name], h = nodeHealth(cont[name]);
      const cls = ['gnode', 'h-' + h];
      if (s.on_critical_path) cls.push('crit');
      const isTarget = active && name === ex.target;
      if (isTarget && ex.phase === 'fault') cls.push('fault');
      else if (isTarget) cls.push('recovering');
      else if (risk.has(name) && ex.phase === 'fault') cls.push('risk');
      const c = cont[name];
      const tip = name + ' (' + s.image + ')\nreplicas: ' + s.replicas + '  restart policy: ' + s.restart_policy +
        '\nhealthcheck: ' + (s.healthcheck ? 'yes' : 'NO') + '  resource limits: ' + (s.has_limits ? 'yes' : 'no') +
        '\ndepends on: ' + ((s.depends_on || []).join(', ') || '-') + '\nfan-in: ' + s.fan_in + '  fan-out: ' + s.fan_out +
        (c ? '\nnow: ' + c.state + (c.health ? ' / ' + c.health : '') + '  restarts: ' + c.restarts : '');
      out += '<g class="' + cls.join(' ') + '" transform="translate(' + p.x + ',' + p.y + ')"><title>' + esc(tip) + '</title>' +
        '<rect class="ring" x="' + (-p.w / 2 - 5) + '" y="' + (-p.h / 2 - 5) + '" width="' + (p.w + 10) + '" height="' + (p.h + 10) + '" rx="15"/>' +
        '<rect class="body" x="' + (-p.w / 2) + '" y="' + (-p.h / 2) + '" width="' + p.w + '" height="' + p.h + '" rx="10"/>' +
        '<text class="nm" text-anchor="middle" y="-18">' + esc(name) + '</text>' +
        '<text class="sub" text-anchor="middle" y="-1">' + esc(tagLine(s)) + '</text>' +
        warnings(s).map(function (w, i) { return '<text class="wrn" text-anchor="middle" y="' + (16 + i * 14) + '">' + esc(w) + '</text>'; }).join('');
      let by = -p.h / 2 - 17;
      if (isTarget && ex.phase === 'fault') out += badge('FAULT: ' + (ex.fault || ''), 'b-fault', by);
      else if (isTarget) out += badge('recovering', 'b-warn', by);
      else if (risk.has(name) && ex.phase === 'fault') out += badge('at risk', 'b-warn', by);
      else if (name === lastTarget) out += badge('last fault: ' + (ex.fault || ''), 'b-dim', by);
      if (observed[name] > 0) out += badge('↻ restarted ×' + observed[name], 'b-fault', p.h / 2 + 17);
      else if (errs[name]) out += badge('user errors', 'b-fault', p.h / 2 + 17);
      out += '</g>';
    });
    return out;
  }

  function renderGraph(el, g, st) {
    if (!el) return;
    const L = (g && g.services && Object.keys(g.services).length) ? layout(g) : { height: 208 };
    el.setAttribute('viewBox', '0 0 ' + VW + ' ' + L.height);
    el.innerHTML = buildSvg(g, st);
  }

  const api = { esc: esc, graphOrder: graphOrder, blastRadius: blastRadius, nodeHealth: nodeHealth,
                layout: layout, buildSvg: buildSvg, renderGraph: renderGraph };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.ChaosGraph = api;
})(typeof window !== 'undefined' ? window : this);
