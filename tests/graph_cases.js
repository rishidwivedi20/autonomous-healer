// Test cases for chaosai/static/graph.js, run by tests/test_graph_js.py with Node.
// Prints one JSON object {case_name: "ok" | "<failure message>"}.
const G = require('../chaosai/static/graph.js');

const svc = (name, o) => Object.assign({ name, image: 'img/' + name, replicas: 1, healthcheck: true, restart_policy: 'unless-stopped',
  has_limits: false, stateful: false, depends_on: [], fan_in: 0, fan_out: 0, on_critical_path: true, entrypoint: false }, o || {});

// the real voting-app graph
const voting = {
  critical_path: ['vote', 'redis', 'worker', 'db', 'result'],
  services: {
    vote: svc('vote', { depends_on: ['redis'], entrypoint: true }),
    redis: svc('redis', { stateful: true, fan_in: 2 }),
    worker: svc('worker', { depends_on: ['redis', 'db'], healthcheck: false }),
    db: svc('db', { stateful: true, fan_in: 2 }),
    result: svc('result', { depends_on: ['db'], entrypoint: true }),
  },
};
const healthy = { vote: { state: 'running', health: 'healthy', restarts: 0 }, redis: { state: 'running', health: 'healthy', restarts: 0 },
  worker: { state: 'running', health: null, restarts: 0 }, db: { state: 'running', health: 'healthy', restarts: 0 },
  result: { state: 'running', health: 'healthy', restarts: 0 } };

const count = (s, sub) => s.split(sub).length - 1;
const eq = (a, b, msg) => { if (JSON.stringify(a) !== JSON.stringify(b)) throw new Error((msg || 'not equal') + ': ' + JSON.stringify(a) + ' vs ' + JSON.stringify(b)); };
const yes = (c, msg) => { if (!c) throw new Error(msg || 'assertion failed'); };

const cases = {
  order_follows_critical_path_then_the_rest() {
    const g = { critical_path: ['c', 'ghost', 'a', 'b'], services: { a: svc('a'), b: svc('b'), c: svc('c'), x: svc('x', { on_critical_path: false }) } };
    eq(G.graphOrder(g), ['c', 'a', 'b', 'x'], 'unknown path entries are dropped, extras appended');
  },
  voting_graph_is_laid_out_left_to_right_with_straight_neighbour_arrows() {
    const L = G.layout(voting);
    eq(L.names, ['vote', 'redis', 'worker', 'db', 'result']);
    const xs = L.names.map(n => L.pos[n].x);
    yes(xs.every((x, i) => i === 0 || x > xs[i - 1]), 'x increases along the path');
    yes(L.names.every(n => L.pos[n].x - L.pos[n].w / 2 >= 0 && L.pos[n].x + L.pos[n].w / 2 <= L.width), 'every node fits in the viewBox');
    eq(L.edges.length, 4);
    yes(L.edges.every(e => e.d.includes(' L') && !e.d.includes('Q')), 'all four edges join neighbours, so they are straight');
    eq(L.edges.map(e => e.from + '>' + e.to).sort(), ['result>db', 'vote>redis', 'worker>db', 'worker>redis']);
  },
  headroom_is_added_only_when_an_arrow_has_to_arc() {
    const chain = G.layout(voting);
    const arcs = G.layout({ critical_path: ['a', 'b', 'c'], services: { a: svc('a', { depends_on: ['c'] }), b: svc('b'), c: svc('c') } });
    yes(chain.height < arcs.height && chain.cy < arcs.cy, 'a chain is shorter than a graph with arcs');
    eq(chain.edges.length, 4);
  },
  no_text_is_wider_than_its_node() {
    const s = G.buildSvg(voting, { containers: healthy, experiment: {} });
    const widest = Math.max(...[...s.matchAll(/class="(?:wrn|sub|nm)"[^>]*>([^<]*)</g)].map(m => m[1].length));
    yes(widest * 6.4 <= G.layout(voting).pos.vote.w, 'the longest line (' + widest + ' chars) fits a ' + G.layout(voting).pos.vote.w + ' px node');
  },
  non_adjacent_dependencies_use_arcs_that_alternate_over_and_under() {
    const g = { critical_path: ['a', 'b', 'c', 'd'], services: { a: svc('a', { depends_on: ['c', 'd'] }), b: svc('b'), c: svc('c'), d: svc('d') } };
    const L = G.layout(g);
    yes(L.edges.every(e => e.d.includes('Q')), 'both edges skip a node, so both are arcs');
    const sy = L.edges.map(e => Number(e.d.split(' ')[0].split(',')[1]));
    yes(sy[0] !== sy[1], 'one arc goes over the row and the next under it');
  },
  dependencies_not_in_the_graph_are_skipped_instead_of_crashing() {
    const g = { critical_path: ['a'], services: { a: svc('a', { depends_on: ['missing'] }) } };
    eq(G.layout(g).edges.length, 0);
  },
  blast_radius_of_each_voting_service() {
    eq([...G.blastRadius(voting, 'redis')].sort(), ['vote', 'worker']);
    eq([...G.blastRadius(voting, 'db')].sort(), ['result', 'worker']);
    eq([...G.blastRadius(voting, 'worker')], [], 'nothing depends on the worker');
    eq([...G.blastRadius(voting, 'vote')], []);
  },
  blast_radius_is_transitive_and_survives_cycles() {
    const chain = { services: { a: svc('a', { depends_on: ['b'] }), b: svc('b', { depends_on: ['c'] }), c: svc('c') } };
    eq([...G.blastRadius(chain, 'c')].sort(), ['a', 'b']);
    const cyc = { services: { a: svc('a', { depends_on: ['b'] }), b: svc('b', { depends_on: ['a'] }) } };
    eq([...G.blastRadius(cyc, 'a')], ['b'], 'terminates and never includes the target');
  },
  health_mapping() {
    eq(G.nodeHealth(undefined), 'unknown');
    eq(G.nodeHealth({ state: 'exited' }), 'down');
    eq(G.nodeHealth({ state: 'paused', health: 'healthy' }), 'down');
    eq(G.nodeHealth({ state: 'running', health: 'unhealthy' }), 'down');
    eq(G.nodeHealth({ state: 'running', health: 'starting' }), 'warn');
    eq(G.nodeHealth({ state: 'running', health: 'healthy' }), 'ok');
    eq(G.nodeHealth({ state: 'running', health: null }), 'plain');
  },
  idle_picture_shows_every_service_its_warnings_and_the_critical_ring() {
    const s = G.buildSvg(voting, { containers: healthy, experiment: { phase: 'idle' } });
    ['vote', 'redis', 'worker', 'db', 'result'].forEach(n => yes(s.includes('>' + n + '</text>'), n + ' is drawn'));
    eq(count(s, 'class="gnode h-ok crit"'), 4, 'four healthy critical-path nodes');
    eq(count(s, 'class="gnode h-plain crit"'), 1, 'the worker runs but has no healthcheck');
    yes(count(s, '>no healthcheck<') === 1 && count(s, '>single replica<') === 5, 'warnings are shown, one short line each');
    yes(!s.includes('FAULT') && !s.includes('at risk') && !s.includes('restarted'), 'no fault markers while idle');
    yes(s.includes('stateful') && s.includes('user-facing'));
  },
  during_a_redis_fault_the_target_pulses_and_its_dependents_are_at_risk() {
    const st = { containers: healthy, experiment: { running: true, phase: 'fault', target: 'redis', fault: 'kill' } };
    const s = G.buildSvg(voting, st);
    eq(count(s, 'FAULT: kill'), 1);
    eq(count(s, 'at risk'), 2, 'vote and worker depend on redis');
    yes(/gnode h-ok crit fault/.test(s) && count(s, ' risk"') === 2);
    eq(count(s, 'class="gedge hot"'), 2, 'only the two edges into redis are highlighted');
    yes(!/FAULT: kill[\s\S]*FAULT: kill/.test(s));
  },
  baseline_phase_flags_nothing_because_no_fault_is_active_yet() {
    const s = G.buildSvg(voting, { containers: healthy, experiment: { running: true, phase: 'baseline', target: 'redis', fault: 'kill' } });
    yes(!s.includes('FAULT') && !s.includes('at risk') && !s.includes('class="gedge hot"'));
  },
  recovery_phase_marks_the_target_recovering_without_the_risk_badges() {
    const s = G.buildSvg(voting, { containers: healthy, experiment: { running: true, phase: 'recovery', target: 'db', fault: 'pause' } });
    eq(count(s, 'recovering'), 2, 'one class + one badge');
    yes(!s.includes('at risk') && !s.includes('FAULT'));
  },
  after_an_experiment_services_that_really_restarted_keep_a_badge() {
    const st = { containers: healthy, experiment: { running: false, phase: 'idle', target: 'db', fault: 'stop', result: { restarts: { worker: 1, result: 2 } } } };
    const s = G.buildSvg(voting, st);
    yes(s.includes('↻ restarted ×1') && s.includes('↻ restarted ×2'));
    eq(count(s, 'restarted'), 2);
    yes(s.includes('last fault: stop') && !s.includes('at risk') && !s.includes('class="gedge hot"'));
  },
  user_visible_errors_get_a_badge_on_the_affected_journey() {
    const s = G.buildSvg(voting, { containers: healthy, experiment: {}, errors: { vote: true, result: false } });
    eq(count(s, 'user errors'), 1);
  },
  down_and_missing_containers_are_coloured_honestly() {
    const c = Object.assign({}, healthy, { redis: { state: 'exited', health: null, restarts: 3 } });
    delete c.worker;
    const s = G.buildSvg(voting, { containers: c, experiment: {} });
    yes(/gnode h-down crit/.test(s), 'an exited redis is red');
    yes(/gnode h-unknown crit/.test(s), 'a service with no container data is grey, not assumed healthy');
  },
  hostile_names_are_escaped() {
    const g = { critical_path: ['<img src=x onerror=alert(1)>'], services: { '<img src=x onerror=alert(1)>': svc('<img src=x onerror=alert(1)>') } };
    const s = G.buildSvg(g, {});
    yes(!s.includes('<img'), 'no raw tag survives') && yes(s.includes('&lt;img'));
  },
  empty_or_missing_graphs_render_nothing_and_a_single_service_is_centred() {
    eq(G.buildSvg(null, {}), ''); eq(G.buildSvg({ services: {} }, {}), ''); eq(G.buildSvg({}, {}), '');
    const one = { critical_path: ['a'], services: { a: svc('a') } };
    eq(G.layout(one).pos.a.x, 450);
    yes(G.buildSvg(one, undefined).includes('>a</text>'));
  },
  render_sets_the_viewbox_and_the_markup() {
    const el = { attrs: {}, innerHTML: '', setAttribute(k, v) { this.attrs[k] = v; } };
    G.renderGraph(el, voting, { containers: healthy, experiment: {} });
    eq(el.attrs.viewBox, '0 0 900 208', 'a plain chain is compact'); yes(el.innerHTML.includes('marker-end="url(#garr)"'));
    G.renderGraph(null, voting, {});   // no element: must not throw
  },
};

const out = {};
for (const [name, fn] of Object.entries(cases)) {
  try { fn(); out[name] = 'ok'; } catch (e) { out[name] = String(e && e.message || e); }
}
console.log(JSON.stringify(out));
