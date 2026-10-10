/* Results panels for the chaos-ai dashboard: resilience score, experiment history, ranked findings.
 *
 * Pure functions that turn the report served at /api/report into HTML strings, so they can be tested with
 * Node; the page only assigns the strings to its panels. Every value goes through esc(): experiment ids,
 * claims and fix snippets come from stored data and the analyzer, so none of it is trusted as markup.
 * No external libraries, so it works offline.
 */
(function (root) {
  'use strict';

  function esc(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }
  function num(x, nd, unit) { return x == null ? 'n/a' : Number(x).toFixed(nd || 0) + (unit || ''); }
  function pct(x) { return x == null ? 'n/a' : Math.round(x * 100) + '%'; }
  function tone(band) { return band === 'good' ? 'ok' : band === 'fair' ? 'warn' : band === 'poor' ? 'bad' : 'mut'; }
  const LABEL = { availability: 'availability', latency: 'latency', recovery: 'recovery', blast_radius: 'blast radius' };

  function empty(rep) { return !rep || rep.empty || !rep.score; }

  /* ---- the score: a big number, four component bars, and how it was computed */
  function scoreHtml(rep) {
    if (empty(rep)) return '<div class="mut">No experiment has been run yet. Press a red button, or run <code>chaos autopilot</code>.</div>';
    const sc = rep.score;
    if (sc.score == null) return '<div class="mut">No score: nothing measurable in the stored experiments.</div>';
    const bars = Object.keys(sc.weights || {}).map(function (k) {
      const v = (sc.by_component || {})[k];
      return '<div class="bar"><span class="bl">' + esc(LABEL[k] || k) + ' <i>(' + esc(sc.weights[k]) + ')</i></span>' +
        '<span class="bt"><span class="bf ' + (v == null ? 'mut' : v >= 85 ? 'ok' : v >= 60 ? 'warn' : 'bad') +
        '" style="width:' + (v == null ? 0 : Math.max(0, Math.min(100, v))) + '%"></span></span>' +
        '<span class="bv">' + num(v, 0) + '</span></div>';
    }).join('');
    return '<div class="scorebox"><div><div class="big ' + tone(sc.band) + '">' + num(sc.score, 1) + '</div>' +
      '<div class="pillband ' + tone(sc.band) + '">' + esc(sc.band) + '</div></div><div class="bars">' + bars + '</div></div>' +
      '<div class="note">Latest run of each of ' + esc(sc.n_experiments) + ' distinct experiments (' + esc(sc.n_runs) + ' runs stored), ' +
      'weighted by how critical the broken service is. Analysis by ' +
      esc(rep.analysis && rep.analysis.by === 'llm' ? 'LLM agent' : 'deterministic rules') + '.</div>' +
      '<details class="note"><summary>How the score is computed</summary>' + esc(sc.formula) + '</details>';
  }

  /* ---- history: one row per stored experiment, newest first; click a row for the score breakdown */
  function historyHtml(rep) {
    if (empty(rep)) return '';
    const rows = rep.experiments.slice().reverse().map(function (e) {
      const rec = e.recovered ? num(e.recovery_s, 0, ' s') : 'not recovered';
      const worst = e.worst_journey ? pct(e.worst_error_rate) + ' (' + esc(e.worst_journey) + ')' : 'n/a';
      const restarts = Object.keys(e.restarts || {}).filter(function (k) { return e.restarts[k]; })
        .map(function (k) { return esc(k) + ' &times;' + esc(e.restarts[k]); }).join(', ') || '&ndash;';
      const parts = (e.components || []).map(function (c) {
        return '<li><b>' + esc(LABEL[c.name] || c.name) + '</b> (' + esc(c.weight) + '): ' + num(c.score, 2) + ' &mdash; ' + esc(c.basis) + '</li>';
      }).join('');
      const extra = (e.missing_signals || []).map(function (m) { return '<li class="warn">missing: ' + esc(m) + '</li>'; }).join('') +
        (e.rollback_errors || []).map(function (m) { return '<li class="bad">rollback error: ' + esc(m) + '</li>'; }).join('');
      return '<tr class="hrow" data-id="' + esc(e.id) + '" tabindex="0"><td><code>' + esc(e.id) + '</code></td><td>' + esc(e.target) + '</td><td>' +
        esc(e.fault) + '</td><td class="' + tone(e.band) + '"><b>' + num(e.score, 1) + '</b></td><td>' + worst + '</td><td>' + esc(rec) + '</td><td>' +
        (e.aborted ? '<span class="bad">yes: ' + esc(e.abort_reason) + '</span>' : 'no') + '</td><td>' + restarts + '</td></tr>' +
        '<tr class="hdetail" data-id="' + esc(e.id) + '" hidden><td colspan="8"><ul>' + parts + extra + '</ul></td></tr>';
    }).join('');
    return '<table class="hist"><thead><tr><th>experiment</th><th>target</th><th>fault</th><th>score</th><th>worst journey errors</th>' +
      '<th>recovery</th><th>aborted</th><th>restarts</th></tr></thead><tbody>' + rows + '</tbody></table>';
  }

  /* ---- findings: ranked weaknesses with their evidence, and the fixes with how to verify them */
  function findingsHtml(rep) {
    if (empty(rep)) return '';
    const ws = rep.weaknesses.map(function (w) {
      return '<div class="find"><div class="fh"><span class="sev ' + esc(w.severity) + '">' + esc(String(w.severity).toUpperCase()) + '</span> ' +
        '<b>' + esc(w.n) + '. ' + esc(w.title) + '</b></div><div class="note">affected: ' + esc((w.services || []).join(', ')) + '</div>' +
        '<div>' + esc(w.description) + '</div><ul class="ev">' +
        (w.evidence || []).map(function (x) { return '<li>' + esc(x.claim) + ' <code>' + esc(x.experiment_id) + '</code></li>'; }).join('') + '</ul></div>';
    }).join('');
    const fx = rep.fixes.map(function (f, i) {
      const v = (f.verification || []).map(function (x) {
        return '<li><code>' + esc(x.command) + '</code> &mdash; was ' + num(x.score_before, 1) + '; ' + esc(x.pass_if) + '</li>';
      }).join('');
      return '<div class="find"><div class="fh"><b>' + (i + 1) + '. ' + esc(f.title) + '</b></div>' +
        '<div class="note">addresses: ' + esc((f.addresses || []).join('; ') || '-') + '</div><div>' + esc(f.rationale) + '</div>' +
        '<pre>' + esc(String(f.change || '').replace(/\s+$/, '')) + '</pre>' + (v ? '<div class="note">Verify by re-running:</div><ul class="ev">' + v + '</ul>' : '') + '</div>';
    }).join('');
    return '<h3>Ranked weaknesses</h3>' + (ws || '<div class="mut">none</div>') + '<h3>Recommended fixes</h3>' + (fx || '<div class="mut">none</div>');
  }

  /* ---- fix validation: what the fixing agent changed, the same experiment before and after, and the measured verdict */
  const FIX_STATUS = { improved: 'measured improvement', inconclusive: 'no measurable improvement (not claimed)', worse: 'made things worse',
                       skipped: 'skipped, nothing changed', stopped: 'stopped', error: 'error', planned: 'planned' };
  function fixesHtml(rep) {
    const none = '<div class="mut">No fix has been validated yet. Run <code>chaos fix</code> or <code>chaos pipeline</code>: the fixing agent applies one ' +
      'vetted fix, runs the same experiment before and after, compares the two, and puts the sandbox back.</div>';
    if (!rep || rep.empty || !rep.fix_validations || !rep.fix_validations.length) return none;
    return rep.fix_validations.map(function (fx) {
      const tone = fx.status === 'improved' ? 'ok' : (fx.status === 'worse' || fx.status === 'error') ? 'bad' : 'warn';
      const rows = (fx.comparison || []).map(function (c) {
        const t = c.verdict === 'better' ? 'ok' : c.verdict === 'worse' ? 'bad' : 'mut';
        return '<tr><td>' + esc(c.label + (c.unit ? ' (' + c.unit + ')' : '')) + '</td><td>' + num(c.before, 1) + '</td><td>' + num(c.after, 1) + '</td><td class="' + t + '">' +
          (c.change_pct == null ? '' : esc((c.change_pct > 0 ? '+' : '') + Math.round(c.change_pct) + '%')) + ' ' + esc(c.verdict) + '</td></tr>';
      }).join('');
      const b = fx.before || {}, a = fx.after || {};
      const score = (b.score == null && a.score == null) ? '' : '<tr><td><b>resilience score</b></td><td>' + num(b.score, 1) + '</td><td>' + num(a.score, 1) + '</td><td></td></tr>';
      return '<div class="find"><div class="fh"><span class="sev ' + (tone === 'ok' ? 'low' : tone === 'bad' ? 'high' : 'medium') + '">' + esc(String(fx.status).toUpperCase()) + '</span> <b>' +
        esc(fx.title) + '</b></div><div class="note">' + esc(FIX_STATUS[fx.status] || fx.status) + (fx.addresses ? '; addresses: ' + esc(fx.addresses) : '') + '</div>' +
        '<div>' + esc(fx.detail) + '</div>' +
        (rows || score ? '<table class="hist"><thead><tr><th>metric</th><th>before</th><th>after</th><th>change</th></tr></thead><tbody>' + score + rows + '</tbody></table>' : '') +
        (fx.claim ? '<div class="note">' + esc(fx.claim) + '</div>' : '') +
        '<div class="note">change: <code>' + esc(fx.change) + '</code>; sandbox afterwards: ' +
        esc(fx.kept ? 'kept as changed' : fx.reverted ? 'reverted to the original' : 'not changed') + '</div></div>';
    }).join('');
  }

  /* ---- small badge for the header cards row */
  function cardHtml(rep) {
    if (empty(rep) || rep.score.score == null) return '';
    const sc = rep.score;
    return { label: 'Resilience score', value: num(sc.score, 1), unit: '/ 100', tone: tone(sc.band), detail: String(sc.band) };
  }

  const api = { esc: esc, scoreHtml: scoreHtml, historyHtml: historyHtml, findingsHtml: findingsHtml, fixesHtml: fixesHtml, cardHtml: cardHtml, tone: tone };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.ChaosReport = api;
})(typeof window !== 'undefined' ? window : this);
