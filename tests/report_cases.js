// Test cases for chaosai/static/report.js, run by tests/test_report_js.py with Node.
// Prints one JSON object {case_name: "ok" | "<failure message>"}.
const R = require('../chaosai/static/report.js');

const eq = (a, b, msg) => { if (JSON.stringify(a) !== JSON.stringify(b)) throw new Error((msg || 'not equal') + ': ' + JSON.stringify(a) + ' vs ' + JSON.stringify(b)); };
const yes = (c, msg) => { if (!c) throw new Error(msg || 'assertion failed'); };
const count = (s, sub) => s.split(sub).length - 1;

const comp = (name, weight, score, basis) => ({ name, weight, score, basis });
const exp = (id, o) => Object.assign({ id, target: 'redis', fault: 'kill', score: 99.2, band: 'good', worst_journey: 'vote', worst_error_rate: 0.02,
  recovered: true, recovery_s: 3, aborted: false, abort_reason: null, restarts: { redis: 1 }, missing_signals: [], rollback_errors: [],
  components: [comp('availability', 40, 0.98, 'worst journey vote: error rate up by 2%'), comp('latency', 20, null, 'no p95 to compare')] }, o || {});
const report = () => ({
  analysis: { by: 'rules' },
  score: { score: 81.4, band: 'fair', n_experiments: 14, n_runs: 30, formula: 'per experiment: 40 + 20 + 25 + 15',
           weights: { availability: 40, latency: 20, recovery: 25, blast_radius: 15 },
           by_component: { availability: 79.2, latency: 61, recovery: 90, blast_radius: null } },
  experiments: [exp('older'), exp('newer', { fault: 'stop', score: 51.5, band: 'poor', aborted: true, abort_reason: 'error rate 100% > 50%', recovered: false })],
  weaknesses: [{ n: 1, title: 'Single worker', severity: 'high', services: ['worker', 'redis'], description: 'one worker',
                 evidence: [{ experiment_id: 'older', claim: 'queue peaked at 309' }] }],
  fixes: [{ title: 'More workers', addresses: ['Single worker'], rationale: 'share the load', change: 'worker:\n  replicas: 3\n',
            verification: [{ command: 'chaos run vote traffic-spike -p multiplier=5', score_before: 75, pass_if: 'score goes up' }] }],
});

const cases = {
  score_shows_the_number_the_band_and_every_component() {
    const h = R.scoreHtml(report());
    yes(h.includes('81.4') && h.includes('>fair<'), 'number and band');
    yes(h.includes('class="big warn"'), 'fair is amber');
    ['availability', 'latency', 'recovery', 'blast radius'].forEach(n => yes(h.includes(n), n));
    yes(h.includes('14 distinct experiments') && h.includes('30 runs stored') && h.includes('deterministic rules'));
    yes(h.includes('per experiment: 40 + 20 + 25 + 15'), 'the formula is shown');
  },
  a_component_with_no_data_shows_na_and_an_empty_bar() {
    const h = R.scoreHtml(report());
    yes(h.includes('<span class="bv">n/a</span>') && h.includes('style="width:0%"'), 'blast radius has no data');
    yes(h.includes('style="width:79.2%"'));
  },
  band_colours() {
    eq([R.tone('good'), R.tone('fair'), R.tone('poor'), R.tone(null)], ['ok', 'warn', 'bad', 'mut']);
    const r = report(); r.score.band = 'good'; yes(R.scoreHtml(r).includes('class="big ok"'));
  },
  llm_analysis_is_labelled() {
    const r = report(); r.analysis.by = 'llm'; yes(R.scoreHtml(r).includes('LLM agent'));
  },
  empty_and_unscored_reports_say_so_instead_of_inventing_a_score() {
    yes(R.scoreHtml({ empty: true }).includes('No experiment has been run yet'));
    yes(R.scoreHtml(null).includes('No experiment has been run yet'));
    const r = report(); r.score.score = null; yes(R.scoreHtml(r).includes('No score'));
    eq([R.historyHtml({ empty: true }), R.findingsHtml({ empty: true }), R.cardHtml({ empty: true }), R.cardHtml(r)], ['', '', '', '']);
  },
  history_lists_newest_first_with_a_hidden_breakdown_row_per_experiment() {
    const h = R.historyHtml(report());
    yes(h.indexOf('newer') < h.indexOf('older'), 'newest first');
    eq(count(h, 'class="hrow"'), 2); eq(count(h, 'class="hdetail"'), 2); eq(count(h, ' hidden>'), 2);
    yes(h.includes('data-id="newer"') && h.includes('worst journey vote: error rate up by 2%'), 'breakdown carries the basis text');
    yes(h.includes('not recovered') && h.includes('error rate 100% &gt; 50%') && h.includes('redis &times;1'));
  },
  missing_signals_and_rollback_errors_are_surfaced_in_the_breakdown() {
    const r = report(); r.experiments[0].missing_signals = ['db_votes: unreadable']; r.experiments[0].rollback_errors = ['limit not restored'];
    const h = R.historyHtml(r);
    yes(h.includes('missing: db_votes: unreadable') && h.includes('rollback error: limit not restored'));
  },
  findings_show_severity_evidence_and_the_fix_with_its_verification() {
    const h = R.findingsHtml(report());
    yes(h.includes('class="sev high">HIGH<') && h.includes('1. Single worker') && h.includes('queue peaked at 309') && h.includes('<code>older</code>'));
    yes(h.includes('More workers') && h.includes('replicas: 3') && h.includes('chaos run vote traffic-spike -p multiplier=5') && h.includes('was 75.0'));
    yes(!h.includes('replicas: 3\n</pre>'), 'trailing whitespace is trimmed');
  },
  hostile_text_from_results_or_the_analyzer_is_escaped_everywhere() {
    const bad = '<img src=x onerror=alert(1)>';
    const r = report();
    r.experiments[0] = exp(bad, { target: bad, fault: bad, abort_reason: bad, aborted: true, restarts: { [bad]: 2 },
      components: [comp(bad, 1, 0.5, bad)], missing_signals: [bad], rollback_errors: [bad] });
    r.weaknesses[0] = { n: 1, title: bad, severity: bad, services: [bad], description: bad, evidence: [{ experiment_id: bad, claim: bad }] };
    r.fixes[0] = { title: bad, addresses: [bad], rationale: bad, change: '</pre><script>x</script>', verification: [{ command: bad, score_before: 1, pass_if: bad }] };
    r.score.formula = bad; r.score.band = bad;
    const all = R.scoreHtml(r) + R.historyHtml(r) + R.findingsHtml(r);
    yes(!all.includes('<img') && !all.includes('<script>') && !all.includes('</pre><script'), 'no raw markup survives');
    yes(all.includes('&lt;img src=x onerror=alert(1)&gt;') && all.includes('&lt;/pre&gt;&lt;script&gt;'));
  },
  fix_validation_shows_the_before_and_after_table_the_verdict_and_the_sandbox_state() {
    const r = report();
    r.fix_validations = [{ status: 'improved', title: 'Run three worker replicas', addresses: 'Single worker', detail: 'the score rose +3.3 points',
      change: 'docker compose up -d --scale worker=3 worker', claim: 'resilience score 93.8 -> 97.1', kept: false, reverted: true,
      before: { score: 93.8 }, after: { score: 97.1 },
      comparison: [{ label: 'redis queue peak', unit: 'votes', before: 109, after: 9, change_pct: -91.7, verdict: 'better' },
                   { label: 'worst journey error rate', unit: '%', before: 0, after: 0, change_pct: 0, verdict: 'same' },
                   { label: 'recovery time', unit: 's', before: 12, after: 40, change_pct: 233, verdict: 'worse' }] }];
    const h = R.fixesHtml(r);
    yes(h.includes('class="sev low">IMPROVED<') && h.includes('Run three worker replicas') && h.includes('measured improvement') && h.includes('addresses: Single worker'));
    yes(h.includes('redis queue peak (votes)') && h.includes('<td>109.0</td><td>9.0</td>') && h.includes('-92% better') && h.includes('class="ok"'), 'the metric row');
    yes(h.includes('<b>resilience score</b></td><td>93.8</td><td>97.1</td>') && h.includes('+233% worse') && h.includes('class="bad"'), 'score row and a regression shown as worse');
    yes(h.includes('0% same') && h.includes('resilience score 93.8 -&gt; 97.1') && h.includes('sandbox afterwards: reverted to the original'));
  },
  fix_validation_words_each_status_and_each_sandbox_state_honestly() {
    const fx = (o) => Object.assign({ status: 'improved', title: 't', detail: 'd', change: 'c', comparison: [], reverted: false, kept: false }, o);
    const html = (o) => R.fixesHtml({ fix_validations: [fx(o)], score: {} });
    yes(html({ status: 'inconclusive' }).includes('no measurable improvement (not claimed)') && html({ status: 'inconclusive' }).includes('class="sev medium">INCONCLUSIVE<'));
    yes(html({ status: 'worse' }).includes('made things worse') && html({ status: 'worse' }).includes('class="sev high">WORSE<'));
    yes(html({ status: 'error' }).includes('class="sev high">ERROR<') && html({ status: 'skipped' }).includes('skipped, nothing changed'));
    yes(html({ kept: true }).includes('kept as changed') && html({}).includes('sandbox afterwards: not changed') && html({ reverted: true }).includes('reverted to the original'));
  },
  fix_validation_without_any_fix_explains_what_the_fixing_agent_does() {
    ['', { empty: true }, null, { fix_validations: [] }].forEach(function (rep) {
      const h = R.fixesHtml(rep);
      yes(h.includes('No fix has been validated yet') && h.includes('chaos fix') && h.includes('chaos pipeline') && h.includes('puts the sandbox back'));
    });
  },
  fix_validation_escapes_everything_it_shows() {
    const bad = '<img src=x onerror=alert(1)>';
    const h = R.fixesHtml({ fix_validations: [{ status: bad, title: bad, addresses: bad, detail: bad, change: bad, claim: bad, kept: false, reverted: true,
      before: { score: 1 }, after: { score: 2 }, comparison: [{ label: bad, unit: bad, before: 1, after: 2, change_pct: 100, verdict: bad }] }] });
    yes(!h.includes('<img') && !h.includes('<script'), 'no raw markup survives');
    yes(h.includes('&lt;img src=x onerror=alert(1)&gt;'));
  },
  card_summarises_the_score_for_the_header_row() {
    const c = R.cardHtml(report());
    eq([c.label, c.value, c.unit, c.tone, c.detail], ['Resilience score', '81.4', '/ 100', 'warn', 'fair']);
  },
};

const out = {};
for (const [name, fn] of Object.entries(cases)) {
  try { fn(); out[name] = 'ok'; } catch (e) { out[name] = String(e && e.message || e); }
}
console.log(JSON.stringify(out));
