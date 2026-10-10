"""Dashboard routes for the results panels: /api/report, the downloads, /report.js, and the page wiring."""
import http.client
import json

import pytest

from chaosai import report
from tests.fixtures_results import make_results
from tests.test_dashboard import call, serve, stop, web  # noqa: F401  (web is a fixture)


def raw_get(port, path, host=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    c.putrequest("GET", path, skip_host=host is not None)
    if host is not None:
        c.putheader("Host", host)
    c.endheaders()
    r = c.getresponse()
    return r.status, {k.lower(): v for k, v in r.getheaders()}, r.read().decode()


@pytest.fixture
def live(tmp_path):
    """A real report built by report.Live from a stored-results folder (the same path `chaos dashboard` uses)."""
    (tmp_path / "fx").mkdir()
    d = make_results(tmp_path / "fx")
    graph = json.loads((d / "graph.json").read_text(encoding="utf-8"))
    return report.Live(d, None, lambda: graph)


def test_report_endpoint_serves_the_findings(tmp_path, live):
    server, orch = serve(tmp_path / "w", report_fn=live.get)
    try:
        status, rep = call(server.server_port, "GET", "/api/report")
        assert status == 200 and rep["score"]["score"] is not None and len(rep["experiments"]) == 2
        assert rep["weaknesses"] and rep["fixes"] and rep["analysis"]["by"] == "rules"
    finally:
        stop(server)


def test_report_endpoint_with_no_experiments_says_empty_not_error(tmp_path):
    server, orch = serve(tmp_path, report_fn=lambda: {"empty": True})
    try:
        assert call(server.server_port, "GET", "/api/report") == (200, {"empty": True})
        status, headers, body = raw_get(server.server_port, "/report.html")
        assert status == 404 and "no experiments yet" in body
    finally:
        stop(server)


def test_report_endpoint_reports_an_unavailable_builder_without_crashing(tmp_path):
    server, orch = serve(tmp_path)
    try:
        status, body = call(server.server_port, "GET", "/api/report")
        assert status == 503 and "not available" in body["error"]
        assert raw_get(server.server_port, "/report.md")[0] == 503
    finally:
        stop(server)


def test_report_endpoint_reports_a_failing_builder_without_crashing(tmp_path):
    def boom():
        raise RuntimeError("results folder unreadable")
    server, orch = serve(tmp_path, report_fn=boom)
    try:
        status, body = call(server.server_port, "GET", "/api/report")
        assert status == 503 and "results folder unreadable" in body["error"]
    finally:
        stop(server)


def test_downloads_are_attachments_with_the_right_type_and_content(tmp_path, live):
    server, orch = serve(tmp_path / "w", report_fn=live.get)
    try:
        port = server.server_port
        s, h, body = raw_get(port, "/report.html")
        assert s == 200 and h["content-type"].startswith("text/html") and 'filename="chaosai-findings.html"' in h["content-disposition"]
        assert body.startswith("<!doctype html>") and "Resilience findings report" in body
        s, h, body = raw_get(port, "/report.md")
        assert s == 200 and h["content-type"].startswith("text/markdown") and h["content-disposition"].endswith('chaosai-findings.md"')
        assert body.startswith("# Resilience findings report")
        s, h, body = raw_get(port, "/report.json")
        assert s == 200 and h["content-type"] == "application/json" and json.loads(body)["score"]["score"] is not None
    finally:
        stop(server)


def test_the_downloaded_json_matches_the_api(tmp_path, live):
    server, orch = serve(tmp_path / "w", report_fn=live.get)
    try:
        port = server.server_port
        assert json.loads(raw_get(port, "/report.json")[2]) == call(port, "GET", "/api/report")[1]
    finally:
        stop(server)


def test_report_routes_enforce_the_same_host_check(web):  # noqa: F811
    for path in ("/api/report", "/report.html", "/report.md", "/report.json", "/report.js"):
        assert call(web.port, "GET", path, host="evil.example")[0] == 403, path


def test_page_wires_the_results_panels_to_the_report_script_and_endpoint(web):  # noqa: F811
    status, page = call(web.port, "GET", "/")
    assert status == 200 and 'src="/report.js"' in page and "/api/report" in page
    for needle in ('id="r_score"', 'id="r_hist"', 'id="r_find"', 'id="r_fix"', "ChaosReport.fixesHtml", 'href="/report.html"', 'href="/report.md"', 'href="/report.json"',
                   "ChaosReport.scoreHtml", "ChaosReport.historyHtml", "ChaosReport.findingsHtml", "ChaosReport.cardHtml"):
        assert needle in page, needle
    s, h, js = raw_get(web.port, "/report.js")
    assert s == 200 and h["content-type"].startswith("application/javascript") and "scoreHtml" in js and "findingsHtml" in js
