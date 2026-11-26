import time

import pytest
import requests

from landslide import net

URL = "https://ags.cuzk.gov.cz/arcgis2/rest/services/dmr5g/ImageServer"


class _FakeResponse:
    status_code = 200

    def raise_for_status(self):
        pass

    def json(self):
        return {"ok": True}


@pytest.fixture
def client(monkeypatch):
    c = net.Client()
    calls = []

    def fake_request(method, url, **kw):
        calls.append((time.monotonic(), method, url, kw))
        return _FakeResponse()

    monkeypatch.setattr(c.session, "request", fake_request)
    c.calls = calls
    return c


@pytest.mark.parametrize("key", ["outSR", "outsr", "OUTSR", " outSR "])
def test_outsr_rejected_in_params(client, key):
    with pytest.raises(net.ForbiddenRequest):
        client.get(URL, params={"f": "json", key: 4326})
    assert client.calls == []


def test_outsr_rejected_in_url_query_and_post(client):
    with pytest.raises(net.ForbiddenRequest):
        client.get(URL + "?f=json&outSR=4326")
    with pytest.raises(net.ForbiddenRequest):
        client.post(URL, data={"outSR": 5514})
    assert client.calls == []


@pytest.mark.parametrize("url", ["https://example.com/x", "http://ags.cuzk.gov.cz/x",
                                 "https://ags.cuzk.gov.cz.evil.org/x"])
def test_host_allowlist(client, url):
    with pytest.raises(net.ForbiddenRequest):
        client.get(url)


def test_user_agent_and_min_interval(client):
    assert client.session.headers["User-Agent"] == net.USER_AGENT
    for _ in range(3):
        client.get(URL, params={"f": "json"})
    t = [c[0] for c in client.calls]
    assert all(b - a >= net.MIN_INTERVAL_S - 1e-3 for a, b in zip(t, t[1:]))
    assert client.n_requests == 3


def test_interval_floor():
    with pytest.raises(ValueError):
        net.Client(min_interval_s=0.1)


def test_arcgis_error_body(client, monkeypatch):
    class Err(_FakeResponse):
        def json(self):
            return {"error": {"code": 400, "message": "bad"}}

    monkeypatch.setattr(client.session, "request", lambda *a, **k: Err())
    with pytest.raises(RuntimeError, match="ArcGIS error"):
        client.get_json(URL)


def test_retries_are_configured():
    adapter = net.Client().session.get_adapter(URL)
    assert isinstance(adapter, requests.adapters.HTTPAdapter)
    assert adapter.max_retries.total >= 3
    assert 500 in adapter.max_retries.status_forcelist
