import json

import pytest
from fastapi.testclient import TestClient
from pytest_bdd import given, parsers, scenario, then, when

from aichat.serve import api_server
from aichat.serve.api_server import app

FEATURE = "features/health.feature"


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@scenario(FEATURE, "Ready endpoint reports ready after startup")
def test_ready_after_startup():
    pass


@scenario(FEATURE, "Ready endpoint reports not ready before startup completes")
def test_ready_not_ready():
    pass


@scenario(FEATURE, "Info endpoint reports the running version")
def test_info_version():
    pass


@given("the gateway has fully started")
def _(client):
    return client


@given("the gateway has not finished starting")
def _(client):
    app.state.ready = False
    return client


@given(parsers.parse('the gateway reports version "{version}"'))
def _(version, monkeypatch):
    monkeypatch.setattr(api_server, "VERSION", version)


@when(parsers.parse("the client requests {method} {path}"), target_fixture="response")
def _(client, method, path):
    return client.request(method, path)


@then(parsers.parse("the response status is {code:d}"))
def _(response, code):
    assert response.status_code == code


@then("the response body is")
def _(response, docstring):
    assert response.json() == json.loads(docstring)
