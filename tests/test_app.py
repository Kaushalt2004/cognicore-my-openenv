"""Tests for cognicore.server.app — REST API endpoints (create_app)."""

import pytest

try:
    from fastapi.testclient import TestClient
    from cognicore.server.app import create_app

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

pytestmark = pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not installed")


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


class TestAppRootEndpoint:
    def test_root_returns_api_name(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "CogniCore API"

    def test_root_returns_version(self, client):
        resp = client.get("/")
        data = resp.json()
        assert "version" in data

    def test_root_returns_environment_count(self, client):
        resp = client.get("/")
        data = resp.json()
        assert "environments" in data
        assert data["environments"] > 0


class TestAppEnvListing:
    def test_list_envs_status_200(self, client):
        resp = client.get("/envs")
        assert resp.status_code == 200

    def test_list_envs_contains_safety_env(self, client):
        resp = client.get("/envs")
        ids = [e["id"] for e in resp.json()["environments"]]
        assert "SafetyClassification-v1" in ids

    def test_list_envs_contains_math_env(self, client):
        resp = client.get("/envs")
        ids = [e["id"] for e in resp.json()["environments"]]
        assert "MathReasoning-v1" in ids

    def test_list_envs_contains_code_env(self, client):
        resp = client.get("/envs")
        ids = [e["id"] for e in resp.json()["environments"]]
        assert "CodeDebugging-v1" in ids

    def test_list_envs_contains_conversation_env(self, client):
        resp = client.get("/envs")
        ids = [e["id"] for e in resp.json()["environments"]]
        assert "Conversation-v1" in ids

    def test_list_envs_contains_planning_env(self, client):
        resp = client.get("/envs")
        ids = [e["id"] for e in resp.json()["environments"]]
        assert "Planning-v1" in ids


class TestAppSessionLifecycle:
    def test_create_session(self, client):
        resp = client.post(
            "/envs/SafetyClassification-v1/create",
            json={"difficulty": "easy"},
        )
        assert resp.status_code == 200
        assert "session_id" in resp.json()

    def test_create_session_returns_env_id(self, client):
        resp = client.post(
            "/envs/SafetyClassification-v1/create",
            json={"difficulty": "easy"},
        )
        assert resp.json()["env_id"] == "SafetyClassification-v1"

    def test_create_invalid_env_returns_404(self, client):
        resp = client.post(
            "/envs/NonExistent-v1/create",
            json={"difficulty": "easy"},
        )
        assert resp.status_code == 404

    def test_reset_session(self, client):
        resp = client.post(
            "/envs/SafetyClassification-v1/create", json={"difficulty": "easy"}
        )
        sid = resp.json()["session_id"]
        resp = client.post(f"/sessions/{sid}/reset")
        assert resp.status_code == 200
        assert "observation" in resp.json()

    def test_step_session(self, client):
        resp = client.post(
            "/envs/SafetyClassification-v1/create", json={"difficulty": "easy"}
        )
        sid = resp.json()["session_id"]
        client.post(f"/sessions/{sid}/reset")
        resp = client.post(
            f"/sessions/{sid}/step",
            json={"action": {"classification": "SAFE"}},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "reward" in data
        assert "done" in data

    def test_delete_session(self, client):
        resp = client.post(
            "/envs/SafetyClassification-v1/create", json={"difficulty": "easy"}
        )
        sid = resp.json()["session_id"]
        resp = client.delete(f"/sessions/{sid}")
        assert resp.status_code == 200
        # Verify it's gone
        resp = client.get(f"/sessions/{sid}/state")
        assert resp.status_code == 404

    def test_step_nonexistent_session_returns_404(self, client):
        resp = client.post(
            "/sessions/nonexistent-session-id/step",
            json={"action": {"classification": "SAFE"}},
        )
        assert resp.status_code == 404


class TestAppMultipleDifficulties:
    def test_create_easy_session(self, client):
        resp = client.post(
            "/envs/MathReasoning-v1/create", json={"difficulty": "easy"}
        )
        assert resp.status_code == 200

    def test_create_medium_session(self, client):
        resp = client.post(
            "/envs/MathReasoning-v1/create", json={"difficulty": "medium"}
        )
        assert resp.status_code == 200

    def test_create_hard_session(self, client):
        resp = client.post(
            "/envs/MathReasoning-v1/create", json={"difficulty": "hard"}
        )
        assert resp.status_code == 200
