"""Tests for CogniCore Memory middleware."""

import os
import tempfile

from cognicore.middleware.memory import Memory


class TestMemoryBasics:
    """Basic store/retrieve functionality."""

    def test_store_and_retrieve(self):
        mem = Memory(similarity_key="category")
        mem.store({"category": "math", "predicted": "42", "correct": True})
        mem.store({"category": "math", "predicted": "43", "correct": False})
        mem.store({"category": "code", "predicted": "pass", "correct": True})

        results = mem.retrieve("math")
        assert len(results) == 2
        assert results[0]["predicted"] == "43"  # most recent first

    def test_retrieve_top_k(self):
        mem = Memory()
        for i in range(10):
            mem.store({"category": "a", "correct": True, "i": i})

        results = mem.retrieve("a", top_k=3)
        assert len(results) == 3
        assert results[0]["i"] == 9

    def test_retrieve_successes_and_failures(self):
        mem = Memory(similarity_key="category")
        mem.store({"category": "x", "correct": True, "id": 1})
        mem.store({"category": "x", "correct": False, "id": 2})
        mem.store({"category": "x", "correct": True, "id": 3})

        successes = mem.retrieve_successes("x")
        assert len(successes) == 2
        assert all(s["correct"] for s in successes)

        failures = mem.retrieve_failures("x")
        assert len(failures) == 1
        assert not failures[0]["correct"]

    def test_get_context(self):
        mem = Memory()
        mem.store({"category": "test", "correct": True, "secret": "data"}, episode=1)

        ctx = mem.get_context("test")
        assert len(ctx) == 1
        # Should strip internal fields
        assert "episode" not in ctx[0]
        assert "_timestamp" not in ctx[0]
        # Should keep user fields
        assert ctx[0]["correct"] is True

    def test_has_seen_group(self):
        mem = Memory()
        assert not mem.has_seen_group("new_group")
        mem.store({"category": "new_group", "correct": True})
        assert mem.has_seen_group("new_group")


class TestMemoryEviction:
    """FIFO eviction when max size is reached."""

    def test_fifo_eviction(self):
        mem = Memory(max_size=3)
        for i in range(5):
            mem.store({"category": "a", "correct": True, "i": i})

        assert len(mem.entries) == 3
        assert mem.entries[0]["i"] == 2  # oldest remaining


class TestMemoryPersistence:
    """JSON save/load."""

    def test_save_and_load(self):
        mem = Memory()
        mem.store({"category": "saved", "correct": True, "data": "hello"})
        mem.store({"category": "saved", "correct": False, "data": "world"})

        # Save to temp file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            path = f.name

        try:
            mem.save(path)

            # Load into new instance
            mem2 = Memory()
            mem2.load(path)

            assert len(mem2.entries) == 2
            assert mem2.entries[0]["data"] == "hello"
        finally:
            os.unlink(path)

    def test_load_nonexistent(self):
        mem = Memory()
        mem.load("nonexistent_file.json")  # Should not raise
        assert len(mem.entries) == 0


class TestMemoryStats:
    def test_stats(self):
        mem = Memory()
        mem.store({"category": "a", "correct": True})
        mem.store({"category": "a", "correct": False})
        mem.store({"category": "b", "correct": True})

        stats = mem.stats()
        assert stats["total_entries"] == 3
        assert stats["successes"] == 2
        assert stats["failures"] == 1
        assert abs(stats["success_rate"] - 2 / 3) < 0.01
        assert set(stats["groups"]) == {"a", "b"}
