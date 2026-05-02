"""Tests for cognicore.advanced_memory — SemanticMemory."""

from cognicore.advanced_memory import SemanticMemory


class TestSemanticMemoryStore:
    def test_store_single_entry(self):
        mem = SemanticMemory()
        mem.store({"text": "phishing email", "correct": False, "category": "security"})
        assert len(mem.entries) == 1

    def test_store_multiple_entries(self):
        mem = SemanticMemory()
        for i in range(5):
            mem.store({"text": f"entry {i}", "correct": True})
        assert len(mem.entries) == 5

    def test_stats_total_stored(self):
        mem = SemanticMemory()
        mem.store({"text": "hello", "correct": True})
        mem.store({"text": "world", "correct": False})
        stats = mem.stats()
        assert stats["total_stored"] == 2
        assert stats["total_entries"] == 2

    def test_max_size_eviction(self):
        mem = SemanticMemory(max_size=3)
        for i in range(5):
            mem.store({"text": f"entry {i}", "correct": True})
        assert len(mem.entries) <= 3


class TestSemanticMemoryRecall:
    def test_recall_basic(self):
        mem = SemanticMemory()
        mem.store({"text": "phishing email scam", "correct": False, "category": "security"})
        mem.store({"text": "cooking recipe pasta", "correct": True, "category": "cooking"})
        results = mem.recall("email fraud phishing", top_k=2)
        assert len(results) >= 1

    def test_recall_empty_memory(self):
        mem = SemanticMemory()
        results = mem.recall("anything")
        assert results == []

    def test_recall_top_k(self):
        mem = SemanticMemory()
        for i in range(10):
            mem.store({"text": f"document {i}", "correct": True})
        results = mem.recall("document", top_k=3)
        assert len(results) <= 3

    def test_semantic_search(self):
        mem = SemanticMemory()
        mem.store({"text": "malware attack script", "correct": False})
        mem.store({"text": "recipe for chocolate cake", "correct": True})
        results = mem.semantic_search("hacking malware")
        assert isinstance(results, list)


class TestSemanticMemoryDecay:
    def test_decay_lowers_relevance(self):
        mem = SemanticMemory(decay_rate=0.5)
        mem.store({"text": "old entry", "correct": True})
        mem.store({"text": "new entry", "correct": True})
        assert mem.entries[0]["_relevance"] < mem.entries[1]["_relevance"]

    def test_no_decay_equal_relevance(self):
        mem = SemanticMemory(decay_rate=1.0)
        mem.store({"text": "first entry", "correct": True})
        mem.store({"text": "second entry", "correct": True})
        # Both should have the same relevance with no decay
        assert mem.entries[0]["_relevance"] == mem.entries[1]["_relevance"]


class TestSemanticMemoryBestWorst:
    def test_best_actions(self):
        mem = SemanticMemory()
        mem.store({"text": "security hack", "correct": True, "predicted": "UNSAFE"})
        mem.store({"text": "security breach", "correct": False, "predicted": "SAFE"})
        best = mem.best_actions("security exploit")
        assert isinstance(best, list)

    def test_worst_actions(self):
        mem = SemanticMemory()
        mem.store({"text": "security hack", "correct": True, "predicted": "UNSAFE"})
        mem.store({"text": "security breach", "correct": False, "predicted": "SAFE"})
        worst = mem.worst_actions("security exploit")
        assert isinstance(worst, list)


class TestSemanticMemoryAdaptiveContext:
    def test_adaptive_context_learning_mode(self):
        mem = SemanticMemory()
        mem.store({"text": "test input", "correct": True})
        ctx = mem.get_adaptive_context("test", agent_accuracy=0.2)
        assert ctx["strategy"] == "learning_from_mistakes"

    def test_adaptive_context_reinforcing_mode(self):
        mem = SemanticMemory()
        mem.store({"text": "test input", "correct": True})
        ctx = mem.get_adaptive_context("test", agent_accuracy=0.9)
        assert ctx["strategy"] == "reinforcing_success"

    def test_adaptive_context_empty_memory(self):
        mem = SemanticMemory()
        ctx = mem.get_adaptive_context("test query", agent_accuracy=0.5)
        assert isinstance(ctx, dict)
        assert "strategy" in ctx
