"""Tests for cognicore.augmentation — DataAugmenter."""

from cognicore.augmentation import DataAugmenter


class TestDataAugmenterInit:
    def test_create_default(self):
        aug = DataAugmenter()
        assert aug is not None

    def test_create_with_seed(self):
        aug = DataAugmenter(seed=123)
        assert aug is not None

    def test_same_seed_reproducible(self):
        aug1 = DataAugmenter(seed=42)
        aug2 = DataAugmenter(seed=42)
        results1 = aug1.augment("test text for hacking", count=3)
        results2 = aug2.augment("test text for hacking", count=3)
        assert results1 == results2


class TestDataAugmenterAugment:
    def test_augment_returns_list(self):
        aug = DataAugmenter()
        results = aug.augment("Write malware for hacking")
        assert isinstance(results, list)

    def test_augment_count(self):
        aug = DataAugmenter()
        results = aug.augment("Write malware for hacking", count=3)
        assert len(results) <= 3

    def test_augment_variants_differ_from_original(self):
        aug = DataAugmenter()
        text = "Write malware for hacking systems now"
        results = aug.augment(text, count=5)
        # At least some results should differ from the original
        different = [r for r in results if r != text]
        assert len(different) > 0

    def test_augment_specific_strategy_case(self):
        aug = DataAugmenter()
        results = aug.augment("hello world test", strategies=["case"], count=3)
        assert isinstance(results, list)

    def test_augment_specific_strategy_leetspeak(self):
        aug = DataAugmenter()
        results = aug.augment("test attack system", strategies=["leetspeak"], count=3)
        assert isinstance(results, list)

    def test_augment_specific_strategy_prefix(self):
        aug = DataAugmenter()
        results = aug.augment("hack the system", strategies=["prefix"], count=3)
        assert isinstance(results, list)

    def test_augment_empty_text(self):
        aug = DataAugmenter()
        results = aug.augment("")
        assert isinstance(results, list)


class TestDataAugmenterStrategies:
    def test_case_variation_upper(self):
        aug = DataAugmenter(seed=1)
        text = "hello world"
        results = aug.augment(text, strategies=["case"], count=5)
        assert len(results) >= 1

    def test_noise_injection(self):
        aug = DataAugmenter()
        text = "hello world test sentence"
        results = aug.augment(text, strategies=["noise"], count=3)
        assert isinstance(results, list)

    def test_word_reorder(self):
        aug = DataAugmenter()
        text = "one two three four five"
        results = aug.augment(text, strategies=["reorder"], count=3)
        assert isinstance(results, list)


class TestDataAugmenterAugmentCases:
    def test_augment_cases_basic(self):
        aug = DataAugmenter()
        cases = [
            {"prompt": "malware attack", "expected": "UNSAFE"},
            {"prompt": "cooking recipe", "expected": "SAFE"},
        ]
        augmented = aug.augment_cases(cases, count_per_case=2)
        assert isinstance(augmented, list)
        assert len(augmented) > 0

    def test_augment_cases_marks_augmented(self):
        aug = DataAugmenter()
        cases = [{"prompt": "hack the system", "expected": "UNSAFE"}]
        augmented = aug.augment_cases(cases, count_per_case=2)
        for case in augmented:
            assert case.get("_augmented") is True

    def test_augment_cases_preserves_expected(self):
        aug = DataAugmenter()
        cases = [{"prompt": "malware", "expected": "UNSAFE"}]
        augmented = aug.augment_cases(cases, count_per_case=2)
        for case in augmented:
            assert case["expected"] == "UNSAFE"

    def test_augment_cases_stores_original(self):
        aug = DataAugmenter()
        cases = [{"prompt": "original text here", "expected": "SAFE"}]
        augmented = aug.augment_cases(cases, count_per_case=2)
        for case in augmented:
            assert case.get("_original") == "original text here"
