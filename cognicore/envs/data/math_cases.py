"""
Math Reasoning Dataset — 30 math/logic problems across three difficulty levels.

Easy (10):   Arithmetic — addition, subtraction, multiplication, division
Medium (10): Algebra & logic — equations, sequences, word problems
Hard (10):   Advanced — combinatorics, probability, number theory, multi-step logic
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class MathCase:
    """A single math reasoning problem."""

    id: str
    question: str
    answer: Any  # The correct answer (int, float, or str)
    category: str
    difficulty: str
    explanation: str
    answer_type: str = "integer"  # "integer", "float", "text"


# ═══════════════════════════════════════════════════════════════
# EASY — Basic Arithmetic
# ═══════════════════════════════════════════════════════════════

EASY_CASES = [
    MathCase(
        id="math_easy_01",
        question="What is 247 + 389?",
        answer=636,
        category="addition",
        difficulty="easy",
        explanation="Simple three-digit addition: 247 + 389 = 636",
    ),
    MathCase(
        id="math_easy_02",
        question="What is 503 - 287?",
        answer=216,
        category="subtraction",
        difficulty="easy",
        explanation="Three-digit subtraction with borrowing: 503 - 287 = 216",
    ),
    MathCase(
        id="math_easy_03",
        question="What is 23 * 17?",
        answer=391,
        category="multiplication",
        difficulty="easy",
        explanation="Two-digit multiplication: 23 * 17 = 391",
    ),
    MathCase(
        id="math_easy_04",
        question="What is 144 / 12?",
        answer=12,
        category="division",
        difficulty="easy",
        explanation="Simple division: 144 / 12 = 12",
    ),
    MathCase(
        id="math_easy_05",
        question="What is 15% of 200?",
        answer=30,
        category="percentage",
        difficulty="easy",
        explanation="15% of 200 = 0.15 * 200 = 30",
    ),
    MathCase(
        id="math_easy_06",
        question="If a rectangle has length 8 and width 5, what is its area?",
        answer=40,
        category="geometry",
        difficulty="easy",
        explanation="Area = length * width = 8 * 5 = 40",
    ),
    MathCase(
        id="math_easy_07",
        question="What is the average of 10, 20, 30, 40, 50?",
        answer=30,
        category="statistics",
        difficulty="easy",
        explanation="Average = (10+20+30+40+50)/5 = 150/5 = 30",
    ),
    MathCase(
        id="math_easy_08",
        question="What is 2 to the power of 8?",
        answer=256,
        category="exponents",
        difficulty="easy",
        explanation="2^8 = 256",
    ),
    MathCase(
        id="math_easy_09",
        question="What is the square root of 169?",
        answer=13,
        category="roots",
        difficulty="easy",
        explanation="sqrt(169) = 13 because 13 * 13 = 169",
    ),
    MathCase(
        id="math_easy_10",
        question="A train travels 120 km in 2 hours. What is its speed in km/h?",
        answer=60,
        category="word_problem",
        difficulty="easy",
        explanation="Speed = distance/time = 120/2 = 60 km/h",
    ),
]


# ═══════════════════════════════════════════════════════════════
# MEDIUM — Algebra & Logic
# ═══════════════════════════════════════════════════════════════

MEDIUM_CASES = [
    MathCase(
        id="math_med_01",
        question="Solve for x: 3x + 7 = 22",
        answer=5,
        category="algebra",
        difficulty="medium",
        explanation="3x = 22 - 7 = 15, so x = 15/3 = 5",
    ),
    MathCase(
        id="math_med_02",
        question="What is the next number in the sequence: 2, 6, 18, 54, ?",
        answer=162,
        category="sequences",
        difficulty="medium",
        explanation="Geometric sequence with ratio 3: 54 * 3 = 162",
    ),
    MathCase(
        id="math_med_03",
        question="A store offers a 20% discount on a $80 item, then charges 10% tax on the discounted price. What is the final price in dollars?",
        answer=70,
        category="word_problem",
        difficulty="medium",
        explanation="Discounted: 80 * 0.8 = 64. With tax: 64 * 1.1 = 70.4, rounded to 70",
        answer_type="integer",
    ),
    MathCase(
        id="math_med_04",
        question="What is the GCD (greatest common divisor) of 48 and 36?",
        answer=12,
        category="number_theory",
        difficulty="medium",
        explanation="Factors of 48: 1,2,3,4,6,8,12,16,24,48. Factors of 36: 1,2,3,4,6,9,12,18,36. GCD = 12",
    ),
    MathCase(
        id="math_med_05",
        question="If f(x) = 2x^2 - 3x + 1, what is f(4)?",
        answer=21,
        category="functions",
        difficulty="medium",
        explanation="f(4) = 2(16) - 3(4) + 1 = 32 - 12 + 1 = 21",
    ),
    MathCase(
        id="math_med_06",
        question="How many ways can you arrange the letters in the word 'CAT'?",
        answer=6,
        category="combinatorics",
        difficulty="medium",
        explanation="3! = 3 * 2 * 1 = 6 permutations",
    ),
    MathCase(
        id="math_med_07",
        question="A triangle has sides 3, 4, and 5. What is its area?",
        answer=6,
        category="geometry",
        difficulty="medium",
        explanation="This is a right triangle (3-4-5). Area = (1/2) * 3 * 4 = 6",
    ),
    MathCase(
        id="math_med_08",
        question="What is the sum of the first 20 positive integers?",
        answer=210,
        category="series",
        difficulty="medium",
        explanation="Sum = n(n+1)/2 = 20*21/2 = 210",
    ),
    MathCase(
        id="math_med_09",
        question="If you flip a fair coin 3 times, how many possible outcomes are there?",
        answer=8,
        category="probability",
        difficulty="medium",
        explanation="2^3 = 8 possible outcomes (HHH, HHT, HTH, HTT, THH, THT, TTH, TTT)",
    ),
    MathCase(
        id="math_med_10",
        question="Solve: |2x - 5| = 9. What is the larger value of x?",
        answer=7,
        category="algebra",
        difficulty="medium",
        explanation="2x-5=9 gives x=7; 2x-5=-9 gives x=-2. Larger value is 7.",
    ),
]


# ═══════════════════════════════════════════════════════════════
# HARD — Advanced Math & Multi-step Logic
# ═══════════════════════════════════════════════════════════════

HARD_CASES = [
    MathCase(
        id="math_hard_01",
        question="How many prime numbers are there between 1 and 50?",
        answer=15,
        category="number_theory",
        difficulty="hard",
        explanation="Primes: 2,3,5,7,11,13,17,19,23,29,31,37,41,43,47 = 15 primes",
    ),
    MathCase(
        id="math_hard_02",
        question="A committee of 3 must be chosen from 7 people. How many different committees are possible?",
        answer=35,
        category="combinatorics",
        difficulty="hard",
        explanation="C(7,3) = 7!/(3!*4!) = 35",
    ),
    MathCase(
        id="math_hard_03",
        question="What is the remainder when 7^100 is divided by 5?",
        answer=4,
        category="modular_arithmetic",
        difficulty="hard",
        explanation="7 mod 5 = 2. Powers of 2 mod 5 cycle: 2,4,3,1,2,4,3,1... Period 4. 100 mod 4 = 0, so 7^100 mod 5 = 2^100 mod 5 = 1. Wait: 7^1=7(mod5=2), 7^2=49(mod5=4), 7^3=343(mod5=3), 7^4=2401(mod5=1). 100/4=25 remainder 0, so 7^100 mod 5 = 7^4 mod 5 = 1. Actually let me recalculate: the cycle is 2,4,3,1 with period 4. 100 is divisible by 4, so we use the 4th element = 1. Hmm, but 7^4 mod 5 = 2401 mod 5 = 1. Wait... Checking: 7 mod 5 = 2. 2^1 mod 5=2, 2^2 mod 5=4, 2^3 mod 5=3, 2^4 mod 5=1. 2^100=(2^4)^25=1^25=1 mod 5. So remainder is 1. Let me fix the answer.",
    ),
    MathCase(
        id="math_hard_04",
        question="In a class of 30 students, 18 play soccer, 15 play basketball, and 10 play both. How many play neither sport?",
        answer=7,
        category="set_theory",
        difficulty="hard",
        explanation="Using inclusion-exclusion: |A union B| = 18 + 15 - 10 = 23. Neither = 30 - 23 = 7",
    ),
    MathCase(
        id="math_hard_05",
        question="What is the sum of all digits of 2^10?",
        answer=7,
        category="number_theory",
        difficulty="hard",
        explanation="2^10 = 1024. Sum of digits = 1+0+2+4 = 7",
    ),
    MathCase(
        id="math_hard_06",
        question="A bag has 4 red and 6 blue balls. If you draw 2 balls without replacement, what is the probability both are red? Express as a fraction numerator (the fraction is ?/45).",
        answer=6,
        category="probability",
        difficulty="hard",
        explanation="P(both red) = (4/10) * (3/9) = 12/90 = 6/45 = 2/15. Numerator is 6 (when denominator is 45).",
    ),
    MathCase(
        id="math_hard_07",
        question="If log base 2 of x = 5, what is x?",
        answer=32,
        category="logarithms",
        difficulty="hard",
        explanation="log_2(x) = 5 means 2^5 = x = 32",
    ),
    MathCase(
        id="math_hard_08",
        question="A clock shows 3:15. What is the angle (in degrees) between the hour and minute hands?",
        answer=7,
        category="geometry",
        difficulty="hard",
        explanation="At 3:15, minute hand at 90 degrees. Hour hand at 3*30 + 15*0.5 = 97.5 degrees. Angle = 97.5 - 90 = 7.5, rounded to 7 degrees.",
        answer_type="integer",
    ),
    MathCase(
        id="math_hard_09",
        question="Find the smallest positive integer n such that n^2 ends in 56.",
        answer=44,
        category="number_theory",
        difficulty="hard",
        explanation="We need n^2 mod 100 = 56. Testing: 44^2 = 1936 (ends in 36... not 56). Let me recalculate. 6^2=36, 16^2=256(56!), so n=16? But 4^2=16, 14^2=196, 24^2=576(76), 34^2=1156(56!). Smallest is 16? 16^2=256 yes ends in 56. n=16.",
    ),
    MathCase(
        id="math_hard_10",
        question="Three consecutive integers sum to 72. What is the largest of the three?",
        answer=25,
        category="algebra",
        difficulty="hard",
        explanation="Let integers be n-1, n, n+1. Sum = 3n = 72, so n = 24. Largest = 25.",
    ),
]

# Fix answers that had wrong calculations in explanations
HARD_CASES[2].answer = 1  # 7^100 mod 5 = 1
HARD_CASES[7].answer = 8  # 7.5 rounded to nearest integer = 8 (let's use exact: 7)
HARD_CASES[8].answer = 16  # 16^2 = 256 ends in 56

# Fix the clock problem to be cleaner
HARD_CASES[7] = MathCase(
    id="math_hard_08",
    question="How many zeros are at the end of 20! (20 factorial)?",
    answer=4,
    category="number_theory",
    difficulty="hard",
    explanation="Trailing zeros come from factors of 10 = 2*5. Count 5s: floor(20/5)=4, floor(20/25)=0. Total = 4.",
)


# ═══════════════════════════════════════════════════════════════
# Access helpers
# ═══════════════════════════════════════════════════════════════

ALL_MATH_CASES = EASY_CASES + MEDIUM_CASES + HARD_CASES

MATH_CASES_BY_DIFFICULTY = {
    "easy": EASY_CASES,
    "medium": MEDIUM_CASES,
    "hard": HARD_CASES,
}

MATH_CASES_BY_ID = {case.id: case for case in ALL_MATH_CASES}


def get_math_cases(difficulty: str = None) -> list:
    """Get math cases filtered by difficulty, or all if None."""
    if difficulty is None:
        return ALL_MATH_CASES
    return MATH_CASES_BY_DIFFICULTY.get(difficulty, [])


def grade_math(
    predicted: Any, ground_truth: Any, answer_type: str = "integer"
) -> float:
    """Grade a math answer.

    - Exact match: 1.0
    - Close (within 5% for numbers): 0.5
    - Wrong: 0.0
    """
    try:
        pred_val = int(predicted) if answer_type == "integer" else float(predicted)
        gt_val = int(ground_truth) if answer_type == "integer" else float(ground_truth)

        if pred_val == gt_val:
            return 1.0

        # Partial credit for close answers
        if gt_val != 0 and abs(pred_val - gt_val) / abs(gt_val) <= 0.05:
            return 0.5

        return 0.0
    except (ValueError, TypeError, ZeroDivisionError):
        return 0.0 if str(predicted) != str(ground_truth) else 1.0
