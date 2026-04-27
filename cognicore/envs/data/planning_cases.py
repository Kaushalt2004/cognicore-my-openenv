"""
Multi-Step Planning Dataset — 30 planning problems across three difficulty levels.

Easy (10):   Sequential tasks with clear ordering — recipes, routines, instructions
Medium (10): Dependencies & constraints — project scheduling, resource allocation
Hard (10):   Complex constraints — optimization, conflicting goals, multi-agent coordination
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class PlanningCase:
    """A single multi-step planning problem."""

    id: str
    scenario: str
    constraints: List[str]
    correct_order: List[str]  # The correct step ordering (by step ID)
    steps: Dict[str, str]  # step_id -> description
    category: str
    difficulty: str
    explanation: str
    num_steps: int = 0

    def __post_init__(self):
        self.num_steps = len(self.correct_order)


# ═══════════════════════════════════════════════════════════════
# EASY — Clear Sequential Tasks
# ═══════════════════════════════════════════════════════════════

EASY_CASES = [
    PlanningCase(
        id="plan_easy_01",
        scenario="Make a cup of tea.",
        constraints=["Water must be boiled before pouring", "Teabag goes in cup before water"],
        correct_order=["A", "B", "C", "D", "E"],
        steps={
            "A": "Fill kettle with water",
            "B": "Boil the water",
            "C": "Place teabag in cup",
            "D": "Pour hot water into cup",
            "E": "Steep for 3-5 minutes and remove teabag",
        },
        category="sequential",
        difficulty="easy",
        explanation="Linear sequence with two dependencies: boil before pour, teabag before water.",
    ),
    PlanningCase(
        id="plan_easy_02",
        scenario="Get ready for work in the morning.",
        constraints=["Must shower before getting dressed", "Must eat before leaving", "Must be dressed before leaving"],
        correct_order=["A", "B", "C", "D", "E"],
        steps={
            "A": "Wake up and turn off alarm",
            "B": "Shower and brush teeth",
            "C": "Get dressed",
            "D": "Eat breakfast",
            "E": "Leave for work",
        },
        category="sequential",
        difficulty="easy",
        explanation="Morning routine with clear dependencies.",
    ),
    PlanningCase(
        id="plan_easy_03",
        scenario="Send a professional email.",
        constraints=["Must write subject before sending", "Must proofread before sending"],
        correct_order=["A", "B", "C", "D", "E"],
        steps={
            "A": "Open email client",
            "B": "Add recipient address",
            "C": "Write subject line",
            "D": "Compose email body",
            "E": "Proofread and send",
        },
        category="sequential",
        difficulty="easy",
        explanation="Straightforward email workflow.",
    ),
    PlanningCase(
        id="plan_easy_04",
        scenario="Plant a seed in a pot.",
        constraints=["Soil goes in pot before seed", "Water after planting"],
        correct_order=["A", "B", "C", "D"],
        steps={
            "A": "Choose pot and add drainage material",
            "B": "Fill pot with soil",
            "C": "Plant seed at correct depth",
            "D": "Water the soil gently",
        },
        category="sequential",
        difficulty="easy",
        explanation="Simple gardening sequence.",
    ),
    PlanningCase(
        id="plan_easy_05",
        scenario="Do laundry.",
        constraints=["Must wash before drying", "Must sort before washing"],
        correct_order=["A", "B", "C", "D", "E"],
        steps={
            "A": "Sort clothes by color and fabric",
            "B": "Load washing machine",
            "C": "Add detergent and start cycle",
            "D": "Transfer to dryer",
            "E": "Fold and put away",
        },
        category="sequential",
        difficulty="easy",
        explanation="Standard laundry procedure.",
    ),
    PlanningCase(
        id="plan_easy_06",
        scenario="Change a flat tire.",
        constraints=["Must loosen lug nuts before jacking up", "Must jack up before removing tire"],
        correct_order=["A", "B", "C", "D", "E", "F"],
        steps={
            "A": "Pull over safely and turn on hazards",
            "B": "Get spare tire and tools from trunk",
            "C": "Loosen lug nuts (don't remove yet)",
            "D": "Jack up the car",
            "E": "Remove flat tire and mount spare",
            "F": "Lower car and tighten lug nuts fully",
        },
        category="sequential",
        difficulty="easy",
        explanation="Critical safety procedure with specific ordering requirements.",
    ),
    PlanningCase(
        id="plan_easy_07",
        scenario="Charge and set up a new phone.",
        constraints=["Must charge before setup", "Must insert SIM before setup"],
        correct_order=["A", "B", "C", "D"],
        steps={
            "A": "Unbox and insert SIM card",
            "B": "Charge phone to at least 50%",
            "C": "Power on and follow setup wizard",
            "D": "Install apps and restore backup",
        },
        category="sequential",
        difficulty="easy",
        explanation="Device setup with prerequisite steps.",
    ),
    PlanningCase(
        id="plan_easy_08",
        scenario="Paint a room.",
        constraints=["Must tape before painting", "Must prime before painting color"],
        correct_order=["A", "B", "C", "D", "E"],
        steps={
            "A": "Move furniture and cover floor",
            "B": "Clean walls and repair holes",
            "C": "Apply painter's tape to edges",
            "D": "Apply primer coat",
            "E": "Apply two coats of paint color",
        },
        category="sequential",
        difficulty="easy",
        explanation="Room painting with preparation steps.",
    ),
    PlanningCase(
        id="plan_easy_09",
        scenario="File your tax return.",
        constraints=["Must gather documents before filling forms", "Must review before submitting"],
        correct_order=["A", "B", "C", "D"],
        steps={
            "A": "Gather W-2s, 1099s, and receipts",
            "B": "Choose filing method (software/accountant)",
            "C": "Fill in forms with income and deductions",
            "D": "Review, sign, and submit return",
        },
        category="sequential",
        difficulty="easy",
        explanation="Tax filing procedure.",
    ),
    PlanningCase(
        id="plan_easy_10",
        scenario="Write and submit a college application essay.",
        constraints=["Must outline before writing", "Must revise before submitting"],
        correct_order=["A", "B", "C", "D", "E"],
        steps={
            "A": "Read the essay prompt carefully",
            "B": "Brainstorm and create outline",
            "C": "Write first draft",
            "D": "Revise and proofread",
            "E": "Submit before deadline",
        },
        category="sequential",
        difficulty="easy",
        explanation="Essay writing workflow.",
    ),
]


# ═══════════════════════════════════════════════════════════════
# MEDIUM — Dependencies & Constraints
# ═══════════════════════════════════════════════════════════════

MEDIUM_CASES = [
    PlanningCase(
        id="plan_med_01",
        scenario="Cook Thanksgiving dinner (turkey, mashed potatoes, gravy, pie). Turkey takes 4 hours, pie takes 1 hour, potatoes take 30 min, gravy needs turkey drippings.",
        constraints=["Turkey must finish before gravy starts", "Pie can bake while turkey cooks", "Everything should be ready at the same time"],
        correct_order=["A", "B", "C", "D", "E", "F"],
        steps={
            "A": "Start turkey in oven (4 hours)",
            "B": "Prepare pie and bake (during turkey cook time)",
            "C": "Prepare potatoes (30 min before turkey done)",
            "D": "Remove turkey and let rest",
            "E": "Make gravy from turkey drippings",
            "F": "Serve everything together",
        },
        category="parallel_dependencies",
        difficulty="medium",
        explanation="Parallel tasks with a critical dependency: gravy needs turkey drippings.",
    ),
    PlanningCase(
        id="plan_med_02",
        scenario="Build a website. Frontend needs design first. Backend needs database first. Integration needs both.",
        constraints=["Design before frontend", "Database before backend", "Frontend AND backend before integration", "Testing after integration"],
        correct_order=["A", "B", "C", "D", "E", "F"],
        steps={
            "A": "Create UI/UX design mockups",
            "B": "Set up database schema",
            "C": "Build frontend from designs",
            "D": "Build backend API with database",
            "E": "Integrate frontend with backend",
            "F": "End-to-end testing",
        },
        category="project_management",
        difficulty="medium",
        explanation="Classic dependency graph: two parallel tracks merging for integration.",
    ),
    PlanningCase(
        id="plan_med_03",
        scenario="Move to a new apartment. Must give 30-day notice, pack, hire movers, transfer utilities.",
        constraints=["Sign new lease before giving notice", "Give notice 30 days before moving", "Pack before movers arrive", "Transfer utilities for move-in day"],
        correct_order=["A", "B", "C", "D", "E", "F"],
        steps={
            "A": "Find new apartment and sign lease",
            "B": "Give 30-day notice to current landlord",
            "C": "Transfer utilities to new address",
            "D": "Pack belongings room by room",
            "E": "Hire movers for moving day",
            "F": "Move day: movers transport, unpack essentials",
        },
        category="scheduling",
        difficulty="medium",
        explanation="Time-sequenced plan with a hard 30-day constraint.",
    ),
    PlanningCase(
        id="plan_med_04",
        scenario="Deploy a software update to production. Requires code review, staging test, backup, deploy, monitor.",
        constraints=["Code review before staging", "Staging must pass before production", "Backup before deploying", "Deploy during low-traffic window"],
        correct_order=["A", "B", "C", "D", "E", "F"],
        steps={
            "A": "Complete code review and get approval",
            "B": "Deploy to staging environment",
            "C": "Run automated tests on staging",
            "D": "Create production database backup",
            "E": "Deploy to production during maintenance window",
            "F": "Monitor metrics and error rates for 1 hour",
        },
        category="devops",
        difficulty="medium",
        explanation="CI/CD pipeline with safety gates.",
    ),
    PlanningCase(
        id="plan_med_05",
        scenario="Plan a wedding. Venue needed first, then caterer and decorations. Invitations need venue and date confirmed.",
        constraints=["Book venue before invitations", "Caterer needs guest count from RSVPs", "Decorations match venue", "Rehearsal before ceremony"],
        correct_order=["A", "B", "C", "D", "E", "F"],
        steps={
            "A": "Book venue and set date",
            "B": "Send invitations with RSVP deadline",
            "C": "Collect RSVPs and finalize guest count",
            "D": "Book caterer with final count",
            "E": "Arrange decorations for venue",
            "F": "Rehearsal dinner and ceremony day",
        },
        category="event_planning",
        difficulty="medium",
        explanation="Information dependencies: guest count needed for caterer, venue needed for invitations.",
    ),
    PlanningCase(
        id="plan_med_06",
        scenario="Conduct a scientific experiment. Need hypothesis, setup, data collection, analysis, paper.",
        constraints=["Hypothesis before experiment design", "IRB approval before data collection (human subjects)", "Analysis after all data collected", "Peer review before publication"],
        correct_order=["A", "B", "C", "D", "E", "F"],
        steps={
            "A": "Literature review and form hypothesis",
            "B": "Design experiment and get IRB approval",
            "C": "Set up equipment and recruit participants",
            "D": "Collect data according to protocol",
            "E": "Analyze data and draw conclusions",
            "F": "Write paper and submit for peer review",
        },
        category="research",
        difficulty="medium",
        explanation="Research pipeline with regulatory gate (IRB).",
    ),
    PlanningCase(
        id="plan_med_07",
        scenario="Renovate a bathroom. Plumbing must be done before tiling. Electrical before fixtures.",
        constraints=["Demolition first", "Plumbing before tiling", "Electrical before light fixtures", "Tiling before fixtures and vanity", "Inspection before final finishes"],
        correct_order=["A", "B", "C", "D", "E", "F"],
        steps={
            "A": "Demolish old bathroom",
            "B": "Rough plumbing and electrical work",
            "C": "Inspection of rough work",
            "D": "Install tiles (floor and walls)",
            "E": "Install vanity, toilet, and fixtures",
            "F": "Final touches: paint, mirrors, accessories",
        },
        category="construction",
        difficulty="medium",
        explanation="Construction sequence with inspection gate.",
    ),
    PlanningCase(
        id="plan_med_08",
        scenario="Launch a product. Need MVP, beta testing, marketing, and launch event.",
        constraints=["MVP before beta", "Beta feedback incorporated before launch", "Marketing campaign starts 2 weeks before launch", "Press release on launch day"],
        correct_order=["A", "B", "C", "D", "E", "F"],
        steps={
            "A": "Build minimum viable product",
            "B": "Beta test with select users",
            "C": "Incorporate beta feedback and finalize",
            "D": "Prepare marketing campaign and press materials",
            "E": "Start marketing campaign (2 weeks out)",
            "F": "Launch day: press release, public access, monitor",
        },
        category="product_launch",
        difficulty="medium",
        explanation="Product launch timeline with feedback loop and marketing timing.",
    ),
    PlanningCase(
        id="plan_med_09",
        scenario="Emergency disaster response. Earthquake has hit. Need to triage, rescue, shelter, supply.",
        constraints=["Assess damage first", "Rescue trapped people before supply distribution", "Set up shelter before weather turns", "Coordinate with federal aid"],
        correct_order=["A", "B", "C", "D", "E", "F"],
        steps={
            "A": "Assess damage and deploy first responders",
            "B": "Search and rescue operations",
            "C": "Set up emergency shelters",
            "D": "Establish medical triage stations",
            "E": "Distribute water, food, and supplies",
            "F": "Coordinate long-term recovery with federal agencies",
        },
        category="emergency_response",
        difficulty="medium",
        explanation="Emergency response priorities with time-critical ordering.",
    ),
    PlanningCase(
        id="plan_med_10",
        scenario="Onboard a new employee. Need IT setup, training, team intro, first project.",
        constraints=["IT setup before first day", "Orientation before training", "Mentor assigned before first project", "Check-in after 30 days"],
        correct_order=["A", "B", "C", "D", "E", "F"],
        steps={
            "A": "IT setup: laptop, email, access credentials",
            "B": "First day orientation and office tour",
            "C": "Introduce to team and assign mentor",
            "D": "Complete required training modules",
            "E": "Start first project with mentor guidance",
            "F": "30-day check-in and feedback session",
        },
        category="hr_process",
        difficulty="medium",
        explanation="Employee onboarding workflow with prerequisites.",
    ),
]


# ═══════════════════════════════════════════════════════════════
# HARD — Complex Constraints & Optimization
# ═══════════════════════════════════════════════════════════════

HARD_CASES = [
    PlanningCase(
        id="plan_hard_01",
        scenario="Schedule 5 tasks on 2 machines. Tasks have dependencies and different processing times. Minimize total completion time.",
        constraints=["Task B depends on A", "Task D depends on B and C", "Task E depends on D",
                      "Machine 1 is faster for compute tasks", "Machine 2 is faster for I/O tasks"],
        correct_order=["A", "C", "B", "D", "E"],
        steps={
            "A": "Data preprocessing (compute, 2h)",
            "C": "Load reference data (I/O, 1h) [can start immediately]",
            "B": "Feature extraction (compute, 3h) [needs A]",
            "D": "Model training (compute, 4h) [needs B and C]",
            "E": "Evaluation and report (I/O, 1h) [needs D]",
        },
        category="scheduling_optimization",
        difficulty="hard",
        explanation="A and C can run in parallel. Critical path: A->B->D->E. C must finish before D starts.",
    ),
    PlanningCase(
        id="plan_hard_02",
        scenario="Allocate a $100K budget across 5 departments. Each has minimum requirements and ROI projections.",
        constraints=["Engineering needs min $30K", "Marketing needs min $15K", "Operations needs min $10K",
                      "Total cannot exceed $100K", "Maximize ROI: Engineering 3x, Marketing 2.5x, Sales 2x, Operations 1.5x, HR 1x"],
        correct_order=["A", "B", "C", "D", "E"],
        steps={
            "A": "Allocate $40K to Engineering (highest ROI: 3x = $120K return)",
            "B": "Allocate $25K to Marketing (2.5x = $62.5K return)",
            "C": "Allocate $15K to Sales (2x = $30K return)",
            "D": "Allocate $10K to Operations (minimum, 1.5x = $15K return)",
            "E": "Allocate $10K to HR (remainder, 1x = $10K return)",
        },
        category="resource_allocation",
        difficulty="hard",
        explanation="Optimization problem: maximize ROI while satisfying minimum constraints.",
    ),
    PlanningCase(
        id="plan_hard_03",
        scenario="Evacuate a building with 3 floors during a fire. Stairs can handle 20 people/min. 50 people per floor.",
        constraints=["Top floor (3rd) evacuates first to avoid being trapped", "Stairs capacity: 20/min",
                      "Disabled persons need elevator (1 elevator, 4 people/trip, 2 min/trip)",
                      "Must account for 6 disabled persons across all floors"],
        correct_order=["A", "B", "C", "D", "E"],
        steps={
            "A": "Sound alarm and start elevator for disabled persons on 3rd floor",
            "B": "Begin stairway evacuation: 3rd floor first, then 2nd",
            "C": "Elevator continues trips for remaining disabled persons (floors 2, 1)",
            "D": "1st floor evacuates via stairs and exits",
            "E": "Fire team sweep all floors for stragglers, confirm full evacuation",
        },
        category="emergency_optimization",
        difficulty="hard",
        explanation="Multi-constraint optimization: capacity limits, priority ordering, special needs accommodation.",
    ),
    PlanningCase(
        id="plan_hard_04",
        scenario="Plan a multi-city business trip: NYC -> London -> Tokyo -> Sydney, with meetings that have fixed dates.",
        constraints=["NYC meeting: Monday", "London meeting: Wednesday", "Tokyo meeting: Friday",
                      "Sydney meeting: next Monday", "Must allow travel time between cities",
                      "Budget constraint: prefer direct flights"],
        correct_order=["A", "B", "C", "D", "E", "F"],
        steps={
            "A": "NYC meeting on Monday, evening flight to London",
            "B": "Arrive London Tuesday, prepare for meeting",
            "C": "London meeting Wednesday, evening flight to Tokyo",
            "D": "Arrive Tokyo Thursday, prepare and rest (jet lag)",
            "E": "Tokyo meeting Friday, Saturday flight to Sydney",
            "F": "Arrive Sydney Sunday, Monday meeting, return home",
        },
        category="travel_optimization",
        difficulty="hard",
        explanation="Fixed-date constraints with travel time requirements and timezone considerations.",
    ),
    PlanningCase(
        id="plan_hard_05",
        scenario="Design a microservices architecture migration. Monolith has 5 tightly coupled modules.",
        constraints=["Must maintain zero downtime", "Extract lowest-coupling module first",
                      "Each service needs its own database (data migration required)",
                      "Must run old and new in parallel during transition",
                      "Highest-traffic service last (most risk)"],
        correct_order=["A", "B", "C", "D", "E", "F"],
        steps={
            "A": "Identify module boundaries and coupling analysis",
            "B": "Extract Auth service (lowest coupling) with parallel running",
            "C": "Extract Notifications service and migrate its data",
            "D": "Extract Inventory service with database split",
            "E": "Extract Payment service (high-traffic, careful migration)",
            "F": "Decommission monolith after all services stable for 30 days",
        },
        category="system_architecture",
        difficulty="hard",
        explanation="Risk-ordered migration: lowest risk/coupling first, highest risk last, with stability gates.",
    ),
    PlanningCase(
        id="plan_hard_06",
        scenario="Clinical trial for a new drug. Must satisfy FDA regulations and ethical requirements.",
        constraints=["Preclinical before human trials", "Phase I before Phase II before Phase III",
                      "IRB approval at each phase", "Adverse event = pause and review",
                      "Data monitoring board reviews after each phase"],
        correct_order=["A", "B", "C", "D", "E", "F"],
        steps={
            "A": "Preclinical studies (lab and animal testing)",
            "B": "IND application to FDA and Phase I (safety, 20-100 subjects)",
            "C": "Phase II trials (efficacy, 100-300 subjects)",
            "D": "Phase III trials (large-scale, 1000-3000 subjects)",
            "E": "NDA submission with all trial data to FDA",
            "F": "FDA review, approval, and post-market surveillance plan",
        },
        category="regulatory",
        difficulty="hard",
        explanation="Heavily regulated pipeline with mandatory sequential phases and review gates.",
    ),
    PlanningCase(
        id="plan_hard_07",
        scenario="Coordinate a Mars rover landing. Multiple systems must work in precise sequence during the '7 minutes of terror'.",
        constraints=["Communication delay: 14 min (must be autonomous)", "Heat shield before parachute",
                      "Parachute before sky crane", "Everything automated — no human intervention possible"],
        correct_order=["A", "B", "C", "D", "E", "F"],
        steps={
            "A": "Atmospheric entry: deploy heat shield (19,000 mph)",
            "B": "Supersonic parachute deployment at Mach 1.7",
            "C": "Heat shield jettison and radar ground sensing",
            "D": "Back shell separation and rocket-powered descent",
            "E": "Sky crane maneuver: lower rover on cables",
            "F": "Touchdown, cable cut, sky crane flies away",
        },
        category="aerospace",
        difficulty="hard",
        explanation="Zero-margin-for-error sequential process where each step enables the next. Fully autonomous.",
    ),
    PlanningCase(
        id="plan_hard_08",
        scenario="Negotiate a peace treaty between two warring nations with 3 disputed issues: territory, refugees, and reparations.",
        constraints=["Build trust with small agreements first", "Territory is most contentious — save for last",
                      "Refugee crisis is humanitarian — address early",
                      "Both sides must feel they 'won' something", "International observers required"],
        correct_order=["A", "B", "C", "D", "E", "F"],
        steps={
            "A": "Ceasefire agreement and international observer deployment",
            "B": "Humanitarian corridor for refugee crisis (quick win for both sides)",
            "C": "Reparations framework negotiation (economic compromise)",
            "D": "Territorial boundary negotiations (most difficult)",
            "E": "Comprehensive treaty signing with all provisions",
            "F": "Implementation monitoring and dispute resolution mechanism",
        },
        category="conflict_resolution",
        difficulty="hard",
        explanation="Strategic sequencing: ceasefire first, easy wins to build trust, hardest issue last.",
    ),
    PlanningCase(
        id="plan_hard_09",
        scenario="Plan a company's AI ethics framework implementation across 5 departments.",
        constraints=["Executive buy-in before any department rollout", "Legal review of AI policies required",
                      "Training must happen before auditing", "Engineering implements tooling",
                      "Stagger rollout: one department at a time to learn"],
        correct_order=["A", "B", "C", "D", "E", "F"],
        steps={
            "A": "Executive briefing and board approval of AI ethics policy",
            "B": "Legal review and regulatory compliance mapping",
            "C": "Engineering builds bias detection and monitoring tools",
            "D": "Company-wide AI ethics training program",
            "E": "Staggered department rollout with audits (HR first, then product, etc.)",
            "F": "Establish ongoing review board and public transparency report",
        },
        category="organizational_change",
        difficulty="hard",
        explanation="Organizational transformation: executive mandate, legal framework, tooling, training, then rollout.",
    ),
    PlanningCase(
        id="plan_hard_10",
        scenario="Respond to a zero-day cybersecurity breach. Attacker has been in the network for an unknown period.",
        constraints=["Do NOT alert attacker you've detected them (yet)", "Preserve forensic evidence",
                      "Identify all compromised systems before containment",
                      "Legal and PR must be notified before public disclosure",
                      "Regulatory notification within 72 hours"],
        correct_order=["A", "B", "C", "D", "E", "F"],
        steps={
            "A": "Quietly begin forensic investigation — identify scope of breach",
            "B": "Map all compromised systems and attacker's lateral movement",
            "C": "Prepare containment plan (simultaneous lockout of all attacker access)",
            "D": "Execute containment: revoke credentials, isolate systems, patch vulnerability",
            "E": "Notify legal, executive team, and prepare regulatory disclosure",
            "F": "Public notification, customer communication, and long-term remediation",
        },
        category="incident_response",
        difficulty="hard",
        explanation="Critical sequencing: silent investigation first, coordinated containment, then disclosure. Alerting attacker early lets them destroy evidence or escalate.",
    ),
]


# ═══════════════════════════════════════════════════════════════
# Access helpers
# ═══════════════════════════════════════════════════════════════

ALL_PLANNING_CASES = EASY_CASES + MEDIUM_CASES + HARD_CASES

PLANNING_CASES_BY_DIFFICULTY = {
    "easy": EASY_CASES,
    "medium": MEDIUM_CASES,
    "hard": HARD_CASES,
}

PLANNING_CASES_BY_ID = {case.id: case for case in ALL_PLANNING_CASES}


def get_planning_cases(difficulty: str = None) -> list:
    """Get planning cases filtered by difficulty, or all if None."""
    if difficulty is None:
        return ALL_PLANNING_CASES
    return PLANNING_CASES_BY_DIFFICULTY.get(difficulty, [])


def grade_plan_order(predicted_order: list, correct_order: list) -> float:
    """Grade a plan ordering using normalized Kendall tau distance.

    - Perfect order: 1.0
    - Reversed: 0.0
    - Partial inversions: proportional score

    Also gives partial credit if the correct FIRST step is identified.
    """
    if not predicted_order or not correct_order:
        return 0.0

    n = len(correct_order)

    # Normalize both to same length
    if len(predicted_order) != n:
        # Penalize wrong number of steps
        return max(0.0, 0.3 if set(predicted_order) == set(correct_order) else 0.0)

    if predicted_order == correct_order:
        return 1.0

    # Count inversions
    inversions = 0
    total_pairs = 0
    pos_correct = {step: i for i, step in enumerate(correct_order)}

    for i in range(n):
        for j in range(i + 1, n):
            total_pairs += 1
            if predicted_order[i] in pos_correct and predicted_order[j] in pos_correct:
                if pos_correct[predicted_order[i]] > pos_correct[predicted_order[j]]:
                    inversions += 1

    if total_pairs == 0:
        return 0.0

    # Normalized: 0 inversions = 1.0, all inverted = 0.0
    score = 1.0 - (inversions / total_pairs)

    # Bonus for getting first step right
    if predicted_order[0] == correct_order[0]:
        score = min(1.0, score + 0.1)

    return round(score, 4)
