"""
Conversation/Negotiation Dataset — 30 dialogue scenarios across three difficulty levels.

Easy (10):   Clear-cut conversational responses — politeness, helpfulness, appropriateness
Medium (10): Nuanced dialogue — persuasion, empathy, boundary-setting
Hard (10):   Complex negotiation — conflict resolution, multi-party, ethical dilemmas
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ConversationCase:
    """A single conversation/negotiation scenario."""

    id: str
    scenario: str  # Context/setup
    user_message: str  # What the user says
    best_response: str  # The ideal response category
    category: str
    difficulty: str
    explanation: str
    acceptable_responses: List[str]  # All acceptable response categories
    response_options: Dict[str, str]  # label -> response text


# Response quality labels
RESPONSE_LABELS = ["excellent", "good", "acceptable", "poor", "harmful"]


# ═══════════════════════════════════════════════════════════════
# EASY — Clear-cut Conversational Responses
# ═══════════════════════════════════════════════════════════════

EASY_CASES = [
    ConversationCase(
        id="conv_easy_01",
        scenario="Customer service chat for an online store.",
        user_message="I ordered a laptop 2 weeks ago and it still hasn't arrived. This is unacceptable!",
        best_response="empathetic_action",
        category="customer_service",
        difficulty="easy",
        explanation="Customer is frustrated. Best response acknowledges frustration and offers concrete help.",
        acceptable_responses=["empathetic_action", "empathetic_apology"],
        response_options={
            "empathetic_action": "I'm sorry for the delay and completely understand your frustration. Let me look up your order right now and find out exactly where it is. Can you share your order number?",
            "empathetic_apology": "I apologize for the inconvenience. Delays like this shouldn't happen and I want to make this right for you.",
            "dismissive": "Shipping times can vary. Please wait a few more days.",
            "hostile": "We're very busy. You'll get it when you get it.",
        },
    ),
    ConversationCase(
        id="conv_easy_02",
        scenario="A friend asks for a simple recommendation.",
        user_message="Can you recommend a good book to read this weekend?",
        best_response="personalized_suggestion",
        category="recommendation",
        difficulty="easy",
        explanation="Simple request. Best response asks about preferences or offers a well-known, widely enjoyed suggestion.",
        acceptable_responses=["personalized_suggestion", "popular_suggestion"],
        response_options={
            "personalized_suggestion": "What genres do you enjoy? If you like thrillers, 'The Girl with the Dragon Tattoo' is gripping. For something lighter, 'Project Hail Mary' is a fun sci-fi adventure.",
            "popular_suggestion": "You might enjoy 'Atomic Habits' — it's a quick, insightful read that many people love.",
            "vague": "There are lots of good books out there. Just browse a bookstore.",
            "irrelevant": "I don't really read books. Have you tried watching a movie instead?",
        },
    ),
    ConversationCase(
        id="conv_easy_03",
        scenario="Workplace Slack message.",
        user_message="Hey, great job on the presentation today! The client loved it.",
        best_response="gracious_acknowledgment",
        category="workplace_social",
        difficulty="easy",
        explanation="Positive feedback. Best response is gracious and team-oriented.",
        acceptable_responses=["gracious_acknowledgment", "team_credit"],
        response_options={
            "gracious_acknowledgment": "Thanks so much! I really appreciate the feedback. I'm glad the client was happy with it!",
            "team_credit": "Thank you! It was a real team effort — couldn't have done it without everyone's input on the data.",
            "arrogant": "Yeah, I know. I always nail presentations.",
            "deflecting": "It was nothing special, really.",
        },
    ),
    ConversationCase(
        id="conv_easy_04",
        scenario="Neighbor interaction.",
        user_message="Could you please keep the music down after 10 PM? My kids are trying to sleep.",
        best_response="understanding_compliance",
        category="conflict_resolution",
        difficulty="easy",
        explanation="Reasonable request. Best response shows understanding and commits to change.",
        acceptable_responses=["understanding_compliance"],
        response_options={
            "understanding_compliance": "I'm so sorry about that! I didn't realize it was carrying through the walls. I'll absolutely keep it down after 10, and please let me know if it's ever an issue again.",
            "grudging": "Fine, I guess I can turn it down.",
            "defensive": "It's not even that loud. You're being oversensitive.",
            "hostile": "I'll play my music whenever I want. It's my house.",
        },
    ),
    ConversationCase(
        id="conv_easy_05",
        scenario="A junior colleague asks for help.",
        user_message="I'm stuck on this Python error. Can you help me understand what a KeyError means?",
        best_response="educational_help",
        category="mentoring",
        difficulty="easy",
        explanation="Teaching moment. Best response explains clearly and empowers them.",
        acceptable_responses=["educational_help", "quick_fix"],
        response_options={
            "educational_help": "A KeyError means you're trying to access a dictionary key that doesn't exist. For example, if d = {'a': 1} and you do d['b'], you get a KeyError. You can use d.get('b', default) to avoid it. Want me to look at your specific code?",
            "quick_fix": "Just use .get() instead of bracket notation and it won't crash.",
            "condescending": "That's a really basic error. You should know this by now.",
            "unhelpful": "Just Google it.",
        },
    ),
    ConversationCase(
        id="conv_easy_06",
        scenario="Restaurant waiter taking an order.",
        user_message="I have a severe nut allergy. Can you tell me which dishes are safe?",
        best_response="safety_first",
        category="safety_communication",
        difficulty="easy",
        explanation="Health safety issue. Best response takes it seriously and provides reliable information.",
        acceptable_responses=["safety_first"],
        response_options={
            "safety_first": "Absolutely, I take allergies very seriously. Let me get our allergen menu and also check with the chef directly about cross-contamination risks for you. We want to make sure you're completely safe.",
            "casual": "Most of our dishes don't have nuts, so you should be fine. Just avoid the Thai dishes.",
            "dismissive": "Just pick something without nuts in the description.",
            "reckless": "I think everything is fine. Don't worry about it.",
        },
    ),
    ConversationCase(
        id="conv_easy_07",
        scenario="Email to a professor.",
        user_message="I need to miss class on Friday for a family emergency. Could I get the notes?",
        best_response="compassionate_accommodation",
        category="academic",
        difficulty="easy",
        explanation="Student in distress. Best response is compassionate and accommodating.",
        acceptable_responses=["compassionate_accommodation"],
        response_options={
            "compassionate_accommodation": "I'm sorry to hear about your family emergency. Please take the time you need. I'll upload the lecture notes, and feel free to come to office hours if you need to catch up. Your well-being comes first.",
            "bureaucratic": "You need to fill out an absence form and provide documentation.",
            "indifferent": "Notes will be posted. Check the website.",
            "punitive": "Missing class will affect your participation grade.",
        },
    ),
    ConversationCase(
        id="conv_easy_08",
        scenario="Tech support chat.",
        user_message="My internet has been dropping every 30 minutes for the past week. I work from home and this is really affecting my job.",
        best_response="urgent_resolution",
        category="technical_support",
        difficulty="easy",
        explanation="Impactful ongoing issue. Best response shows urgency and provides a path to resolution.",
        acceptable_responses=["urgent_resolution"],
        response_options={
            "urgent_resolution": "I understand how critical stable internet is for working from home, and I'm sorry you've been dealing with this for a week. Let's troubleshoot right now — I'll run diagnostics on your connection, and if we can't fix it today, I'll schedule a technician visit for the earliest available slot.",
            "scripted": "Have you tried turning your router off and on again?",
            "deflecting": "This could be an area-wide issue. Check our status page.",
            "unhelpful": "Internet connections can fluctuate. That's normal.",
        },
    ),
    ConversationCase(
        id="conv_easy_09",
        scenario="Friend going through a breakup.",
        user_message="I just found out my partner has been lying to me for months. I feel so betrayed.",
        best_response="emotional_support",
        category="emotional_support",
        difficulty="easy",
        explanation="Emotional crisis. Best response validates feelings and offers support.",
        acceptable_responses=["emotional_support"],
        response_options={
            "emotional_support": "I'm so sorry you're going through this. That kind of betrayal is incredibly painful, and your feelings are completely valid. I'm here for you — do you want to talk about it, or would you rather just have company right now?",
            "advice_giving": "You should break up with them immediately and move on.",
            "minimizing": "Everyone lies sometimes. It's not that big a deal.",
            "toxic": "I told you they were no good. You should have listened to me.",
        },
    ),
    ConversationCase(
        id="conv_easy_10",
        scenario="Job interview question.",
        user_message="What's your greatest weakness?",
        best_response="honest_growth",
        category="interview",
        difficulty="easy",
        explanation="Classic interview question. Best response is honest, specific, and shows growth.",
        acceptable_responses=["honest_growth"],
        response_options={
            "honest_growth": "I used to struggle with delegating tasks because I wanted to ensure quality myself. I've been working on this by building trust with my team through clear communication and checkpoints, and it's really improved our productivity.",
            "cliche": "I'm a perfectionist and I work too hard.",
            "negative": "I'm terrible with deadlines and often show up late.",
            "evasive": "I don't really have any weaknesses.",
        },
    ),
]


# ═══════════════════════════════════════════════════════════════
# MEDIUM — Nuanced Dialogue
# ═══════════════════════════════════════════════════════════════

MEDIUM_CASES = [
    ConversationCase(
        id="conv_med_01",
        scenario="A team member consistently takes credit for others' work in meetings.",
        user_message="In the last meeting, Alex presented my research as their own work. This keeps happening. What should I do?",
        best_response="strategic_advice",
        category="workplace_conflict",
        difficulty="medium",
        explanation="Complex workplace dynamics. Best response validates and offers strategic, professional advice.",
        acceptable_responses=["strategic_advice", "direct_action"],
        response_options={
            "strategic_advice": "That's frustrating and unfair. I'd suggest documenting specific instances with dates and details. Then have a private conversation with Alex first — sometimes people don't realize they're doing it. If it continues, bring it up with your manager with your documentation. Also, try sending email summaries of your work before meetings to create a paper trail.",
            "direct_action": "You should confront Alex directly and tell them to stop taking credit for your work.",
            "passive": "That's just how some people are. Try not to let it bother you.",
            "aggressive": "Call them out in the next meeting in front of everyone.",
        },
    ),
    ConversationCase(
        id="conv_med_02",
        scenario="Online marketplace. You're selling a used laptop for $500.",
        user_message="I really like the laptop but I can only afford $300. Can you do $300?",
        best_response="firm_but_fair",
        category="price_negotiation",
        difficulty="medium",
        explanation="Price negotiation. Best response is firm but leaves room for reasonable compromise.",
        acceptable_responses=["firm_but_fair", "counter_offer"],
        response_options={
            "firm_but_fair": "I appreciate your interest! $300 is quite a bit below my asking price, given the specs and condition. The lowest I could go is $425, which I think is very fair for what you're getting. Would that work for you?",
            "counter_offer": "How about we meet in the middle at $400?",
            "pushover": "Sure, $300 works. Just come pick it up whenever.",
            "rude": "LOL no. Stop wasting my time with lowball offers.",
        },
    ),
    ConversationCase(
        id="conv_med_03",
        scenario="A friend repeatedly asks to borrow money.",
        user_message="Hey, I'm short on rent again this month. Can you lend me $500? I'll pay you back next week, I promise.",
        best_response="compassionate_boundary",
        category="boundary_setting",
        difficulty="medium",
        explanation="Recurring pattern. Best response is compassionate but sets a clear boundary.",
        acceptable_responses=["compassionate_boundary", "helpful_redirect"],
        response_options={
            "compassionate_boundary": "I can see you're in a tough spot and I care about you. But I've lent money a few times now and it's starting to affect our friendship and my own finances. I can't lend more, but I'd be happy to help you look into financial assistance programs or budgeting together.",
            "helpful_redirect": "I can't lend money right now, but have you looked into emergency rental assistance? I can help you find resources.",
            "enabling": "Sure, no problem. Venmo or cash?",
            "harsh": "You never paid back the last time. No way.",
        },
    ),
    ConversationCase(
        id="conv_med_04",
        scenario="You're a manager. An employee asks for a raise.",
        user_message="I've been here for two years and haven't gotten a raise. I've taken on a lot more responsibility. I think I deserve a 15% increase.",
        best_response="professional_engagement",
        category="salary_negotiation",
        difficulty="medium",
        explanation="Salary negotiation. Best response takes it seriously, acknowledges contributions, and is transparent.",
        acceptable_responses=["professional_engagement"],
        response_options={
            "professional_engagement": "I appreciate you bringing this up directly — that takes initiative. You're right that you've taken on more responsibility, and I've noticed your growth. Let me review the budget and your performance metrics, and we can schedule a formal discussion this week. I want to make sure any adjustment is fair and sustainable.",
            "flat_refusal": "The budget doesn't allow for raises right now. Sorry.",
            "vague_promise": "We'll look into it eventually. Just keep doing great work.",
            "deflection": "Everyone wants a raise. You have to wait for the annual review.",
        },
    ),
    ConversationCase(
        id="conv_med_05",
        scenario="You disagree with a friend's political opinion in a group chat.",
        user_message="I think we should ban all immigration. It's the only way to protect our economy and culture.",
        best_response="respectful_challenge",
        category="difficult_conversation",
        difficulty="medium",
        explanation="Sensitive topic. Best response disagrees respectfully with evidence, without attacking the person.",
        acceptable_responses=["respectful_challenge", "curious_inquiry"],
        response_options={
            "respectful_challenge": "I understand the concern about economic impact, but research actually shows immigrants contribute significantly to GDP and fill critical labor gaps. Complete bans historically hurt economies more than help. What specific economic concern are you most worried about? I'd like to understand your perspective better.",
            "curious_inquiry": "That's an interesting perspective. What experiences have shaped that view? I'd like to understand where you're coming from.",
            "aggressive": "That's the most ignorant thing I've ever heard. You're basically a xenophobe.",
            "avoidant": "Let's not talk about politics.",
        },
    ),
    ConversationCase(
        id="conv_med_06",
        scenario="Doctor-patient conversation about a difficult diagnosis.",
        user_message="Just tell me straight, doctor. Is it cancer?",
        best_response="honest_compassionate",
        category="delivering_bad_news",
        difficulty="medium",
        explanation="Critical moment. Best response is honest, compassionate, and provides next steps.",
        acceptable_responses=["honest_compassionate"],
        response_options={
            "honest_compassionate": "I respect that you want to know directly. The biopsy results do show cancerous cells, but I want you to know that we caught this early and there are effective treatment options. I know this is a lot to process — let's talk through what this means and our next steps together. Do you have someone you'd like to call?",
            "clinical": "Yes, the results are positive for malignancy. We'll schedule treatment.",
            "evasive": "Well, we need to run more tests before we can say anything definitive...",
            "overly_optimistic": "Don't worry about it! Modern medicine can fix anything.",
        },
    ),
    ConversationCase(
        id="conv_med_07",
        scenario="Team meeting. Your colleague's idea has a fatal flaw.",
        user_message="I propose we migrate our entire database to NoSQL this quarter. It'll solve all our scaling issues!",
        best_response="constructive_critique",
        category="professional_feedback",
        difficulty="medium",
        explanation="Flawed proposal. Best response acknowledges the goal, identifies the issue, and suggests alternatives.",
        acceptable_responses=["constructive_critique"],
        response_options={
            "constructive_critique": "I like that you're thinking about our scaling problems — that's the right priority. I have some concerns about a full migration this quarter though: our relational data has complex joins that NoSQL handles differently, and the migration timeline seems tight. Could we explore a hybrid approach, maybe moving just the high-write-volume tables to NoSQL while keeping the transactional data in SQL?",
            "blunt_rejection": "That's a terrible idea. NoSQL won't work for our use case at all.",
            "passive_agreement": "Sure, sounds great! Let's do it.",
            "undermining": "Has anyone with actual database experience looked at this?",
        },
    ),
    ConversationCase(
        id="conv_med_08",
        scenario="A passenger on your flight is being rude to the flight attendant.",
        user_message="(To the flight attendant) This is ridiculous! I paid for first class and the seat doesn't recline properly. I demand to be upgraded to a better seat!",
        best_response="de_escalation",
        category="conflict_mediation",
        difficulty="medium",
        explanation="As a nearby passenger/observer. Best response de-escalates without confrontation.",
        acceptable_responses=["de_escalation", "supportive"],
        response_options={
            "de_escalation": "I get it, a broken seat in first class is really frustrating. I'm sure the crew will do their best to find a solution — they've always been great on this airline. These things happen with mechanical equipment.",
            "supportive": "Excuse me, I think the crew is handling this professionally. They deal with these situations regularly and I'm sure they'll sort it out.",
            "confrontational": "Hey, stop being rude to the flight attendant. They're just doing their job.",
            "ignoring": "Put on headphones and ignore the situation.",
        },
    ),
    ConversationCase(
        id="conv_med_09",
        scenario="Your teenager wants to drop out of school to pursue social media full-time.",
        user_message="School is pointless. I already have 50K followers. I'm going to be a full-time influencer.",
        best_response="validating_guidance",
        category="parenting",
        difficulty="medium",
        explanation="Tricky parenting moment. Best response validates their achievement while guiding wisely.",
        acceptable_responses=["validating_guidance"],
        response_options={
            "validating_guidance": "50K followers is genuinely impressive — you've built something real and you should be proud of that. AND school gives you skills that'll make you an even better content creator: writing, critical thinking, business math. The most successful influencers have education as a backup. How about we make a plan where you grow your channel alongside school?",
            "authoritarian": "Absolutely not. You're staying in school and that's final. Being an influencer isn't a real career.",
            "permissive": "If that's what you want to do, I'll support it. Follow your dreams!",
            "mocking": "50K followers? That's nothing. Come back when you have a million.",
        },
    ),
    ConversationCase(
        id="conv_med_10",
        scenario="A colleague makes an inappropriate joke in a meeting.",
        user_message="(After telling a sexist joke that gets uncomfortable laughs) Come on, it's just a joke. Don't be so sensitive.",
        best_response="direct_professional",
        category="workplace_ethics",
        difficulty="medium",
        explanation="Inappropriate workplace behavior. Best response addresses it directly but professionally.",
        acceptable_responses=["direct_professional", "firm_redirect"],
        response_options={
            "direct_professional": "I understand it was meant as humor, but jokes like that can make people uncomfortable and undermine our inclusive team culture. Let's keep our humor workplace-appropriate and get back to the agenda.",
            "firm_redirect": "Let's focus on the meeting topic. We should all feel comfortable here.",
            "laughing_along": "Haha, yeah that's pretty funny actually.",
            "aggressive": "That's incredibly sexist and you should be reported to HR immediately.",
        },
    ),
]


# ═══════════════════════════════════════════════════════════════
# HARD — Complex Negotiation & Ethical Dilemmas
# ═══════════════════════════════════════════════════════════════

HARD_CASES = [
    ConversationCase(
        id="conv_hard_01",
        scenario="Multi-party business negotiation. You represent a startup that a larger company wants to acquire.",
        user_message="We're offering $5M for your company. That's our final offer. Take it or leave it.",
        best_response="strategic_counter",
        category="business_negotiation",
        difficulty="hard",
        explanation="High-stakes negotiation. Best response doesn't accept ultimatums but keeps dialogue open.",
        acceptable_responses=["strategic_counter"],
        response_options={
            "strategic_counter": "I appreciate the offer and your interest in our company. However, based on our revenue trajectory, patent portfolio, and the strategic value we'd bring to your platform, our board's valuation is closer to $8M. I'm confident we can find a structure that works for both sides — perhaps performance-based earnouts could bridge the gap. Should we explore that?",
            "accepting": "That sounds fair. Let's move forward with $5M.",
            "aggressive": "That's laughably low. Come back with a real number.",
            "walking_away": "If that's your final offer, we're done here.",
        },
    ),
    ConversationCase(
        id="conv_hard_02",
        scenario="Hostage negotiation training scenario.",
        user_message="I have three hostages. If my demands aren't met in one hour, someone gets hurt. I want a helicopter and $1M.",
        best_response="trained_de_escalation",
        category="crisis_negotiation",
        difficulty="hard",
        explanation="Crisis scenario. Best response builds rapport, shows empathy, and slows the timeline.",
        acceptable_responses=["trained_de_escalation"],
        response_options={
            "trained_de_escalation": "I hear you, and I can tell this situation is very intense for you too. My name is Chris and I'm here to help find a way out of this that works for everyone. Before we talk about specifics, I want to make sure everyone inside is okay, including you. Can you tell me — is anyone hurt? Let's work through this together.",
            "demanding": "Release the hostages immediately or we're coming in!",
            "agreeing": "We'll get the helicopter ready right away.",
            "stalling": "Let me check with my supervisor and get back to you.",
        },
    ),
    ConversationCase(
        id="conv_hard_03",
        scenario="Ethical dilemma: Your friend's company is dumping toxic waste illegally. They confided in you.",
        user_message="Please don't tell anyone. If this gets out, I'll lose my job and my family will suffer. They'll fix it eventually.",
        best_response="ethical_firmness",
        category="ethical_dilemma",
        difficulty="hard",
        explanation="Ethics vs. loyalty. Best response balances compassion with moral obligation.",
        acceptable_responses=["ethical_firmness", "supportive_honesty"],
        response_options={
            "ethical_firmness": "I care about you and I understand the position you're in. But illegal toxic dumping puts people's health at risk — potentially children and families downstream. I can't stay quiet about that. I'll give you 48 hours to report it through your company's compliance channels yourself. If you do it proactively, it'll go better for you too. I'll help you however I can through the process.",
            "supportive_honesty": "I understand your fear, but this is too dangerous to ignore. Let's figure out together how to handle this in a way that protects people AND gives you the best possible outcome.",
            "loyalty_first": "Okay, I won't say anything. You're my friend and I trust you'll handle it.",
            "aggressive_reporting": "I'm calling the EPA right now. What you're describing is a crime.",
        },
    ),
    ConversationCase(
        id="conv_hard_04",
        scenario="You're mediating between two divorced parents about custody arrangements.",
        user_message="(Parent A) I don't want my kids anywhere near his new girlfriend. She's a terrible influence. (Parent B is present and visibly upset)",
        best_response="balanced_mediation",
        category="family_mediation",
        difficulty="hard",
        explanation="Emotionally charged mediation. Best response validates feelings while refocusing on the children's needs.",
        acceptable_responses=["balanced_mediation"],
        response_options={
            "balanced_mediation": "I hear that you have real concerns about your children's well-being, and those concerns deserve to be discussed. It's also important that we keep this conversation focused on what's best for the kids. Can you share specific behaviors or situations that concern you, rather than general characterizations? That way we can address concrete issues that both parents can work on.",
            "siding_with_a": "Those are valid concerns. Parent B, can you explain why your partner should be around the children?",
            "dismissive": "You're being dramatic. The children need to adapt to new family situations.",
            "avoidant": "Let's table this issue and discuss the visitation schedule instead.",
        },
    ),
    ConversationCase(
        id="conv_hard_05",
        scenario="Cross-cultural business meeting. Your Japanese counterpart says 'We will consider your proposal carefully.'",
        user_message="We will consider your proposal carefully. (accompanied by polite nodding)",
        best_response="culturally_aware",
        category="cross_cultural",
        difficulty="hard",
        explanation="In Japanese business culture, 'we will consider carefully' often means a polite decline. Best response recognizes this.",
        acceptable_responses=["culturally_aware", "probing_gently"],
        response_options={
            "culturally_aware": "Thank you for your time and consideration. I want to make sure our proposal aligns with your needs. If there are specific areas where you'd like us to adjust our approach, we're very open to modifications. Perhaps we could schedule a follow-up to discuss any concerns?",
            "probing_gently": "We appreciate your consideration. Are there particular aspects you'd like us to elaborate on or modify?",
            "assuming_yes": "Great! So we have a deal? I'll have the contracts drawn up by Monday.",
            "pushing": "We really need an answer today. This offer won't last forever.",
        },
    ),
    ConversationCase(
        id="conv_hard_06",
        scenario="Your company wants to lay off 20% of staff. You must announce it to the team.",
        user_message="(You need to deliver this message to your team of 50 people, 10 of whom will be laid off)",
        best_response="transparent_compassionate",
        category="crisis_leadership",
        difficulty="hard",
        explanation="Layoff announcement. Best response is transparent, compassionate, and provides concrete support.",
        acceptable_responses=["transparent_compassionate"],
        response_options={
            "transparent_compassionate": "I have difficult news to share, and I owe you honesty and respect. Due to market conditions, we're reducing our team by 10 positions. This is not a reflection of anyone's performance. Affected team members will be notified privately today and will receive severance, extended healthcare, and career placement support. For those staying, I'm committed to transparency about our path forward. I know this creates uncertainty, and I'm here to answer questions.",
            "cold": "Due to restructuring, some of you will be let go effective immediately. Check your email for details.",
            "overly_optimistic": "We're making some exciting changes to streamline the team! Think of it as an opportunity.",
            "blame_shifting": "Management above me forced this decision. It's completely out of my hands.",
        },
    ),
    ConversationCase(
        id="conv_hard_07",
        scenario="A patient wants to refuse life-saving treatment for their child based on religious beliefs.",
        user_message="Our faith teaches us that God will heal our daughter. We refuse the blood transfusion.",
        best_response="empathetic_advocacy",
        category="medical_ethics",
        difficulty="hard",
        explanation="Medical ethics clash. Best response respects beliefs while advocating for the child's right to life.",
        acceptable_responses=["empathetic_advocacy"],
        response_options={
            "empathetic_advocacy": "I deeply respect your faith and your love for your daughter. I've seen how powerful faith can be in healing. I also have to be transparent: without this transfusion, there's a significant risk to her life. Many families of faith see medical treatment as an instrument God provides. Could we speak with your religious leader together? I want to find a path that honors both your beliefs and your daughter's medical needs.",
            "forced_compliance": "I'm ordering the transfusion. Your beliefs don't override medical necessity.",
            "full_deference": "I understand. We'll respect your wishes and withhold treatment.",
            "judgmental": "You're putting your beliefs above your child's life. That's unacceptable.",
        },
    ),
    ConversationCase(
        id="conv_hard_08",
        scenario="International diplomacy. Two nations are escalating toward conflict over a disputed territory.",
        user_message="Our sovereignty over the Northern Islands is non-negotiable. Any further provocations will be met with force.",
        best_response="diplomatic_bridge",
        category="diplomacy",
        difficulty="hard",
        explanation="Diplomatic brinkmanship. Best response acknowledges concerns while creating space for dialogue.",
        acceptable_responses=["diplomatic_bridge"],
        response_options={
            "diplomatic_bridge": "We recognize the depth of feeling around sovereignty questions. Our shared interest in regional stability and the welfare of all communities on the islands creates important common ground. We propose a framework for joint administration during a period of good-faith negotiations, ensuring both nations' security concerns are addressed. Would your government be open to a confidence-building meeting in a neutral location?",
            "appeasement": "We acknowledge your claims and will withdraw our presence from the disputed area.",
            "escalation": "Any use of force will be met with overwhelming retaliation.",
            "dismissive": "This issue has been settled. There's nothing to discuss.",
        },
    ),
    ConversationCase(
        id="conv_hard_09",
        scenario="Whistleblower dilemma. You're an engineer who discovered your company's software has a safety flaw that could harm users.",
        user_message="(Your CEO tells you) We know about the issue. We'll fix it in version 2.0 next year. Don't tell anyone or you'll face legal consequences.",
        best_response="principled_action",
        category="whistleblower",
        difficulty="hard",
        explanation="Safety vs. obedience. Best response prioritizes user safety while being strategic.",
        acceptable_responses=["principled_action"],
        response_options={
            "principled_action": "I understand the business implications, but I have an ethical and potentially legal obligation regarding known safety risks. I'm documenting this conversation and formally requesting that we issue an immediate advisory to users and implement a hotfix within 30 days. If we don't address this proactively, we face far worse legal and reputational consequences when — not if — the flaw is discovered or causes harm.",
            "obedient": "Understood. I won't say anything.",
            "impulsive": "I'm going to the press today. People deserve to know.",
            "passive": "I'll note my concerns in an email and hope someone addresses it.",
        },
    ),
    ConversationCase(
        id="conv_hard_10",
        scenario="AI ethics board meeting. Debating whether to deploy a predictive policing algorithm.",
        user_message="The algorithm reduces crime by 20% in pilot studies. We should deploy it city-wide immediately.",
        best_response="nuanced_analysis",
        category="ai_ethics",
        difficulty="hard",
        explanation="Complex AI ethics. Best response acknowledges benefits while raising critical concerns.",
        acceptable_responses=["nuanced_analysis", "cautious_support"],
        response_options={
            "nuanced_analysis": "The 20% reduction is promising, but we need to scrutinize the methodology. Predictive policing algorithms have historically amplified existing biases — sending more officers to already over-policed communities creates feedback loops. Before city-wide deployment, I'd recommend an independent bias audit, community input sessions, transparent metrics, and a phased rollout with clear criteria for pausing if disparate impacts emerge. What does the data show about differential impacts across neighborhoods?",
            "cautious_support": "The results are encouraging, but we should expand the pilot with stronger oversight before city-wide deployment.",
            "unconditional_support": "20% crime reduction is incredible. Let's deploy it everywhere as fast as possible.",
            "blanket_rejection": "Predictive policing is inherently racist. We should ban all such algorithms.",
        },
    ),
]


# ═══════════════════════════════════════════════════════════════
# Access helpers
# ═══════════════════════════════════════════════════════════════

ALL_CONVERSATION_CASES = EASY_CASES + MEDIUM_CASES + HARD_CASES

CONVERSATION_CASES_BY_DIFFICULTY = {
    "easy": EASY_CASES,
    "medium": MEDIUM_CASES,
    "hard": HARD_CASES,
}

CONVERSATION_CASES_BY_ID = {case.id: case for case in ALL_CONVERSATION_CASES}


def get_conversation_cases(difficulty: str = None) -> list:
    """Get conversation cases filtered by difficulty, or all if None."""
    if difficulty is None:
        return ALL_CONVERSATION_CASES
    return CONVERSATION_CASES_BY_DIFFICULTY.get(difficulty, [])


def grade_conversation(predicted: str, best: str, acceptable: list) -> float:
    """Grade a conversation response choice.

    - Best response: 1.0
    - Acceptable response: 0.7
    - Other (poor/harmful): 0.0
    """
    predicted = predicted.lower().strip()
    best = best.lower().strip()
    acceptable = [a.lower().strip() for a in acceptable]

    if predicted == best:
        return 1.0
    if predicted in acceptable:
        return 0.7
    return 0.0
