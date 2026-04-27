"""
Text Summarization Dataset — 30 passages with reference summaries.

Easy (10):   Short, clear texts — news headlines, simple paragraphs
Medium (10): Technical/academic texts requiring key point extraction
Hard (10):   Long, nuanced texts with multiple viewpoints
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class SummarizationCase:
    """A single text summarization task."""

    id: str
    text: str
    reference_summary: str
    key_points: List[str]
    category: str
    difficulty: str
    max_summary_length: int = 100  # words


EASY_CASES = [
    SummarizationCase(
        id="sum_easy_01",
        text="Scientists at MIT have developed a new solar panel that is 30% more efficient than current models. The breakthrough uses a novel perovskite material that captures a wider spectrum of sunlight. The technology could be commercially available within two years, potentially reducing the cost of solar energy by half.",
        reference_summary="MIT scientists created a solar panel 30% more efficient using perovskite material, potentially halving solar energy costs within two years.",
        key_points=["MIT", "30% more efficient", "perovskite", "two years", "half cost"],
        category="science_news",
        difficulty="easy",
    ),
    SummarizationCase(
        id="sum_easy_02",
        text="The city council voted 7-2 to approve a new public transit line connecting downtown to the airport. The $2 billion project will create 5,000 jobs and reduce commute times by 40 minutes. Construction is expected to begin in spring 2027.",
        reference_summary="City council approved a $2B transit line from downtown to airport, creating 5,000 jobs and cutting commutes by 40 minutes, starting spring 2027.",
        key_points=["7-2 vote", "$2 billion", "5000 jobs", "40 minutes", "spring 2027"],
        category="local_news",
        difficulty="easy",
    ),
    SummarizationCase(
        id="sum_easy_03",
        text="Apple announced its quarterly earnings today, reporting revenue of $94 billion, a 5% increase from last year. iPhone sales drove most of the growth, with services revenue also hitting a record high. CEO Tim Cook expressed optimism about the company's AI initiatives.",
        reference_summary="Apple reported $94B quarterly revenue (up 5%), driven by iPhone and record services revenue. CEO optimistic about AI initiatives.",
        key_points=["$94 billion", "5% increase", "iPhone", "services", "AI"],
        category="business_news",
        difficulty="easy",
    ),
    SummarizationCase(
        id="sum_easy_04",
        text="A new study published in Nature found that walking just 30 minutes a day can reduce the risk of heart disease by 25%. Researchers tracked 50,000 participants over 10 years. The benefits were consistent across all age groups and body types.",
        reference_summary="Nature study of 50,000 people over 10 years found 30 minutes of daily walking reduces heart disease risk by 25%, regardless of age or body type.",
        key_points=["30 minutes", "heart disease", "25%", "50,000 participants", "10 years"],
        category="health_news",
        difficulty="easy",
    ),
    SummarizationCase(
        id="sum_easy_05",
        text="The Mars rover Perseverance has discovered organic molecules in rock samples from Jezero Crater. While this doesn't confirm life, it suggests the crater once had conditions suitable for microbial life. NASA plans to return the samples to Earth by 2033.",
        reference_summary="Perseverance found organic molecules in Mars' Jezero Crater rocks, suggesting past habitable conditions. Samples to return to Earth by 2033.",
        key_points=["organic molecules", "Jezero Crater", "microbial life", "NASA", "2033"],
        category="space_news",
        difficulty="easy",
    ),
    SummarizationCase(
        id="sum_easy_06",
        text="The World Cup final between France and Argentina drew 1.5 billion viewers worldwide, making it the most-watched sporting event in history. Argentina won 4-2 on penalties after a dramatic 3-3 draw.",
        reference_summary="Argentina beat France 4-2 on penalties (after 3-3 draw) in the most-watched World Cup final ever, with 1.5 billion viewers.",
        key_points=["1.5 billion", "World Cup", "Argentina", "France", "penalties"],
        category="sports_news",
        difficulty="easy",
    ),
    SummarizationCase(
        id="sum_easy_07",
        text="Google has released Gemini 2.0, its most advanced AI model. The model outperforms GPT-4 on most benchmarks and can process text, images, and video. It will be integrated into Search, Gmail, and Docs.",
        reference_summary="Google launched Gemini 2.0, outperforming GPT-4 on most benchmarks with multimodal capabilities, coming to Search, Gmail, and Docs.",
        key_points=["Gemini 2.0", "GPT-4", "multimodal", "Search", "Gmail"],
        category="tech_news",
        difficulty="easy",
    ),
    SummarizationCase(
        id="sum_easy_08",
        text="A magnitude 6.2 earthquake struck central Italy early this morning. No casualties have been reported, but several historic buildings sustained damage. Emergency services are assessing the situation.",
        reference_summary="A 6.2 earthquake hit central Italy with no casualties but damage to historic buildings. Emergency assessment ongoing.",
        key_points=["6.2 magnitude", "Italy", "no casualties", "historic buildings", "emergency"],
        category="world_news",
        difficulty="easy",
    ),
    SummarizationCase(
        id="sum_easy_09",
        text="Tesla reported delivering 500,000 vehicles in Q4, a new company record. The Model Y was the best-selling car globally for the second year. Analysts expect Tesla to reach 2 million annual deliveries next year.",
        reference_summary="Tesla delivered a record 500K vehicles in Q4. Model Y remains global best-seller, with 2M annual deliveries expected next year.",
        key_points=["500,000", "Q4 record", "Model Y", "best-selling", "2 million"],
        category="business_news",
        difficulty="easy",
    ),
    SummarizationCase(
        id="sum_easy_10",
        text="The United Nations climate summit concluded with 195 nations agreeing to phase out fossil fuels by 2050. The agreement includes a $100 billion annual fund for developing nations. Critics say the timeline is too slow.",
        reference_summary="195 nations agreed at the UN summit to phase out fossil fuels by 2050 with $100B annual funding for developing nations, though critics want faster action.",
        key_points=["195 nations", "fossil fuels", "2050", "$100 billion", "critics"],
        category="climate_news",
        difficulty="easy",
    ),
]

MEDIUM_CASES = [
    SummarizationCase(
        id="sum_med_01",
        text="Quantum computing has reached a significant milestone with IBM's 1,121-qubit processor, codenamed Condor. Unlike classical bits that are either 0 or 1, qubits can exist in superposition, enabling quantum computers to solve certain problems exponentially faster. However, error rates remain a challenge — current quantum computers require extensive error correction, and practical quantum advantage for real-world problems is still years away. IBM's roadmap targets error-corrected quantum computing by 2029.",
        reference_summary="IBM's 1,121-qubit Condor processor marks a quantum computing milestone, but high error rates mean practical quantum advantage remains years away, with IBM targeting error-corrected computing by 2029.",
        key_points=["1121 qubits", "Condor", "superposition", "error rates", "2029"],
        category="technology",
        difficulty="medium",
    ),
    SummarizationCase(
        id="sum_med_02",
        text="CRISPR gene editing has been approved for clinical use in treating sickle cell disease. The FDA-approved therapy, Casgevy, modifies a patient's own stem cells to produce functional hemoglobin. While the one-time treatment costs $2.2 million, it could eliminate the need for lifelong blood transfusions and pain management. Ethical concerns remain about germline editing and accessibility. Insurance coverage negotiations are ongoing.",
        reference_summary="The FDA approved Casgevy, a $2.2M one-time CRISPR therapy for sickle cell disease that modifies stem cells to produce functional hemoglobin, though ethical and accessibility concerns persist.",
        key_points=["CRISPR", "Casgevy", "sickle cell", "$2.2 million", "ethical concerns"],
        category="medicine",
        difficulty="medium",
    ),
    SummarizationCase(
        id="sum_med_03",
        text="The European Union's AI Act, the world's first comprehensive AI regulation, categorizes AI systems by risk level. High-risk systems like facial recognition in public spaces face strict requirements including transparency, human oversight, and bias audits. General-purpose AI models must disclose training data and comply with copyright law. Critics argue the regulation may stifle innovation, while supporters say it sets a global standard for responsible AI development. Penalties for non-compliance can reach 7% of global revenue.",
        reference_summary="The EU AI Act categorizes AI systems by risk level, requiring transparency and bias audits for high-risk systems. Penalties reach 7% of global revenue. Debate continues between innovation concerns and responsible AI standards.",
        key_points=["EU AI Act", "risk levels", "transparency", "7% revenue", "innovation debate"],
        category="regulation",
        difficulty="medium",
    ),
    SummarizationCase(
        id="sum_med_04",
        text="Recent research in neuroscience has revealed that the brain's default mode network (DMN) is far more active during creative thinking than previously believed. The DMN, once thought to be active only during rest, actually coordinates with the executive control network during creative problem-solving. This challenges the popular notion that creativity is purely a right-brain activity. The findings suggest that techniques like meditation and mind-wandering may enhance creativity by strengthening DMN connectivity.",
        reference_summary="New neuroscience research shows the brain's default mode network actively coordinates with executive control during creativity, challenging the right-brain myth. Meditation may enhance creativity through DMN connectivity.",
        key_points=["default mode network", "creative thinking", "executive control", "right-brain myth", "meditation"],
        category="neuroscience",
        difficulty="medium",
    ),
    SummarizationCase(
        id="sum_med_05",
        text="The global semiconductor shortage that began in 2020 has reshaped chip manufacturing strategy. TSMC is building a $40 billion fab in Arizona, Samsung is investing $17 billion in Texas, and Intel is spending $20 billion on facilities in Ohio. These investments aim to reduce dependence on East Asian manufacturing, which currently produces 75% of the world's chips. However, analysts warn that the new fabs won't be operational until 2025-2026 and may face workforce challenges.",
        reference_summary="The chip shortage has driven $77B+ in US fab investments from TSMC ($40B Arizona), Samsung ($17B Texas), and Intel ($20B Ohio) to reduce reliance on East Asia's 75% market share, with operations starting 2025-2026.",
        key_points=["chip shortage", "TSMC $40B", "Samsung $17B", "Intel $20B", "75% East Asia"],
        category="industry",
        difficulty="medium",
    ),
    SummarizationCase(
        id="sum_med_06",
        text="A longitudinal study following 10,000 remote workers over three years found that fully remote employees reported 23% higher job satisfaction but 15% lower promotion rates compared to in-office peers. Hybrid workers (3 days in office) showed the best outcomes across all metrics. The study also found that remote work disproportionately benefited parents, people with disabilities, and those living far from urban centers, but increased feelings of isolation among younger workers under 30.",
        reference_summary="A 3-year study of 10,000 workers found remote employees had 23% higher satisfaction but 15% lower promotion rates. Hybrid (3 days in-office) performed best overall. Remote work benefited parents and disabled workers but increased isolation for under-30s.",
        key_points=["10,000 workers", "23% satisfaction", "15% lower promotion", "hybrid best", "isolation under-30"],
        category="workplace",
        difficulty="medium",
    ),
    SummarizationCase(
        id="sum_med_07",
        text="Microplastics have been detected in human blood for the first time, according to a study by VU Amsterdam. Researchers found plastic particles in 77% of blood samples tested, with PET (used in drink bottles) and polystyrene (food packaging) being the most common types. The health implications are unknown, but animal studies suggest microplastics can cause inflammation and cellular damage. The study calls for urgent research into long-term health effects.",
        reference_summary="VU Amsterdam found microplastics (mainly PET and polystyrene) in 77% of human blood samples for the first time. Health effects are unknown but animal studies show inflammation risk, prompting calls for urgent research.",
        key_points=["microplastics", "human blood", "77%", "PET", "inflammation"],
        category="health_research",
        difficulty="medium",
    ),
    SummarizationCase(
        id="sum_med_08",
        text="SpaceX's Starship completed its first successful orbital flight, marking a major milestone for reusable heavy-lift rockets. The 120-meter vehicle, designed to carry 100+ tons to orbit, is central to NASA's Artemis moon landing program and Musk's Mars colonization plans. The flight lasted 90 minutes and included a successful booster catch using the tower's 'chopstick' arms. Competitors view the achievement with concern, as Starship's cost per kilogram to orbit could be 10x cheaper than current rockets.",
        reference_summary="SpaceX's Starship completed its first orbital flight with a successful booster catch, achieving a milestone for NASA's moon program and Mars plans. Its potential 10x cost reduction per kilogram threatens competitors.",
        key_points=["Starship", "orbital flight", "booster catch", "100+ tons", "10x cheaper"],
        category="aerospace",
        difficulty="medium",
    ),
    SummarizationCase(
        id="sum_med_09",
        text="The World Health Organization declared loneliness a global health priority, citing research that social isolation increases mortality risk by 26% — comparable to smoking 15 cigarettes per day. The initiative calls for governments to appoint 'loneliness ministers' and invest in community infrastructure. Japan, UK, and Australia have already created such positions. The problem is particularly acute among elderly and young adults aged 18-25.",
        reference_summary="WHO declared loneliness a global health priority, noting it raises mortality risk by 26% (equivalent to 15 cigarettes/day). The initiative urges governments to appoint loneliness ministers, targeting elderly and young adults 18-25.",
        key_points=["WHO", "loneliness", "26% mortality", "15 cigarettes", "loneliness ministers"],
        category="public_health",
        difficulty="medium",
    ),
    SummarizationCase(
        id="sum_med_10",
        text="Researchers at DeepMind have developed AlphaFold 3, which can now predict the structures of all biological molecules, not just proteins. The system can model DNA, RNA, ligands, and how they interact with each other. This breakthrough could accelerate drug discovery by years, as understanding molecular interactions is crucial for designing effective medications. The model's predictions are 50% more accurate than previous methods for protein-ligand interactions.",
        reference_summary="DeepMind's AlphaFold 3 predicts structures of all biological molecules (DNA, RNA, ligands) and their interactions, with 50% better accuracy for protein-ligand modeling, potentially accelerating drug discovery by years.",
        key_points=["AlphaFold 3", "all molecules", "DNA RNA ligands", "50% more accurate", "drug discovery"],
        category="ai_research",
        difficulty="medium",
    ),
]

HARD_CASES = [
    SummarizationCase(
        id="sum_hard_01",
        text="The debate over universal basic income (UBI) has intensified following results from the largest-ever randomized controlled trial. GiveDirectly's 12-year study in Kenya gave $22/month to 20,000 recipients. Results showed a 42% increase in household assets, 18% increase in earnings, and improved mental health. However, critics point to inflation in recipient villages, no significant increase in hours worked, and questions about scalability. Proponents argue the results prove UBI works as a floor, not a ceiling — it doesn't discourage work but provides stability for risk-taking and entrepreneurship. The cost of a US-wide UBI at $1,000/month would be approximately $3 trillion annually, roughly 13% of GDP, though proponents note it could replace existing welfare programs costing $1.2 trillion. The political feasibility remains the biggest obstacle, with support split along unusual lines — tech libertarians and progressive activists both support it, while fiscal conservatives and some labor unions oppose it.",
        reference_summary="The world's largest UBI trial (20,000 Kenyans, 12 years, $22/month) showed 42% more assets and 18% higher earnings with improved mental health, but raised concerns about inflation and scalability. A US-wide $1K/month UBI would cost $3T (13% GDP), partially offset by replacing $1.2T in existing welfare. Political support is unusually bipartisan but faces opposition from fiscal conservatives and some labor unions.",
        key_points=["12-year trial", "20,000 recipients", "42% assets", "$3 trillion US cost", "political feasibility"],
        category="economics",
        difficulty="hard",
        max_summary_length=150,
    ),
    SummarizationCase(
        id="sum_hard_02",
        text="The intersection of AI and climate change presents both promise and paradox. AI models are accelerating climate science — DeepMind's weather model is 1000x faster than traditional simulations, and AI-optimized power grids have reduced energy waste by 15% in pilot programs. However, training a single large language model produces 300 tons of CO2, equivalent to 125 round-trip flights from NY to London. Data centers now consume 1.5% of global electricity and are projected to reach 4% by 2030. Some researchers propose that AI's net climate impact will be positive — optimization savings will outweigh training costs by 100:1. Others argue this ignores the rebound effect, where efficiency gains lead to increased overall consumption. The truth likely depends on regulation, renewable energy adoption, and whether AI companies prioritize efficiency over scale.",
        reference_summary="AI both helps and hurts climate efforts: it accelerates climate science (1000x faster weather models, 15% grid optimization) but training large models emits 300 tons of CO2 each, with data centers consuming 1.5% of global electricity (4% by 2030). Net impact depends on regulation and whether optimization savings (potentially 100:1) outweigh the rebound effect of increased consumption.",
        key_points=["1000x faster", "300 tons CO2", "1.5% electricity", "100:1 ratio", "rebound effect"],
        category="ai_and_climate",
        difficulty="hard",
        max_summary_length=150,
    ),
    SummarizationCase(
        id="sum_hard_03",
        text="The future of work is being reshaped by three simultaneous trends: AI automation, demographic decline, and the gig economy. AI could automate 30% of current jobs by 2030 according to McKinsey, but the same period will see 85 million fewer working-age people globally due to falling birth rates. The gig economy now employs 36% of US workers but offers fewer benefits and less stability. These trends create contradictory pressures — automation threatens jobs while demographics create labor shortages. Policy solutions range from retraining programs and portable benefits to robot taxes and reduced work weeks. The countries that navigate this transition best will likely be those that invest in education while building social safety nets that don't depend on traditional employment.",
        reference_summary="Three converging trends reshape work: AI automating 30% of jobs by 2030, demographic decline (85M fewer workers), and the expanding gig economy (36% of US workers). The paradox of job automation during labor shortages requires balancing retraining, portable benefits, and safety nets independent of traditional employment.",
        key_points=["30% automation", "85 million fewer workers", "36% gig economy", "contradictory pressures", "portable benefits"],
        category="future_of_work",
        difficulty="hard",
        max_summary_length=150,
    ),
    SummarizationCase(
        id="sum_hard_04",
        text="Consciousness remains the hardest problem in science. While neuroscience has mapped brain regions to functions, the 'hard problem' — why physical processes give rise to subjective experience — remains unsolved. Integrated Information Theory (IIT) proposes that consciousness arises from integrated information, measurable as phi. Global Workspace Theory (GWT) suggests consciousness is a 'broadcast' mechanism. Recent experiments using adversarial collaboration have tested both theories simultaneously, and preliminary results favor GWT for some predictions and IIT for others. The implications extend to AI: if IIT is correct, sufficiently integrated AI systems might be conscious; if GWT is correct, current AI architectures lack the necessary global workspace. Some philosophers argue both theories miss the point — consciousness may be a fundamental property of matter, like mass or charge.",
        reference_summary="The hard problem of consciousness — why physical processes create subjective experience — remains unsolved. Two leading theories, IIT (integrated information) and GWT (broadcast mechanism), were tested adversarially with mixed results. The debate has AI implications: IIT suggests integrated AI could be conscious, while GWT says current architectures lack the necessary architecture. Some philosophers propose consciousness may be a fundamental property of matter.",
        key_points=["hard problem", "IIT vs GWT", "adversarial collaboration", "AI consciousness", "fundamental property"],
        category="philosophy_of_mind",
        difficulty="hard",
        max_summary_length=150,
    ),
    SummarizationCase(
        id="sum_hard_05",
        text="Central bank digital currencies (CBDCs) represent the most significant change to money since the end of the gold standard. 130 countries representing 98% of global GDP are exploring CBDCs. China's digital yuan has 260 million users. Proponents argue CBDCs enable instant payments, financial inclusion for the unbanked (1.4 billion globally), and better monetary policy transmission. Critics warn of surveillance risks — governments could track every transaction, freeze accounts without judicial oversight, and implement 'programmable money' that restricts how citizens spend. Cryptocurrency advocates argue CBDCs are the worst of both worlds — central control without privacy. The design choices being made now will determine whether CBDCs become tools of empowerment or control for decades to come.",
        reference_summary="130 countries (98% of GDP) are exploring CBDCs, with China's digital yuan at 260M users. Benefits include instant payments and financial inclusion for 1.4B unbanked people. Risks include government surveillance, account freezing, and programmable spending restrictions. Design choices now will determine if CBDCs empower or control citizens for decades.",
        key_points=["130 countries", "260M users", "1.4B unbanked", "surveillance risks", "programmable money"],
        category="fintech",
        difficulty="hard",
        max_summary_length=150,
    ),
    SummarizationCase(
        id="sum_hard_06",
        text="Antibiotic resistance has been called the next pandemic. Drug-resistant infections already kill 1.27 million people annually — more than HIV or malaria. By 2050, that number could reach 10 million, surpassing cancer. The economic cost is projected at $100 trillion in lost GDP. The problem is driven by overuse in healthcare (30% of prescriptions unnecessary), agriculture (70% of antibiotics used in livestock), and lack of new drugs (only 12 new antibiotics approved since 2017, none for the most dangerous resistant bacteria). Solutions include phage therapy, AI-driven drug discovery, rapid point-of-care diagnostics, and economic incentives like subscription models to make antibiotic development profitable again.",
        reference_summary="Antibiotic resistance kills 1.27M annually (potentially 10M by 2050, $100T GDP loss), driven by unnecessary prescriptions (30%), livestock use (70% of antibiotics), and only 12 new drugs since 2017. Solutions include phage therapy, AI drug discovery, rapid diagnostics, and subscription-model economic incentives.",
        key_points=["1.27M deaths", "10M by 2050", "70% livestock", "12 new drugs", "phage therapy"],
        category="global_health",
        difficulty="hard",
        max_summary_length=150,
    ),
    SummarizationCase(
        id="sum_hard_07",
        text="Nuclear fusion has achieved net energy gain for the second time at the National Ignition Facility, confirming the December 2022 breakthrough was not a fluke. The latest experiment produced 3.88 MJ from 2.05 MJ of laser energy — an energy gain of 1.89x. However, this only accounts for the fusion reaction itself; the entire facility uses 300 MJ per shot, meaning the system-level efficiency is about 1.3%. Private fusion companies like Commonwealth Fusion Systems and TAE Technologies claim they can achieve commercial fusion by 2035. Skeptics note that the engineering challenges — sustaining plasma at 100 million degrees, developing materials that can withstand neutron bombardment, and building tritium breeding blankets — may take decades longer. If achieved, fusion would provide virtually unlimited clean energy, fundamentally transforming civilization.",
        reference_summary="NIF achieved net fusion energy gain for the second time (1.89x), but system-level efficiency is only 1.3% (3.88 MJ output vs 300 MJ facility input). Private companies target commercial fusion by 2035, but engineering challenges with extreme temperatures, neutron-resistant materials, and tritium breeding may take decades. Success would mean unlimited clean energy.",
        key_points=["1.89x gain", "1.3% system efficiency", "300 MJ facility", "2035 target", "100M degrees"],
        category="energy",
        difficulty="hard",
        max_summary_length=150,
    ),
    SummarizationCase(
        id="sum_hard_08",
        text="The ethics of longevity research have become urgent as several interventions show promise. Metformin, a cheap diabetes drug, is being tested in the TAME trial (Targeting Aging with Metformin) involving 3,000 participants. Yamanaka factors can reprogram cells to a younger state. Senolytics clear senescent 'zombie' cells. The Hevolution Foundation has committed $1 billion to aging research. If these therapies work, they could add 10-20 healthy years. But the societal implications are staggering: pension systems designed for 15 years of retirement would face 35-year obligations. Population growth, already straining resources, would accelerate. Access would likely be unequal — creating a longevity divide between rich and poor nations. Some bioethicists argue that extending life is a fundamental right; others say resources should focus on extending healthspan in developing countries rather than lifespan in wealthy ones.",
        reference_summary="Longevity research (metformin TAME trial, Yamanaka factors, senolytics, $1B Hevolution Foundation) could add 10-20 healthy years. But implications include unsustainable pensions (35 vs 15 years), population pressure, and a rich-poor longevity divide. The ethical debate centers on whether to prioritize lifespan extension or healthspan equity across nations.",
        key_points=["metformin TAME trial", "Yamanaka factors", "10-20 years", "pension crisis", "longevity divide"],
        category="bioethics",
        difficulty="hard",
        max_summary_length=150,
    ),
    SummarizationCase(
        id="sum_hard_09",
        text="Space debris has reached a critical threshold. Over 36,000 objects larger than 10cm orbit Earth, along with millions of smaller fragments traveling at 28,000 km/h. The Kessler Syndrome — a cascading collision scenario that could make low Earth orbit unusable — is no longer theoretical. The 2021 Russian anti-satellite weapon test created 1,500 trackable fragments. SpaceX's Starlink constellation (5,000+ satellites) and competitors' plans could double orbital population by 2030. Proposed solutions include laser-powered deorbiting, magnetic capture systems, and the ESA's ClearSpace-1 mission. But there's no enforceable international law governing debris cleanup or constellation limits. The tragedy of the commons in space mirrors climate change: everyone benefits from access, but no one bears responsibility for cleanup.",
        reference_summary="Space debris (36,000+ large objects, millions of fragments at 28,000 km/h) risks the Kessler Syndrome of cascading collisions. Mega-constellations like Starlink could double orbital objects by 2030. Solutions include laser deorbiting and ESA's ClearSpace-1, but no international cleanup laws exist — mirroring climate change's tragedy of the commons.",
        key_points=["36,000 objects", "Kessler Syndrome", "28,000 km/h", "Starlink doubling", "no international law"],
        category="space_policy",
        difficulty="hard",
        max_summary_length=150,
    ),
    SummarizationCase(
        id="sum_hard_10",
        text="The rise of deepfakes poses an existential threat to information integrity. Current AI can generate photorealistic videos, clone voices from 3 seconds of audio, and produce text indistinguishable from human writing. In 2024, deepfake fraud cost businesses $26 billion globally. Political deepfakes have already influenced elections in multiple countries. Detection tools exist but face an arms race — each advance in detection is quickly countered by better generation. Proposed solutions include mandatory watermarking (C2PA standard), blockchain-verified provenance, and AI literacy education. However, some researchers argue we may be approaching a 'post-truth' equilibrium where no video or audio evidence is inherently trusted, fundamentally changing how society establishes facts, trusts institutions, and administers justice.",
        reference_summary="Deepfakes — photorealistic videos, 3-second voice clones, AI-generated text — cost $26B in fraud (2024) and influenced elections. Detection tools face an arms race with generation. Solutions include C2PA watermarking and blockchain provenance, but society may be approaching a 'post-truth equilibrium' where no media evidence is inherently trusted.",
        key_points=["$26B fraud", "3-second voice clone", "detection arms race", "C2PA watermarking", "post-truth"],
        category="information_security",
        difficulty="hard",
        max_summary_length=150,
    ),
]


ALL_SUMMARIZATION_CASES = EASY_CASES + MEDIUM_CASES + HARD_CASES

SUMMARIZATION_CASES_BY_DIFFICULTY = {
    "easy": EASY_CASES,
    "medium": MEDIUM_CASES,
    "hard": HARD_CASES,
}


def get_summarization_cases(difficulty=None):
    if difficulty is None:
        return ALL_SUMMARIZATION_CASES
    return SUMMARIZATION_CASES_BY_DIFFICULTY.get(difficulty, [])


def grade_summary(predicted: str, reference: str, key_points: list) -> float:
    """Grade a summary using key point coverage.

    - Each key point found: proportional score
    - Length penalty if too long (>2x reference)
    - Bonus for conciseness
    """
    if not predicted or not predicted.strip():
        return 0.0

    predicted_lower = predicted.lower()
    ref_lower = reference.lower()

    # Key point coverage (70% of score)
    hits = sum(1 for kp in key_points if kp.lower() in predicted_lower)
    coverage = hits / len(key_points) if key_points else 0.0

    # Length penalty (20% of score)
    pred_words = len(predicted.split())
    ref_words = len(reference.split())
    if ref_words > 0:
        length_ratio = pred_words / ref_words
        if length_ratio <= 1.5:
            length_score = 1.0
        elif length_ratio <= 3.0:
            length_score = 0.5
        else:
            length_score = 0.0
    else:
        length_score = 0.5

    # Basic quality (10% of score) — check if any reference words match
    ref_words_set = set(ref_lower.split())
    pred_words_set = set(predicted_lower.split())
    overlap = len(ref_words_set & pred_words_set) / len(ref_words_set) if ref_words_set else 0
    quality_score = min(1.0, overlap * 2)

    return round(0.7 * coverage + 0.2 * length_score + 0.1 * quality_score, 4)
