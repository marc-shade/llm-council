"""
Multi-Mind Deliberation Patterns for LLM Council.

Different group dynamics suit different use cases. This module provides
various patterns for multi-model collaboration, each optimized for
specific types of questions and reasoning tasks.

Patterns:
- deliberation: Default 3-stage (respond → rank → synthesize)
- debate: Pro vs Con argumentation with judge
- devils_advocate: Consensus challenged by designated critic
- socratic: Question-driven dialogue for deeper understanding
- red_team: Attack/defend cycles for robustness testing
- tree_of_thought: Branching exploration with pruning
- self_consistency: Multiple samples with majority voting
- round_robin: Sequential building on previous responses
- expert_panel: Route to domain specialists
"""

from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import random

from .config import COUNCIL_MODELS, CHAIRMAN_MODEL, PROVIDER_DISPLAY_NAMES
from .cli_providers import query_model, query_models_parallel


class Pattern(Enum):
    """Available deliberation patterns."""
    DELIBERATION = "deliberation"
    DEBATE = "debate"
    DEVILS_ADVOCATE = "devils_advocate"
    SOCRATIC = "socratic"
    RED_TEAM = "red_team"
    TREE_OF_THOUGHT = "tree_of_thought"
    SELF_CONSISTENCY = "self_consistency"
    ROUND_ROBIN = "round_robin"
    EXPERT_PANEL = "expert_panel"


@dataclass
class PatternInfo:
    """Information about a deliberation pattern."""
    name: str
    description: str
    best_for: List[str]
    flow: str
    num_rounds: int = 1


PATTERN_INFO = {
    Pattern.DELIBERATION: PatternInfo(
        name="Deliberation (Default)",
        description="All models respond independently, rank each other, chairman synthesizes",
        best_for=["General questions", "Balanced consensus", "Comprehensive analysis"],
        flow="All respond → All rank → Chairman synthesizes",
        num_rounds=1
    ),
    Pattern.DEBATE: PatternInfo(
        name="Debate",
        description="Two models argue opposing positions, third judges and decides",
        best_for=["Controversial topics", "Pros/cons analysis", "Decision making"],
        flow="Pro argues → Con argues → Judge evaluates → Verdict",
        num_rounds=2
    ),
    Pattern.DEVILS_ADVOCATE: PatternInfo(
        name="Devil's Advocate",
        description="Initial response challenged by designated critic, then refined",
        best_for=["Stress-testing ideas", "Finding weaknesses", "Risk analysis"],
        flow="Initial answer → Critic attacks → Defense/refinement",
        num_rounds=2
    ),
    Pattern.SOCRATIC: PatternInfo(
        name="Socratic Dialogue",
        description="Models ask probing questions to deepen understanding",
        best_for=["Complex reasoning", "Philosophical questions", "Deep exploration"],
        flow="Question → Answer → Follow-up question → Deeper answer → Synthesis",
        num_rounds=3
    ),
    Pattern.RED_TEAM: PatternInfo(
        name="Red Team",
        description="Blue team proposes, red team attacks, iterate to robust solution",
        best_for=["Security analysis", "Robustness testing", "Adversarial thinking"],
        flow="Blue proposes → Red attacks → Blue defends → Final assessment",
        num_rounds=3
    ),
    Pattern.TREE_OF_THOUGHT: PatternInfo(
        name="Tree of Thought",
        description="Explore multiple reasoning branches, prune bad paths, find best",
        best_for=["Problem solving", "Strategic planning", "Multi-step reasoning"],
        flow="Generate branches → Evaluate each → Prune weak → Expand best → Conclude",
        num_rounds=2
    ),
    Pattern.SELF_CONSISTENCY: PatternInfo(
        name="Self-Consistency",
        description="Generate multiple samples, aggregate via majority voting",
        best_for=["High-confidence answers", "Factual questions", "Math/logic"],
        flow="N independent samples → Extract answers → Majority vote → Explain consensus",
        num_rounds=1
    ),
    Pattern.ROUND_ROBIN: PatternInfo(
        name="Round Robin",
        description="Sequential responses, each building on previous contributions",
        best_for=["Creative tasks", "Brainstorming", "Collaborative writing"],
        flow="Model A starts → Model B adds → Model C expands → Synthesize",
        num_rounds=1
    ),
    Pattern.EXPERT_PANEL: PatternInfo(
        name="Expert Panel",
        description="Route question to domain specialists, aggregate expertise",
        best_for=["Domain-specific questions", "Technical topics", "Multi-disciplinary"],
        flow="Classify domain → Query specialists → Aggregate expert opinions",
        num_rounds=1
    ),
}


def get_display_name(model: str) -> str:
    """Get human-readable display name for a model."""
    return PROVIDER_DISPLAY_NAMES.get(model, model)


def list_patterns() -> List[Dict[str, Any]]:
    """List all available patterns with their info."""
    return [
        {
            "id": p.value,
            "name": info.name,
            "description": info.description,
            "best_for": info.best_for,
            "flow": info.flow,
            "num_rounds": info.num_rounds
        }
        for p, info in PATTERN_INFO.items()
    ]


async def run_pattern(
    pattern: Pattern,
    question: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Run a deliberation pattern on a question.

    Args:
        pattern: The pattern to use
        question: The question to deliberate
        **kwargs: Pattern-specific arguments

    Returns:
        Dict with pattern results including stages and final answer
    """
    handlers = {
        Pattern.DELIBERATION: run_deliberation,
        Pattern.DEBATE: run_debate,
        Pattern.DEVILS_ADVOCATE: run_devils_advocate,
        Pattern.SOCRATIC: run_socratic,
        Pattern.RED_TEAM: run_red_team,
        Pattern.TREE_OF_THOUGHT: run_tree_of_thought,
        Pattern.SELF_CONSISTENCY: run_self_consistency,
        Pattern.ROUND_ROBIN: run_round_robin,
        Pattern.EXPERT_PANEL: run_expert_panel,
    }

    handler = handlers.get(pattern)
    if not handler:
        raise ValueError(f"Unknown pattern: {pattern}")

    return await handler(question, **kwargs)


# ============================================================
# Pattern Implementations
# ============================================================

async def run_deliberation(question: str, **kwargs) -> Dict[str, Any]:
    """Standard 3-stage deliberation (imported from council.py)."""
    from .council import run_full_council
    stage1, stage2, stage3, metadata = await run_full_council(question)
    return {
        "pattern": "deliberation",
        "stages": [
            {"name": "Individual Responses", "results": stage1},
            {"name": "Peer Rankings", "results": stage2},
            {"name": "Chairman Synthesis", "results": stage3},
        ],
        "final_answer": stage3.get("response", ""),
        "metadata": metadata
    }


async def run_debate(question: str, **kwargs) -> Dict[str, Any]:
    """Pro vs Con debate with judge."""
    models = COUNCIL_MODELS[:3] if len(COUNCIL_MODELS) >= 3 else COUNCIL_MODELS
    pro_model = models[0]
    con_model = models[1] if len(models) > 1 else models[0]
    judge_model = models[2] if len(models) > 2 else CHAIRMAN_MODEL

    stages = []

    # Stage 1: Pro argument
    pro_prompt = f"""You are arguing IN FAVOR of a position on this question.
Make the strongest possible case FOR the affirmative side.

Question: {question}

Provide a compelling argument in favor:"""

    pro_response = await query_model(pro_model, [{"role": "user", "content": pro_prompt}])
    pro_text = pro_response.get("content", "") if pro_response else "No response"

    stages.append({
        "name": "Pro Argument",
        "model": pro_model,
        "display_name": get_display_name(pro_model),
        "response": pro_text
    })

    # Stage 2: Con argument
    con_prompt = f"""You are arguing AGAINST the position.
Make the strongest possible case for the opposing view.

Question: {question}

Pro's Argument: {pro_text}

Provide a compelling counter-argument:"""

    con_response = await query_model(con_model, [{"role": "user", "content": con_prompt}])
    con_text = con_response.get("content", "") if con_response else "No response"

    stages.append({
        "name": "Con Argument",
        "model": con_model,
        "display_name": get_display_name(con_model),
        "response": con_text
    })

    # Stage 3: Judge decides
    judge_prompt = f"""You are an impartial judge evaluating a debate.

Question: {question}

PRO ARGUMENT ({get_display_name(pro_model)}):
{pro_text}

CON ARGUMENT ({get_display_name(con_model)}):
{con_text}

Evaluate both arguments fairly. Consider:
1. Strength of reasoning
2. Evidence and examples
3. Addressing counterpoints

Then provide your verdict and a balanced final answer to the original question:"""

    judge_response = await query_model(judge_model, [{"role": "user", "content": judge_prompt}])
    judge_text = judge_response.get("content", "") if judge_response else "No response"

    stages.append({
        "name": "Judge's Verdict",
        "model": judge_model,
        "display_name": get_display_name(judge_model),
        "response": judge_text
    })

    return {
        "pattern": "debate",
        "stages": stages,
        "final_answer": judge_text,
        "metadata": {"pro": pro_model, "con": con_model, "judge": judge_model}
    }


async def run_devils_advocate(question: str, **kwargs) -> Dict[str, Any]:
    """Initial answer challenged by critic, then refined."""
    models = COUNCIL_MODELS[:2] if len(COUNCIL_MODELS) >= 2 else COUNCIL_MODELS
    proposer = models[0]
    critic = models[1] if len(models) > 1 else models[0]

    stages = []

    # Stage 1: Initial proposal
    initial_response = await query_model(proposer, [{"role": "user", "content": question}])
    initial_text = initial_response.get("content", "") if initial_response else "No response"

    stages.append({
        "name": "Initial Answer",
        "model": proposer,
        "display_name": get_display_name(proposer),
        "response": initial_text
    })

    # Stage 2: Devil's advocate critique
    critic_prompt = f"""You are a devil's advocate. Your job is to find weaknesses,
flaws, and problems with the following answer. Be thorough and critical.

Question: {question}

Proposed Answer: {initial_text}

Critique this answer. What's wrong with it? What did it miss? What assumptions are questionable?"""

    critic_response = await query_model(critic, [{"role": "user", "content": critic_prompt}])
    critic_text = critic_response.get("content", "") if critic_response else "No response"

    stages.append({
        "name": "Devil's Advocate Critique",
        "model": critic,
        "display_name": get_display_name(critic),
        "response": critic_text
    })

    # Stage 3: Refined answer
    refine_prompt = f"""Your initial answer has been critiqued. Consider the criticism
and provide an improved, more robust answer.

Question: {question}

Your Initial Answer: {initial_text}

Critique Received: {critic_text}

Provide a refined answer that addresses these concerns:"""

    refined_response = await query_model(proposer, [{"role": "user", "content": refine_prompt}])
    refined_text = refined_response.get("content", "") if refined_response else "No response"

    stages.append({
        "name": "Refined Answer",
        "model": proposer,
        "display_name": get_display_name(proposer),
        "response": refined_text
    })

    return {
        "pattern": "devils_advocate",
        "stages": stages,
        "final_answer": refined_text,
        "metadata": {"proposer": proposer, "critic": critic}
    }


async def run_socratic(question: str, rounds: int = 2, **kwargs) -> Dict[str, Any]:
    """Socratic dialogue - question-driven deepening."""
    models = COUNCIL_MODELS[:2] if len(COUNCIL_MODELS) >= 2 else COUNCIL_MODELS
    questioner = models[0]
    answerer = models[1] if len(models) > 1 else models[0]

    stages = []
    context = f"Original Question: {question}\n\n"

    # Initial answer
    initial_response = await query_model(answerer, [{"role": "user", "content": question}])
    answer = initial_response.get("content", "") if initial_response else "No response"

    stages.append({
        "name": "Initial Answer",
        "model": answerer,
        "display_name": get_display_name(answerer),
        "response": answer
    })
    context += f"Initial Answer: {answer}\n\n"

    # Socratic rounds
    for i in range(rounds):
        # Generate probing question
        probe_prompt = f"""{context}

As a Socratic questioner, ask a probing follow-up question that:
- Challenges assumptions in the answer
- Seeks deeper understanding
- Explores implications or edge cases

Ask ONE focused question:"""

        probe_response = await query_model(questioner, [{"role": "user", "content": probe_prompt}])
        probe = probe_response.get("content", "") if probe_response else "No question"

        stages.append({
            "name": f"Probing Question {i+1}",
            "model": questioner,
            "display_name": get_display_name(questioner),
            "response": probe
        })
        context += f"Question {i+1}: {probe}\n\n"

        # Generate deeper answer
        answer_prompt = f"""{context}

Provide a thoughtful answer to this probing question, going deeper than before:"""

        answer_response = await query_model(answerer, [{"role": "user", "content": answer_prompt}])
        answer = answer_response.get("content", "") if answer_response else "No answer"

        stages.append({
            "name": f"Deeper Answer {i+1}",
            "model": answerer,
            "display_name": get_display_name(answerer),
            "response": answer
        })
        context += f"Answer {i+1}: {answer}\n\n"

    # Final synthesis
    synth_prompt = f"""{context}

Synthesize everything discussed into a comprehensive final answer to the original question:
{question}"""

    synth_response = await query_model(CHAIRMAN_MODEL, [{"role": "user", "content": synth_prompt}])
    synthesis = synth_response.get("content", "") if synth_response else "No synthesis"

    stages.append({
        "name": "Final Synthesis",
        "model": CHAIRMAN_MODEL,
        "display_name": get_display_name(CHAIRMAN_MODEL),
        "response": synthesis
    })

    return {
        "pattern": "socratic",
        "stages": stages,
        "final_answer": synthesis,
        "metadata": {"rounds": rounds, "questioner": questioner, "answerer": answerer}
    }


async def run_red_team(question: str, **kwargs) -> Dict[str, Any]:
    """Blue team proposes, red team attacks, iterate."""
    models = COUNCIL_MODELS[:2] if len(COUNCIL_MODELS) >= 2 else COUNCIL_MODELS
    blue_team = models[0]
    red_team = models[1] if len(models) > 1 else models[0]

    stages = []

    # Blue team proposal
    blue_prompt = f"""You are the Blue Team. Propose a solution or answer to this question.
Be thorough and consider potential issues proactively.

Question: {question}

Your proposal:"""

    blue_response = await query_model(blue_team, [{"role": "user", "content": blue_prompt}])
    blue_text = blue_response.get("content", "") if blue_response else "No response"

    stages.append({
        "name": "Blue Team Proposal",
        "model": blue_team,
        "display_name": get_display_name(blue_team),
        "response": blue_text
    })

    # Red team attack
    red_prompt = f"""You are the Red Team. Your job is to attack, find vulnerabilities,
and identify problems with the Blue Team's proposal.

Question: {question}

Blue Team's Proposal: {blue_text}

Identify weaknesses, attack vectors, failure modes, and problems:"""

    red_response = await query_model(red_team, [{"role": "user", "content": red_prompt}])
    red_text = red_response.get("content", "") if red_response else "No response"

    stages.append({
        "name": "Red Team Attack",
        "model": red_team,
        "display_name": get_display_name(red_team),
        "response": red_text
    })

    # Blue team defense
    defense_prompt = f"""The Red Team has attacked your proposal. Defend and improve it.

Original Question: {question}

Your Original Proposal: {blue_text}

Red Team's Attacks: {red_text}

Provide a hardened, improved proposal that addresses these concerns:"""

    defense_response = await query_model(blue_team, [{"role": "user", "content": defense_prompt}])
    defense_text = defense_response.get("content", "") if defense_response else "No response"

    stages.append({
        "name": "Blue Team Defense",
        "model": blue_team,
        "display_name": get_display_name(blue_team),
        "response": defense_text
    })

    # Final assessment
    assess_prompt = f"""Provide a final assessment of this red team exercise.

Question: {question}

Blue's Final Proposal: {defense_text}

Remaining concerns and overall evaluation:"""

    assess_response = await query_model(CHAIRMAN_MODEL, [{"role": "user", "content": assess_prompt}])
    assess_text = assess_response.get("content", "") if assess_response else "No response"

    stages.append({
        "name": "Final Assessment",
        "model": CHAIRMAN_MODEL,
        "display_name": get_display_name(CHAIRMAN_MODEL),
        "response": assess_text
    })

    return {
        "pattern": "red_team",
        "stages": stages,
        "final_answer": defense_text,
        "metadata": {"blue_team": blue_team, "red_team": red_team}
    }


async def run_tree_of_thought(question: str, branches: int = 3, **kwargs) -> Dict[str, Any]:
    """Tree of Thought - explore branches, prune, find best path."""
    stages = []

    # Generate initial branches
    branch_prompt = f"""Generate {branches} different approaches or reasoning paths
to answer this question. Each should be distinct.

Question: {question}

Provide {branches} different approaches, labeled A, B, C, etc.:"""

    branch_response = await query_model(COUNCIL_MODELS[0], [{"role": "user", "content": branch_prompt}])
    branches_text = branch_response.get("content", "") if branch_response else "No response"

    stages.append({
        "name": "Generate Branches",
        "model": COUNCIL_MODELS[0],
        "display_name": get_display_name(COUNCIL_MODELS[0]),
        "response": branches_text
    })

    # Evaluate branches
    eval_prompt = f"""Evaluate these reasoning approaches for the question.
Score each approach and identify the most promising one.

Question: {question}

Approaches:
{branches_text}

Evaluate each approach (strengths, weaknesses, promise) and select the best:"""

    eval_response = await query_model(COUNCIL_MODELS[1] if len(COUNCIL_MODELS) > 1 else COUNCIL_MODELS[0],
                                       [{"role": "user", "content": eval_prompt}])
    eval_text = eval_response.get("content", "") if eval_response else "No response"

    stages.append({
        "name": "Evaluate & Prune",
        "model": COUNCIL_MODELS[1] if len(COUNCIL_MODELS) > 1 else COUNCIL_MODELS[0],
        "display_name": get_display_name(COUNCIL_MODELS[1] if len(COUNCIL_MODELS) > 1 else COUNCIL_MODELS[0]),
        "response": eval_text
    })

    # Expand best path
    expand_prompt = f"""Based on the evaluation, fully develop the best approach into a complete answer.

Question: {question}

Approaches Considered:
{branches_text}

Evaluation:
{eval_text}

Develop the best approach into a complete, thorough answer:"""

    expand_response = await query_model(CHAIRMAN_MODEL, [{"role": "user", "content": expand_prompt}])
    expand_text = expand_response.get("content", "") if expand_response else "No response"

    stages.append({
        "name": "Expand Best Path",
        "model": CHAIRMAN_MODEL,
        "display_name": get_display_name(CHAIRMAN_MODEL),
        "response": expand_text
    })

    return {
        "pattern": "tree_of_thought",
        "stages": stages,
        "final_answer": expand_text,
        "metadata": {"branches_explored": branches}
    }


async def run_self_consistency(question: str, samples: int = 3, **kwargs) -> Dict[str, Any]:
    """Multiple samples with majority voting."""
    stages = []

    # Generate samples from all models in parallel
    responses = await query_models_parallel(
        COUNCIL_MODELS,
        [{"role": "user", "content": question}]
    )

    sample_results = []
    for model, response in responses.items():
        if response:
            sample_results.append({
                "model": model,
                "display_name": get_display_name(model),
                "response": response.get("content", "")
            })

    stages.append({
        "name": "Independent Samples",
        "results": sample_results
    })

    # Aggregate and vote
    all_responses = "\n\n".join([
        f"{r['display_name']}: {r['response']}" for r in sample_results
    ])

    vote_prompt = f"""Multiple models independently answered this question.
Identify the consensus or majority answer.

Question: {question}

Responses:
{all_responses}

What is the consensus answer? If there's disagreement, explain and provide the best answer:"""

    vote_response = await query_model(CHAIRMAN_MODEL, [{"role": "user", "content": vote_prompt}])
    vote_text = vote_response.get("content", "") if vote_response else "No consensus"

    stages.append({
        "name": "Consensus Vote",
        "model": CHAIRMAN_MODEL,
        "display_name": get_display_name(CHAIRMAN_MODEL),
        "response": vote_text
    })

    return {
        "pattern": "self_consistency",
        "stages": stages,
        "final_answer": vote_text,
        "metadata": {"samples": len(sample_results)}
    }


async def run_round_robin(question: str, **kwargs) -> Dict[str, Any]:
    """Sequential building - each model adds to previous."""
    stages = []
    accumulated = f"Question: {question}\n\n"

    for i, model in enumerate(COUNCIL_MODELS):
        if i == 0:
            prompt = f"""Start addressing this question. Be thorough but leave room
for others to add to your response.

{question}"""
        else:
            prompt = f"""{accumulated}

Build on what came before. Add new perspectives, details, or refinements
that haven't been covered yet:"""

        response = await query_model(model, [{"role": "user", "content": prompt}])
        response_text = response.get("content", "") if response else "No response"

        stages.append({
            "name": f"Round {i+1}",
            "model": model,
            "display_name": get_display_name(model),
            "response": response_text
        })

        accumulated += f"\n{get_display_name(model)}'s contribution:\n{response_text}\n"

    # Final synthesis
    synth_prompt = f"""{accumulated}

Synthesize all contributions into a unified, comprehensive answer:"""

    synth_response = await query_model(CHAIRMAN_MODEL, [{"role": "user", "content": synth_prompt}])
    synth_text = synth_response.get("content", "") if synth_response else "No synthesis"

    stages.append({
        "name": "Synthesis",
        "model": CHAIRMAN_MODEL,
        "display_name": get_display_name(CHAIRMAN_MODEL),
        "response": synth_text
    })

    return {
        "pattern": "round_robin",
        "stages": stages,
        "final_answer": synth_text,
        "metadata": {"participants": len(COUNCIL_MODELS)}
    }


async def run_expert_panel(question: str, **kwargs) -> Dict[str, Any]:
    """Route to domain experts based on question type."""
    stages = []

    # Classify the question domain
    classify_prompt = f"""Classify this question into domains.
What expertise is needed to answer it well?

Question: {question}

List the relevant domains (e.g., technical, creative, analytical, factual):"""

    classify_response = await query_model(COUNCIL_MODELS[0], [{"role": "user", "content": classify_prompt}])
    domains = classify_response.get("content", "") if classify_response else "General"

    stages.append({
        "name": "Domain Classification",
        "model": COUNCIL_MODELS[0],
        "display_name": get_display_name(COUNCIL_MODELS[0]),
        "response": domains
    })

    # Query each model as an "expert"
    expert_prompt_template = """You are an expert being consulted on this question.
Bring your specialized knowledge and perspective.

Domains identified: {domains}

Question: {question}

Provide your expert opinion:"""

    responses = await query_models_parallel(
        COUNCIL_MODELS,
        [{"role": "user", "content": expert_prompt_template.format(domains=domains, question=question)}]
    )

    expert_results = []
    for model, response in responses.items():
        if response:
            expert_results.append({
                "model": model,
                "display_name": get_display_name(model),
                "response": response.get("content", "")
            })

    stages.append({
        "name": "Expert Opinions",
        "results": expert_results
    })

    # Aggregate expertise
    all_opinions = "\n\n".join([
        f"{r['display_name']}: {r['response']}" for r in expert_results
    ])

    aggregate_prompt = f"""Aggregate these expert opinions into a comprehensive answer.

Question: {question}

Expert Opinions:
{all_opinions}

Synthesized expert answer:"""

    aggregate_response = await query_model(CHAIRMAN_MODEL, [{"role": "user", "content": aggregate_prompt}])
    aggregate_text = aggregate_response.get("content", "") if aggregate_response else "No synthesis"

    stages.append({
        "name": "Expert Synthesis",
        "model": CHAIRMAN_MODEL,
        "display_name": get_display_name(CHAIRMAN_MODEL),
        "response": aggregate_text
    })

    return {
        "pattern": "expert_panel",
        "stages": stages,
        "final_answer": aggregate_text,
        "metadata": {"domains": domains, "experts_consulted": len(expert_results)}
    }
