"""
Dual verifier pipeline:
  1. Claim extractor — decomposes response into atomic claims,
     checks each against retrieved passages via NLI-style prompting
  2. Factuality critic — holistic check for confabulation and
     unsupported assertions not covered by retrieved context

Both run in parallel. Results merged into a VerificationResult.
"""
import asyncio
import json
import re
from dataclasses import dataclass, field

from app.core.ollama_client import get_ollama_client
from app.rag.pipeline import RetrievedChunk
from config.settings import get_settings

settings = get_settings()


@dataclass
class ClaimVerdict:
    claim: str
    verdict: str          # "supported" | "unsupported" | "contradicted" | "unverifiable"
    confidence: float
    evidence: str = ""


@dataclass
class VerificationResult:
    passed: bool
    overall_confidence: float
    claim_verdicts: list[ClaimVerdict] = field(default_factory=list)
    critic_score: float = 0.0
    critic_feedback: str = ""
    needs_regeneration: bool = False
    regeneration_hint: str = ""

    def to_dict(self):
        return {
            "passed": self.passed,
            "overall_confidence": round(self.overall_confidence, 3),
            "critic_score": round(self.critic_score, 3),
            "critic_feedback": self.critic_feedback,
            "needs_regeneration": self.needs_regeneration,
            "regeneration_hint": self.regeneration_hint,
            "claim_verdicts": [
                {
                    "claim": v.claim,
                    "verdict": v.verdict,
                    "confidence": round(v.confidence, 3),
                    "evidence": v.evidence,
                }
                for v in self.claim_verdicts
            ],
        }


class ClaimExtractor:
    """
    Decomposes the response into atomic factual claims and checks
    each claim against the retrieved context using NLI-style prompting.
    """

    def __init__(self):
        self.ollama = get_ollama_client()

    async def extract_and_verify(
        self, response: str, context_chunks: list[RetrievedChunk]
    ) -> list[ClaimVerdict]:
        context_str = "\n\n".join(
            f"[Source: {c.source}]\n{c.content}" for c in context_chunks
        )

        prompt = f"""You are a fact-checking assistant. Given a response and retrieved context passages, do two things:

1. Extract all distinct factual claims from the response (ignore opinions/caveats)
2. For each claim, judge if it is: "supported", "unsupported", "contradicted", or "unverifiable" based on the context

Return ONLY valid JSON in this format:
{{
  "claims": [
    {{
      "claim": "The claim text",
      "verdict": "supported|unsupported|contradicted|unverifiable",
      "confidence": 0.0-1.0,
      "evidence": "brief quote or reason from context"
    }}
  ]
}}

CONTEXT:
{context_str[:3000]}

RESPONSE TO CHECK:
{response[:2000]}

Return ONLY the JSON object, no other text."""

        try:
            raw = await self.ollama.chat(
                model=settings.ollama_verifier_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            # Extract JSON from response
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                data = json.loads(match.group())
                verdicts = []
                for item in data.get("claims", []):
                    verdicts.append(
                        ClaimVerdict(
                            claim=item.get("claim", ""),
                            verdict=item.get("verdict", "unverifiable"),
                            confidence=float(item.get("confidence", 0.5)),
                            evidence=item.get("evidence", ""),
                        )
                    )
                return verdicts
        except Exception as e:
            pass

        # Fallback: return a single unverifiable claim
        return [ClaimVerdict(claim=response[:100], verdict="unverifiable", confidence=0.5)]


class FactualityCritic:
    """
    Holistic factuality critic that checks the full response for:
    - Internal contradictions
    - Overconfident assertions lacking support
    - Confabulated details (plausible-sounding but fabricated)
    """

    def __init__(self):
        self.ollama = get_ollama_client()

    async def critique(
        self, query: str, response: str, context_chunks: list[RetrievedChunk]
    ) -> tuple[float, str]:
        """Returns (score 0.0-1.0, feedback string)"""
        context_str = "\n\n".join(
            f"[Source: {c.source}]\n{c.content}" for c in context_chunks
        )

        prompt = f"""You are a critical factuality evaluator for an AI system. Your job is to detect hallucinations.

Evaluate the response for:
1. Internal contradictions
2. Overconfident claims not supported by the context
3. Specific numbers, names, dates that appear fabricated
4. Plausible-sounding but unverifiable assertions

Return ONLY valid JSON:
{{
  "factuality_score": 0.0-1.0,
  "issues": ["list of specific issues found, empty if none"],
  "recommendation": "approve|revise|reject",
  "feedback": "one sentence explaining the main concern or approval"
}}

QUERY: {query}

CONTEXT:
{context_str[:2000]}

RESPONSE:
{response[:2000]}

Return ONLY the JSON object."""

        try:
            raw = await self.ollama.chat(
                model=settings.ollama_verifier_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                data = json.loads(match.group())
                score = float(data.get("factuality_score", 0.7))
                feedback = data.get("feedback", "")
                issues = data.get("issues", [])
                if issues:
                    feedback = f"{feedback} Issues: {'; '.join(issues[:3])}"
                return score, feedback
        except Exception:
            pass

        return 0.7, "Critic evaluation unavailable."


class DualVerifier:
    """Runs ClaimExtractor and FactualityCritic in parallel."""

    def __init__(self):
        self.claim_extractor = ClaimExtractor()
        self.critic = FactualityCritic()

    async def verify(
        self,
        query: str,
        response: str,
        context_chunks: list[RetrievedChunk],
    ) -> VerificationResult:
        # Run both verifiers in parallel
        claim_task = self.claim_extractor.extract_and_verify(response, context_chunks)
        critic_task = self.critic.critique(query, response, context_chunks)

        verdicts, (critic_score, critic_feedback) = await asyncio.gather(
            claim_task, critic_task
        )

        # Compute claim-level confidence
        if verdicts:
            verdict_weights = {
                "supported": 1.0,
                "unverifiable": 0.6,
                "unsupported": 0.3,
                "contradicted": 0.0,
            }
            claim_score = sum(
                verdict_weights.get(v.verdict, 0.5) * v.confidence
                for v in verdicts
            ) / len(verdicts)
        else:
            claim_score = 0.6

        # Weighted average: claims 60%, critic 40%
        overall = (claim_score * 0.6) + (critic_score * 0.4)

        passed = overall >= settings.confidence_threshold
        needs_regen = not passed

        regen_hint = ""
        if needs_regen:
            contradicted = [v for v in verdicts if v.verdict == "contradicted"]
            unsupported = [v for v in verdicts if v.verdict == "unsupported"]
            if contradicted:
                regen_hint = f"Avoid contradicting: {contradicted[0].claim}"
            elif unsupported:
                regen_hint = f"Provide evidence for: {unsupported[0].claim}"
            else:
                regen_hint = critic_feedback

        return VerificationResult(
            passed=passed,
            overall_confidence=overall,
            claim_verdicts=verdicts,
            critic_score=critic_score,
            critic_feedback=critic_feedback,
            needs_regeneration=needs_regen,
            regeneration_hint=regen_hint,
        )


# Singleton
_verifier: DualVerifier | None = None


def get_verifier() -> DualVerifier:
    global _verifier
    if _verifier is None:
        _verifier = DualVerifier()
    return _verifier
