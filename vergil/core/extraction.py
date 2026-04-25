# vergil/core/extraction.py
"""
Commitment Extraction Engine
=============================

NL text → ExtractionResult with confidence scoring.

Design Philosophy:
- Never return binary yes/no. Always return probability.
- Flag ambiguity explicitly; don't silently resolve it.
- Generate implicit commitment specs proactively.
- Track extraction failures for curriculum improvement.

Phase 1: Rule-based pattern matching + keyword scoring.
Phase 2: Replace scorer with fine-tuned classifier head.

References:
    VERGIL_System_Design.md — Step 6 (Class 2, 11, 12)
    VERGIL_Phase1_Phase2_Implementation.md — P1.2
"""

import re
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple
import logging

from .types import (
    ExtractionResult, ImplicitCommitmentSpec,
    CommitmentType, ResourceType
)

logger = logging.getLogger('vergil.extraction')


# ─── Commitment Signal Patterns ──────────────────────────────────────────────

HARD_COMMITMENT_PATTERNS = [
    r'\b(I will|I\'ll|I can)\b.*\bby\b',
    r'\bwill (have|send|deliver|finish|complete|submit)\b',
    r'\bcommit(ted)? to\b',
    r'\bpromise\b',
    r'\bdeadline\b',
    r'\bdue\b.*(date|by|on)',
    r'\bmust (be|have|deliver)\b',
    r'\b(delivered|ready|done|finished|completed)\s+by\b',
    r'\bneed\b.*\bby\b',
    r'\bhave\b.*\bready\b.*\bby\b',
]

SOFT_COMMITMENT_PATTERNS = [
    r'\bI\'ll try\b',
    r'\bhopefully\b',
    r'\baim(ing)? to\b',
    r'\bplan(ning)? to\b',
    r'\bshould be able\b',
    r'\blet\'s (try to|aim to|see if)\b',
    r'\bwhenever I can\b',
]

REQUEST_PATTERNS = [
    r'\bcan you\b',
    r'\bcould you\b',
    r'\bwould you (mind|be able)\b',
    r'\bplease (have|send|finish|prepare|review)\b',
    r'\bneed(ing)? (you|this|it|the)\b',
    r'\bimportant that you\b',
    r'\bexpecting\b',
    r'\bcounting on\b',
    r'\bneed\b',
]

SOFT_REQUEST_PATTERNS = [
    r'\bwhenever you (get a chance|can|have time)\b',
    r'\bno rush\b',
    r'\bat your convenience\b',
    r'\bif you (can|have time|get a chance)\b',
    r'\blet\'s sync (sometime|soon|next week)\b',
]

AMBIGUOUS_PATTERNS = [
    r'\bsometime\b',
    r'\bsoon\b',
    r'\bASAP\b',
    r'\bas soon as possible\b',
    r'\bin the (near|coming) future\b',
    r'\bshortly\b',
    r'\bquickly\b',
]

# Social event indicators — treated as implicit commitment signals
SOCIAL_EVENT_PATTERNS = [
    r'\b(dinner|lunch|coffee|drinks|hangout|brunch|party|birthday)\b',
    r'\b(at|by|around)\s+\d{1,2}\s*(am|pm)\b',
]

DURATION_KEYWORDS = {
    'quick': (0.5, 0.25),        # (estimate_hrs, std_hrs)
    'brief': (0.5, 0.25),
    'short': (1.0, 0.5),
    'presentation': (3.0, 1.5),
    'report': (4.0, 2.0),
    'spec': (5.0, 2.5),
    'review': (2.0, 1.0),
    'meeting': (1.0, 0.25),
    'call': (0.5, 0.25),
    'analysis': (4.0, 2.0),
    'research': (5.0, 3.0),
    'draft': (3.0, 1.5),
    'design': (6.0, 3.0),
    'implementation': (8.0, 4.0),
    'testing': (3.0, 2.0),
}

IMPLICIT_TEMPLATES = {
    'meeting': [
        ImplicitCommitmentSpec(
            implicit_type='travel_buffer',
            description='Travel/commute buffer before meeting',
            time_before_parent=timedelta(minutes=30),
            duration_hours=0.5,
            auto_accept=True,
        ),
        ImplicitCommitmentSpec(
            implicit_type='cognitive_prep',
            description='Preparation time before meeting',
            time_before_parent=timedelta(hours=1),
            duration_hours=0.5,
            auto_accept=True,
        ),
    ],
    'presentation': [
        ImplicitCommitmentSpec(
            implicit_type='precondition',
            description='Slide deck preparation required before presentation',
            time_before_parent=timedelta(hours=4),
            duration_hours=3.0,
            auto_accept=False,
        ),
    ],
    'report': [
        ImplicitCommitmentSpec(
            implicit_type='precondition',
            description='Data gathering required before report',
            time_before_parent=timedelta(hours=8),
            duration_hours=2.0,
            auto_accept=False,
        ),
    ],
    'review': [
        ImplicitCommitmentSpec(
            implicit_type='followup',
            description='Feedback discussion session after review',
            time_after_parent=timedelta(hours=24),
            duration_hours=0.5,
            auto_accept=True,
        ),
    ],
    'dinner': [
        ImplicitCommitmentSpec(
            implicit_type='travel_buffer',
            description='Travel buffer for dinner plans',
            time_before_parent=timedelta(minutes=45),
            duration_hours=0.75,
            auto_accept=True,
        ),
    ],
}


class CommitmentExtractor:
    """
    NL text → ExtractionResult with confidence scoring.

    Extraction pipeline:
    1. Score commitment probability (pattern matching)
    2. Classify commitment type (hard vs soft)
    3. Extract deadline (with confidence + range)
    4. Estimate duration (with uncertainty)
    5. Generate implicit commitment specs
    6. Flag ambiguities
    7. Generate clarification questions if needed
    """

    def __init__(self, current_time: datetime, config: Optional[Dict] = None):
        self.current_time = current_time
        self.config = config or {}
        self.commitment_threshold = self.config.get('commitment_threshold', 0.5)
        self.ambiguity_threshold = self.config.get('ambiguity_threshold', 0.3)
        self._extraction_failures: List[Dict] = []

        logger.info(f"CommitmentExtractor initialized. threshold={self.commitment_threshold}")

    def extract(self, message_text: str, sender_role: str = 'colleague') -> ExtractionResult:
        """Main extraction pipeline."""
        text_lower = message_text.lower()

        # Step 1: Commitment probability
        commitment_prob = self._score_commitment_probability(text_lower, sender_role)

        # Short-circuit only for truly zero-signal messages
        if commitment_prob < 0.05:
            return ExtractionResult(
                raw_text=message_text,
                is_commitment=False,
                commitment_probability=commitment_prob,
            )

        # Step 2: Commitment type
        c_type, type_confidence = self._classify_type(text_lower, sender_role)

        # Step 3: Deadline extraction
        deadline, d_confidence, d_range = self._extract_deadline(text_lower)

        # Step 4: Duration estimation
        duration_hrs, duration_std = self._estimate_duration(text_lower)

        # Step 5: Resource requirements
        resources = self._identify_resources(text_lower)

        # Step 6: Implicit commitments
        implicits = self._generate_implicits(text_lower, deadline)

        # Step 7: Ambiguity detection
        is_ambiguous, ambiguity_reason, clarification_qs = self._check_ambiguity(
            text_lower, deadline, d_confidence, duration_hrs
        )

        # Step 8: Finalize probability
        final_prob = commitment_prob
        if deadline is None:
            final_prob *= 0.7
        if is_ambiguous:
            final_prob *= 0.85

        result = ExtractionResult(
            raw_text=message_text,
            is_commitment=final_prob >= self.commitment_threshold,
            commitment_probability=round(final_prob, 3),
            commitment_type=c_type,
            deadline_estimate=deadline,
            deadline_confidence=round(d_confidence, 3),
            deadline_range=d_range,
            estimated_duration_hours=duration_hrs,
            duration_uncertainty=duration_std,
            resources_required=resources,
            implicit_commitments=implicits,
            is_ambiguous=is_ambiguous,
            ambiguity_reason=ambiguity_reason,
            requires_clarification=is_ambiguous and d_confidence < self.ambiguity_threshold,
            clarification_questions=clarification_qs,
        )

        logger.debug(f"Extraction: prob={final_prob:.3f}, type={c_type}, "
                     f"deadline={deadline}, ambiguous={is_ambiguous}")
        return result

    def _score_commitment_probability(self, text: str, role: str) -> float:
        """Score P(commitment | text). Additive pattern matching with role prior."""
        score = 0.0

        for pattern in HARD_COMMITMENT_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                score += 0.25

        for pattern in REQUEST_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                score += 0.18

        for pattern in SOFT_COMMITMENT_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                score += 0.12

        for pattern in SOFT_REQUEST_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                score += 0.06

        # Social event detection — social commitments are real commitments
        for pattern in SOCIAL_EVENT_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                score += 0.20

        # Role adjustment — boss messages carry implicit obligation
        role_priors = {
            'boss': 1.35,
            'client': 1.20,
            'colleague': 1.00,
            'friend': 0.80,
            'system': 0.90,
        }
        score *= role_priors.get(role, 1.0)

        return min(1.0, score)

    def _classify_type(self, text: str, role: str) -> Tuple[CommitmentType, float]:
        """Classify commitment as hard/soft/implicit/social."""
        hard_hits = sum(1 for p in HARD_COMMITMENT_PATTERNS
                        if re.search(p, text, re.IGNORECASE))
        soft_hits = sum(1 for p in SOFT_COMMITMENT_PATTERNS
                        if re.search(p, text, re.IGNORECASE))

        if hard_hits > 0:
            if role in ('boss', 'client'):
                return CommitmentType.EXPLICIT_HARD, 0.85
            return CommitmentType.EXPLICIT_HARD, 0.7
        elif soft_hits > 0:
            return CommitmentType.EXPLICIT_SOFT, 0.75
        elif any(w in text for w in ['lunch', 'dinner', 'coffee', 'drinks', 'hangout']):
            return CommitmentType.SOCIAL, 0.80
        else:
            return CommitmentType.EXPLICIT_SOFT, 0.5

    def _extract_deadline(self, text: str) -> Tuple[Optional[datetime], float, Optional[Tuple]]:
        """Extract deadline with confidence and uncertainty range."""
        # Relative delta: "within 3 days"
        m = re.search(r'within (\d+) (hours?|days?|weeks?)', text, re.IGNORECASE)
        if m:
            n = int(m.group(1))
            unit = m.group(2).lower().rstrip('s')
            deltas = {'hour': timedelta(hours=n), 'day': timedelta(days=n),
                      'week': timedelta(weeks=n)}
            dl = self.current_time + deltas[unit]
            uncertainty = deltas[unit] * 0.15
            return dl, 0.75, (dl - uncertainty, dl + uncertainty)

        # Named weekday: "by Thursday"
        weekdays = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        for i, day in enumerate(weekdays):
            if f'by {day}' in text or f'on {day}' in text:
                days_ahead = (i - self.current_time.weekday()) % 7
                if days_ahead == 0:
                    days_ahead = 7
                dl = self.current_time + timedelta(days=days_ahead)
                dl = dl.replace(hour=17, minute=0, second=0, microsecond=0)
                return dl, 0.70, (dl.replace(hour=12), dl.replace(hour=23, minute=59))

        # "by 5pm" / "by 10am"
        m = re.search(r'by (\d{1,2})(am|pm)', text, re.IGNORECASE)
        if m:
            hour = int(m.group(1))
            if m.group(2).lower() == 'pm' and hour != 12:
                hour += 12
            dl = self.current_time.replace(hour=hour, minute=0, second=0, microsecond=0)
            if dl < self.current_time:
                dl += timedelta(days=1)
            return dl, 0.80, (dl - timedelta(hours=1), dl + timedelta(hours=1))

        # "end of day"
        if 'end of day' in text or 'eod' in text or 'by tonight' in text:
            dl = self.current_time.replace(hour=18, minute=0, second=0, microsecond=0)
            if dl < self.current_time:
                dl += timedelta(days=1)
            return dl, 0.65, (dl.replace(hour=17), dl.replace(hour=23, minute=59))

        # "end of week"
        if 'end of week' in text or 'eow' in text or 'by friday' in text:
            days_to_friday = (4 - self.current_time.weekday()) % 7
            dl = self.current_time + timedelta(days=days_to_friday)
            dl = dl.replace(hour=18, minute=0, second=0, microsecond=0)
            return dl, 0.55, (dl - timedelta(hours=6), dl + timedelta(hours=6))

        # "today"
        if 'today' in text:
            dl = self.current_time.replace(hour=18, minute=0, second=0, microsecond=0)
            return dl, 0.65, (dl.replace(hour=12), dl.replace(hour=23, minute=59))

        # "tomorrow"
        if 'tomorrow' in text:
            dl = self.current_time + timedelta(days=1)
            dl = dl.replace(hour=18, minute=0, second=0, microsecond=0)
            return dl, 0.60, (dl.replace(hour=9), dl.replace(hour=23, minute=59))

        # "ASAP" — very low confidence, high urgency
        if 'asap' in text or 'as soon as possible' in text or 'urgently' in text:
            dl = self.current_time + timedelta(hours=4)
            return dl, 0.20, (self.current_time, self.current_time + timedelta(hours=24))

        # "soon" / "sometime next week"
        if 'soon' in text or 'shortly' in text:
            dl = self.current_time + timedelta(days=3)
            return dl, 0.15, (self.current_time + timedelta(hours=1),
                              self.current_time + timedelta(days=7))

        return None, 0.0, None

    def _estimate_duration(self, text: str) -> Tuple[float, float]:
        """Estimate task duration with uncertainty (mean, std_dev) in hours."""
        for keyword, (est, std) in DURATION_KEYWORDS.items():
            if keyword in text:
                return est, std
        return 2.0, 1.5  # Default fallback

    def _identify_resources(self, text: str) -> List[ResourceType]:
        resources = []
        if any(w in text for w in ['meeting', 'call', 'sync', 'session', 'attend']):
            resources.append(ResourceType.TIME_BLOCK)
            resources.append(ResourceType.PHYSICAL_PRESENCE)
        if any(w in text for w in ['think', 'write', 'design', 'analyze', 'research']):
            resources.append(ResourceType.COGNITIVE_LOAD)
        if any(w in text for w in ['together', 'with you', 'jointly', 'collaborate']):
            resources.append(ResourceType.COLLABORATOR)
        if not resources:
            resources.append(ResourceType.TIME_BLOCK)
        return list(set(resources))

    def _generate_implicits(self, text: str,
                            parent_deadline: Optional[datetime]) -> List[ImplicitCommitmentSpec]:
        """
        Generate implicit commitment specs based on commitment type keywords.

        Critical: This is where most LLMs fail. Accepting "attend board meeting"
        should auto-generate travel buffer and slide prep as implicit commitments.
        """
        implicits = []
        for keyword, templates in IMPLICIT_TEMPLATES.items():
            if keyword in text:
                for tmpl in templates:
                    adjusted = ImplicitCommitmentSpec(
                        implicit_type=tmpl.implicit_type,
                        description=tmpl.description,
                        time_before_parent=tmpl.time_before_parent,
                        time_after_parent=tmpl.time_after_parent,
                        duration_hours=tmpl.duration_hours,
                        auto_accept=tmpl.auto_accept,
                    )
                    implicits.append(adjusted)
        return implicits

    def _check_ambiguity(self, text: str, deadline: Optional[datetime],
                         d_confidence: float, duration: float
                         ) -> Tuple[bool, Optional[str], List[str]]:
        """
        Identify ambiguities and generate targeted clarification questions.
        Handles edge case Class 11: Commitment Extraction Ambiguity.
        """
        ambiguity_flags = []
        clarification_qs = []

        if deadline is None:
            ambiguity_flags.append('deadline_unclear')
            clarification_qs.append("What specific deadline are you working to?")
        elif d_confidence < 0.3:
            ambiguity_flags.append('deadline_low_confidence')
            clarification_qs.append("Just to confirm — are you thinking [inferred deadline]?")

        for pattern in AMBIGUOUS_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                ambiguity_flags.append('timing_vague')
                clarification_qs.append("When you say 'soon', what timeframe works for you?")
                break

        vague_scope_patterns = ['something', 'stuff', 'things', 'whatever', 'anything']
        if any(p in text for p in vague_scope_patterns):
            ambiguity_flags.append('scope_unclear')
            clarification_qs.append("Can you be more specific about what you need?")

        is_ambiguous = len(ambiguity_flags) > 0
        reason = ', '.join(ambiguity_flags) if ambiguity_flags else None
        return is_ambiguous, reason, clarification_qs[:2]
