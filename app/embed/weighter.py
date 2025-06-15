import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict
from typing import List
from typing import Optional


class AcademicRole(str, Enum):
    FACULTY = "faculty"
    INSTRUCTOR = "instructor"
    TA = "teaching_assistant"
    STUDENT = "student"


@dataclass
class WeightingComponents:
    """Container for all weighting components"""

    role_keywords: List[str]
    type_keywords: List[str]
    context_keywords: List[str]
    bridge_text: Optional[str]
    authority_prefix: str
    should_apply: bool  # Whether weighting should be applied


class RoleKeywordWeighter:
    """Handles role-based keyword weighting for academic discourse"""

    # Role-based keyword mappings
    ROLE_KEYWORDS = {
        AcademicRole.FACULTY: {
            "primary": ["[FACULTY]", "[COURSE-DESIGNER]", "[PROFESSOR]"],
            "authority": ["[OFFICIAL]", "[POLICY]", "[CURRICULUM]"],
            "weight": 3.0,
        },
        AcademicRole.INSTRUCTOR: {
            "primary": ["[INSTRUCTOR]", "[TEACHER]", "[EXPERT]"],
            "authority": ["[OFFICIAL]", "[LECTURE]", "[GRADING]"],
            "weight": 2.5,
        },
        AcademicRole.TA: {
            "primary": ["[TA]", "[ASSISTANT]", "[HELPER]"],
            "authority": ["[SUPPORT]", "[TUTORIAL]", "[CLARIFICATION]"],
            "weight": 2.0,
        },
        AcademicRole.STUDENT: {
            "primary": ["[STUDENT]", "[LEARNER]"],
            "authority": [],
            "weight": 1.0,
        },
    }

    # Content-context keyword patterns
    CONTENT_PATTERNS = {
        "assignment": {
            "triggers": ["assignment", "homework", "task", "submit", "due"],
            "keywords": ["[ASSIGNMENT]", "[TASK]", "[HOMEWORK]"],
        },
        "solution": {
            "triggers": [
                "solution",
                "answer",
                "solved",
                "fix",
                "here's how",
                "try this",
            ],
            "keywords": ["[SOLUTION]", "[ANSWER]", "[SOLVED]"],
        },
        "announcement": {
            "triggers": [
                "announcement",
                "notice",
                "update",
                "important",
                "please note",
            ],
            "keywords": ["[ANNOUNCEMENT]", "[NOTICE]", "[IMPORTANT]"],
        },
        "deadline": {
            "triggers": ["deadline", "due date", "extended", "postponed", "urgent"],
            "keywords": ["[DEADLINE]", "[DUE]", "[URGENT]"],
        },
        "clarification": {
            "triggers": [
                "doubt",
                "confusion",
                "clarify",
                "explain",
                "help",
                "understand",
            ],
            "keywords": ["[CLARIFICATION]", "[HELP]", "[EXPLANATION]"],
        },
        "resource": {
            "triggers": ["material", "resource", "reference", "link", "reading"],
            "keywords": ["[RESOURCE]", "[MATERIAL]", "[REFERENCE]"],
        },
        "evaluation": {
            "triggers": ["grade", "marks", "evaluation", "feedback", "score"],
            "keywords": ["[GRADE]", "[EVALUATION]", "[FEEDBACK]"],
        },
        "discussion": {
            "triggers": ["discuss", "opinion", "thoughts", "what do you think"],
            "keywords": ["[DISCUSSION]", "[FORUM]", "[DEBATE]"],
        },
    }

    # Post-type specific keyword mappings
    POST_TYPE_KEYWORDS = {
        "question": ["[QUESTION]", "[ASKING]", "[HELP-NEEDED]"],
        "answer": {
            AcademicRole.FACULTY: [
                "[OFFICIAL-ANSWER]",
                "[AUTHORITATIVE]",
                "[RESPONSE]",
            ],
            AcademicRole.INSTRUCTOR: [
                "[EXPERT-ANSWER]",
                "[AUTHORITATIVE]",
                "[RESPONSE]",
            ],
            AcademicRole.TA: ["[TA-ANSWER]", "[HELPFUL]", "[RESPONSE]"],
            AcademicRole.STUDENT: ["[STUDENT-ANSWER]", "[RESPONSE]"],
        },
        "announcement": ["[ANNOUNCEMENT]", "[OFFICIAL]", "[NOTICE]"],
        "discussion": ["[DISCUSSION]", "[TOPIC]", "[FORUM]"],
    }

    # QA matching bridge patterns
    QA_BRIDGE_PATTERNS = {
        # Dashboard/scoring related
        r"\b(appears?|shows?|displays?)\b": "question about how it appears shows displays",
        r"\b(dashboard|screen|interface)\b": "question about dashboard screen interface",
        r"\b(score|grade|marks?)\b": "question about score grade marks",
        r"\b(bonus)\b": "question about bonus",
        # System behavior
        r"\b(will see|you see|can see)\b": "question about what you see",
        r"\b(works?|working)\b": "question about how it works",
        r"\b(happens?|occurred?)\b": "question about what happens",
    }

    def __init__(
        self, vip_weights: Dict[str, float], user_role_mapping: Dict[str, AcademicRole]
    ):
        self.vip_weights = vip_weights
        self.user_role_mapping = user_role_mapping

    def get_user_role(self, username: str) -> AcademicRole:
        """Get academic role for username"""
        return self.user_role_mapping.get(username, AcademicRole.STUDENT)

    def detect_content_context(self, content: str) -> List[str]:
        """Detect content patterns and return relevant keywords"""
        content_lower = content.lower()
        detected_keywords = []

        for _pattern_name, pattern_data in self.CONTENT_PATTERNS.items():
            # Check if any trigger words are present
            if any(trigger in content_lower for trigger in pattern_data["triggers"]):
                # Add primary keyword for this pattern
                detected_keywords.extend(
                    pattern_data["keywords"][:1]
                )  # Take first keyword

        return detected_keywords[:3]  # Limit to 3 context keywords

    def build_role_prefix(self, role: AcademicRole, weight: float) -> List[str]:
        """Build role-based prefix keywords"""
        role_data = self.ROLE_KEYWORDS[role]
        prefix_keywords = []

        if weight > 1.0:
            # Primary role keywords
            repetitions = min(int(weight), 3)  # Cap at 3 repetitions
            primary_keywords = role_data["primary"][:2]  # Take top 2

            for keyword in primary_keywords:
                prefix_keywords.extend([keyword] * repetitions)

            # Add authority keywords for senior roles
            if (
                role in [AcademicRole.FACULTY, AcademicRole.INSTRUCTOR]
                and role_data["authority"]
            ):
                prefix_keywords.extend(
                    role_data["authority"][:1]
                )  # Add 1 authority keyword

        return prefix_keywords

    def get_post_type_keywords(self, post_type: str, role: AcademicRole) -> List[str]:
        """Get keywords specific to post type and user role"""
        post_keywords = self.POST_TYPE_KEYWORDS.get(post_type, [])

        if isinstance(post_keywords, dict):
            # Role-specific keywords (e.g., for answers)
            return post_keywords.get(role, post_keywords.get(AcademicRole.STUDENT, []))
        # Generic keywords for post type
        return post_keywords

    def get_qa_bridge_text(self, content: str) -> Optional[str]:
        """Get QA bridge text for better question-answer matching"""
        content_lower = content.lower()
        bridge_terms = []

        for pattern, bridge in self.QA_BRIDGE_PATTERNS.items():
            if re.search(pattern, content_lower):
                bridge_terms.append(bridge)

        if bridge_terms:
            # Return bridge terms as text
            return " ".join(bridge_terms[:3])  # Limit to 3 bridges

        return None

    def get_authority_prefix(self, username: str, role: AcademicRole) -> str:
        """Get authority prefix for the user"""
        weight = self.vip_weights.get(username, 1.0)

        if weight <= 1.0:
            return ""

        authority_level = "highly trusted" if weight >= 2.5 else "trusted"
        role_name = role.value.replace("_", " ")

        return f"[Expert content from {authority_level} {role_name} {username}]"

    def get_weighting_components(
        self, username: str, content: str, post_type: str = None
    ) -> WeightingComponents:
        """Get all weighting components without modifying content"""
        # Get user role and weight
        role = self.get_user_role(username)
        weight = self.vip_weights.get(username, 1.0)

        # Check if weighting should be applied
        should_apply = weight > 1.0

        if not should_apply:
            return WeightingComponents(
                role_keywords=[],
                type_keywords=[],
                context_keywords=[],
                bridge_text=None,
                authority_prefix="",
                should_apply=False,
            )

        # 1. Role-based keywords
        role_keywords = self.build_role_prefix(role, weight)

        # 2. Post-type specific keywords
        type_keywords = []
        if post_type:
            type_keywords = self.get_post_type_keywords(post_type, role)

        # 3. Content-context keywords
        context_keywords = self.detect_content_context(content)

        # 4. QA bridge text (only for answers)
        bridge_text = None
        if post_type == "answer":
            bridge_text = self.get_qa_bridge_text(content)

        # 5. Authority prefix
        authority_prefix = self.get_authority_prefix(username, role)

        return WeightingComponents(
            role_keywords=role_keywords,
            type_keywords=type_keywords,
            context_keywords=context_keywords,
            bridge_text=bridge_text,
            authority_prefix=authority_prefix,
            should_apply=True,
        )

    def get_search_boost_multiplier(self, username: str) -> float:
        """Get multiplicative boost for search ranking"""
        role = self.get_user_role(username)
        base_weight = self.vip_weights.get(username, 1.0)

        # Role-based search priority multipliers
        role_multipliers = {
            AcademicRole.FACULTY: 3.0,  # Highest priority
            AcademicRole.INSTRUCTOR: 2.5,  # High priority
            AcademicRole.TA: 2.0,  # Medium priority
            AcademicRole.STUDENT: 1.0,  # Standard priority
        }

        return base_weight * role_multipliers[role]

    def get_role_metadata(self, username: str) -> Dict[str, any]:
        """Get complete role-based metadata for a user"""
        role = self.get_user_role(username)
        weight = self.vip_weights.get(username, 1.0)

        return {
            "username": username,
            "role": role.value,
            "authority_weight": weight,
            "search_boost": self.get_search_boost_multiplier(username),
            "is_authority": weight >= 2.0,
            "role_level": self.ROLE_KEYWORDS[role]["weight"],
        }


# Utility function to apply weighting components to content
def apply_weighting_to_content(content: str, components: WeightingComponents) -> str:
    """Apply weighting components to content - use this in your processing loop"""
    if not components.should_apply:
        return content

    # Collect all keywords
    all_keywords = []
    all_keywords.extend(components.role_keywords)
    all_keywords.extend(components.type_keywords)
    all_keywords.extend(components.context_keywords)

    # Limit total keywords to prevent token explosion
    final_keywords = all_keywords[:5]  # Max 5 keywords total

    # Build final weighted content
    parts = []

    if components.authority_prefix:
        parts.append(components.authority_prefix)

    if final_keywords:
        parts.append(" ".join(final_keywords))

    parts.append(content)

    if components.bridge_text:
        parts.append(f"[CONTEXTUAL-MATCH: {components.bridge_text}]")

    return " ".join(parts)
