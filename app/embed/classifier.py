import re
from enum import Enum
from typing import Dict
from typing import List


class PostType(str, Enum):
    QUESTION = "question"
    ANSWER = "answer"
    DISCUSSION = "discussion"
    ANNOUNCEMENT = "announcement"


class ContentClassifier:
    """Classifies discourse posts as questions, answers, or other types"""

    # Question indicators
    QUESTION_PATTERNS = [
        # Direct question words
        r"\b(what|how|why|when|where|which|who)\b",
        # Question phrases
        r"\b(how to|how do|how can|what is|what are|how would)\b",
        # Specific question contexts
        r"\b(scores?|appear|dashboard|display|show)\b.*\?",
        # Help-seeking patterns
        r"\b(help|doubt|confused|clarify|explain)\b",
    ]

    # Answer indicators
    ANSWER_PATTERNS = [
        # Direct answer phrases
        r"\b(here\'s how|the answer is|you can|try this|this will)\b",
        r"\b(it appears|it shows|it displays|will appear|will show)\b",
        # Solution indicators
        r"\b(solution|fix|resolved|works|working)\b",
        # Explanation patterns
        r"\b(in the dashboard|on the screen|you will see)\b",
    ]

    # Authority response indicators (from faculty/TAs)
    AUTHORITY_ANSWER_PATTERNS = [
        r"\b(yes|no|correct|incorrect|exactly|that\'s right)\b",
        r"\b(the dashboard|system|grade|score|bonus)\b.*\b(shows?|displays?|appears?)\b",
    ]

    # Announcement patterns
    ANNOUNCEMENT_PATTERNS = [
        r"\b(announcement|notice|important|update|please note)\b",
        r"\b(deadline|extended|postponed|cancelled|rescheduled)\b",
        r"\b(assignment|exam|quiz|test)\b.*\b(due|submission|date)\b",
    ]

    def __init__(self, vip_weights: Dict[str, float]):
        self.vip_weights = vip_weights

    def classify_post_type(self, content: str, username: str) -> PostType:
        """Classify post as question, answer, announcement, or discussion"""
        content_lower = content.lower()

        # Check for announcements first (highest priority)
        announcement_score = sum(
            1
            for pattern in self.ANNOUNCEMENT_PATTERNS
            if re.search(pattern, content_lower, re.IGNORECASE)
        )

        # Check for questions
        question_score = sum(
            1
            for pattern in self.QUESTION_PATTERNS
            if re.search(pattern, content_lower, re.IGNORECASE)
        )

        # Check for answers
        answer_score = sum(
            1
            for pattern in self.ANSWER_PATTERNS
            if re.search(pattern, content_lower, re.IGNORECASE)
        )

        # Authority users get answer boost
        is_authority = (
            username in self.vip_weights and self.vip_weights[username] >= 2.0
        )
        if is_authority:
            authority_score = sum(
                1
                for pattern in self.AUTHORITY_ANSWER_PATTERNS
                if re.search(pattern, content_lower, re.IGNORECASE)
            )
            answer_score += authority_score * 2  # Double weight for authority answers

            # Authority users posting without clear indicators likely making announcements
            if announcement_score == 0 and answer_score == 0 and question_score == 0:
                announcement_score = 1

        # Classification logic with priority order
        if announcement_score > 0 and is_authority:
            return PostType.ANNOUNCEMENT
        if content.count("?") >= 1 and question_score > 0:
            return PostType.QUESTION
        if answer_score > question_score and answer_score > 0:
            return PostType.ANSWER
        if question_score > 0:
            return PostType.QUESTION
        return PostType.DISCUSSION

    def get_content_confidence(self, content: str, username: str) -> float:
        """Return confidence score (0-1) for the classification"""
        content_lower = content.lower()

        total_indicators = 0
        matched_indicators = 0

        # Count all possible indicators
        all_patterns = (
            self.QUESTION_PATTERNS + self.ANSWER_PATTERNS + self.ANNOUNCEMENT_PATTERNS
        )

        for pattern in all_patterns:
            total_indicators += 1
            if re.search(pattern, content_lower, re.IGNORECASE):
                matched_indicators += 1

        # Authority users get confidence boost
        is_authority = (
            username in self.vip_weights and self.vip_weights[username] >= 2.0
        )
        confidence = matched_indicators / max(total_indicators, 1)

        if is_authority:
            confidence = min(1.0, confidence * 1.2)  # 20% boost for authority

        # Question marks provide strong confidence for questions
        if "?" in content:
            confidence = min(1.0, confidence + 0.3)

        return confidence

    def extract_key_topics(self, content: str) -> List[str]:
        """Extract key academic topics from content"""
        content_lower = content.lower()

        # Academic topic patterns
        topic_patterns = {
            "assignment": r"\b(assignment|homework|task|project)\b",
            "exam": r"\b(exam|test|quiz|midterm|final)\b",
            "grade": r"\b(grade|score|marks|points|percentage)\b",
            "deadline": r"\b(deadline|due|submission|extension)\b",
            "dashboard": r"\b(dashboard|portal|system|interface)\b",
            "bonus": r"\b(bonus|extra\s+credit|additional\s+points)\b",
            "lecture": r"\b(lecture|class|session|meeting)\b",
            "material": r"\b(material|resource|reference|reading)\b",
        }

        detected_topics = []
        for topic, pattern in topic_patterns.items():
            if re.search(pattern, content_lower):
                detected_topics.append(topic)

        return detected_topics

    def is_urgent_content(self, content: str, username: str) -> bool:
        """Determine if content requires urgent attention"""
        content_lower = content.lower()

        urgent_indicators = [
            r"\b(urgent|emergency|asap|immediately)\b",
            r"\b(deadline.*today|due.*today|submit.*today)\b",
            r"\b(cancelled|postponed|rescheduled)\b",
            r"\b(important.*announcement|critical.*update)\b",
        ]

        is_authority = (
            username in self.vip_weights and self.vip_weights[username] >= 2.0
        )

        # Authority users posting urgent content gets priority
        has_urgent_indicator = any(
            re.search(pattern, content_lower, re.IGNORECASE)
            for pattern in urgent_indicators
        )

        return has_urgent_indicator and is_authority
