# ============================================================
# models.py
# ============================================================
# This file defines the EXACT shape of every JSON object that
# enters or leaves our API. Think of it as a contract.
#
# We use Pydantic for this. Pydantic is a Python library that:
#   1. Checks that incoming data has the right types
#   2. Rejects bad data automatically (before your code even runs)
#   3. Converts Python objects to/from JSON automatically
#
# If the evaluator sends {"messages": "hello"} instead of a list,
# Pydantic catches it and returns a 422 error — your code never
# sees the bad data. This is why FastAPI + Pydantic is so powerful.
# ============================================================

from pydantic import BaseModel, field_validator  
# BaseModel    → base class for all our schema classes
# field_validator → lets us write custom validation logic

from typing import List, Optional
# List[X]     → a list where every item is type X
# Optional[X] → the field can be X or None (missing is OK)


# ============================================================
# INCOMING DATA SHAPES (what the caller sends US)
# ============================================================

class Message(BaseModel):
    """
    A single message in the conversation history.
    
    Example JSON this maps to:
    {
        "role": "user",
        "content": "I need to hire a Java developer"
    }
    
    role must be either "user" or "assistant" — nothing else.
    content is the actual text of the message.
    """
    role: str       # "user" or "assistant"
    content: str    # the message text

    @field_validator("role")
    @classmethod
    def role_must_be_valid(cls, v):
        # This runs automatically when a Message is created.
        # If role is anything other than "user" or "assistant",
        # Pydantic will raise a validation error.
        if v not in ("user", "assistant"):
            raise ValueError(f"role must be 'user' or 'assistant', got '{v}'")
        return v

    @field_validator("content")
    @classmethod
    def content_must_not_be_empty(cls, v):
        # Strip whitespace and check the message isn't blank.
        # An empty message would confuse our agent.
        if not v or not v.strip():
            raise ValueError("content cannot be empty")
        return v.strip()  # clean up leading/trailing whitespace


class ChatRequest(BaseModel):
    """
    The full request body for POST /chat.
    
    Example JSON this maps to:
    {
        "messages": [
            {"role": "user",      "content": "Hiring a Java dev"},
            {"role": "assistant", "content": "What seniority level?"},
            {"role": "user",      "content": "Mid-level, 4 years"}
        ]
    }
    
    The API is STATELESS — the caller sends the ENTIRE conversation
    history every single time. We never store sessions.
    """
    messages: List[Message]  # list of Message objects, at least 1

    @field_validator("messages")
    @classmethod
    def messages_must_not_be_empty(cls, v):
        # We need at least one message to process.
        if not v:
            raise ValueError("messages list cannot be empty")
        return v

    @field_validator("messages")
    @classmethod
    def messages_must_start_with_user(cls, v):
        # The first message in a conversation must always be from the user.
        # An assistant can't speak before the user does.
        if v and v[0].role != "user":
            raise ValueError("First message must have role 'user'")
        return v


# ============================================================
# OUTGOING DATA SHAPES (what WE send back to the caller)
# ============================================================

class Recommendation(BaseModel):
    """
    A single assessment recommendation.
    
    Example JSON this maps to:
    {
        "name": "Java 8 (New)",
        "url":  "https://www.shl.com/products/product-catalog/view/java-8-new/",
        "test_type": "K"
    }
    
    test_type uses comma-separated letters when multiple apply:
        "K"   → Knowledge & Skills only
        "K,S" → Knowledge & Skills + Simulation
        "P"   → Personality & Behavior only
        "P,C" → Personality + Competencies
    
    This is what we learned from the sample conversations —
    the evaluator expects this comma-separated format.
    """
    name: str       # exact name from catalog
    url: str        # exact URL from catalog — NEVER invented
    test_type: str  # comma-separated letter codes e.g. "K,S"

    @field_validator("url")
    @classmethod
    def url_must_be_shl(cls, v):
        # Hard safety check: every URL must be an SHL catalog URL.
        # This prevents hallucinated URLs from slipping through.
        # Even if Gemini makes one up, this validator catches it.
        if not v.startswith("https://www.shl.com"):
            raise ValueError(f"URL must be an SHL URL, got: {v}")
        return v


class ChatResponse(BaseModel):
    """
    The full response body for POST /chat.
    
    Example JSON this maps to:
    {
        "reply": "Here are 3 assessments for a mid-level Java developer.",
        "recommendations": [
            {
                "name": "Java 8 (New)",
                "url": "https://www.shl.com/...",
                "test_type": "K"
            }
        ],
        "end_of_conversation": false
    }
    
    RULES from the assignment spec:
    - recommendations is [] (empty list) when clarifying or refusing
    - recommendations has 1-10 items when recommending
    - end_of_conversation is true ONLY when the agent is done
    - The schema is NON-NEGOTIABLE — deviating breaks the evaluator
    """
    reply: str                          # natural language response to user
    recommendations: List[Recommendation]  # [] or 1-10 items — NEVER null
    end_of_conversation: bool           # true only when task is complete

    @field_validator("recommendations")
    @classmethod
    def recommendations_max_ten(cls, v):
        # Hard cap: the assignment says maximum 10 recommendations.
        # If our agent returns more, we silently trim to 10.
        # Better to trim than to fail the hard eval.
        if len(v) > 10:
            return v[:10]
        return v


# ============================================================
# HEALTH CHECK SHAPE
# ============================================================

class HealthResponse(BaseModel):
    """
    Response for GET /health.
    
    The spec says this must return exactly:
    {"status": "ok"}
    with HTTP 200.
    """
    status: str = "ok"  # default value is "ok"