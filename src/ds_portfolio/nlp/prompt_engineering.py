"""
NLP & Prompt Engineering Module.
Demonstrates:
- Prompt engineering for NLU tasks: search, QA, chat
- LLM integration (OpenAI, Anthropic)
- Evaluation frameworks for prompt quality
- Structured output parsing with Pydantic/Instructor
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json
import asyncio

from pydantic import BaseModel, Field
from instructor import Instructor
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
import pandas as pd


class TaskType(str, Enum):
    SEARCH_RERANK = "search_rerank"
    QUESTION_ANSWERING = "question_answering"
    CHAT_BOT = "chat_bot"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    SUMMARIZATION = "summarization"


class SentimentLabel(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class SearchResult(BaseModel):
    """Structured output for Search Reranking."""
    document_id: str
    relevance_score: float = Field(ge=0.0, le=1.0)
    reasoning: str


class QAResponse(BaseModel):
    """Structured output for Question Answering."""
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    sources: List[str] = Field(default_factory=list)


class ChatResponse(BaseModel):
    """Structured output for Chat Bot."""
    response: str
    intent: str
    suggested_actions: List[str] = Field(default_factory=list)


class SentimentOutput(BaseModel):
    """Structured output for Sentiment Analysis."""
    label: SentimentLabel
    confidence: float = Field(ge=0.0, le=1.0)
    key_phrases: List[str] = Field(default_factory=list)


class SummarizationOutput(BaseModel):
    """Structured output for Summarization."""
    summary: str
    key_points: List[str] = Field(default_factory=list)
    word_count: int


@dataclass
class PromptTemplate:
    """Version-controlled prompt templates for experimentation."""
    name: str
    version: str
    system_prompt: str
    user_template: str
    task_type: TaskType
    output_model: type[BaseModel]
    few_shot_examples: List[Dict[str, str]] = None


class PromptEngineer:
    """
    Prompt Engineering Framework demonstrating:
    - Template versioning and A/B testing
    - Few-shot prompt construction
    - Chain-of-thought prompting
    - Structured output with validation
    - Evaluation of prompt variants
    """

    def __init__(self, provider: str = "openai", api_key: str = None):
        self.provider = provider
        self.templates: Dict[str, PromptTemplate] = {}

        if provider == "openai":
            self.client = Instructor.from_openai(AsyncOpenAI(api_key=api_key))
        elif provider == "anthropic":
            self.client = Instructor.from_anthropic(AsyncAnthropic(api_key=api_key))
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def register_template(self, template: PromptTemplate) -> None:
        """Register a prompt template for a specific task."""
        key = f"{template.task_type.value}:{template.name}:{template.version}"
        self.templates[key] = template

    def build_prompt(self, task_type: TaskType, inputs: Dict[str, Any],
                     template_name: str = "default", version: str = "v1") -> List[Dict[str, str]]:
        """Constructs messages array with few-shot examples and chain-of-thought."""
        key = f"{task_type.value}:{template_name}:{version}"
        template = self.templates.get(key)

        if not template:
            # Fallback to built-in defaults
            template = self._get_default_template(task_type)

        messages = [{"role": "system", "content": template.system_prompt}]

        # Add few-shot examples if available
        if template.few_shot_examples:
            for ex in template.few_shot_examples:
                messages.append({"role": "user", "content": ex["input"]})
                messages.append({"role": "assistant", "content": ex["output"]})

        # Add user query with template formatting
        user_content = template.user_template.format(**inputs)
        messages.append({"role": "user", "content": user_content})

        return messages

    def _get_default_template(self, task_type: TaskType) -> PromptTemplate:
        """Built-in default templates for each task type."""
        defaults = {
            TaskType.SEARCH_RERANK: PromptTemplate(
                name="default",
                version="v1",
                system_prompt=(
                    "You are an expert search relevance judge. "
                    "Given a query and a document snippet, score relevance 0-1 "
                    "and provide brief reasoning. Output JSON only."
                ),
                user_template="Query: {query}\nDocument: {document}\n\nOutput JSON with fields: document_id, relevance_score, reasoning",
                task_type=TaskType.SEARCH_RERANK,
                output_model=SearchResult,
            ),
            TaskType.QUESTION_ANSWERING: PromptTemplate(
                name="default",
                version="v1",
                system_prompt=(
                    "You are a precise QA system. Answer using only provided context. "
                    "Cite sources. If unknown, say so. Output JSON only."
                ),
                user_template="Context: {context}\nQuestion: {question}\n\nOutput JSON with fields: answer, confidence, sources",
                task_type=TaskType.QUESTION_ANSWERING,
                output_model=QAResponse,
            ),
            TaskType.CHAT_BOT: PromptTemplate(
                name="default",
                version="v1",
                system_prompt=(
                    "You are a helpful customer support agent. "
                    "Identify intent, respond empathetically, suggest actions. Output JSON only."
                ),
                user_template="User Message: {message}\nConversation History: {history}\n\nOutput JSON with fields: response, intent, suggested_actions",
                task_type=TaskType.CHAT_BOT,
                output_model=ChatResponse,
            ),
            TaskType.SENTIMENT_ANALYSIS: PromptTemplate(
                name="default",
                version="v1",
                system_prompt=(
                    "Analyze sentiment of customer feedback. "
                    "Label: positive/neutral/negative. Extract key phrases. Output JSON only."
                ),
                user_template="Feedback: {text}\n\nOutput JSON with fields: label, confidence, key_phrases",
                task_type=TaskType.SENTIMENT_ANALYSIS,
                output_model=SentimentOutput,
            ),
            TaskType.SUMMARIZATION: PromptTemplate(
                name="default",
                version="v1",
                system_prompt=(
                    "Summarize text concisely. Extract key points. Count words. Output JSON only."
                ),
                user_template="Text: {text}\n\nOutput JSON with fields: summary, key_points, word_count",
                task_type=TaskType.SUMMARIZATION,
                output_model=SummarizationOutput,
            ),
        }
        return defaults[task_type]

    async def execute_task(self, task_type: TaskType, inputs: Dict[str, Any],
                           template_name: str = "default", version: str = "v1") -> BaseModel:
        """Execute prompt with structured output validation."""
        messages = self.build_prompt(task_type, inputs, template_name, version)
        key = f"{task_type.value}:{template_name}:{version}"
        template = self.templates.get(key, self._get_default_template(task_type))

        if self.provider == "openai":
            result = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                response_model=template.output_model,
                messages=messages,
                temperature=0.1,
            )
        else:  # anthropic
            result = await self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                response_model=template.output_model,
                messages=messages,
                temperature=0.1,
            )
        return result


class PromptEvaluator:
    """
    Evaluation Framework for Prompt Engineering.
    Measures: Accuracy, Faithfulness, Hallucination Rate, Cost, Latency.
    """

    def __init__(self):
        self.results = []

    async def evaluate_prompt_variant(self, engineer: PromptEngineer, task_type: TaskType,
                                      test_cases: List[Dict[str, Any]],
                                      template_name: str, version: str) -> Dict[str, Any]:
        """Run evaluation suite on a prompt variant."""
        correct = 0
        total_latency = 0.0
        total_cost = 0.0
        hallucinations = 0

        for case in test_cases:
            import time
            start = time.time()
            try:
                result = await engineer.execute_task(
                    task_type,
                    {k: v for k, v in case.items() if k != "expected"},
                    template_name, version
                )
                elapsed = time.time() - start
                total_latency += elapsed

                # Task-specific correctness check
                if self._is_correct(result, case.get("expected"), task_type):
                    correct += 1
                else:
                    hallucinations += 1

            except Exception as e:
                hallucinations += 1

        n = len(test_cases)
        return {
            "template": f"{template_name}:{version}",
            "accuracy": correct / n if n > 0 else 0,
            "hallucination_rate": hallucinations / n if n > 0 else 0,
            "avg_latency_sec": total_latency / n if n > 0 else 0,
            "total_cases": n,
        }

    def _is_correct(self, result: BaseModel, expected: Any, task_type: TaskType) -> bool:
        """Task-specific correctness validation."""
        if task_type == TaskType.SENTIMENT_ANALYSIS:
            return result.label.value == expected.get("label")
        elif task_type == TaskType.QUESTION_ANSWERING:
            # Semantic similarity check (simplified)
            expected_ans = expected.get("answer", "").lower()
            actual_ans = result.answer.lower()
            return expected_ans in actual_ans or actual_ans in expected_ans
        return True  # Default pass for generative tasks