"""
BrowseComp Environment - Web search reasoning benchmark

A single-turn evaluation environment with 1,266 encrypted questions requiring
web search. Agents must research questions, then submit answers with explanation
and confidence. Answers are graded by an LLM judge (gpt-5-mini).
"""

import pandas as pd
import openai
from pydantic import BaseModel, Field
from typing import Dict, List
from tavily import AsyncTavilyClient

from openreward.environments import Environment, JSONObject, Server, TextBlock, ToolOutput, tool

from decrypt import decrypt_task
from constants import BROWSECOMP_CSV


# Grader prompt template for LLM-based answer evaluation
GRADER_PROMPT_TEMPLATE = """You are evaluating whether an agent's answer to a research question is correct.

Question: {question}

Correct Answer: {correct_answer}

Agent's Response:
- Explanation: {explanation}
- Exact Answer: {exact_answer}
- Confidence: {confidence}

Task: Determine if the agent's "Exact Answer" is semantically equivalent to the correct answer.

Consider:
1. Does the exact answer capture the key factual content?
2. Are minor formatting/phrasing differences acceptable? (e.g., "Paris" vs "Paris, France")
3. Is the answer factually accurate according to the correct answer provided?
4. For numerical answers, allow small rounding differences

Provide a brief analysis (2-3 sentences), then conclude with either "CORRECT" or "INCORRECT" on a new line."""


# Pydantic schemas for type safety
class BrowseCompTaskSpec(BaseModel):
    """Task specification for BrowseComp environment"""
    id: str
    problem: str
    answer: str


class WebSearchInput(BaseModel):
    """Parameters for web_search tool"""
    query: str = Field(..., description="Search query")


class FetchUrlInput(BaseModel):
    """Parameters for fetch_url tool"""
    url: str = Field(..., description="URL to fetch")


class SubmitAnswerParams(BaseModel):
    """Parameters for submit_answer tool"""
    explanation: str = Field(
        ...,
        description="Your detailed reasoning and sources (2-4 sentences)"
    )
    exact_answer: str = Field(
        ...,
        description="The precise answer to the question (concise)"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Your confidence level (0.0 to 1.0)"
    )


def load_browsecomp_data() -> Dict[str, List[Dict]]:
    """
    Load and decrypt BrowseComp CSV dataset.

    Returns:
        Dict with "test" split containing list of decrypted task dicts

    Raises:
        FileNotFoundError: If CSV file not found at expected path
        ValueError: If decryption fails
    """
    print(f"Loading BrowseComp data from: {BROWSECOMP_CSV}")

    if not BROWSECOMP_CSV.exists():
        raise FileNotFoundError(
            f"BrowseComp CSV not found at {BROWSECOMP_CSV}. "
            f"Please download from: "
            f"https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv"
        )

    df = pd.read_csv(BROWSECOMP_CSV)

    tasks = []
    for idx, row in df.iterrows():
        try:
            # Decrypt problem and answer using canary password
            problem, answer = decrypt_task(
                row['problem'],
                row['answer'],
                row['canary']
            )

            tasks.append({
                "id": f"browsecomp_{idx}",
                "problem": problem,
                "answer": answer,
            })
        except Exception as e:
            print(f"Warning: Failed to decrypt task {idx}: {e}")
            continue

    print(f"Successfully loaded and decrypted {len(tasks)} tasks")
    return {"test": tasks}


# Load dataset once at module level (AIME pattern)
ALL_DATA = load_browsecomp_data()


class BrowseComp(Environment):
    """
    BrowseComp environment: encrypted web research questions with LLM grading.

    Agent workflow:
    1. Receives a research question requiring web search
    2. Uses web_search tool (provided by client) to research
    3. Submits answer with explanation, exact_answer, and confidence
    4. Answer is graded by gpt-5-mini comparing to correct answer
    5. Receives reward (1.0 correct, 0.0 incorrect) and feedback
    """

    def __init__(self, task_spec: JSONObject, secrets: dict[str, str] = {}) -> None:
        """
        Initialize BrowseComp environment instance.

        Args:
            task_spec: Task specification with id, problem, answer
            secrets: Must contain "openai_api_key" for grading and "tavily_api_key" for search

        Raises:
            ValueError: If required API keys missing or task_spec invalid
        """
        super().__init__(task_spec)
        self.config = BrowseCompTaskSpec.model_validate(task_spec)

        # Require OpenAI API key for grader - fail fast if missing
        openai_api_key = secrets.get("openai_api_key")
        if not openai_api_key:
            raise ValueError(
                "openai_api_key required in secrets parameter for LLM grading. "
                "Pass secrets={'openai_api_key': 'sk-...', 'tavily_api_key': 'tvly-...'} when creating session."
            )

        # Require Tavily API key for web search - fail fast if missing
        tavily_api_key = secrets.get("tavily_api_key")
        if not tavily_api_key:
            print(
                "Note: tavily_api_key required in secrets parameter for web search. "
                "Pass secrets={'openai_api_key': 'sk-...', 'tavily_api_key': 'tvly-...'} when creating session."
            )

        self.openai_client = openai.AsyncClient(api_key=openai_api_key)
        self.tavily_client = AsyncTavilyClient(api_key=tavily_api_key)

    @classmethod
    def list_splits(cls) -> list[str]:
        """Return available data splits"""
        return ["test"]

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        """
        List all tasks for a given split.

        Args:
            split: Data split name (only "test" available)

        Returns:
            List of task specifications (problem + answer)

        Raises:
            ValueError: If split is unknown
        """
        if split != "test":
            raise ValueError(f"Unknown split: {split}. Available splits: test")

        # Return all task fields including answer (needed for grading)
        return [
            {
                "id": task["id"],
                "problem": task["problem"],
                "answer": task["answer"],
            }
            for task in ALL_DATA["test"]
        ]

    def get_prompt(self) -> list[TextBlock]:
        """
        Generate prompt for the agent.

        Returns:
            List containing single TextBlock with question and instructions
        """
        prompt_text = f"""Research Question: {self.config.problem}

Your task is to research this question using web search and provide a comprehensive answer.

Available Tools:
1. web_search(query: str) - Search the web for information
   - Returns search results with titles, URLs, and snippets
   - You can search multiple times to gather information from different angles
2. fetch_url(url: str) - Fetch full content from a specific URL
   - Use this after web_search to read complete pages
   - Helpful for extracting detailed information
3. submit_answer(...) - Submit your final answer (ends the episode)

Instructions:
1. Use web_search to find relevant information
   - Consider searching for key entities, dates, or concepts mentioned in the question
   - Questions often require multi-hop reasoning across multiple searches
2. Use fetch_url to get complete content from promising URLs
3. When ready, use submit_answer with:
   - explanation: Your detailed reasoning and sources cited (2-4 sentences)
   - exact_answer: The precise, concise answer to the question
   - confidence: Your confidence level (0.0 to 1.0)

Important: Questions in this benchmark are deliberately challenging and often require multi-hop reasoning. Take your time to research thoroughly before submitting."""

        return [TextBlock(type="text", text=prompt_text)]

    @tool
    async def web_search(self, params: WebSearchInput) -> ToolOutput:
        """
        Search the web using Tavily. Returns search results with titles, URLs, and snippets.
        Use fetch_url tool to get full content from specific URLs if needed.
        """
        try:
            # Use Tavily search API
            response = await self.tavily_client.search(
                query=params.query,
                search_depth="basic",
                max_results=5
            )

            # Format results
            results = response.get("results", [])
            if not results:
                return ToolOutput(
                    blocks=[TextBlock(type="text", text="No search results found.")],
                    metadata={"query": params.query, "results": []},
                    reward=0.0,
                    finished=False
                )

            # Build display text
            display_parts = [f"Search results for: {params.query}\n"]
            for i, result in enumerate(results, 1):
                title = result.get("title", "No title")
                url = result.get("url", "")
                snippet = result.get("content", "")
                display_parts.append(f"{i}. {title}\n   URL: {url}\n   {snippet}\n")

            display_text = "\n".join(display_parts)

            return ToolOutput(
                blocks=[TextBlock(type="text", text=display_text)],
                metadata={
                    "query": params.query,
                    "results": results,
                    "count": len(results)
                },
                reward=0.0,
                finished=False
            )
        except Exception as e:
            return ToolOutput(
                blocks=[TextBlock(type="text", text=f"Web search failed: {str(e)}")],
                metadata={"query": params.query, "error": str(e)},
                reward=0.0,
                finished=False
            )

    @tool
    async def fetch_url(self, params: FetchUrlInput) -> ToolOutput:
        """
        Fetch and return the full text content from a specific URL using Tavily's extract method.
        Use this after web_search to get complete information from a page.
        """
        try:
            # Use Tavily's extract method
            response = await self.tavily_client.extract(urls=[params.url])

            # Get the extracted content
            results = response.get("results", [])
            if not results:
                return ToolOutput(
                    blocks=[TextBlock(type="text", text=f"No content extracted from {params.url}")],
                    metadata={"url": params.url, "results": []},
                    reward=0.0,
                    finished=False
                )

            # Get the first result (we only passed one URL)
            result = results[0]
            raw_content = result.get("raw_content", "")

            # Truncate if too long
            max_length = 8000
            if len(raw_content) > max_length:
                raw_content = raw_content[:max_length] + "...\n[Content truncated]"

            return ToolOutput(
                blocks=[TextBlock(type="text", text=f"Content from {params.url}:\n\n{raw_content}")],
                metadata={
                    "url": params.url,
                    "length": len(raw_content)
                },
                reward=0.0,
                finished=False
            )
        except Exception as e:
            return ToolOutput(
                blocks=[TextBlock(type="text", text=f"Failed to fetch URL: {str(e)}")],
                metadata={"url": params.url, "error": str(e)},
                reward=0.0,
                finished=False
            )

    async def _grade_answer(
        self,
        explanation: str,
        exact_answer: str,
        confidence: float
    ) -> Dict:
        """
        Use LLM grader to evaluate answer correctness.

        Args:
            explanation: Agent's reasoning
            exact_answer: Agent's submitted answer
            confidence: Agent's confidence score

        Returns:
            Dict with keys: is_correct, grading_response, confidence

        Note: Uses gpt-5-mini
        """
        grader_prompt = GRADER_PROMPT_TEMPLATE.format(
            question=self.config.problem,
            correct_answer=self.config.answer,
            explanation=explanation,
            exact_answer=exact_answer,
            confidence=confidence
        )

        # Use gpt-5-mini as recommended for graders (cost-effective, reliable)
        response = await self.openai_client.chat.completions.create(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": grader_prompt}],
        )

        grading_text = response.choices[0].message.content or ""

        # Parse verdict (case-insensitive, must have CORRECT without INCORRECT)
        upper_text = grading_text.upper()
        is_correct = "CORRECT" in upper_text and "INCORRECT" not in upper_text

        return {
            "is_correct": is_correct,
            "grading_response": grading_text,
            "confidence": confidence
        }

    @tool
    async def submit_answer(self, params: SubmitAnswerParams) -> ToolOutput:
        """
        Submit your final answer to the research question.

        This tool grades your answer using an LLM judge and returns a reward.
        The episode ends after calling this tool.

        Args:
            explanation: Your reasoning and sources (2-4 sentences)
            exact_answer: The precise answer to the question
            confidence: Your confidence level (0.0 to 1.0)

        Returns:
            ToolOutput with grading result, reward, and feedback
        """
        # Grade the answer using LLM judge
        grading_result = await self._grade_answer(
            params.explanation,
            params.exact_answer,
            params.confidence
        )

        reward = 1.0 if grading_result["is_correct"] else 0.0
        result_status = "✅ Correct" if grading_result["is_correct"] else "❌ Incorrect"

        # Format display output for the agent
        display_text = f"""{result_status}

Grading Analysis:
{grading_result['grading_response']}

Your Confidence: {params.confidence:.2f}
Reward: {reward:.1f}

Expected Answer: {self.config.answer}
Your Answer: {params.exact_answer}"""

        return ToolOutput(
            blocks=[TextBlock(type="text", text=display_text)],
            metadata={
                "task_id": self.config.id,
                "is_correct": grading_result["is_correct"],
                "grading_response": grading_result["grading_response"],
                "submitted_answer": params.exact_answer,
                "submitted_explanation": params.explanation,
                "confidence": params.confidence,
                "correct_answer": self.config.answer,  # For analysis
                "question": self.config.problem,
            },
            reward=reward,
            finished=True
        )


if __name__ == "__main__":
    # Start the environment server
    Server([BrowseComp]).run()
