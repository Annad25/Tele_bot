from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from services.llm_gateway import LLMGateway


class LLMGatewayTests(unittest.IsolatedAsyncioTestCase):
    @patch("services.llm_gateway.AsyncOpenAI")
    async def test_generate_uses_prior_user_queries_only_for_continuity(
        self, mock_openai
    ) -> None:
        create = AsyncMock(
            return_value=SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))]
            )
        )
        mock_openai.return_value = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )

        gateway = LLMGateway(api_key="test-key")
        result = await gateway.generate(
            query="What about the equipment policy?",
            context="[company_remote_policy.md]\nThe company provides a laptop.",
            history=[
                {
                    "query": "What are the core hours?",
                    "response": "Core hours are 10 AM to 4 PM.",
                }
            ],
        )

        self.assertEqual(result, "answer")

        messages = create.await_args.kwargs["messages"]
        self.assertEqual(len(messages), 2)
        self.assertEqual(create.await_args.kwargs["max_completion_tokens"], 1024)
        self.assertIn("What are the core hours?", messages[1]["content"])
        self.assertNotIn("Core hours are 10 AM to 4 PM.", messages[1]["content"])
