"""Factory for creating specific LLM instances using AWS Bedrock"""

from enum import Enum

import boto3
from langchain_aws import ChatBedrock
from langchain.chat_models.base import BaseChatModel

from realest_tate_agent.config import AWS_REGION


class LlmTier(Enum):
    """Available LLM based on their cost/intelligence."""

    SMALL = "us.anthropic.claude-3-5-haiku-20241022-v1:0"
    STANDARD = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
    LARGE = "us.anthropic.claude-sonnet-4-20250514-v1:0"


class LlmFactory:
    """Factory class to create LLM instances using AWS Bedrock."""

    def __init__(self):
        """
        Initialize the LLM factory with AWS Bedrock configuration

        Args:
            region_name: AWS region where Bedrock is available
        """
        self.bedrock_runtime = boto3.client(
            service_name="bedrock-runtime", region_name=AWS_REGION
        )

    def instantiate_llm(self, tier: LlmTier, temperature: float = 0.0) -> BaseChatModel:
        """Instantiate an LLM instance based on the specified tier

        Args:
            tier: The LLM tier (SMALL, STANDARD, LARGE)
            temperature: Model temperature (0.0 = deterministic, 1.0 = creative)

        Returns:
            ChatBedrock instance configured for the specified tier
        """

        return ChatBedrock(
            model_id=tier.value,
            client=self.bedrock_runtime,
            model_kwargs={
                "temperature": temperature,
                # "max_tokens": 4000,
            },
        )
