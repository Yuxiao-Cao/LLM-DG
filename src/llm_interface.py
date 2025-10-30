"""
LLM API interface for interacting with various language models
"""

import requests
import json
from typing import Dict, Any, Optional, List
import time
from abc import ABC, abstractmethod
import os
from dotenv import load_dotenv


# Import OpenAI library for Doubao API compatibility
try:
    from openai import OpenAI, base_url

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI library not available. Install with: pip install openai")

load_dotenv()
# load_dotenv(dotenv_path="../.env")


class LLMInterface(ABC):
    """Abstract base class for LLM interfaces"""

    @abstractmethod
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate response from the language model"""
        pass


class DoubaoInterface(LLMInterface):
    """
    Interface for Doubao API (doubao-seed-1-6-thinking-250715 model)
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize Doubao interface

        Args:
            api_key: API key for authentication
            base_url: Base URL for the API endpoint
        """
        self.api_key = api_key or os.getenv("DOUBAO_API_KEY")
        self.base_url = base_url or os.getenv("DOUBAO_BASE_URL")

        if not self.api_key:
            raise ValueError("API key is required. Set ARK_API_KEY environment variable or pass api_key parameter")

        self.model_name = os.getenv("DOUBAO_MODEL_NAME")

        # Initialize OpenAI client with Doubao endpoint
        if OPENAI_AVAILABLE:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
        else:
            raise ImportError("OpenAI library is required for Doubao API. Install with: pip install openai")

    def generate_response(self, prompt: str, max_tokens: int = 10**5, temperature: float = 0.7, **kwargs) -> str:
        """
        Generate response from Doubao model

        Args:
            prompt: Input prompt for the model
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            **kwargs: Additional parameters

        Returns:
            Model response as string
        """
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert autonomous driving decision-making AI with deep knowledge of game theory, traffic safety, and vehicle dynamics. Always provide detailed reasoning for your decisions."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]#,
                # max_tokens=max_tokens,
                # temperature=temperature
            )

            if completion.choices and len(completion.choices) > 0:
                return completion.choices[0].message.content
            else:
                raise ValueError("No choices returned from API")

        except Exception as e:
            raise RuntimeError(f"Doubao API request failed: {e}")

    def generate_batch_responses(self, prompts: List[str], **kwargs) -> List[str]:
        """
        Generate responses for multiple prompts

        Args:
            prompts: List of input prompts
            **kwargs: Additional parameters for generation

        Returns:
            List of model responses
        """
        responses = []
        for i, prompt in enumerate(prompts):
            try:
                response = self.generate_response(prompt, **kwargs)
                responses.append(response)

                # Add small delay between requests to avoid rate limiting
                if i < len(prompts) - 1:
                    time.sleep(0.1)

            except Exception as e:
                print(f"Error generating response for prompt {i}: {e}")
                responses.append(f"Error: {str(e)}")

        return responses


class OpenAIInterface(LLMInterface):
    """
    Interface for OpenAI-compatible APIs (can be used for testing)
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None,
                 model_name: Optional[str] = None):
        """
        Initialize OpenAI interface

        Args:
            api_key: API key for authentication
            base_url: Base URL for the API endpoint
            model_name: Name of the model to use
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.model_name = model_name or os.getenv("OPENAI_MODEL_NAME")

        if not self.api_key:
            raise ValueError("API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter")

        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })

    def generate_response(self, prompt: str, max_tokens: int = 10**5, temperature: float = 0.7, **kwargs) -> str:
        """
        Generate response from OpenAI-compatible model

        Args:
            prompt: Input prompt for the model
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            **kwargs: Additional parameters

        Returns:
            Model response as string
        """
        url = f"{self.base_url}/chat/completions"

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert autonomous driving decision-making AI with deep knowledge of game theory, traffic safety, and vehicle dynamics. Always provide detailed reasoning for your decisions."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature
        }

        try:
            response = self.session.post(url, json=payload, timeout=500)
            response.raise_for_status()

            result = response.json()

            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            else:
                raise ValueError(f"Unexpected response format: {result}")

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"API request failed: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error: {e}")


class DeepseekInterface(LLMInterface):
    """
    Interface for Deepseek API
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None,
                 model_name: Optional[str] = None):
        """
        Initialize Deepseek interface

        Args:
            api_key: API key for authentication
            base_url: Base URL for the API endpoint
            model_name: Name of the model to use
        """
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.base_url = base_url or os.getenv("DEEPSEEK_BASE_URL")
        self.model_name = model_name or os.getenv("DEEPSEEK_MODEL_NAME")

        if not self.api_key:
            raise ValueError("API key is required. Set DEEPSEEK_API_KEY environment variable or pass api_key parameter")

        # Initialize OpenAI client with Deepseek endpoint
        if OPENAI_AVAILABLE:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
        else:
            raise ImportError("OpenAI library is required for Deepseek API. Install with: pip install openai")

    def generate_response(self, prompt: str, max_tokens: int = 10**5, temperature: float = 0.7, **kwargs) -> str:
        """
        Generate response from Deepseek model

        Args:
            prompt: Input prompt for the model
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            **kwargs: Additional parameters

        Returns:
            Model response as string
        """
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert autonomous driving decision-making AI with deep knowledge of game theory, traffic safety, and vehicle dynamics. Always provide detailed reasoning for your decisions."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )

            if completion.choices and len(completion.choices) > 0:
                return completion.choices[0].message.content
            else:
                raise ValueError("No choices returned from API")

        except Exception as e:
            raise RuntimeError(f"Deepseek API request failed: {e}")


class QwenInterface(LLMInterface):
    """
    Interface for Qwen API (Alibaba Cloud)
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None,
                 model_name: Optional[str] = None):
        """
        Initialize Qwen interface

        Args:
            api_key: API key for authentication
            base_url: Base URL for the API endpoint
            model_name: Name of the model to use
        """
        self.api_key = api_key or os.getenv("QWEN_API_KEY")
        self.base_url = base_url or os.getenv("QWEN_BASE_URL")
        self.model_name = model_name or os.getenv("QWEN_MODEL_NAME")

        if not self.api_key:
            raise ValueError("API key is required. Set QWEN_API_KEY environment variable or pass api_key parameter")

        # Initialize OpenAI client with Qwen endpoint
        if OPENAI_AVAILABLE:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
        else:
            raise ImportError("OpenAI library is required for Qwen API. Install with: pip install openai")

    def generate_response(self, prompt: str, max_tokens: int = 10**5, temperature: float = 0.7, **kwargs) -> str:
        """
        Generate response from Qwen model

        Args:
            prompt: Input prompt for the model
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            **kwargs: Additional parameters

        Returns:
            Model response as string
        """
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert autonomous driving decision-making AI with deep knowledge of game theory, traffic safety, and vehicle dynamics. Always provide detailed reasoning for your decisions."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )

            if completion.choices and len(completion.choices) > 0:
                return completion.choices[0].message.content
            else:
                raise ValueError("No choices returned from API")

        except Exception as e:
            raise RuntimeError(f"Qwen API request failed: {e}")

class GeminiInterface(LLMInterface):
    """
    Initialize Gemini interface
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None,
                 model_name: Optional[str] = None):
        """
        Initialize GEMINI interface

        Args:
            api_key: API key for authentication
            base_url: Base URL for the API endpoint
            model_name: Name of the model to use
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.base_url = base_url or os.getenv("GEMINI_BASE_URL")
        self.model_name = model_name or os.getenv("GEMINI_MODEL_NAME")

        if not self.api_key:
            raise ValueError("API key is required. Set GEMINI_API_KEY environment variable or pass api_key parameter")

        # Initialize OpenAI client with GEMINI endpoint
        if OPENAI_AVAILABLE:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
        else:
            raise ImportError("OpenAI library is required for GEMINI API. Install with: pip install openai")

    def generate_response(self, prompt: str, max_tokens: int = 10**5, temperature: float = 0.7, **kwargs) -> str:
        """
        Generate response from GEMINI model

        Args:
            prompt: Input prompt for the model
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            **kwargs: Additional parameters

        Returns:
            Model response as string
        """
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert autonomous driving decision-making AI with deep knowledge of game theory, traffic safety, and vehicle dynamics. Always provide detailed reasoning for your decisions."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )

            if completion.choices and len(completion.choices) > 0:
                return completion.choices[0].message.content
            else:
                raise ValueError("No choices returned from API")

        except Exception as e:
            raise RuntimeError(f"GEMINI API request failed: {e}")


class ClaudeInterface(LLMInterface):
    """
    Initialize Claude interface
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None,
                 model_name: Optional[str] = None):
        """
        Initialize Claude interface

        Args:
            api_key: API key for authentication
            base_url: Base URL for the API endpoint
            model_name: Name of the model to use
        """
        self.api_key = api_key or os.getenv("CLAUDE_API_KEY")
        self.base_url = base_url or os.getenv("CLAUDE_BASE_URL")
        self.model_name = model_name or os.getenv("CLAUDE_MODEL_NAME")

        if not self.api_key:
            raise ValueError("API key is required. Set CLAUDE_API_KEY environment variable or pass api_key parameter")

        # Initialize OpenAI client with CLAUDE endpoint
        if OPENAI_AVAILABLE:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
        else:
            raise ImportError("OpenAI library is required for CLAUDE API. Install with: pip install openai")

    def generate_response(self, prompt: str, max_tokens: int = 10**5, temperature: float = 0.7, **kwargs) -> str:
        """
        Generate response from CLAUDE model

        Args:
            prompt: Input prompt for the model
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            **kwargs: Additional parameters

        Returns:
            Model response as string
        """
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert autonomous driving decision-making AI with deep knowledge of game theory, traffic safety, and vehicle dynamics. Always provide detailed reasoning for your decisions."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )

            if completion.choices and len(completion.choices) > 0:
                return completion.choices[0].message.content
            else:
                raise ValueError("No choices returned from API")

        except Exception as e:
            raise RuntimeError(f"CLAUDE API request failed: {e}")



def create_llm_interface(model_type: str, **kwargs) -> LLMInterface:
    """
    Factory function to create LLM interface

    Args:
        model_type: Type of model interface ("doubao", "openai", "deepseek", "qwen", "gemini", "claude", "mock")
        **kwargs: Additional parameters for interface initialization

    Returns:
        LLMInterface instance
    """
    model_type = model_type.lower()

    if model_type == "doubao":
        return DoubaoInterface(**kwargs)
    elif model_type == "openai":
        return OpenAIInterface(**kwargs)
    elif model_type == "deepseek":
        return DeepseekInterface(**kwargs)
    elif model_type == "qwen":
        return QwenInterface(**kwargs)
    elif model_type == "gemini":
        return GeminiInterface(**kwargs)
    elif model_type == "claude":
        return ClaudeInterface(**kwargs)
    else:
        supported_models = ["doubao", "openai", "deepseek", "qwen", "gemini", "claude"]
        raise ValueError(f"Unsupported model type: {model_type}. Supported types: {', '.join(supported_models)}")