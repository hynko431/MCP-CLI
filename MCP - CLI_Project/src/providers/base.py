"""
Base LLM Provider abstraction layer

This module defines the abstract interface for all LLM providers,
enabling pluggable implementations for Ollama, OpenRouter, Grok, and others.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any, AsyncIterator
import asyncio
from datetime import datetime


class MessageRole(str, Enum):
    """Message role in chat context"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class ChatMessage:
    """Single message in a chat conversation"""
    role: MessageRole
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role.value,
            "content": self.content,
            **self.metadata
        }


@dataclass
class ChatResponse:
    """Response from LLM provider"""
    content: str
    model: str
    provider: str
    finish_reason: str  # "stop", "length", "error"
    usage: Dict[str, int] = field(default_factory=dict)  # token counts
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ProviderConfig:
    """Configuration for LLM provider"""
    provider_type: str  # "ollama", "openrouter", "grok", etc.
    api_key: Optional[str] = None
    api_endpoint: Optional[str] = None
    model: Optional[str] = None
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderInfo:
    """Provider metadata and capabilities"""
    name: str
    description: str
    supported_models: List[str]
    cost_per_1k_tokens: Dict[str, float]  # {"input": 0.001, "output": 0.002}
    capabilities: List[str]  # ["chat", "stream", "vision", "tools"]
    health_status: str  # "healthy", "degraded", "unhealthy"
    last_check: Optional[datetime] = None
    supported_roles: List[str] = field(default_factory=lambda: ["user", "assistant", "system"])


class LLMProvider(ABC):
    """
    Abstract base class for all LLM providers
    
    Defines the interface that all provider implementations must follow.
    This enables pluggable provider support while maintaining a consistent API.
    """

    def __init__(self, config: ProviderConfig):
        """
        Initialize provider with configuration
        
        Args:
            config: ProviderConfig with provider-specific settings
        """
        self.config = config
        self._health_status = "unknown"
        self._last_health_check = None
        self._models_cache: Optional[List[str]] = None
        self._cache_ttl = 300  # 5 minutes

    @abstractmethod
    async def chat(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> ChatResponse:
        """
        Send chat message(s) to LLM and get response
        
        Args:
            messages: List of ChatMessage objects
            model: Model name (uses config.model if not specified)
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens in response
            stop_sequences: Stop generation on these strings
            **kwargs: Provider-specific parameters
            
        Returns:
            ChatResponse with LLM output
            
        Raises:
            ValueError: If configuration or messages are invalid
            TimeoutError: If request exceeds timeout
            Exception: Provider-specific errors
        """
        pass

    @abstractmethod
    async def stream_chat(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Stream chat response token-by-token
        
        Args:
            messages: List of ChatMessage objects
            model: Model name (uses config.model if not specified)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            stop_sequences: Stop generation on these strings
            **kwargs: Provider-specific parameters
            
        Yields:
            Response tokens as they're generated
            
        Raises:
            ValueError: If configuration or messages are invalid
            TimeoutError: If request exceeds timeout
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if provider is healthy and accessible
        
        Returns:
            True if provider is operational, False otherwise
        """
        pass

    @abstractmethod
    async def get_models(self, force_refresh: bool = False) -> List[str]:
        """
        List available models from provider
        
        Args:
            force_refresh: Skip cache and fetch fresh list
            
        Returns:
            List of available model identifiers
        """
        pass

    @abstractmethod
    async def get_provider_info(self) -> ProviderInfo:
        """
        Get provider metadata and capabilities
        
        Returns:
            ProviderInfo with provider details
        """
        pass

    async def validate_config(self) -> bool:
        """
        Validate provider configuration
        
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        if not self.config.provider_type:
            raise ValueError("provider_type is required")
        if not self.config.api_endpoint and self.config.provider_type != "ollama":
            raise ValueError(f"{self.config.provider_type} requires api_endpoint")
        return True

    def get_health_status(self) -> Dict[str, Any]:
        """
        Get current health status
        
        Returns:
            Dict with status, last check time, and other health metrics
        """
        return {
            "provider": self.config.provider_type,
            "status": self._health_status,
            "last_check": self._last_health_check.isoformat() if self._last_health_check else None,
            "is_healthy": self._health_status == "healthy"
        }

    async def close(self) -> None:
        """
        Cleanup and close connections (optional override)
        """
        pass

    async def __aenter__(self):
        """Context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        await self.close()


class SyncLLMProvider(LLMProvider):
    """
    Wrapper for synchronous LLM providers
    
    Adapts sync implementations to async interface using thread pool
    """

    async def chat(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> ChatResponse:
        """Async wrapper for sync chat method"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._chat_sync,
            messages,
            model,
            temperature,
            max_tokens,
            stop_sequences,
            kwargs
        )

    async def stream_chat(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """Async wrapper for sync stream method"""
        loop = asyncio.get_event_loop()
        generator = await loop.run_in_executor(
            None,
            self._stream_chat_sync,
            messages,
            model,
            temperature,
            max_tokens,
            stop_sequences,
            kwargs
        )
        for chunk in generator:
            yield chunk

    async def health_check(self) -> bool:
        """Async wrapper for sync health check"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._health_check_sync)

    async def get_models(self, force_refresh: bool = False) -> List[str]:
        """Async wrapper for sync model listing"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_models_sync, force_refresh)

    # Sync methods to be implemented by subclasses
    def _chat_sync(self, messages, model, temperature, max_tokens, stop_sequences, kwargs) -> ChatResponse:
        raise NotImplementedError

    def _stream_chat_sync(self, messages, model, temperature, max_tokens, stop_sequences, kwargs):
        raise NotImplementedError

    def _health_check_sync(self) -> bool:
        raise NotImplementedError

    def _get_models_sync(self, force_refresh: bool) -> List[str]:
        raise NotImplementedError
