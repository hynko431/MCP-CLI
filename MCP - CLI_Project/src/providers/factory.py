"""
Provider Factory

Factory pattern for creating LLM provider instances.
Enables pluggable provider creation and registration.
"""

from typing import Dict, Type, Optional
import logging

from .base import LLMProvider, ProviderConfig
from .ollama import OllamaProvider
from .openrouter import OpenRouterProvider
from .grok import GrokProvider

logger = logging.getLogger(__name__)


class ProviderFactory:
    """
    Factory for creating LLM provider instances
    
    Manages provider registration and instantiation.
    Supports built-in providers and custom provider registration.
    """

    # Registry of available providers
    _providers: Dict[str, Type[LLMProvider]] = {
        "ollama": OllamaProvider,
        "openrouter": OpenRouterProvider,
        "grok": GrokProvider,
    }

    @classmethod
    def register_provider(
        cls,
        provider_type: str,
        provider_class: Type[LLMProvider]
    ) -> None:
        """
        Register a new provider implementation
        
        Args:
            provider_type: Unique provider identifier (e.g., "azure_openai")
            provider_class: Provider class (must inherit from LLMProvider)
            
        Raises:
            ValueError: If provider_type already registered or class invalid
            TypeError: If provider_class doesn't inherit from LLMProvider
        """
        if not issubclass(provider_class, LLMProvider):
            raise TypeError(f"{provider_class} must inherit from LLMProvider")
        
        if provider_type in cls._providers:
            logger.warning(f"Overwriting existing provider: {provider_type}")
        
        cls._providers[provider_type] = provider_class
        logger.info(f"Registered provider: {provider_type}")

    @classmethod
    def create_provider(
        cls,
        config: ProviderConfig
    ) -> LLMProvider:
        """
        Create a provider instance from configuration
        
        Args:
            config: ProviderConfig with provider_type and settings
            
        Returns:
            LLMProvider instance
            
        Raises:
            ValueError: If provider_type is not registered
            Exception: Provider-specific initialization errors
        """
        provider_type = config.provider_type.lower()
        
        if provider_type not in cls._providers:
            available = ", ".join(cls._providers.keys())
            raise ValueError(
                f"Provider '{provider_type}' not found. "
                f"Available providers: {available}"
            )
        
        provider_class = cls._providers[provider_type]
        
        try:
            provider = provider_class(config)
            logger.info(f"Created provider instance: {provider_type}")
            return provider
        except Exception as e:
            logger.error(f"Failed to create provider {provider_type}: {e}")
            raise

    @classmethod
    def get_available_providers(cls) -> Dict[str, Type[LLMProvider]]:
        """
        Get all registered providers
        
        Returns:
            Dict of provider_type -> provider_class
        """
        return cls._providers.copy()

    @classmethod
    def is_provider_available(cls, provider_type: str) -> bool:
        """
        Check if provider is registered
        
        Args:
            provider_type: Provider identifier
            
        Returns:
            True if provider is available
        """
        return provider_type.lower() in cls._providers

    @classmethod
    def get_provider_info(cls) -> Dict[str, str]:
        """
        Get list of available providers with descriptions
        
        Returns:
            Dict of provider_type -> description
        """
        info = {}
        for provider_type, provider_class in cls._providers.items():
            info[provider_type] = provider_class.__doc__ or "No description"
        return info


# Convenience function for quick provider creation
def create_provider(
    provider_type: str,
    api_key: Optional[str] = None,
    api_endpoint: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs
) -> LLMProvider:
    """
    Convenience function to create a provider with minimal configuration
    
    Args:
        provider_type: Provider type (e.g., "ollama", "openrouter")
        api_key: API key for provider (if required)
        api_endpoint: Custom API endpoint (optional)
        model: Default model to use
        **kwargs: Additional configuration
        
    Returns:
        LLMProvider instance
        
    Example:
        provider = create_provider(
            "ollama",
            model="llama2"
        )
    """
    config = ProviderConfig(
        provider_type=provider_type,
        api_key=api_key,
        api_endpoint=api_endpoint,
        model=model,
        metadata=kwargs
    )
    return ProviderFactory.create_provider(config)
