from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """
    Abstract base class for trading agents.
    Defines the interface for all specialized and ensemble agents.
    """
    @abstractmethod
    def choose_action(self, observation):
        """
        Given an observation (or dict of observations for multi-agent),
        return action, log_prob, value estimation.
        """
        pass

    @abstractmethod
    def remember(self, observation, action, log_prob, value, reward, done):
        """Store experience for learning."""
        pass

    @abstractmethod
    def learn(self):
        """Perform learning update using stored experiences."""
        pass

    @abstractmethod
    def save_models(self, path: str):
        """Save model parameters to disk."""
        pass

    @abstractmethod
    def load_models(self, path: str):
        """Load model parameters from disk."""
        pass