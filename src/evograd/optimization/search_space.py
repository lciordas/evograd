"""
Search space definitions for Bayesian optimization.

This module provides classes for defining hyperparameter
search spaces that can be used with Optuna for optimization.
"""

from abc    import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import optuna   # type: ignore

class Parameter(ABC):
    """Abstract base class for a single hyperparameter."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def suggest(self, trial: optuna.Trial) -> Any:
        """Generate a suggestion for this parameter using an Optuna trial."""
        pass

class FloatParameter(Parameter):
    """A floating-point parameter."""

    def __init__(self, name: str, low: float, high: float, log: bool = False, step: Optional[float] = None):
        """
        Parameters:
            name: Parameter name
            low:  Lower bound
            high: Upper bound
            log:  Use logarithmic scale
            step: Step size for discrete values (None for continuous)
        """
        super().__init__(name)
        self.low  = low
        self.high = high
        self.log  = log
        self.step = step

    def suggest(self, trial: optuna.Trial) -> float:
        """Generate a float suggestion."""
        if self.step is not None:
            return trial.suggest_float(self.name, self.low, self.high, step=self.step)
        else:
            return trial.suggest_float(self.name, self.low, self.high, log=self.log)

class IntParameter(Parameter):
    """An integer parameter."""

    def __init__(self, name: str, low: int, high: int, log: bool = False, step: int = 1):
        """
        Parameters:
            name: Parameter name
            low:  Lower bound
            high: Upper bound
            log:  Use logarithmic scale
            step: Step size
        """
        super().__init__(name)
        self.low  = low
        self.high = high
        self.log  = log
        self.step = step

    def suggest(self, trial: optuna.Trial) -> int:
        """Generate an integer suggestion."""
        return trial.suggest_int(self.name, self.low, self.high, log=self.log, step=self.step)

class CategoricalParameter(Parameter):
    """Categorical parameter with discrete choices."""

    def __init__(self, name: str, choices: List[Any]):
        """
        Parameters:
            name:    Parameter name
            choices: List of possible values
        """
        super().__init__(name)
        self.choices = choices

    def suggest(self, trial: optuna.Trial) -> Any:
        """Generate a categorical suggestion."""
        return trial.suggest_categorical(self.name, self.choices)

class SearchSpace:
    """
    Container for defining hyperparameter search spaces.

    This class allows you to programmatically define which 
    hyperparameters to optimize and their ranges/choices.

    Example:
        >>> search_space = SearchSpace()
        >>> search_space.add_float('learning_rate', 0.001, 0.1, log=True)
        >>> search_space.add_int('population_size', 50, 500)
        >>> search_space.add_categorical('activation', ['relu', 'tanh', 'sigmoid'])
    """

    def __init__(self):
        self.parameters : List[Parameter]      = []  # List of parameters in insertion order
        self._param_dict: Dict[str, Parameter] = {}  # Fast lookup by parameter name

    def add_float(self, name: str, 
                  low: float, high: float, log: bool = False, 
                  step: Optional[float] = None) -> 'SearchSpace':
        """
        Add a continuous float parameter to the search space.

        Parameters:
            name: Parameter name (must match Config attribute name)
            low:  Lower bound (inclusive)
            high: Upper bound (inclusive)
            log:  Whether to use log scale sampling
            step: Step size for discrete sampling (None for continuous)

        Returns:
            self for method chaining
        """
        param = FloatParameter(name, low, high, log, step)
        self.parameters.append(param)
        self._param_dict[name] = param
        return self

    def add_int(self, name: str, low: int, high: int, log: bool = False, step: int = 1) -> 'SearchSpace':
        """
        Add an integer parameter to the search space.

        Parameters:
            name: Parameter name (must match Config attribute name)
            low:  Lower bound (inclusive)
            high: Upper bound (inclusive)
            log:  Whether to use log scale sampling
            step: Step size

        Returns:
            self for method chaining
        """
        param = IntParameter(name, low, high, log, step)
        self.parameters.append(param)
        self._param_dict[name] = param
        return self

    def add_categorical(self, name: str, choices: List[Any]) -> 'SearchSpace':
        """
        Add a categorical parameter to the search space.

        Parameters:
            name:    Parameter name (must match Config attribute name)
            choices: List of possible values

        Returns:
            self for method chaining
        """
        param = CategoricalParameter(name, choices)
        self.parameters.append(param)
        self._param_dict[name] = param
        return self

    def suggest(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Generate parameter suggestions for an Optuna trial.

        Parameters:
            trial: Optuna trial object

        Returns:
            Dictionary mapping parameter names to suggested values
        """
        suggestions = {}
        for param in self.parameters:
            suggestions[param.name] = param.suggest(trial)
        return suggestions

    def get_param_names(self) -> List[str]:
        """Get list of all parameter names in the search space."""
        return [p.name for p in self.parameters]

    def __len__(self) -> int:
        """Number of parameters in the search space."""
        return len(self.parameters)

    def __repr__(self) -> str:
        """String representation of the search space."""
        param_strs = []
        for param in self.parameters:
            if isinstance(param, FloatParameter):
                param_strs.append(f"FloatParameter('{param.name}', {param.low}, {param.high}, "
                                  f"log={param.log}, step={param.step})")
            elif isinstance(param, IntParameter):
                param_strs.append(f"IntParameter('{param.name}', {param.low}, {param.high}, "
                                  f"log={param.log}, step={param.step})")
            elif isinstance(param, CategoricalParameter):
                param_strs.append(f"CategoricalParameter('{param.name}', {param.choices})")

        params_str = ',\n    '.join(param_strs)
        return f"SearchSpace([\n    {params_str}\n])"