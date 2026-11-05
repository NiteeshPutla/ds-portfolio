"""DS Portfolio - Data Science Portfolio Project."""
from ds_portfolio.core.base import (
    ScalerProtocol,
    EstimatorProtocol,
    EvaluationMetrics,
    AbstractModelPipeline,
)

from ds_portfolio.data.pipeline import (
    DataPipeline,
    SyntheticDataConfig,
    PriorityLevel,
)

from ds_portfolio.ml.trainer import (
    ECommerceConversionPipeline,
    TabularPyTorchDataset,
    DeepTabularClassifier,
)

from ds_portfolio.nlp.prompt_engineering import (
    PromptEngineer,
    PromptEvaluator,
    PromptTemplate,
    TaskType,
    SearchResult,
    QAResponse,
    ChatResponse,
    SentimentOutput,
    SummarizationOutput,
)

from ds_portfolio.viz.visualization import (
    EDAVisualizer,
    InteractiveDashboard,
    ChartConfig,
)

from ds_portfolio.eval.evaluation import (
    ModelEvaluator,
    ABTester,
    PromptEvalFramework,
    ModelEvaluationResult,
    ABTestResult,
)

__all__ = [
    # Core
    "ScalerProtocol",
    "EstimatorProtocol",
    "EvaluationMetrics",
    "AbstractModelPipeline",
    # Data
    "DataPipeline",
    "SyntheticDataConfig",
    "PriorityLevel",
    # ML
    "ECommerceConversionPipeline",
    "TabularPyTorchDataset",
    "DeepTabularClassifier",
    # NLP/Prompt Engineering
    "PromptEngineer",
    "PromptEvaluator",
    "PromptTemplate",
    "TaskType",
    "SearchResult",
    "QAResponse",
    "ChatResponse",
    "SentimentOutput",
    "SummarizationOutput",
    # Viz
    "EDAVisualizer",
    "InteractiveDashboard",
    "ChartConfig",
    # Eval
    "ModelEvaluator",
    "ABTester",
    "PromptEvalFramework",
    "ModelEvaluationResult",
    "ABTestResult",
]

__version__ = "0.1.0"