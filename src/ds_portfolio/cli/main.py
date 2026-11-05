"""
CLI Entry Point for DS Portfolio.
Demonstrates Typer for production CLI design.
"""

import typer
from typing import Optional
from rich.console import Console
from rich.table import Table

from ds_portfolio.data.pipeline import DataPipeline, SyntheticDataConfig
from ds_portfolio.ml.trainer import ECommerceConversionPipeline
from ds_portfolio.eval.evaluation import ModelEvaluator, ABTester
from ds_portfolio.viz.visualization import EDAVisualizer, InteractiveDashboard, save_fig
import pandas as pd
import numpy as np

app = typer.Typer(help="Data Science Portfolio CLI - Demonstrates all JD requirements")
console = Console()


@app.command()
def generate_data(
    n_samples: int = typer.Option(10000, help="Number of samples to generate"),
    output: str = typer.Option("data/ecommerce.parquet", help="Output path"),
    seed: int = typer.Option(42, help="Random seed"),
) -> None:
    """Generate synthetic e-commerce dataset with annotation priorities."""
    config = SyntheticDataConfig(n_samples=n_samples, random_seed=seed)
    pipeline = DataPipeline(config)

    console.print("[bold green]Generating synthetic e-commerce data...[/bold green]")
    df = pipeline.generate_e_commerce_dataset()

    console.print("[bold green]Creating annotation strategy...[/bold green]")
    df = pipeline.create_annotation_strategy(df)

    df.to_parquet(output, index=False)
    console.print(f"[bold green]Saved {len(df)} rows to {output}[/bold green]")

    # Show sample
    console.print("\n[bold]Sample data:[/bold]")
    console.print(df.head().to_string())


@app.command()
def train_model(
    data_path: str = typer.Option("data/ecommerce.parquet", help="Input data path"),
    model_type: str = typer.Option("random_forest", help="Model type: random_forest or pytorch"),
    output_model: str = typer.Option("models/conversion_model.pkl", help="Output model path"),
) -> None:
    """Train conversion prediction model."""
    import joblib

    console.print(f"[bold green]Loading data from {data_path}...[/bold green]")
    df = pd.read_parquet(data_path)

    console.print(f"[bold green]Training {model_type} model...[/bold green]")
    pipeline = ECommerceConversionPipeline(model_type=model_type)

    X, y = pipeline.preprocess(df)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pipeline.train(X_train, y_train)
    metrics = pipeline.evaluate(X_test, y_test)

    # Save model
    joblib.dump(pipeline, output_model)

    # Display metrics
    table = Table(title="Model Evaluation Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    for k, v in metrics.to_dict().items():
        if v is not None:
            table.add_row(k, f"{v:.4f}" if isinstance(v, float) else str(v))

    console.print(table)
    console.print(f"[bold green]Model saved to {output_model}[/bold green]")


@app.command()
def evaluate_models(
    data_path: str = typer.Option("data/ecommerce.parquet", help="Input data path"),
) -> None:
    """Compare multiple models with cross-validation."""
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression

    console.print("[bold green]Loading data...[/bold green]")
    df = pd.read_parquet(data_path)

    pipeline = ECommerceConversionPipeline(model_type="random_forest")
    X, y = pipeline.preprocess(df)

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
        "LogisticRegression": LogisticRegression(random_state=42, max_iter=1000),
    }

    evaluator = ModelEvaluator(cv_folds=5)
    results_df = evaluator.compare_models(models, X, y)

    console.print("\n[bold]Model Comparison Results:[/bold]")
    console.print(results_df.to_string(index=False))


@app.command()
def run_ab_test(
    metric: str = typer.Option("conversion_rate", help="Metric to test"),
    n_a: int = typer.Option(1000, help="Variant A sample size"),
    n_b: int = typer.Option(1000, help="Variant B sample size"),
    conv_a: float = typer.Option(0.12, help="Variant A conversion rate"),
    conv_b: float = typer.Option(0.15, help="Variant B conversion rate"),
) -> None:
    """Run A/B test simulation."""
    import numpy as np

    np.random.seed(42)
    variant_a = np.random.binomial(1, conv_a, n_a)
    variant_b = np.random.binomial(1, conv_b, n_b)

    tester = ABTester()

    if metric == "conversion_rate":
        result = tester.run_proportion_ztest(
            successes_a=int(variant_a.sum()), trials_a=n_a,
            successes_b=int(variant_b.sum()), trials_b=n_b
        )
    else:
        result = tester.run_ttest(variant_a, variant_b, metric)

    table = Table(title="A/B Test Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    for field in ["metric_name", "variant_a_mean", "variant_b_mean", "p_value",
                  "effect_size", "is_significant", "recommendation"]:
        val = getattr(result, field)
        table.add_row(field, f"{val:.4f}" if isinstance(val, float) else str(val))

    console.print(table)


@app.command()
def visualize(
    data_path: str = typer.Option("data/ecommerce.parquet", help="Input data path"),
    output_dir: str = typer.Option("outputs/viz", help="Output directory for charts"),
) -> None:
    """Generate EDA visualizations."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    console.print("[bold green]Loading data...[/bold green]")
    df = pd.read_parquet(data_path)

    # Create target
    df["high_purchase"] = (df["purchase_amount"] > 50).astype(int)

    visualizer = EDAVisualizer()

    console.print("[bold green]Generating target distribution...[/bold green]")
    fig = visualizer.plot_target_distribution(df, "high_purchase")
    save_fig(fig, f"{output_dir}/target_distribution.png")

    console.print("[bold green]Generating numerical distributions...[/bold green]")
    fig = visualizer.plot_numerical_distributions(df,
        ["session_duration_sec", "pages_visited", "purchase_amount"], "high_purchase")
    save_fig(fig, f"{output_dir}/numerical_distributions.png")

    console.print("[bold green]Generating correlation heatmap...[/bold green]")
    fig = visualizer.plot_correlation_heatmap(df,
        ["session_duration_sec", "pages_visited", "purchase_amount"])
    save_fig(fig, f"{output_dir}/correlation_heatmap.png")

    console.print("[bold green]Generating categorical analysis...[/bold green]")
    fig = visualizer.plot_categorical_vs_target(df, ["category", "device"], "high_purchase")
    save_fig(fig, f"{output_dir}/categorical_vs_target.png")

    console.print(f"[bold green]Visualizations saved to {output_dir}/[/bold green]")


@app.command()
def demo_prompt_engineering(
    task: str = typer.Option("sentiment_analysis", help="Task to demo"),
    provider: str = typer.Option("openai", help="LLM provider: openai or anthropic"),
) -> None:
    """Demonstrate prompt engineering capabilities."""
    import asyncio
    from ds_portfolio.nlp.prompt_engineering import PromptEngineer, TaskType

    async def run_demo():
        engineer = PromptEngineer(provider=provider)

        if task == "sentiment_analysis":
            test_inputs = [
                {"text": "Absolutely love this product! Best purchase ever."},
                {"text": "It's okay, nothing special. Average quality."},
                {"text": "Terrible experience. Broken on arrival, waste of money."},
            ]
            for inp in test_inputs:
                result = await engineer.execute_task(TaskType.SENTIMENT_ANALYSIS, inp)
                console.print(f"Input: {inp['text']}")
                console.print(f"Result: {result.model_dump_json(indent=2)}")
                console.print("---")

        elif task == "qa":
            test_inputs = [
                {"context": "The company was founded in 2020. It has 500 employees.", "question": "When was the company founded?"},
                {"context": "Our return policy allows 30 days for full refund.", "question": "What is the return window?"},
            ]
            for inp in test_inputs:
                result = await engineer.execute_task(TaskType.QUESTION_ANSWERING, inp)
                console.print(f"Q: {inp['question']}")
                console.print(f"A: {result.answer} (confidence: {result.confidence})")
                console.print("---")

    asyncio.run(run_demo())


if __name__ == "__main__":
    app()