Business qustion:
Can we trust our retail sales forecasts?


I am imagining I am the ML engineer responsible for building a forecasting system for Rossmann.

I want to predict future sales.

Before deployment, I need to answer four questions:

| Decision                    | What you need to establish                                                         |
| --------------------------- | ---------------------------------------------------------------------------------- |
| Model evaluation            | Does our validation process give us a trustworthy estimate of future performance?  |
| History availability        | Does forecasting performance deteriorate when a store has less historical data?    |
| Model architecture strategy | Should we train one global model or individual models per store?                   |
| Hyperparameter tuning       | Does tuning genuinely improve performance, or does it overfit the validation data? |


Questions I want to answer:

1. Validation: Am I measuring model performance correctly?
2. History: How much historical information does the model need?
3. Model strategy: Should stores share a model or have individual models?
4. Tuning: Am I genuinely improving the model or just fitting my validation data?

That's the engineering problem.

And one correction to our original challenge: don't treat these four investigations as mandatory regardless of what you discover. If an earlier experiment reveals that a proposed comparison isn't meaningful, revise the investigation and document why.


Run yourself:
```bash
# Create virtual environment
python -m venv .venv

# Activate it
. .venv\Scripts\activate

# Install everything from requirements.txt
uv pip install -r requirements.txt
```

rossmann-forecasting-reliability/
│
├── data/
│
├── notebooks/
│   ├── 01_data_investigation.ipynb
│   ├── 02_validation_experiments.ipynb
│   └── 03_hyperparameter_experiments.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── features.py
│   ├── validation.py
│   ├── training.py
│   └── evaluation.py
│
├── results/
│
└── docs/
    ├── experimental_design.md
    └── findings.md