# Superstore Analysis Dashboard

An interactive Superstore sales & profit dashboard built with Preswald.  
Features:
- Sidebar controls (row slider + "All Segments" selector)  
- Five Plotly visualizations (scatter, bar, histogram, line, choropleth)  
- Built-in data prep (state code mapping & profit-margin calculation)  
- Configurable via `preswald.toml`  
- Packaged example under `my_first_preswald_app/hello.py`

---

## Table of Contents

1. [Prerequisites](#prerequisites)  
2. [Installation](#installation)  
3. [Configuration](#configuration)  
4. [Running the Dashboard](#running-the-dashboard)  
5. [Project Layout](#project-layout)  
6. [Examples](#examples)  
7. [Testing](#testing)  
8. [Contributing](#contributing)  
9. [License](#license)  

---

## Prerequisites

- Python 3.10+  
- `pip` or `poetry`  
- Internet access to download Preswald and Plotly  

---

## Installation

```bash
# Install Preswald and dependencies
pip install preswald plotly pandas
```

---

## Configuration

Your `preswald.toml` (at project root) should look like this:

```toml
[project]
title       = "Superstore Analysis Dashboard"
version     = "0.1.0"
port        = 8501
slug        = "my-first-preswald-app-540052"
entrypoint  = "hello.py"

[branding]
name         = "Superstore Analysis Dashboard"
logo         = "images/logo.png"
favicon      = "images/favicon.ico"
primaryColor = "#F89613"

[data.my_sample_superstore]
type = "csv"
path = "data/my_sample_superstore.csv"

[logging]
level  = "INFO"
format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

* **title**: Human-readable name.
* **version**: App version.
* **port**: Default web port (e.g. Streamlit default).
* **slug**: Unique identifier for deployments.
* **entrypoint**: Path to your example script.
* **branding**: Appearance in the UI header.
* **data.***: Define your data sources.
* **logging**: Control verbosity and format.

---

## Running the Dashboard

From your project directory:

```bash
preswald run my_first_preswald_app/hello.py
```

Then open your browser at `http://localhost:8500` (or the port you configured).

---

## Project Layout

```
.
├── data/
│   └── my_sample_superstore.csv
├── images/
│   ├── logo.png
│   └── favicon.ico
├── my_first_preswald_app/
│   ├── __init__.py
│   └── hello.py
├── preswald.toml
├── README.md
└── tests/
    └── test_hello_example.py
```

---

## Examples

### Superstore Dashboard

* **Location**: `my_first_preswald_app/hello.py`
* **Controls**:

  * "Number of Rows to Display" slider
  * "Customer Segment" select box (with "All Segments" default)
* **Visuals**:

  1. Sales vs Profit scatter
  2. Sales by Category bar chart
  3. Profit-Margin histogram
  4. Orders over time line chart
  5. Profit by State choropleth

---

## Testing

We include basic smoke and data-integrity tests:

```bash
pytest tests/test_hello_example.py
```

* Checks that `Profit Margin` column exists and is float.
* Verifies complete state-code mappings.
* Ensures `main()` runs without errors.

---

## Contributing

1. Fork the repo.
2. Create a feature branch (`git checkout -b feat/my-feature`).
3. Make your changes, write tests, and update docs.
4. Submit a PR with a clear title and description.

Please follow the `feat(example): …` commit prefix for demo/example changes.

---

## License

MIT © Your Name

