# Realest Tate Agent

This is a Python+LangGraph agent example with CLI human interaction and LangSmith for tracing.

It implements a flow with multiple conditions as a workflow. It does not use a swarm with a supervisor, since the steps are well-defined beforehand.

![Graph visualization](./graph.png)

## Requirements

* `uv` for dependency management
* LangSmith API key for tracing
* AWS profile for Bedrock LLMs

## Use

1. Install the project locally with `uv sync`
2. Create `.env` and/or `.secrets` file with the variables demonstrated in the .env.example file
3. Activate the virtual environment created by uv in `.venv` and run the `src/realest_tate_agent/flow.py` script, or run it with `uv run realest_tate_agent`
4. Enjoy

## Potential improvements

* Return state updates instead of forced overwrites
* Implement sub-graphs with independent states
* Implement human interrupts for bubbling to a frontend, and a backend endpoint for the resume command
* Replace mock properties with:
    * injected variables in a SQL query
    * semantic searches in a vector DB
* Implement more robust structured output parsing
* Control renting user's final output format (LLMs want to output md)
