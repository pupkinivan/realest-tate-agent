# Realest Tate Agent

This is a Python+LangGraph agent example with CLI human interaction (without interrupts).

It implements a flow with multiple conditions as a workflow. It does not use a swarm with a supervisor, since the steps are well-defined beforehand.

![Graph visualization](./graph.png)

## Potential improvements

* Return state updates instead of forced overwrites
* Implement sub-graphs with independent states
* Implement human interrupts for bubbling to a frontend, and a backend endpoint for the resume command
* Replace mock properties with:
    * injected variables in a SQL query
    * semantic searches in a vector DB
* Implement more robust structured output parsing
* Control renting user's final output format (LLMs want to output md)
