# AI Models, Prompts and Configuration

This directory stores reusable AI assets used by the CRM platform.

## prompts/

Contains reusable prompt templates used by the AI agent.

The files mirror the prompts currently embedded in `src/agents.py`; the runtime code still uses the embedded prompts for stability.

- customer_agent.md
- ticket_summary.md
- factual_router.md
- system.md

## configs/

Contains model configuration files.

Examples:

- ollama_config.json

## Agent State

The project uses LangGraph for workflow orchestration.

Agent state is maintained in memory during workflow execution and is not persisted to disk because persistent state is unnecessary for the current CRM use case.
