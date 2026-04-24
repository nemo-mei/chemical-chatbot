# Project Structure

## Root files
- `app_langchain.py`: primary Gradio app for the LangChain version
- `app.py`: fallback / simpler router app
- `langchain_bot.py`: LangChain tool wiring and bot orchestration
- `router.py`: deterministic routing fallback
- `requirements.txt`: Python dependencies
- `.env.example`: environment variable template
- `README.md`: portfolio-facing project overview
- `DEPLOY_TO_HF_SPACES.md`: deployment steps for Hugging Face Spaces
- `SUBMISSION_CUSTOMIZATION.md`: last-minute placeholder replacement guide
- `DEMO_SCRIPT.md`: short presentation script for class or portfolio demo

## Data and knowledge base
- `data/chemicals.csv`: structured chemical catalog
- `data/pricing_rules.csv`: quote rules by region
- `docs/*.md`: FAQ and policy documents used for retrieval
- `assets/routing_logic.svg`: routing diagram for the README

## Operational files
- `logs/conversations.csv`: runtime conversation log
- `logs/escalations.csv`: human handoff queue
- `logs/demo_conversations.csv`: sample exchanges for the README

## Source code
- `tools/chemical_lookup.py`: catalog lookup tool
- `tools/quote_generator.py`: deterministic quote tool
- `tools/faq_rag.py`: RAG / fallback FAQ tool
- `tools/escalation.py`: escalation logging tool

## Tests
- `tests/test_tools.py`
- `tests/test_bot_flow.py`
- `tests/test_langchain_bot.py`
- `tests/test_faq_rag_upgrade.py`
