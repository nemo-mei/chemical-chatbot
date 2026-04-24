# 2-Minute Demo Script

## Opening
This project is a LangChain-based customer support chatbot for the chemical industry. It handles four main tasks: chemical property lookup, preliminary quote generation, FAQ answering through RAG, and escalation for cases that need a human representative.

## Demo flow
1. Ask: `What is the storage condition for acetone?`
   - Explain that this triggers the structured chemical lookup tool.

2. Ask: `Give me a quote for 250 kg ethanol shipped to California.`
   - Explain that the bot combines product data and pricing rules to estimate a preliminary quote.

3. Ask: `What is your minimum order quantity?`
   - Explain that this goes through the FAQ knowledge base and uses RAG when the full LangChain stack is available.

4. Ask: `I need a human representative for a damaged shipment complaint.`
   - Explain that the bot records the escalation and sends the case for human follow-up.

## Closing
The main thing I learned was that multi-tool design is more reliable than a single free-form prompt. Structured tasks work best with deterministic tools, while policy questions benefit from retrieval, and exception handling should always escalate cleanly.
