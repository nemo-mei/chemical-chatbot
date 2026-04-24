# Deploy to Hugging Face Spaces

## 1. Create the Space
Create a new **Gradio** Space on Hugging Face.

Recommended settings:
- SDK: `gradio`
- App file: `app_langchain.py`
- Python version: `3.11`

## 2. Upload the repository
Upload the full contents of this repository root, including:
- `app_langchain.py`
- `langchain_bot.py`
- `router.py`
- `tools/`
- `data/`
- `docs/`
- `assets/`
- `requirements.txt`
- `README.md`

Do **not** upload your local `.env` file.

## 3. Add secrets
In the Space settings, add:
- `GOOGLE_API_KEY` = your Gemini API key

Optional:
- `GOOGLE_GENAI_MODEL` = your preferred Gemini model name

## 4. Verify startup
The Space should install dependencies from `requirements.txt` and launch `app_langchain.py`.

## 5. Smoke test the app
Run at least these four prompts:
1. `What is the storage condition for acetone?`
2. `Give me a quote for 250 kg ethanol shipped to California.`
3. `What is your minimum order quantity?`
4. `I need a human representative for a damaged shipment complaint.`

## 6. Finish the portfolio details
After deployment:
- Replace `YOUR_HUGGING_FACE_SPACE_URL` in `README.md`
- Replace `YOUR_GITHUB_REPO_URL` in `README.md`
- Commit and push the final README update
