# Final Submission Checklist

## Customize before submission
- [ ] Replace `YOUR_GITHUB_REPO_URL` in `README.md`
- [ ] Replace `YOUR_HUGGING_FACE_SPACE_URL` in `README.md`
- [ ] Confirm the repo folder name matches the README examples
- [ ] Confirm `.env` is **not** committed

## Validate locally
- [ ] Install dependencies with `pip install -r requirements.txt`
- [ ] Run `pytest -q`
- [ ] Launch `python app_langchain.py`
- [ ] Test all four core flows: lookup, quote, FAQ, escalation

## Deploy
- [ ] Create a Hugging Face Gradio Space
- [ ] Upload all repository files
- [ ] Add `GOOGLE_API_KEY` as a Space secret
- [ ] Confirm `app_langchain.py` is the entry point
- [ ] Test the live Space with at least four example queries

## Turn it into a portfolio piece
- [ ] Push the final repository to GitHub
- [ ] Pin the repo on your GitHub profile
- [ ] Add the live Space link to the README
- [ ] Prepare 3–4 sample conversations for class/demo
- [ ] Record a short demo video if you want the optional showcase
