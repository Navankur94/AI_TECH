# Push this repository to GitHub

Target repository: https://github.com/Navankur94/AI_TECH

## 1. Unzip and check that no secret will be committed

```bash
unzip AI_TECH.zip && cd AI_TECH
git init -b main
git add .
git status                      # .env files must NOT appear; only .env.example files should
git check-ignore -v rag-knowledge-assistant/.env    # should print the .gitignore rule that ignores it
```

## 2. Commit and push

If the GitHub repository is empty:

```bash
git commit -m "Add RAG assistant, LangGraph triage agent and LLM gateway with evals"
git remote add origin https://github.com/Navankur94/AI_TECH.git
git push -u origin main
```

If the repository already has commits (for example a README created on GitHub):

```bash
git commit -m "Add RAG assistant, LangGraph triage agent and LLM gateway with evals"
git remote add origin https://github.com/Navankur94/AI_TECH.git
git pull origin main --allow-unrelated-histories --no-edit
git push -u origin main
```

GitHub asks you to sign in when you push over HTTPS: use a personal access token as the password, or
sign in with the GitHub CLI (`gh auth login`).

## 3. After the push

- Open the **Actions** tab. The `CI` workflow runs each project's tests and builds each Docker image.
  Fix anything that fails before you share the repository.
- Run one project end to end with your own `OPENAI_API_KEY` (see its README) before describing it as
  working on OpenAI.
- If a real key was ever committed, revoke it at the provider immediately and create a new one.
  Deleting the file in a later commit does not remove it from history.
