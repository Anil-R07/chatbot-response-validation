# Chatbot Response Validation — Langfuse Setup

## 1. Self-host Langfuse (single node, good enough for this use case)

```bash
git clone https://github.com/langfuse/langfuse.git
cd langfuse
docker compose up -d
```

This starts Postgres, ClickHouse, Redis, MinIO (S3-compatible storage), and
the Langfuse web app together - no separate infrastructure to provision for
a solo POC. Open http://localhost:3000, create an account (local, not tied
to any org), and create a project. Grab the project's API keys from
Settings -> API Keys.

## 2. Set environment variables

```bash
export LANGFUSE_PUBLIC_KEY="pk-lf-..."
export LANGFUSE_SECRET_KEY="sk-lf-..."
export LANGFUSE_HOST="http://localhost:3000"
export OPENAI_API_KEY="sk-..."     # never hardcode this in any script
```

## 3. Upload your ground truth once

```bash
pip install langfuse openai
python dataset.py
```

Re-run this any time you add a new test case (e.g. after a false alert from
a live session, per Section 5 of the POC — add the real message with the
correct label, then re-upload).

## 4. Run the evaluation gate

```bash
python script.py "chatbot-v1-baseline"
```

Run it again with a new name whenever you change `CHATBOT_SYSTEM_PROMPT` or
`CHATBOT_MODEL` (e.g. `"chatbot-v2-tightened-refusal"`). Every run shows up
in Langfuse under the dataset's **Runs** tab, so you can compare pass rate,
per-category breakdown, and the zero-tolerance leakage category side by
side against the previous run before deciding whether to promote a change.

## 5. Set up the annotation queue for judge spot-checks (Section 9)

This part is done in the Langfuse UI, not code:

1. Go to your project -> **Annotation Queues** -> **New Queue**.
2. Name it e.g. `judge-spot-checks`.
3. Add a scoring config matching your judge's verdict (e.g. a categorical
   score: agree / disagree with the judge).
4. From the **Traces** table, filter to a run, sort/filter to the FAIL
   verdicts (and a random sample of PASS verdicts), select them, and
   **Add to queue**.
5. Work through the queue yourself under **Annotation Queues** -> your
   queue - each item shows the question, chatbot response, and judge
   reasoning together so you can confirm or override the verdict.

Since you're the only reviewer for now, there's no need to configure
multi-reviewer assignment - just work the queue directly.

## Notes

- `script.py` currently re-runs every test case against a live
  API on every call, which costs real API tokens each time. Fine for a POC
  of a few dozen-to-hundred cases; if the ground-truth set grows into the
  thousands, consider caching chatbot responses the way the earlier
  notebook POCs did (`predictions.json`).
- The judge model (`gpt-4o`) should stay equal-or-stronger than the chatbot
  model (`gpt-4o-mini`), per Section 3 of the POC - don't downgrade the
  judge to save cost, since that undermines the whole gate.