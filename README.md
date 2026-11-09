# SLM Agents Demo — a mini router

A small, readable demo for the paper [*Small Language Models are the Future of Agentic AI*](https://arxiv.org/abs/2506.02153) (Belcak et al., NVIDIA Research, 2025).

It runs the same 10 agent requests (English and Egyptian Arabic) twice:

1. **All large**: every request goes to a 7B model.
2. **Routed**: a 1.5B model first decides what kind of request it is. Intent detection, field extraction and tool calls stay on the small model. Open-ended planning goes to the large one.

Then it compares **latency, tokens, estimated cost and accuracy**, logs every call to `results/calls.jsonl` (step S1 of the paper's conversion algorithm), and saves a chart to `results/summary.png`.

## Run it

```bash
# 1. Install Ollama (https://ollama.com) and pull two models
ollama pull qwen2.5:1.5b
ollama pull qwen2.5:7b

# 2. Install Python packages
pip install -r requirements.txt

# 3. Run
python router.py

# Rehearse without any model (simulated numbers, clearly marked as mock)
python router.py --mock
```

Swap models with environment variables, for example `SMALL_MODEL=qwen2.5:3b LARGE_MODEL=llama3.1:8b python router.py`.

## What you see

- A table of every routed call: language, what the router decided, which model answered, whether the answer was correct, latency and cost.
- A summary table: all-large vs routed, with the change in latency, cost per 1,000 requests, accuracy, and the share of calls the small model handled.
- `results/summary.png`: the same comparison as a chart, ready for a slide.

## Notes

- **Cost is an estimate.** Both models run locally for free. The cost column prices tokens as if each tier were a hosted API of its size, using the illustrative prices in `config.py`. Put your own provider's prices there.
- **Accuracy is checked automatically** for intent, extraction and tool calls. Planning answers are open-ended and not scored.
- **The router's own call counts.** Its latency and tokens are added to every routed request, so routing is not free.
- Ten tasks is a demo, not a benchmark. Add your own to `tasks.json`.

## Files

| File | What it does |
| --- | --- |
| `router.py` | Routing, model calls, scoring, tables and chart |
| `tasks.json` | The 10 requests with expected answers |
| `config.py` | Model names, prices, which task types stay small |

---

## بالعربي

ديمو بسيط لفكرة الورقة: بدل ما كل طلب يروح لموديل كبير، موديل صغير (1.5B) بيحدد نوع الطلب الأول. المهام البسيطة زي فهم الـ intent واستخراج البيانات والـ tool calls بتفضل عند الموديل الصغير، والأسئلة المفتوحة اللي محتاجة تفكير بتروح للموديل الكبير (7B).

الكود بيشغّل نفس الـ 10 طلبات (إنجليزي ومصري) مرتين، مرة كله على الكبير ومرة بالـ router، ويقارن الـ latency والتكلفة التقديرية والدقة، ويسجّل كل call في `results/calls.jsonl`، ويطلّع chart جاهز للسلايد.

للتشغيل: نزّلي Ollama والموديلين، `pip install -r requirements.txt`، وبعدين `python router.py`. ولو عايزة تتمرني من غير موديلات: `python router.py --mock`.
