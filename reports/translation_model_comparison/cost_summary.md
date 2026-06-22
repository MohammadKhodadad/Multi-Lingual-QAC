# Translation model comparison — cost & token usage

Same 30 English documents translated to simplified Chinese by each model, using the identical translation prompt from `translate_to_chinese.py`.

| Model | slug | ok | avg in tok | avg out tok | reasoning tok (total) | measured cost (30 docs) | est. 100 docs (measured) | est. 100 docs (list price) |
|---|---|---|---|---|---|---|---|---|
| gemma-4-31b | `google/gemma-4-31b-it` | 30/30 | 631 | 1780 | 50066 | $0.0242 | $0.0807 | $0.0699 |
| qwen3.6-35b-a3b | `qwen/qwen3.6-35b-a3b` | 30/30 | 629 | 3603 | 114185 | $0.1330 | $0.4432 | $0.3691 |
| qwen3.7-max | `qwen/qwen3.7-max` | 30/30 | 629 | 1548 | 41309 | $0.1978 | $0.6592 | $0.6592 |
| gpt-5.5 | `gpt-5.5` | 30/30 | 606 | 462 | 7376 | $0.5066 | $1.6885 | $1.6885 |

- **measured cost** uses the provider-reported dollar cost for the OpenRouter models (promo-inclusive) and list-price × token counts for gpt-5.5.
- **est. 100 docs (measured)** = measured cost / ok docs × 100.
- **est. 100 docs (list price)** is a promo-free cross-check from the list rates in `PRICING` × observed average tokens.
- `qwen/qwen3.7-max` measured cost reflects the current 50% OpenRouter promo; the list-price column shows the un-discounted estimate.
