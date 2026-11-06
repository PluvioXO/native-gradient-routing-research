"""Joint MLP/attention sweep; one pointer task, eight reserved task slots."""
CHANNELS = (1, 2, 4, 6, 13, 26, 51, 103, 117, 206, 411)
RANKS = (1, 2, 4, 8, 16, 18, 32, 64)
SEEDS = (31, 47, 71)
LAYERS = (0, 3, 6, 9, 13, 16, 19, 23)
NATIVE_SCALARS_PER_CHANNEL = 3 * 896 * len(LAYERS)
LORA_MLP_SCALARS_PER_RANK = 3 * (896 + 4864) * len(LAYERS)
LORA_ATTENTION_SCALARS_PER_RANK = (2 * (896 + 896) + 2 * (896 + 128)) * len(LAYERS)
LORA_SCALARS_PER_RANK = LORA_MLP_SCALARS_PER_RANK + LORA_ATTENTION_SCALARS_PER_RANK
ACTIVE_TASKS = (1,)


def attention_features(channels, intermediate=4864, kv_width=128):
    return max(1, round(channels * kv_width / intermediate))


def cells(smoke=False):
    widths, ranks = ((1, 117, 411), (1, 18, 64)) if smoke else (CHANNELS, RANKS)
    result = []
    for c in widths:
        kv = attention_features(c)
        mlp, attention = c * NATIVE_SCALARS_PER_CHANNEL, 2 * 896 * (7 + 1) * kv * len(LAYERS)
        result.append(dict(support="native", channels=c, rank=None, budget_id=f"c{c:03d}",
            attention_kv_features=kv, attention_query_features=7 * kv,
            mlp_scalars_per_slot=mlp, attention_scalars_per_slot=attention,
            scalars_per_slot=mlp + attention))
    result += [dict(support="lora", channels=None, rank=r, budget_id=f"r{r:03d}",
                    attention_kv_features=None, attention_query_features=None,
                    mlp_scalars_per_slot=r * LORA_MLP_SCALARS_PER_RANK,
                    attention_scalars_per_slot=r * LORA_ATTENTION_SCALARS_PER_RANK,
                    scalars_per_slot=r * LORA_SCALARS_PER_RANK) for r in ranks]
    return result
