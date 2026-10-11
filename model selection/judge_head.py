"""Project Qwen's final hidden state in float32 before any output rounding."""


def project_hidden(model, hidden, token_ids=(32, 33)):
    import mlx.core as mx
    if getattr(model, "model_type", None) != "qwen3":
        raise ValueError("This precision path requires the audited Qwen3 architecture.")
    head = model.model.embed_tokens if model.args.tie_word_embeddings else model.lm_head
    if hasattr(head, "scales"):
        raise ValueError("Quantized output heads require a separate precision audit.")
    weight = head.weight if token_ids is None else head.weight[mx.array(token_ids)]
    scores = hidden.astype(mx.float32) @ weight.astype(mx.float32).T
    if not model.args.tie_word_embeddings and "bias" in head:
        bias = head.bias if token_ids is None else head.bias[mx.array(token_ids)]
        scores = scores + bias.astype(mx.float32)
    return scores
