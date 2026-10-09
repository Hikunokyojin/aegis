"""Verifies CodeBERT embeddings have the expected shape and carry real
semantic signal (a near-duplicate function embeds closer than an unrelated
one). Requires network access on first run to download the model weights."""
import torch
import torch.nn.functional as F

from aegis.embeddings.codebert_embedder import embed, load_model

ORIGINAL = """void copy_input(const char* user_input) {
    char buffer[64];
    strcpy(buffer, user_input);
    printf("Copied: %s\\n", buffer);
}"""

NEAR_DUPLICATE = """void handle_request(const char* raw_data) {
    char temp[64];
    strcpy(temp, raw_data);
    printf("Received: %s\\n", temp);
}"""

UNRELATED = """int compute_average(int a, int b, int c) {
    int total = a + b + c;
    return total / 3;
}"""


def test_load_model_places_model_on_the_requested_device():
    _, model = load_model(device="cpu")

    assert next(model.parameters()).device.type == "cpu"


def test_embedding_shape_is_768_dim():
    tokenizer, model = load_model()
    vector, token_count = embed(tokenizer, model, ORIGINAL)

    assert len(vector) == 768
    assert token_count > 0


def test_near_duplicate_embeds_closer_than_unrelated():
    tokenizer, model = load_model()
    v_original, _ = embed(tokenizer, model, ORIGINAL)
    v_near, _ = embed(tokenizer, model, NEAR_DUPLICATE)
    v_unrelated, _ = embed(tokenizer, model, UNRELATED)

    e_original = torch.tensor(v_original).unsqueeze(0)
    e_near = torch.tensor(v_near).unsqueeze(0)
    e_unrelated = torch.tensor(v_unrelated).unsqueeze(0)

    sim_near = F.cosine_similarity(e_original, e_near).item()
    sim_unrelated = F.cosine_similarity(e_original, e_unrelated).item()

    assert sim_near > sim_unrelated
