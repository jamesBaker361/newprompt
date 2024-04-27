import torch

def prepare_textual_inversion(placeholder:str, tokenizer:object,text_encoder:object,initializer_token:str="thing"):
    placeholder_tokens=[placeholder]
    tokenizer.add_tokens(placeholder_tokens)
    token_ids = tokenizer.encode(initializer_token, add_special_tokens=False)
    initializer_token_id = token_ids[0]
    placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)

    # Resize the token embeddings as we are adding new special tokens to the tokenizer
    text_encoder.resize_token_embeddings(len(tokenizer))

    # Initialise the newly added placeholder token with the embeddings of the initializer token
    token_embeds = text_encoder.get_input_embeddings().weight.data
    with torch.no_grad():
        for token_id in placeholder_token_ids:
            token_embeds[token_id] = token_embeds[initializer_token_id].clone()

    text_encoder.get_input_embeddings().requires_grad_(True)
    return tokenizer,text_encoder,placeholder_token_ids