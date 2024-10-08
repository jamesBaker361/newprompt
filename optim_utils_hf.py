import random
import numpy as np
from PIL import Image
from statistics import mean
import copy
import json
from typing import Any, Mapping
from transformers import AutoTokenizer, T5EncoderModel,CLIPImageProcessor,CLIPVisionModelWithProjection,CLIPModel,CLIPProcessor
from transformers.models.clip.modeling_clip import CLIPTextEmbeddings,CLIPOutput,CLIPTextTransformer
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, ImageClassifierOutput
from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask
from torchvision.transforms import Compose
import torch.nn.functional as F
import os
import torch
from typing import Optional,Union,Tuple
from PIL import Image

import torch

from sentence_transformers.util import (semantic_search, 
                                        dot_score, 
                                        normalize_embeddings)

from transformers import AutoTokenizer, T5EncoderModel
from huggingface_hub import hf_hub_download,snapshot_download

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def nn_project(curr_embeds, embedding_layer, print_hits=False):
    with torch.no_grad():
        bsz,seq_len,emb_dim = curr_embeds.shape
        
        # Using the sentence transformers semantic search which is 
        # a dot product exact kNN search between a set of 
        # query vectors and a corpus of vectors
        curr_embeds = curr_embeds.reshape((-1,emb_dim))
        curr_embeds = normalize_embeddings(curr_embeds) # queries

        embedding_matrix = embedding_layer.weight
        embedding_matrix = normalize_embeddings(embedding_matrix)
        
        hits = semantic_search(curr_embeds, embedding_matrix, 
                                query_chunk_size=curr_embeds.shape[0], 
                                top_k=1,
                                score_function=dot_score)

        if print_hits:
            all_hits = []
            for hit in hits:
                all_hits.append(hit[0]["score"])
            print(f"mean hits:{mean(all_hits)}")
        
        nn_indices = torch.tensor([hit[0]["corpus_id"] for hit in hits], device=curr_embeds.device)
        nn_indices = nn_indices.reshape((bsz,seq_len))

        projected_embeds = embedding_layer(nn_indices)

    return projected_embeds, nn_indices


def forward_hidden(
        self:CLIPTextTransformer,
        hidden_states: Optional[torch.Tensor] = None,
        input_ids:Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        

        input_shape = input_ids.size()

        #hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)

        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = _create_4d_causal_attention_mask(
            input_shape, hidden_states.dtype, device=hidden_states.device
        )

        # expand attention_mask
        if attention_mask is not None and not self._use_flash_attention_2:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        if self.eos_token_id == 2:
            # The `eos_token_id` was incorrect before PR #24773: Let's keep what have been done here.
            # A CLIP model with such `eos_token_id` in the config can't work correctly with extra new tokens added
            # ------------------------------------------------------------
            # text_embeds.shape = [batch_size, sequence_length, transformer.width]
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
            pooled_output = last_hidden_state[
                torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
            ]
        else:
            # The config gets updated `eos_token_id` from PR #24773 (so the use of exta new tokens is possible)
            pooled_output = last_hidden_state[
                torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                # We need to get the first position of `eos_token_id` value (`pad_token_ids` might equal to `eos_token_id`)
                # Note: we assume each sequence (along batch dim.) contains an  `eos_token_id` (e.g. prepared by the tokenizer)
                (input_ids.to(dtype=torch.int, device=last_hidden_state.device) == self.eos_token_id)
                .int()
                .argmax(dim=-1),
            ]

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )



def optimize_prompt(args:dict,
                    clip_processor:CLIPProcessor,
                       clip_model:CLIPModel,
                       tokenizer:AutoTokenizer,
                       image:Image.Image,
                       device:str)->str:
    text_prompt=" ".join([tokenizer.bos_token for _ in range(args.prompt_len)])
    image_input = clip_processor.image_processor(images=image, return_tensors="pt").to(device)

    token_embedding=clip_model.text_model.embeddings.token_embedding.to(device)
    

    

    # Forward pass through the model to get the image embedding (without gradient tracking)
    with torch.no_grad():
        image_features = clip_model.get_image_features(**image_input)

    # Normalize the image embedding
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    # Tokenize the text prompt
    text_input = clip_processor.tokenizer(text=text_prompt, return_tensors="pt", padding=True).to(device)
    print(text_input)

    # Enable gradient tracking on the tokenized text input embeddings
    text_outputs = clip_model.text_model(
            **text_input
        )

    text_states=text_outputs[0]
    optimized_text_states=text_states.clone().detach().requires_grad_(True)
    
    #optimized_text_embeds = text_embeddings.clone().detach().requires_grad_(True)
    # Define optimizer for the text embeddings
    optimizer = torch.optim.Adam([optimized_text_states], lr=0.01)
    
    # Optimization loop to update text embeddings
    for step in range(args.iter):  # Run for 100 iterations
        optimizer.zero_grad()
        text_outputs=forward_hidden(clip_model.text_model,optimized_text_states,text_input["input_ids"])
        text_embeddings=text_outputs[1]

        text_embeddings = clip_model.text_projection(text_embeddings)
        # Normalize the text embedding
        normalized_text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

        # Compute cosine similarity and loss (minimizing distance)
        cosine_similarity = torch.cosine_similarity(image_features, normalized_text_embeddings)
        loss = 1 - cosine_similarity.mean()  # Minimize the distance

        # Backpropagation to compute gradients
        loss.backward()

        # Update the text embeddings
        optimizer.step()

        if step % 10 == 0 or step==args.iter-1:

            # After optimization, convert the optimized text embeddings back to tokens
            projected_embeds, nn_indices = nn_project(optimized_text_states, token_embedding, print_hits=True)

            # Use Hugging Face's tokenizer to decode tokens back to text (approximation)
            text_prompt = clip_processor.tokenizer.decode(nn_indices[0], skip_special_tokens=True)
            print(f"Step {step} - Loss: {loss.item()} prompt: {text_prompt}")
    return text_prompt

    

def set_random_seed(seed=0):
    torch.manual_seed(seed + 0)
    torch.cuda.manual_seed(seed + 1)
    torch.cuda.manual_seed_all(seed + 2)
    np.random.seed(seed + 3)
    torch.cuda.manual_seed_all(seed + 4)
    random.seed(seed + 5)
