import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Optional, Tuple
from transformers.modeling_outputs import CausalLMOutputWithPast

from transformers import Qwen2ForCausalLM
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
from transformers.generation.utils import *
from transformers.generation.utils import _split_model_inputs
import ipdb
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)                       
sys.path.insert(0, parent_dir)
from pretrain_verifier.layers import LatentModelwithVerifier_RATT

class VanillaRoPE(nn.Module):
    def __init__(self, hidden_size, device=None):
        super(VanillaRoPE, self).__init__()
        base = 1000
        dim = hidden_size
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

class SelfAttentionLayer(torch.nn.Module):
    def __init__(self, hidden_size, end_k = -1):
        super().__init__()
        self.hidden_size = hidden_size
        # need the end_k states to calculate
        # if set to -1, all of hidden_states needed
        self.end_k = end_k

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.scale_score = 1.0/(self.hidden_size ** 0.5)
        self.rope = VanillaRoPE(hidden_size)

    @staticmethod
    def mask_to_weights(attention_mask, thought_id_idx, end_k=-1):

        if thought_id_idx is not None and attention_mask.size(0) != thought_id_idx.size(0): 
            raise ValueError("attention_mask must have the same size as thought_id_idx")

        if end_k != -1:
            
            for i in range(attention_mask.size(0)):
                idx = thought_id_idx[i].item()
                start_idx = max(0, idx - end_k)
                attention_mask[i].zero_()
                attention_mask[i, start_idx:idx] = 1
        else:
            for i in range(attention_mask.size(0)):
                idx = thought_id_idx[i].item()
                attention_mask[i, idx:] = 0
           
        # Convert mask to weights: 1 -> 0, 0 -> -inf
        attention_weight = torch.zeros_like(attention_mask, dtype=torch.float32)
        attention_weight = attention_weight.masked_fill(~attention_mask.bool(), float('-inf'))
        return attention_weight

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor,
                thought_id_idx) -> torch.Tensor:
        attention_mask = SelfAttentionLayer.mask_to_weights(attention_mask.clone(), thought_id_idx, self.end_k) # (bs, seq_len)

        cos, sin = self.rope(
            hidden_states,
            position_ids=torch.arange(0, hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)
        )

        batch_size, seq_len, _ = hidden_states.shape

        Q = self.query(hidden_states)  # [batch_size, seq_len, embed_dim]
        K = self.key(hidden_states)
        V = self.value(hidden_states)

        Q, K = apply_rotary_pos_emb(Q, K, cos.squeeze(0), sin.squeeze(0), unsqueeze_dim=0)

        if thought_id_idx is None:
            indices = torch.full((batch_size,), seq_len - 1, device=hidden_states.device)
        else:
            indices = thought_id_idx - 1
            if (indices < 0).any() or (indices >= seq_len).any():
                raise ValueError("thought_id_idx-1 exceeds valid sequence indices")

        Q_selected = Q[torch.arange(batch_size), indices].unsqueeze(1)

        attn_scores = torch.matmul(Q_selected, K.transpose(-2, -1)) * self.scale_score

        attn_scores += attention_mask.unsqueeze(1)  # broadcast to [batch_size, 1, seq_len]

        attn_weights = F.softmax(attn_scores, dim=-1)
        output_selected = torch.matmul(attn_weights, V)  # [batch_size, 1, embed_dim]
        output = output_selected.squeeze(1) # [batch_size, embed_dim]
        return output


class Noise(nn.Module):
    def __init__(self, hidden_size):
        super(Noise, self).__init__()
        self.mu = nn.Linear(hidden_size, hidden_size, bias=False)
        self.log_var = torch.nn.Parameter(torch.tensor([float('-10')] * hidden_size))

    def forward(self, x):
        x = self.mu(x)
        std = torch.exp(self.log_var / 2)
        return x + torch.randn_like(x).mul(std)


class LatentModel_MS(Qwen2ForCausalLM):

    def __init__(self, config):
        super().__init__(config)
        # self.mlp = Noise(config.hidden_size)
        self.attention = SelfAttentionLayer(config.hidden_size)
        self.n_thought = config.n_thought

    def generate_embs(self, input_ids, input_embeds, attention_mask):
        
        if input_ids is not None and input_embeds is None:
            output_mid = super().forward(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        elif input_embeds is not None:
            output_mid = super().forward(inputs_embeds=input_embeds, attention_mask=attention_mask, output_hidden_states=True)
        
        # generate embeddings
        thought_ids = self.model.embed_tokens.num_embeddings - 1  # thought id flag
        where_thought_ids = torch.nonzero(input_ids == thought_ids)
        # print("*** where thought ids", where_thought_ids)
        hidden_states = output_mid['hidden_states']
        input_embs = self.model.embed_tokens(input_ids) if input_embeds is None else input_embeds

        input_embs[where_thought_ids[:, 0], where_thought_ids[:, 1]] = self.attention(
            hidden_states=hidden_states[-1], attention_mask=attention_mask,
            thought_id_idx=where_thought_ids[:, 1],
        ).to(input_embs.dtype)

        return input_embs, where_thought_ids

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        # single-step reasoning: take the last latent hidden state before the thought token to replace the thought token's embedding, then take the final hidden state
        # multi-step reasoning (training): after each replacement, increment thought_idx by 1, then append a new padding column, then update attention_mask (set the old thought_idx position to 1, leave the rest unchanged). Update input_embeds by shifting the embeddings from the old thought_idx onward to the new thought_idx position, and update labels accordingly
        # multi-step reasoning (inference): keep the kv cache, only use the latest input_ids; attention_mask and cache_position need to be adjusted externally when updated
        
        device = input_ids.device

        if input_ids is not None and inputs_embeds is None and input_ids.size() == attention_mask.size(): # during inference, first time call forward
            # move inside forward(), one-time creation
            thought_token_id = torch.tensor([self.model.embed_tokens.num_embeddings - 1], device=input_ids.device, dtype=input_ids.dtype)
            thought_token_embed = self.model.embed_tokens(thought_token_id.to(self.model.embed_tokens.weight.device)) #.to(inputs_embeds.dtype) #.detach()

            for thought_idx in range(self.n_thought):
                input_ids_for_mark = input_ids.clone()

                if input_ids is not None and inputs_embeds is None and input_ids.size() == attention_mask.size():
                    inputs_embeds, where_thought_ids = self.generate_embs(input_ids, None, attention_mask)
                    # input_ids = None
                else:
                    inputs_embeds, where_thought_ids = self.generate_embs(input_ids_for_mark, inputs_embeds, attention_mask)

                # update thought idx and attention mask
                next_prefix_embeds = torch.zeros(inputs_embeds.shape[0], inputs_embeds.shape[1]+1, inputs_embeds.shape[2]).to(device).type_as(inputs_embeds) # (bs, seq_len+1, dim)
                next_prefix_ids = torch.zeros(input_ids.shape[0], input_ids.shape[1]+1).to(device).type_as(input_ids) # next_prefix_ids add a new column
            

                where_thought_ids = where_thought_ids.clone()
                where_thought_ids[:,1] = where_thought_ids[:,1] + 1
                # where_thought_ids[:,1] += 1 # update thought ids idx

                # next_labels = torch.cat([labels, labels[:,-1].unsqueeze(1)], dim=1)
                next_labels = torch.ones(labels.shape[0], labels.shape[1]+1).type_as(labels) if labels is not None else None
                next_attention_mask = torch.ones(attention_mask.shape[0], attention_mask.shape[1]+1).type_as(attention_mask)
                
                # update input_embeds, input_ids, labels, attention_mask
                for idx, thought_id_idx in enumerate(where_thought_ids[:,1]):
                    next_prefix_embeds[idx] = torch.cat([inputs_embeds[idx, :thought_id_idx, :], thought_token_embed, inputs_embeds[idx, thought_id_idx:, :]], dim=0) # update input_embeds
                    next_prefix_ids[idx] = torch.cat([input_ids[idx, :thought_id_idx], thought_token_id, input_ids[idx, thought_id_idx:]], dim=0)  # update input_ids
                    next_prefix_ids[idx, thought_id_idx-1] = 0
                    next_attention_mask[idx] = torch.cat([attention_mask[idx, :thought_id_idx], torch.LongTensor([1]).type_as(attention_mask), attention_mask[idx, thought_id_idx:]], dim=0) # update attention mask
                    if labels is not None:
                        next_labels[idx] = torch.cat([labels[idx, :thought_id_idx], torch.LongTensor([-100]).type_as(labels), labels[idx, thought_id_idx:]])             
                        next_labels[idx, thought_id_idx] = -100 # update labels

                input_ids = next_prefix_ids.clone()
                inputs_embeds = next_prefix_embeds.clone()
                labels = next_labels.clone() if labels is not None else None
                attention_mask = next_attention_mask.clone()

                if position_ids is not None:
                    next_pos = position_ids[:, -1:] + 1  
                    position_ids = torch.cat([position_ids, next_pos], dim=1)
                if cache_position is not None:
                    next_cache = cache_position[-1:] + 1  
                    cache_position = torch.cat([cache_position, next_cache], dim=0)

            input_ids = None # clear after reasoning for item title generation

        # during inference, kv cache is on
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask, # (bs, seq_len+n_thought)
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, # (bs, seq_len+n_thought, dim)
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            num_logits_to_keep=num_logits_to_keep,
            **kwargs,
        )
    
    def _beam_search(
        self,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        **model_kwargs,
    ) -> Union[GenerateBeamOutput, torch.LongTensor]:
        # init values
        pad_token_id = generation_config._pad_token_tensor
        eos_token_id = generation_config._eos_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        sequential = generation_config.low_memory
        do_sample = generation_config.do_sample

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        beam_indices = (
            tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
        )
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
        # of the first beam are considered to avoid sampling the exact same tokens across all beams.
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        this_peer_finished = False

        decoder_prompt_len = input_ids.shape[-1]  # record the prompt length of decoder
        reasoning_flag = True
        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            
            # prepare variable output controls (note: some models won't accept all output controls)
            model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
            model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

            # if sequential is True, split the input to batches of batch_size and run sequentially
            if sequential:
                if any(
                    model_name in self.__class__.__name__.lower()
                    for model_name in [
                        "fsmt",
                        "reformer",
                        "ctrl",
                        "gpt_bigcode",
                        "transo_xl",
                        "xlnet",
                        "cpm",
                        "jamba",
                    ]
                ):
                    raise RuntimeError(
                        f"Currently generation for {self.__class__.__name__} is not supported "
                        f"for `low_memory beam_search`. Please open an issue on GitHub if you need this feature."
                    )

                inputs_per_sub_batches = _split_model_inputs(
                    model_inputs,
                    split_size=batch_size,
                    full_batch_size=batch_beam_size,
                    config=self.config.get_text_config(),
                )
                outputs_per_sub_batch = [
                    self(**inputs_per_sub_batch, return_dict=True) for inputs_per_sub_batch in inputs_per_sub_batches
                ]

                outputs = stack_model_outputs(outputs_per_sub_batch, self.config.get_text_config())

            else:  # Unchanged original behavior
                outputs = self(**model_inputs, return_dict=True)

            # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue

            # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            # .float() is needed to retain precision for later logits manipulations
            next_token_logits = outputs.logits[:, -1, :].clone().float()
            next_token_logits = next_token_logits.to(input_ids.device)
            next_token_scores = nn.functional.log_softmax(
                next_token_logits, dim=-1
            )  # (batch_size * num_beams, vocab_size)

            next_token_scores_processed = logits_processor(input_ids, next_token_scores)
            next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(
                next_token_scores_processed
            )

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores_processed,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)
                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            # Beam token selection: pick 1 + eos_token_id.shape[0] next tokens for each beam so we have at least 1
            # non eos token per beam.
            n_eos_tokens = eos_token_id.shape[0] if eos_token_id is not None else 0
            n_tokens_to_keep = max(2, 1 + n_eos_tokens) * num_beams
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=n_tokens_to_keep)
                next_token_scores = torch.gather(next_token_scores, -1, next_tokens)
                next_token_scores, _indices = torch.sort(next_token_scores, descending=True, dim=1)
                next_tokens = torch.gather(next_tokens, -1, _indices)
            else:
                next_token_scores, next_tokens = torch.topk(
                    next_token_scores, n_tokens_to_keep, dim=1, largest=True, sorted=True
                )

            next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                beam_indices=beam_indices,
                decoder_prompt_len=decoder_prompt_len,
            )

            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            # IMPORTANT: Note that this should appear BEFORE the call to _reorder_cache() to save the maximum memory
            # (that way the memory peak does not include outputs.logits)
            del outputs

            if model_kwargs.get("past_key_values", None) is not None:
                model_kwargs["past_key_values"] = self._temporary_reorder_cache(
                    model_kwargs["past_key_values"], beam_idx
                )
            if reasoning_flag:
                model_kwargs["attention_mask"] = torch.cat(
                    [model_kwargs["attention_mask"], model_kwargs["attention_mask"].new_ones((model_kwargs["attention_mask"].shape[0], self.n_thought))], dim=-1
                )
                if model_kwargs.get("use_cache", True):
                    model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + self.n_thought
                reasoning_flag = False

            if return_dict_in_generate and output_scores:
                beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))

            # increase cur_len
            cur_len = cur_len + 1

            if beam_scorer.is_done or all(stopping_criteria(input_ids, scores)):
                this_peer_finished = True

        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
            beam_indices=beam_indices,
            decoder_prompt_len=decoder_prompt_len,
        )

        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None

            if self.config.is_encoder_decoder:
                return GenerateBeamEncoderDecoderOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    logits=raw_logits,
                    beam_indices=sequence_outputs["beam_indices"],
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateBeamDecoderOnlyOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    logits=raw_logits,
                    beam_indices=sequence_outputs["beam_indices"],
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return sequence_outputs["sequences"]

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> Dict[str, Any]:
        # update past_key_values keeping its naming used in model code
        for possible_cache_name in ALL_CACHE_NAMES:
            if possible_cache_name in outputs:
                # (joao): remove output/input mismatch when these old models (xlnet, reformer) are deprecated
                if possible_cache_name in ("past_buckets_states", "mems"):
                    cache_name = "past_key_values"
                else:
                    cache_name = possible_cache_name
                model_kwargs[cache_name] = getattr(outputs, possible_cache_name)
                break

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        if not is_encoder_decoder:
            # update attention mask
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )
        else:
            # update decoder attention mask
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                model_kwargs["decoder_attention_mask"] = torch.cat(
                    [decoder_attention_mask, decoder_attention_mask.new_ones((decoder_attention_mask.shape[0], 1))],
                    dim=-1,
                )

        if model_kwargs.get("use_cache", True):
            model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + num_new_tokens
        else:
            past_positions = model_kwargs.pop("cache_position")
            new_positions = torch.arange(
                past_positions[-1] + 1, past_positions[-1] + num_new_tokens + 1, dtype=past_positions.dtype
            ).to(past_positions.device)
            model_kwargs["cache_position"] = torch.cat((past_positions, new_positions))
        return model_kwargs
    

class LatentModelwithVerifier_RATT_MS(LatentModelwithVerifier_RATT):
    '''
        This class contain the RATT, and is the finetune version (i.e., support multi reasoning thoughts)
        workflow: reasoning then generate
        training: reasoning loss is verifier loss, generation loss is MLE, and a monoticity regularization
    '''
    def __init__(self, config):
        super().__init__(config)
        self.verifiers = nn.ModuleList([nn.Linear(config.hidden_size, config.num_category) for _ in range(config.num_verifier)])
        self.num_verifier = config.num_verifier
        self.num_category = config.num_category 
        self.v_weight = config.v_weight
        self.mono_weight = config.mono_weight
       
    def open_llm_params(self):
        # open llm and RATT params
        for param in self.model.parameters():
            param.requires_grad = True
        for param in self.attention.parameters():
            param.requires_grad = True
        # open verifier
        for param in self.verifiers.parameters():
            param.requires_grad = True
        
    def freeze_verifier(self):
        # freeze verifier
        for param in self.verifiers.parameters():
            param.requires_grad = False

    def verify(self, thought_embeds, strategy="max"):
        '''
            input: thought embedding
            output: weight, guidance emb
        '''
        v_logits = []
        for v_idx, v in enumerate(self.verifiers):
            logit = v(thought_embeds)
            v_logits.append(logit.unsqueeze(0))
        v_logits = torch.cat(v_logits, dim=0) # (n_verifier, bs, n_cat)

        v_probs = F.softmax(v_logits, dim=-1) # (n_verifier, bs, n_cat)
        # evaluation feedback f_i via entropy (Eq. 8)
        entropy = -torch.sum(v_probs * torch.log(v_probs + 1e-12), dim=-1)  # (n_verifier, bs)
        max_entropy = torch.log(torch.tensor(v_probs.size(-1), dtype=torch.float, device=v_probs.device))
        weight = torch.mean(entropy / max_entropy, dim=0)  # (bs,)
        if strategy == "max": 
            # verifier average
            v_probs_mean = v_probs.mean(dim=0)  # (bs, n_cat)
            max_prob, max_prob_idx = v_probs_mean.max(dim=-1)  # (bs,)
            # guidance vector g_i = verifier's last-layer weight prototype (Eq. 9)
            weight_matrix = self.verifiers[0].weight  # (n_cat, dim)
            guidance_emb = weight_matrix[max_prob_idx]  # (bs, dim)
        
        elif strategy == "mean":
            guidance_emb = torch.mean(self.verifier.weight, dim=1)
        return weight, guidance_emb, entropy

    def generate_embs(self, input_ids, input_embeds, attention_mask):
        
        if input_ids is not None and input_embeds is None:
            output_mid = Qwen2ForCausalLM.forward(self, input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        elif input_embeds is not None:
            output_mid = Qwen2ForCausalLM.forward(self, inputs_embeds=input_embeds, attention_mask=attention_mask, output_hidden_states=True)
        
        # generate embeddings
        thought_ids = self.model.embed_tokens.num_embeddings - 1  # thought id flag
        where_thought_ids = torch.nonzero(input_ids == thought_ids)

        input_embs = self.model.embed_tokens(input_ids) if input_embeds is None else input_embeds
        # ipdb.set_trace()
        hidden_states = output_mid['hidden_states'][-1] # (bs, seq_len, dim)
        thought_embeds = self.attention(
            hidden_states=hidden_states, attention_mask=attention_mask,
            thought_id_idx=where_thought_ids[:, 1],
        ).to(input_embs.dtype) # (bs, dim)
        
        weight, guidance_emb, entropy = self.verify(thought_embeds, strategy="max")
        # confidence-guided adjustment r* (Eq. 10)
        input_embs[where_thought_ids[:, 0], where_thought_ids[:, 1]] = (1-weight.unsqueeze(1).to(dtype=input_embs.dtype)) * thought_embeds + weight.unsqueeze(1).to(dtype=input_embs.dtype) * guidance_emb
        return input_embs, where_thought_ids, thought_embeds, guidance_emb, entropy
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        device = input_ids.device
        v_loss = 0
        H_last = None
        loss_mono = 0
        all_thought_embeds = []
        if input_ids is not None and inputs_embeds is None and input_ids.size() == attention_mask.size(): # during inference, first time call forward
            # move inside forward(), one-time creation
            thought_token_id = torch.tensor([self.model.embed_tokens.num_embeddings - 1], device=input_ids.device, dtype=input_ids.dtype)
            thought_token_embed = self.model.embed_tokens(thought_token_id.to(self.model.embed_tokens.weight.device)) #.to(inputs_embeds.dtype) #.detach()

            for thought_idx in range(self.n_thought):
                input_ids_for_mark = input_ids.clone()

                if input_ids is not None and inputs_embeds is None and input_ids.size() == attention_mask.size():
                    inputs_embeds, where_thought_ids, thought_embeds, guidance_embeds, H = self.generate_embs(input_ids, None, attention_mask)
                else:
                    inputs_embeds, where_thought_ids, thought_embeds, guidance_embeds, H = self.generate_embs(input_ids_for_mark, inputs_embeds, attention_mask)

                # compute the entropy difference
                if H_last is None:
                    H_last = H
                else:
                    difference = H - H_last
                    H_last = H
                    # monotonicity regularization (Eq. 13)
                    loss_mono += torch.exp(torch.mean(torch.clamp_min(difference.squeeze(0), 0.0)))

                # update thought idx and attention mask
                next_prefix_embeds = torch.zeros(inputs_embeds.shape[0], inputs_embeds.shape[1]+1, inputs_embeds.shape[2]).to(device).type_as(inputs_embeds) # (bs, seq_len+1, dim)
                next_prefix_ids = torch.zeros(input_ids.shape[0], input_ids.shape[1]+1).to(device).type_as(input_ids) # next_prefix_ids add a new column

                where_thought_ids = where_thought_ids.clone()
                where_thought_ids[:,1] = where_thought_ids[:,1] + 1

                next_labels = torch.ones(labels.shape[0], labels.shape[1]+1).type_as(labels) if labels is not None else None
                next_attention_mask = torch.ones(attention_mask.shape[0], attention_mask.shape[1]+1).type_as(attention_mask)

                # update input_embeds, input_ids, labels, attention_mask
                for idx, thought_id_idx in enumerate(where_thought_ids[:,1]):
                    next_prefix_embeds[idx] = torch.cat([inputs_embeds[idx, :thought_id_idx, :], thought_token_embed, inputs_embeds[idx, thought_id_idx:, :]], dim=0) # update input_embeds
                    next_prefix_ids[idx] = torch.cat([input_ids[idx, :thought_id_idx], thought_token_id, input_ids[idx, thought_id_idx:]], dim=0)  # update input_ids
                    next_prefix_ids[idx, thought_id_idx-1] = 0
                    next_attention_mask[idx] = torch.cat([attention_mask[idx, :thought_id_idx], torch.LongTensor([1]).type_as(attention_mask), attention_mask[idx, thought_id_idx:]], dim=0) # update attention mask
                    if labels is not None:
                        next_labels[idx] = torch.cat([labels[idx, :thought_id_idx], torch.LongTensor([-100]).type_as(labels), labels[idx, thought_id_idx:]])
                        next_labels[idx, thought_id_idx] = -100 # update labels

                input_ids = next_prefix_ids.clone()
                inputs_embeds = next_prefix_embeds.clone()
                labels = next_labels.clone() if labels is not None else None
                attention_mask = next_attention_mask.clone()

                if position_ids is not None:
                    next_pos = position_ids[:, -1:] + 1
                    position_ids = torch.cat([position_ids, next_pos], dim=1)
                if cache_position is not None:
                    next_cache = cache_position[-1:] + 1
                    cache_position = torch.cat([cache_position, next_cache], dim=0)
                all_thought_embeds.append(thought_embeds.unsqueeze(0)) # collect all thoughts embeds for classification loss
            input_ids = None # clear after reasoning for item title generation

            # verifier loss (Eq. 11)
            all_thought_embeds = torch.cat(all_thought_embeds, dim=0) # (n_thoughts, bs, dim)
            output_logits = []
            for v_idx in range(self.num_verifier):
                output_logits.append(self.verifiers[v_idx](all_thought_embeds).unsqueeze(2)) # (n_thoughts, bs, 1, n_cat)

            output_logits = torch.cat(output_logits, dim=2) # (n_thoughts, bs, n_verifier, n_cat)
            output_logits = output_logits.view(self.n_thought, -1, self.num_category) # (n_thoughts, bs*n_verifier, n_cat)

            v_loss = 0
            if "category_label" in kwargs:
                # ipdb.set_trace()
                v_labels = kwargs["category_label"] # (bs)
                # print("v_labels shape", v_labels.shape)
                # print("n verifier", self.num_verifier)
                # print("n_thought", self.n_thought)
                v_labels = v_labels.squeeze(1)
                v_labels = v_labels.repeat(self.num_verifier).unsqueeze(0).repeat(self.n_thought, 1)  # (n_thoughts, bs*n_verifier, )
                loss_func = nn.CrossEntropyLoss()
                output_logits = output_logits.view(-1, self.num_category) # (n_thoughts * bs * n_verifier, n_category)
                v_labels = v_labels.view(-1) # (n_thoughts * bs * n_verifier)
                v_loss = loss_func(output_logits, v_labels)

            # overall loss = rec loss + beta * verifier loss + gamma * monotonicity reg (Eq. 14)
            v_loss += self.mono_weight * loss_mono

        # during inference, kv cache is on
        return self.qwen_forward(
            input_ids=input_ids,
            attention_mask=attention_mask, # (bs, seq_len+n_thought)
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, # (bs, seq_len+n_thought, dim)
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            num_logits_to_keep=num_logits_to_keep,
            loss=v_loss,
            **kwargs,
        )

    def qwen_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        loss: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        if loss is not None:
            v_loss = loss

        loss = None
        if labels is not None:
            # recommendation loss (Eq. 12)
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
            # overall loss: rec loss + verifier loss (incl. monotonicity reg) (Eq. 14)
            loss = loss + self.v_weight * v_loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    def _beam_search(
        self,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        **model_kwargs,
    ) -> Union[GenerateBeamOutput, torch.LongTensor]:
        # init values
        pad_token_id = generation_config._pad_token_tensor
        eos_token_id = generation_config._eos_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        sequential = generation_config.low_memory
        do_sample = generation_config.do_sample

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        beam_indices = (
            tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
        )
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
        # of the first beam are considered to avoid sampling the exact same tokens across all beams.
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        this_peer_finished = False

        decoder_prompt_len = input_ids.shape[-1]  # record the prompt length of decoder
        reasoning_flag = True
        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            
            # prepare variable output controls (note: some models won't accept all output controls)
            model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
            model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

            # if sequential is True, split the input to batches of batch_size and run sequentially
            if sequential:
                if any(
                    model_name in self.__class__.__name__.lower()
                    for model_name in [
                        "fsmt",
                        "reformer",
                        "ctrl",
                        "gpt_bigcode",
                        "transo_xl",
                        "xlnet",
                        "cpm",
                        "jamba",
                    ]
                ):
                    raise RuntimeError(
                        f"Currently generation for {self.__class__.__name__} is not supported "
                        f"for `low_memory beam_search`. Please open an issue on GitHub if you need this feature."
                    )

                inputs_per_sub_batches = _split_model_inputs(
                    model_inputs,
                    split_size=batch_size,
                    full_batch_size=batch_beam_size,
                    config=self.config.get_text_config(),
                )
                outputs_per_sub_batch = [
                    self(**inputs_per_sub_batch, return_dict=True) for inputs_per_sub_batch in inputs_per_sub_batches
                ]

                outputs = stack_model_outputs(outputs_per_sub_batch, self.config.get_text_config())

            else:  # Unchanged original behavior
                outputs = self(**model_inputs, return_dict=True)

            # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue

            # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            # .float() is needed to retain precision for later logits manipulations
            next_token_logits = outputs.logits[:, -1, :].clone().float()
            next_token_logits = next_token_logits.to(input_ids.device)
            next_token_scores = nn.functional.log_softmax(
                next_token_logits, dim=-1
            )  # (batch_size * num_beams, vocab_size)

            next_token_scores_processed = logits_processor(input_ids, next_token_scores)
            next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(
                next_token_scores_processed
            )

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores_processed,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)
                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            # Beam token selection: pick 1 + eos_token_id.shape[0] next tokens for each beam so we have at least 1
            # non eos token per beam.
            n_eos_tokens = eos_token_id.shape[0] if eos_token_id is not None else 0
            n_tokens_to_keep = max(2, 1 + n_eos_tokens) * num_beams
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=n_tokens_to_keep)
                next_token_scores = torch.gather(next_token_scores, -1, next_tokens)
                next_token_scores, _indices = torch.sort(next_token_scores, descending=True, dim=1)
                next_tokens = torch.gather(next_tokens, -1, _indices)
            else:
                next_token_scores, next_tokens = torch.topk(
                    next_token_scores, n_tokens_to_keep, dim=1, largest=True, sorted=True
                )

            next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                beam_indices=beam_indices,
                decoder_prompt_len=decoder_prompt_len,
            )

            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            # IMPORTANT: Note that this should appear BEFORE the call to _reorder_cache() to save the maximum memory
            # (that way the memory peak does not include outputs.logits)
            del outputs

            if model_kwargs.get("past_key_values", None) is not None:
                model_kwargs["past_key_values"] = self._temporary_reorder_cache(
                    model_kwargs["past_key_values"], beam_idx
                )
            if reasoning_flag:
                model_kwargs["attention_mask"] = torch.cat(
                    [model_kwargs["attention_mask"], model_kwargs["attention_mask"].new_ones((model_kwargs["attention_mask"].shape[0], self.n_thought))], dim=-1
                )
                if model_kwargs.get("use_cache", True):
                    model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + self.n_thought
                reasoning_flag = False

            if return_dict_in_generate and output_scores:
                beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))

            # increase cur_len
            cur_len = cur_len + 1

            if beam_scorer.is_done or all(stopping_criteria(input_ids, scores)):
                this_peer_finished = True

        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
            beam_indices=beam_indices,
            decoder_prompt_len=decoder_prompt_len,
        )

        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None

            if self.config.is_encoder_decoder:
                return GenerateBeamEncoderDecoderOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    logits=raw_logits,
                    beam_indices=sequence_outputs["beam_indices"],
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateBeamDecoderOnlyOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    logits=raw_logits,
                    beam_indices=sequence_outputs["beam_indices"],
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return sequence_outputs["sequences"]

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        r"""

        Generates sequences of token ids for models with a language modeling head.

        <Tip warning={true}>

        Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
        model's default generation configuration. You can override any `generation_config` by passing the corresponding
        parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`.

        For an overview of generation strategies and code examples, check out the [following
        guide](../generation_strategies).

        </Tip>

        Parameters:
            inputs (`torch.Tensor` of varying shape depending on the modality, *optional*):
                The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the
                method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs`
                should be in the format of `input_ids`. For encoder-decoder models *inputs* can represent any of
                `input_ids`, `input_values`, `input_features`, or `pixel_values`.
            generation_config ([`~generation.GenerationConfig`], *optional*):
                The generation configuration to be used as base parametrization for the generation call. `**kwargs`
                passed to generate matching the attributes of `generation_config` will override them. If
                `generation_config` is not provided, the default will be used, which has the following loading
                priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
                configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
                default values, whose documentation should be checked to parameterize generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                Custom logits processors that complement the default logits processors built from arguments and
                generation config. If a logit processor is passed that is already created with the arguments or a
                generation config an error is thrown. This feature is intended for advanced users.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                Custom stopping criteria that complements the default stopping criteria built from arguments and a
                generation config. If a stopping criteria is passed that is already created with the arguments or a
                generation config an error is thrown. If your stopping criteria depends on the `scores` input, make
                sure you pass `return_dict_in_generate=True, output_scores=True` to `generate`. This feature is
                intended for advanced users.
            prefix_allowed_tokens_fn (`Callable[[int, torch.Tensor], List[int]]`, *optional*):
                If provided, this function constraints the beam search to allowed tokens only at each step. If not
                provided no constraint is applied. This function takes 2 arguments: the batch ID `batch_id` and
                `input_ids`. It has to return a list with the allowed tokens for the next generation step conditioned
                on the batch ID `batch_id` and the previously generated tokens `inputs_ids`. This argument is useful
                for constrained generation conditioned on the prefix, as described in [Autoregressive Entity
                Retrieval](https://arxiv.org/abs/2010.00904).
            synced_gpus (`bool`, *optional*):
                Whether to continue running the while loop until max_length. Unless overridden, this flag will be set
                to `True` if using `FullyShardedDataParallel` or DeepSpeed ZeRO Stage 3 with multiple GPUs to avoid
                deadlocking if one GPU finishes generating before other GPUs. Otherwise, defaults to `False`.
            assistant_model (`PreTrainedModel`, *optional*):
                An assistant model that can be used to accelerate generation. The assistant model must have the exact
                same tokenizer. The acceleration is achieved when forecasting candidate tokens with the assistant model
                is much faster than running generation with the model you're calling generate from. As such, the
                assistant model should be much smaller.
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            negative_prompt_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                The negative prompt needed for some processors such as CFG. The batch size must match the input batch
                size. This is an experimental feature, subject to breaking API changes in future versions.
            negative_prompt_attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Attention_mask for `negative_prompt_ids`.
            kwargs (`Dict[str, Any]`, *optional*):
                Ad hoc parametrization of `generation_config` and/or additional model-specific kwargs that will be
                forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder
                specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.

        Return:
            [`~utils.ModelOutput`] or `torch.LongTensor`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`
            or when `config.return_dict_in_generate=True`) or a `torch.LongTensor`.

                If the model is *not* an encoder-decoder model (`model.config.is_encoder_decoder=False`), the possible
                [`~utils.ModelOutput`] types are:

                    - [`~generation.GenerateDecoderOnlyOutput`],
                    - [`~generation.GenerateBeamDecoderOnlyOutput`]

                If the model is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible
                [`~utils.ModelOutput`] types are:

                    - [`~generation.GenerateEncoderDecoderOutput`],
                    - [`~generation.GenerateBeamEncoderDecoderOutput`]
        """

        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        self._validate_model_class()
        tokenizer = kwargs.pop("tokenizer", None)  # Pull this out first, we only use it for stopping criteria
        assistant_tokenizer = kwargs.pop("assistant_tokenizer", None)  # only used for assisted generation

        generation_config, model_kwargs = self._prepare_generation_config(generation_config, **kwargs)
        self._validate_model_kwargs(model_kwargs.copy())
        self._validate_assistant(assistant_model, tokenizer, assistant_tokenizer)

        # 2. Set generation parameters if not already defined
        if synced_gpus is None:
            synced_gpus = (is_deepspeed_zero3_enabled() or is_fsdp_managed_module(self)) and dist.get_world_size() > 1

        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
        requires_attention_mask = "encoder_outputs" not in model_kwargs
        kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None

        # 3. Define model inputs
        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        batch_size = inputs_tensor.shape[0]

        device = inputs_tensor.device
        self._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=device)

        # decoder-only models must use left-padding for batched generation.
        if not self.config.is_encoder_decoder and not is_torchdynamo_compiling():
            # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
            # Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
            if (
                generation_config._pad_token_tensor is not None
                and batch_size > 1
                and len(inputs_tensor.shape) == 2
                and torch.sum(inputs_tensor[:, -1] == generation_config._pad_token_tensor) > 0
            ):
                logger.warning(
                    "A decoder-only architecture is being used, but right-padding was detected! For correct "
                    "generation results, please set `padding_side='left'` when initializing the tokenizer."
                )

        # 4. Define other model kwargs
        # decoder-only models with inputs_embeds forwarding must use caching (otherwise we can't detect whether we are
        # generating the first new token or not, and we only want to use the embeddings for the first new token)
        if not self.config.is_encoder_decoder and model_input_name == "inputs_embeds":
            generation_config.use_cache = True

        if not kwargs_has_attention_mask and requires_attention_mask and accepts_attention_mask:
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                inputs_tensor, generation_config, model_kwargs
            )
        elif kwargs_has_attention_mask:
            # (joao): generalize this check with other types of inputs
            if model_input_name == "input_ids" and len(model_kwargs["attention_mask"].shape) > 2:
                raise ValueError("`attention_mask` passed to `generate` must be 2D.")

        if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
            # if model is encoder decoder encoder_outputs are created and added to `model_kwargs`
            model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name, generation_config
            )

        # 5. Prepare `input_ids` which will be used for auto-regressive generation
        if self.config.is_encoder_decoder:
            input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
                batch_size=batch_size,
                model_input_name=model_input_name,
                model_kwargs=model_kwargs,
                decoder_start_token_id=generation_config._decoder_start_token_tensor,
                device=inputs_tensor.device,
            )
        else:
            input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

        if generation_config.token_healing:
            input_ids = self.heal_tokens(input_ids, tokenizer)

        if streamer is not None:
            streamer.put(input_ids.cpu())

        # 6. Prepare `max_length` depending on other stopping criteria.
        input_ids_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            has_default_min_length=has_default_min_length,
            model_input_name=model_input_name,
            inputs_tensor=inputs_tensor,
            input_ids_length=input_ids_length,
        )

        # If the model supports `logits_to_keep` in forward(), set it to 1 to avoid computing the whole
        # logit matrix. This can save a lot of memory during the first forward pass. Note that assisted decoding
        # dynamically overrides this value as it can need more than the last token logits
        if self._supports_num_logits_to_keep() and "num_logits_to_keep" not in model_kwargs:
            model_kwargs["num_logits_to_keep"] = 1

        self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

        # 7. Prepare the cache.
        # - `model_kwargs` may be updated in place with a cache as defined by the parameters in `generation_config`.
        # - different models have a different cache name expected by the model (default = "past_key_values")
        # - `max_length`, prepared above, is used to determine the maximum cache length
        max_cache_length = generation_config.max_length - 1
        if (
            inputs_tensor.shape[1] != input_ids_length
            and model_input_name == "inputs_embeds"
            and not self.config.is_encoder_decoder
        ):
            max_cache_length += inputs_tensor.shape[1]
        self._prepare_cache_for_generation(
            generation_config, model_kwargs, assistant_model, batch_size, max_cache_length, device
        )

        # 8. determine generation mode
        generation_mode = generation_config.get_generation_mode(assistant_model)

        if streamer is not None and (generation_config.num_beams > 1):
            raise ValueError(
                "`streamer` cannot be used with beam search (yet!). Make sure that `num_beams` is set to 1."
            )

        if not is_torchdynamo_compiling() and self.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )

        # 9. prepare logits processors and stopping criteria
        prepared_logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_length,
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
            device=inputs_tensor.device,
            model_kwargs=model_kwargs,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
        )
        prepared_stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config, stopping_criteria=stopping_criteria, tokenizer=tokenizer, **kwargs
        )

        # Set model_kwargs `use_cache` so we can use it later in forward runs
        model_kwargs["use_cache"] = generation_config.use_cache

        # 10. go into different generation modes
        if generation_mode == GenerationMode.ASSISTED_GENERATION:
            if generation_config.num_return_sequences > 1:
                raise ValueError(
                    "num_return_sequences has to be 1 when doing assisted generate, "
                    f"but is {generation_config.num_return_sequences}."
                )
            if batch_size > 1:
                raise ValueError("assisted generate is only supported for batch_size = 1")
            if not model_kwargs["use_cache"]:
                raise ValueError("assisted generate requires `use_cache=True`")
            if generation_config.cache_implementation in ["static", "hybrid", "sliding_window"]:
                raise ValueError("assisted generate is not supported with Static cache classes`")
            if self._is_stateful:
                # In assisted generation we need the ability to confirm whether the model would pick certain tokens,
                # which is not possible with stateful models (they can't reset to a previous subset of generated text)
                raise ValueError(
                    f"assisted generation is not supported with stateful models, such as {self.__class__.__name__}"
                )

            # 11. Get the candidate generator, given the parameterization
            candidate_generator = self._get_candidate_generator(
                generation_config=generation_config,
                input_ids=input_ids,
                inputs_tensor=inputs_tensor,
                assistant_model=assistant_model,
                logits_processor=logits_processor,
                target_tokenizer=tokenizer,
                assistant_tokenizer=assistant_tokenizer,
                model_kwargs=model_kwargs,
            )

            # 12. run assisted generate
            result = self._assisted_decoding(
                input_ids,
                candidate_generator=candidate_generator,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )
        elif generation_mode == GenerationMode.DOLA_GENERATION:
            if self._is_stateful:
                # DoLa decoding was not designed for stateful models, and would require some changes
                raise ValueError(
                    f"dola decoding is not supported with stateful models, such as {self.__class__.__name__}"
                )
            result = self._dola_decoding(
                input_ids,
                dola_layers=generation_config.dola_layers,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )

        elif generation_mode == GenerationMode.CONTRASTIVE_SEARCH:
            if not model_kwargs["use_cache"]:
                raise ValueError("Contrastive search requires `use_cache=True`")
            if self._is_stateful:
                # Just like assisted generation, we need to be able to rollback to a previous state (see comment above)
                raise ValueError(
                    f"contrastive search is not supported with stateful models, such as {self.__class__.__name__}"
                )

            result = self._contrastive_search(
                input_ids,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )

        elif generation_mode in (GenerationMode.SAMPLE, GenerationMode.GREEDY_SEARCH):
            # 11. expand input_ids with `num_return_sequences` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_return_sequences,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )

            # 12. run sample (it degenerates to greedy search when `generation_config.do_sample=False`)
            result = self._sample(
                input_ids,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )

        elif generation_mode in (GenerationMode.BEAM_SAMPLE, GenerationMode.BEAM_SEARCH):
            # 11. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=generation_config.num_beams,
                device=inputs_tensor.device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                num_beam_hyps_to_keep=generation_config.num_return_sequences,
                max_length=generation_config.max_length,
            )

            # 12. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )

            # 13. run beam sample
            result = self._beam_search(
                input_ids,
                beam_scorer,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif generation_mode == GenerationMode.GROUP_BEAM_SEARCH:
            # 11. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=generation_config.num_beams,
                device=inputs_tensor.device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                num_beam_hyps_to_keep=generation_config.num_return_sequences,
                num_beam_groups=generation_config.num_beam_groups,
                max_length=generation_config.max_length,
            )
            # 12. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )
            # 13. run beam search
            result = self._group_beam_search(
                input_ids,
                beam_scorer,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif generation_mode == GenerationMode.CONSTRAINED_BEAM_SEARCH:
            final_constraints = []
            if generation_config.constraints is not None:
                final_constraints = generation_config.constraints

            if generation_config.force_words_ids is not None:

                def typeerror():
                    raise ValueError(
                        "`force_words_ids` has to either be a `List[List[List[int]]]` or `List[List[int]]` "
                        f"of positive integers, but is {generation_config.force_words_ids}."
                    )

                if (
                    not isinstance(generation_config.force_words_ids, list)
                    or len(generation_config.force_words_ids) == 0
                ):
                    typeerror()

                for word_ids in generation_config.force_words_ids:
                    if isinstance(word_ids[0], list):
                        if not isinstance(word_ids, list) or len(word_ids) == 0:
                            typeerror()
                        if any(not isinstance(token_ids, list) for token_ids in word_ids):
                            typeerror()
                        if any(
                            any((not isinstance(token_id, int) or token_id < 0) for token_id in token_ids)
                            for token_ids in word_ids
                        ):
                            typeerror()

                        constraint = DisjunctiveConstraint(word_ids)
                    else:
                        if not isinstance(word_ids, list) or len(word_ids) == 0:
                            typeerror()
                        if any((not isinstance(token_id, int) or token_id < 0) for token_id in word_ids):
                            typeerror()

                        constraint = PhrasalConstraint(word_ids)
                    final_constraints.append(constraint)

            # 11. prepare beam search scorer
            constrained_beam_scorer = ConstrainedBeamSearchScorer(
                constraints=final_constraints,
                batch_size=batch_size,
                num_beams=generation_config.num_beams,
                device=inputs_tensor.device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                num_beam_hyps_to_keep=generation_config.num_return_sequences,
                max_length=generation_config.max_length,
            )
            # 12. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )
            # 13. run beam search
            result = self._constrained_beam_search(
                input_ids,
                constrained_beam_scorer=constrained_beam_scorer,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        # Convert to legacy cache format if requested
        if (
            generation_config.return_legacy_cache is True
            and not is_torchdynamo_compiling()
            and hasattr(result, "past_key_values")
            and getattr(result.past_key_values, "to_legacy_cache") is not None
        ):
            result.past_key_values = result.past_key_values.to_legacy_cache()
        return result
    
    def stage2_forward_upperbound(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        '''
            1. train verifier, this forward will only calculate the category classification loss
            2. this forward function will replace the original forward function
            3. the classification loss will be calculated on every step.
        '''
        device = input_ids.device
        v_loss = 0
        if input_ids is not None and inputs_embeds is None and input_ids.size() == attention_mask.size(): # during inference, first time call forward
            # move inside forward(), one-time creation
            thought_token_id = torch.tensor([self.model.embed_tokens.num_embeddings - 1], device=input_ids.device, dtype=input_ids.dtype)
            thought_token_embed = self.model.embed_tokens(thought_token_id.to(self.model.embed_tokens.weight.device)) #.to(inputs_embeds.dtype) #.detach()

            for thought_idx in range(self.n_thought):
                input_ids_for_mark = input_ids.clone()

                if input_ids is not None and inputs_embeds is None and input_ids.size() == attention_mask.size():
                    inputs_embeds, where_thought_ids, thought_embeds = self.generate_embs(input_ids, None, attention_mask)
                    # input_ids = None
                else:
                    inputs_embeds, where_thought_ids, thought_embeds = self.generate_embs(input_ids_for_mark, inputs_embeds, attention_mask)

                # update thought idx and attention mask
                next_prefix_embeds = torch.zeros(inputs_embeds.shape[0], inputs_embeds.shape[1]+1, inputs_embeds.shape[2]).to(device).type_as(inputs_embeds) # (bs, seq_len+1, dim)
                next_prefix_ids = torch.zeros(input_ids.shape[0], input_ids.shape[1]+1).to(device).type_as(input_ids) # next_prefix_ids add a new column
            

                where_thought_ids = where_thought_ids.clone()
                where_thought_ids[:,1] = where_thought_ids[:,1] + 1
                # where_thought_ids[:,1] += 1 # update thought ids idx

                # next_labels = torch.cat([labels, labels[:,-1].unsqueeze(1)], dim=1)
                next_labels = torch.ones(labels.shape[0], labels.shape[1]+1).type_as(labels) if labels is not None else None
                next_attention_mask = torch.ones(attention_mask.shape[0], attention_mask.shape[1]+1).type_as(attention_mask)
                
                # update input_embeds, input_ids, labels, attention_mask
                for idx, thought_id_idx in enumerate(where_thought_ids[:,1]):
                    next_prefix_embeds[idx] = torch.cat([inputs_embeds[idx, :thought_id_idx, :], thought_token_embed, inputs_embeds[idx, thought_id_idx:, :]], dim=0) # update input_embeds
                    next_prefix_ids[idx] = torch.cat([input_ids[idx, :thought_id_idx], thought_token_id, input_ids[idx, thought_id_idx:]], dim=0)  # update input_ids
                    next_prefix_ids[idx, thought_id_idx-1] = 0
                    next_attention_mask[idx] = torch.cat([attention_mask[idx, :thought_id_idx], torch.LongTensor([1]).type_as(attention_mask), attention_mask[idx, thought_id_idx:]], dim=0) # update attention mask
                    if labels is not None:
                        next_labels[idx] = torch.cat([labels[idx, :thought_id_idx], torch.LongTensor([-100]).type_as(labels), labels[idx, thought_id_idx:]])             
                        next_labels[idx, thought_id_idx] = -100 # update labels

                input_ids = next_prefix_ids.clone()
                inputs_embeds = next_prefix_embeds.clone()
                labels = next_labels.clone() if labels is not None else None
                attention_mask = next_attention_mask.clone()

                if position_ids is not None:
                    next_pos = position_ids[:, -1:] + 1  
                    position_ids = torch.cat([position_ids, next_pos], dim=1)
                if cache_position is not None:
                    next_cache = cache_position[-1:] + 1  
                    cache_position = torch.cat([cache_position, next_cache], dim=0)

            input_ids = None # clear after reasoning for item title generation

            # train verifiers
            output_logits = []
            for v_idx in range(self.num_verifier):
                output_logits.append(self.verifiers[v_idx](thought_embeds)) # (bs, 1, n_cat)
            
            output_logits = torch.cat(output_logits, dim=1) # (bs, n_verifier, n_cat) 
            output_logits = output_logits.view(-1, self.num_category) # (bs*n_verifier, n_cat)

            v_loss = 0
            if "category_label" in kwargs:
                v_labels = kwargs["category_label"]
                v_labels = v_labels.repeat(self.num_verifier)  # (bs*n_verifier, )
                loss_func = nn.CrossEntropyLoss()
                v_loss = loss_func(output_logits, v_labels)
        
        return CausalLMOutputWithPast(
            loss=v_loss
        )
    
class LatentModelwithVerifier_RATT_MS_MoV(LatentModelwithVerifier_RATT_MS):
    def __init__(self, config):
        LatentModelwithVerifier_RATT.__init__(self, config)
        # self.verifiers = nn.ModuleList([nn.Linear(config.hidden_size, config.num_category) for _ in range(config.num_verifier)])

        # different verifier output dimension [3, 5, 2]
        self.verifier_dims = config.verifier_dims

        self.num_verifier = config.num_verifier
        self.hidden_size = config.hidden_size
        self.num_category = config.num_category 

        self.mono_weight = config.mono_weight
        self.v_weight = config.v_weight if "v_weight" in config else 0
        self.neg_weight = config.neg_weight if "neg_weight" in config else 0
        
        # mixture of verifiers, one head per aspect (Eq. 5)
        self.verifiers = nn.ModuleList([
            nn.Linear(self.hidden_size, out_dim) for out_dim in self.verifier_dims
        ])
        # personalized router g(.) (Eq. 6)
        self.gate = nn.Linear(self.hidden_size, self.num_verifier)

    def setup_verifier(self, config):
        self.verifier_dims = list(getattr(config, "verifier_dims", self.verifier_dims))
        self.num_verifier = len(self.verifier_dims)
        del self.verifiers
        self.verifiers = nn.ModuleList([
            nn.Linear(self.hidden_size, out_dim) for out_dim in self.verifier_dims
        ])
        self.gate = nn.Linear(self.hidden_size, self.num_verifier)

    def verify(self, thought_embeds):
        bs, _ = thought_embeds.shape
        eps = 1e-12

        # personalized router weights w_i (Eq. 6)
        gate_w = F.softmax(self.gate(thought_embeds), dim=-1) # (bs, k)

        probs_list = []    # (bs, n_cat_i)
        entropy_list = []  # (bs,)
        guid_list = []     # (bs, dim)

        for i, v in enumerate(self.verifiers):
            # multi-dimensional evaluation: per-aspect verifier V_i (Eq. 5)
            logits_i = v(thought_embeds)                 # (bs, n_cat_i)
            probs_i  = F.softmax(logits_i, dim=-1)
            probs_list.append(probs_i)

            # evaluation feedback f_i via entropy (Eq. 8)
            ent_i = -torch.sum(probs_i * torch.log(probs_i + eps), dim=-1)  # (bs,)
            entropy_list.append(ent_i)

            # guidance vector g_i = verifier's last-layer weight prototype (Eq. 9)
            idx_i = torch.argmax(probs_i, dim=-1)        # (bs,)
            Wi = v.weight                                # (n_cat_i, dim)
            guid_i = Wi[idx_i]                           # (bs, dim)
            guid_list.append(guid_i)

        # stack
        entropy_mat = torch.stack(entropy_list, dim=0)        # (k, bs)
        guidance_stack = torch.stack(guid_list, dim=1)        # (bs, k, dim)

        # normalize entropy and weight it by the gate -> fused weight
        # each expert is normalized by its own log(n_cat_i)
        denom = torch.tensor(
            [max(1, d) for d in self.verifier_dims],
            device=thought_embeds.device, dtype=thought_embeds.dtype
        )
        denom = torch.log(denom) + eps                         # (k,)
        norm_entropy = entropy_mat / denom.view(-1, 1)         # (k, bs)
        # router-weighted confidence score (Eq. 10)
        weight = torch.sum(gate_w.transpose(0,1) * norm_entropy, dim=0)  # (bs,)

        # router-weighted guidance vector (Eq. 9, 10)
        guidance_emb = torch.sum(gate_w.unsqueeze(-1) * guidance_stack, dim=1).to(thought_embeds.dtype)  # (bs, dim)

        return weight, guidance_emb, entropy_mat, gate_w
    

    def generate_embs(self, input_ids, input_embeds, attention_mask):
        
        if input_ids is not None and input_embeds is None:
            output_mid = Qwen2ForCausalLM.forward(self, input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        elif input_embeds is not None:
            output_mid = Qwen2ForCausalLM.forward(self, inputs_embeds=input_embeds, attention_mask=attention_mask, output_hidden_states=True)
        
        # generate embeddings
        thought_ids = self.model.embed_tokens.num_embeddings - 1  # thought id flag
        where_thought_ids = torch.nonzero(input_ids == thought_ids)

        input_embs = self.model.embed_tokens(input_ids) if input_embeds is None else input_embeds
        hidden_states = output_mid['hidden_states'][-1] # (bs, seq_len, dim)
        thought_embeds = self.attention(
            hidden_states=hidden_states, attention_mask=attention_mask,
            thought_id_idx=where_thought_ids[:, 1],
        ).to(input_embs.dtype) # (bs, dim)
        
        weight, guidance_emb, entropy, gate_w = self.verify(thought_embeds)
        # confidence-guided adjustment r* (Eq. 10)
        input_embs[where_thought_ids[:, 0], where_thought_ids[:, 1]] = (1-weight.unsqueeze(1).to(dtype=input_embs.dtype)) * thought_embeds + weight.unsqueeze(1).to(dtype=input_embs.dtype) * guidance_emb
        return input_embs, where_thought_ids, thought_embeds, guidance_emb, entropy, gate_w
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        device = input_ids.device
        v_loss = 0
        H_last = None
        loss_mono = 0
        all_thought_embeds = []
        all_entropy = []
        if input_ids is not None and inputs_embeds is None and input_ids.size() == attention_mask.size(): # during inference, first time call forward
            # move inside forward(), one-time creation
            thought_token_id = torch.tensor([self.model.embed_tokens.num_embeddings - 1], device=input_ids.device, dtype=input_ids.dtype)
            thought_token_embed = self.model.embed_tokens(thought_token_id.to(self.model.embed_tokens.weight.device)) #.to(inputs_embeds.dtype) #.detach()

            for thought_idx in range(self.n_thought):
                input_ids_for_mark = input_ids.clone()

                if input_ids is not None and inputs_embeds is None and input_ids.size() == attention_mask.size():
                    inputs_embeds, where_thought_ids, thought_embeds, guidance_embeds, H, gate_w = self.generate_embs(input_ids, None, attention_mask)
                else:
                    inputs_embeds, where_thought_ids, thought_embeds, guidance_embeds, H, gate_w = self.generate_embs(input_ids_for_mark, inputs_embeds, attention_mask)

                # compute the entropy difference
                if H_last is None:
                    H_last = H
                else:
                    difference = H - H_last
                    H_last = H
                    # monotonicity regularization (Eq. 13)
                    loss_mono += torch.exp(torch.mean(torch.clamp_min(difference.squeeze(0), 0.0)))

                # update thought idx and attention mask
                next_prefix_embeds = torch.zeros(inputs_embeds.shape[0], inputs_embeds.shape[1]+1, inputs_embeds.shape[2]).to(device).type_as(inputs_embeds) # (bs, seq_len+1, dim)
                next_prefix_ids = torch.zeros(input_ids.shape[0], input_ids.shape[1]+1).to(device).type_as(input_ids) # next_prefix_ids add a new column

                where_thought_ids = where_thought_ids.clone()
                where_thought_ids[:,1] = where_thought_ids[:,1] + 1

                next_labels = torch.ones(labels.shape[0], labels.shape[1]+1).type_as(labels) if labels is not None else None
                next_attention_mask = torch.ones(attention_mask.shape[0], attention_mask.shape[1]+1).type_as(attention_mask)
                
                # update input_embeds, input_ids, labels, attention_mask
                for idx, thought_id_idx in enumerate(where_thought_ids[:,1]):
                    next_prefix_embeds[idx] = torch.cat([inputs_embeds[idx, :thought_id_idx, :], thought_token_embed, inputs_embeds[idx, thought_id_idx:, :]], dim=0) # update input_embeds
                    next_prefix_ids[idx] = torch.cat([input_ids[idx, :thought_id_idx], thought_token_id, input_ids[idx, thought_id_idx:]], dim=0)  # update input_ids
                    next_prefix_ids[idx, thought_id_idx-1] = 0
                    next_attention_mask[idx] = torch.cat([attention_mask[idx, :thought_id_idx], torch.LongTensor([1]).type_as(attention_mask), attention_mask[idx, thought_id_idx:]], dim=0) # update attention mask
                    if labels is not None:
                        next_labels[idx] = torch.cat([labels[idx, :thought_id_idx], torch.LongTensor([-100]).type_as(labels), labels[idx, thought_id_idx:]])             
                        next_labels[idx, thought_id_idx] = -100 # update labels

                input_ids = next_prefix_ids.clone()
                inputs_embeds = next_prefix_embeds.clone()
                labels = next_labels.clone() if labels is not None else None
                attention_mask = next_attention_mask.clone()

                if position_ids is not None:
                    next_pos = position_ids[:, -1:] + 1  
                    position_ids = torch.cat([position_ids, next_pos], dim=1)
                if cache_position is not None:
                    next_cache = cache_position[-1:] + 1  
                    cache_position = torch.cat([cache_position, next_cache], dim=0)
                all_thought_embeds.append(thought_embeds.unsqueeze(0)) # collect all thoughts embeds for classification loss
                all_entropy.append(H.unsqueeze(0)) # (1, n_verifier, bs)
            input_ids = None # clear after reasoning for item title generation

            # verifier loss (Eq. 11), router-weighted across verifiers
            all_thought_embeds = torch.cat(all_thought_embeds, dim=0)  # (n_thoughts, bs, dim)
            n_thoughts, bs, _ = all_thought_embeds.shape

            # compute CE for each expert
            if "category_label" in kwargs:
                labels_per_expert = kwargs["category_label"]
                assert labels_per_expert.shape[1] == self.num_verifier, f"category label length {len(labels_per_expert)} != num_verifier {self.num_verifier}"

                per_exp_nll = []
                for k_idx, v in enumerate(self.verifiers):
                    logits_k = v(all_thought_embeds) # (n_thought, bs, n_cat_k)
                    probs_k  = F.softmax(logits_k, dim=-1) # (n_thoughts, bs, n_cat_k)
                    label_k = labels_per_expert[:, k_idx].unsqueeze(0).repeat(n_thoughts, 1)  # (n_thoughts, bs)
                    picked = torch.gather(probs_k, dim=-1, index=label_k.unsqueeze(-1)).squeeze(-1)  # (n_thoughts, bs)
                    nll_k = -torch.log(picked + 1e-12)                    # (n_thoughts, bs)
                    per_exp_nll.append(nll_k)  # (n_verifier, n_thoguhts, bs)

                per_exp_nll = torch.stack(per_exp_nll, dim=2)           # (n_thoughts, bs, n_verifier)
                # weighted by gate
                nll = torch.sum(gate_w.unsqueeze(0) * per_exp_nll, dim=2)            # (n_thoughts, bs)

                if "correct" in kwargs and kwargs["correct"] is not None:
                    correct_tensor = kwargs["correct"].unsqueeze(0).repeat(n_thoughts, 1)  # (n_thoughts, bs)
                else:
                    correct_tensor = torch.ones_like(nll)

                v_loss = (nll * correct_tensor).mean()

                # overall loss = rec loss + beta * verifier loss + gamma * monotonicity reg (Eq. 14)
                v_loss += self.mono_weight * loss_mono

        # during inference, kv cache is on
        return self.qwen_forward(
            input_ids=input_ids,
            attention_mask=attention_mask, # (bs, seq_len+n_thought)
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, # (bs, seq_len+n_thought, dim)
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            num_logits_to_keep=num_logits_to_keep,
            loss=v_loss,
            **kwargs,
        )