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

        # 定义Q, K, V的线性变换层
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.scale_score = 1.0/(self.hidden_size ** 0.5)
        self.rope = VanillaRoPE(hidden_size)

    @staticmethod
    def mask_to_weights(attention_mask, thought_id_idx, end_k=-1):

        if thought_id_idx is not None and attention_mask.size(0) != thought_id_idx.size(0): 
            ipdb.set_trace()
            raise ValueError("attention_mask must have the same size as thought_id_idx")

        if end_k != -1:
            # 把thought_id前end_k个位置的内容考虑进去
            # 这里的代码有一点问题：也就是我们认为end_k不会长过前面的句子长度
            # 否则会把padding的部分置为1
            for i in range(attention_mask.size(0)):
                idx = thought_id_idx[i].item()
                start_idx = max(0, idx - end_k)
                attention_mask[i].zero_()
                attention_mask[i, start_idx:idx] = 1
        else:
            # 把thought_id前所有的内容都考虑进去 (要屏蔽padding和thought_id后面的部分)
            # 这里想一下
            # 在训练的时候，padding在右侧
            # 在推理的时候，padding在左侧
            # 而在推理的时候，我们传进来的attention_mask实际上是正确的，所以不需要全部置0
            # 训练时候，代码可以改为: attention_mask[i, idx:] = 0
            # 推理时候, idx == seq_len, 也是可以和训练时候一样去改，所以修改
            for i in range(attention_mask.size(0)):
                idx = thought_id_idx[i].item()
                attention_mask[i, idx:] = 0
            # TODO: 这个地方的attention mask的操作？为什么是thought_idx这个地方? n_thought == 1: n_thought这个地方是看不到自己的，

        # Convert mask to weights: 1 -> 0, 0 -> -inf
        attention_weight = torch.zeros_like(attention_mask, dtype=torch.float32)
        attention_weight = attention_weight.masked_fill(~attention_mask.bool(), float('-inf'))
        return attention_weight

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor,
                thought_id_idx) -> torch.Tensor:
        # ipdb.set_trace() # TODO: check hidden state (bs, seq_len, dim), attention_mask (bs, seq_len), thought_id_idx (bs,) - column idx
        # hidden_states = hidden_states.to(torch.float32)
        attention_mask = SelfAttentionLayer.mask_to_weights(attention_mask.clone(), thought_id_idx, self.end_k) # (bs, seq_len)

        cos, sin = self.rope(
            hidden_states,
            position_ids=torch.arange(0, hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)
        )

        batch_size, seq_len, _ = hidden_states.shape

        # 生成Q, K, V
        Q = self.query(hidden_states)  # [batch_size, seq_len, embed_dim]
        K = self.key(hidden_states)
        V = self.value(hidden_states)

        # 加入位置编码
        Q, K = apply_rotary_pos_emb(Q, K, cos.squeeze(0), sin.squeeze(0), unsqueeze_dim=0)

        # 确定目标位置索引
        if thought_id_idx is None:
            # 取每个样本的最后一个位置（seq_len - 1）
            indices = torch.full((batch_size,), seq_len - 1, device=hidden_states.device)
        else:
            indices = thought_id_idx - 1
            if (indices < 0).any() or (indices >= seq_len).any():
                raise ValueError("thought_id_idx-1 exceeds valid sequence indices")

        # 提取目标位置的Q向量 [batch_size, 1, embed_dim]
        Q_selected = Q[torch.arange(batch_size), indices].unsqueeze(1)

        # 计算注意力分数 [batch_size, 1, seq_len]
        attn_scores = torch.matmul(Q_selected, K.transpose(-2, -1)) * self.scale_score

        # 应用attention_mask（形状需匹配为[batch_size, 1, seq_len]）
        attn_scores += attention_mask.unsqueeze(1)  # 广播至[batch_size, 1, seq_len]

        # 计算注意力权重与输出
        attn_weights = F.softmax(attn_scores, dim=-1)
        output_selected = torch.matmul(attn_weights, V)  # [batch_size, 1, embed_dim]
        output = output_selected.squeeze(1) # [batch_size, embed_dim]
        return output


class Noise(nn.Module):
    def __init__(self, hidden_size):
        super(Noise, self).__init__()
        # 这里的初始化就放在from_pretrained之后去做
        # 对于 log_var  model.mlp.log_var = torch.nn.Parameter(torch.tensor([float('-inf')] * model.config.hidden_size))
        # 对于 mu       nn.init.eyes
        self.mu = nn.Linear(hidden_size, hidden_size, bias=False)
        self.log_var = torch.nn.Parameter(torch.tensor([float('-10')] * hidden_size))

    def forward(self, x):
        x = self.mu(x)
        std = torch.exp(self.log_var / 2)
        return x + torch.randn_like(x).mul(std)


class LatentModel(Qwen2ForCausalLM):

    def __init__(self, config):
        super().__init__(config)
        # self.mlp = Noise(config.hidden_size)
        self.attention = SelfAttentionLayer(config.hidden_size)

    def generate_embs(self, input_ids, attention_mask):
        output_mid = super().forward(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True) 
        # generate embeddings
        thought_ids = self.model.embed_tokens.num_embeddings - 1  # thought id flag
        where_thought_ids = torch.nonzero(input_ids == thought_ids)
        hidden_states = output_mid['hidden_states']
        input_embs = self.model.embed_tokens(input_ids)
        # ipdb.set_trace() # attention mask (bs, seq_len)
        input_embs[where_thought_ids[:, 0], where_thought_ids[:, 1]] = self.attention(
            hidden_states=hidden_states[-1], attention_mask=attention_mask,
            thought_id_idx=where_thought_ids[:, 1],
        ).to(input_embs.dtype)

        return input_embs


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
        if input_ids is not None and inputs_embeds is None and input_ids.size() == attention_mask.size():
            inputs_embeds = self.generate_embs(input_ids, attention_mask)
            input_ids = None
        # elif input_ids is not None and inputs_embeds is None and input_ids.size()!=attention_mask.size():
            # 这个时候是在做生成任务，已经有了cache所以用input_ids即可
        ipdb.set_trace()
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            num_logits_to_keep=num_logits_to_keep,
            **kwargs,
        )


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

    def update_thought_token_info(self):
        pass
        # self.thought_token_id = torch.LongTensor([self.model.embed_tokens.num_embeddings - 1])
        # self.thought_token_embeds = self.model.embed_tokens(self.thought_token_id)

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
        
        # 单步reasoning实现逻辑 - 拿thought token之前的最后一个latent hidden state，替换掉thought token的embedding，取最后的hidden state
        # 多部reasoning实现逻辑 - 训练：每次替换之后，要把thought_idx + 1, 然后，在后面加一列padding，随后更新attention_mask，老的thought_idx位置 置1，后面不变. 更新input_embeds, 把原来thought_idx以及之后的input_embeds（替换latent hidden state到旧thought token之前）放到 新的thought_idx 以及之后的位置，还要更新labels
        # 多部reasoning实现逻辑 - inference：kv cache需要保留，只用最新的input_ids, attention_mask 和 cache position在外部更新时需要调整更改
        
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
                # TODO (joao): remove output/input mismatch when these old models (xlnet, reformer) are deprecated
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