from transformers.models.gpt2.modeling_gpt2 import BaseModelOutputWithPastAndCrossAttentions, logging
import torch
from typing import Optional, Tuple, Union
from transformers import GPT2LMHeadModel
import hydra
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from .gpt2_model_with_hyena import GPT2ModelWithHyena

 
logger = logging.get_logger(__name__)


class GPT2(GPT2LMHeadModel):
    def __init__(self, config, **kwargs):
        super(GPT2, self).__init__(config)
        config.num_sequences = kwargs["dataset"]["num_sequences"]
        self.transformer = GPT2ModelWithHyena(config, hyena_model=kwargs["hyena_model"])
        self.stage = kwargs["train_stage"]
        
        self.jianji_head = torch.nn.Linear(self.transformer.wte.weight.shape[0], config.dim_of_hyena_hidden)
       
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        start_index_of_seq: Optional[torch.LongTensor] = None,
        input_ids_of_seqs: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        transformer_outputs = self.transformer(
            input_ids,
            start_index_of_seq,
            input_ids_of_seqs,
            labels,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0][0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            if self.stage == 1: # pretrain
                shift_logits_cells_list = []
                for i in range(len(input_ids_of_seqs)): # batchsize
                    shift_logits_cells = lm_logits[i, start_index_of_seq[i]-1:start_index_of_seq[i] + self.transformer.num_seqs-1, :].contiguous()
                    shift_logits_cells = self.jianji_head(shift_logits_cells)
                    shift_logits_cells_list.append(shift_logits_cells)
                shift_logits_cells = torch.stack(shift_logits_cells_list, dim=0)
                jianji_embedding_labels = transformer_outputs[1]
                loss_fct_jianji = MSELoss()
                loss_jianji = loss_fct_jianji(shift_logits_cells.view(-1, shift_logits_cells.size(-1)), jianji_embedding_labels.view(-1, jianji_embedding_labels.size(-1)))
                loss += loss_jianji
        if not return_dict:
            output = (lm_logits,) + transformer_outputs[0][1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs[0].past_key_values,
            hidden_states=transformer_outputs[0].hidden_states,
            attentions=transformer_outputs[0].attentions,
            cross_attentions=transformer_outputs[0].cross_attentions,
        )
