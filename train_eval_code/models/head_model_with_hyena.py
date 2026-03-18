from transformers.models.gpt2.modeling_gpt2 import BaseModelOutputWithPastAndCrossAttentions, logging
import torch
from torch import nn
from typing import Optional, Tuple, Union
from transformers import AutoModelForSequenceClassification, PreTrainedModel, BertForSequenceClassification
import hydra
from .head_config import HeadConfig

logger = logging.get_logger(__name__)
from torch import nn

# class HeadModelWithHyena(BertForSequenceClassification):
#     def __init__(self, config, **kwargs):
#         super(HeadModelWithHyena, self).__init__(config)
#         self.hyena_model = hydra.utils.instantiate(kwargs["hyena_model"])
#         self.hyena_model.eval()
#         self.num_seqs = kwargs["num_seqs"]
         
#         # Initialize weights and apply final processing
#         self.post_init()
       
#     def forward(
#         self,
#         input_ids_of_seqs: Optional[torch.LongTensor] = None,
#         labels: Optional[torch.LongTensor] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
       
#         batch_size, num_seqs, num_base = input_ids_of_seqs.shape
#         batch_seq_embeds = self.hyena_model(input_ids_of_seqs.view(-1, num_base))[0][:, 0, :]
#         batch_seq_embeds = batch_seq_embeds.view(batch_size, num_seqs, -1)

#         outputs = self.bert(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )

#         pooled_output = outputs[1]

#         pooled_output = self.dropout(pooled_output)
#         logits = self.classifier(pooled_output)

#         loss = None
#         if labels is not None:
#             if self.config.problem_type is None:
#                 if self.num_labels == 1:
#                     self.config.problem_type = "regression"
#                 elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
#                     self.config.problem_type = "single_label_classification"
#                 else:
#                     self.config.problem_type = "multi_label_classification"

#             if self.config.problem_type == "regression":
#                 loss_fct = MSELoss()
#                 if self.num_labels == 1:
#                     loss = loss_fct(logits.squeeze(), labels.squeeze())
#                 else:
#                     loss = loss_fct(logits, labels)
#             elif self.config.problem_type == "single_label_classification":
#                 loss_fct = CrossEntropyLoss()
#                 loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
#             elif self.config.problem_type == "multi_label_classification":
#                 loss_fct = BCEWithLogitsLoss()
#                 loss = loss_fct(logits, labels)
#         if not return_dict:
#             output = (logits,) + outputs[2:]
#             return ((loss,) + output) if loss is not None else output

#         return SequenceClassifierOutput(
#             loss=loss,
#             logits=logits,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )

#         return logits

class HeadModelWithHyena(nn.Module):
    def __init__(self, hyena_model, head, num_seqs):
        super(HeadModelWithHyena, self).__init__()
        self.hyena_model = hydra.utils.instantiate(hyena_model)
        self.hyena_model.eval()
        self.head = hydra.utils.instantiate(head)
        self.num_seqs = num_seqs
         
        # Initialize weights and apply final processing
        self.post_init()
       
    def forward(
        self,
        input_ids_of_seqs: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
       
        batch_size, num_seqs, num_base = input_ids_of_seqs.shape
        batch_seq_embeds = self.hyena_model(input_ids_of_seqs.view(-1, num_base))[0][:, 0, :]
        batch_seq_embeds = batch_seq_embeds.view(batch_size, num_seqs, -1)

        logits = self.head(batch_seq_embeds)

        return logits

# from transformers import 
# class HeadModelWithHyena(PreTrainedModel):
#     config_class = HeadConfig

#     def __init__(self, config):
#         super(HeadModelWithHyena, self).__init__(config)
#         self.hyena_model = hydra.utils.instantiate(config["hyena_model"])
#         self.hyena_model.eval()
#         self.head = hydra.utils.instantiate(config["head"])
#         self.num_seqs = config["num_seqs"]
         
#         # Initialize weights and apply final processing
#         self.post_init()
       
#     def forward(
#         self,
#         input_ids_of_seqs: Optional[torch.LongTensor] = None,
#         labels: Optional[torch.LongTensor] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
       
#         batch_size, num_seqs, num_base = input_ids_of_seqs.shape
#         batch_seq_embeds = self.hyena_model(input_ids_of_seqs.view(-1, num_base))[0][:, 0, :]
#         batch_seq_embeds = batch_seq_embeds.view(batch_size, num_seqs, -1)

#         logits = self.head(batch_seq_embeds)

#         return logits