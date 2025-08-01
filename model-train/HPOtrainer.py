import torch
import torch.nn.functional as F
from torch import nn
import re
from typing import Dict, Optional, Union, Tuple, Any, Literal, List
from dataclasses import dataclass, field
import warnings
from contextlib import nullcontext
import torch.amp as amp

from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    AutoModelForCausalLM,
    BaseImageProcessor,
    FeatureExtractionMixin,
    ProcessorMixin
)
from accelerate import PartialState, Accelerator
from datasets import Dataset, IterableDataset
from transformers.data.data_collator import DataCollatorMixin
from transformers.utils import is_peft_available, is_torch_xpu_available

from trl.trainer.dpo_trainer import (
    DPOTrainer,
    maybe_extract_prompt,
    maybe_apply_chat_template,
    DPOConfig
)

from HPOconfig import HPOConfig

from trl.trainer.utils import (
    RunningMoments,
    cap_exp,
    disable_dropout_in_model,
    empty_cache,
    flush_left,
    generate_model_card,
    get_comet_experiment_url,
    log_table_to_comet_experiment,
    pad,
    pad_to_length,
    peft_module_casting_to_bf16,
    selective_log_softmax,
)

if is_peft_available():
    pass

@dataclass
class CustomDataCollatorForPreferenceWithSegMasks(DataCollatorMixin):
    pad_token_id: int
    return_tensors: str = "pt"

    def torch_call(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not examples:
            return {}

        prompt_input_ids = [torch.tensor(example["prompt_input_ids"]) for example in examples]
        prompt_attention_mask = [torch.ones_like(ids) for ids in prompt_input_ids]
        chosen_input_ids = [torch.tensor(example["chosen_input_ids"]) for example in examples]
        chosen_attention_mask = [torch.ones_like(ids) for ids in chosen_input_ids]
        rejected_input_ids = [torch.tensor(example["rejected_input_ids"]) for example in examples]
        rejected_attention_mask = [torch.ones_like(ids) for ids in rejected_input_ids]

        process_chosen_seg_masks = "chosen_prefix_mask" in examples[0]
        process_rejected_seg_masks = "rejected_prefix_mask" in examples[0]

        if process_chosen_seg_masks:
            chosen_prefix_masks = [torch.tensor(example["chosen_prefix_mask"], dtype=torch.bool) for example in examples]
            chosen_mid_masks    = [torch.tensor(example["chosen_mid_mask"], dtype=torch.bool) for example in examples]
            chosen_suffix_masks = [torch.tensor(example["chosen_suffix_mask"], dtype=torch.bool) for example in examples]
        
        if process_rejected_seg_masks:
            rejected_prefix_masks = [torch.tensor(example["rejected_prefix_mask"], dtype=torch.bool) for example in examples]
            rejected_mid_masks    = [torch.tensor(example["rejected_mid_mask"], dtype=torch.bool) for example in examples]
            rejected_suffix_masks = [torch.tensor(example["rejected_suffix_mask"], dtype=torch.bool) for example in examples]

        if "pixel_values" in examples[0]:
            pixel_values = [torch.tensor(example["pixel_values"]) for example in examples]
        if "pixel_attention_mask" in examples[0]:
            pixel_attention_mask = [torch.tensor(example["pixel_attention_mask"]) for example in examples]
        if "ref_chosen_logps" in examples[0] and "ref_rejected_logps" in examples[0]:
            ref_chosen_logps = torch.tensor([example["ref_chosen_logps"] for example in examples])
            ref_rejected_logps = torch.tensor([example["ref_rejected_logps"] for example in examples])

        output = {}
        output["prompt_input_ids"] = pad(prompt_input_ids, padding_value=self.pad_token_id, padding_side="left")
        output["prompt_attention_mask"] = pad(prompt_attention_mask, padding_value=0, padding_side="left")
        output["chosen_input_ids"] = pad(chosen_input_ids, padding_value=self.pad_token_id)
        output["chosen_attention_mask"] = pad(chosen_attention_mask, padding_value=0)
        output["rejected_input_ids"] = pad(rejected_input_ids, padding_value=self.pad_token_id)
        output["rejected_attention_mask"] = pad(rejected_attention_mask, padding_value=0)
        if "pixel_values" in examples[0]:
            output["pixel_values"] = pad(pixel_values, padding_value=0.0)
        if "pixel_attention_mask" in examples[0]:
            output["pixel_attention_mask"] = pad(pixel_attention_mask, padding_value=0)
        if "image_sizes" in examples[0]:
            output["image_sizes"] = torch.tensor([example["image_sizes"] for example in examples])
        if "ref_chosen_logps" in examples[0] and "ref_rejected_logps" in examples[0]:
            output["ref_chosen_logps"] = ref_chosen_logps
            output["ref_rejected_logps"] = ref_rejected_logps

        if process_chosen_seg_masks:
            output["chosen_prefix_mask"] = pad(chosen_prefix_masks, padding_value=0)
            output["chosen_mid_mask"]    = pad(chosen_mid_masks,    padding_value=0)
            output["chosen_suffix_mask"] = pad(chosen_suffix_masks, padding_value=0)
        
        if process_rejected_seg_masks:
            output["rejected_prefix_mask"] = pad(rejected_prefix_masks, padding_value=0)
            output["rejected_mid_mask"]    = pad(rejected_mid_masks,    padding_value=0)
            output["rejected_suffix_mask"] = pad(rejected_suffix_masks, padding_value=0)

        return output   

class HPOTrainer(DPOTrainer):
    def __init__(self, 
                 model: Optional[Union[PreTrainedModel, nn.Module, str]] = None, 
                 ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None, 
                 args: Optional[HPOConfig] = None,
                 data_collator: Optional[Any] = None,
                 train_dataset: Optional[Dataset] = None, 
                 eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None, 
                 processing_class: Optional[Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]] = None, 
                 **kwargs):

        if args is None:
            raise ValueError("HPOTrainer requires 'args' (an instance of HPOConfig).")
        
        if args.padding_value is not None:
            self.padding_value = args.padding_value
        else:
            if hasattr(processing_class, "pad_token_id") and processing_class.pad_token_id is not None:
                self.padding_value = processing_class.pad_token_id
            elif hasattr(processing_class, "tokenizer") and processing_class.tokenizer.pad_token_id is not None:
                self.padding_value = processing_class.tokenizer.pad_token_id
            else:
                raise ValueError(
                    "`padding_value` is not specified in `DPOConfig`, and `pad_token_id` is missing in the "
                    "`processing_class`. Please either set the `padding_value` argument in `DPOConfig`, or set "
                    "`tokenizer.pad_token` (e.g., `tokenizer.pad_token = tokenizer.eos_token`) before instantiating "
                    "the trainer."
                )

        if data_collator is None:
            data_collator = CustomDataCollatorForPreferenceWithSegMasks(pad_token_id=self.padding_value)

        super().__init__(
            model=model,
            ref_model=ref_model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            **kwargs
        )

        if not isinstance(self.args, HPOConfig):
            raise ValueError(
                 "Internal error or misconfiguration: self.args is not an instance of HPOConfig "
                 "after parent __init__. Ensure 'args' parameter to HPOTrainer is correct."
            )

        self.kl_prefix_weight = self.args.kl_prefix_weight
        self.kl_suffix_weight = self.args.kl_suffix_weight
        self.mid_loss_weight = self.args.mid_dpo_weight

        new_signature_columns = [
            "chosen_prefix_mask", "chosen_mid_mask", "chosen_suffix_mask",
            "rejected_prefix_mask", "rejected_mid_mask", "rejected_suffix_mask",
        ]
        if not hasattr(self, '_signature_columns') or self._signature_columns is None:
             warnings.warn(
                 "_signature_columns was not initialized by parent Trainer as expected. "
                 "Initializing a new one. This might indicate an issue with the TRL/Transformers version or setup.", 
                 UserWarning
            )
             self._signature_columns = []

        for col in new_signature_columns:
            if col not in self._signature_columns:
                self._signature_columns.append(col)

    def _prepare_dataset(
        self,
        dataset: Union[Dataset, IterableDataset],
        processing_class: Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin],
        args: HPOConfig, 
        dataset_name: str,
    ) -> Union[Dataset, IterableDataset]:
        map_kwargs = {"writer_batch_size": args.dataset_writer_batch_size if hasattr(args, 'dataset_writer_batch_size') else 1000}
        
        if isinstance(dataset, Dataset):
            if hasattr(args, 'dataset_num_proc') and args.dataset_num_proc is not None and args.dataset_num_proc > 0:
                map_kwargs["num_proc"] = args.dataset_num_proc
        
        with PartialState().main_process_first():
            if isinstance(dataset, Dataset):
                map_kwargs["desc"] = f"Extracting prompt in {dataset_name} dataset"
            dataset = dataset.map(maybe_extract_prompt, **map_kwargs)
            if isinstance(dataset, Dataset):
                map_kwargs["desc"] = f"Applying chat template to {dataset_name} dataset"
            dataset = dataset.map(
                maybe_apply_chat_template, fn_kwargs={"tokenizer": processing_class, "tools": args.tools}, **map_kwargs
            )
            def process_diff_and_add_masks_revised(example):
                chosen_clean, chosen_char_span = self.process_diff_tags(example["chosen"])
                chosen_enc = processing_class(chosen_clean, return_offsets_mapping=True, add_special_tokens=False, truncation=True, max_length=args.max_completion_length)
                chosen_enc["input_ids"] = chosen_enc["input_ids"] + [processing_class.eos_token_id]
                chosen_enc["offset_mapping"] = chosen_enc["offset_mapping"] + [(0, 0)]
                if args.max_completion_length is not None:
                    chosen_enc["input_ids"] = chosen_enc["input_ids"][:args.max_completion_length]
                if chosen_enc["input_ids"]:
                    chosen_prefix_mask, chosen_mid_mask, chosen_suffix_mask = self.generate_masks_from_offsets(chosen_enc["offset_mapping"], chosen_char_span, len(chosen_enc["input_ids"]))
                else: 
                    chosen_prefix_mask, chosen_mid_mask, chosen_suffix_mask = torch.empty(0, dtype=torch.bool), torch.empty(0, dtype=torch.bool), torch.empty(0, dtype=torch.bool)
                
                rejected_clean, rejected_char_span = self.process_diff_tags(example["rejected"])
                rejected_enc = processing_class(rejected_clean, return_offsets_mapping=True, add_special_tokens=False, truncation=True, max_length=args.max_completion_length)
                rejected_enc["input_ids"] = rejected_enc["input_ids"] + [processing_class.eos_token_id]
                rejected_enc["offset_mapping"] = rejected_enc["offset_mapping"] + [(0, 0)]
                if args.max_completion_length is not None:
                    rejected_enc["input_ids"] = rejected_enc["input_ids"][:args.max_completion_length]
                if rejected_enc["input_ids"]:
                    rejected_prefix_mask, rejected_mid_mask, rejected_suffix_mask = self.generate_masks_from_offsets(rejected_enc["offset_mapping"], rejected_char_span, len(rejected_enc["input_ids"]))
                else:
                    rejected_prefix_mask, rejected_mid_mask, rejected_suffix_mask = torch.empty(0, dtype=torch.bool), torch.empty(0, dtype=torch.bool), torch.empty(0, dtype=torch.bool)

                example["chosen"] = chosen_clean 
                example["rejected"] = rejected_clean 
                
                example["chosen_prefix_mask"] = chosen_prefix_mask.tolist()
                example["chosen_mid_mask"] = chosen_mid_mask.tolist()
                example["chosen_suffix_mask"] = chosen_suffix_mask.tolist()
                example["rejected_prefix_mask"] = rejected_prefix_mask.tolist()
                example["rejected_mid_mask"] = rejected_mid_mask.tolist()
                example["rejected_suffix_mask"] = rejected_suffix_mask.tolist()

                return example

            if isinstance(dataset, Dataset):
                map_kwargs["desc"] = f"Processing diff tags and generating masks for {dataset_name} dataset"
            dataset = dataset.map(process_diff_and_add_masks_revised, **map_kwargs)

            columns_to_remove_after_tokenization = [
                col for col in ["prompt", "chosen", "rejected"] if col in dataset.column_names
            ]

            if isinstance(dataset, Dataset):
                map_kwargs["desc"] = f"Tokenizing {dataset_name} dataset"

            dataset = dataset.map(
                self.tokenize_row if not self.is_vision_model else self.process_row,
                remove_columns=["prompt", "chosen", "rejected"],
                fn_kwargs={
                    "processing_class": processing_class,
                    "max_prompt_length": args.max_prompt_length,
                    "max_completion_length": args.max_completion_length,
                    "add_special_tokens": False,
                },
                **map_kwargs,
            )
       
        return dataset

    def process_diff_tags(self, text: str) -> Tuple[str, Tuple[int, int]]:
        diff_pattern = r'<diff>(.*?)</diff>'
        match = re.search(diff_pattern, text, re.DOTALL)
        if not match:
            return text, (0, len(text))
        start_tag_pos = match.start()
        end_tag_pos = match.end()
        diff_content = match.group(1)
        cleaned_text = text[:start_tag_pos] + diff_content + text[end_tag_pos:]
        char_span_start = start_tag_pos
        char_span_end = start_tag_pos + len(diff_content)
        return cleaned_text, (char_span_start, char_span_end)

    def generate_masks_from_offsets(
        self,
        offsets: list[Tuple[int, int]],
        diff_char_span: Tuple[int, int],
        completion_seq_len: int
    ) -> Tuple[torch.BoolTensor, torch.BoolTensor, torch.BoolTensor]:

        if completion_seq_len == 0:
            return (
                torch.zeros(0, dtype=torch.bool),
                torch.zeros(0, dtype=torch.bool),
                torch.zeros(0, dtype=torch.bool),
            )

        start_char, end_char = diff_char_span

        if not offsets:
            return (
                torch.zeros(0, dtype=torch.bool),
                torch.zeros(0, dtype=torch.bool),
                torch.zeros(0, dtype=torch.bool),
            )
            
        offsets_tensor = torch.tensor(offsets, dtype=torch.long)
        token_starts = offsets_tensor[:, 0]
        token_ends = offsets_tensor[:, 1]

        is_valid_token_mask = token_starts != token_ends

        prefix_mask = (token_ends <= start_char) & is_valid_token_mask

        suffix_mask = (token_starts >= end_char) & is_valid_token_mask

        raw_prefix_condition = token_ends <= start_char
        raw_suffix_condition = token_starts >= end_char
        
        mid_mask = (~raw_prefix_condition) & (~raw_suffix_condition) & is_valid_token_mask

        return prefix_mask, mid_mask, suffix_mask

    def calculate_mid_avg_logps(
        self,
        logps: torch.FloatTensor,
        mid_mask: torch.BoolTensor,
    ) -> Tuple[torch.FloatTensor]:
        device = logps.device
        mid_mask = mid_mask.to(device)
        mid_sum = (logps * mid_mask).sum(dim=-1)
        mid_avg_logps = mid_sum / mid_mask.sum(dim=-1)
        return mid_avg_logps
    
    def calculate_kl_loss(
        self,
        logps: torch.FloatTensor,
        ref_logps: torch.FloatTensor,
        mask: torch.BoolTensor,
    ) -> Tuple[torch.FloatTensor]:
        device = logps.device
        mask = mask.to(device)
        kl_loss = F.kl_div(logps, ref_logps, reduction="none", log_target=True)

        kl_loss = (kl_loss * mask).sum(dim=-1)

        mask_sum = mask.sum(dim=-1)
        epsilon = 1e-20
        kl_loss = kl_loss / (mask_sum + epsilon)
        kl_loss = torch.where(
            mask_sum > 0,
            kl_loss,
            torch.zeros_like(kl_loss) 
        )

        return kl_loss

    def dpo_loss(
        self,
        chosen_logps: torch.FloatTensor,
        rejected_logps: torch.FloatTensor,
        ref_chosen_logps: torch.FloatTensor,
        ref_rejected_logps: torch.FloatTensor,
        chosen_per_token_logps: torch.FloatTensor,
        rejected_per_token_logps: torch.FloatTensor,
        ref_chosen_per_token_logps: torch.FloatTensor,
        ref_rejected_per_token_logps: torch.FloatTensor,
        chosen_prefix_mask: torch.BoolTensor,
        chosen_mid_mask: torch.BoolTensor,
        chosen_suffix_mask: torch.BoolTensor,
        rejected_prefix_mask: torch.BoolTensor,
        rejected_mid_mask: torch.BoolTensor,
        rejected_suffix_mask: torch.BoolTensor,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        device = self.accelerator.device
        chosen_mid_avg_logps= self.calculate_mid_avg_logps(chosen_per_token_logps, chosen_mid_mask)
        rejected_mid_avg_logps= self.calculate_mid_avg_logps(rejected_per_token_logps, rejected_mid_mask)
        ref_chosen_mid_avg_logps= self.calculate_mid_avg_logps(ref_chosen_per_token_logps, chosen_mid_mask)
        ref_rejected_mid_avg_logps= self.calculate_mid_avg_logps(ref_rejected_per_token_logps, rejected_mid_mask)
        prefix_kl_loss = self.calculate_kl_loss(chosen_per_token_logps, ref_chosen_per_token_logps, chosen_prefix_mask)
        suffix_kl_loss = self.calculate_kl_loss(chosen_per_token_logps, ref_chosen_per_token_logps, chosen_suffix_mask)
        mid_logratios = ((chosen_mid_avg_logps - rejected_mid_avg_logps) - (ref_chosen_mid_avg_logps - ref_rejected_mid_avg_logps)).to(device)

        if self.loss_type == "sigmoid":
            mid_dpo_loss = (-F.logsigmoid(self.beta * mid_logratios) * (1 - self.label_smoothing)
                            - F.logsigmoid(-self.beta * mid_logratios) * self.label_smoothing)
        elif self.loss_type == "robust":
            denominator = (1 - 2 * self.label_smoothing)
            if denominator == 0: 
                mid_dpo_loss = torch.tensor(0.0, device=device) 
                if self.label_smoothing > 0: 
                     warnings.warn("Denominator for robust DPO loss is zero due to label_smoothing. Setting mid_dpo_loss to 0.", UserWarning)
            else:
                mid_dpo_loss = ((-F.logsigmoid(self.beta * mid_logratios) * (1 - self.label_smoothing)
                                + F.logsigmoid(-self.beta * mid_logratios) * self.label_smoothing) /
                                denominator)
        elif self.loss_type == "hinge":
            mid_dpo_loss = torch.relu(1 - self.beta * mid_logratios)
        elif self.loss_type == "ipo":
            mid_dpo_loss = (mid_logratios - 1 / (2 * self.beta)) ** 2
        else:
            warnings.warn(f"Unsupported loss_type {self.loss_type}, defaulting to sigmoid.", UserWarning)
            mid_dpo_loss = -F.logsigmoid(self.beta * mid_logratios)

        prefix_kl_loss = prefix_kl_loss ** 2
        suffix_kl_loss = suffix_kl_loss ** 2
        losses =(self.mid_loss_weight * mid_dpo_loss +
                  self.kl_prefix_weight * prefix_kl_loss +
                  self.kl_suffix_weight * suffix_kl_loss)
        
        chosen_rewards = self.beta * (chosen_logps.sum(dim=-1).to(device) - ref_chosen_logps.sum(dim=-1).to(device)).detach()
        rejected_rewards = self.beta * (rejected_logps.sum(dim=-1).to(device) - ref_rejected_logps.sum(dim=-1).to(device)).detach()
        return losses, chosen_rewards, rejected_rewards

    def get_batch_loss_metrics(
        self,
        model,
        batch: dict[str, Union[list, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        metrics = {}
        model_output = self.concatenated_forward(model, batch)
        
        if "ref_chosen_logps" in batch and "ref_rejected_logps" in batch:
            ref_chosen_logps = batch["ref_chosen_logps"]
            ref_rejected_logps = batch["ref_rejected_logps"]
            ref_chosen_per_token_logps = batch["ref_chosen_per_token_logps"]
            ref_rejected_per_token_logps = batch["ref_rejected_per_token_logps"]
        else:
            context_manager = torch.no_grad() if (self.ref_model is not None and self.ref_model is not self.model) else nullcontext()
            with context_manager:
                 ref_chosen_logps, ref_rejected_logps, ref_chosen_per_token_logps, ref_rejected_per_token_logps = self.compute_ref_log_probs(batch)

        chosen_prefix_mask = model_output["chosen_prefix_mask"].bool().to(self.accelerator.device)
        chosen_mid_mask = model_output["chosen_mid_mask"].bool().to(self.accelerator.device)
        chosen_suffix_mask = model_output["chosen_suffix_mask"].bool().to(self.accelerator.device)
        rejected_prefix_mask = model_output["rejected_prefix_mask"].bool().to(self.accelerator.device)
        rejected_mid_mask = model_output["rejected_mid_mask"].bool().to(self.accelerator.device)
        rejected_suffix_mask = model_output["rejected_suffix_mask"].bool().to(self.accelerator.device)

        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            model_output["chosen_logps"], model_output["rejected_logps"],
            ref_chosen_logps, ref_rejected_logps,
            model_output["chosen_per_token_logps"], model_output["rejected_per_token_logps"],
            ref_chosen_per_token_logps, ref_rejected_per_token_logps,
            chosen_prefix_mask, chosen_mid_mask, chosen_suffix_mask,
            rejected_prefix_mask, rejected_mid_mask, rejected_suffix_mask,
        )
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        if hasattr(self.args, 'rpo_alpha') and self.args.rpo_alpha is not None and "nll_loss" in model_output:
            losses = losses + self.args.rpo_alpha * model_output["nll_loss"]
            
        if hasattr(self, 'use_weighting') and self.use_weighting and "policy_weights" in model_output:
             losses = losses * model_output["policy_weights"]
        
        if self.aux_loss_enabled and "aux_loss" in model_output : 
             losses = losses + self.aux_loss_coef * model_output["aux_loss"]

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = self.accelerator.gather_for_metrics(chosen_rewards).mean().item()
        metrics[f"{prefix}rewards/rejected"] = self.accelerator.gather_for_metrics(rejected_rewards).mean().item()
        metrics[f"{prefix}rewards/accuracies"] = self.accelerator.gather_for_metrics(reward_accuracies).mean().item()
        metrics[f"{prefix}rewards/margins"] = (self.accelerator.gather_for_metrics(chosen_rewards - rejected_rewards).mean().item())
        metrics[f"{prefix}logps/chosen_total"] = (self.accelerator.gather_for_metrics(model_output["chosen_logps"].sum(dim=-1)).detach().mean().item())
        metrics[f"{prefix}logps/rejected_total"] = (self.accelerator.gather_for_metrics(model_output["rejected_logps"].sum(dim=-1)).detach().mean().item())
        
        if "mean_chosen_logits" in model_output and "mean_rejected_logits" in model_output:
            metrics[f"{prefix}logits/chosen_mean"] = (self.accelerator.gather_for_metrics(model_output["mean_chosen_logits"]).detach().mean().item())
            metrics[f"{prefix}logits/rejected_mean"] = (self.accelerator.gather_for_metrics(model_output["mean_rejected_logits"]).detach().mean().item())
        
        if hasattr(self.args, 'rpo_alpha') and self.args.rpo_alpha is not None and "nll_loss" in model_output:
            metrics[f"{prefix}nll_loss"] = (self.accelerator.gather_for_metrics(model_output["nll_loss"]).detach().mean().item())
            metrics[f"{prefix}nll_loss_mid"] = (self.accelerator.gather_for_metrics(model_output["nll_loss_mid"]).detach().mean().item())
        
        if self.aux_loss_enabled and "aux_loss" in model_output:
            metrics[f"{prefix}aux_loss"] = (self.accelerator.gather_for_metrics(model_output["aux_loss"]).detach().mean().item())

        chosen_mid_avg_logps= self.calculate_mid_avg_logps(model_output["chosen_per_token_logps"], chosen_mid_mask)
        rejected_mid_avg_logps= self.calculate_mid_avg_logps(model_output["rejected_per_token_logps"], rejected_mid_mask)
        ref_chosen_mid_avg_logps= self.calculate_mid_avg_logps(ref_chosen_per_token_logps, chosen_mid_mask)
        ref_rejected_mid_avg_logps= self.calculate_mid_avg_logps(ref_rejected_per_token_logps, rejected_mid_mask)
        prefix_kl_loss = self.calculate_kl_loss(model_output["chosen_per_token_logps"], ref_chosen_per_token_logps, chosen_prefix_mask)
        suffix_kl_loss = self.calculate_kl_loss(model_output["chosen_per_token_logps"], ref_chosen_per_token_logps, chosen_suffix_mask)

        mid_logps_metric = (((chosen_mid_avg_logps - rejected_mid_avg_logps) - (ref_chosen_mid_avg_logps - ref_rejected_mid_avg_logps)).to(self.accelerator.device))

        metrics[f"{prefix}loss_components/prefix_kl_penalty"] = self.accelerator.gather_for_metrics(prefix_kl_loss).mean().item()
        metrics[f"{prefix}loss_components/suffix_kl_penalty"] = self.accelerator.gather_for_metrics(suffix_kl_loss).mean().item()
        metrics[f"{prefix}loss_components/mid_dpo_loss"] = self.accelerator.gather_for_metrics(mid_logps_metric).mean().item()
        return losses.mean(), metrics
    
    def concatenated_forward(self, model: nn.Module, batch: dict[str, Union[list, torch.LongTensor]]):
        num_examples = batch["prompt_input_ids"].shape[0]

        concatenated_batch = self.concatenated_inputs(batch, padding_value=self.padding_value)

        model_kwargs = {}
        if self.aux_loss_enabled:
            model_kwargs["output_router_logits"] = True

        if "pixel_values" in concatenated_batch:
            model_kwargs["pixel_values"] = concatenated_batch["pixel_values"]
        if "pixel_attention_mask" in concatenated_batch:
            model_kwargs["pixel_attention_mask"] = concatenated_batch["pixel_attention_mask"]
        if "image_sizes" in concatenated_batch:
            model_kwargs["image_sizes"] = concatenated_batch["image_sizes"]

        prompt_input_ids = concatenated_batch["prompt_input_ids"]
        prompt_attention_mask = concatenated_batch["prompt_attention_mask"]
        completion_input_ids = concatenated_batch["completion_input_ids"]
        completion_attention_mask = concatenated_batch["completion_attention_mask"]

        chosen_prefix_mask = concatenated_batch["chosen_prefix_mask"]
        chosen_mid_mask = concatenated_batch["chosen_mid_mask"]
        chosen_suffix_mask = concatenated_batch["chosen_suffix_mask"]
        rejected_prefix_mask = concatenated_batch["rejected_prefix_mask"]
        rejected_mid_mask = concatenated_batch["rejected_mid_mask"]
        rejected_suffix_mask = concatenated_batch["rejected_suffix_mask"]

        if self.is_encoder_decoder:
            labels = completion_input_ids
            labels[completion_attention_mask == 0] = self.label_pad_token_id
            outputs = model(
                input_ids=prompt_input_ids,
                attention_mask=prompt_attention_mask,
                labels=labels,
                **model_kwargs,
            )
            logits = outputs.logits
            loss_mask = completion_attention_mask.bool()
        else:
            input_ids = torch.cat((prompt_input_ids, completion_input_ids), dim=1)
            attention_mask = torch.cat((prompt_attention_mask, completion_attention_mask), dim=1)
            loss_mask = torch.cat(
                (torch.zeros_like(prompt_attention_mask), completion_attention_mask),
                dim=1,
            )

            prefix_mask = torch.cat((chosen_prefix_mask, rejected_prefix_mask), dim=0)
            mid_mask = torch.cat((chosen_mid_mask, rejected_mid_mask), dim=0)
            suffix_mask = torch.cat((chosen_suffix_mask, rejected_suffix_mask), dim=0)

            attention_mask, input_ids, loss_mask, prefix_mask, mid_mask, suffix_mask = flush_left(attention_mask, input_ids, loss_mask, prefix_mask, mid_mask, suffix_mask)
            if self.max_length is not None:
                if self.truncation_mode == "keep_end":
                    input_ids = input_ids[:, -self.max_length :]
                    attention_mask = attention_mask[:, -self.max_length :]
                    loss_mask = loss_mask[:, -self.max_length :]
                elif self.truncation_mode == "keep_start":
                    input_ids = input_ids[:, : self.max_length]
                    attention_mask = attention_mask[:, : self.max_length]
                    loss_mask = loss_mask[:, : self.max_length]
                else:
                    raise ValueError(
                        f"Unknown truncation mode: '{self.truncation_mode}'. Should be one of ['keep_end', "
                        "'keep_start']."
                    )

            if self.use_logits_to_keep:
                first_compute_index = loss_mask.nonzero(as_tuple=True)[1].min()
                logits_to_keep = (loss_mask.shape[1] - first_compute_index).item() + 1
                model_kwargs["logits_to_keep"] = logits_to_keep

            if self.padding_free:
                input_ids = input_ids[attention_mask.bool()].unsqueeze(0)
                loss_mask = loss_mask[attention_mask.bool()].unsqueeze(0)
                position_ids = attention_mask.cumsum(1)[attention_mask.bool()].unsqueeze(0) - 1
                model_kwargs["position_ids"] = position_ids
            else:
                model_kwargs["attention_mask"] = attention_mask

            outputs = model(input_ids, **model_kwargs)
            logits = outputs.logits

            labels = torch.roll(input_ids, shifts=-1, dims=1)
            loss_mask = torch.roll(loss_mask, shifts=-1, dims=1).bool()

            if self.use_logits_to_keep:
                labels = labels[:, -logits_to_keep:]
                loss_mask = loss_mask[:, -logits_to_keep:]

        if logits.shape[:2] != labels.shape[:2]:
            seq_len = labels.shape[1]
            logits = logits[:, -seq_len:]

        labels[~loss_mask] = 0
        per_token_logps = selective_log_softmax(logits, labels)
        per_token_logps[~loss_mask] = 0
        per_token_logps = torch.roll(per_token_logps, shifts=1, dims=1)

        if self.padding_free:
            batch_size, seq_len = attention_mask.shape
            per_token_logps_ = torch.zeros(
                batch_size, seq_len, device=outputs.logits.device, dtype=outputs.logits.dtype
            )
            per_token_logps_[attention_mask.bool()] = per_token_logps
            per_token_logps = per_token_logps_

        all_logps = per_token_logps.sum(-1)

        output = {}

        if self.use_weighting:
            with torch.no_grad():
                logprobs = F.log_softmax(logits, dim=-1)
                weights_adjustment_factor = torch.logsumexp(2 * logprobs, dim=-1)
                per_token_logps_adjusted = per_token_logps - weights_adjustment_factor
                all_weights = (per_token_logps_adjusted * loss_mask).sum(-1) / loss_mask.sum(-1)
                chosen_weights = all_weights[:num_examples]
                rejected_weights = all_weights[num_examples:]
                output["policy_weights"] = torch.clamp(torch.exp(chosen_weights + rejected_weights), max=1)

        if self.args.rpo_alpha is not None:
            chosen_logits = logits[:num_examples]
            chosen_labels = labels[:num_examples]
            labels_for_mid_nll = labels[:num_examples].clone()
            labels_for_mid_nll[~mid_mask[:num_examples]] = 0

            output["nll_loss"] = F.cross_entropy(
                torch.flatten(chosen_logits, end_dim=1), torch.flatten(chosen_labels, end_dim=1), ignore_index=0
            )

            output["nll_loss_mid"] = F.cross_entropy(
                torch.flatten(chosen_logits, end_dim=1), torch.flatten(labels_for_mid_nll, end_dim=1), ignore_index=0
            )

        if self.loss_type == "ipo":
            all_logps = all_logps / loss_mask.sum(-1)

        output["chosen_per_token_logps"] = per_token_logps[:num_examples]
        output["rejected_per_token_logps"] = per_token_logps[num_examples:]
        output["chosen_logps"] = all_logps[:num_examples]
        output["rejected_logps"] = all_logps[num_examples:]

        if self.padding_free:
            split_idx = (position_ids == 0).nonzero(as_tuple=True)[1][num_examples]
            mean_chosen_logits = logits[0, :split_idx][loss_mask[0, :split_idx]].mean()
            mean_rejected_logits = logits[0, split_idx:][loss_mask[0, split_idx:]].mean()
        else:
            mean_chosen_logits = logits[:num_examples][loss_mask[:num_examples]].mean()
            mean_rejected_logits = logits[num_examples:][loss_mask[num_examples:]].mean()

        output["mean_chosen_logits"] = mean_chosen_logits
        output["mean_rejected_logits"] = mean_rejected_logits

        if self.aux_loss_enabled:
            output["aux_loss"] = outputs.aux_loss

        output["chosen_prefix_mask"] = prefix_mask[:num_examples]
        output["chosen_mid_mask"] = mid_mask[:num_examples]
        output["chosen_suffix_mask"] = suffix_mask[:num_examples]
        output["rejected_prefix_mask"] = prefix_mask[num_examples:]
        output["rejected_mid_mask"] = mid_mask[num_examples:]
        output["rejected_suffix_mask"] = suffix_mask[num_examples:]

        return output
    
    @staticmethod
    def concatenated_inputs(
        batch: dict[str, Union[list, torch.LongTensor]], padding_value: int
    ) -> dict[str, torch.LongTensor]:
        output = {}

        output["prompt_input_ids"] = torch.cat([batch["prompt_input_ids"], batch["prompt_input_ids"]], dim=0)
        output["prompt_attention_mask"] = torch.cat(
            [batch["prompt_attention_mask"], batch["prompt_attention_mask"]], dim=0
        )
        if "pixel_values" in batch:
            output["pixel_values"] = torch.cat([batch["pixel_values"], batch["pixel_values"]], dim=0)

        if "pixel_attention_mask" in batch:
            output["pixel_attention_mask"] = torch.cat(
                [batch["pixel_attention_mask"], batch["pixel_attention_mask"]], dim=0
            )
        if "image_sizes" in batch:
            output["image_sizes"] = torch.cat([batch["image_sizes"], batch["image_sizes"]], dim=0)

        max_completion_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])
        output["completion_input_ids"] = torch.cat(
            (
                pad_to_length(batch["chosen_input_ids"], max_completion_length, pad_value=padding_value),
                pad_to_length(batch["rejected_input_ids"], max_completion_length, pad_value=padding_value),
            ),
        )
        output["completion_attention_mask"] = torch.cat(
            (
                pad_to_length(batch["chosen_attention_mask"], max_completion_length, pad_value=0),
                pad_to_length(batch["rejected_attention_mask"], max_completion_length, pad_value=0),
            ),
        )

        prompt_mask = torch.zeros(
            (batch["prompt_input_ids"].shape[0],batch["prompt_input_ids"].shape[1]),
            device=batch["prompt_input_ids"].device,
            dtype=torch.bool
        )

        output["chosen_prefix_mask"] = pad_to_length(batch["chosen_prefix_mask"],max_completion_length,pad_value=False)            
        output["chosen_prefix_mask"] = torch.cat((prompt_mask,output["chosen_prefix_mask"]),dim=1)

        output["chosen_mid_mask"] = pad_to_length(batch["chosen_mid_mask"],max_completion_length,pad_value=False)
        output["chosen_mid_mask"] = torch.cat((prompt_mask,output["chosen_mid_mask"]),dim=1)

        output["chosen_suffix_mask"] = pad_to_length(batch["chosen_suffix_mask"],max_completion_length,pad_value=False)
        output["chosen_suffix_mask"] = torch.cat((prompt_mask,output["chosen_suffix_mask"]),dim=1)

        output["rejected_prefix_mask"] = pad_to_length(batch["rejected_prefix_mask"],max_completion_length,pad_value=False)
        output["rejected_prefix_mask"] = torch.cat((prompt_mask,output["rejected_prefix_mask"]),dim=1)

        output["rejected_mid_mask"] = pad_to_length(batch["rejected_mid_mask"],max_completion_length,pad_value=False)
        output["rejected_mid_mask"] = torch.cat((prompt_mask,output["rejected_mid_mask"]),dim=1)

        output["rejected_suffix_mask"] = pad_to_length(batch["rejected_suffix_mask"],max_completion_length,pad_value=False)
        output["rejected_suffix_mask"] = torch.cat((prompt_mask,output["rejected_suffix_mask"]),dim=1)

        return output
    
    def compute_ref_log_probs(self, batch: dict[str, torch.LongTensor]) -> dict:
        device_type = "xpu" if is_torch_xpu_available() else "cuda"
        compte_ref_context_manager = amp.autocast(device_type) if self._peft_has_been_casted_to_bf16 else nullcontext()
        with torch.no_grad(), compte_ref_context_manager:
            if self.ref_model is None:
                with self.null_ref_context():
                    ref_model_output = self.concatenated_forward(self.model, batch)
            else:
                ref_model_output = self.concatenated_forward(self.ref_model, batch)
        return ref_model_output["chosen_logps"], ref_model_output["rejected_logps"], ref_model_output["chosen_per_token_logps"], ref_model_output["rejected_per_token_logps"]