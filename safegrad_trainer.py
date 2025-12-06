from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
import torch
from packaging import version
from transformers import Trainer
from transformers import logging
import torch.nn.functional as F
import transformers

if version.parse(torch.__version__) >= version.parse("1.6"):
    from torch.cuda.amp import autocast

logger = logging.get_logger(__name__)
IGNORE_INDEX = -100

class SafeGrad_KLD_Trainer(Trainer):

    def __init__(self, *args, projection=True, ref_model=None, ref_model_name_or_path: Optional[str] = None, **kwargs):
        super().__init__(*args, **kwargs)

        if ref_model is None and ref_model_name_or_path is None:
            raise ValueError("Either 'ref_model' or 'ref_model_name_or_path' must be provided for KLD Trainer.")

        if ref_model:
            print("Creating a deepcopy of the provided reference model to ensure independence.")
            self.ref_model = ref_model
        else:
            self.ref_model = transformers.AutoModelForCausalLM.from_pretrained(
                ref_model_name_or_path,
                load_in_8bit=False,
                torch_dtype=torch.float16 if self.args.fp16 else (torch.bfloat16 if self.args.bf16 else torch.float32),
                device_map="auto",
            )

        self.ref_model.to(self.accelerator.device)
        self.ref_model.eval()
        self.projection = projection

        for param in self.ref_model.parameters():
            param.requires_grad = False

        print("SafeGrad_KLD_Trainer initialized: Alignment is based on output distribution (KL Divergence).")


    def get_alignment_dataloader(self, alignment_dataset) -> DataLoader:
        """
        Builds DataLoader for alignment dataset.
        """
       
        from transformers.trainer_utils import (
            seed_worker
        )
        from torch.utils.data import DataLoader, RandomSampler
        data_collator = self.data_collator
        sampler = RandomSampler(alignment_dataset)
        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }
        if not isinstance(alignment_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = sampler
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
        return self.accelerator.prepare(DataLoader(alignment_dataset, **dataloader_params))

    def init(self, alignment_dataset):
        self.clock = 0
        self.steps = 0
        if self.args.guide_data_num > 0:
            self.alignment_dataloader = self.get_alignment_dataloader(alignment_dataset)
            self.data_iter = iter(self.alignment_dataloader)

    def sample_from_alignment(self):
        try:
            batch = next(self.data_iter)
        except (StopIteration):
            self.data_iter = iter(self.alignment_dataloader)
            batch = next(self.data_iter)
        return batch

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()

        # ----- Step 1: Compute user task gradient (g_task) -----
        finetune_inputs = self._prepare_inputs(inputs)
        with self.compute_loss_context_manager():
            loss_task = self.compute_loss(model, finetune_inputs, return_outputs=False)
        self.accelerator.backward(loss_task)
        g_task = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
        model.zero_grad()

        # ----- Step 2: Compute alignment gradient (g_align) via KL divergence -----
        alignment_inputs = self.sample_from_alignment()
        alignment_inputs = self._prepare_inputs(alignment_inputs)

        with torch.no_grad():
            # Reference model logits (P_ref)
            ref_outputs = self.ref_model(**alignment_inputs)
            logits_ref = ref_outputs.logits
        
        # Current model logits (P_theta)
        finetune_outputs = model(**alignment_inputs)
        logits_theta = finetune_outputs.logits
        

        with torch.no_grad():
            p_ref = F.softmax(logits_ref, dim=-1)
        # Convert to (log) probability distributions
        log_p_theta = F.log_softmax(logits_theta, dim=-1)
        
        # KL(P_theta || P_ref): log_target=False indicates p_ref is normal probs
        loss_align_total = F.kl_div(log_p_theta, p_ref, reduction='sum', log_target=False)
        num_valid_tokens = (alignment_inputs["labels"] != -100).sum()

        # Manually calculate the average loss
        if num_valid_tokens > 0:
            loss_align = loss_align_total / num_valid_tokens
        else:
            loss_align = torch.tensor(0.0, device=logits_theta.device, dtype=logits_theta.dtype)

        self.accelerator.backward(loss_align)
        g_align = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
        model.zero_grad()

        # ----- Step 3: Global projection and gradient merge (DiGraP) -----
        global_dot_product = torch.tensor(0.0, device=self.accelerator.device)
        global_align_grad_norm_sq = torch.tensor(0.0, device=self.accelerator.device)
        projection_scalar = torch.tensor(0.0, device=self.accelerator.device)

        if self.projection:
            for name in g_task:
                if name in g_align:
                    # Convert the gradient to float32 for accumulation to prevent
                    # precision loss or overflow during the accumulation process.
                    global_dot_product += torch.sum(g_task[name].to(torch.float32) * g_align[name].to(torch.float32))
                    global_align_grad_norm_sq += torch.sum(g_align[name].to(torch.float32) * g_align[name].to(torch.float32))

            # If conflict detected (dot product < 0), compute projection scalar
            if global_dot_product < 0:
                if global_align_grad_norm_sq > 1e-9: 
                    projection_scalar = global_dot_product / global_align_grad_norm_sq

        for name, param in model.named_parameters():
            if name in g_task and name in g_align:
                projected_task_grad = g_task[name] - projection_scalar.to(g_task[name].dtype) * g_align[name]
                final_grad = projected_task_grad + self.args.rho * g_align[name]
                param.grad = final_grad
            elif name in g_task:
                param.grad = g_task[name]


        logs = {
            "loss_task": loss_task.item(),
            "loss_align_kld": loss_align.item(), 
            "global_dot_product": global_dot_product.item(),
            "projection_scalar": projection_scalar.item(),
        }
        self.log(logs)

        return loss_task.detach() / self.args.gradient_accumulation_steps
    
