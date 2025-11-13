# src/Model/train_llm.py
#!/usr/bin/env python3
"""
Supervised fine-tuning with DECISION outputs (no quoting) + label masking.
Saves/evaluates every 200 steps; optimized for A100/L40S (bf16 + TF32).
Adds HF token support for gated repos (use HF_TOKEN env var).
"""
import os, sys, torch
from pathlib import Path
from math import ceil
from time import time
from tqdm import tqdm

try:
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer,
        TrainingArguments, Trainer, DataCollatorForLanguageModeling,
        BitsAndBytesConfig, TrainerCallback, TrainerState, TrainerControl
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from datasets import load_dataset
except ImportError as e:
    print(f"[ERROR] {e}\nInstall: pip install transformers peft datasets bitsandbytes accelerate")
    sys.exit(1)

from llm_config import LLMConfig, LightweightConfig

# ---- HF gated repo support ----
HF_TOKEN = os.getenv("HF_TOKEN")
hf_kwargs = {"token": HF_TOKEN} if HF_TOKEN else {}

# ---------------- progress callback ----------------
class _TP:
    def __init__(self): self.t=time(); self.s=0; self.r=0.0

class ProgressCallback(TrainerCallback):
    def __init__(self, total_steps:int, tokens_per_step:int, log_path:Path):
        self.total_steps=total_steps; self.tps=tokens_per_step
        self.tp=_TP(); self.pbar=None
        self.log_path=log_path; log_path.parent.mkdir(parents=True, exist_ok=True)
    def _log(self,line): open(self.log_path,"a",encoding="utf-8").write(line.rstrip()+"\n")
    def on_train_begin(self, args, state, control, **kw):
        self.pbar=tqdm(total=self.total_steps,desc="TRAIN",leave=True,ncols=100)
        self.tp.t=time(); self._log("[START] Training")
    def on_log(self, args, state, control, **kw):
        if not state.log_history: return
        rec=state.log_history[-1]; step=state.global_step; now=time()
        dt=max(1e-9,now-self.tp.t); ds=max(0,step-self.tp.s)
        if ds>0: self.tp.r=(ds*self.tps)/dt; self.tp.t=now; self.tp.s=step
        loss=rec.get("loss",rec.get("train_loss")); lr=rec.get("learning_rate")
        remain=max(0,self.total_steps-step); eta=remain*self.tps/max(1e-9,self.tp.r)
        m=int(eta//60); s=int(eta%60)
        self.pbar.n=min(step,self.total_steps)
        status=f"step={step}/{self.total_steps}"
        if loss is not None: status+=f"  loss={loss:.4f}"
        if lr   is not None: status+=f"  lr={lr:.2e}"
        status+=f"  tok/s={self.tp.r:,.0f}  ETA={m:02d}:{s:02d}"
        self.pbar.set_postfix_str(status[:60]); self.pbar.refresh(); self._log(f"[LOG] {status}")
    def on_save(self, args, state, control, **kw): self._log(f"[SAVE] checkpoint {state.global_step}")
    def on_train_end(self, args, state, control, **kw):
        self.pbar.n=min(self.total_steps,state.global_step); self.pbar.close(); self._log("[DONE] Training finished")

# ---------------- trainer ----------------
class ECUTuningTrainer:
    def __init__(self, cfg: LLMConfig):
        self.cfg=cfg
        self.device="cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Device: {self.device}")
        if self.device=="cuda":
            print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
            print(f"[INFO] Mem: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
        self.tok=None; self.model=None

    def load_model_and_tokenizer(self):
        print(f"\n[LOAD] Base: {self.cfg.base_model}")
        bnb=None
        if self.cfg.use_4bit:
            print("[LOAD] QLoRA 4-bit")
            bnb=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if self.device=="cuda" else torch.float32,
                bnb_4bit_use_double_quant=True
            )
        elif self.cfg.use_8bit:
            print("[LOAD] 8-bit"); bnb=BitsAndBytesConfig(load_in_8bit=True)

        print("[LOAD] Tokenizer…")
        self.tok = AutoTokenizer.from_pretrained(
            self.cfg.base_model,
            trust_remote_code=True,
            padding_side="right",
            **hf_kwargs
        )
        if self.tok.pad_token is None: self.tok.pad_token = self.tok.eos_token

        print("[LOAD] Weights…")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.base_model,
            quantization_config=bnb,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if self.device=="cuda" else torch.float32,
            **hf_kwargs
        )
        if bnb is not None:
            print("[LOAD] Prepare k-bit training…")
            self.model = prepare_model_for_kbit_training(self.model)
        print("[OK] Model+Tokenizer loaded")

    def apply_lora(self):
        if not self.cfg.use_lora: return
        print("\n[SETUP] LoRA…")
        lora=LoraConfig(
            r=self.cfg.lora_r, lora_alpha=self.cfg.lora_alpha, lora_dropout=self.cfg.lora_dropout,
            target_modules=self.cfg.lora_target_modules or ["q_proj","k_proj","v_proj","o_proj"],
            bias="none", task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(self.model, lora)
        self.model.print_trainable_parameters()
        print("[OK] LoRA applied")

    # ----- DECISION template + label masking -----
    def _format_example(self, ex:dict)->dict:
        prompt = f"""{self.cfg.system_prompt}

### Instruction:
{ex['instruction']}

### Context:
{ex.get('context','')}

### Response:
"""
        resp = ex.get("response","").strip()
        decision = ex.get("decision") or "Apply conservative bounded deltas based on logs."
        why      = ex.get("why") or "Observed behavior suggests conservative adjustment."
        safety   = ex.get("safety") or "Respect octane timing caps, lambda floors at high load, and WGDC limits."
        target = f"Decision: {decision}\nWhy: {why}\nSafety: {safety}\n"
        if resp:
            target += f"\nSummary: {resp[:600]}\n"
        target += (self.tok.eos_token or "</s>")
        return {"prompt": prompt, "target": target}

    def _tokenize_mask_supervised(self, batch):
        prompts=batch["prompt"]; targets=batch["target"]
        p_tok = self.tok(prompts, truncation=True, max_length=self.cfg.max_seq_length, padding=False, add_special_tokens=True)
        with self.tok.as_target_tokenizer():
            t_tok = self.tok(targets, truncation=True, max_length=self.cfg.max_seq_length, padding=False, add_special_tokens=False)
        input_ids=[]; attn=[]; labels=[]
        for pi, ti in zip(p_tok["input_ids"], t_tok["input_ids"]):
            ids = pi + ti
            am  = [1] * len(ids)
            lb  = [-100]*len(pi) + ti      # mask prompt tokens => no loss on prompt
            input_ids.append(ids[:self.cfg.max_seq_length])
            attn.append(am[:self.cfg.max_seq_length])
            labels.append(lb[:self.cfg.max_seq_length])
        return {"input_ids":input_ids,"attention_mask":attn,"labels":labels}

    def load_training_data(self):
        if not self.cfg.train_data_path.exists():
            print(f"[ERROR] Missing train file: {self.cfg.train_data_path}"); sys.exit(1)
        ds = load_dataset("json", data_files={"train":str(self.cfg.train_data_path),
                                              "validation":str(self.cfg.val_data_path)})
        print(f"[DATA] Train={len(ds['train'])}  Val={len(ds['validation'])}")
        ds = ds.map(self._format_example)
        ds = ds.map(self._tokenize_mask_supervised, batched=True, remove_columns=ds["train"].column_names)
        return ds

    def _estimate(self, args, n:int):
        world=max(1,getattr(args,"world_size",1)); mbs=args.per_device_train_batch_size; ga=args.gradient_accumulation_steps
        steps_per_epoch=ceil(n/(mbs*world)); total=steps_per_epoch*int(args.num_train_epochs)
        if ga>1: total=ceil(total/ga)
        tokens_per_step=self.cfg.max_seq_length*(mbs*world)
        return total, tokens_per_step

    def train(self):
        print("\n[TRAIN] Starting…")
        args = TrainingArguments(
            output_dir=str(self.cfg.output_dir),
            num_train_epochs=self.cfg.num_epochs,
            per_device_train_batch_size=self.cfg.batch_size,
            per_device_eval_batch_size=self.cfg.batch_size,
            gradient_accumulation_steps=self.cfg.gradient_accumulation_steps,
            learning_rate=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
            lr_scheduler_type="cosine",
            warmup_ratio=0.08,
            max_grad_norm=1.0,

            logging_dir=str(self.cfg.logs_dir),
            logging_steps=20,
            evaluation_strategy="steps",
            eval_steps=200,
            save_strategy="steps",
            save_steps=200,            # every 200 steps
            save_total_limit=3,

            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,

            bf16=(self.device=="cuda"),
            fp16=False,
            tf32=True,
            gradient_checkpointing=True,
            report_to=[],
            optim="paged_adamw_8bit" if self.cfg.use_4bit else "adamw_torch",
        )

        ds = self.load_training_data()
        collator = DataCollatorForLanguageModeling(tokenizer=self.tok, mlm=False)
        trainer = Trainer(model=self.model, args=args,
                          train_dataset=ds["train"], eval_dataset=ds["validation"],
                          data_collator=collator)

        total, tps = self._estimate(args, len(ds["train"]))
        log_path = self.cfg.logs_dir / "training_progress.log"
        trainer.add_callback(ProgressCallback(total, tps, log_path))
        print(f"[INFO] Progress log -> {log_path}")

        trainer.train()
        print("\n[OK] Training complete")

        print(f"[SAVE] -> {self.cfg.output_dir}")
        trainer.save_model(); self.tok.save_pretrained(self.cfg.output_dir)

    def run(self):
        self.load_model_and_tokenizer()
        self.apply_lora()
        self.train()

def main():
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--lightweight", action="store_true")
    p.add_argument("--model", type=str)
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    a=p.parse_args()

    cfg = LightweightConfig() if a.lightweight else LLMConfig()
    if a.model: cfg.base_model=a.model
    if a.epochs: cfg.num_epochs=a.epochs
    if a.batch_size: cfg.batch_size=a.batch_size

    ECUTuningTrainer(cfg).run()

if __name__=="__main__":
    main()
