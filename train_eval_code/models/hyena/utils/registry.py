optimizer = {
    "adam": "torch.optim.Adam",
    "adamw": "torch.optim.AdamW",
    "rmsprop": "torch.optim.RMSprop",
    "sgd": "torch.optim.SGD",
    "lamb": "models.hyena.utils.optim.lamb.JITLamb",
}

scheduler = {
    "constant": "transformers.get_constant_schedule",
    "plateau": "torch.optim.lr_scheduler.ReduceLROnPlateau",
    "step": "torch.optim.lr_scheduler.StepLR",
    "multistep": "torch.optim.lr_scheduler.MultiStepLR",
    "cosine": "torch.optim.lr_scheduler.CosineAnnealingLR",
    "constant_warmup": "transformers.get_constant_schedule_with_warmup",
    "linear_warmup": "transformers.get_linear_schedule_with_warmup",
    "cosine_warmup": "transformers.get_cosine_schedule_with_warmup",
    "cosine_warmup_timm": "models.hyena.utils.optim.schedulers.TimmCosineLRScheduler",
}

model = {
    # Backbones from this repo
    "model": "models.hyena.sequence.SequenceModel",
    "lm": "models.hyena.sequence.long_conv_lm.ConvLMHeadModel",
    "lm_simple": "models.hyena.sequence.simple_lm.SimpleLMHeadModel",
    "vit_b_16": "src.models.baselines.vit_all.vit_base_patch16_224",
    "dna_embedding": "models.hyena.sequence.dna_embedding.DNAEmbeddingModel",
    "bpnet": "models.hyena.sequence.hyena_bpnet.HyenaBPNet"
}

layer = {
    "id": "models.hyena.sequence.base.SequenceIdentity",
    "ff": "models.hyena.sequence.ff.FF",
    "mha": "models.hyena.sequence.mha.MultiheadAttention",
    "s4d": "models.hyena.sequence.ssm.s4d.S4D",
    "s4_simple": "models.hyena.sequence.ssm.s4_simple.SimpleS4Wrapper",
    "long-conv": "models.hyena.sequence.long_conv.LongConv",
    "h3": "models.hyena.sequence.h3.H3",
    "h3-conv": "models.hyena.sequence.h3_conv.H3Conv",
    "hyena": "models.hyena.sequence.hyena.HyenaOperator",
    "hyena-filter": "models.hyena.sequence.hyena.HyenaFilter",
    "vit": "models.hyena.sequence.mha.VitAttention",
}

callbacks = {
    "timer": "src.callbacks.timer.Timer",
    "params": "src.callbacks.params.ParamsLog",
    "learning_rate_monitor": "pytorch_lightning.callbacks.LearningRateMonitor",
    "model_checkpoint": "pytorch_lightning.callbacks.ModelCheckpoint",
    "early_stopping": "pytorch_lightning.callbacks.EarlyStopping",
    "swa": "pytorch_lightning.callbacks.StochasticWeightAveraging",
    "rich_model_summary": "pytorch_lightning.callbacks.RichModelSummary",
    "rich_progress_bar": "pytorch_lightning.callbacks.RichProgressBar",
    "progressive_resizing": "src.callbacks.progressive_resizing.ProgressiveResizing",
    "seqlen_warmup": "src.callbacks.seqlen_warmup.SeqlenWarmup",
    "seqlen_warmup_reload": "src.callbacks.seqlen_warmup_reload.SeqlenWarmupReload",
    "gpu_affinity": "src.callbacks.gpu_affinity.GpuAffinity"
}

model_state_hook = {
    'load_backbone': 'models.hyena.sequence.long_conv_lm.load_backbone',
}
