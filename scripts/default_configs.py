"""
    DEFAULT CONFIGS BLOCK -- start
        Do not make any new changes to this block because it has to be used for future references
"""
accelerator = "gpu"
devices = [0]
precision = "16-mixed"
debugger = False
models = ["clipseg", "cris", "biomedclipseg", "biomedclipseg_d"]
freeze_encoder = False

models_configs = {
    "clipseg": {"batch_size": 128, "lr": 0.002},
    "cris": {"batch_size": 32, "lr": 0.00002},
}
non_rad_prompts = [f"p{i}" for i in range(10)]
chexlocalze_prompts = [f"p{i}" for i in range(7)]
camus_prompts = [f"p{i}" for i in range(8)]
busi_prompts = [f"p{i}" for i in range(7)]

dataset_prompts = {
    "kvasir_polyp": non_rad_prompts,
    "bkai_polyp": non_rad_prompts,
    "clinicdb_polyp": non_rad_prompts,
    "isic": non_rad_prompts,
    "dfu": non_rad_prompts,
    "camus": camus_prompts,
    "busi": busi_prompts,
    "chexlocalize": chexlocalze_prompts,
    "pooled_polyp": non_rad_prompts,
    "pooled_all": ["random"]
}

"""
    DEFAULT CONFIGS BLOCK -- end
"""
