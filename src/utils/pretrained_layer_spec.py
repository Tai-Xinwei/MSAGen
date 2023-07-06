from deepspeed.pipe import LayerSpec
import torch
from deepspeed.utils import logger as ds_logger

class PretrainedLayerSpec(LayerSpec):
    def __init__(self, typename, *module_args, **module_kwargs):
        self.pretrained_ckpt_path = module_kwargs.pop("pretrained_ckpt_path", None)
        super().__init__(typename, *module_args, **module_kwargs)
    
    def build(self, device='cpu', log=False, load=False):
        layer = super().build(log=log)
        if self.pretrained_ckpt_path is not None and load:
            checkpoints_state = torch.load(self.pretrained_ckpt_path, map_location="cpu")
            if type(checkpoints_state) == dict and 'module' in checkpoints_state:
                checkpoints_state = checkpoints_state['module']
            elif type(checkpoints_state) == dict and 'model' in checkpoints_state:
                checkpoints_state = checkpoints_state['model']

            if self.pretrained_ckpt_path.split('/')[-1] == 'model.hybrid_emb.pt' or self.pretrained_ckpt_path.split('/')[-1] == 'layer_37-model_states.pt':
                IncompatibleKeys = layer.load_state_dict(checkpoints_state, strict=False)
            else:
                IncompatibleKeys = layer.load_state_dict(checkpoints_state, strict=False)
            IncompatibleKeys = IncompatibleKeys._asdict()

            if len(IncompatibleKeys['missing_keys']) > 0:
                ds_logger.info("{} Missing keys in {}: {}".format(device, self.pretrained_ckpt_path, IncompatibleKeys['missing_keys']))
            if len(IncompatibleKeys['unexpected_keys']) > 0:
                ds_logger.info("{} Unexpected keys {}: {}".format(device, self.pretrained_ckpt_path, IncompatibleKeys['unexpected_keys']))

            ds_logger.info(f"{device} Loaded from {self.pretrained_ckpt_path}")
        return layer


from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training

class LoraLayerSpec(LayerSpec):
    def __init__(self, typename, ifload=False, lora=True, *module_args, **module_kwargs):
        self.ifload = ifload
        self.pretrained_ckpt_path = module_kwargs.pop("pretrained_ckpt_path", None)
        self.lora = lora
        super().__init__(typename, *module_args, **module_kwargs)
    
    def build(self, device='cpu', log=False, load=False):
        layer = super().build(log=log)
        if self.pretrained_ckpt_path is not None and self.ifload:
            checkpoints_state = torch.load(self.pretrained_ckpt_path, map_location="cpu")
            if type(checkpoints_state) == dict and 'module' in checkpoints_state:
                checkpoints_state = checkpoints_state['module']
            elif type(checkpoints_state) == dict and 'model' in checkpoints_state:
                checkpoints_state = checkpoints_state['model']

            # nl = int(self.pretrained_ckpt_path.split('/')[-1].split('-')[0].split('_')[-1])
            # if nl > 0 and nl < 37:
            #     for key in list(checkpoints_state.keys()):
            #         new_key = "base_model.model." + key
            #         checkpoints_state[new_key] = checkpoints_state.pop(key)

            # if self.pretrained_ckpt_path.split('/')[-1] == 'model.hybrid_emb.pt' or self.pretrained_ckpt_path.split('/')[-1] == 'layer_37-model_states.pt':
            #     layer.load_state_dict(checkpoints_state, strict=False)
            # else:
            #     layer.load_state_dict(checkpoints_state)

            for key in list(checkpoints_state.keys()):
                if key.find('base_model.model') != -1:
                    new_key = key.split("_model.model.")[-1]
                    checkpoints_state[new_key] = checkpoints_state.pop(key) 

            IncompatibleKeys = layer.load_state_dict(checkpoints_state, strict=False)
            IncompatibleKeys = IncompatibleKeys._asdict()

            if len(IncompatibleKeys['missing_keys']) > 0:
                ds_logger.info("{} Missing keys in {}: {}".format(device, self.pretrained_ckpt_path, IncompatibleKeys['missing_keys']))
            if len(IncompatibleKeys['unexpected_keys']) > 0:
                ds_logger.info("{} Unexpected keys {}: {}".format(device, self.pretrained_ckpt_path, IncompatibleKeys['unexpected_keys']))
                
            ds_logger.info(f"{device} Loaded from {self.pretrained_ckpt_path}")

        return self.create_peft_model(layer, self.lora)
    
    def create_peft_model(self, model, lora=True):
        LORA_R = 8
        LORA_ALPHA = 16
        LORA_DROPOUT = 0.1
        if lora:
            TARGET_MODULES = [
                "q_proj",
                "k_proj",
                "v_proj",
                # "down_proj",
                # "gate_proj",
                # "up_proj",
            ]
        else:
            TARGET_MODULES = ["dummy"]
        
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            # task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, lora_config)

        return model