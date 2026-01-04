import torch
from torch import nn

class CPSVD_Linear(nn.Module):
    """
    一个实现了分段SVD因式分解的nn.Linear替代层。
    [重构后版本]: 只需 col_indices 即可确定 svd_indices。
    """
    def __init__(self, in_features: int, out_features: int, rank: int, 
                 col_indices: list, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        # --- 核心改动：动态计算 svd_indices ---
        # 1. 将 col_indices 转为集合以提高查找效率
        col_indices_set = set(col_indices)
        
        # 2. 计算 svd_indices (所有索引中的补集)
        # 为了保证顺序一致性，我们对结果进行排序
        all_indices = set(range(in_features))
        svd_indices_list = sorted(list(all_indices - col_indices_set))
        
        if len(svd_indices_list) + len(col_indices_set) != in_features:
            raise ValueError("col_indices 和计算出的 svd_indices 之和与 in_features 不匹配！")

        # 3. 将索引注册为 buffer
        self.register_buffer('svd_indices', torch.tensor(svd_indices_list, dtype=torch.long))
        self.register_buffer('col_indices', torch.tensor(sorted(col_indices), dtype=torch.long)) # 同样排序以保证一致性

        self.is_pure_svd = len(self.col_indices) == 0

        self.v_proj = nn.Linear(len(self.svd_indices), rank, bias=False)
        self.u_proj = nn.Linear(rank, out_features, bias=bias)

        if not self.is_pure_svd:
            self.col_proj = nn.Linear(len(self.col_indices), out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # forward 方法无需任何改动
        if self.is_pure_svd:
            return self.u_proj(self.v_proj(x))
        else:
            x_svd = x.index_select(-1, self.svd_indices)
            x_col = x.index_select(-1, self.col_indices)
            svd_output = self.u_proj(self.v_proj(x_svd))
            col_output = self.col_proj(x_col)
            return svd_output + col_output

    def __repr__(self):
        return (f"{self.__class__.__name__}(in_features={self.in_features}, "
                f"out_features={self.out_features}, rank={self.rank}, "
                f"mode={'pure_svd' if self.is_pure_svd else 'mixed'})")

    def get_config(self) -> dict:
        """
        [实例方法] 返回一个可序列化为JSON的配置字典。
        现在不再需要保存 svd_indices。
        """
        return {
            "layer_type": "CPSVD_Linear",
            "in_features": self.in_features,
            "out_features": self.out_features,
            "rank": self.rank,
            # svd_indices 已移除
            "col_indices": self.col_indices.cpu().tolist(),
            "bias": self.u_proj.bias is not None
        }

    @classmethod
    def from_config(cls, config: dict):
        """
        [类方法] 从配置字典创建一个新的 CPSVD_Linear 实例。
        现在它会自动计算 svd_indices。
        """
        if config.get("layer_type") != "CPSVD_Linear":
            raise ValueError("配置字典类型不匹配 CPSVD_Linear")

        # 调用新的 __init__，它会自动处理 svd_indices 的计算
        return cls(
            in_features=config["in_features"],
            out_features=config["out_features"],
            rank=config["rank"],
            col_indices=config["col_indices"], # 只需传递 col_indices
            bias=config.get("bias", False)
        )
    

from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

# 假设 _get_parent_module_and_name 和 replace_linear_with_svd 函数已定义

def save_model_with_svd_config(model, path):
    """
    保存模型，通过调用 get_config() 将 CPSVD_Linear 的配置写入 config.json。
    """
    svd_config = {}
    for name, module in model.named_modules():
        if isinstance(module, CPSVD_Linear):
            # 直接调用模块的 get_config 方法
            svd_config[name] = module.get_config()
    
    if not svd_config:
        print("警告: 模型中未找到 CPSVD_Linear 层，将按标准方式保存。")
    else:
        model.config.svd_config = svd_config
        model.config.model_type = "cpsvd_llama"
        print(f"SVD配置已生成并附加到 model.config.svd_config")
    
    model.save_pretrained(path)
    print(f"模型已保存至 {path}")

def load_model_with_svd(path, device_map='auto'):
    """
    加载模型，并根据 config.json 中的SVD配置，
    通过调用 from_config() 重建 CPSVD_Linear 层。
    """
    model = AutoModelForCausalLM.from_pretrained(path, device_map=device_map, torch_dtype=torch.float16, trust_remote_code=True, cache_dir=None)
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    # if not hasattr(model.config, 'svd_config'):
    #     print("信息：在 model.config 中未找到 'svd_config'，返回原始模型。")
    #     return model
        
    # print("标准模型已加载，现在开始重建 SVD 层...")
    # svd_config = model.config.svd_config

    # for name, config in svd_config.items():
    #     # 直接通过类方法从配置创建新模块
    #     new_svd_layer = CPSVD_Linear.from_config(config)
        
    #     parent_module, module_name = get_parent_module_and_name(model, name)
    #     setattr(parent_module, module_name, new_svd_layer)
    #     print(f"  - 正在重建: {name}")
        
    # print("SVD 层重建完成！")
    return model, tokenizer

import torch.nn as nn
from typing import Tuple

def get_parent_module_and_name(model: nn.Module, name: str) -> Tuple[nn.Module, str]:
    """
    根据模块的完整名称（例如 'model.layers.0.self_attn.q_proj'）,
    返回其直接父模块和它在父模块中的名称。

    Args:
        model (nn.Module): 顶层模型。
        name (str): 目标模块的点分路径字符串。

    Returns:
        Tuple[nn.Module, str]: 一个元组，包含父模块和目标模块的名称。
    
    Example:
        >>> model = nn.Sequential(nn.Linear(10, 20), nn.ReLU())
        >>> parent, name = _get_parent_module_and_name(model, '0')
        >>> print(parent)
        Sequential(
          (0): Linear(in_features=10, out_features=20, bias=True)
          (1): ReLU()
        )
        >>> print(name)
        '0'
        >>> getattr(parent, name) # This gives the Linear layer itself
    """
    if '.' not in name:
        return model, name

    # 使用 rsplit('.', 1) 从右边分割一次，效率更高
    # 'layers.0.self_attn.q_proj' -> ('layers.0.self_attn', 'q_proj')
    parent_name, module_name = name.rsplit('.', 1)
    
    # 遍历模型以找到父模块
    parent_module = model
    for part in parent_name.split('.'):
        parent_module = getattr(parent_module, part)
        
    return parent_module, module_name

from transformers import LlamaConfig, LlamaForCausalLM, AutoConfig, AutoModelForCausalLM

class CPSVDLlamaConfig(LlamaConfig):
    """
    Custom config class. The model_type attribute is the key that links this
    config to the custom model class during loading.
    """
    model_type = "cpsvd_llama"

class CPSVDLlamaForCausalLM(LlamaForCausalLM):
    """
    Custom model class.
    """
    config_class = CPSVDLlamaConfig

    def __init__(self, config):
        # 1. First, initialize the standard LlamaForCausalLM. 
        #    This builds the model with regular nn.Linear layers.
        super().__init__(config)

        # 2. Check if the config has our custom SVD information.
        #    This is the "surgery" step: we replace the nn.Linear layers
        #    with our CPSVD_Linear layers *during initialization*.
        if hasattr(config, "svd_config"):
            print("Found SVD config. Rebuilding model with CPSVD_Linear layers...")
            for name, layer_cfg in config.svd_config.items():
                # Create the custom layer from its specific configuration
                new_layer = CPSVD_Linear.from_config(layer_cfg)
                
                # Find the parent of the layer to be replaced and set the new layer
                parent_module, module_name = get_parent_module_and_name(self, name)
                setattr(parent_module, module_name, new_layer)
            print("Model rebuilt successfully.")

from transformers import Qwen3ForCausalLM, Qwen3Config

class CPSVDQwen3Config(Qwen3Config):
    model_type = "cpsvd_qwen3"

class CPSVDQwen3ForCausalLM(Qwen3ForCausalLM):
    config_class = CPSVDQwen3Config

    def __init__(self, config):

        super().__init__(config)

        if hasattr(config, "svd_config"):
            print("Found SVD config. Rebuilding model with CPSVD_Linear layers...")
            for name, layer_cfg in config.svd_config.items():
                # Create the custom layer from its specific configuration
                new_layer = CPSVD_Linear.from_config(layer_cfg)
                
                # Find the parent of the layer to be replaced and set the new layer
                parent_module, module_name = get_parent_module_and_name(self, name)
                setattr(parent_module, module_name, new_layer)
            print("Model rebuilt successfully.")

from transformers import MistralConfig, MistralForCausalLM

class CPSVDMistralConfig(MistralConfig):
    model_type = 'cpsvd_mistral'

class CPSVDMistralForCausalLM(MistralForCausalLM):
    config_class =  CPSVDMistralConfig

    def __init__(self, config):

        super().__init__(config)

        if hasattr(config, "svd_config"):
            print("Found SVD config. Rebuilding model with CPSVD_Linear layers...")
            for name, layer_cfg in config.svd_config.items():
                # Create the custom layer from its specific configuration
                new_layer = CPSVD_Linear.from_config(layer_cfg)
                
                # Find the parent of the layer to be replaced and set the new layer
                parent_module, module_name = get_parent_module_and_name(self, name)
                setattr(parent_module, module_name, new_layer)
            print("Model rebuilt successfully.")   


AutoConfig.register(CPSVDLlamaConfig.model_type, CPSVDLlamaConfig)
AutoModelForCausalLM.register(CPSVDLlamaConfig, CPSVDLlamaForCausalLM)

AutoConfig.register(CPSVDQwen3Config.model_type, CPSVDQwen3Config)
AutoModelForCausalLM.register(CPSVDQwen3Config, CPSVDQwen3ForCausalLM)

AutoConfig.register(CPSVDMistralConfig.model_type, CPSVDMistralConfig)
AutoModelForCausalLM.register(CPSVDMistralConfig, CPSVDMistralForCausalLM)