import torch
import torch.nn as nn
from transformers import AutoTokenizer, MixtralForCausalLM
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import argparse
import json
import os
import random
import sys

sys.path.append("./")
from moon import FiddlerMoon

class Config:
    """配置参数"""
    model_name = "/home/share/bz/model/Moonlight-16B-A3B-Instruct"  # HF模型名称
    # dataset_path = "/home/share/bz/dataset/merged_dpo_zh_emoji_for_firefly.jsonl"  # 修改为JSON文件路径
    # dataset_type = 'dpo'  # 新增：数据集类型    
    # output_dir = "emo_predictors"  # 输出目录
    # dataset_path = "/home/share/bz/dataset/llava_instruct_150k.json"  # 修改为JSON文件路径
    # dataset_type = 'llava'  # 新增：数据集类型    
    # output_dir = "lla_predictors"  # 输出目录    
    dataset_path = "/home/share/bz/dataset/ShareGPT_V3_unfiltered_cleaned_split.json"  # 修改为JSON文件路径
    dataset_type = 'share'  # 新增：数据集类型    
    output_dir = "expert_predictors"  # 输出目录        
    topk=6
    projection_dim = 256  # 增大降维维度(原128)
    hidden_dim = 128  # 增大隐藏层维度(原64)
    mlp_layers = 3  # 增加MLP层数(原2)
    num_experts = 64  # 保持64专家(n_routed_experts)
    batch_size = 32  
    lr = 0.001  
    epochs = 20  
    max_sequence_length = 163840  # 匹配max_position_embeddings
    max_samples = 1000  
    save_every = 100  
    num_layers = 27  # 保持27层(num_hidden_layers)
def parse_args():
    parser = argparse.ArgumentParser()

    # 训练模式参数
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--train-epochs", type=int, default=20)
    
    # 性能测试模式参数
    parser.add_argument("--model", type=str, default="/home/share/bz/Moonlight-16B-A3B-Instruct")
    parser.add_argument("--cpu-offload", type=int, default=1, choices=[0, 1])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--beam-width", type=int, default=1)
    parser.add_argument("--input-tokens", type=int, nargs="+", default=[512])
    parser.add_argument("--output-tokens", type=int, nargs="+", default=[32])
    parser.add_argument("--n-sample", type=int, default=5)
    
    return parser.parse_args()
def load_dataset(path, dataset_type='sharegpt'):
    """加载数据集，支持多种格式"""
    if dataset_type == 'dpo':
        texts = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                # 使用chosen中的用户提问作为输入
                texts.append(data['chosen'][0]['content'])
        return texts
    elif dataset_type == 'llava':
        texts = []
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                # LLaVA格式通常包含conversations字段
                for conv in item.get('conversations', []):
                    if conv['from'] == 'human':  # 只取用户输入
                        texts.append(conv['value'])
        return texts        
    else:  # 默认sharegpt格式
        with open(path, "r") as f:
            data = json.load(f)
        texts = []
        for d in data:
            if len(d.get("conversations", [])) > 0:
                texts.append(" ".join(d["conversations"][0]["value"].split()))
        return texts

    


class MoEPredictor(nn.Module):
    def __init__(self, layer_idx, config):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx  # 新增：记录层索引
        
        # 动态调整隐藏层维度（中层增加容量）
        # 低层(0-9)：64维，中层(10-25)：128维，高层(26+):64维
        dynamic_hidden_dim = 256 if 8 <= layer_idx <= 20 else 128  # 调整中层范围
        input_dim = config.projection_dim + config.num_experts
        output_dim = config.num_experts
        # 动态调整深度（中层增加层数）
        # 基础2层，中层额外增加2层
        dynamic_mlp_layers = config.mlp_layers + (2 if 8 <= layer_idx <= 20 else 0)  # 调整中层范围
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, config.projection_dim),
            nn.GELU(),
            nn.Dropout(0.4 if layer_idx > 20 else 0.3)  # 高层更强的dropout
        )        

        # 降维层（所有层统一）
        # self.projection = nn.Linear(input_dim, config.projection_dim)
        # self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(0.4 if layer_idx > 25 else 0.3)  # 高层更强的dropout
        
        # MLP结构 - 添加残差连接（提升中层训练稳定性）
        layers = []
        dim_in = config.projection_dim
        
        for i in range(dynamic_mlp_layers):
            # 仅在中层添加残差块
            if 8 <= layer_idx <= 20 and i > 0:
                layers.append(ResidualBlock(dim_in))  # 新增残差块
            else:
                layers.append(nn.Linear(dim_in, dynamic_hidden_dim))
                layers.append(nn.GELU())  # 改用GELU激活
                layers.append(nn.Dropout(0.4 if layer_idx > 20 else 0.3))
                dim_in = dynamic_hidden_dim
        
        # 输出层
        layers.append(nn.Linear(dim_in, config.num_experts))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.input_proj(x)
        return self.mlp(x)

# 新增残差块模块（针对中层）
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(dim, dim)
        )
    
    def forward(self, x):
        return x + self.block(x)  # 残差连接

class MoEDataset(Dataset):
    """MoE数据集"""
    
    def __init__(self, layer_idx, config):
        self.config = config
        self.layer_idx = layer_idx
        current_data = np.load(
            os.path.join(config.output_dir, f"layer_{layer_idx}.npz"), 
            allow_pickle=True
        )
        prev_data = np.load(
            os.path.join(config.output_dir, f"layer_{layer_idx-1}.npz"), 
            allow_pickle=True
        ) if layer_idx > 1 else None
        

        # 加载降维器（训练后保存）
        proj_path = os.path.join(config.output_dir, f"layer_{layer_idx}_projector.pt")
        if os.path.exists(proj_path):
            self.projection = torch.load(proj_path)
        else:
            from sklearn.decomposition import PCA
            self.projection = PCA(n_components=config.projection_dim)
            self.projection.fit(current_data["hidden_states"])
            torch.save(self.projection, proj_path)
      
        
        # 应用降维
        self.current_hidden = self.projection.transform(current_data["hidden_states"])
        self.current_experts = current_data["expert_indices"]
        if prev_data is not None:
            self.prev_hidden = self.projection.transform(prev_data["hidden_states"])
            self.prev_experts = prev_data["expert_indices"]
        else:
            # 第0层没有前一层，用零填充
            self.prev_hidden = np.zeros_like(self.current_hidden)
            self.prev_experts = np.zeros_like(self.current_experts)
            
    
    def __len__(self):
        return len(self.current_hidden)
    
    def __getitem__(self, idx):
        return (
            torch.cat([
                torch.tensor(self.prev_hidden[idx], dtype=torch.float32),
                torch.tensor(self.prev_experts[idx], dtype=torch.float32)
            ]),  # 输入: 前一层hidden_state + 前一层专家激活
            torch.tensor(self.current_experts[idx], dtype=torch.float32)  # 输出: 当前层专家激活
        )

def train_predictors(config):
    """训练每层的预测器"""
    os.makedirs(config.output_dir, exist_ok=True)
    layer_best_val_losses = []
    log_file = os.path.join(config.output_dir, "training_log.txt")    
    # 为每层训练单独的预测器
    for layer_idx in range(config.num_layers):
        layer_file = os.path.join(config.output_dir, f"layer_{layer_idx}.npz")
        
        # 跳过没有数据的层
        if not os.path.exists(layer_file):
            print(f"Skipping layer {layer_idx} - no data")
            continue
            
        print(f"\nTraining predictor for layer {layer_idx}")
        
        # 创建数据集
        dataset = MoEDataset(layer_idx, config)
        
        # 分割数据集
        total_size = len(dataset)
        train_size = int(0.8 * total_size)
        val_size = total_size - train_size
        
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config.batch_size
        )
        
        all_indices = []
        for _, labels in train_dataset:
            # 确保标签转换为numpy数组
            if isinstance(labels, torch.Tensor):
                labels = labels.cpu().numpy()
            all_indices.append(labels)
        if layer_idx > 0:
            prev_layer_file = os.path.join(config.output_dir, f"layer_{layer_idx-1}.npz")
            if not os.path.exists(prev_layer_file):
                print(f"Skipping layer {layer_idx} - previous layer data not found")
                continue        
        expert_counts = np.zeros(config.num_experts, dtype=int)
        
        if len(all_indices) > 0:
            # 处理不同维度的标签
            if all(isinstance(idx, (int, float)) for idx in all_indices):
                indices_array = np.array(all_indices, dtype=int)
            else:
                indices_array = np.concatenate(
                    [np.array(idx).flatten() for idx in all_indices if idx.size > 0]
                ).astype(int)
            
            if indices_array.size > 0:
                counts = np.bincount(indices_array, minlength=config.num_experts)
                expert_counts = counts[:config.num_experts]  # 确保长度匹配
        
        # 安全计算权重
        epsilon = 1e-5
        total = np.sum(config.num_experts / (expert_counts + epsilon))
        expert_weights = torch.tensor(
            config.num_experts / (expert_counts + epsilon) / total,
            dtype=torch.float32
        )


        # 创建模型
        model = MoEPredictor(
            layer_idx=layer_idx,
            config=config
        ).float()
        

        lr = 0.002 if 8 <= layer_idx <= 20 else config.lr
        weight_decay = 1e-3 if layer_idx > 20 else 1e-4
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        # 优化器和损失
        # optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        # criterion = nn.BCEWithLogitsLoss()
        class HybridLoss(nn.Module):
            def __init__(self, alpha=expert_weights, gamma=2.0):
                super().__init__()
                self.bce = nn.BCEWithLogitsLoss(weight=alpha)
                self.gamma = gamma
                
            def forward(self, outputs, targets):
                bce_loss = self.bce(outputs, targets)
                
                # Focal Loss组件（关注难分类样本）
                probas = torch.sigmoid(outputs)
                focal_loss = -targets * (1 - probas).pow(self.gamma) * torch.log(probas + 1e-8)
                focal_loss = focal_loss.mean()
                
                return bce_loss + 0.3 * focal_loss  # 组合损失
        
        criterion = HybridLoss()        
        # 训练循环
        best_val_loss = float("inf")
        train_losses, val_losses = [], []
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs
        )

        patience = 5
        best_epoch = 0        
        for epoch in range(config.epochs):
            # 训练阶段
            model.train()
            epoch_train_loss = 0
            for inputs, labels in train_loader:
                if 8 <= layer_idx <= 20:  # 仅对中层
                    inputs += torch.normal(0, 0.03, inputs.shape)  # 添加3%噪声                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪                
                optimizer.step()
                epoch_train_loss += loss.item() * inputs.size(0)
            scheduler.step()  # 更新学习率            
            epoch_train_loss /= len(train_dataset)
            train_losses.append(epoch_train_loss)
            
            # 验证阶段
            model.eval()
            epoch_val_loss = 0
            with torch.no_grad():
                total_hits = 0
                total_samples = 0
                for inputs, labels in val_loader:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    epoch_val_loss += loss.item() * inputs.size(0)
                    
                    # 计算top-2命中率
                    k = config.topk  # 使用模型配置的专家数量
                    _, pred_topk = torch.topk(outputs, k=k, dim=1)  # [batch_size, k]
                    _, true_topk = torch.topk(labels, k=k, dim=1)   # [batch_size, k]
                    # print("pre",pred_topk)
                    # print("true",true_topk)
                    # 计算交集数量（每个样本预测和真实专家有多少重合）
                    batch_matches = 0
                    for pred, true in zip(pred_topk, true_topk):
                        pred_set = set(pred.tolist())
                        true_set = set(true.tolist())
                        batch_matches += len(pred_set & true_set)  # 当前batch的交集总数
                    
                    # 更新统计
                    total_hits += batch_matches
                    # print("hit",batch_matches)
                    total_samples += len(true_topk)*k
                    # print("total",len(true_topk)*k)

                # 计算并记录命中率
                hit_rate = total_hits / total_samples if total_samples > 0 else 0



            epoch_val_loss /= len(val_dataset)
            # scheduler.step(epoch_val_loss)
            val_losses.append(epoch_val_loss)
            log_msg = f"layer {layer_idx},Epoch {epoch+1}/{config.epochs}: " \
                    f"Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, " \
                    f"Hit Rate: {hit_rate:.2%}" 
            print(log_msg)
            with open(log_file, 'a') as f:
                f.write(log_msg + '\n')
            
            # 保存最佳模型
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_epoch = epoch
                torch.save(model.state_dict(), 
                          os.path.join(config.output_dir, f"layer_{layer_idx}_mlp.pth"))
            elif epoch - best_epoch >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        layer_best_val_losses.append((layer_idx, best_val_loss))
        # 绘制损失曲线
    plt.figure(figsize=(12, 6))
    layers, losses = zip(*sorted(layer_best_val_losses, key=lambda x: x[0]))
    plt.bar(range(len(layers)), losses)
    plt.xticks(range(len(layers)), layers)
    plt.xlabel("Layer Index")
    plt.ylabel("Best Validation Loss")
    plt.title("Best Validation Loss Across Layers")
    for i, v in enumerate(losses):
        plt.text(i, v, f"{v:.4f}", ha='center', va='bottom')
    
    plt.savefig(os.path.join(config.output_dir, "all_layers_val_loss.png"))
    plt.close()


class ExpertPredictor:
    """专家预测器的使用接口"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载模型和分词器
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = FiddlerMoon(config)
        
        # 加载预测器模型
        self.predictors = {}
        self.projections = {}
        
        for layer_idx in range(config.num_layers):
            proj_path = os.path.join(config.output_dir, f"layer_{layer_idx}_projector.pt")
            mlp_path = os.path.join(config.output_dir, f"layer_{layer_idx}_mlp.pth")
            
            if os.path.exists(proj_path) and os.path.exists(mlp_path):
                # 加载降维器
                self.projections[layer_idx] = torch.load(proj_path)
                
                # 加载MLP
                mlp = MoEPredictor(
                    input_dim=config.projection_dim,
                    output_dim=config.num_experts,
                    config=config
                ).to(self.device).float()
                mlp.load_state_dict(torch.load(mlp_path))
                mlp.eval()
                self.predictors[layer_idx] = mlp
    
    def predict_for_layer(self, text, layer_idx):
        """预测给定文本在特定层的专家激活情况"""
        # 分词
        # inputs = self.tokenizer(
        #     text, return_tensors="pt", truncation=True, max_length=config.max_sequence_length
        # ).to(self.device)
        
        # # 获取目标层的hidden state
        # def get_hidden_state_hook(module, inputs, outputs):
        #     hidden_states = inputs[0].detach().cpu().numpy()
        #     self.hidden_states = hidden_states
        self.model.generate(
                    text, output_token=32, input_token=512
                )
        hidden_states=self.model.layer_data[layer_idx]["hidden_states"]       
        # 注册钩子
        # layer = self.model.model.layers[layer_idx]
        # hook = layer.block_sparse_moe.register_forward_hook(get_hidden_state_hook)
        
        # 前向传播
        # with torch.no_grad():
        #     outputs = self.model(**inputs)
        
        # 移除钩子
        # hook.remove()
        
        # 如果没有收集到数据，返回空
        if not hasattr(self, 'hidden_states'):
            return []
        
        # 处理每个token
        results = []
        batch_size, seq_len, _ = self.hidden_states.shape
        
        for b in range(batch_size):
            for s in range(seq_len):
                # 降维
                hidden_state = self.hidden_states[b, s]
                projected = self.projections[layer_idx].transform([hidden_state])[0]
                
                # 预测
                input_tensor = torch.tensor(projected, dtype=torch.float32).to(self.device)
                with torch.no_grad():
                    prediction = self.predictors[layer_idx](input_tensor).sigmoid()
                    prediction = prediction.cpu().numpy()
                
                # 获取top2专家
                top2_indices = np.argsort(prediction[0])[-2:][::-1]
                
                results.append({
                    "token": self.tokenizer.decode(inputs.input_ids[b, s]),
                    "layer": layer_idx,
                    "position": s,
                    "predicted_experts": top2_indices.tolist(),
                    "expert_probabilities": prediction[0].tolist()
                })
        
        return results

if __name__ == "__main__":
    config = Config()
    
    # 步骤1: 收集数据
    args = argparse.Namespace(
        model=config.model_name,
        cpu_offload=1,
        batch_size=config.batch_size,
        beam_width=1
    )
    
    # 读取数据集
    # print("开始通过推理收集训练数据...")
    # texts = load_dataset(config.dataset_path, config.dataset_type)

    # random.seed(0)
    # random.shuffle(texts)
    # model = FiddlerMoon(args)
    # n_sample = 20
    # for input_token in [16]:
    #     for output_token in [16]:
    #         idx_text = 0
    #         batchid = 0
    #         prefill_time_sum, decode_time_sum, hit_rate_sum = 0, 0, 0
    #         for _ in range(n_sample):
    #             batch_texts = []
    #             # 允许重复使用文本直到填满batch_size
    #             while len(batch_texts) < args.batch_size:
    #                 if idx_text >= len(texts):
    #                     idx_text = 0  # 循环使用文本
    #                 text = texts[idx_text]
    #                 idx_text += 1
    #                 if len(text.split()) >= input_token:
    #                     batch_texts.append(text)
    #                     print("batchid", len(batch_texts))
                
    #             # 确保batch_texts长度正确
    #             batch_texts = batch_texts[:args.batch_size]
    #             # print(f"生成的batch_texts内容: {batch_texts}")                        
    #             prefill_time, decode_time, hit_rate, stats = model.generate(
    #                 batch_texts, output_token=output_token, input_token=input_token
    #             )

    
    # # 保存收集到的数据
    # print("保存收集到的数据...")
    # os.makedirs(config.output_dir, exist_ok=True)
    # for layer_idx in model.layer_data:
    #     np.savez_compressed(
    #         os.path.join(config.output_dir, f"layer_{layer_idx}.npz"),
    #         hidden_states=np.array(model.layer_data[layer_idx]["hidden_states"]),
    #         expert_indices=np.array(model.layer_data[layer_idx]["expert_indices"])
    #     )
    # 步骤2: 训练预测器
    train_predictors(config)
    
    # 步骤3: 使用预测器（示例）
    # predictor = ExpertPredictor(config)
    
    # sample_text = "The capital of France is"
    # results = predictor.predict_for_layer(sample_text, layer_idx=0)
    
    # print("\nPredicted expert activations:")
    # for result in results:
    #     print(f"Token: {result['token']}, Position: {result['position']}, "
    #           f"Experts: {result['predicted_experts']}")