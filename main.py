

from typing import Dict, Tuple, Optional

import torch, os, math
import torch.nn as nn
import torch.nn.functional as F
import freerec # 推荐系统框架 FreeRec

from optimizers.Adam import AdamSEvo
from optimizers.AdamW import AdamWSEvo
from optimizers.utils import Smoother

freerec.declare(version='1.0.1')

cfg = freerec.parser.Parser()
cfg.add_argument("--embedding-dim", type=int, default=64)
cfg.add_argument("--num-layers", type=int, default=3, help="the number of layers for FSC/BSC")

# 多模态输入文件和参数
cfg.add_argument("--mfiles", type=str, default="textual_modality.pkl,visual_modality.pkl", help="the files saving modality")
cfg.add_argument("--num-neighbors", type=str, default='5-1', help="for kNN graph") # 每种模态的k近邻数
cfg.add_argument("--gamma", type=float, default=0.2) # 平滑参数 γ

cfg.set_defaults(
    description="STAIR",
    root="../../data",
    dataset='Amazon2014Baby_550_MMRec',
    epochs=500,
    batch_size=1024,
    optimizer='adamwsevo',
    lr=1e-3,
    weight_decay=0.1,
    seed=1
)
cfg.compile()

# 分割多模态文件名字符串为列表
cfg.mfiles = cfg.mfiles.split(',')
# 将 “5-1” 解析成整数列表 [5, 1]
cfg.num_neighbors = list(map(int, cfg.num_neighbors.split('-')))

# beta3 是 BSC 层平滑衰减参数，用于控制不同维度的加权
cfg.beta3 = (0.1 + 0.9 * (torch.arange(cfg.embedding_dim) / cfg.embedding_dim).pow(cfg.gamma)).to(cfg.device)


# ===============================
# 模型定义：STAIR 模型
# ===============================
class STAIR(freerec.models.GenRecArch):

    def __init__(
        self, dataset: freerec.data.datasets.RecDataSet
    ) -> None:
        super().__init__(dataset)

        # 设置图卷积层数
        self.num_layers = cfg.num_layers

        # 用户嵌入矩阵初始化
        self.User.add_module(
            "embeddings", nn.Embedding(
                self.User.count, cfg.embedding_dim
            )
        )

        # 物品嵌入矩阵初始化
        self.Item.add_module(
            "embeddings", nn.Embedding(
                self.Item.count, cfg.embedding_dim
            )
        )

        # 注册图的归一化邻接矩阵 (用户-物品二分图)
        self.register_buffer(
            "Adj",
            self.dataset.train().to_normalized_adj(
                normalization='sym'
            )
        )

        # 重置模型参数
        self.reset_parameters()

        # 准备多模态数据
        self.prepare(dataset.path)

        # 定义 BPR 损失函数
        self.criterion = freerec.criterions.BPRLoss(reduction='mean')

    def reset_parameters(self):
        # 遍历所有模块初始化参数
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 线性层使用Kaiming正态初始化
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Embedding):
                # 嵌入层使用正态分布初始化
                nn.init.normal_(m.weight, std=1.e-4)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                # 批归一化层初始化
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)

    def marked_params(self):
        # 返回带平滑器的参数分组（实现BSC）
        params = [
            {
                'params': self.User.parameters(), # 用户参数
                'smoother': None # 不使用平滑器
            },
            {
                'params': self.Item.parameters(),  # 物品参数
                'smoother': Smoother(self.mAdj, beta=cfg.beta3, L=cfg.num_layers, aggr='neumann') # 使用Neumann平滑器
            },
        ]
        return params

    def whitening(self, feats: torch.Tensor):
        # 多模态嵌入初始化：特征白化处理：去均值+SVD降维
        feats = feats - feats.mean(0, keepdim=True)
        feats, _, _ = torch.linalg.svd(feats, full_matrices=False)
        # 缩放特征
        return feats[:, :cfg.embedding_dim] * math.sqrt(self.Item.count / cfg.embedding_dim)

    def get_knn_graph(self, features: torch.Tensor, k: int = 5):
        r"""
        计算k近邻图
        """
        features = F.normalize(features, dim=-1) # L2归一化特征 (N, D)
        sim = features @ features.t() # 计算相似度矩阵 (N, N)
        sim.fill_diagonal_(-10.) # 将对角线设为最小值，避免自连接
        # 获取k近邻图
        edge_index, _ = freerec.graph.get_knn_graph(
            sim, k, symmetric=False
        )
        return edge_index

    def prepare(self, path: str):
        from freerec.utils import import_pickle

        # 多模态嵌入初始化：加载多模态特征数据
        mfeats = [
            import_pickle(
                os.path.join(path, mfile)
            )
            for mfile in cfg.mfiles
        ]

        # 构建多模态k近邻图（用于BSC）
        edge_index = torch.cat(
            [self.get_knn_graph(feats, k) for feats, k in zip(mfeats, cfg.num_neighbors)],
            dim=1 # 拼接不同模态的边
        )
        edge_weight = torch.ones_like(edge_index[0], dtype=torch.float) # 初始化边权重为1
        # 合并重复边
        edge_index, edge_weight = freerec.graph.coalesce(
            edge_index, edge_weight, reduce='sum'
        )
        # 转换为无向图
        edge_index, edge_weight = freerec.graph.to_undirected(
            edge_index, edge_weight, reduce='max'
        )
        # 归一化邻接矩阵
        edge_index, edge_weight = freerec.graph.to_normalized(
            edge_index, edge_weight,
            normalization='sym'
        )
        # 创建多模态邻接矩阵
        mAdj = torch.sparse_coo_tensor(
            edge_index, edge_weight,
            size=(self.Item.count, self.Item.count)
        )
        # 注册多模态邻接矩阵缓冲区
        self.register_buffer(
            'mAdj',
            mAdj.to_sparse_csr() # 转换为CSR格式
        )

        # 多模态嵌入初始化：多模态特征融合（MI）
        mfeats = [self.whitening(mfeat) * k for mfeat, k in zip(mfeats, cfg.num_neighbors)] # 白化并加权
        mfeats = sum(mfeats).div(sum(cfg.num_neighbors)) # 加权平均
        self.Item.embeddings.weight.data.copy_(mfeats)

        # 获取用户-物品交互图
        edge_index = self.dataset.train().to_bigraph(edge_type='u2i')['u2i'].edge_index
        # 左归一化
        edge_index, edge_weight = freerec.graph.to_normalized(edge_index, normalization='left')
        # 创建交互矩阵R
        R = torch.sparse_coo_tensor(
            edge_index, edge_weight, size=(self.User.count, self.Item.count)
        ).to_sparse_csr()

        # 多模态嵌入初始化：用交互矩阵初始化用户嵌入
        self.User.embeddings.weight.data.copy_(R @ mfeats)

    def sure_trainpipe(self, batch_size: int):
        # 创建训练数据管道
        return self.dataset.train().shuffled_pairs_source(
        ).gen_train_sampling_neg_( # 生成负采样
            num_negatives=1 # 每个正样本采样1个负样本
        ).batch_(batch_size).tensor_() # 分批并转换为张量

    def encode(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # 拼接用户和物品嵌入
        allEmbds = torch.cat(
            (self.User.embeddings.weight, self.Item.embeddings.weight), dim=0
        ) # (N, D)

        features = allEmbds
        smoothed = allEmbds
        
        # FSC（特征平滑卷积）前向传播
        beta = 1 - cfg.beta3 # beta3 是维度相关的权重向量
        norm_correction = 1 - beta ** (self.num_layers + 1) # 归一化修正项
        for _ in range(self.num_layers):
            features = self.Adj @ features * beta # 图卷积
            smoothed = smoothed + features # 累积平滑特征
        # 计算平均嵌入
        avgEmbds = smoothed.mul(1 - beta).div(norm_correction)
        # 分割为用户和物品嵌入
        userEmbds, itemEmbds = torch.split(
            avgEmbds, (self.User.count, self.Item.count)
        )
        return userEmbds, itemEmbds

    def fit(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        # 前向传播获取嵌入
        userEmbds, itemEmbds = self.encode()
        # 提取用户、正样本、负样本
        users, positives, negatives = data[self.User], data[self.Item], data[self.INeg]
        userEmbds = userEmbds[users] # (B, 1, D)
        iposEmbds = itemEmbds[positives] # (B, 1, D)
        inegEmbds = itemEmbds[negatives] # (B, K, D)

        # 计算BPR损失
        rec_loss = self.criterion(
            torch.einsum("BKD,BKD->BK", userEmbds, iposEmbds),
            torch.einsum("BKD,BKD->BK", userEmbds, inegEmbds)
        )
        return rec_loss

    def reset_ranking_buffers(self):
        """评估前执行：重置排名缓冲区"""
        userEmbds, itemEmbds = self.encode()
        self.ranking_buffer = dict()
        # 缓存用户和物品嵌入用于评估
        self.ranking_buffer[self.User] = userEmbds.detach().clone()
        self.ranking_buffer[self.Item] = itemEmbds.detach().clone()

    def recommend_from_full(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        # 全量推荐：计算用户与所有物品的得分
        userEmbds = self.ranking_buffer[self.User][data[self.User]] # (B, 1, D)
        itemEmbds = self.ranking_buffer[self.Item]
        return torch.einsum("BKD,ND->BN", userEmbds, itemEmbds)

    def recommend_from_pool(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        # 池化推荐：计算用户与候选物品的得分
        userEmbds = self.ranking_buffer[self.User][data[self.User]] # (B, 1, D)
        itemEmbds = self.ranking_buffer[self.Item][data[self.IUnseen]] # (B, 101, D)
        return torch.einsum("BKD,BKD->BK", userEmbds, itemEmbds)


class CoachForSTAIR(freerec.launcher.Coach):
    """STAIR模型的训练教练类"""

    def set_optimizer(self):
        # 根据配置选择优化器
        if self.cfg.optimizer.lower() == 'sgd':
            self.optimizer = torch.optim.SGD(
                self.model.marked_params(), lr=self.cfg.lr, 
                momentum=self.cfg.momentum,
                nesterov=self.cfg.nesterov,
                weight_decay=self.cfg.weight_decay
            )
        elif self.cfg.optimizer.lower() == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.marked_params(), lr=self.cfg.lr,
                betas=(self.cfg.beta1, self.cfg.beta2),
                weight_decay=self.cfg.weight_decay
            )
        elif self.cfg.optimizer.lower() == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.model.marked_params(), lr=self.cfg.lr,
                betas=(self.cfg.beta1, self.cfg.beta2),
                weight_decay=self.cfg.weight_decay
            )
        elif self.cfg.optimizer.lower() == 'adamsevo':
            self.optimizer = AdamSEvo(
                self.model.marked_params(), lr=self.cfg.lr,
                betas=(self.cfg.beta1, self.cfg.beta2),
                weight_decay=self.cfg.weight_decay
            )
        elif self.cfg.optimizer.lower() == 'adamwsevo':
            self.optimizer = AdamWSEvo(
                self.model.marked_params(), lr=self.cfg.lr,
                betas=(self.cfg.beta1, self.cfg.beta2),
                weight_decay=self.cfg.weight_decay
            )
        else:
            raise NotImplementedError(
                f"Unexpected optimizer {self.cfg.optimizer} ..."
            )

    def train_per_epoch(self, epoch: int):
        # 每个训练周期的迭代
        for data in self.dataloader:
            data = self.dict_to_device(data) # 数据转移到设备
            loss = self.model(data) # 前向传播计算损失

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 监控损失
            self.monitor(
                loss.item(), 
                n=len(data[self.User]), reduction="mean", 
                mode='train', pool=['LOSS']
            )


def main():
    """主函数：模型训练流程"""

    try:
        # 尝试通过名称获取数据集
        dataset = getattr(freerec.data.datasets, cfg.dataset)(root=cfg.root)
    except AttributeError:
        # 回退到通用数据集加载
        dataset = freerec.data.datasets.RecDataSet(cfg.root, cfg.dataset, tasktag=cfg.tasktag)

    # 初始化模型
    model = STAIR(dataset)

    # 创建数据管道
    trainpipe = model.sure_trainpipe(cfg.batch_size)
    validpipe = model.sure_validpipe(cfg.ranking)
    testpipe = model.sure_testpipe(cfg.ranking)

    # 初始化训练教练
    coach = CoachForSTAIR(
        dataset=dataset,
        trainpipe=trainpipe,
        validpipe=validpipe,
        testpipe=testpipe,
        model=model,
        cfg=cfg
    )
    # 开始训练
    coach.fit()


if __name__ == "__main__":
    main()