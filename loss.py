import torch
import torch.nn.functional as F
from torch import nn


def triplet_loss(features, labels, margin=1.0):

    loss = 0.0

    for i in range(len(features)):
        anchor = features[i]
        anchor_label = labels[i]

        positive_distances = []
        negative_distances = []

        for j in range(len(features)):
            if i != j:
                other = features[j]
                other_label = labels[j]

                distance = F.pairwise_distance(anchor, other)

                if anchor_label == other_label:
                    positive_distances.append(distance)
                else:
                    negative_distances.append(distance)

        if positive_distances and negative_distances:
            positive_distance = torch.stack(positive_distances).mean()
            negative_distance = torch.stack(negative_distances).mean()

            triplet_loss = torch.clamp(positive_distance - negative_distance + margin, min=0.0)
            loss += triplet_loss

    # loss = torch.tensor(loss,requires_grad=True)
    return loss


def cosine_loss(features, labels):

    batch_size = features.size(0)
    num_samples = features.size(1)
    feature_dim = features.size(2)

    # 构建每对样本的索引对
    pairs = []
    for i in range(num_samples):
        for j in range(num_samples):
            if i != j:
                pairs.append((i, j))

    # 逐对计算余弦嵌入损失
    criterion = nn.CosineEmbeddingLoss()
    total_loss = 0
    for pair in pairs:
        i, j = pair
        input1 = features[:, i, :].view(batch_size, feature_dim)
        input2 = features[:, j, :].view(batch_size, feature_dim)
        target = torch.tensor([1] * batch_size)  # 根据实际情况提供目标标签
        loss = criterion(input1, input2, target)
        total_loss += loss

    avg_loss = total_loss / len(pairs)
    return avg_loss