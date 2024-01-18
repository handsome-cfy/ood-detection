import torch
import torch.nn.functional as F


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

    loss = torch.tensor(loss,requires_grad=True)
    return loss
