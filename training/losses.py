import torch.nn.functional as F


def BCELogit_Loss(score_map, labels):
    """ The Binary Cross-Correlation with Logits Loss.
    Args:
        score_map (torch.Tensor): The score map tensor of shape [B,1,H,W]
        labels (torch.Tensor): The label tensor of shape [B,H,W,2] where the
            fourth dimension is separated in two maps, the first indicates whether
            the pixel is negative (0) or positive (1) and the second one whether
            the pixel is positive/negative (1) or neutral (0) in which case it
            will simply be ignored.
    Return:
        loss (scalar torch.Tensor): The BCE Loss with Logits for the score map and labels.
    """
    labels = labels.unsqueeze(1)
    loss = F.binary_cross_entropy_with_logits(score_map, labels[:, :, :, :, 0],
                                              weight=labels[:, :, :, :, 1],
                                              reduction='mean')
    return loss
