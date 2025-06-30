import torch


# def KL_div(p_mu, p_sigma, q_mu, q_sigma):
#     div = (
#             torch.log(q_sigma / p_sigma)
#             + (p_sigma ** 2 + (p_mu - q_mu) ** 2) / (2 * q_sigma ** 2)
#             - 0.5
#     )
#
#     div = div.nanmean(dim=-1)
#
#     return div


def KL_div(p_mu, p_logvar, q_mu, q_logvar):
    p_sigma = torch.exp(0.5 * p_logvar)
    q_sigma = torch.exp(0.5 * q_logvar)

    div = (
            torch.log(q_sigma / p_sigma)  # 先计算对数比值
            + (p_sigma ** 2 + (p_mu - q_mu) ** 2) / (2 * q_sigma ** 2)  # 方差部分
            - 0.5  # 常数项
    )

    div = div.mean(dim=-1)

    return div
