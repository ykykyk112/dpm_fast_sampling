import torch
import torch.nn as nn

class DPMSolver(nn.Module) :
    def __init__(self) :
        super(DPMSolver, self).__init__()

    pass

class NoiseSchedule :
    def __init__(
            self,
            schedule = 'discrete',
            betas = None,
            alphas_cumprod = None,
            continuous_beta_0 = 0.1,
            continuous_beta_1 = 20.,
            dtype = torch.float32
    ) :
        self.schedule = schedule

        if schedule == 'discrete' :
            if betas is not None :
                log_alphas = torch.sqrt(1. - betas).cumsum(dim = 0)
            else :
                log_alphas = torch.log(alphas_cumprod)
            self.T = 1.
            self.log_alpha_array = self.numerical_clip_alpha(log_alphas).reshape((1, -1)).to(dtype=dtype)
            self.total_N = self.log_alpha_array.shape[1]
            self.t_array = torch.linspace(0., 1., self.total_N + 1).reshape((1, -1)).to(dtype=dtype)
        else :
            self.T = 1.
            self.total_N = 1000
            self.beta_0 = continuous_beta_0
            self.beta_1 = continuous_beta_1

    def numerical_clip_alpha(self, log_alphas, clipped_lambda = -5.1) :
        # sigma = 1 - alpha^2
        log_sigmas = 0.5 * torch.log(1. - torch.exp(log_alphas * 2.))
        # exp(lambda) = alpha / sigma
        lambdas = log_alphas - log_sigmas
        idx = torch.searchsorted(torch.flip(lambdas, [0]), clipped_lambda)
        if idx > 0 :
            log_alphas = log_alphas[:-idx]
        return log_alphas
    
    def marginal_log_mean_coeff(self, t) :
        # argument t : t-th step in context of alpha_t
        if self.schedule == 'discrete' :
            # t-th step에 해당하는 log_alphas_t를 계산한다.
            return interpolate_fn(t.reshape((-1, 1)), self.t_array.to(t.device), self.log_alpha_array.to(t.device)).reshape((-1))
        elif self.schedule == 'linear' :
            # implementation in the paper, supplementary D.4
            return -0.25 * (self.beta_1 - self.beta_0) * (t ** 2) - self.beta_0 * t * 0.5

    def marginal_alpha(self, t) :
        # exp(log_alphas) -> alphas
        return torch.exp(self.marginal_log_mean_coeff(t))
    
    def marginal_std(self, t) :
        # standard deviation = sigma
        # sigma ** 2 = 1 - alphas ** 2
        return torch.sqrt(1. - torch.exp(self.marginal_log_mean_coeff(t) * 2.))

    def marginal_lambda(self, t) :
        # log_alphas - log_sigma
        log_mean_coeff = self.marginal_log_mean_coeff(t)
        log_sigma = self.marginal_std(t)
        return log_mean_coeff - log_sigma
    
    def inverse_lambda(self, lamb) :
        # get time t from lambda
        if self.schedule == 'linear':
            tmp = 2. * (self.beta_1 - self.beta_0) * torch.logaddexp(-2. * lamb, torch.zeros((1,)).to(lamb))
            Delta = self.beta_0**2 + tmp
            return tmp / (torch.sqrt(Delta) + self.beta_0) / (self.beta_1 - self.beta_0)
        elif self.schedule == 'discrete':
            log_alpha = -0.5 * torch.logaddexp(torch.zeros((1,)).to(lamb.device), -2. * lamb)
            t = interpolate_fn(log_alpha.reshape((-1, 1)), torch.flip(self.log_alpha_array.to(lamb.device), [1]), torch.flip(self.t_array.to(lamb.device), [1]))
            return t.reshape((-1,))
        
class DPM_Solver :
    def __init__(
            self,
            model_fn,
            noise_schedule,
            correcting_x0_fn = None,
            correcting_xt_fn = None,
            thresholding_max_val = 1.,
            dynamic_thresholding_ratio = 0.995,
    ) :
        self.model = lambda x, t : model_fn(x, t.expand((x.shape[0])))
        self.noise_schedule = noise_schedule
        self.algorithm_type = 'dpmsolver'

    def noise_prediction_fn(self, x, t) :
        return self.model(x, t)
    
    def model_fn(self, x, t) :
        return self.noise_prediction_fn(x, t)
    
    def get_time_steps(self, skip_type, t_T, t_0, N, device) :
        if skip_type == 'logSNR' :
            lambda_T = self.noise_schedule.marginal_lambda(torch.tensor(t_T).to(device))
            lambda_0 = self.noise_schedule.marginal_lambda(torch.tensor(t_0).to(device))
            logSNR_steps = torch.linspace(lambda_T.cpu().item(), lambda_0.cpu().item(), N + 1).to(device)
            return self.noise_schedule.inverse_lambda(logSNR_steps)
        elif skip_type == 'time_uniform' :
            return torch.linspace(t_T, t_0, N + 1).to(device)
        elif skip_type == 'time_quadratic' :
            t_order = 2
            t = torch.linspace(t_T ** (1. / t_order), t_0 ** (1. / t_order), N + 1).pow(t_order).to(device)
            return t
        else :
            raise ValueError('Unsupported skip_type {}'.format(skip_type))
        
    

def interpolate_fn(x, xp, yp):
    """
    A piecewise linear function y = f(x), using xp and yp as keypoints.
    We implement f(x) in a differentiable way (i.e. applicable for autograd).
    The function f(x) is well-defined for all x-axis. (For x beyond the bounds of xp, we use the outmost points of xp to define the linear function.)

    Args:
        x: PyTorch tensor with shape [N, C], where N is the batch size, C is the number of channels (we use C = 1 for DPM-Solver).
        xp: PyTorch tensor with shape [C, K], where K is the number of keypoints.
        yp: PyTorch tensor with shape [C, K].
    Returns:
        The function values f(x), with shape [N, C].
    """
    N, K = x.shape[0], xp.shape[1]
    all_x = torch.cat([x.unsqueeze(2), xp.unsqueeze(0).repeat((N, 1, 1))], dim=2)
    sorted_all_x, x_indices = torch.sort(all_x, dim=2)
    x_idx = torch.argmin(x_indices, dim=2)
    cand_start_idx = x_idx - 1
    start_idx = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(1, device=x.device),
        torch.where(
            torch.eq(x_idx, K), torch.tensor(K - 2, device=x.device), cand_start_idx,
        ),
    )
    end_idx = torch.where(torch.eq(start_idx, cand_start_idx), start_idx + 2, start_idx + 1)
    start_x = torch.gather(sorted_all_x, dim=2, index=start_idx.unsqueeze(2)).squeeze(2)
    end_x = torch.gather(sorted_all_x, dim=2, index=end_idx.unsqueeze(2)).squeeze(2)
    start_idx2 = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(0, device=x.device),
        torch.where(
            torch.eq(x_idx, K), torch.tensor(K - 2, device=x.device), cand_start_idx,
        ),
    )
    y_positions_expanded = yp.unsqueeze(0).expand(N, -1, -1)
    start_y = torch.gather(y_positions_expanded, dim=2, index=start_idx2.unsqueeze(2)).squeeze(2)
    end_y = torch.gather(y_positions_expanded, dim=2, index=(start_idx2 + 1).unsqueeze(2)).squeeze(2)
    cand = start_y + (x - start_x) * (end_y - start_y) / (end_x - start_x)
    return cand