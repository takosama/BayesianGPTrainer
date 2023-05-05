
class BayesianGPTrainer(nn.Module):
    def __init__(self, lambda1=1.0, lambda2=0.1):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def forward(self, input, target, attention_mask=None):
        batch_size, sequence_length, num_classes = input.size()
        log_likelihood = torch.gather(
            input, dim=2, index=target.unsqueeze(2)).squeeze(2)
        log_likelihood = log_likelihood * attention_mask
        log_likelihood = torch.abs_(log_likelihood)

        bayesian_term = (self.lambda1 * log_likelihood).mean()

        entropy = (input.exp() *
                   input).sum(dim=2)*attention_mask
        entropy = entropy.mean()

        entropy_term = self.lambda2 * torch.log(-entropy)

        loss = (bayesian_term + entropy_term)
        return loss
