def likelihood(X_train, model, device):
    ##########################################################
    # YOUR CODE HERE
    loss = -model.log_prob(X_train).mean()
    ##########################################################

    return loss
