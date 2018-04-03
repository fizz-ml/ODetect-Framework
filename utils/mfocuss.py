import numpy as np

def mfocuss(Phi, Y, reg = 1e-10):
    p = 0.8
    epsilon = 1e-8
    min_gamma = 1e-4
    prune_gamma 1e-3
    max_iters = 800

    [N, M] = Phi.shape
    gamma = ones(M,1)
    keep_list = np.arange(0,M)
    mu = zeros(M,L)

    while(True):
        # remove
        if (min(gamma) < prune_gamma):
            # only keep the indices where gamma is above the prune threshold
            Phi = Phi[:, gamma > prune_gamma]
            keep_list = keep_list[gamma > prune_gamma]
            gamma = gamma[gamma > prune_gamma]
            # When there is no more gammas, stop
            if len(gamma) == 0:
                break;

        # get all combination products
        G = np.matlib.repmat(np.transpose(np.sqrt(gamma)), N, 1)
        # element wise mult
        PhiG = Phi * G
        [U,S,V] = np.linalg.svd(PhiG)

        [d1, d2] = size(S)
        if d > 1:
            diag_S = diag(S)
        else:
            diag_S = S[0]
        templ = np.transpose(diag_S/np.square(diag_S)+sqrt(lambda)+1e-16)
        U_scaled = U*repmat(templ, N, 1)
        Xi = np.transpose(G)*(V*np.transpose(U_scaled))

        mu_old = mu
        mu = Xi*Y

        gamma_old = gamma
        mu2_bar = np.sum(np.square(np.abs(mu)), 2)
        gamma = np.pow((mu2_bar/L), (1-p/2))
        count += 1
        print(np.abs(gamma-gamma_old))
        if count >= max_iters:
            break

    X = zeros(M, L)
    X[keep_list, :] = mu
    return X

N = 1600
M = 320
testPhi = np.arange(0,N) @ np.transpose(np.arange(0,M))
testPhi = np.exp(2*np.pi*testPhi)
