functions{

    int[] idx(int[] row, int[] idx_info){
        return segment(idx_info, row[2], row[1]);
    }

    matrix coordinates(int[] idxs, matrix coords){
        matrix[size(idxs), 3] selected;
        for(i in 1:size(idxs)){
            selected[i] = coords[idxs[i]];
        }
        return selected;
    }

        // Given theta AND distance matrices
    // return covariance function
    matrix compute_C(
        real a, real c, real kappa, real sig_sq, // cov params theta
        matrix spatdist, matrix tempdist // spatial and temporal data
    ){
        int n = dims(spatdist)[1];
        int m = dims(spatdist)[2];
        matrix[n, m] aup1;
        matrix[n, m] aup1pow;
        matrix[n, m] C;

        aup1 = a*tempdist+1;

        for(i in 1:n){
            for(j in 1:m){
                aup1pow[i][j] = pow(aup1[i][j], kappa/2.0);
            }
        }
        C = (sig_sq ./ aup1) .* exp( (-c * spatdist) ./ (aup1pow));
        return C;
    }


    matrix get_distance(matrix coords){
        int n = dims(coords)[1];
        matrix[n, n] dist = rep_matrix(0, n, n);
        for(i in 1:n){
            for(j in i:n){
                dist[i, j] = distance(coords[i], coords[j]);
                dist[j, i] = dist[i, j];
            }
        }
        return dist;
    }

    vector get_w(vector w, int[] idx_pi){
        vector[size(idx_pi)] w_pi;
        for(i in 1:size(idx_pi)){
            w_pi[i] = w[idx_pi[i]];
        }
        return w_pi;
    }

    matrix get_HwR(real a, real c, real kappa, real sig_sq,
                int i, int z,
                vector w,
                int K, int[,] ptts_train, int[] ptts_idx, int[,] pptts_train, int[] pptts_idx, matrix coords){

        int ptt[3] = ptts_train[i];
        int pptt[3] = pptts_train[(i-1) * K + z];

        int n_i = ptt[1];
        int n_pi = pptt[1];

        matrix[n_i, 1 + n_i] HwR = rep_matrix(0, n_i, 1 + n_i);

        matrix[n_i, 3] coords_i = coordinates(idx(ptt, ptts_idx), coords);

        if(n_pi==0){
            matrix[n_i, n_i] spatdist = get_distance(block(coords_i, 1, 1, n_i, 2));
            matrix[n_i, n_i] tempdist = get_distance(block(coords_i, 1, 3, n_i, 1));

            matrix[n_i, n_i] R = compute_C(a, c, kappa, sig_sq, spatdist, tempdist);

            vector[n_i] Hw = rep_vector(0, n_i);
            HwR = append_col(Hw, R);
        }
        else{
            int idx_pi[n_pi] = idx(pptt, pptts_idx);
            matrix[n_pi, 3] coords_pi = coordinates(idx_pi, coords);
            matrix[n_i+n_pi, 3] coords_ipi = append_row(coords_i, coords_pi);
            matrix[n_i+n_pi, n_i+n_pi] spatdist = get_distance(block(coords_ipi, 1, 1, n_i+n_pi, 2));
            matrix[n_i+n_pi, n_i+n_pi] tempdist = get_distance(block(coords_ipi, 1, 3, n_i+n_pi, 1));

            matrix[n_i+n_pi, n_i+n_pi] C = compute_C(a, c, kappa, sig_sq, spatdist, tempdist);

            matrix[n_i, n_i] Ci = block(C, 1, 1, n_i, n_i);
            matrix[n_pi, n_pi] Cj = block(C, n_i+1, n_i+1, n_pi, n_pi);
            matrix[n_i, n_pi] Cij = block(C, 1, n_i+1, n_i, n_pi);

            matrix[n_i, n_pi] H = Cij * inverse(Cj);
            matrix[n_i, n_i] R = Ci - H * Cij';

            vector[n_pi] w_pi = get_w(w, idx_pi);

            HwR = append_col(H * w_pi, R);
        }
        return HwR;
    }

}

data{
    int n;
    int n_train;
    int n_test;
    int n_pptts_test;
    int n_pptts_train;

    int p;
    int M;
    int K;

    vector[n_train] y;
    matrix[n, p] X;
    matrix[n, 3] coords;

    int ptts_train[M, 3];
    int pptts_train[M * K, 3];

    int ptts_test[M, 3];
    int pptts_test[M * K, 3];

    int ptts_idx[n_train];
    int pptts_idx[n_pptts_train];

    int ptts_test_idx[n_test];
    int pptts_test_idx[n_pptts_test];
}
transformed data{
    real alpha = 1.0/K;
}

parameters{
    vector[p] beta;
    real<lower=0> tau_sq;

    real<lower=0> a;
    real<lower=0> c;
    real<lower=0,upper=1> kappa;
    real<lower=0> sig_sq;

    simplex[K] pi[M];
    vector[n_train] w;
}

transformed parameters{
    vector[K] lp[M];

    for(i in 1:M){
        for(z in 1:K){
            int n_i = ptts_train[i, 1];
            int idxs[n_i] = idx(ptts_train[i], ptts_idx);
            matrix[n_i, 1 + n_i] HwR;
            vector[n_i] Hw;
            matrix[n_i, n_i] R;

            if(pi[i, z]==0){
                continue;
            }
            HwR = get_HwR(a, c, kappa, sig_sq, i, z, w, K, ptts_train, ptts_idx, pptts_train, pptts_idx, coords);

            Hw = col(HwR, 1);
            R = block(HwR, 1, 2, n_i, n_i);
            lp[i, z] = log(pi[i, z]) + multi_normal_lpdf(get_w(w, idxs) | Hw, R);
        }
    }
}

model{
    a ~ uniform(0, 1000);
    c ~ uniform(0, 1000);
    kappa ~ uniform(0, 1);
    sig_sq ~ inv_gamma(2, 1);

    beta ~ normal(0,  100);
    tau_sq ~ inv_gamma(2, 0.1);

    y ~ normal(X[1:n_train] * beta + w, tau_sq);

    for(i in 1:M){
        pi[i] ~ dirichlet(rep_vector(alpha, K));
        target += log_sum_exp(lp[i]);
    }
}

generated quantities {
  vector[n_test] y_rep;
  {
  vector[n_test] w_rep = rep_vector(0, n_test);

    for(i in 1:M){
        for(z in 1:K){
            int n_i = ptts_test[i, 1];
            int idxs[n_i] = idx(ptts_test[i], ptts_test_idx);
            matrix[n_i, 1 + n_i] HwR;
            vector[n_i] Hw;
            matrix[n_i, n_i] R;
            vector[n_i] tmp;

            if(pi[i, z]==0){
                continue;
            }
            HwR = get_HwR(a, c, kappa, sig_sq, i, z, w, K, ptts_test, ptts_test_idx, pptts_test, pptts_test_idx, coords);

            Hw = col(HwR, 1);
            R = block(HwR, 1, 2, n_i, n_i);

            tmp = multi_normal_rng(Hw, R) * pi[i, z];

            for(j in 1:n_i){
                w_rep[idxs[j]-n_train] += tmp[j];
            }

        }
    }
  for(i in 1:n_test){
    y_rep[i] = normal_rng(X[n_train+i]*beta + w_rep[i], tau_sq);
  }
  }
  print("GEN");
}
