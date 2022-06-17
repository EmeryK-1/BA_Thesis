data{
    int n;
    int n_train;
    int n_test;

    int p;

    vector[n_train] y;
    matrix[n, p] X;
}

parameters{
    vector[p] beta;
    real<lower=0> tau_sq;
}

model{
    beta ~ normal(0,  100);
    tau_sq ~ inv_gamma(2, 0.1);

    y ~ normal(X[1:n_train] * beta, tau_sq);
}

generated quantities {
  vector[n_test] y_rep;
  for(i in 1:n_test){
    y_rep[i] = normal_rng(X[n_train+i]*beta, tau_sq);
  }
}
