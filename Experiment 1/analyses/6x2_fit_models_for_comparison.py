import numpy as np
import pymc as pm  # if using pymc >=4
import arviz as az
import matplotlib.pyplot as plt
import json
from pytensor.tensor import cumsum
import pytensor.tensor as at
from glob import glob
import mvEEG.mvEEG as mvEEG
from os import path
import argparse


def fit_models():
    parser = argparse.ArgumentParser(description="Fit models for comparison")
    parser.add_argument(
        "--metric", type=str, default="signedDprime", help="Distance metric to use for model comparison"
    )
    parser.add_argument("--overwrite", action="store_true", help="Whether to overwrite existing results")
    parser.add_argument(
        "--prior_sd", type=float, default=0.25, help="Standard deviation of the priors for the model parameters"
    )
    args = parser.parse_args()

    METRIC = args.metric
    if METRIC == "signedCorrWhite":
        RESCALE = 1e6  # rescaling factor for RDMs to improve numerical stability in Bayesian models
    else:
        RESCALE = 1  # if not rescaling, then set to 1 (i.e., no rescaling)
    OVERWRITE = args.overwrite
    PRIOR_SD = args.prior_sd

    print(f"Fitting models for comparison with prior_sd={PRIOR_SD}")

    # set up other constants and variables
    conditions = [
        "A2N",
        "A4N",
        "A6N",
        "A8N",
        "A2P",
        "A4P",
        "A6P",
        "A8P",
        "M2N",
        "M4N",
        "M6N",
        "M8N",
        "M2P",
        "M4P",
        "M6P",
        "M8P",
    ]

    ridx, cidx = np.triu_indices(len(conditions), k=1)

    ROOT = "../data/derivatives"
    DATA_DIR = ROOT
    EXPERIMENT_NAME = "banyan"

    wrangl = mvEEG.Wrangler(
        data_dir=DATA_DIR,
        experiment_name=EXPERIMENT_NAME,
        dropped_chans={},
        t_win=50,
        t_step=25,
        trim_timepoints=[-350, 1000],
        trial_bin_size=15,
        n_folds=2,  # NOTE: _not_ using wrangler n_folds here
    )

    delay_window = (wrangl.t > 300) & (
        wrangl.t < 900
    )  # time window to average distances over, only includes when probes could appear

    with open("rejected_subs.json", "r") as f:
        rejected_subs = json.load(f)

    # get the RDM files for each subject
    rdm_files = sorted(glob(f"../data/derivatives/*/RDMs/*desc-{METRIC}*npy"))
    rdm_files = [f for f in rdm_files if f.split("-")[1].split("/")[0] not in rejected_subs]

    # build up the design matrix for the models
    # Number of Items in WM
    WM_setsizes = np.array([int(cond[1]) if (cond[0] == "M") else 0 for cond in conditions])
    WMss_options = sorted(np.unique(WM_setsizes))
    WM_setsize_map = {ss: i for i, ss in enumerate(WMss_options)}
    cond_WMss_idx = np.array([WM_setsize_map[ss] for ss in WM_setsizes])

    # Number of attended areas, assuming that people can filter our distractors
    condition_ATTN = np.array([int(cond[1]) for cond in conditions])
    ATTNss_options = sorted(np.unique(condition_ATTN))
    ATTN_setsize_map = {ss: i for i, ss in enumerate(ATTNss_options)}
    cond_ATTNss_idx = np.array([ATTN_setsize_map[ss] for ss in condition_ATTN])

    # stimulus energy, carved up for a monotonic function
    condition_stimEnergy = np.array([int(cond[1]) if cond[2] == "N" else 10 for cond in conditions])
    stimEnergy_options = sorted(np.unique(condition_stimEnergy))
    stimEnergy_map = {ss: i for i, ss in enumerate(stimEnergy_options)}
    cond_stimEnergy_idx = np.array([stimEnergy_map[ss] for ss in condition_stimEnergy])

    q = np.array(
        [
            cond_WMss_idx[ridx],
            cond_WMss_idx[cidx],
            cond_ATTNss_idx[ridx],
            cond_ATTNss_idx[cidx],
            cond_stimEnergy_idx[ridx],
            cond_stimEnergy_idx[cidx],
        ]
    ).T

    # build up distances and inputs
    for i, f in enumerate(rdm_files):
        sub_rdms = np.load(f)
        sub_rdm_delay = sub_rdms[..., delay_window].mean(-1)

        sub_dists = sub_rdm_delay[ridx, cidx] * RESCALE  # vectorize upper triangle, rescale for numerical stability

        sub_q = np.concatenate(
            [
                q,
                np.array([[i] * q.shape[0]]).T,  # add column to index subjects
            ],
            axis=1,
        )

        if i == 0:
            pop_dists = sub_dists
            pop_q = sub_q
        else:
            pop_dists = np.concatenate((pop_dists, sub_dists))
            pop_q = np.concatenate((pop_q, sub_q), axis=0)

    # MODEL DEFINITIONS

    # possible bends (SS) for each function. The last option (8) corresponds to the full model with fewest contraints on the shape of the function
    # every bend is going to be compared via Savage-Dickey density ratio to a model with one fewer bend.
    WM_possible_bends = [4, 6, 8]
    WM_SS_to_idx = {0: 0, 2: 1, 4: 2, 6: 3, 8: 4}

    def build_model(d, q, WM_bend=8):
        """
        Build a model to estimate the WM and spatial attention functions with the following assumptions:
        1. The WM function is a guassian random walk, **is the same with or without placeholders**
        2. The spatial attention function is monotonic, **and is the same with or without placeholders**
        3. The stimulus energy function is monotonic

        **s indicate assumptions specific to this model

        Arguments
        ---------
        d : array of shape (n_distances,)
            The vectorized distances in upper (or lower) triangle of RDM
        q : array of shape (n_distances, n_regressors)
            Condition indices
        For this model the following column structure is expected:
        0. WM set size at first condition
        1. WM set size at second condition
        2. Spatial attention set size at first condition
        3. Spatial attention set size at second condition
        4. Stimulus energy at first condition
        5. Stimulus energy at second condition
        6. unique subject index for each distance (i.e., same for all distances from the same subject)

        """
        N = len(d)
        wmK = np.unique(q[:, :2]).size  # number of WM set sizes
        attnK = np.unique(q[:, 2:4]).size  # number of attended areas
        StimEnergyK = np.unique(q[:, 4:6]).size  # number of stimulus energy levels
        nsub = np.unique(q[:, 6]).size  # number of subjects

        # figure out where the plateau/bend occurs in the indices for each function, which determines how many values we need to estimate before the plateau
        # adding 1 to include the bend point itself. For example, if the bend is at SS4, then we need to estimate values for SS1, SS2, SS3, and SS4 (i.e., 4 values), and then the function plateaus (repeats) after that.
        wmMax = WM_SS_to_idx[WM_bend] + 1

        # check that the quad indices are in the expected format
        assert np.min(q) == 0
        assert q.shape[0] == N
        assert (np.max(q[:, :2]) + 1) == wmK
        assert (np.max(q[:, 2:4]) + 1) == attnK
        assert (np.max(q[:, 4:6]) + 1) == StimEnergyK

        # steps between conditions we'd like to fit to data
        coords = {
            "innovations_num_WM": np.arange(1, wmK),
            "innovations_num_Attn": np.arange(1, attnK),
            "innovations_num_SE": np.arange(1, StimEnergyK),
            # for population-level functions, which can plateau after the bend
            "innovations_num_WM_POP": np.arange(1, wmMax),
            "subject": np.arange(0, nsub),
            "trial_num": np.arange(1, N + 1),
        }

        with pm.Model(coords=coords) as model:

            # prior on noise in RDM values, used in likelihood below
            RDM_sigma_pop = pm.Exponential("RDM_sigma_pop", lam=5)  # OLD lam=10
            RDM_sigma_subs = pm.Exponential("RDM_sigma_subs", lam=1 / RDM_sigma_pop, dims=("subject",))

            # ~~~~~~~~~~~~ 1. Working Memory ~~~~~~~~~~~~

            # only assumption is that the values are greater than 0.
            # population-level random values
            mu_WM_to_bend = pm.TruncatedNormal(
                "mu_WM_to_bend", mu=0.5, sigma=PRIOR_SD, lower=0, dims=("innovations_num_WM_POP",)
            )  # population-level function values at each set size (except first, which is fixed to 0)

            # get the final innovation, which is the difference between the last two values before the bend
            # to look at savage dickey later
            mu_WM_innov_final = pm.Deterministic("mu_WM_innov_final", mu_WM_to_bend[-1] - mu_WM_to_bend[-2])
            # concatenate the last value back on to produce the full function, which plateaus after the bend
            mu_WM = pm.Deterministic(
                "mu_WM",
                at.concatenate((mu_WM_to_bend, at.repeat(mu_WM_to_bend[-1:], wmK - wmMax)), axis=0),
                dims=("innovations_num_WM",),
            )

            sigma_lvls_prior_WM = pm.Exponential(
                "sigma_lvls_hyperprior_WM", lam=4
            )  # prior on variability of the subjects at each set size - setting it broad to include very small values
            sigma_lvls_WM = pm.Exponential("sigma_lvls_WM", scale=sigma_lvls_prior_WM, dims=("innovations_num_WM",))

            sub_vals_WM = pm.TruncatedNormal(
                "sub_vals_WM_minus_first",
                mu=mu_WM,
                sigma=sigma_lvls_WM,
                lower=0,
                dims=("subject", "innovations_num_WM"),
            )
            sub_vals_WM = pm.Deterministic("sub_vals_WM", at.concatenate((np.zeros((nsub, 1)), sub_vals_WM), axis=1))

            # ~~~~~~~~~~~~ 2. Spatial Attention ~~~~~~~~~~~~
            # define the population-level dirichlet prior
            pop_dir_Attn = pm.Dirichlet("pop_dir_Attn", a=np.ones(attnK - 1), dims=("innovations_num_Attn",))
            pop_dir_scale_Attn = pm.Exponential(
                "pop_dir_scale_Attn", lam=5
            )  # scale of dirichlet concentration, higher = less variance
            pop_concentration_Attn = pop_dir_scale_Attn * pop_dir_Attn

            # each subject's dirichlet is drawn from a scaled version of the population dirichlet
            sub_dir_Attn = pm.Dirichlet(
                "sub_dir_Attn",
                a=pop_concentration_Attn,
                dims=("subject", "innovations_num_Attn"),
            )
            sub_mu_Attn = pm.Deterministic("sub_mu_Attn_minus_first", cumsum(sub_dir_Attn, axis=1))
            sub_mu_Attn = pm.Deterministic("sub_mu_Attn", at.concatenate((np.zeros((nsub, 1)), sub_mu_Attn), axis=1))

            # subject scale is also hierarchical
            pop_scale_Attn = pm.Exponential("pop_scale_Attn", lam=5)
            sub_scale_Attn = pm.Exponential("sub_scale_Attn", lam=1 / pop_scale_Attn, dims=("subject",))

            # differences - scaled by subject scale above

            # ~~~~~~~~~~~~ 3. Stimulus Energy ~~~~~~~~~~~~
            # define the population-level dirichlet prior
            pop_dir_SE = pm.Dirichlet("pop_dir_SE", a=np.ones(StimEnergyK - 1), dims=("innovations_num_SE",))
            pop_dir_scale_SE = pm.Exponential("pop_dir_scale_SE", lam=5)
            pop_concentration_SE = pop_dir_scale_SE * pop_dir_SE

            # each subject's dirichlet is drawn from a scaled version of the population dirichlet
            sub_dir_SE = pm.Dirichlet(
                "sub_dir_SE",
                a=pop_concentration_SE,
                dims=("subject", "innovations_num_SE"),
            )
            sub_mu_SE = pm.Deterministic("sub_mu_SE_minus_first", cumsum(sub_dir_SE, axis=1))
            sub_mu_SE = pm.Deterministic("sub_mu_SE", at.concatenate((np.zeros((nsub, 1)), sub_mu_SE), axis=1))

            # subject scale is also hierarchical
            pop_scale_SE = pm.Exponential("pop_scale_SE", lam=5)
            sub_scale_SE = pm.Exponential("sub_scale_SE", lam=1 / pop_scale_SE, dims=("subject",))

            # ~~~~~~~~~~~~ 4. Combine into full model ~~~~~~~~~~~~

            # start by getting difference scores between pairs of conds based on WM, Attn, and StimEnergy levels
            # WM
            mu_i_WM = sub_vals_WM[q[:, 6].astype(int), q[:, 0].astype(int)]
            mu_j_WM = sub_vals_WM[q[:, 6].astype(int), q[:, 1].astype(int)]
            mu_delta_WM = pm.math.abs(mu_i_WM - mu_j_WM)

            # NOTE: attn and stim energy are flipped axes (subject, levels) compared to WM
            # Attn
            mu_i_Attn = sub_mu_Attn[q[:, 6].astype(int), q[:, 2].astype(int)]
            mu_j_Attn = sub_mu_Attn[q[:, 6].astype(int), q[:, 3].astype(int)]
            mu_delta_Attn = pm.math.abs(mu_i_Attn - mu_j_Attn) * sub_scale_Attn[q[:, 6].astype(int)]

            # StimEnergy
            mu_i_SE = sub_mu_SE[q[:, 6].astype(int), q[:, 4].astype(int)]
            mu_j_SE = sub_mu_SE[q[:, 6].astype(int), q[:, 5].astype(int)]
            mu_delta_SE = pm.math.abs(mu_i_SE - mu_j_SE) * sub_scale_SE[q[:, 6].astype(int)]

            # likelihood function
            dist_est = mu_delta_WM + mu_delta_Attn + mu_delta_SE
            pm.Normal("d", mu=dist_est, sigma=RDM_sigma_subs[q[:, 6].astype(int)], observed=d, dims=("trial_num",))

        return model

    # Iterate across models
    for WM_bend in WM_possible_bends:
        model_fn = f"model_comparison/model_{METRIC}_priorSD{PRIOR_SD*100:.0f}_WMbend{WM_bend}.nc"
        if path.exists(model_fn) and not OVERWRITE:
            print(f"Model with WM bend {WM_bend} already exists, skipping fit")
            continue
        print(f"Fitting model with WM bend {WM_bend}")

        model = build_model(pop_dists, pop_q, WM_bend)  # only pass the condition indices to the model
        with model:
            idata = pm.sample(
                10_000,  # real samples per chain
                tune=10_000,  # warmup samples per chain
                chains=4,
                target_accept=0.9999,
                random_seed=2026,
                return_inferencedata=True,
                idata_kwargs={"log_likelihood": True},
                nuts_sampler="numpyro",
            )
            idata.extend(
                pm.sample_prior_predictive(40_000)
            )  # add prior predictive samples to the idata for later checks
            # check for issues
            summary_table = az.summary(idata)
            if len(summary_table.loc[summary_table.r_hat > 1.01]) > 0:
                print(f"Model with WM bend {WM_bend} and prior SD {PRIOR_SD} has convergence issues:")
                print(summary_table.loc[summary_table.r_hat > 1.01].head(100))  # print any parameters with r_hat > 1.01

        idata.to_netcdf(model_fn)


# run script
if __name__ == "__main__":
    fit_models()
