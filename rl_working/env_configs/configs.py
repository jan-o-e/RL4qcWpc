def get_plot_elem_names(env_name):
    env_to_elem_names = {
        "simple_stirap": [
            "det-p",
            "det-s",
            "env-p",
            "env-s",
            "noise-amp",
            "noise-freq",
        ],
        "multi_stirap": [
            "det-p",
            "det-s",
            "env-p",
            "env-s",
            "noise-amp",
            "noise-freq",
        ],
        "rydberg": [
            "det-p",
            "env-p",
            "noise-amp",
            "noise-freq",
        ],
        "rydberg_two": [
            "det-p",
            "det-s",
            "env-p",
            "env-s",
            "noise-amp",
            "noise-freq",
        ],
        "transmon_reset": [
            "freq",
            "amp",
            "noise-amp",
            "noise-freq",
        ],
    }

    try:
        return env_to_elem_names[env_name]
    except KeyError:
        print(
            f"Warning: Environment '{env_name}' not recognized. Returning an empty list."
        )
        return []


def get_simple_stirap_params(args):
    return {
        "gamma": args.gamma_ss,
        "omega_0": args.omega_ss,
        "delta_s": args.delta_ss,
        "delta_p": args.delta_ss,
        "mxstep": args.mxstep_solver,
        "t_total": 1.0,
        "x_penalty": 0.5,
        "fid_factor": 3.0,
        "pulse_area_penalty": args.area_pen_ss,
        "smoothness_calc_amp": args.smoothness_calc_amp,
        "smoothness_calc_det": args.smoothness_calc_det,
        "convolution_std_amp": args.kernel_std_amp,
        "convolution_std_freq": args.kernel_std_amp,
        "smoothness_cutoff_freq": args.smoothness_cutoff_freq,
        "smoothness_penalty_env": args.smoothness_pen_ss,
        "smoothness_penalty_det": args.smoothness_pen_ss,
        'mx_step_penalty': args.mx_step_penalty,
        "fixed_endpoints": 1,
        "gaussian_kernel_window_length": [50, 25],
        "initial_state": [1, 0],
        "final_state": [args.final_state_zero_ss, 1],
        "n_action_steps": 50,
        "dissipator": True,
        "dim": 5,
        "x_detuning": args.x_detuning_ss,
        "noise": args.noise,
        "gaussian_noise_scaling": [args.sigma_phase, args.sigma_amp]
    }


def get_multi_stirap_params(args):
    return get_simple_stirap_params(args)


def get_rydberg_cz_params(args):
    return {
        "gamma": 0.003,
        "omega_0": 4 * 25,
        "delta_p": 4 * 10,
        "blockade_strength": args.blockade_strength,
        "mxstep": args.mxstep_solver,
        "t_total": 0.25,
        "fid_reward": 4.0,
        "log_fidelity": args.log_fidelity,
        "x_penalty": 1,
        "pulse_area_penalty": 0.0,
        "smoothness_calc_amp": args.smoothness_calc_amp,
        "smoothness_calc_det": args.smoothness_calc_det,
        "smoothness_cutoff_freq": args.smoothness_cutoff_freq,
        "smoothness_penalty_env": args.smoothness_pen_ss,
        "smoothness_penalty_det": args.smoothness_pen_ss,
        'mx_step_penalty': args.mx_step_penalty,
        "fixed_endpoints": args.fix_endpoints_ss,
        "gaussian_kernel_window_length": [50, 25],
        "n_action_steps": 50,
        "dissipator": True,
        "dim": 2,
        "noise": args.noise,
    }


def get_rydberg_two_params(args):
    return {
        "gamma": 3,
        "blockade_strength": args.blockade_strength,
        "omega_e": 2 * 14.4,
        "omega_r": 2 * 10,
        "delta_r": 2 * 10,
        "delta_e": -2500,
        "mxstep": args.mxstep_solver,
        "t_total": 0.5,
        "fid_reward": 4.0,
        "x_penalty": 1,
        "log_fidelity": args.log_fidelity,
        "pulse_area_penalty": 0.0,
        "convolution_std_amp": args.kernel_std_amp,
        "convolution_std_freq": args.kernel_std_amp,
        "smoothness_calc_amp": args.smoothness_calc_amp,
        "smoothness_calc_det": args.smoothness_calc_amp,
        "smoothness_cutoff_freq": args.smoothness_cutoff_freq,
        "smoothness_penalty_env": args.smoothness_pen_ss,
        "smoothness_penalty_det": args.smoothness_pen_ss,
        "mx_step_penalty": 0,
        "fixed_endpoints": args.fix_endpoints_ss,
        "gaussian_kernel_window_length": [50, 25],
        "n_action_steps": 50,
        "dissipator": True,
        "const_freq_pump": args.const_freq_pump_rydberg_two,
        "const_amp_stokes": args.const_amp_stokes_rydberg_two,
        "e_dissipation": 1,
        "r_dissipation": 0.001,
        "dim": 2,
        "noise": args.noise,
    }


def get_transmon_reset_params(args):
    return {
        "kappa": 5.,  # 4.25 initially
        "chi": -0.335,
        "delta": 1934,
        "anharm": -330.,
        "g_coupling": 67.,
        "gamma": 1 / 500.,
        "omega_max": 330.,
        "delta_max": 10.,
        "sim_t1": 0.20,
        "transmon_reset_coeff": 10.,
        "smoothness_coeff": 0.,
        "amp_pen_coeff": 0.,
        "deviation_coeff": 50.,
        "steps_pen_coeff": 20.,
        "max_grad": 43.,
        "k_factor": -0.001,
        "max_deviation": 0.2,
        "max_steps": args.mxstep_solver,
        "gauss_std":args.kernel_std_amp,
    }
