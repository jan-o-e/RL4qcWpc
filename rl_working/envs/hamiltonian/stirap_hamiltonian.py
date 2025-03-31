# hamiltonian_parameters.py
import jax.numpy as jnp


def get_hamiltonian_operators(dim, gamma, _dtype):
    if dim == 4:

        def projector_four(i, j):
            ret_array = jnp.zeros((4, 4), _dtype)
            ret_array = ret_array.at[i, j].set(1)
            return ret_array

        H_p = 0.5 * (projector_four(1, 0) + projector_four(0, 1))
        H_s = 0.5 * (projector_four(2, 1) + projector_four(1, 2))
        H_dp = projector_four(1, 1)
        H_ds = projector_four(2, 2)
        L = jnp.sqrt(gamma) * projector_four(3, 1)
        ham_ops = jnp.array([H_dp, H_ds, H_p, H_s], dtype=_dtype)
        c_ops = jnp.array([L], dtype=_dtype)
    elif dim == 5:

        def projector_five(i, j):
            ret_array = jnp.zeros((5, 5), dtype=_dtype)
            ret_array = ret_array.at[i, j].set(1)
            return ret_array

        H_p = 0.5 * (projector_five(1, 0) + projector_five(0, 1))
        H_s = 0.5 * (projector_five(2, 1) + projector_five(1, 2))
        H_p_x = 0.5 * (projector_five(3, 0) + projector_five(0, 3))
        H_s_x = 0.5 * (-projector_five(3, 2) - projector_five(2, 3))
        H_dp = projector_five(1, 1)
        H_ds = projector_five(2, 2)
        H_d_x = projector_five(3, 3)
        L = jnp.sqrt(gamma) * (1 / 2 *
                               (projector_five(4, 1) + projector_five(4, 3)))
        ham_ops = jnp.array([H_dp, H_ds, H_d_x, H_p, H_s, H_p_x, H_s_x],
                            dtype=_dtype)
        c_ops = jnp.array([L], dtype=_dtype)
    else:
        raise ValueError(f"Unsupported dimension: {dim}")

    return ham_ops, c_ops
