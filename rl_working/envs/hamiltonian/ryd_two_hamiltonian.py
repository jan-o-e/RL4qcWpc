import jax.numpy as jnp


def generate_hamiltonian_and_c_ops(reduced_hspace, complex_dtype, gamma,
                                   e_dissipation, r_dissipation):

    if reduced_hspace == 1:  # For reduced_hspace = 10
        size = 10

        def projector(i, j):
            ret_array = jnp.zeros((size, size), dtype=complex_dtype)
            ret_array = ret_array.at[i, j].set(1)
            return ret_array

        H_omega_p = 1 / 2 * (projector(1, 3) + jnp.sqrt(2) * projector(2, 5) +
                             projector(6, 7) + projector(3, 1) +
                             jnp.sqrt(2) * projector(5, 2) + projector(7, 6))

        H_omega_s = 1 / 2 * (projector(3, 4) + projector(5, 6) +
                             jnp.sqrt(2) * projector(8, 9) + projector(4, 3) +
                             projector(6, 5) + jnp.sqrt(2) * projector(9, 8))

        H_delta_s = projector(6, 6)
        H_delta_p = projector(5, 5)
        H_delta_sp = projector(7, 7)
        H_v = projector(8, 8)

        c_ops = (jnp.sqrt(gamma) * jnp.array([
            e_dissipation * projector(9, 5),
            e_dissipation * projector(9, 3),
            r_dissipation * projector(9, 4),
            r_dissipation * projector(9, 6),
            r_dissipation * projector(9, 7),
            r_dissipation * projector(9, 8),
        ]))

        h_ops = jnp.array(
            [H_omega_p, H_omega_s, H_delta_p, H_delta_s, H_delta_sp, H_v])

    elif reduced_hspace == 0:  # For reduced_hspace = 13
        size = 13

        def projector(i, j):
            ret_array = jnp.zeros((size, size), dtype=complex_dtype)
            ret_array = ret_array.at[i, j].set(1)
            return ret_array

        H_omega_p = 1 / 2 * (projector(1, 4) + jnp.sqrt(2) * projector(3, 8) +
                             projector(9, 10) + projector(2, 5) +
                             projector(4, 1) + jnp.sqrt(2) * projector(8, 3) +
                             projector(10, 9) + projector(5, 2))

        H_omega_s = 1 / 2 * (projector(4, 7) + projector(5, 6) + projector(
            8, 9) + jnp.sqrt(2) * projector(10, 11) + projector(7, 4) +
                             projector(6, 5) + projector(9, 8) +
                             jnp.sqrt(2) * projector(11, 10))

        H_delta_s = projector(9, 9)
        H_delta_p = projector(8, 8)
        H_delta_sp = projector(10, 10)
        H_v = projector(11, 11)

        c_ops = (jnp.sqrt(gamma) * jnp.array([
            e_dissipation * projector(12, 4),
            e_dissipation * projector(12, 5),
            e_dissipation * projector(12, 8),
            r_dissipation * projector(12, 9),
            r_dissipation * projector(12, 10),
            r_dissipation * projector(12, 11),
            r_dissipation * projector(12, 6),
            r_dissipation * projector(12, 7),
        ]))

        h_ops = jnp.array(
            [H_omega_p, H_omega_s, H_delta_p, H_delta_s, H_delta_sp, H_v])

    else:
        raise ValueError(
            "Invalid value for reduced_hspace. Use 0 for 10-dimensional or 1 for 13-dimensional space."
        )

    return h_ops, c_ops
