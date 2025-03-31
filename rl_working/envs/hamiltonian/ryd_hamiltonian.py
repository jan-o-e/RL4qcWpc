import jax.numpy as jnp


def generate_hamiltonian_and_c_ops(complex_dtype, gamma):
    def projector_nine(i, j):
        ret_array = jnp.zeros((9, 9), dtype=complex_dtype)
        ret_array = ret_array.at[i, j].set(1)
        return ret_array

    H_omega = (1 / 2 * (
        (projector_nine(4, 7) + projector_nine(4, 5) + projector_nine(5, 8) +
         projector_nine(7, 8) + projector_nine(7, 4) + projector_nine(5, 4) +
         projector_nine(8, 5) + projector_nine(8, 7) + projector_nine(1, 2) +
         projector_nine(2, 1) + projector_nine(3, 6) + projector_nine(6, 3))))

    H_delta = (-(projector_nine(7, 7) + projector_nine(5, 5) +
                 projector_nine(7, 5) + projector_nine(5, 7)) -
               2 * projector_nine(8, 8) - projector_nine(6, 6) -
               projector_nine(2, 2))
    H_v = projector_nine(8, 8)

    # NOTE: Dissipator currently defines fidelity lower bound as it only considers decay to a sink state and not to an existing ground state
    c_ops = (
        jnp.sqrt(gamma) * 1 * (
            jnp.array([
                #projector_ten(9, 2),
                #projector_ten(9, 7),
                #projector_ten(9, 5),
                #projector_ten(9, 8),
                #projector_ten(9, 6),
                projector_nine(0, 6),
                projector_nine(3, 6),
                projector_nine(1, 2),
                projector_nine(0, 2),
                projector_nine(4, 5),
                projector_nine(3, 5),
                projector_nine(1, 7),
                projector_nine(4, 7),
                #projector_nine(0, 8),
                #projector_nine(3, 8),
                #projector_nine(1, 8),
                #projector_nine(4, 8),
                projector_nine(7, 8),
                projector_nine(5, 8),
                projector_nine(2, 8),
                projector_nine(6, 8)
            ])))

    h_ops = jnp.array([H_omega, H_delta, H_v])

    return h_ops, c_ops
