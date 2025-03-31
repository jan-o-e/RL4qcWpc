import jax.numpy as jnp


def get_hamiltonian_operators(transmon_dim, res_dim, chi, kappa, gamma,
                              g_bar_factor, _dtype):
    # Ladder operators for resonator and transmon
    a = jnp.diag(jnp.sqrt(jnp.arange(1, res_dim, dtype=_dtype)), 1)
    adag = jnp.diag(jnp.sqrt(jnp.arange(1, res_dim, dtype=_dtype)), -1)
    q = jnp.diag(jnp.sqrt(jnp.arange(1, transmon_dim, dtype=_dtype)), 1)
    qdag = jnp.diag(jnp.sqrt(jnp.arange(1, transmon_dim, dtype=_dtype)), -1)

    # Identity matrices
    trans_ident = jnp.eye(transmon_dim, dtype=_dtype)
    res_ident = jnp.eye(res_dim, dtype=_dtype)

    # Kronecker products to expand operators
    a_op = jnp.kron(trans_ident, a).astype(_dtype)
    ad_op = jnp.kron(trans_ident, adag).astype(_dtype)
    q_op = jnp.kron(q, res_ident).astype(_dtype)
    qd_op = jnp.kron(qdag, res_ident).astype(_dtype)

    # Hamiltonian terms with enforced dtype
    H_STATIC = (0.5 * chi * qd_op @ q_op @ ad_op @ a_op).astype(_dtype)
    H_DISSIPATE = [
        (jnp.sqrt(kappa) * a_op).astype(_dtype),
        (jnp.sqrt(gamma) * q_op).astype(_dtype),
    ]
    H_DRIVE = [
        (g_bar_factor * (qd_op @ qd_op @ a_op + ad_op @ q_op @ q_op) /
         jnp.sqrt(2)).astype(_dtype),
        (qd_op @ q_op).astype(_dtype),
    ]
    return H_STATIC, H_DISSIPATE, H_DRIVE, (qd_op @ q_op).astype(_dtype)
