import jax.numpy as jnp
import jax
from jax.scipy.linalg import sqrtm

# Check if GPU is available
if "cuda" in str(jax.devices()):
    print("Connected to a GPU")
    processor = "gpu"
else:
    jax.config.update("jax_platform_name", "cpu")
    print("Not connected to a GPU")
    jax.config.update("jax_enable_x64", True)
    processor = "cpu"

# dtype
if processor == "gpu":
    float_dtype = jnp.float32
    complex_dtype = jnp.complex64
else:
    float_dtype = jnp.float64
    complex_dtype = jnp.complex128


def ket(_dim, i):
    """input args:
    i: (int) index of the desired ket
    """
    psi_vec = jnp.zeros(_dim, dtype=complex_dtype)
    psi_vec = psi_vec.at[i].set(1)
    return psi_vec


def projector(_dim, i, j):
    """input args:
    i: (int) index of the desired bra
    j: (int) index of the desired ket
    """
    ret_array = jnp.zeros((_dim, _dim), dtype=complex_dtype)
    ret_array = ret_array.at[i, j].set(1)
    return ret_array


def statevector_to_rho(psi):
    """input args:
    psi: (jnp.ndarray) state vector
    """
    return jnp.outer(psi, jnp.conjugate(psi))


def fidelity(rho1, rho2):
    """
    Calculate the fidelity between two density matrices rho1 and rho2
    input args:
    rho1: (jnp.ndarray) density matrix 1
    rho2: (jnp.ndarray) density matrix 2"""
    if rho1.ndim == 1:
        rho_1 = statevector_to_rho(rho1)
    elif rho1.ndim == 2:
        rho_1 = rho1
    else:
        raise TypeError("rho1 cannot be larger than a 2D matrix")

    if rho2.ndim == 1:
        rho_2 = statevector_to_rho(rho2)
    elif rho2.ndim == 2:
        rho_2 = rho2
    else:
        raise TypeError("rho2 cannot be larger than a 2D matrix")

    # Compute the square root of rho1
    sqrt_rho1 = sqrtm(rho_1)

    # Compute the product of sqrt_rho1 and rho2
    product = jnp.matmul(jnp.matmul(sqrt_rho1, rho_2), sqrt_rho1)

    # Compute the square root of the result
    sqrt_result = sqrtm(product)

    # Compute the trace of the square root of the result
    trace = jnp.trace(sqrt_result)

    # Compute the fidelity
    fid = jnp.real(trace)**2

    return fid


def mixed_pure_state_fidelity(psi, rho):
    """Compute the fidelity between a pure state and a mixed state."""
    # Compute the expectation value of rho with respect to psi
    expectation_value = jnp.vdot(psi, jnp.matmul(rho, psi))

    # Compute the fidelity
    fid = jnp.real(expectation_value)

    return fid


def pure_state_fidelity(psi1, psi2):
    """Compute the fidelity between two pure states."""
    # Compute the inner product of the two states
    inner_product = jnp.vdot(psi1, psi2)

    # Compute the fidelity
    fid = jnp.abs(inner_product)**2

    return fid


def cz_gate_two_photon():
    """Create the CZ gate for two photons transition."""
    # Create a diagonal matrix filled with zeros
    diagonal_elements = jnp.zeros(13)

    # Set the specified diagonal elements
    diagonal_elements = diagonal_elements.at[0].set(1)
    diagonal_elements = diagonal_elements.at[1].set(1)
    diagonal_elements = diagonal_elements.at[2].set(1)
    diagonal_elements = diagonal_elements.at[3].set(-1)

    # Create the diagonal matrix
    diagonal_matrix = jnp.diag(diagonal_elements)
    diagonal_matrix = diagonal_matrix.astype(complex_dtype)

    return diagonal_matrix


def tensor(a, b):
    """input args:
    a: (jnp.ndarray) matrix a
    b: (jnp.ndarray) matrix b
    """
    return jnp.tensordot(a, b, axes=0)


def lowering_op(_dim):
    return jnp.sqrt(jnp.diag(jnp.arange(_dim - 1) + 1, k=1))


def raising_op(_dim):
    return jnp.sqrt(jnp.diag(jnp.arange(_dim - 1) + 1, k=-1))


def res_transmon_ket(res_dim, transmon_dim, res_state, transmon_state):
    transmon_ket = ket(transmon_dim, transmon_state)
    res_ket = ket(res_dim, res_state)

    total_ket = jnp.kron(res_ket, transmon_ket)

    return total_ket


def res_transmon_dm(res_dim, transmon_dim, res_state, transmon_state):
    transmon_ket = ket(transmon_dim, transmon_state)
    res_ket = ket(res_dim, res_state)

    total_ket = jnp.kron(res_ket, transmon_ket)
    total_dm = statevector_to_rho(total_ket)

    return total_dm
