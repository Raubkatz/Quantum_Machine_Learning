import numpy as np
from scipy.linalg import expm

def is_hermitian(matrix):
    """
    Check if a matrix is Hermitian.

    A matrix is Hermitian if it is equal to its own conjugate transpose.
    In other words, the matrix is Hermitian if it is unchanged when
    replaced by its conjugate transpose.

    Parameters
    ----------
    matrix : array-like
        A square matrix.

    Returns
    -------
    bool
        True if the matrix is Hermitian, False otherwise.
    """
    # Conjugate transpose of the matrix
    matrix_conj_transpose = matrix.conj().T
    # Check if the original matrix is close to its conjugate transpose
    return np.allclose(matrix, matrix_conj_transpose)


def random_element(G):
    """
    Generate a random element of the group represented by the list of generators G.

    A random element is created by forming a linear combination of the generators
    with random coefficients, and then exponentiating the resulting matrix.

    Parameters
    ----------
    G : list of array-like
        List of generators of the group.

    Returns
    -------
    array-like
        A random element of the group.
    """
    # Get the dimension of the group from the first generator
    N = G[0].shape[0]
    # Get the number of generators
    num_generators = len(G)

    # Generate a set of random coefficients
    coefficients = np.random.rand(num_generators)

    # Create a linear combination of the generators with the coefficients
    linear_combination = sum(coefficients[i] * G[i] for i in range(num_generators))

    # Apply the matrix exponential operation
    R = expm(1j * linear_combination)

    return R


def check_group_element(R, N, group_type):
    """
    Check if a matrix R is an element of the group specified by group_type.

    The matrix R is checked against the defining properties of the group:
    - SU(N): unitary and determinant 1
    - SL(N): determinant 1
    - SO(N): orthogonal

    Parameters
    ----------
    R : array-like
        The matrix to check.
    N : int
        The dimension of the matrix.
    group_type : str
        The type of the group ('SU', 'SL', 'SO').

    Returns
    -------
    bool
        True if the matrix R is an element of the specified group, False otherwise.
    """
    # Check the shape of the matrix
    if R.shape != (N, N):
        print(f"Shape Mismatch: The matrix shape is {R.shape}, expected {(N, N)}")
        return False

    # Check if the matrix is unitary
    unitary_check = np.allclose(R @ R.conj().T, np.eye(N))
    if not unitary_check:
        print("Unitarity Violation: The matrix is not unitary.")
        return False

    # Check the determinant based on group type
    if group_type in ['SU', 'SL']:
        det_check = np.isclose(np.linalg.det(R), 1)
        if not det_check:
            print(f"Determinant Mismatch: The determinant of the matrix is {np.linalg.det(R)}, expected 1")
            return False

    # Check the orthogonality for SO group
    if group_type == 'SO':
        transpose_check = np.allclose(R, R.T)
        if not transpose_check:
            print("Transpose Mismatch: The matrix is not equal to its transpose.")
            return False

    print(f"The matrix is a member of {group_type}({N}).")
    return True

def generate_SU(n):
    """
    Generate the generators of the special unitary group SU(n).

    The generators are traceless Hermitian matrices that can be used to
    generate any element of the group through linear combinations and
    exponentiation.

    Parameters
    ----------
    n : int
        The dimension of the group.

    Returns
    -------
    array-like
        A set of generators for SU(n).
    """

    def traceless_diag(n):
        """
        Helper function to generate a traceless diagonal array.

        The diagonal contains -1's and a single (n - 1) on the last
        entry to ensure the trace is zero.

        Parameters
        ----------
        n : int
            The size of the diagonal.

        Returns
        -------
        array-like
            A traceless diagonal array of size n.
        """
        tot_arr = np.zeros(n)
        for i in range(n - 1):
            tot_arr[i] = -1
        tot_arr[n - 1] = n - 1
        return tot_arr

    # Check the dimension n
    if n < 2:
        return 'Choose n>=2'

    # Case n=2
    if n == 2:
        SU2_gens = np.zeros((3, 2, 2), dtype=np.complex128)
        SU2_gens[0] = np.array([[0., 1], [1, 0]])
        SU2_gens[1] = np.array([[0., -1j], [1j, 0]])
        SU2_gens[2] = np.array([[1., 0], [0, -1]])
        return SU2_gens

    # Case n>2
    else:
        dim = n ** 2 - 1
        dim_m_1 = (n - 1) ** 2 - 1
        gens = np.zeros((dim, n, n), dtype=np.complex128)
        gens_m_1 = generate_SU(n - 1)

        # Extend generators of SU(n-1) to SU(n)
        for i in range(dim_m_1):
            gens[i] = np.append(np.append(gens_m_1[i], np.zeros((n - 1, 1)), axis=1),
                                np.zeros((1, n)), axis=0)

        # Generators with 1 entries
        for a in range(dim_m_1, dim_m_1 + (n - 1)):
            gens[a, a % (n - 1), n - 1] = 1
            gens[a, n - 1, a % (n - 1)] = 1

        # Generators with 1j entries
        for a in range(dim_m_1 + (n - 1), dim_m_1 + 2 * (n - 1)):
            gens[a, a % (n - 1), n - 1] = -1j
            gens[a, n - 1, a % (n - 1)] = 1j

        # Generator with diagonal entries
        gens[dim - 1] = (2 ** 0.5 / (n * (n - 1)) ** 0.5) * np.diag(traceless_diag(n))

        return gens

def generate_SO(n):
    """
    Generate the generators of the special orthogonal group SO(n).

    The generators are skew-symmetric matrices that can be used to
    generate any element of the group through linear combinations and
    exponentiation.

    Parameters
    ----------
    n : int
        The dimension of the group.

    Returns
    -------
    array-like
        A set of generators for SO(n).
    """

    dim = int(n * (n - 1) / 2)
    gens = np.zeros((dim, n, n), dtype=np.complex128)
    ij_pair = [(i, j) for i in range(n) for j in range(n) if i < j]

    # Construct the generators
    for a, (i, j) in enumerate(ij_pair):
        gens[a, i, j] = 1.
        gens[a, j, i] = -1.

    return gens


def generate_SL_from_SU(n):
    """
    Generate the generators of the special linear group SL(n)
    from the generators of SU(n).

    The generators of SL(n) can be obtained from the generators of SU(n)
    by multiplying all the generators with purely imaginary entries by 1j.

    Parameters
    ----------
    n : int
        The dimension of the group.

    Returns
    -------
    array-like
        A set of generators for SL(n).
    """

    gens = generate_SU(n)

    # Change generators with purely imaginary entries
    for i in range(len(gens)):
        if np.iscomplex(gens[i]).any():
            gens[i] = gens[i] * 1j

    return gens