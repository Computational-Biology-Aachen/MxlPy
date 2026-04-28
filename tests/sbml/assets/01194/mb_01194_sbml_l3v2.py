from mxlpy import Model


def A(A_amount: float, Cell: float) -> float:
    return A_amount / Cell


def B(B_amount: float, Cell: float) -> float:
    return B_amount / Cell


def C(C_amount: float, Cell: float) -> float:
    return C_amount / Cell


def D(D_amount: float, Cell: float) -> float:
    return D_amount / Cell


def E(E_amount: float, Cell: float) -> float:
    return E_amount / Cell


def F(F_amount: float, Cell: float) -> float:
    return F_amount / Cell


def G(G_amount: float, Cell: float) -> float:
    return G_amount / Cell


def H(H_amount: float, Cell: float) -> float:
    return H_amount / Cell


def I(I_amount: float, Cell: float) -> float:
    return I_amount / Cell


def J(J_amount: float, Cell: float) -> float:
    return J_amount / Cell


def K(K_amount: float, Cell: float) -> float:
    return K_amount / Cell


def L(L_amount: float, Cell: float) -> float:
    return L_amount / Cell


def M(M_amount: float, Cell: float) -> float:
    return M_amount / Cell


def N(N_amount: float, Cell: float) -> float:
    return N_amount / Cell


def O(O_amount: float, Cell: float) -> float:
    return O_amount / Cell


def P(P_amount: float, Cell: float) -> float:
    return P_amount / Cell


def Q(Q_amount: float, Cell: float) -> float:
    return Q_amount / Cell


def R(R_amount: float, Cell: float) -> float:
    return R_amount / Cell


def S(S_amount: float, Cell: float) -> float:
    return S_amount / Cell


def T(T_amount: float, Cell: float) -> float:
    return T_amount / Cell


def U(U_amount: float, Cell: float) -> float:
    return U_amount / Cell


def X(X_amount: float, Cell: float) -> float:
    return X_amount / Cell


def Y(Y_amount: float, Cell: float) -> float:
    return Y_amount / Cell


def init_A_amount(Cell: float) -> float:
    return 0


def init_B_amount(Cell: float) -> float:
    return 0


def init_C_amount(Cell: float) -> float:
    return 0


def init_D_amount(Cell: float) -> float:
    return 0


def init_E_amount(Cell: float) -> float:
    return 0


def init_F_amount(Cell: float) -> float:
    return 0


def init_G_amount(Cell: float) -> float:
    return 0


def init_H_amount(Cell: float) -> float:
    return 0


def init_I_amount(Cell: float) -> float:
    return 0


def init_J_amount(Cell: float) -> float:
    return 0


def init_K_amount(Cell: float) -> float:
    return 0


def init_L_amount(Cell: float) -> float:
    return 0


def init_M_amount(Cell: float) -> float:
    return 0


def init_N_amount(Cell: float) -> float:
    return 0


def init_O_amount(Cell: float) -> float:
    return 0


def init_P_amount(Cell: float) -> float:
    return 0


def init_Q_amount(Cell: float) -> float:
    return 0


def init_R_amount(Cell: float) -> float:
    return 0


def init_S_amount(Cell: float) -> float:
    return 0


def init_T_amount(Cell: float) -> float:
    return 0


def init_U_amount(Cell: float) -> float:
    return 0


def init_X_amount(Cell: float) -> float:
    return 0


def init_Y_amount(Cell: float) -> float:
    return 0


def create_model() -> Model:
    return (
        Model()
        .add_variable("A_amount", initial_value=0.0)
        .add_variable("B_amount", initial_value=0.0)
        .add_variable("C_amount", initial_value=0.0)
        .add_variable("D_amount", initial_value=0.0)
        .add_variable("E_amount", initial_value=0.0)
        .add_variable("F_amount", initial_value=0.0)
        .add_variable("G_amount", initial_value=0.0)
        .add_variable("H_amount", initial_value=0.0)
        .add_variable("I_amount", initial_value=0.0)
        .add_variable("J_amount", initial_value=0.0)
        .add_variable("K_amount", initial_value=0.0)
        .add_variable("L_amount", initial_value=0.0)
        .add_variable("M_amount", initial_value=0.0)
        .add_variable("N_amount", initial_value=0.0)
        .add_variable("O_amount", initial_value=0.0)
        .add_variable("P_amount", initial_value=0.0)
        .add_variable("Q_amount", initial_value=0.0)
        .add_variable("R_amount", initial_value=0.0)
        .add_variable("S_amount", initial_value=0.0)
        .add_variable("T_amount", initial_value=0.0)
        .add_variable("U_amount", initial_value=0.0)
        .add_variable("X_amount", initial_value=0.0)
        .add_variable("Y_amount", initial_value=0.0)
        .add_variable("Cell", initial_value=1.00000000000000)
        .add_derived(
            "A",
            fn=A,
            args=["A_amount", "Cell"],
        )
        .add_derived(
            "B",
            fn=B,
            args=["B_amount", "Cell"],
        )
        .add_derived(
            "C",
            fn=C,
            args=["C_amount", "Cell"],
        )
        .add_derived(
            "D",
            fn=D,
            args=["D_amount", "Cell"],
        )
        .add_derived(
            "E",
            fn=E,
            args=["E_amount", "Cell"],
        )
        .add_derived(
            "F",
            fn=F,
            args=["F_amount", "Cell"],
        )
        .add_derived(
            "G",
            fn=G,
            args=["G_amount", "Cell"],
        )
        .add_derived(
            "H",
            fn=H,
            args=["H_amount", "Cell"],
        )
        .add_derived(
            "I",
            fn=I,
            args=["I_amount", "Cell"],
        )
        .add_derived(
            "J",
            fn=J,
            args=["J_amount", "Cell"],
        )
        .add_derived(
            "K",
            fn=K,
            args=["K_amount", "Cell"],
        )
        .add_derived(
            "L",
            fn=L,
            args=["L_amount", "Cell"],
        )
        .add_derived(
            "M",
            fn=M,
            args=["M_amount", "Cell"],
        )
        .add_derived(
            "N",
            fn=N,
            args=["N_amount", "Cell"],
        )
        .add_derived(
            "O",
            fn=O,
            args=["O_amount", "Cell"],
        )
        .add_derived(
            "P",
            fn=P,
            args=["P_amount", "Cell"],
        )
        .add_derived(
            "Q",
            fn=Q,
            args=["Q_amount", "Cell"],
        )
        .add_derived(
            "R",
            fn=R,
            args=["R_amount", "Cell"],
        )
        .add_derived(
            "S",
            fn=S,
            args=["S_amount", "Cell"],
        )
        .add_derived(
            "T",
            fn=T,
            args=["T_amount", "Cell"],
        )
        .add_derived(
            "U",
            fn=U,
            args=["U_amount", "Cell"],
        )
        .add_derived(
            "X",
            fn=X,
            args=["X_amount", "Cell"],
        )
        .add_derived(
            "Y",
            fn=Y,
            args=["Y_amount", "Cell"],
        )
        .add_initial_assignment(
            "A_amount",
            fn=init_A_amount,
            args=["Cell"],
        )
        .add_initial_assignment(
            "B_amount",
            fn=init_B_amount,
            args=["Cell"],
        )
        .add_initial_assignment(
            "C_amount",
            fn=init_C_amount,
            args=["Cell"],
        )
        .add_initial_assignment(
            "D_amount",
            fn=init_D_amount,
            args=["Cell"],
        )
        .add_initial_assignment(
            "E_amount",
            fn=init_E_amount,
            args=["Cell"],
        )
        .add_initial_assignment(
            "F_amount",
            fn=init_F_amount,
            args=["Cell"],
        )
        .add_initial_assignment(
            "G_amount",
            fn=init_G_amount,
            args=["Cell"],
        )
        .add_initial_assignment(
            "H_amount",
            fn=init_H_amount,
            args=["Cell"],
        )
        .add_initial_assignment(
            "I_amount",
            fn=init_I_amount,
            args=["Cell"],
        )
        .add_initial_assignment(
            "J_amount",
            fn=init_J_amount,
            args=["Cell"],
        )
        .add_initial_assignment(
            "K_amount",
            fn=init_K_amount,
            args=["Cell"],
        )
        .add_initial_assignment(
            "L_amount",
            fn=init_L_amount,
            args=["Cell"],
        )
        .add_initial_assignment(
            "M_amount",
            fn=init_M_amount,
            args=["Cell"],
        )
        .add_initial_assignment(
            "N_amount",
            fn=init_N_amount,
            args=["Cell"],
        )
        .add_initial_assignment(
            "O_amount",
            fn=init_O_amount,
            args=["Cell"],
        )
        .add_initial_assignment(
            "P_amount",
            fn=init_P_amount,
            args=["Cell"],
        )
        .add_initial_assignment(
            "Q_amount",
            fn=init_Q_amount,
            args=["Cell"],
        )
        .add_initial_assignment(
            "R_amount",
            fn=init_R_amount,
            args=["Cell"],
        )
        .add_initial_assignment(
            "S_amount",
            fn=init_S_amount,
            args=["Cell"],
        )
        .add_initial_assignment(
            "T_amount",
            fn=init_T_amount,
            args=["Cell"],
        )
        .add_initial_assignment(
            "U_amount",
            fn=init_U_amount,
            args=["Cell"],
        )
        .add_initial_assignment(
            "X_amount",
            fn=init_X_amount,
            args=["Cell"],
        )
        .add_initial_assignment(
            "Y_amount",
            fn=init_Y_amount,
            args=["Cell"],
        )
    )
