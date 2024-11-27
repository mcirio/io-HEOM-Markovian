from qutip import *

sigmaz_op = basis(2,1) * basis(2,1).dag() - basis(2,0) * basis(2,0).dag()
sigmay_op = -1j * basis(2,1) * basis(2,0).dag() + 1j * basis(2,0) * basis(2,1).dag()
sigmax_op = basis(2,1) * basis(2,0).dag() + basis(2,0) * basis(2,1).dag()
id_op = basis(2,1) * basis(2,1).dag() + basis(2,0) * basis(2,0).dag()
sigmap_op = (sigmax_op + 1j * sigmay_op ) / 2.
sigmam_op = (sigmax_op - 1j * sigmay_op ) / 2.


def Hamiltonian_single(H):
    return - 1j * (spre(H) - spost(H))
def Lindblad_single(c):
    return (2*spre(c) * spost(c.dag()) - (spre(c.dag() * c) + spost(c.dag() * c)))
