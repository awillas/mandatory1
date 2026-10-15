import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

from scipy import sparse
from scipy.sparse import linalg as sparse_linalg
from scipy.interpolate import interpn

from math import floor, ceil, pi, sqrt


x, y, t = sp.symbols("x,y,t")


import numpy as np
import sympy as sp
from scipy import sparse

from math import pi, sqrt

x, y, t = sp.symbols("x y t")


class Wave2D:
    """Class for solving the 2D wave equation."""

    def create_mesh(
        self, N: int, sparse: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return 2D mesh created using np.meshgrid

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        sparse : bool, optional
            Whether to create a sparse mesh or not. Default is False.
        Returns
        -------
        xij : 2D array
            The x-coordinates of the mesh
        yij : 2D array
            The y-coordinates of the mesh"""
        unit_interval = np.linspace(0, 1, N+1)
        return np.meshgrid(unit_interval, unit_interval, indexing="ij", sparse=sparse)
        
    def D2(self, N: int) -> sparse.lil_matrix:
        """Return second order differentiation matrix

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        Returns
        -------
        D : scipy sparse LIL matrix
            The second order differentiation matrix
        """
        return sparse.diags([1.0, -2.0, 1.0], [-1, 0, 1], (N+1, N+1), 'lil') # For the Dirichlet problem, the boundary values are always zero

    @property
    def w(self):
        """Return the dispersion coefficient"""
        return pi*self.c*sqrt(self.mx**2 + self.my**2)
    
    def ue(self, mx: int, my: int) -> sp.Expr:
        """Return the exact standing wave

        Parameters
        ----------
        mx, my : int
            Parameters for the standing wave
        Returns
        -------
        ue : Sympy expression
            The exact solution as a Sympy expression in x, y and t
        """
        return sp.sin(mx * sp.pi * x) * sp.sin(my * sp.pi * y) * sp.cos(self.w * t)


    def initialize(self, N: int, mx: int, my: int) -> np.ndarray:
        r"""Initialize the solution at $U^{n}$ and $U^{n-1}$

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        mx, my : int
            Parameters for the standing wave
        """

        """
        xij, yij = self.create_mesh(N)
        Unp1, Un, Unm1 = np.zeros((3, N+1, N+1))
        Unm1[:] = np.sin(pi*mx*xij) * np.sin(pi*my*yij) # Since cos(0) = 1, we omit the cosine factor
        D = self.D2(N)/self.h**2
        Un[:] = Unm1 + 0.5*(self.c*self.dt)**2 * (D@Unm1 + Unm1@D.T)
        #Un *= self.interior_indicator_matrix # Set boundary values to zero
        Un[0,:] = 0
        Un[:,0] = 0
        Un[-1,:] = 0
        Un[:,-1] = 0
        self.Unp1, self.Un, self.Unm1 = Unp1, Un, Unm1
        """

        xij, yij = self.create_mesh(N)
        return np.sin(pi*mx*xij) * np.sin(pi*my*yij) # Since cos(0) = 1, we omit the cosine factor

    @property
    def dt(self) -> float:
        """Return the time step"""
        return self.cfl*self.h/self.c

    
    def l2_error(self, u: np.ndarray, t0: float) -> float:
        """Return l2-error norm

        Parameters
        ----------
        u : array
            The solution mesh function
        t0 : number
            The time of the comparison
        """
        xij, yij = self.create_mesh(u.shape[0]-1)
        u_exact = np.sin(pi*self.mx*xij) * np.sin(pi*self.my*yij) * np.cos(self.w*t0)
        return self.h*np.sqrt(np.sum((u-u_exact)**2))

    def apply_bcs(self, u: np.ndarray):
        """Apply Dirichlet boundary conditions."""
        u *= self.interior_indicator_matrix
        #u[0,:] = 0
        #u[:,0] = 0
        #u[-1,:] = 0
        #u[:,-1] = 0

    def __call__(
        self,
        N: int,
        Nt: int,
        cfl: float = 0.5,
        c: float = 1.0,
        mx: int = 3,
        my: int = 3,
        store_data: int = -1,
    ):
        """Solve the wave equation

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        Nt : int
            Number of time steps
        cfl : number
            The CFL number
        c : number
            The wave speed
        mx, my : int
            Parameters for the standing wave
        store_data : int
            Store the solution every store_data time step
            Note that if store_data is -1 then you should return the l2-error
            instead of data for plotting. This is used in `convergence_rates`.

        Returns
        -------
        If store_data > 0, then return a dictionary with key, value = timestep, solution
        If store_data == -1, then return the two-tuple (h, l2-error)
        """

        # We need these in some of the other methods
        self.h = 1/N
        self.cfl = cfl
        self.c = c
        self.mx = mx
        self.my = my

        # Create a matrix representing the interior of our domain
        self.interior_indicator_matrix = np.zeros((N+1, N+1), dtype=float)
        self.interior_indicator_matrix[1:-1,1:-1] = 1.0
        interior_indicator_matrix = self.interior_indicator_matrix

        dt = self.dt
        D = self.D2(N)/self.h**2

        Unp1, Un, Unm1 = np.zeros((3, N+1, N+1))
        Unm1[:] = self.initialize(N, mx, my)
        Un[:] = Unm1[:] + 0.5*(self.c*self.dt)**2 * (D@Unm1 + Unm1@D.T)
        self.apply_bcs(Un) #Un *= self.interior_indicator_matrix
        
        if store_data > 0:
            frames = { 0: Unm1.copy() }
            for n in range(1, Nt):
                Unp1[:] = 2.0*Un - Unm1 + (c*dt)**2 * (D@Un + Un@D.T)
                self.apply_bcs(Unp1) #Unp1 *= interior_indicator_matrix # Enforce the Dirichlet boundary condition
                Unm1[:] = Un
                Un[:] = Unp1
                
                frames[n] = Unm1.copy()

            return frames
            
        elif store_data == -1:
            for n in range(1, Nt):
                Unp1[:] = 2.0*Un - Unm1 + (c*dt)**2 * (D@Un + Un@D.T)
                self.apply_bcs(Unp1) #Unp1 *= interior_indicator_matrix # Enforce the Dirichlet boundary condition
                Unm1[:] = Un
                Un[:] = Unp1
                
            return self.h, self.l2_error(Un, Nt*dt)
        
    
    def convergence_rates(
        self,
        m: int = 4,
        cfl: float = 0.1,
        Nt: int = 10,
        mx: int = 3,
        my: int = 3,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute convergence rates for a range of discretizations."""

        E = []
        h = []

        N0 = 8

        for _ in range(m):
            dx, err = self(
                N0,
                Nt,
                cfl=cfl,
                mx=mx,
                my=my,
                store_data=-1,
            )

            E.append(err)
            h.append(dx)

            N0 *= 2
            Nt *= 2

        r = [
            np.log(E[i - 1] / E[i])
            / np.log(h[i - 1] / h[i])
            for i in range(1, m)
        ]

        return np.array(r), np.array(E), np.array(h)



class Wave2D_Neumann(Wave2D):
    """2D wave equation with homogeneous Neumann boundary conditions."""

    def D2(self, N: int) -> sparse.lil_matrix:
        D = sparse.diags([1.0, -2.0, 1.0], [-1, 0, 1], (N+1, N+1), 'lil')
        D[0,0] = -2.0
        D[0,1] = 2.0
        D[-1,-2] = 2.0
        D[-1,-1] = -2.0
        return D

    def initialize(
        self,
        N: int,
        mx: int,
        my: int,
        #c: float,
    ) -> np.ndarray:
        """Initialize cosine standing wave."""
        xij, yij = self.create_mesh(N)
        return np.cos(mx*pi*xij)*np.cos(my*pi*yij)

    def l2_error(self, u: np.ndarray, t0: float) -> float:
        """Return l2-error norm

        Parameters
        ----------
        u : array
            The solution mesh function
        t0 : number
            The time of the comparison
        """
        xij, yij = self.create_mesh(u.shape[0]-1)
        u_exact = np.cos(pi*self.mx*xij) * np.cos(pi*self.my*yij) * np.cos(self.w*t0)
        return self.h*np.sqrt(np.sum((u-u_exact)**2))

    def apply_bcs(self, u: np.ndarray):
        """Apply homogeneous Neumann boundary conditions."""
        pass



def test_convergence_wave2d():
    sol = Wave2D()
    r, _, _ = sol.convergence_rates(m=5, mx=2, my=3)
    assert abs(r[-1] - 2) < 1e-2, r

def test_convergence_wave2d_neumann():
    solN = Wave2D_Neumann()
    r, _, _ = solN.convergence_rates(mx=3, my=3)
    assert abs(r[-1] - 2) < 0.05

def test_exact_wave2d():
    l2_error_for_the_dirichlet_problem = Wave2D()(100, 100, cfl=1/sqrt(2), mx=2, my=2)[1]
    assert l2_error_for_the_dirichlet_problem < 1.0e-12

    l2_error_for_the_neumann_problem = Wave2D_Neumann()(100, 100, cfl=1/sqrt(2), mx=2, my=2)[1]
    assert l2_error_for_the_neumann_problem < 1.0e-12


if __name__ == "__main__":
    test_convergence_wave2d()
    test_convergence_wave2d_neumann()
    test_exact_wave2d()
    print("All tests passed!")
