

import numpy as np
import pandas as pd
from scipy.linalg import expm,svd
import math
import time
from os.path import exists

def vec_conj(a):
    return np.conjugate(a)

def vec_dot(a,b):
    # second vector is conjuagated
    return np.dot(a,vec_conj(b))

def dagger(M):
    return np.conjugate(np.transpose(M))

def normalize(psi):
    return psi/np.sqrt(vec_dot(psi,psi))

def s_x():
    sx = np.matrix([[0,1],[1,0]])
    return sx

def s_y():
    sy = np.matrix([[0,-1j],[1j,0]])
    return sy

def s_z():
    sz = np.matrix([[1,0],[0,-1]])
    return sz
    
def kron4(A,B,C,D):
    return np.kron(A,np.kron(B,np.kron(C,D)))

def I():
    return np.eye(2)

def lattice_gauge(h,J,K,m):
    hi_start = h*(kron4(s_z(),I(),I(),I())+kron4(I(),s_z(),I(),I())+kron4(I(),I(),s_z()/2,I())+kron4(I(),I(),I(),s_z()/2)) - (J/2)*(kron4(I(),s_x(),I(),I())-kron4(s_z(),s_x(),s_z(),I())) - (J/2)*(kron4(I(),I(),s_x(),I())-kron4(I(),s_z(),s_x(),s_z())) - (K/2)*(kron4(I(),s_x(),I(),I()) + kron4(s_z(),s_x(),s_z(),I())) - (K/2)*(kron4(I(),I(),s_x(),I()) + kron4(I(),s_z(),s_x(),s_z())) - (m/2)*(kron4(s_z(),s_z(),I(),I()) + kron4(I(),s_z(),s_z(),I()) + (1/2)*kron4(I(),I(),s_z(),s_z()))
    hi = h*(kron4(s_z()/2,I(),I(),I())+kron4(I(),s_z()/2,I(),I())+kron4(I(),I(),s_z()/2,I())+kron4(I(),I(),I(),s_z()/2)) - (J/2)*(kron4(I(),s_x(),I(),I())-kron4(s_z(),s_x(),s_z(),I())) - (J/2)*(kron4(I(),I(),s_x(),I())-kron4(I(),s_z(),s_x(),s_z())) - (K/2)*(kron4(I(),s_x(),I(),I()) + kron4(s_z(),s_x(),s_z(),I())) - (K/2)*(kron4(I(),I(),s_x(),I()) + kron4(I(),s_z(),s_x(),s_z())) - (m/2)*((1/2)*kron4(s_z(),s_z(),I(),I()) + kron4(I(),s_z(),s_z(),I()) + (1/2)*kron4(I(),I(),s_z(),s_z()))
    hi_end = h*(kron4(s_z()/2,I(),I(),I())+kron4(I(),s_z()/2,I(),I())+kron4(I(),I(),s_z(),I())+kron4(I(),I(),I(),s_z())) - (J/2)*(kron4(I(),s_x(),I(),I())-kron4(s_z(),s_x(),s_z(),I())) - (J/2)*(kron4(I(),I(),s_x(),I())-kron4(I(),s_z(),s_x(),s_z())) - (K/2)*(kron4(I(),s_x(),I(),I()) + kron4(s_z(),s_x(),s_z(),I())) - (K/2)*(kron4(I(),I(),s_x(),I()) + kron4(I(),s_z(),s_x(),s_z())) - (m/2)*((1/2)*kron4(s_z(),s_z(),I(),I()) + kron4(I(),s_z(),s_z(),I()) + kron4(I(),I(),s_z(),s_z()))
    return [hi_start,hi,hi_end]

# def lattice_gauge(h,J):
#     hi_start = h*(kron4(s_z(),I(),I(),I())+kron4(I(),s_z(),I(),I())+kron4(I(),I(),s_z()/2,I())+kron4(I(),I(),I(),s_z()/2)) - (J/2)*(kron4(I(),s_x(),I(),I())-kron4(s_z(),s_x(),s_z(),I())) - (J/2)*(kron4(I(),I(),s_x(),I())-kron4(I(),s_z(),s_x(),s_z()))
#     hi = h*(kron4(s_z()/2,I(),I(),I())+kron4(I(),s_z()/2,I(),I())+kron4(I(),I(),s_z()/2,I())+kron4(I(),I(),I(),s_z()/2)) - (J/2)*(kron4(I(),s_x(),I(),I())-kron4(s_z(),s_x(),s_z(),I())) - (J/2)*(kron4(I(),I(),s_x(),I())-kron4(I(),s_z(),s_x(),s_z()))
#     hi_end = h*(kron4(s_z()/2,I(),I(),I())+kron4(I(),s_z()/2,I(),I())+kron4(I(),I(),s_z(),I())+kron4(I(),I(),I(),s_z())) - (J/2)*(kron4(I(),s_x(),I(),I())-kron4(s_z(),s_x(),s_z(),I())) - (J/2)*(kron4(I(),I(),s_x(),I())-kron4(I(),s_z(),s_x(),s_z()))
#     return [hi_start,hi,hi_end]

def init_psi(L):
    psi = np.zeros((L,2),dtype=np.complex128)
    for i in range(L):
        if i == int(L/2) or i == int(L/2)-1 or i == int(L/2)+1:
            psi[i] += np.array([1,0])
        else:
            psi[i] += np.array([0,1])
    return psi

def init_psi_tq(L):
    psi = np.zeros((L,2),dtype=np.complex128)
    for i in range(L):
        if i == int(L/2)-1 or i == int(L/2)+1:
            psi[i] += np.array([1,0])
        else:
            psi[i] += np.array([0,1])
    return psi

def init_psi_vac(L):
    psi = np.zeros((L,2),dtype=np.complex128)
    for i in range(L):
        psi[i] += np.array([0,1])
    return psi

def init_psi_i(L):
    psi = np.zeros((L,2),dtype=np.complex128)
    for i in range(L):
        if i == int(L/2):
            psi[i] += np.array([1,0])
        else:
            psi[i] += np.array([0,1])
    return psi

def init_psi2(L):
    psi = np.zeros((L,2),dtype=np.complex128)
    for i in range(L):
        if i == int(L/4) or i == int(3*L/4):
            psi[i] += np.array([1,0])
        else:
            psi[i] += np.array([0,1])
    return psi

def init_psi_m(L):
    psi = np.zeros((L,2),dtype=np.complex128)
    for i in range(L):
        if (i+1)%8 in [1,2,7,0]:
            psi[i] += np.array([0,1])
        elif (i+1)%8 in [3,4,5,6]:
            psi[i] += np.array([1,0])
    return psi

def init_MPS_dict(L):
    psi = init_psi_vac(L)
    A_dict = {}
    for i in np.arange(0,L,2):
        key = str("A"+str(i))
        A_temp1 = np.reshape(psi[i],(1,2,1))
        A_temp2 = np.reshape(psi[i+1],(1,2,1))
        A_temp = np.einsum('aib,bjc->aijc',A_temp1,A_temp2,optimize='optimal')
        A_dict[key] = np.reshape(A_temp,(1,4,1))
    return A_dict


class MPS:
    # Hamiltonian parameters
    J = #JJ#
    h = #HH#
    K = #KK#
    m = #MM#
    
    def __init__(self,L,chi,T,N):
        self.L = L
        self.chi = chi
        self.T = T
        self.N = N
        self.dt = self.T/self.N
        
        # Initialize U
        
        Hi_start, Hi, Hi_end = lattice_gauge(self.h,self.J,self.K,self.m)
        self.U = expm(-1j*self.dt*Hi)
        self.U_start = expm(-1j*self.dt*Hi_start)
        self.U_end = expm(-1j*self.dt*Hi_end)
        
        
        # Initialize MPS
        self.A_dict = init_MPS_dict(self.L)
        self.lmbd_position = 0
        
        
        # Initialize measurements
        self.ni_persite = np.zeros(self.L-1,dtype=np.complex128)
        self.E_total_TEBD = 0
        self.n_tot = 0
        self.tqe = 0
        self.tqe1 = 0
        self.tqe2 = 0
        self.tqe_total = 0
        self.tqe_persite = np.zeros(self.L-4,dtype=np.complex128)
        self.tqe_width = 0
        self.n_1meson = 0
        self.n_1meson_persite = np.zeros(self.L-2,dtype=np.complex128)
        self.n_2meson = 0
        self.n_2meson_persite = np.zeros(self.L-3,dtype=np.complex128)
        self.n_3meson = 0
        self.n_3meson_persite = np.zeros(self.L-4,dtype=np.complex128)
        self.n_4meson = 0
        self.n_4meson_persite = np.zeros(self.L-5,dtype=np.complex128)
        self.avg_ml = 0
        
        
    # Returns the position of lmbd based on the trace
    def lmbd_pos(self):
        for i in range(self.L):
            key = "A"+str(i)
            lmbd_trace = np.linalg.norm(np.einsum('aib,aib',self.A_dict[key],np.conjugate(self.A_dict[key])))
            if (math.isclose(lmbd_trace,1)):
                break
        return i
    
    # Check lmbd_position using the previous function
    def lmbd_check(self,ind):
        if (ind[0] == self.lmbd_pos() or ind[1] == self.lmbd_pos()):
            return True
        else:
            print('Warning: Lmbd not in position')
            return False
    
    
    # Updates the MPS and runs applyU_schrodinger
    def applyU(self,ind,dirc,U,lm=False):
        
        # This part relocates lmbd to the right position
        if lm == False:
            if dirc == 'left':
                self.lmbd_relocate(ind[1])
            elif dirc == 'right':
                self.lmbd_relocate(ind[0])
        
        # lm checks if we want to apply U or move lmbd
        if lm == False:
#             self.lmbd_check(ind)
            if sch_bool == True:
                self.applyU_schrodinger(ind[0],U)
            U = np.reshape(U,(4,4,4,4))
            
        elif lm == True:
            U = np.reshape(np.eye(16),(4,4,4,4))
            
        A1 = self.A_dict["A"+str(ind[0])]
        A2 = self.A_dict["A"+str(ind[1])]
        chi1 = np.shape(A1)[0]
        chi2 = np.shape(A2)[2]
    
        #s1 = np.einsum('aib,bjc,ijkl->aklc',A1,A2,U)
        s1 = np.einsum('ijkl,akb,blc->aijc',U,A1,A2,optimize='optimal')
        s2 = np.reshape(s1,(4*chi1,4*chi2))
        try:
            Lp,lmbd,R=np.linalg.svd(s2,full_matrices=False)
        except np.linalg.LinAlgError as err:
            if "SVD did not converge" in str(err):
                Lp,lmbd,R=svd(s2,full_matrices=False,lapack_driver='gesvd')
                f = open("py_print.txt","a")
                f.write("SVD convergence issue")
                f.close()
            else:
                raise
        chi12 = np.min([4*chi1,4*chi2])
        chi12_p = np.min([self.chi,chi12])
        lmbd = np.diag(lmbd)
    
        # Truncation step
        lmbd = lmbd[:chi12_p,:chi12_p]
        Lp = Lp[:,:chi12_p]
        R = R[:chi12_p,:]
        
        norm_lmbd = np.einsum('ab,ab',lmbd,np.conjugate(lmbd))
        lmbd = lmbd/np.sqrt(norm_lmbd)
        
        if (dirc == 'left'):
            A1 = np.reshape(np.dot(Lp,lmbd),(chi1,4,chi12_p))
            A2 = np.reshape(R,(chi12_p,4,chi2))
            self.lmbd_position = ind[0]
            
        elif (dirc == 'right'):
            A1 = np.reshape(Lp,(chi1,4,chi12_p))
            A2 = np.reshape(np.dot(lmbd,R),(chi12_p,4,chi2))
            self.lmbd_position = ind[1]
        
        self.A_dict["A"+str(ind[0])] = A1
        self.A_dict["A"+str(ind[1])] = A2
        
        # Checks the TEBD and Schrodinger wavefunctions match
        if sch_bool == True:
            if not self.check_schrodinger_psi():
                print('Warning: Wavefunctions do not match')
    
    # Function to move lmbd right
    def move_lmbd_right(self,ind):
        self.applyU([ind,ind+2],'right',1,lm=True)
    
    # Function to move lmbd left
    def move_lmbd_left(self,ind):
        self.applyU([ind,ind+2],'left',1,lm=True)
    
    # Sweeps over the entire system and updates the MPS and the Schrodinger wavefunction (1 time step)
    def sweepU(self):
    
        sites = [[i,i+2] for i in np.arange(0,self.L-2,2)]
        
        
        for i in sites:
            if i[0] == 0:
                self.applyU(i,'right',self.U_start)
            elif i[1] == self.L-2:
                self.applyU(i,'right',self.U_end)
            else:
                self.applyU(i,'right',self.U)
                
        sites.reverse()
            
        for i in sites:
            if i[0] == 0:
                self.applyU(i,'left',self.U_start)
            elif i[1] == self.L-2:
                self.applyU(i,'left',self.U_end)
            else:
                self.applyU(i,'left',self.U)
        self.measure_TEBD()
        
    
    
    # Relocates lmbd from lmbd_position to ind
    def lmbd_relocate(self,ind):
        step = ind - self.lmbd_position
        for i in range(int(np.abs(step)/2)):
            if step > 0:
                self.move_lmbd_right(self.lmbd_position)
            elif step < 0:
                self.move_lmbd_left(self.lmbd_position-2)
    
        
    # Measures the expectation value using the MPS
    def measure_TEBD(self):
        
        self.left_trace = []
        self.right_trace = []
        self.build_left()
        self.build_right()
        
        self.SzSz_build = np.zeros(self.L-1,dtype=np.complex128)
        self.SzISz_build = np.zeros(self.L-2,dtype=np.complex128)
        self.SzIISz_build = np.zeros(self.L-3,dtype=np.complex128)
        self.SzIIISz_build = np.zeros(self.L-4,dtype=np.complex128)
        self.SzSzSzSz_build = np.zeros(self.L-3,dtype=np.complex128)
        self.SzSzISzSz_build = np.zeros(self.L-4,dtype=np.complex128)
        self.SzISzSzSz_build = np.zeros(self.L-4,dtype=np.complex128)
        self.SzSzSzISz_build = np.zeros(self.L-4,dtype=np.complex128)
        self.SzSzIISzSz_build = np.zeros(self.L-5,dtype=np.complex128)
        self.SzSzSzSzSzSz_build = np.zeros(self.L-5,dtype=np.complex128)
        self.SzSzSzIISz_build = np.zeros(self.L-5,dtype=np.complex128)
        self.SzIISzSzSz_build = np.zeros(self.L-5,dtype=np.complex128)
        self.SzSzISzISz_build = np.zeros(self.L-5,dtype=np.complex128)
        self.SzISzISzSz_build = np.zeros(self.L-5,dtype=np.complex128)
        self.SzSzIISzSz_build = np.zeros(self.L-5,dtype=np.complex128)
        self.SzISzSzISz_build = np.zeros(self.L-5,dtype=np.complex128)
        self.SzIIIISz_build = np.zeros(self.L-5,dtype=np.complex128)
        
        self.SzSz()
        self.SzISz()
        self.SzIISz()
        self.SzIIISz()
        self.SzSzSzSz()
        self.SzSzISzSz()
        self.SzISzSzSz()
        self.SzSzSzISz()
        self.SzSzIISzSz()
        self.SzSzSzSzSzSz()
        self.SzSzSzIISz()
        self.SzIISzSzSz()
        self.SzSzISzISz()
        self.SzISzISzSz()
        self.SzSzIISzSz()
        self.SzISzSzISz()
        self.SzIIIISz()
        

        # total particle number
        self.n_tot = 0
        for i in range(self.L-1):
            if i%2 == 0:
                self.lmbd_relocate(i)
                sz_tmp = np.einsum('aib,ij,ajb',np.conjugate(self.A_dict["A"+str(i)]),np.kron(s_z(),s_z()),self.A_dict["A"+str(i)],optimize='optimal')
                self.ni_persite[i] = (1-sz_tmp)/2
                self.n_tot += (1-sz_tmp)/2
            else:
                self.lmbd_relocate(i-1)
                sz_tmp = np.einsum('aib,ij,ajc->bc',np.conjugate(self.A_dict["A"+str(i-1)]),np.kron(I(),s_z()),self.A_dict["A"+str(i-1)],optimize='optimal')
                sz_tmp = np.einsum('bc,bid->dic',sz_tmp,np.conjugate(self.A_dict["A"+str(i+1)]),optimize='optimal')
                sz_tmp = np.einsum('dic,ij,cjd',sz_tmp,np.kron(s_z(),I()),self.A_dict["A"+str(i+1)],optimize='optimal')
                self.ni_persite[i] = (1-sz_tmp)/2
                self.n_tot += (1-sz_tmp)/2

        # tetraquark width
        top_indices = np.argpartition(-self.ni_persite, 4)[:4]
        sorted_top_indices = top_indices[np.argsort(-self.ni_persite[top_indices])]

        self.tqe_width = np.abs(sorted_top_indices[0] - sorted_top_indices[3])


        
        # measure total energy
        self.E_total_TEBD = 0
        for i in np.arange(0,self.L,2):
            self.lmbd_relocate(i)
            self.E_total_TEBD += self.h*np.einsum('aib,ij,ajb',np.conjugate(self.A_dict["A"+str(i)]),np.kron(s_z(),np.eye(2)),self.A_dict["A"+str(i)],optimize='optimal')
            self.E_total_TEBD += self.h*np.einsum('aib,ij,ajb',np.conjugate(self.A_dict["A"+str(i)]),np.kron(np.eye(2),s_z()),self.A_dict["A"+str(i)],optimize='optimal')
            if i == 0:
                self.E_total_TEBD += -(1/2)*(self.J+self.K)*np.einsum('aib,ij,ajb',np.conjugate(self.A_dict["A"+str(i)]),np.kron(np.eye(2),s_x()),self.A_dict["A"+str(i)],optimize='optimal')
            elif i == self.L-2:
                self.E_total_TEBD += -(1/2)*(self.J+self.K)*np.einsum('aib,ij,ajb',np.conjugate(self.A_dict["A"+str(i)]),np.kron(s_x(),np.eye(2)),self.A_dict["A"+str(i)],optimize='optimal')
                self.E_total_TEBD += -(self.m/2)*np.einsum('aib,ij,ajb',np.conjugate(self.A_dict["A"+str(i)]),np.kron(s_z(),s_z()),self.A_dict["A"+str(i)],optimize='optimal')
            else:
                self.E_total_TEBD += -(1/2)*(self.J+self.K)*np.einsum('aib,ij,ajb',np.conjugate(self.A_dict["A"+str(i)]),np.kron(s_x(),np.eye(2)),self.A_dict["A"+str(i)],optimize='optimal')
                self.E_total_TEBD += -(1/2)*(self.J+self.K)*np.einsum('aib,ij,ajb',np.conjugate(self.A_dict["A"+str(i)]),np.kron(np.eye(2),s_x()),self.A_dict["A"+str(i)],optimize='optimal')
            
            if i != self.L-2:
                tmp_e1 = np.einsum('aib,ij,ajc->bc',np.conjugate(self.A_dict["A"+str(i)]),np.kron(s_z(),s_x()),self.A_dict["A"+str(i)],optimize='optimal')
                tmp_e1 = np.einsum('bc,bid->dic',tmp_e1,np.conjugate(self.A_dict["A"+str(i+2)]),optimize='optimal')
                tmp_e1 = np.einsum('dic,ij,cjd',tmp_e1,np.kron(s_z(),np.eye(2)),self.A_dict["A"+str(i+2)],optimize='optimal')
                self.E_total_TEBD += (1/2)*(self.J-self.K)*tmp_e1
            
                tmp_e2 = np.einsum('aib,ij,ajc->bc',np.conjugate(self.A_dict["A"+str(i)]),np.kron(np.eye(2),s_z()),self.A_dict["A"+str(i)],optimize='optimal')
                tmp_e2 = np.einsum('bc,bid->dic',tmp_e2,np.conjugate(self.A_dict["A"+str(i+2)]),optimize='optimal')
                tmp_e2 = np.einsum('dic,ij,cjd',tmp_e2,np.kron(s_x(),s_z()),self.A_dict["A"+str(i+2)],optimize='optimal')
                self.E_total_TEBD += (1/2)*(self.J-self.K)*tmp_e2

                self.E_total_TEBD += -(self.m/2)*np.einsum('aib,ij,ajb',np.conjugate(self.A_dict["A"+str(i)]),np.kron(s_z(),s_z()),self.A_dict["A"+str(i)],optimize='optimal')

                tmp_m = np.einsum('aib,ij,ajc->bc',np.conjugate(self.A_dict["A"+str(i)]),np.kron(I(),s_z()),self.A_dict["A"+str(i)],optimize='optimal')
                tmp_m = np.einsum('bc,bid->dic',tmp_m,np.conjugate(self.A_dict["A"+str(i+2)]),optimize='optimal')
                tmp_m = np.einsum('dic,ij,cjd',tmp_m,np.kron(s_z(),I()),self.A_dict["A"+str(i+2)],optimize='optimal')
                self.E_total_TEBD += -(self.m/2)*tmp_m

        # measure tetraquark exp value
        self.tqe = 0
        for i in range(self.L-4):
            tq_tmp = (1/16)*(1 - self.SzSz_build[i] - self.SzSz_build[i+1] - self.SzSz_build[i+2] - self.SzSz_build[i+3] + self.SzISz_build[i] + self.SzISz_build[i+1] + self.SzISz_build[i+2] - self.SzIISz_build[i] - self.SzIISz_build[i+1] + self.SzIIISz_build[i] + self.SzSzSzSz_build[i] + self.SzSzSzSz_build[i+1] + self.SzSzISzSz_build[i] - self.SzISzSzSz_build[i] - self.SzSzSzISz_build[i])
            self.tqe_persite[i] = tq_tmp
            self.tqe += tq_tmp
            
        self.tqe1 = 0
        for i in range(self.L-5):
            tq_tmp1 = (1/16)*(1 - self.SzSz_build[i+2] - self.SzSz_build[i+3] + self.SzISz_build[i+2] - self.SzSz_build[i+4] + self.SzSzSzSz_build[i+2] + self.SzISz_build[i+3] - self.SzIISz_build[i+2] - self.SzSz_build[i] + self.SzSzSzSz_build[i] + self.SzSzISzSz_build[i] - self.SzSzSzISz_build[i] + self.SzSzIISzSz_build[i] - self.SzSzSzSzSzSz_build[i] - self.SzSzISzISz_build[i] + self.SzSzSzIISz_build[i])
            self.tqe1 += tq_tmp1
            
        self.tqe2 = 0
        for i in range(self.L-5):
            tq_tmp2 = (1/16)*(1 - self.SzSz_build[i+2] - self.SzSz_build[i+4] + self.SzSzSzSz_build[i+2] - self.SzSz_build[i+1] + self.SzISz_build[i+1] + self.SzSzISzSz_build[i+1] - self.SzISzSzSz_build[i+1] + self.SzSzSzSz_build[i] - self.SzSz_build[i] + self.SzSzIISzSz_build[i] - self.SzSzSzSzSzSz_build[i] + self.SzISz_build[i] - self.SzIISz_build[i] - self.SzISzISzSz_build[i] + self.SzIISzSzSz_build[i])
            self.tqe2 += tq_tmp2
        
        self.tqe_total = self.tqe + self.tqe1 + self.tqe2
            
        # Measure number of mesons and average meson length
        self.ml_num = 0

        self.n_1meson = 0
        for i in range(self.L-2):
            nm1_tmp = (1/4)*(1-self.SzSz_build[i]-self.SzSz_build[i+1]+self.SzISz_build[i])
            self.n_1meson_persite[i] = nm1_tmp
            self.n_1meson += nm1_tmp
            self.ml_num += 1*nm1_tmp
            
        self.n_2meson = 0
        for i in range(self.L-3):
            nm2_tmp = (1/8)*(1-self.SzSz_build[i]+self.SzSz_build[i+1]-self.SzISz_build[i]-self.SzSz_build[i+2]+self.SzSzSzSz_build[i]-self.SzISz_build[i+1]-self.SzIISz_build[i])
            self.n_2meson_persite[i] = nm2_tmp
            self.n_2meson += nm1_tmp
            self.ml_num += 2*nm2_tmp
            
        self.n_3meson = 0
        for i in range(self.L-4):
            nm3_tmp = (1/16)*(1-self.SzSz_build[i]+self.SzSz_build[i+1]-self.SzISz_build[i]-self.SzSzSzSz_build[i]+self.SzISz_build[i+1]-self.SzIISz_build[i]+self.SzSz_build[i+2]-self.SzSz_build[i+3]+self.SzSzISzSz_build[i]-self.SzSzSzSz_build[i+1]+self.SzISzSzSz_build[i]-self.SzISz_build[i+2]+self.SzSzSzISz_build[i]-self.SzIISz_build[i+1]+self.SzIIISz_build[i])
            self.n_3meson_persite[i] = nm3_tmp
            self.n_3meson = self.n_3meson_persite[int(L/2)-2]
            self.ml_num += 3*nm3_tmp
            
        self.n_4meson = 0
        for i in range(self.L-5):
            nm4_tmp = (1/32)*(1+self.SzSz_build[i+3]-self.SzSz_build[i+4]-self.SzISz_build[i+3]+self.SzSz_build[i+2]+self.SzISz_build[i+2]-self.SzSzSzSz_build[i+2]-self.SzIISz_build[i+2]+self.SzSz_build[i+1]+self.SzSzSzSz_build[i+1]-self.SzSzISzSz_build[i+1]-self.SzSzSzISz_build[i+1]+self.SzISz_build[i+1]+self.SzIISz_build[i+1]-self.SzISzSzSz_build[i+1]-self.SzIIISz_build[i+1]-self.SzSz_build[i]-self.SzSzISzSz_build[i]+self.SzSzIISzSz_build[i]+self.SzSzISzISz_build[i]-self.SzSzSzSz_build[i]-self.SzSzSzISz_build[i]+self.SzSzSzSzSzSz_build[i]+self.SzSzSzIISz_build[i]-self.SzISz_build[i]-self.SzISzSzSz_build[i]+self.SzISzISzSz_build[i]+self.SzISzSzISz_build[i]-self.SzIISz_build[i]-self.SzIIISz_build[i]+self.SzIISzSzSz_build[i]+self.SzIIIISz_build[i])
            self.n_4meson_persite[i] = nm4_tmp
            self.n_4meson += nm4_tmp
            self.ml_num += 4*nm4_tmp
        
        self.avg_ml = self.ml_num/(self.n_1meson+self.n_2meson+self.n_3meson+self.n_4meson)

        
                

            
        
    
    # Returns the wavefunction from the MPS
    def MPS_to_wf(self):
        temp = self.A_dict['A0']
        for i in range(self.L-1):
            temp = np.tensordot(temp,mps_evolve.A_dict['A'+str(i+1)],axes=1)
        self.TEBD_psi = temp.flatten()
    
    # Function to check the TEBD and Schrodinger wavefunction
    def check_schrodinger_psi(self):
        self.MPS_to_wf()
        if np.allclose(self.TEBD_psi,self.schrodinger_psi):
            return True
        else:
            return False
        
    def build_left(self):
        temp = np.reshape(1.+0.*1j,(1,1))
        self.left_trace.append(temp)
        for i in np.arange(2,self.L,2):
            temp = np.einsum('ij,iak->kaj',temp,np.conjugate(self.A_dict["A"+str(i-2)]),optimize='optimal')
            temp = np.einsum('kaj,jal->kl',temp,self.A_dict["A"+str(i-2)],optimize='optimal')
            self.left_trace.append(temp)
        
    def build_right(self):
        temp = np.reshape(1.+0.*1j,(1,1))
        self.right_trace.append(temp)
        loop_arr = np.arange(self.L-4,-4,-2)
        for i in loop_arr:
            temp = np.einsum('ij,kai->kaj',temp,np.conjugate(self.A_dict["A"+str(i+2)]),optimize='optimal')
            temp = np.einsum('kaj,laj->kl',temp,self.A_dict["A"+str(i+2)],optimize='optimal')
            self.right_trace.append(temp)
        self.right_trace.reverse()

    def SzSz(self):
        for ind in range(self.L-1):
            if ind%2 == 0:
                self.lmbd_relocate(ind)
                tmp_szsz = np.einsum('aib,ij,ajb',np.conjugate(self.A_dict["A"+str(ind)]),np.kron(s_z(),s_z()),self.A_dict["A"+str(ind)],optimize='optimal')
            elif ind%2 != 0:
                self.lmbd_relocate(ind-1)
                tmp_szsz = np.einsum('aib,ij,ajc->bc',np.conjugate(self.A_dict["A"+str(ind-1)]),np.kron(I(),s_z()),self.A_dict["A"+str(ind-1)],optimize='optimal')
                tmp_szsz = np.einsum('bc,bid->dic',tmp_szsz,np.conjugate(self.A_dict["A"+str(ind+1)]),optimize='optimal')
                tmp_szsz = np.einsum('dic,ij,cjd',tmp_szsz,np.kron(s_z(),I()),self.A_dict["A"+str(ind+1)],optimize='optimal')
            self.SzSz_build[ind] = tmp_szsz
    
    def SzISz(self):
        for ind in range(self.L-2):
            if ind%2 == 0:
                self.lmbd_relocate(ind)
                szIsz = np.einsum('aib,ij,ajc->bc',np.conjugate(self.A_dict["A"+str(ind)]),np.kron(s_z(),I()),self.A_dict["A"+str(ind)],optimize='optimal')
                szIsz = np.einsum('bc,bid->dic',szIsz,np.conjugate(self.A_dict["A"+str(ind+2)]),optimize='optimal')
                szIsz = np.einsum('dic,ij,cjd',szIsz,np.kron(s_z(),I()),self.A_dict["A"+str(ind+2)],optimize='optimal')
            elif ind%2 != 0:
                self.lmbd_relocate(ind-1)
                szIsz = np.einsum('aib,ij,ajc->bc',np.conjugate(self.A_dict["A"+str(ind-1)]),np.kron(I(),s_z()),self.A_dict["A"+str(ind-1)],optimize='optimal')
                szIsz = np.einsum('bc,bid->dic',szIsz,np.conjugate(self.A_dict["A"+str(ind+1)]),optimize='optimal')
                szIsz = np.einsum('dic,ij,cjd',szIsz,np.kron(I(),s_z()),self.A_dict["A"+str(ind+1)],optimize='optimal')
            self.SzISz_build[ind] = szIsz
    
    def SzIISz(self):
        for ind in range(self.L-3):
            if ind%2 == 0:
                self.lmbd_relocate(ind)
                szIIsz = np.einsum('aib,ij,ajc->bc',np.conjugate(self.A_dict["A"+str(ind)]),np.kron(s_z(),I()),self.A_dict["A"+str(ind)],optimize='optimal')
                szIIsz = np.einsum('bc,bid->dic',szIIsz,np.conjugate(self.A_dict["A"+str(ind+2)]),optimize='optimal')
                szIIsz = np.einsum('dic,ij,cjd',szIIsz,np.kron(I(),s_z()),self.A_dict["A"+str(ind+2)],optimize='optimal')
            elif ind%2 != 0:
                self.lmbd_relocate(ind-1)
                szIIsz = np.einsum('aib,ij,ajc->bc',np.conjugate(self.A_dict["A"+str(ind-1)]),np.kron(I(),s_z()),self.A_dict["A"+str(ind-1)],optimize='optimal')
                szIIsz = np.einsum('bc,bid->dic',szIIsz,np.conjugate(self.A_dict["A"+str(ind+1)]),optimize='optimal')
                szIIsz = np.einsum('dic,ij,cje->de',szIIsz,np.kron(I(),I()),self.A_dict["A"+str(ind+1)],optimize='optimal')
                szIIsz = np.einsum('de,dif->fie',szIIsz,np.conjugate(self.A_dict["A"+str(ind+3)]),optimize='optimal')
                szIIsz = np.einsum('fie,ij,ejf',szIIsz,np.kron(s_z(),I()),self.A_dict["A"+str(ind+3)],optimize='optimal')
            self.SzIISz_build[ind] = szIIsz
    
    def SzIIISz(self):
        for ind in range(self.L-4):
            if ind%2 == 0:
                self.lmbd_relocate(ind)
                szIIIsz = np.einsum('aib,ij,ajc->bc',np.conjugate(self.A_dict["A"+str(ind)]),np.kron(s_z(),I()),self.A_dict["A"+str(ind)],optimize='optimal')
                szIIIsz = np.einsum('bc,bid->dic',szIIIsz,np.conjugate(self.A_dict["A"+str(ind+2)]),optimize='optimal')
                szIIIsz = np.einsum('dic,ij,cje->de',szIIIsz,np.kron(I(),I()),self.A_dict["A"+str(ind+2)],optimize='optimal')
                szIIIsz = np.einsum('de,dif->fie',szIIIsz,np.conjugate(self.A_dict["A"+str(ind+4)]),optimize='optimal')
                szIIIsz = np.einsum('fie,ij,ejf',szIIIsz,np.kron(s_z(),I()),self.A_dict["A"+str(ind+4)],optimize='optimal')
            elif ind%2 != 0:
                self.lmbd_relocate(ind-1)
                szIIIsz = np.einsum('aib,ij,ajc->bc',np.conjugate(self.A_dict["A"+str(ind-1)]),np.kron(I(),s_z()),self.A_dict["A"+str(ind-1)],optimize='optimal')
                szIIIsz = np.einsum('bc,bid->dic',szIIIsz,np.conjugate(self.A_dict["A"+str(ind+1)]),optimize='optimal')
                szIIIsz = np.einsum('dic,ij,cje->de',szIIIsz,np.kron(I(),I()),self.A_dict["A"+str(ind+1)],optimize='optimal')
                szIIIsz = np.einsum('de,dif->fie',szIIIsz,np.conjugate(self.A_dict["A"+str(ind+3)]),optimize='optimal')
                szIIIsz = np.einsum('fie,ij,ejf',szIIIsz,np.kron(I(),s_z()),self.A_dict["A"+str(ind+3)],optimize='optimal')
            self.SzIIISz_build[ind] = szIIIsz
    
    def SzSzSzSz(self):
        for ind in range(self.L-3):
            if ind%2 == 0:
                self.lmbd_relocate(ind)
                szszszsz = np.einsum('aib,ij,ajc->bc',np.conjugate(self.A_dict["A"+str(ind)]),np.kron(s_z(),s_z()),self.A_dict["A"+str(ind)],optimize='optimal')
                szszszsz = np.einsum('bc,bid->dic',szszszsz,np.conjugate(self.A_dict["A"+str(ind+2)]),optimize='optimal')
                szszszsz = np.einsum('dic,ij,cjd',szszszsz,np.kron(s_z(),s_z()),self.A_dict["A"+str(ind+2)],optimize='optimal')
            elif ind%2 != 0:   
                self.lmbd_relocate(ind-1)
                szszszsz = np.einsum('aib,ij,ajc->bc',np.conjugate(self.A_dict["A"+str(ind-1)]),np.kron(I(),s_z()),self.A_dict["A"+str(ind-1)],optimize='optimal')
                szszszsz = np.einsum('bc,bid->dic',szszszsz,np.conjugate(self.A_dict["A"+str(ind+1)]),optimize='optimal')
                szszszsz = np.einsum('dic,ij,cje->de',szszszsz,np.kron(s_z(),s_z()),self.A_dict["A"+str(ind+1)],optimize='optimal')
                szszszsz = np.einsum('de,dif->fie',szszszsz,np.conjugate(self.A_dict["A"+str(ind+3)]),optimize='optimal')
                szszszsz = np.einsum('fie,ij,ejf',szszszsz,np.kron(s_z(),I()),self.A_dict["A"+str(ind+3)],optimize='optimal')
            self.SzSzSzSz_build[ind] = szszszsz
    
    def SzSzISzSz(self):
        for ind in range(self.L-4):
            if ind%2 == 0:
                self.lmbd_relocate(ind)
                szszIszsz = np.einsum('aib,ij,ajc->bc',np.conjugate(self.A_dict["A"+str(ind)]),np.kron(s_z(),s_z()),self.A_dict["A"+str(ind)],optimize='optimal')
                szszIszsz = np.einsum('bc,bid->dic',szszIszsz,np.conjugate(self.A_dict["A"+str(ind+2)]),optimize='optimal')
                szszIszsz = np.einsum('dic,ij,cje->de',szszIszsz,np.kron(I(),s_z()),self.A_dict["A"+str(ind+2)],optimize='optimal')
                szszIszsz = np.einsum('de,dif->fie',szszIszsz,np.conjugate(self.A_dict["A"+str(ind+4)]),optimize='optimal')
                szszIszsz = np.einsum('fie,ij,ejf',szszIszsz,np.kron(s_z(),I()),self.A_dict["A"+str(ind+4)],optimize='optimal')
            elif ind%2 != 0:
                self.lmbd_relocate(ind-1)
                szszIszsz = np.einsum('aib,ij,ajc->bc',np.conjugate(self.A_dict["A"+str(ind-1)]),np.kron(I(),s_z()),self.A_dict["A"+str(ind-1)],optimize='optimal')
                szszIszsz = np.einsum('bc,bid->dic',szszIszsz,np.conjugate(self.A_dict["A"+str(ind+1)]),optimize='optimal')
                szszIszsz = np.einsum('dic,ij,cje->de',szszIszsz,np.kron(s_z(),I()),self.A_dict["A"+str(ind+1)],optimize='optimal')
                szszIszsz = np.einsum('de,dif->fie',szszIszsz,np.conjugate(self.A_dict["A"+str(ind+3)]),optimize='optimal')
                szszIszsz = np.einsum('fie,ij,ejf',szszIszsz,np.kron(s_z(),s_z()),self.A_dict["A"+str(ind+3)],optimize='optimal')
            self.SzSzISzSz_build[ind] = szszIszsz
    
    def SzISzSzSz(self):
        for ind in range(self.L-4):
            if ind%2 == 0:
                self.lmbd_relocate(ind)
                szIszszsz = np.einsum('aib,ij,ajc->bc',np.conjugate(self.A_dict["A"+str(ind)]),np.kron(s_z(),I()),self.A_dict["A"+str(ind)],optimize='optimal')
                szIszszsz = np.einsum('bc,bid->dic',szIszszsz,np.conjugate(self.A_dict["A"+str(ind+2)]),optimize='optimal')
                szIszszsz = np.einsum('dic,ij,cje->de',szIszszsz,np.kron(s_z(),s_z()),self.A_dict["A"+str(ind+2)],optimize='optimal')
                szIszszsz = np.einsum('de,dif->fie',szIszszsz,np.conjugate(self.A_dict["A"+str(ind+4)]),optimize='optimal')
                szIszszsz = np.einsum('fie,ij,ejf',szIszszsz,np.kron(s_z(),I()),self.A_dict["A"+str(ind+4)],optimize='optimal')
            elif ind%2 != 0:
                self.lmbd_relocate(ind-1)
                szIszszsz = np.einsum('aib,ij,ajc->bc',np.conjugate(self.A_dict["A"+str(ind-1)]),np.kron(I(),s_z()),self.A_dict["A"+str(ind-1)],optimize='optimal')
                szIszszsz = np.einsum('bc,bid->dic',szIszszsz,np.conjugate(self.A_dict["A"+str(ind+1)]),optimize='optimal')
                szIszszsz = np.einsum('dic,ij,cje->de',szIszszsz,np.kron(I(),s_z()),self.A_dict["A"+str(ind+1)],optimize='optimal')
                szIszszsz = np.einsum('de,dif->fie',szIszszsz,np.conjugate(self.A_dict["A"+str(ind+3)]),optimize='optimal')
                szIszszsz = np.einsum('fie,ij,ejf',szIszszsz,np.kron(s_z(),s_z()),self.A_dict["A"+str(ind+3)],optimize='optimal')
            self.SzISzSzSz_build[ind] = szIszszsz
    
    def SzSzSzISz(self):
        for ind in range(self.L-4):
            if ind%2 == 0:
                self.lmbd_relocate(ind)
                szIszszsz = np.einsum('aib,ij,ajc->bc',np.conjugate(self.A_dict["A"+str(ind)]),np.kron(s_z(),s_z()),self.A_dict["A"+str(ind)],optimize='optimal')
                szIszszsz = np.einsum('bc,bid->dic',szIszszsz,np.conjugate(self.A_dict["A"+str(ind+2)]),optimize='optimal')
                szIszszsz = np.einsum('dic,ij,cje->de',szIszszsz,np.kron(s_z(),I()),self.A_dict["A"+str(ind+2)],optimize='optimal')
                szIszszsz = np.einsum('de,dif->fie',szIszszsz,np.conjugate(self.A_dict["A"+str(ind+4)]),optimize='optimal')
                szIszszsz = np.einsum('fie,ij,ejf',szIszszsz,np.kron(s_z(),I()),self.A_dict["A"+str(ind+4)],optimize='optimal')
            elif ind%2 != 0:
                self.lmbd_relocate(ind-1)
                szIszszsz = np.einsum('aib,ij,ajc->bc',np.conjugate(self.A_dict["A"+str(ind-1)]),np.kron(I(),s_z()),self.A_dict["A"+str(ind-1)],optimize='optimal')
                szIszszsz = np.einsum('bc,bid->dic',szIszszsz,np.conjugate(self.A_dict["A"+str(ind+1)]),optimize='optimal')
                szIszszsz = np.einsum('dic,ij,cje->de',szIszszsz,np.kron(s_z(),s_z()),self.A_dict["A"+str(ind+1)],optimize='optimal')
                szIszszsz = np.einsum('de,dif->fie',szIszszsz,np.conjugate(self.A_dict["A"+str(ind+3)]),optimize='optimal')
                szIszszsz = np.einsum('fie,ij,ejf',szIszszsz,np.kron(I(),s_z()),self.A_dict["A"+str(ind+3)],optimize='optimal')
            self.SzSzSzISz_build[ind] = szIszszsz
            
    def SzSzIISzSz(self):
        for ind in range(self.L-5):
            if ind%2 == 0:
                self.lmbd_relocate(ind)
                szszIIszsz = np.einsum('aib,ij,ajc->bc',np.conjugate(self.A_dict["A"+str(ind)]),np.kron(s_z(),s_z()),self.A_dict["A"+str(ind)],optimize='optimal')
                szszIIszsz = np.einsum('bc,bid->dic',szszIIszsz,np.conjugate(self.A_dict["A"+str(ind+2)]),optimize='optimal')
                szszIIszsz = np.einsum('dic,ij,cje->de',szszIIszsz,np.kron(I(),I()),self.A_dict["A"+str(ind+2)],optimize='optimal')
                szszIIszsz = np.einsum('de,dif->fie',szszIIszsz,np.conjugate(self.A_dict["A"+str(ind+4)]),optimize='optimal')
                szszIIszsz = np.einsum('fie,ij,ejf',szszIIszsz,np.kron(s_z(),s_z()),self.A_dict["A"+str(ind+4)],optimize='optimal')
            elif ind%2 != 0:
                self.lmbd_relocate(ind-1)
                szszIIszsz = np.einsum('aib,ij,ajc->bc',np.conjugate(self.A_dict["A"+str(ind-1)]),np.kron(I(),s_z()),self.A_dict["A"+str(ind-1)],optimize='optimal')
                szszIIszsz = np.einsum('bc,bid->dic',szszIIszsz,np.conjugate(self.A_dict["A"+str(ind+1)]),optimize='optimal')
                szszIIszsz = np.einsum('dic,ij,cje->de',szszIIszsz,np.kron(s_z(),I()),self.A_dict["A"+str(ind+1)],optimize='optimal')
                szszIIszsz = np.einsum('de,dif->fie',szszIIszsz,np.conjugate(self.A_dict["A"+str(ind+3)]),optimize='optimal')
                szszIIszsz = np.einsum('fie,ij,ejg->fg',szszIIszsz,np.kron(I(),s_z()),self.A_dict["A"+str(ind+3)],optimize='optimal')
                szszIIszsz = np.einsum('fg,fih->hig',szszIIszsz,np.conjugate(self.A_dict["A"+str(ind+5)]),optimize='optimal')
                szszIIszsz = np.einsum('hig,ij,gjh',szszIIszsz,np.kron(s_z(),I()),self.A_dict["A"+str(ind+5)],optimize='optimal')
            self.SzSzIISzSz_build[ind] = szszIIszsz
        
    def SzSzSzSzSzSz(self):
        for ind in range(self.L-5):
            if ind%2 == 0:
                self.lmbd_relocate(ind)
                szszszszszsz = np.einsum('aib,ij,ajc->bc',np.conjugate(self.A_dict["A"+str(ind)]),np.kron(s_z(),s_z()),self.A_dict["A"+str(ind)],optimize='optimal')
                szszszszszsz = np.einsum('bc,bid->dic',szszszszszsz,np.conjugate(self.A_dict["A"+str(ind+2)]),optimize='optimal')
                szszszszszsz = np.einsum('dic,ij,cje->de',szszszszszsz,np.kron(s_z(),s_z()),self.A_dict["A"+str(ind+2)],optimize='optimal')
                szszszszszsz = np.einsum('de,dif->fie',szszszszszsz,np.conjugate(self.A_dict["A"+str(ind+4)]),optimize='optimal')
                szszszszszsz = np.einsum('fie,ij,ejf',szszszszszsz,np.kron(s_z(),s_z()),self.A_dict["A"+str(ind+4)],optimize='optimal')
            elif ind%2 != 0:
                self.lmbd_relocate(ind-1)
                szszszszszsz = np.einsum('aib,ij,ajc->bc',np.conjugate(self.A_dict["A"+str(ind-1)]),np.kron(I(),s_z()),self.A_dict["A"+str(ind-1)],optimize='optimal')
                szszszszszsz = np.einsum('bc,bid->dic',szszszszszsz,np.conjugate(self.A_dict["A"+str(ind+1)]),optimize='optimal')
                szszszszszsz = np.einsum('dic,ij,cje->de',szszszszszsz,np.kron(s_z(),s_z()),self.A_dict["A"+str(ind+1)],optimize='optimal')
                szszszszszsz = np.einsum('de,dif->fie',szszszszszsz,np.conjugate(self.A_dict["A"+str(ind+3)]),optimize='optimal')
                szszszszszsz = np.einsum('fie,ij,ejg->fg',szszszszszsz,np.kron(s_z(),s_z()),self.A_dict["A"+str(ind+3)],optimize='optimal')
                szszszszszsz = np.einsum('fg,fih->hig',szszszszszsz,np.conjugate(self.A_dict["A"+str(ind+5)]),optimize='optimal')
                szszszszszsz = np.einsum('hig,ij,gjh',szszszszszsz,np.kron(s_z(),I()),self.A_dict["A"+str(ind+5)],optimize='optimal')
            self.SzSzSzSzSzSz_build[ind] = szszszszszsz
            
    def SzSzSzIISz(self):
        for ind in range(self.L-5):
            if ind%2 == 0:
                self.lmbd_relocate(ind)
                szszszIIsz = np.einsum('aib,ij,ajc->bc',np.conjugate(self.A_dict["A"+str(ind)]),np.kron(s_z(),s_z()),self.A_dict["A"+str(ind)],optimize='optimal')
                szszszIIsz = np.einsum('bc,bid->dic',szszszIIsz,np.conjugate(self.A_dict["A"+str(ind+2)]),optimize='optimal')
                szszszIIsz = np.einsum('dic,ij,cje->de',szszszIIsz,np.kron(s_z(),I()),self.A_dict["A"+str(ind+2)],optimize='optimal')
                szszszIIsz = np.einsum('de,dif->fie',szszszIIsz,np.conjugate(self.A_dict["A"+str(ind+4)]),optimize='optimal')
                szszszIIsz = np.einsum('fie,ij,ejf',szszszIIsz,np.kron(I(),s_z()),self.A_dict["A"+str(ind+4)],optimize='optimal')
            elif ind%2 != 0:
                self.lmbd_relocate(ind-1)
                szszszIIsz = np.einsum('aib,ij,ajc->bc',np.conjugate(self.A_dict["A"+str(ind-1)]),np.kron(I(),s_z()),self.A_dict["A"+str(ind-1)],optimize='optimal')
                szszszIIsz = np.einsum('bc,bid->dic',szszszIIsz,np.conjugate(self.A_dict["A"+str(ind+1)]),optimize='optimal')
                szszszIIsz = np.einsum('dic,ij,cje->de',szszszIIsz,np.kron(s_z(),s_z()),self.A_dict["A"+str(ind+1)],optimize='optimal')
                szszszIIsz = np.einsum('de,dif->fie',szszszIIsz,np.conjugate(self.A_dict["A"+str(ind+3)]),optimize='optimal')
                szszszIIsz = np.einsum('fie,ij,ejg->fg',szszszIIsz,np.kron(I(),I()),self.A_dict["A"+str(ind+3)],optimize='optimal')
                szszszIIsz = np.einsum('fg,fih->hig',szszszIIsz,np.conjugate(self.A_dict["A"+str(ind+5)]),optimize='optimal')
                szszszIIsz = np.einsum('hig,ij,gjh',szszszIIsz,np.kron(s_z(),I()),self.A_dict["A"+str(ind+5)],optimize='optimal')
            self.SzSzSzIISz_build[ind] = szszszIIsz
            
    def SzIISzSzSz(self):
        for ind in range(self.L-5):
            if ind%2 == 0:
                self.lmbd_relocate(ind)
                szIIszszsz = np.einsum('aib,ij,ajc->bc',np.conjugate(self.A_dict["A"+str(ind)]),np.kron(s_z(),I()),self.A_dict["A"+str(ind)],optimize='optimal')
                szIIszszsz = np.einsum('bc,bid->dic',szIIszszsz,np.conjugate(self.A_dict["A"+str(ind+2)]),optimize='optimal')
                szIIszszsz = np.einsum('dic,ij,cje->de',szIIszszsz,np.kron(I(),s_z()),self.A_dict["A"+str(ind+2)],optimize='optimal')
                szIIszszsz = np.einsum('de,dif->fie',szIIszszsz,np.conjugate(self.A_dict["A"+str(ind+4)]),optimize='optimal')
                szIIszszsz = np.einsum('fie,ij,ejf',szIIszszsz,np.kron(s_z(),s_z()),self.A_dict["A"+str(ind+4)],optimize='optimal')
            elif ind%2 != 0:
                self.lmbd_relocate(ind-1)
                szIIszszsz = np.einsum('aib,ij,ajc->bc',np.conjugate(self.A_dict["A"+str(ind-1)]),np.kron(I(),s_z()),self.A_dict["A"+str(ind-1)],optimize='optimal')
                szIIszszsz = np.einsum('bc,bid->dic',szIIszszsz,np.conjugate(self.A_dict["A"+str(ind+1)]),optimize='optimal')
                szIIszszsz = np.einsum('dic,ij,cje->de',szIIszszsz,np.kron(I(),I()),self.A_dict["A"+str(ind+1)],optimize='optimal')
                szIIszszsz = np.einsum('de,dif->fie',szIIszszsz,np.conjugate(self.A_dict["A"+str(ind+3)]),optimize='optimal')
                szIIszszsz = np.einsum('fie,ij,ejg->fg',szIIszszsz,np.kron(s_z(),s_z()),self.A_dict["A"+str(ind+3)],optimize='optimal')
                szIIszszsz = np.einsum('fg,fih->hig',szIIszszsz,np.conjugate(self.A_dict["A"+str(ind+5)]),optimize='optimal')
                szIIszszsz = np.einsum('hig,ij,gjh',szIIszszsz,np.kron(s_z(),I()),self.A_dict["A"+str(ind+5)],optimize='optimal')
            self.SzIISzSzSz_build[ind] = szIIszszsz
            
    def SzSzISzISz(self):
        for ind in range(self.L-5):
            if ind%2 == 0:
                self.lmbd_relocate(ind)
                szszIszIsz = np.einsum('aib,ij,ajc->bc',np.conjugate(self.A_dict["A"+str(ind)]),np.kron(s_z(),s_z()),self.A_dict["A"+str(ind)],optimize='optimal')
                szszIszIsz = np.einsum('bc,bid->dic',szszIszIsz,np.conjugate(self.A_dict["A"+str(ind+2)]),optimize='optimal')
                szszIszIsz = np.einsum('dic,ij,cje->de',szszIszIsz,np.kron(I(),s_z()),self.A_dict["A"+str(ind+2)],optimize='optimal')
                szszIszIsz = np.einsum('de,dif->fie',szszIszIsz,np.conjugate(self.A_dict["A"+str(ind+4)]),optimize='optimal')
                szszIszIsz = np.einsum('fie,ij,ejf',szszIszIsz,np.kron(I(),s_z()),self.A_dict["A"+str(ind+4)],optimize='optimal')
            elif ind%2 != 0:
                self.lmbd_relocate(ind-1)
                szszIszIsz = np.einsum('aib,ij,ajc->bc',np.conjugate(self.A_dict["A"+str(ind-1)]),np.kron(I(),s_z()),self.A_dict["A"+str(ind-1)],optimize='optimal')
                szszIszIsz = np.einsum('bc,bid->dic',szszIszIsz,np.conjugate(self.A_dict["A"+str(ind+1)]),optimize='optimal')
                szszIszIsz = np.einsum('dic,ij,cje->de',szszIszIsz,np.kron(s_z(),I()),self.A_dict["A"+str(ind+1)],optimize='optimal')
                szszIszIsz = np.einsum('de,dif->fie',szszIszIsz,np.conjugate(self.A_dict["A"+str(ind+3)]),optimize='optimal')
                szszIszIsz = np.einsum('fie,ij,ejg->fg',szszIszIsz,np.kron(s_z(),I()),self.A_dict["A"+str(ind+3)],optimize='optimal')
                szszIszIsz = np.einsum('fg,fih->hig',szszIszIsz,np.conjugate(self.A_dict["A"+str(ind+5)]),optimize='optimal')
                szszIszIsz = np.einsum('hig,ij,gjh',szszIszIsz,np.kron(s_z(),I()),self.A_dict["A"+str(ind+5)],optimize='optimal')
            self.SzSzISzISz_build[ind] = szszIszIsz
            
    def SzISzISzSz(self):
        for ind in range(self.L-5):
            if ind%2 == 0:
                self.lmbd_relocate(ind)
                szIszIszsz = np.einsum('aib,ij,ajc->bc',np.conjugate(self.A_dict["A"+str(ind)]),np.kron(s_z(),I()),self.A_dict["A"+str(ind)],optimize='optimal')
                szIszIszsz = np.einsum('bc,bid->dic',szIszIszsz,np.conjugate(self.A_dict["A"+str(ind+2)]),optimize='optimal')
                szIszIszsz = np.einsum('dic,ij,cje->de',szIszIszsz,np.kron(s_z(),I()),self.A_dict["A"+str(ind+2)],optimize='optimal')
                szIszIszsz = np.einsum('de,dif->fie',szIszIszsz,np.conjugate(self.A_dict["A"+str(ind+4)]),optimize='optimal')
                szIszIszsz = np.einsum('fie,ij,ejf',szIszIszsz,np.kron(s_z(),s_z()),self.A_dict["A"+str(ind+4)],optimize='optimal')
            elif ind%2 != 0:
                self.lmbd_relocate(ind-1)
                szIszIszsz = np.einsum('aib,ij,ajc->bc',np.conjugate(self.A_dict["A"+str(ind-1)]),np.kron(I(),s_z()),self.A_dict["A"+str(ind-1)],optimize='optimal')
                szIszIszsz = np.einsum('bc,bid->dic',szIszIszsz,np.conjugate(self.A_dict["A"+str(ind+1)]),optimize='optimal')
                szIszIszsz = np.einsum('dic,ij,cje->de',szIszIszsz,np.kron(I(),s_z()),self.A_dict["A"+str(ind+1)],optimize='optimal')
                szIszIszsz = np.einsum('de,dif->fie',szIszIszsz,np.conjugate(self.A_dict["A"+str(ind+3)]),optimize='optimal')
                szIszIszsz = np.einsum('fie,ij,ejg->fg',szIszIszsz,np.kron(I(),s_z()),self.A_dict["A"+str(ind+3)],optimize='optimal')
                szIszIszsz = np.einsum('fg,fih->hig',szIszIszsz,np.conjugate(self.A_dict["A"+str(ind+5)]),optimize='optimal')
                szIszIszsz = np.einsum('hig,ij,gjh',szIszIszsz,np.kron(s_z(),I()),self.A_dict["A"+str(ind+5)],optimize='optimal')
            self.SzISzISzSz_build[ind] = szIszIszsz
            
    def SzSzIISzSz(self):
        for ind in range(self.L-5):
            if ind%2 == 0:
                self.lmbd_relocate(ind)
                szszIIszsz = np.einsum('aib,ij,ajc->bc',np.conjugate(self.A_dict["A"+str(ind)]),np.kron(s_z(),s_z()),self.A_dict["A"+str(ind)],optimize='optimal')
                szszIIszsz = np.einsum('bc,bid->dic',szszIIszsz,np.conjugate(self.A_dict["A"+str(ind+2)]),optimize='optimal')
                szszIIszsz = np.einsum('dic,ij,cje->de',szszIIszsz,np.kron(I(),I()),self.A_dict["A"+str(ind+2)],optimize='optimal')
                szszIIszsz = np.einsum('de,dif->fie',szszIIszsz,np.conjugate(self.A_dict["A"+str(ind+4)]),optimize='optimal')
                szszIIszsz = np.einsum('fie,ij,ejf',szszIIszsz,np.kron(s_z(),s_z()),self.A_dict["A"+str(ind+4)],optimize='optimal')
            elif ind%2 != 0:
                self.lmbd_relocate(ind-1)
                szszIIszsz = np.einsum('aib,ij,ajc->bc',np.conjugate(self.A_dict["A"+str(ind-1)]),np.kron(I(),s_z()),self.A_dict["A"+str(ind-1)],optimize='optimal')
                szszIIszsz = np.einsum('bc,bid->dic',szszIIszsz,np.conjugate(self.A_dict["A"+str(ind+1)]),optimize='optimal')
                szszIIszsz = np.einsum('dic,ij,cje->de',szszIIszsz,np.kron(s_z(),I()),self.A_dict["A"+str(ind+1)],optimize='optimal')
                szszIIszsz = np.einsum('de,dif->fie',szszIIszsz,np.conjugate(self.A_dict["A"+str(ind+3)]),optimize='optimal')
                szszIIszsz = np.einsum('fie,ij,ejg->fg',szszIIszsz,np.kron(I(),s_z()),self.A_dict["A"+str(ind+3)],optimize='optimal')
                szszIIszsz = np.einsum('fg,fih->hig',szszIIszsz,np.conjugate(self.A_dict["A"+str(ind+5)]),optimize='optimal')
                szszIIszsz = np.einsum('hig,ij,gjh',szszIIszsz,np.kron(s_z(),I()),self.A_dict["A"+str(ind+5)],optimize='optimal')
            self.SzSzIISzSz_build[ind] = szszIIszsz
            
    def SzISzSzISz(self):
        for ind in range(self.L-5):
            if ind%2 == 0:
                self.lmbd_relocate(ind)
                szIszszIsz = np.einsum('aib,ij,ajc->bc',np.conjugate(self.A_dict["A"+str(ind)]),np.kron(s_z(),I()),self.A_dict["A"+str(ind)],optimize='optimal')
                szIszszIsz = np.einsum('bc,bid->dic',szIszszIsz,np.conjugate(self.A_dict["A"+str(ind+2)]),optimize='optimal')
                szIszszIsz = np.einsum('dic,ij,cje->de',szIszszIsz,np.kron(s_z(),s_z()),self.A_dict["A"+str(ind+2)],optimize='optimal')
                szIszszIsz = np.einsum('de,dif->fie',szIszszIsz,np.conjugate(self.A_dict["A"+str(ind+4)]),optimize='optimal')
                szIszszIsz = np.einsum('fie,ij,ejf',szIszszIsz,np.kron(I(),s_z()),self.A_dict["A"+str(ind+4)],optimize='optimal')
            elif ind%2 != 0:
                self.lmbd_relocate(ind-1)
                szIszszIsz = np.einsum('aib,ij,ajc->bc',np.conjugate(self.A_dict["A"+str(ind-1)]),np.kron(I(),s_z()),self.A_dict["A"+str(ind-1)],optimize='optimal')
                szIszszIsz = np.einsum('bc,bid->dic',szIszszIsz,np.conjugate(self.A_dict["A"+str(ind+1)]),optimize='optimal')
                szIszszIsz = np.einsum('dic,ij,cje->de',szIszszIsz,np.kron(I(),s_z()),self.A_dict["A"+str(ind+1)],optimize='optimal')
                szIszszIsz = np.einsum('de,dif->fie',szIszszIsz,np.conjugate(self.A_dict["A"+str(ind+3)]),optimize='optimal')
                szIszszIsz = np.einsum('fie,ij,ejg->fg',szIszszIsz,np.kron(s_z(),I()),self.A_dict["A"+str(ind+3)],optimize='optimal')
                szIszszIsz = np.einsum('fg,fih->hig',szIszszIsz,np.conjugate(self.A_dict["A"+str(ind+5)]),optimize='optimal')
                szIszszIsz = np.einsum('hig,ij,gjh',szIszszIsz,np.kron(s_z(),I()),self.A_dict["A"+str(ind+5)],optimize='optimal')
            self.SzISzSzISz_build[ind] = szIszszIsz
            
    def SzIIIISz(self):
        for ind in range(self.L-5):
            if ind%2 == 0:
                self.lmbd_relocate(ind)
                szIIIIsz = np.einsum('aib,ij,ajc->bc',np.conjugate(self.A_dict["A"+str(ind)]),np.kron(s_z(),I()),self.A_dict["A"+str(ind)],optimize='optimal')
                szIIIIsz = np.einsum('bc,bid->dic',szIIIIsz,np.conjugate(self.A_dict["A"+str(ind+2)]),optimize='optimal')
                szIIIIsz = np.einsum('dic,ij,cje->de',szIIIIsz,np.kron(I(),I()),self.A_dict["A"+str(ind+2)],optimize='optimal')
                szIIIIsz = np.einsum('de,dif->fie',szIIIIsz,np.conjugate(self.A_dict["A"+str(ind+4)]),optimize='optimal')
                szIIIIsz = np.einsum('fie,ij,ejf',szIIIIsz,np.kron(I(),s_z()),self.A_dict["A"+str(ind+4)],optimize='optimal')
            elif ind%2 != 0:
                self.lmbd_relocate(ind-1)
                szIIIIsz = np.einsum('aib,ij,ajc->bc',np.conjugate(self.A_dict["A"+str(ind-1)]),np.kron(I(),s_z()),self.A_dict["A"+str(ind-1)],optimize='optimal')
                szIIIIsz = np.einsum('bc,bid->dic',szIIIIsz,np.conjugate(self.A_dict["A"+str(ind+1)]),optimize='optimal')
                szIIIIsz = np.einsum('dic,ij,cje->de',szIIIIsz,np.kron(I(),I()),self.A_dict["A"+str(ind+1)],optimize='optimal')
                szIIIIsz = np.einsum('de,dif->fie',szIIIIsz,np.conjugate(self.A_dict["A"+str(ind+3)]),optimize='optimal')
                szIIIIsz = np.einsum('fie,ij,ejg->fg',szIIIIsz,np.kron(I(),I()),self.A_dict["A"+str(ind+3)],optimize='optimal')
                szIIIIsz = np.einsum('fg,fih->hig',szIIIIsz,np.conjugate(self.A_dict["A"+str(ind+5)]),optimize='optimal')
                szIIIIsz = np.einsum('hig,ij,gjh',szIIIIsz,np.kron(s_z(),I()),self.A_dict["A"+str(ind+5)],optimize='optimal')
            self.SzIIIISz_build[ind] = szIIIIsz

# Main code

chi = #CHI#
sch_bool = False
bound_diff = False
L = #LL#
T = #TT#
N = #NN#


ni_psite = np.zeros((L-1,N),dtype=np.complex128)
tqe_psite = np.zeros((L-4,N),dtype=np.complex128)
n_1meson_psite = np.zeros((L-2,N),dtype=np.complex128)
n_3meson_psite = np.zeros((L-4,N),dtype=np.complex128)
Et_TEBD = []
nt_TEBD = []
tqe_TEBD = []
tqe_tot_TEBD = []
tqe_width = []
tqe_long = []
n1m_TEBD = []
n2m_TEBD = []
n3m_TEBD = []
n4m_TEBD = []
av_ml_TEBD = []


mps_evolve = MPS(L,chi,T,N)
if exists("py_print.txt"):
    f = open("py_print.txt","w")
    f.write('New run\n')
    f.close()
else:
    f = open("py_print.txt","x")
    f.write('New run\n')
    f.close()
for i in range(N):
    t1 = time.time()
    mps_evolve.sweepU()
    t2 = time.time()
    f = open("py_print.txt","a")
    f.write('Time step = '+str(i)+', time taken = '+str(t2-t1)+' for each step\n')
    print('Time step = '+str(i)+', time taken = '+str(t2-t1)+' for each step')
    f.close()
    
    for j in range(L-1):
        ni_psite[j][i] = mps_evolve.ni_persite[j]

    for j in range(L-4):
        tqe_psite[j][i] = mps_evolve.tqe_persite[j]
    
    for j in range(L-2):
        n_1meson_psite[j][i] = mps_evolve.n_1meson_persite[j]
        
    for j in range(L-4):
        n_3meson_psite[j][i] = mps_evolve.n_3meson_persite[j]
    
    Et_TEBD.append(mps_evolve.E_total_TEBD)
    nt_TEBD.append(mps_evolve.n_tot)
    tqe_TEBD.append(mps_evolve.tqe)
    tqe_width.append(mps_evolve.tqe_width)
    tqe_long.append(mps_evolve.tqe1 + mps_evolve.tqe2)
    n1m_TEBD.append(mps_evolve.n_1meson)
    n2m_TEBD.append(mps_evolve.n_2meson)
    n3m_TEBD.append(mps_evolve.n_3meson)
    n4m_TEBD.append(mps_evolve.n_4meson)
    av_ml_TEBD.append(mps_evolve.avg_ml)
    tqe_tot_TEBD.append(mps_evolve.tqe_total)
        
# Saving data
final_array = np.zeros((N,4*L))
col_names = []
for i in range(L-1):
    col_names.append('ni'+str(i))
for i in range(L-4):
    col_names.append('tqe'+str(i))
for i in range(L-2):
    col_names.append('n_1meson'+str(i))
for i in range(L-4):
    col_names.append('n_3meson'+str(i))
col_names.append('E-total')
col_names.append('n-total')
col_names.append('Tqe-1')
col_names.append('Tqe-width')
col_names.append('Tqe-long')
col_names.append('n1-meson')
col_names.append('n2-meson')
col_names.append('n3-meson')
col_names.append('n4-meson')
col_names.append('Avg-meson')
col_names.append('Tqe-total')


    
df = pd.DataFrame(final_array,columns=np.array(col_names))

    
for i in range(L-1):
    temp_list_ni = ni_psite[i]
    df['ni'+str(i)] = np.real(temp_list_ni)
    
for i in range(L-4):
    df['tqe'+str(i)] = np.real(tqe_psite[i])
    
for i in range(L-2):
    df['n_1meson'+str(i)] = np.real(n_1meson_psite[i])
    
for i in range(L-4):
    df['n_3meson'+str(i)] = np.real(n_3meson_psite[i])

temp_list_Et = np.array(Et_TEBD)
df['E-total'] = np.real(temp_list_Et)

df['n-total'] = np.real(np.array(nt_TEBD))

df['Tqe-1'] = np.real(np.array(tqe_TEBD))

df['Tqe-width'] = np.real(np.array(tqe_width))

df['Tqe-long'] = np.real(np.array(tqe_long))

df['n1-meson'] = np.real(np.array(n1m_TEBD))

df['n2-meson'] = np.real(np.array(n2m_TEBD))

df['n3-meson'] = np.real(np.array(n3m_TEBD))

df['n4-meson'] = np.real(np.array(n4m_TEBD))

df['Avg-meson'] = np.real(np.array(av_ml_TEBD))

df['Tqe-total'] = np.real(np.array(tqe_tot_TEBD))


df.to_csv('MPS_lattice_vac_L'+str(L)+'_chi'+str(chi)+'_T'+str(T)+'_N'+str(N)+'_J'+str(mps_evolve.J)+'_K'+str(mps_evolve.K)+'_m'+str(mps_evolve.m)+'_h'+str(mps_evolve.h)+'.csv')


