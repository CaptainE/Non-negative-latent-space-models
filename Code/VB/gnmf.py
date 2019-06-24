from scipy.special import gammaln,digamma
from scipy.stats import gamma
from scipy import exp,log
import numpy as np
from gnmf_solvebynewton import gnmf_solvebynewton

class nmf_vb:
	def __init__(self,a_tm, b_tm, a_ve, b_ve,tie_a_tm, tie_b_tm, tie_a_ve, tie_b_ve):
		self.a_tm = a_tm
		self.b_tm = b_tm
		self.a_ve = a_ve
		self.b_ve = b_ve
		self.W = b_tm.shape[0]
		self.K = a_ve.shape[1]
		self.I = b_tm.shape[1]
		self.K = a_ve.shape[1]
		self.tie_a_tm = tie_a_tm
		self.tie_b_tm = tie_b_tm
		self.tie_a_ve = tie_a_ve
		self.tie_b_ve = tie_b_ve
        
        
	def VB_parm_calc(self,X,epoch=1000,update=10,print_period=500):
		# step (1) initialise
		t_init = gamma.rvs(self.a_tm,self.b_tm/self.a_tm)
		v_init = gamma.rvs(self.a_ve,self.b_ve/self.a_ve)
		
		X[np.isnan(X)] = 0
		M = np.ones(X.shape)
		M[np.isnan(X)] = 0
		L_t = t_init
		L_v = v_init
		E_t = t_init
		E_v = v_init
		Sig_t = t_init
		Sig_v = v_init

		# for step (5)
		gammalnX = gammaln(X+1)

		# step (2) for-loop
		for e in range(epoch):
			# step (3) sufficient statistics
			LtLv = L_t@L_v
			tmp = X/LtLv
			Sig_t = L_t*(tmp@L_v.T)
			Sig_v = L_v*(L_t.T@tmp)

			# step (4) means
			alpha_tm = self.a_tm+Sig_t
			beta_tm = 1/((self.a_tm/self.b_tm) +(M@E_v.T))
			E_t = alpha_tm*beta_tm
			
			alpha_ve = self.a_ve+Sig_v
			beta_ve = 1/((self.a_ve/self.b_ve)+(E_t.T@M))
			E_v = alpha_ve*beta_ve
			
            #Compute the bound and get approximated matrices g_E_T, g_E_V
			if (e+1) % print_period == 0 or e == epoch-1:
				print("Optimzing: " + str(e+1) + '/' + str(epoch))
				g_E_T = E_t
				g_E_logT = np.log(L_t+1e-10)
				g_E_V = E_v
				g_E_logV = np.log(L_v+1e-10)

				# step (5) compute bound
				g_Bound =- sum(sum(M*(g_E_T@g_E_V) +  gammalnX )) + sum(sum( -X*( ((L_t*g_E_logT)@L_v + L_t@(L_v*g_E_logV))/(LtLv) -  log(LtLv) )  ))  + sum(sum(-self.a_tm/self.b_tm*g_E_T - gammaln(self.a_tm) + self.a_tm*log(self.a_tm/self.b_tm)  ))   + sum(sum(-self.a_ve/self.b_ve*g_E_V - gammaln(self.a_ve) + self.a_ve*log(self.a_ve/self.b_ve)  ))  + sum(sum( gammaln(alpha_tm) + alpha_tm*(log(beta_tm) + 1)  ))   + sum(sum( gammaln(alpha_ve)  + alpha_ve*(log(beta_ve) + 1)  ))

				g = (g_E_T,g_E_logT,g_E_V,g_E_logV,g_Bound,self.a_ve,self.b_ve,self.a_tm,self.b_tm)
				
			# step (6) means of logs
			L_t = exp(digamma(alpha_tm))*beta_tm
			L_v = exp(digamma(alpha_ve))*beta_ve
			
			# step (7) update hyperparameters
			if e > update:
				if self.tie_a_tm != 'clamp':
					Z = E_t/self.b_tm - (log(L_t) - log(self.b_tm))
					if self.tie_a_tm == 'free':
						self.a_tm = gnmf_solvebynewton(Z,self.a_tm)
					else:
						self.a_tm = gnmf_solvebynewton((sum(Z.ravel())/(self.W*self.I)).reshape(-1,1),self.a_tm)

				if self.tie_b_tm == 'free':
					self.b_tm = E_t
				elif self.tie_b_tm == 'tie_all':
					self.b_tm = sum(sum(self.a_tm*E_t))/sum(self.a_tm.ravel())*np.ones((self.W,self.I))

				if self.tie_a_ve != 'clamp':
					Z = E_v/self.b_ve - (log(L_v) - log(self.b_ve))
					if self.tie_a_ve == 'free':
						self.a_ve = gnmf_solvebynewton(Z,self.a_ve)
					else:
						self.a_ve = gnmf_solvebynewton((sum(Z.ravel())/(self.I*self.K)).reshape(-1,1),self.a_ve)

				if self.tie_b_ve == 'free':
					self.b_ve = E_v
				elif self.tie_b_ve == 'tie_all':
					self.b_ve = sum(sum(self.a_ve*E_v))/sum(self.a_ve.ravel())*np.ones((self.I,self.K))
		return g