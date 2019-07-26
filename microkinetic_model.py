import numpy as np

#To run the model at any pH, remove the comment at the start of the line in both reaction energies and activation barriers sections
#Current settings produce Figure S16 and Figure S17 (at pH = 7, T = 350 K)

# Reaction conditions
T = 350 # K
PHNO2 = 0.03 # bar
PH2 = 0.02 # bar
PH2O = 1 # bar
PNH3 = 0.0 #bar
A = 0.8  # Scaling parameter for activation barriers 

# Physical constants and conversion factors
J2eV = 6.24150974E18 # eV/J
Na = 6.0221415E23 # mol-1
h = 6.626068E-34 * J2eV # in eV*s
kb = 1.3806503E-23 * J2eV # in eV/K
kbT = kb * T # in eV

def get_rate_constants(dGrxn,dGact):
   # Calculate equilibrium and rate constants
   K =[0]*len(dGrxn) # equilibrium constants for 19 reactions
   kf = [0]*len(dGact) # forward rate constants for 19 reactions
   kr = [0]*len(dGact) # reverse rate constants for 19 reactions
   
   for i in range(len(dGrxn)):
       K [i] = np.exp(-dGrxn [i]/kbT)
       kf [i] = kbT/h * np.exp(-dGact [i]/kbT)
       kr [i] = kf [i]/K [i] # enforce thermodynamic consistency
   return (kf,kr)

def get_rates(theta,kf,kr):
   # returns the rates depending on the current coverages theta
   # Extract elements of theta and assign them
   # to more meaningful variables
   tNO = theta [0] # theta of NO
   tOH = theta [1] # theta of OH
   tO = theta [2] # theta of O
   tN = theta [3] # theta of N
   tNH = theta [4] # theta of NH
   tNH2 = theta [5] # theta of NH2
   tNH3 = theta [6] # theta of NH3
   tH2O = theta [7] # theta of H2O
   tH = theta [8] # theta of H
   tNO2 = theta [9] # theta of NO2
   tNOH = theta [10] #theta of NOH
   tHNO = theta [11] #theta of HNO
   tHNOH = theta [12] #theta of HNOH
   tstar = 1.0 - tNO - tOH - tO - tN - tNH - tNH2 - tNH3 - tH2O - tH - tNO2 - tNOH - tHNO - tHNOH  # site balance for tstar
   
   # Caluclate the rates:
   rate = [0]*19 # array with 19 zeros, one for each reaction
   
   rate [0] = kf [0] * PHNO2 * tstar * tstar   - kr [0] * tNO * tOH        # HNO2 + 2* <--> NO* + OH*
   rate [1] = kf [1] * PH2 * tstar * tstar     - kr [1] * tH * tH          # H2 + 2* <--> 2H*
   rate [2] = kf [2] * tOH * tH                - kr [2] * tH2O * tstar     # OH* + H* <--> H2O* + * 
   rate [3] = kf [3] * tNO * tstar             - kr [3] * tN * tO          # NO* + * <--> N* + O*
   rate [4] = kf [4] * tO * tH                 - kr [4] * tOH * tstar      # O* + H* <--> OH* + *
   rate [5] = kf [5] * tN * tH                 - kr [5] * tNH * tstar      # N* + H* <--> NH* + *
   rate [6] = kf [6] * tNH * tH                - kr [6] * tNH2 * tstar     # NH* + H* <--> NH2* + *
   rate [7] = kf [7] * tNH2 * tH               - kr [7] * tNH3 * tstar     # NH2* + H* <--> NH3* + * 
   rate [8] = kf [8] * tNH3                    - kr [8] * PNH3 * tstar     # NH3* <--> NH3 + *
   rate [9] = kf [9] * tH2O                    - kr [9] * PH2O * tstar     # H2O* <--> H2O + *
   rate [10] = kf [10] * PHNO2 * tstar * tstar - kr [10] * tNO2 * tH       # HNO2 + 2* <--> NO2* + H*
   rate [11] = kf [11] * tNO2 * tstar          - kr [11] * tNO * tO        # NO2* + * <--> NO* + O*
   rate [12] = kf [12] * tNO * tH              - kr [12] * tNOH * tstar    # NO* + H* <--> NOH* + *
   rate [13] = kf [13] * tNO * tH              - kr [13] * tHNO * tstar    # NO* + H* <--> HNO* + *
   rate [14] = kf [14] * tNOH * tH             - kr [14] * tHNOH * tstar   # NOH* + H* <--> HNOH* + *
   rate [15] = kf [15] * tNOH * tstar          - kr [15] * tN * tOH        # NOH* + * <--> N* + OH*
   rate [16] = kf [16] * tHNO * tH             - kr [16] * tHNOH * tstar   # HNO* + H* <--> HNOH* + *
   rate [17] = kf [17] * tHNO * tstar          - kr [17] * tNH * tO        # HNO* + * <--> NH* + O*
   rate [18] = kf [18] * tHNOH * tstar         - kr [18] * tNH * tOH       # HNOH* + * <--> NH* + OH*
   return rate

def get_odes(theta,t,kf,kr):
   # returns the system of ODEs d(theta)/dt
   # calculated at the current value of theta (and time t, not used)
   rate = get_rates(theta,kf,kr) # calculate the current rates
   # Time derivatives of theta
   dt = [0]*13 # 13 surface species
   
   dt [0] = rate [0] - rate [3] + rate [11] - rate [12] - rate [13]  # d(tNO)/dt
   dt [1] = rate [0] + rate [15] + rate [18] - rate [2] + rate [4]   # d(tOH)/dt
   dt [2] = rate [3] + rate [17] + rate [11] - rate [4]              # d(tO)/dt
   dt [3] = rate [3] + rate [15] - rate [5]                          # d(tN)/dt
   dt [4] = rate [5] + rate [17] + rate [18] - rate [6]              # d(tNH)/dt
   dt [5] = rate [6] - rate [7]                                      # d(tNH2)/dt
   dt [6] = rate [7] - rate [8]                                      # d(tNH3)/dt
   dt [7] = rate [2] - rate [9]                                      # d(tH2O)/dt
   dt [8] = 2 * rate [1] + rate [10] - rate [2] - rate [4] - rate [5] - rate [6] - rate [7] - rate [12] - rate[13] - rate [14] - rate [16]  # d(tH)/dt
   dt [9] = rate [10] - rate [11]                                    # d(tNO2)/dt
   dt [10] = rate [12] - rate [14] - rate [15]                       # d(NOH)/dt] 
   dt [11] = rate [13] - rate [16] - rate [17]                       # d(HNO)/dt]
   dt [12] = rate [14] + rate [16] - rate [18]                       # d(HNOH)/dt]
   return dt

def print_output(theta,kf,kr):
   # Prints the solution of the model
   rates = get_rates(theta,kf,kr)
   
   print
   for r,rate in enumerate(rates):
       print ("Step",r,": rate =",rate,", kf =",kf [r],", kr=",kr [r])
   print
   print ("The coverages for NO*, OH*, O*, N*, NH*, NH2*, NH3*, H2O*, H*, NO2*, NOH*, HNO*, HNOH* are:")
   for t in theta:
       print (t)
   print


#Reaction Energies
#To run the model at any pH, remove the comment at the start of the line in both reaction energies and activation barriers sections
#E = [-2.908, -1.165, 0.06, -0.85, 0.22, -0.256, 0.323, -0.153, 0.837, 0.656, -1.778, -1.64, 0.501, 0.599, 0.449, -1.133, 0.351, -1.708, -1.839 ] #3.25 
#E = [-2.864, -1.165, 0.06, -0.85, 0.22, -0.256, 0.323, -0.153, 0.882, 0.656, -1.734, -1.64, 0.501, 0.599, 0.449, -1.133, 0.351, -1.708, -1.839 ] #4
#E = [-2.804, -1.165, 0.06, -0.85, 0.22, -0.256, 0.323, -0.153, 0.941, 0.656, -1.674, -1.64, 0.501, 0.599, 0.449, -1.133, 0.351, -1.708, -1.839 ] #5
#E = [-2.745, -1.165, 0.06, -0.85, 0.22, -0.256, 0.323, -0.153, 1.001, 0.656, -1.614, -1.64, 0.501, 0.599, 0.449, -1.133, 0.351, -1.708, -1.839 ] #6
E = [-2.685, -1.165, 0.06, -0.85, 0.22, -0.256, 0.323, -0.153, 1.060, 0.656, -1.555, -1.64, 0.501, 0.599, 0.449, -1.133, 0.351, -1.708, -1.839 ] #7
#E = [-2.626, -1.165, 0.06, -0.85, 0.22, -0.256, 0.323, -0.153, 1.120, 0.656, -1.495, -1.64, 0.501, 0.599, 0.449, -1.133, 0.351, -1.708, -1.839 ] #8
#E = [-2.566, -1.165, 0.06, -0.85, 0.22, -0.256, 0.323, -0.153, 1.179, 0.656, -1.436, -1.64, 0.501, 0.599, 0.449, -1.133, 0.351, -1.708, -1.839 ] #9
#E = [-2.507, -1.165, 0.06, -0.85, 0.22, -0.256, 0.323, -0.153, 1.194, 0.656, -1.376, -1.64, 0.501, 0.599, 0.449, -1.133, 0.351, -1.708, -1.839 ] #10
#E = [-2.447, -1.165, 0.06, -0.85, 0.22, -0.256, 0.323, -0.153, 1.194, 0.656, -1.317, -1.64, 0.501, 0.599, 0.449, -1.133, 0.351, -1.708, -1.839 ] #11
#E = [-2.387, -1.165, 0.06, -0.85, 0.22, -0.256, 0.323, -0.153, 1.194, 0.656, -1.257, -1.64, 0.501, 0.599, 0.449, -1.133, 0.351, -1.708, -1.839 ] #12
#E = [-2.328, -1.165, 0.06, -0.85, 0.22, -0.256, 0.323, -0.153, 1.194, 0.656, -1.198, -1.64, 0.501, 0.599, 0.449, -1.133, 0.351, -1.708, -1.839 ] #13
#E = [-2.268, -1.165, 0.06, -0.85, 0.22, -0.256, 0.323, -0.153, 1.194, 0.656, -1.138, -1.64, 0.501, 0.599, 0.449, -1.133, 0.351, -1.708, -1.839 ] #14

def free_energies():
     #calculate the Gibbs free energy for each reaction using Erxn and Srxn
     S = [-1.965E-3, -1.836E-3, 2.737E-4, -1.336E-4, 9.791E-5, -4.539E-7, 7.032E-5, 2.258E-4, 1.533E-3, 1.4021E-3, -2.954E-3, -8.705E-5, 1.268E-4, -7.899E-5, -1.26E-4, -1.625E-4, 7.978E-5, -5.505E-5, -3.693E-5]
     Grxn = [ E [i] - T * S [i] for i in range (len(E))]
     return Grxn

dGrxn = free_energies()
print ("Reaction Free Energies:", dGrxn)

#Activation Barriers
#To run the model at any pH, remove the comment at the start of the line in both reaction energies and activation barriers sections
#E = [0.000, 0.000, 0.825*A, 1.432*A, 1.250*A, 0.892*A, 1.164*A, 1.053*A, 0.837, 0.656, 0.0, 0.267*A, 1.621*A, 1.492*A, 1.34*A, 0.70*A, 0.73*A, 0.75*A, 0.0 ] #pH=3.25
#E = [0.045, 0.000, 0.825*A, 1.432*A, 1.250*A, 0.892*A, 1.164*A, 1.053*A, 0.882, 0.656, 0.0, 0.267*A, 1.621*A, 1.492*A, 1.34*A, 0.70*A, 0.73*A, 0.75*A, 0.0 ] #pH=4
#E = [0.104, 0.000, 0.825*A, 1.432*A, 1.250*A, 0.892*A, 1.164*A, 1.053*A, 0.941, 0.656, 0.0, 0.267*A, 1.621*A, 1.492*A, 1.34*A, 0.70*A, 0.73*A, 0.75*A, 0.0 ] #pH=5
#E = [0.164, 0.000, 0.825*A, 1.432*A, 1.250*A, 0.892*A, 1.164*A, 1.053*A, 1.001, 0.656, 0.0, 0.267*A, 1.621*A, 1.492*A, 1.34*A, 0.70*A, 0.73*A, 0.75*A, 0.0 ] #pH=6
E = [0.223, 0.000, 0.825*A, 1.432*A, 1.250*A, 0.892*A, 1.164*A, 1.053*A, 1.060, 0.656, 0.0, 0.267*A, 1.621*A, 1.492*A, 1.34*A, 0.70*A, 0.73*A, 0.75*A, 0.0 ] #pH=7
#E = [0.283, 0.000, 0.825*A, 1.432*A, 1.250*A, 0.892*A, 1.164*A, 1.053*A, 1.120, 0.656, 0.0, 0.267*A, 1.621*A, 1.492*A, 1.34*A, 0.70*A, 0.73*A, 0.75*A, 0.0 ] #pH=8
#E = [0.342, 0.000, 0.825*A, 1.432*A, 1.250*A, 0.892*A, 1.164*A, 1.053*A, 1.179, 0.656, 0.0, 0.267*A, 1.621*A, 1.492*A, 1.34*A, 0.70*A, 0.73*A, 0.75*A, 0.0 ] #pH=9
#E = [0.402, 0.000, 0.825*A, 1.432*A, 1.250*A, 0.892*A, 1.164*A, 1.053*A, 1.194, 0.656, 0.0, 0.267*A, 1.621*A, 1.492*A, 1.34*A, 0.70*A, 0.73*A, 0.75*A, 0.0 ] #pH=10
#E = [0.461, 0.000, 0.825*A, 1.432*A, 1.250*A, 0.892*A, 1.164*A, 1.053*A, 1.194, 0.656, 0.0, 0.267*A, 1.621*A, 1.492*A, 1.34*A, 0.70*A, 0.73*A, 0.75*A, 0.0 ] #pH=11
#E = [0.521, 0.000, 0.825*A, 1.432*A, 1.250*A, 0.892*A, 1.164*A, 1.053*A, 1.194, 0.656, 0.0, 0.267*A, 1.621*A, 1.492*A, 1.34*A, 0.70*A, 0.73*A, 0.75*A, 0.0 ] #pH=12
#E = [0.581, 0.000, 0.825*A, 1.432*A, 1.250*A, 0.892*A, 1.164*A, 1.053*A, 1.194, 0.656, 0.0, 0.267*A, 1.621*A, 1.492*A, 1.34*A, 0.70*A, 0.73*A, 0.75*A, 0.0 ] #pH=13
#E = [0.640, 0.000, 0.825*A, 1.432*A, 1.250*A, 0.892*A, 1.164*A, 1.053*A, 1.194, 0.656, 0.0, 0.267*A, 1.621*A, 1.492*A, 1.34*A, 0.70*A, 0.73*A, 0.75*A, 0.0 ] #pH=14

def Activation_free_energies():
      S = [-5.133E-4, -3.776E-4, 8.264E-5, -1.107E-4, 8.137E-5, 4.401E-5, 8.583E-5, 8.480E-5, 1.063E-3, 9.301E-4, -1.397E-3, -6.158E-5, -2.03E-5, 3.36E-5 , 6.285E-5, -1.68E-4, -2.29E-4, 5E-5, 1E-8 ]
      Gact = [ E [i]  - T * S [i]  for i in range (len(E))]
      return Gact
   
dGact = Activation_free_energies()
print ("Free Energy Barriers:", dGact)
        
(kf,kr) = get_rate_constants(dGrxn,dGact) 

# Use scipy’s odeint to solve the system of ODEs
from scipy.integrate import odeint
# As initial guess we assume an empty surface
theta0 = (0., 0., 0., 0., 0., 0., 0., 0., 0.,0., 0., 0., 0.)
# Integrate the ODEs for t' seconds
theta = odeint(get_odes, # system of ODEs
        theta0, # initial guess
        [0,1E6], # integrate ODE unitl we reach time of interest (t')
        args = (kf,kr), # additional arguments to get_odes()
        h0 = 1E-36, # initial time step
        mxstep =9000000, # maximum number of steps
        rtol = 1E-12, # relative tolerance
        atol = 1E-12) # absolute tolerance
print_output(theta [-1,:],kf,kr)
theta_final = theta [-1,:]

# Degree of rate control
print("Degree of Rate Control")

#Degree of Rate Control, Xrc - run the mkm model for a short timestep picking up where the main mkm stopped
theta_o = odeint(get_odes, # system of ODEs
        theta_final, # state of the surface at t'
        [0,1], # short time span
        args = (kf,kr), # additional arguments to get_odes()
        h0 = 1E-36, # initial time step
        mxstep =9000000, # maximum number of steps
        rtol = 1E-12, # relative tolerance
        atol = 1E-12) # absolute tolerance

ro = get_rates(theta[-1,:],kf,kr)[8] #save rate
ratesdrc = [0]*19 #initialize DRC rates for each Step
Xrc = [0]*19 #initialize Xrc for each Step
x = 0.002 #set change in barrier height

for s in range(len(dGact)):
   dGactdrc = dGact[:] #reset barriers
   dGactdrc[s]=dGact[s]-x #modify barrier of step "s"
   (kfdrc,krdrc) = get_rate_constants(dGrxn,dGactdrc) #get rate constants with modified barrier for step "s"

   thetadrc = odeint(get_odes, # system of ODEs
        theta_final, # state of the surface at t'
        [0,1], # short time span
        args = (kfdrc,krdrc), # additional arguments to get_odes()
        h0 = 1E-36, # initial time step
        mxstep =9000000, # maximum number of steps
        rtol = 1E-12, # relative tolerance
        atol = 1E-12) # absolute tolerance
  
   ratesdrc[s] = get_rates(thetadrc[-1,:],kfdrc,krdrc)[8] #compute new rate 
   Xrc[s] = ((ratesdrc[s] - ro) / ro)*(kf[s] / (kfdrc[s] - kf[s])) #compute Xrc for step "s"
   print ("Step",s,": Xrc =",Xrc[s])

print ("Sum:", sum(Xrc))

# Degree of transient rate control
print("Degree of Transient Rate Control")
ro = get_rates(theta_final,kf,kr)[8] #save rate
ratesdrc = [0]*19 #initialize DRC rates for each Step
Xrc = [0]*19 #initialize Xrc for each Step
x = 0.002 #set change in barrier height

for s in range(len(dGact)):
   dGactdrc = dGact[:] #reset barriers
   dGactdrc[s]=dGact[s]-x #modify barrier of Step "s"
   (kfdrc,krdrc) = get_rate_constants(dGrxn,dGactdrc) #get rate constants with modified barrier for step "s"

   thetadrc = odeint(get_odes, # system of ODEs
        theta0, # state of the surface at t = 0
        [0,1E6], # time span t = 0 to t'
        args = (kfdrc,krdrc), # additional arguments to get_odes()
        h0 = 1E-36, # initial time step
        mxstep =9000000, # maximum number of steps
        rtol = 1E-12, # relative tolerance
        atol = 1E-12) # absolute tolerance
  
   ratesdrc[s] = get_rates(thetadrc[-1,:],kfdrc,krdrc)[8] #compute new rate 
   Xrc[s] = ((ratesdrc[s] - ro) / ro)*(kf[s] / (kfdrc[s] - kf[s])) #compute Xrc for step "s"
   print ("Step",s,": Xrc, transient =",Xrc[s])

print ("Sum:", sum(Xrc))