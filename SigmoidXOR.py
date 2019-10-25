import numpy as np 
import matplotlib.pyplot as plt 

#   each column of S is a stimulus vector
S = [[1, 0, 1, 0 ],
     [1, 1, 0, 0 ]]
S = np.array(S)

#   each column of O is a target vector
O = [[1, 0, 0, 0],  # AND
     [1, 1, 1, 0],  # OR
     [0, 1, 1, 0]]  # XOR
O = np.array(O)

def SigmoidNet( S, O, h_len, numSteps=10000, gamma=.4 ):

  #   Define some size constants
  s_len = S.shape[0]    # number of input units
  o_len = O.shape[0]    # number of output units
  numS  = S.shape[1]    # number of stimuli

  #   Initialize weights
  V = np.random.normal(scale=.1 , size=[o_len, h_len])
  W = np.random.normal(scale=.1 , size=[h_len, s_len])

  #   Display lists and values
  ln_list = []
  P = np.zeros((o_len,numS))

  #   Training Loop (Batch)
  for b in range(0, numSteps):

    #   Important variable values over stimuli
    #   for display purposes
    ln = 0

    #   Derivatives for updates
    dln_dVvec = np.zeros((1,o_len*h_len))
    dln_dWvec = np.zeros((1,h_len*s_len))

    for i in range(0, numS):
      #   Grab our stimulus and target pair
      s_i = S[:,i,np.newaxis]
      o_i = O[:,i,np.newaxis]

      #   Forward pass
      r_i = W @ s_i
      h_i = 1/(1+np.exp(-r_i))
      f_i = V @ h_i
      p_i = 1/(1+np.exp(-f_i))
      P[:,i] = np.squeeze(p_i)
      o_i_T = np.transpose(o_i)
      c_i = -o_i_T@np.log(p_i)-(1 - o_i_T)@np.log(1 - p_i)

      #   Backward derivative pass
      dci_dfi   = - np.transpose(o_i - p_i)
      dfi_dVvec = np.kron(np.transpose(h_i), np.eye(o_len))
      dfi_dhi   = V
      dhi_dri   = np.diag(np.squeeze(h_i * (1 - h_i)))
      dri_dWvec = np.kron(np.transpose(s_i), np.eye(h_len))

      #   Chain rules
      dci_dVvec = dci_dfi @ dfi_dVvec
      dci_dWvec = dci_dfi @ dfi_dhi @ dhi_dri @ dri_dWvec

      #   Add to global derivatives
      dln_dVvec = dln_dVvec + 1/numS*dci_dVvec
      dln_dWvec = dln_dWvec + 1/numS*dci_dWvec

      #   Calculate global objective function value
      ln = ln + 1/numS*c_i

    #   vec-1 versions of derivatives
    dln_dV = np.reshape(dln_dVvec, V.shape, order='F')
    dln_dW = np.reshape(dln_dWvec, W.shape, order='F')

    #   Perform weight update equations
    V = V - gamma*dln_dV
    W = W - gamma*dln_dW

    #   Append to display lists
    ln_list.append(np.squeeze(ln))


  plt.plot(ln_list)

  print("FinalLn : ",ln)

  print("\nDesired Output Columns: ")
  print(O)

  print("\nCurrent Output Columns: ")
  print(np.round(P,2))


SigmoidNet(S, O, 10)  # 10 hidden units

