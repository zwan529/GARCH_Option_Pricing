import numpy as np
from scipy.optimize import minimize
import pandas 
from py_vollib import black_scholes

def GARCH_LL(garchParam,ret,rf):
    ## garchParam:  alpha_0, alpha_1, lambda, beta, size of 4
    ## ret:         vector of log returns
    ## rf:          risk-free rate 
    epsilon = np.zeros(len(ret))  # error term: epsilon
    h = np.zeros(len(ret))        # conditional variance h
    LL = np.zeros(len(ret))       # log likelihood

    epsilon[0] = 0
    h[0] = 0.24  
    LL[0] = -0.5 * np.log(2 * np.pi * h[0]) - 0.5 * (epsilon[0] ** 2) / h[0]   ## log likelihood -1/2 log(2 \pi h0) - 1/2 * (x - mu)^2 / h0

    sum = 0
    if (garchParam[1:] < 0).any():
        return 1e100
    for i in range(len(ret)):
        if (i==0):
            continue
        h[i] = garchParam[0] + garchParam[1] * (epsilon[i-1] - garchParam[2] * np.sqrt(h[i-1])) ** 2 + garchParam[3] * h[i-1]
        epsilon[i] = ret[i] - rf + 0.5*h[i]
        LL[i] = -0.5 * np.log(2 * np.pi * h[i]) - 0.5 * (epsilon[i] ** 2) / h[i]
        sum = sum + LL[i]
    return -sum
    
def mainProgram():
    #maxIter = 300           ## Max number of iterations
    Tolerance = 1e-10       ## Tolerance 
    N = 4                   ## Number of GArch parameters 

    S = 4458.58              ## STock
    K = 4468                 ## Strike
    T = 30                  ## time to maturity 
    MaT = T / 360           ## times in year
    rf = 0                  ## risk-Free rate
    q = 0                   ## dividend yield
    Nsims = 400             ## number of simulations

    ## SP500 Price levels 
    SPX_file_Path = "/Users/zhiwang/Desktop/PythonPackage/Data/SPXIndex_1Y_20200910.csv"
    close_price_str = " Close"
    SPX_hist_data = pandas.read_csv(SPX_file_Path)
    level = SPX_hist_data[close_price_str][::-1]
    
    ## log returns 
    logReturn = np.diff(np.log(level))

    ## GARCH Variable starting point
    garchParam = np.zeros((N,N+1))
    garchParam[0][0] = 0.00002; garchParam[0][1] = 0.00001; garchParam[0][2] = 0.00002; garchParam[0][3] = 0.00001; garchParam[0][4] = 0.00002 #alpha 0
    garchParam[1][0] = 0.21;    garchParam[1][1] = 0.19;    garchParam[1][2] = 0.16;    garchParam[1][3] = 0.18;    garchParam[1][4] = 0.22 #alpha 1
    garchParam[2][0] = 0.01;    garchParam[2][1] = 0.72;    garchParam[2][2] = 0.69;    garchParam[2][3] = 0.65;    garchParam[2][4] = 0.73 #lambda
    garchParam[3][0] = 0.71;   garchParam[3][1] = 0.008;   garchParam[3][2] = 0.006;   garchParam[3][3] = 0.004;   garchParam[3][4] = 0.005 #beta

    obj_func = lambda x: GARCH_LL(x, logReturn, rf)
    res = minimize(obj_func,garchParam[:,0], method='Nelder-Mead', tol= Tolerance)   ## Nelder-Mead minimizer 
    estimated_param = res.x
    stationaryVariance = estimated_param[0] / (1 - (1 + estimated_param[2] ** 2) * estimated_param[1] - estimated_param[3] )
    print("Nelder-Mead method " + str(res.success) + "with " + res.message)
    print("Number of iterations: " + str(res.nit))
    print("Obj_Func_Value: " + str(res.fun))
    print("Alpha_0:" + str(estimated_param[0]))
    print("Alpha_1:" + str(estimated_param[1]))
    print("lambda:" + str(estimated_param[2]))
    print("beta:" + str(estimated_param[3]))
    print("Stationary Variance:" + str(stationaryVariance))
    print("Stationary annulized volatility:" + str(np.sqrt(stationaryVariance * 360)))
    print("+++++++++++++++++++++++++++++")

    ## Monte-Carlo simulation for 

    # Monte-Carlo simulation for option price.
    SS = np.zeros((Nsims,T)) ## A simulated price for each day 
    h = np.zeros((Nsims,T))  ## conditional variance 
    e = np.zeros((Nsims,T)) ## error term 
    SS[:,0] = S 
    h[:,0] = stationaryVariance ## initialize to start value to conditional variance
    for t in range(T):
        if t == 0:
            continue
        h[:,t] = estimated_param[0] + estimated_param[1] * (e[:,t-1] - estimated_param[2] * np.sqrt(h[:,t-1])) ** 2 + estimated_param[3] * h[:,t-1]
        e[:,t] = np.random.normal(0,np.sqrt(h[:,t]))
        SS[:,t] = SS[:,t-1] * np.exp(rf - 0.5 * h[:,t] + e[:,t])
    call = np.maximum(SS[:,-1] - K,0)
    put = np.maximum(K - SS[:,-1],0)
    GARCHCall = np.mean(call)
    GARCHPut = np.mean(put)
    print("GARCHCall Price is" + str(GARCHCall))
    print("GARCHPut price is" + str(GARCHPut))
    BSCall = black_scholes.black_scholes('c',S, K, MaT ,rf, np.sqrt(stationaryVariance * 360))
    BSPut = black_scholes.black_scholes('p',S, K, MaT ,rf, np.sqrt(stationaryVariance * 360))
    print("Black-Scholes Call Price: " +  str(BSCall))
    print("Black-Scholes Put Price: " +  str(BSPut))
if __name__ == "__main__":
    mainProgram()