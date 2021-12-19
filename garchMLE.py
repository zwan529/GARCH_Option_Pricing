
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

TWO_OVER_PI_SQRT = np.sqrt(2 / np.pi)

class garchMLE():
    def __init__(self, combinedData):
        ## CombinedData: pandas dataframe. Must have column "SPX.Close" and "VIX.Close"
        ##    index should be corresponding datetime object
        self._spxClose = combinedData["SPX.Close"].to_numpy()
        self._vixClose = combinedData["VIX.Close"].to_numpy()
        self._dates =  combinedData.index.to_numpy()
        self._start = np.min(self._dates)
        self._end = np.max(self._dates)
        if (np.isnan(self._spxClose).any() or np.isnan(self._vixClose).any() or np.isnan(self._dates).any()):
            raise Exception("Nan value encountered in data")
        self._size = len(self._spxClose)
        self.init()
    
    def init(self):
        ## Step1: Reverse the array 
        if (self._dates[0] != self._start):
            self._spxClose = np.flip(self._spxClose)
            self._vixClose = np.flip(self._vixClose)
            self._dates = np.flip(self._dates)
        ## Step2: Calculate log return, readjuest the start and end
        self._logreturn = np.diff(np.log(self._spxClose))
        self._spxClose = np.delete(self._spxClose,[0])
        self._vixClose = np.delete(self._vixClose,[0])
        self._dates = np.delete(self._dates,[0])
        self._start = self._dates[1]
        self._size -= 1
        self._spxVar = np.var(self._logreturn)
    
class GARCH11(garchMLE):
    def __init__(self, combinedData, paramArray, ismLRNVR = False):
        ## paramArray: size of 4: alpha_0, alpha_1, beta_1, lambda 
        ##  if ismLRNVE == True, additional parameter of lambda2
        super().__init__(combinedData)
        self._alpha_0 = paramArray[0]
        self._alpha_1 = paramArray[1]
        self._beta_1 = paramArray[2]
        self._lambda = paramArray[3]
        self._impliedVIX = np.array([])
        self._impliedspx = np.array([])
        self._ismLRNVR = ismLRNVR
        if (ismLRNVR):
            self._lambda2 = paramArray[4]

    def updateParam(self, paramArray, ismLRNVE = False):
        self._alpha_0 = paramArray[0]
        self._alpha_1 = paramArray[1]
        self._beta_1 = paramArray[2]
        self._lambda = paramArray[3]
        if (self._ismLRNVR):
            self._lambda2 = paramArray[4]

    def LogLikehood(self, mode = "SPX", replaced = False):
        ## Under risk netural measure
        ## Assume rf = 0 here
        ## Initial conditional variance = variance of SPX log price over sample period
        ## Initional error = 0 
        ## Valid mode value: "SPX", "VIX", "COMBINED"
        ## Returned value is - log-likelihoood, so need to minimize that value 
        ## Stationary condition check: \alpha1(1 + \lambda ** 2 ) + \beta_1 < 1
        beta_star = self._beta_1
        if (self._ismLRNVR):
            beta_star = self._beta_1 - np.sqrt(2) * self._alpha_1 * self._lambda2
        if (self._alpha_1 < 0 or self._beta_1 < 0 or self._lambda < 0):
            return 1e100
        if (self._alpha_1 * (1 + self._lambda ** 2) + beta_star  >= 1):
            return 1e100
        h_0 = self._spxVar
        ## Initialize array, phisical measure
        epsilon = np.zeros(self._size)
        ht = np.zeros(self._size)
        ht[0] = h_0
        epsilon[0] = 0
        # Risk-netiral measure
        epsilon_q = np.zeros(self._size)
        ht_q = np.zeros(self._size)
        ht_q[0] = h_0
        epsilon_q[0] = 0

        for i in range(self._size):
            if (i == 0):
                continue
            ## Under P
            ht[i] = self._alpha_0 + self._alpha_1 * (epsilon[i - 1]) ** 2 + \
                self._beta_1 * ht[i-1]
            epsilon[i] = self._logreturn[i] + 0.5 * ht[i] - self._lambda * np.sqrt(ht[i]) ## TODO: ADD risk-free rate r here
            
            ## Under Q
            ht_q[i] = self._alpha_0 + self._alpha_1 * (epsilon_q[i - 1] - self._lambda * np.sqrt(ht_q[i - 1])) ** 2 + \
                beta_star * ht_q[i-1]
            epsilon_q[i] = self._logreturn[i] + 0.5 * ht_q[i] ## TODO: ADD risk-free rate r here
        
        LL_SPX = - self._size / 2 * np.log(2 * np.pi) - 0.5 * np.sum( np.log(ht) + epsilon ** 2 / ht  )

        eta = self._alpha_1 * (1 + self._lambda ** 2) + beta_star 
        calculationdays = 30 ## TODO: Use trading day or Calender day?? 
        vix_B = (1 - eta ** calculationdays) / (calculationdays * (1 - eta))
        vix_A = self._alpha_0 / ( 1 - eta ) * ( 1 - vix_B)
        ## vix_t = A + Bh_{t+1}
        vix_implied = np.sqrt(vix_A + vix_B * ht_q) 
        vix_market = self._vixClose / 100 / np.sqrt(252)
        ## Valid period are [startDate, endDate - 1]
        vix_implied = np.roll(vix_implied, -1)[0 : len(vix_implied) - 1]
        vix_market = vix_market[0 : len(self._vixClose) - 1]
        vix_epsilon = vix_market - vix_implied
        vix_var = np.var(vix_epsilon)

        LL_VIX = - self._size / 2 * np.log(2 * np.pi * vix_var) - 1 / (2 * vix_var) * np.sum(vix_epsilon ** 2)

        if (replaced):
            self._impliedVIX = vix_implied * np.sqrt(252) * 100
            self._impliedspx = -0.5 * ht  ## TODO: Add risk-free rate
        if (mode == "SPX"):
            return - LL_SPX
        elif (mode == "VIX"):
            return -LL_VIX 
        else:
            return - (LL_VIX + LL_SPX)


    def MaximumLikeliHood(self, mode = "SPX", Tolerance = 1e-10):
        def callable(paramArray):
            self.updateParam(paramArray)
            return self.LogLikehood(mode = mode)
        initial = np.array([self._alpha_0, self._alpha_1, self._beta_1,self._lambda ])
        if (self._ismLRNVR): 
            initial = np.array([self._alpha_0, self._alpha_1, self._beta_1,self._lambda, self._lambda2])
        res = minimize(callable, initial, method='Nelder-Mead', tol= Tolerance)
        ## Result Diaplay 
        estimated_param = res.x
        print("Nelder-Mead method " + str(res.success) + "with " + res.message)
        print("Number of iterations: " + str(res.nit))
        print("Obj_Func_Value: " + str(res.fun))
        print("Alpha_0:" + str(estimated_param[0]))
        print("Alpha_1:" + str(estimated_param[1]))
        print("beta:" + str(estimated_param[2]))
        print("lambda:" + str(estimated_param[3]))
        if (self._ismLRNVR):
            print("lambda2:" + str(estimated_param[4]))
        print("+++++++++++++++++++++++++++++")
        if (self._ismLRNVR):
            print("%.6f & %.6f & %.6f & %.6f & %.6f"%(estimated_param[0], estimated_param[1], estimated_param[2], \
            estimated_param[3],estimated_param[4]))
        else: 
            print("%.6f & %.6f & %.6f & %.6f &"%(estimated_param[0], estimated_param[1], estimated_param[2], \
            estimated_param[3]))
        
class EGARCG11(garchMLE):
    def __init__(self, combinedData, paramArray):
        ## paramArray: size of 4: alpha_0, alpha_1,beta_1, kappa, lambda 
        super().__init__(combinedData)
        self._alpha_0 = paramArray[0]
        self._alpha_1 = paramArray[1]
        self._beta_1 = paramArray[2]
        self._kappa = paramArray[3]
        self._lambda = paramArray[4]
        self.prepareVIXhelper()

    def updateParam(self, paramArray):
        if (len(paramArray) != 5):
            raise Exception("Garch11 paramArray must have size of 5")
        self._alpha_0 = paramArray[0]
        self._alpha_1 = paramArray[1]
        self._beta_1 = paramArray[2]
        self._kappa = paramArray[3]
        self._lambda = paramArray[4]
        self.prepareVIXhelper()

    def prepareVIXhelper(self):
        periodDays = 30
        vixhelperMap = np.ones(periodDays)
        beta_i = 1   ## beta_1 ** i
        for i in range(periodDays):
            _a = -beta_i * (self._alpha_1 - self._kappa) * self._lambda + ( beta_i * (self._alpha_1 - self._kappa) ) ** 2 / 2
            _b = -beta_i * (self._alpha_1 + self._kappa) * self._lambda + ( beta_i * (self._alpha_1 + self._kappa) ) ** 2 / 2
            _na = self._lambda - beta_i * (self._alpha_1 - self._kappa)
            _nb = beta_i * (self._alpha_1 + self._kappa) - self._lambda
            _0 = beta_i * (self._alpha_0 - self._kappa * TWO_OVER_PI_SQRT)
            if (i == 0):
                vixhelperMap[i] = np.exp(_0) * (np.exp(_a) * norm.cdf(_na) + np.exp(_b) * norm.cdf(_nb))
            else:
                vixhelperMap[i] = vixhelperMap[i - 1] * np.exp(_0) * (np.exp(_a) * norm.cdf(_na) + np.exp(_b) * norm.cdf(_nb))
            beta_i = beta_i * self._beta_1
        self.vixHelper = vixhelperMap

    def error_function(self,zt):
        return self._alpha_1 * zt + self._kappa( np.abs(zt) - TWO_OVER_PI_SQRT)

    def LogLikehood(self, mode = "SPX", replaced = False):
        ## Under risk netural measure
        ## Assume rf = 0 here
        ## Initial conditional variance = variance of SPX log price over sample period
        ## Initional error = 0 
        ## Valid mode value: "SPX", "VIX", "COMBINED"
        ## Returned value is - log-likelihoood, so need to minimize that value 
        ## Stationary condition check: \alpha1(1 + \lambda ** 2 ) + \beta_1 < 1
        if (np.abs(self._beta_1)  >= 1):
            return 1e100
        h_0 = self._spxVar
        ## Initialize array, phisical measure
        epsilon = np.zeros(self._size)
        norm_epsilon = np.zeros(self._size)
        loght = np.zeros(self._size)

        loght[0] = np.log(h_0)
        epsilon[0] = 0
        norm_epsilon[0] = epsilon[0] / loght[0]
        # Risk-netiral measure
        epsilon_q = np.zeros(self._size)
        norm_epsilon_q = np.zeros(self._size)
        loght_q = np.zeros(self._size)

        loght_q[0] = np.log(h_0)
        epsilon_q[0] = 0
        norm_epsilon_q[0] = epsilon_q[0] / loght_q[0]

        for i in range(self._size):
            if (i == 0):
                continue
            ## Under P
            loght[i] = self._alpha_0 + self._beta_1 * loght[i-1] + self.error_function(norm_epsilon[i - 1])
            ht_at_i = np.exp(loght[i])
            epsilon[i] = self._logreturn[i] + 0.5 * ht_at_i - self._lambda * np.sqrt(ht_at_i) ## TODO: ADD risk-free rate r here
            norm_epsilon[i] = epsilon[i] / np.sqrt(ht_at_i)

            ## Under Q
            loght_q[i] = self._alpha_0 + self._beta_1 * loght_q[i-1] + self.error_function(norm_epsilon_q[i - 1] - self._lambda)
            ht_at_i = np.exp(loght_q[i])
            epsilon_q[i] = self._logreturn[i] + 0.5 * ht_at_i - self._lambda * np.sqrt(ht_at_i) ## TODO: ADD risk-free rate r here
            norm_epsilon_q[i] = epsilon_q[i] / np.sqrt(ht_at_i)
        
        LL_SPX = - self._size / 2 * np.log(2 * np.pi) - 0.5 * np.sum( loght + epsilon ** 2 /  np.exp(loght) )

        calculationdays = 30 ## TODO: Use trading day or Calender day?? 
        vix_B = (1 - eta ** calculationdays) / (calculationdays * (1 - eta))
        vix_A = self._alpha_0 / ( 1 - eta ) * ( 1 - vix_B)
        ## vix_t = A + Bh_{t+1}
        vix_implied = np.sqrt(vix_A + vix_B * ht_q) 

        
        vix_market = self._vixClose / 100 / np.sqrt(252)
        ## Valid period are [startDate, endDate - 1]
        vix_implied = np.roll(vix_implied, -1)[0 : len(vix_implied) - 1]
        vix_market = vix_market[0 : len(self._vixClose) - 1]
        vix_epsilon = vix_market - vix_implied
        vix_var = np.var(vix_epsilon)

        LL_VIX = - self._size / 2 * np.log(2 * np.pi * vix_var) - 1 / (2 * vix_var) * np.sum(vix_epsilon ** 2)

        if (replaced):
            self._impliedVIX = vix_implied * np.sqrt(252) * 100
            self._impliedspx = -0.5 * ht  ## TODO: Add risk-free rate
        if (mode == "SPX"):
            return - LL_SPX
        elif (mode == "VIX"):
            return -LL_VIX 
        else:
            return - (LL_VIX + LL_SPX)


'''
import garchUtilities
start = "2000-01-02"
end = "2021-10-12"
SPXfilePath = "/Users/zhiwang/Desktop/PythonPackage/GARCHOptionPricing/Data/HistoricalPrices.csv"
VIXfilePath = "/Users/zhiwang/Desktop/PythonPackage/GARCHOptionPricing/Data/VIX_History.csv"
spxdf = garchUtilities.preprocessSPXdata(SPXfilePath, start, end)
vixdf = garchUtilities.preProcessVIXData(VIXfilePath, start, end)
combineddf = garchUtilities.combineVixSPX(spxdf,vixdf)
combineddf = combineddf.rename(columns={" Close":"SPX.Close", "CLOSE":"VIX.Close"})
paramArray = np.array([1.6746e-6,0.0473,0.9498,0.2068])
mygarch = GARCH11(combineddf, paramArray)
mygarch.MaximumLikeliHood()
print(mygarch.LogLikehood(mode = "Combined"))
'''