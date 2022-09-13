# -*- coding: utf-8 -*-
"""
Cross Entropy Algorithm 
Python implementation of adaptive Importance Sampling by Cross Entropy 
M. Balesdent and L. Brevault of ONERA, the French Aerospace Lab for the 
openTURNS consortium

source :  J. Morio & M. Balesdent, Estimation of Rare Event failureProbabilitybilities 
in Complex Aerospace and Other Systems, A Practical Approach, Elsevier, 2015

"""

import numpy as np
import copy
import openturns as ot

## Container of CE results
class CEResult(ot.SimulationResult):
    
    """
    Class providing results of Cross Entropy algorithm
    
    """
    
    def __init__(self):
        self.probabilityEstimate = 0.
        self.varianceEstimate = 0.
        self.auxiliaryInputSample = None
        self.auxiliaryOutputSample = None
        self.auxiliaryDistribution = None
        self.numberOfSample = 0

    # Get the probability estimate
    def getProbabilityEstimate(self):
        return self.probabilityEstimate

    # Set the probability estimate
    def setProbabilityEstimate(self,probabilityEstimate):
        self.probabilityEstimate = probabilityEstimate
        return None
    
    # Get the variance estimator of the failure probability
    def getVarianceEstimate(self):
        return self.varianceEstimate
    
    # Set the variance estimator of the failure probability
    def setVarianceEstimate(self,varianceEstimate):
        self.varianceEstimate = varianceEstimate
        return None
    
    # Get the auxiliary input samples
    def getAuxiliaryInputSample(self):
        return self.auxiliaryInputSample
    
    # Set the auxiliary input samples
    def setAuxiliaryInputSample(self,auxiliaryInputSample):
        self.auxiliaryInputSample = auxiliaryInputSample
        return None
    
    # Get the auxiliary input samples
    def getAuxiliaryOutputSample(self):
        return self.auxiliaryInputSample
    
    # Set the auxiliary output samples
    def setAuxiliaryOutputSample(self,auxiliaryOutputSample):
        self.auxiliaryOutputSample = auxiliaryOutputSample
        return None
        
    # Get the auxiliary distribution
    def getAuxiliaryDistribution(self):
        return self.auxiliaryDistribution 
    
    # Set the auxiliary distribution
    def setAuxiliaryDistribution(self,distribution):
        self.auxiliaryDistribution  = distribution
        return None

    # Get the auxiliary distribution
    def getNumberOfSample(self):
        return self.numberOfSample 
    
    # Set the auxiliary distribution
    def setNumberOfSample(self,numberOfSample):
        self.numberOfSample  = numberOfSample
        return None
    
    # Get coefficient of variation
    def getCoefficientOfVariation(self):
        return np.sqrt(self.varianceEstimate)/self.probabilityEstimate
        

class GenericCrossEntropy(object):
    """
    Virtual class for the Generic Cross Entropy algorithm


    Attributes
    ----------
    :numberOfAuxiliaryInputSample: number of samples of each Cross Entropy iteration (int)
    
    :maximalIterationNumber: maximal number of iterations for the Cross Entropy (int)

    :event: Threshold event defining the reliability problem (:class:`~openturns.ThresholdEvent`)
    
    :rhoQuantile: quantile level defining the intermediate thresholds (float)
    
    :activeParameters: active parameters of auxiliary distribution (list of booleans)

    """
    def __init__(self,args):
        
        self.numberOfAuxiliaryInputSample  = args[0] 
        self.maximalIterationNumber  = args[1] 

        self.event = args[2]
        self.limitStateFunction = self.event.getFunction() #limit state function
        self.eventThreshold = self.event.getThreshold() #Failure threshold
        self.sampleDimension = self.event.getAntecedent().getDimension() #dimension of input space		
        self.operator = self.event.getOperator() # event operator
        self.initialDistribution = self.event.getAntecedent().getDistribution() #initial distribution
        
        if self.operator(1,2)==True: 
            self.rhoQuantile = args[3] #definition of rho quantile if exceedance failureProbabilitybility
        else:
            self.rhoQuantile = 1 - args[3] #definition of rho quantile in case g<0

        self.activeParameters = np.array(args[4]) #active parameters
        
        self.failureProbability = 0.
        

        #copy of auxiliary distribution
        self.auxiliaryInputSamples = None # Current auxiliaryInputSamples
        self.result = CEResult()

        
    #Accessor to results
    def getResult(self):
        """
        Accessor to the result of Cross Entropy
        
        """
        return self.result
    
    #main function that computes the failure failureProbability
    def run(self):
        """
        Method to estimate the probability of failure
        
        """        

        if self.operator(self.eventThreshold,self.eventThreshold+1) == True:
            eventThresholdLocal = ot.Point([self.eventThreshold+1])
        else: 
            eventThresholdLocal = ot.Point([self.eventThreshold-1])
			
        currentAuxiliaryDistributionParameters = self.initialAuxiliaryDistributionParameters
        iterationNumber  = 0
		#main loop of adaptive importance sampling
        while self.operator(self.eventThreshold,eventThresholdLocal[0]) and iterationNumber < self.maximalIterationNumber:
            
            #theta_courant = currentAuxiliaryDistributionParameters
            self.updateAuxiliaryDistribution(currentAuxiliaryDistributionParameters)
            auxiliaryInputSample= self.auxiliaryDistribution.getSample(self.numberOfAuxiliaryInputSample ) # drawing of auxiliaryInputSamples using auxiliary density
            auxiliaryOutputSample = self.computeOutputSample(auxiliaryInputSample)
            	
            eventThresholdLocal = auxiliaryOutputSample.computeQuantile(self.rhoQuantile) #computation of current quantile
            #print(eventThresholdLocal)
            currentAuxiliaryDistributionParameters = self.optimizeParameters(auxiliaryInputSample,
                                                                             auxiliaryOutputSample,
                                                                             eventThresholdLocal,
                                                                             currentAuxiliaryDistributionParameters)
            iterationNumber += 1
            
        if iterationNumber == self.maximalIterationNumber:
            print('WNG : maximal number of iterations reached')
            
        #Estimate failureProbability
        #self.updateAuxiliaryDistribution(currentAuxiliaryDistributionParameters) #update of auxiliary density 
        y= np.array([self.operator(auxiliaryOutputSample[i][0],self.eventThreshold) for i in range(auxiliaryOutputSample.getSize())]) #find failure points
        indicesCritic=np.where(y==True)[0].tolist() # find failure auxiliaryInputSamples indices
        
        auxiliaryInputSample_critic = auxiliaryInputSample.select(indicesCritic)
        
        criticSamplesInitialPDFValue = self.initialDistribution.computePDF(auxiliaryInputSample_critic) #evaluate initial PDF on failure auxiliaryInputSamples
        
        criticSamplesAuxiliaryPDFValue = self.auxiliaryDistribution.computePDF(auxiliaryInputSample_critic) #evaluate auxiliary PDF on failure auxiliaryInputSamples
        probabilityEstimate = 1./self.numberOfAuxiliaryInputSample * np.sum(np.array([criticSamplesInitialPDFValue])/np.array([criticSamplesAuxiliaryPDFValue])) #Calculation of failure failureProbabilitybility
        
        
        varianceEstimate =  1./self.numberOfAuxiliaryInputSample * 1./(self.numberOfAuxiliaryInputSample - 1) * \
                            np.sum(np.array([criticSamplesInitialPDFValue])**2/np.array([criticSamplesAuxiliaryPDFValue])**2-probabilityEstimate**2)

        self.probabilityEstimate = probabilityEstimate
        self.varianceEstimate = varianceEstimate

        self.auxiliaryInputSamples = auxiliaryInputSample
        
        # Save of data in CEResult
        self.result.setProbabilityEstimate(probabilityEstimate)
        self.result.setVarianceEstimate(varianceEstimate)
        self.result.setAuxiliaryInputSample(auxiliaryInputSample)
        self.result.setAuxiliaryOutputSample(auxiliaryOutputSample)
        self.result.setAuxiliaryDistribution(self.auxiliaryDistribution)
        self.result.setNumberOfSample(iterationNumber*self.numberOfAuxiliaryInputSample)
    
        return None

        
class CrossEntropyAlgorithmPhysicalSpace(GenericCrossEntropy):
    """
    Class for the  Cross Entropy algorithm in the physical space.


    Attributes
    ----------
    
    :numberOfAuxiliaryInputSample: number of samples of each Cross Entropy iteration (int)
    
    :maximalIterationNumber: maximal number of iterations for the Cross Entropy (int)

    :event: Threshold event defining the reliability problem (:class:`~openturns.ThresholdEvent`)
    
    :rhoQuantile: quantile level defining the intermediate thresholds (float)
    
    :activeParameters: active parameters of auxiliary distribution (list of booleans)
    
    :bounds: :class:`~openturns.Interval` bounds on the parameters of auxiliary distribution to be optimized
    
    :auxiliaryDistribution: auxiliary Distribution
    
    :initialAuxiliaryDistributionParameters: initial values of auxiliary distribution parameters for optimization
        
    """
    def __init__(self,*args):
        
        super().__init__(args)

        self.bounds = args[5] #bounds of the active parameters

        self.auxiliaryDistribution = copy.deepcopy(args[6])
        self.auxiliaryDistributionOptimization = copy.deepcopy(args[6])

        self.initialAuxiliaryDistributionParameters = args[7] #initial values of the active parameters	
        self.auxiliaryDistributionParametersDimension = len(self.auxiliaryDistribution.getParameter())#dimension of active parameters
            
        #Check of active parameters list validity
        if len(self.activeParameters )!=len(self.auxiliaryDistribution.getParameter()):
            raise ValueError('Wrong number of active parameters')
			
            #Check of validity of initial theta vector
            if np.sum(self.activeParameters)!=len(self.initialAuxiliaryDistributionParameters):
                raise ValueError('Wrong correspondance between the number of active parameters and the given initial vector of parameters')


    #definition of objective function for Cross entropy
    def CEobjectiveFunction(self,
                            auxiliaryInputSample,
                            auxiliaryOutputSample,
                            currentAuxiliaryDistributionParameters,
                            eventThresholdLocal):
        """
        Method to compute the objective function used to optimize the auxiliary density
        
        :auxiliaryInputSample: list of input :py:class:`openturns.Sample`
        
        :auxiliaryOutputSample: list of output :py:class:`openturns.Sample`
        
        :currentAuxiliaryDistributionParameters: parameters of auxiliaryDistribution (list of floats)
        
        :eventThresholdLocal: local level of quantile at the corresponding iteration
            
        """

        self.updateDistribution(currentAuxiliaryDistributionParameters,self.auxiliaryDistributionOptimization) # update of auxiliary distribution
        
                
        y= np.array([self.operator(auxiliaryOutputSample[i,0],eventThresholdLocal[0]) for i in range(auxiliaryOutputSample.getSize())]) #find failure points
        indicesCritic=np.where(y==True)[0].tolist()  # find failure auxiliaryInputSamples indices
        auxiliaryOutputSample_critic = auxiliaryOutputSample.select(indicesCritic)
        auxiliaryInputSample_critic = auxiliaryInputSample.select(indicesCritic) # select failure auxiliaryInputSamples

        criticSamplesInitialPDFValue = self.initialDistribution.computePDF(auxiliaryInputSample_critic) #evaluate initial PDF on failure auxiliaryInputSamples
        criticSamplesAuxiliaryPDFValue = self.auxiliaryDistributionOptimization.computePDF(auxiliaryInputSample_critic)#evaluate auxiliary PDF on failure auxiliaryInputSamples
        f = 1/self.numberOfAuxiliaryInputSample  * np.sum(np.array([criticSamplesInitialPDFValue])/np.array([criticSamplesAuxiliaryPDFValue])*np.log(criticSamplesAuxiliaryPDFValue)) #calculation of objective function
        return [f]


    def optimizeParameters(self,
                           auxiliaryInputSample,
                           auxiliaryOutputSample,
                           eventThresholdLocal,
                           currentAuxiliaryDistributionParameters):
        """
        Method to optimize the auxiliary density parameters
        
        :auxiliaryInputSample: list of input :py:class:`openturns.Sample`
        
        :auxiliaryOutputSample: list of output :py:class:`openturns.Sample`
                
        :eventThresholdLocal: local level of quantile at the corresponding iteration        
        
        :currentAuxiliaryDistributionParameters: parameters of auxiliaryDistribution (list of floats)
        
        
        """
        f_opt = lambda theta : self.CEobjectiveFunction(auxiliaryInputSample,
                                                        auxiliaryOutputSample,
                                                        theta,
                                                        eventThresholdLocal) #definition of objective function for CE
            
        objective = ot.PythonFunction(np.sum(self.activeParameters), 1, f_opt)
            
        problem = ot.OptimizationProblem(objective) # Definition of CE optimization  problemof auxiliary distribution parameters
        problem.setBounds(self.bounds)
        problem.setMinimization(False)
        
        # TNC algorithm for the optimization-----------------------------------
        solver = ot.TNC(problem)

        optimizationAlgorithm = solver
        optimizationAlgorithm.setMaximumEvaluationNumber(500)
        optimizationAlgorithm.setStartingPoint(currentAuxiliaryDistributionParameters)
        optimizationAlgorithm.run()
        
        # retrieve results
        result = optimizationAlgorithm.getResult()
        currentAuxiliaryDistributionParameters = result.getOptimalPoint()
        
        return currentAuxiliaryDistributionParameters
    
    def computeOutputSample(self,
                            inputSample):
        """
        Method to compute the outputSample from the inputSample
        
        :inputSample: list of input :py:class:`openturns.Sample`
        
        
        """
        outputSample = self.limitStateFunction(inputSample) #evaluation on limit state function
        return outputSample
    
    #definition of function that updates the auxiliary distribution based on new theta values
    def updateAuxiliaryDistribution(self,
                                    auxiliaryDistributionParameters): 
        """
        Method to update the auxiliary distribution from the distribution parameters
        
        :auxiliaryDistributionParameters: parameters of auxiliaryDistribution (list of floats)
        
        
        """
        theta_ = np.array(self.auxiliaryDistribution.getParameter())
        theta_[self.activeParameters] = auxiliaryDistributionParameters
        self.auxiliaryDistribution.setParameter(theta_)
        return None

    def updateDistribution(self,
                           auxiliaryDistributionParameters,
                           distribution): 
        """
        Method to update the auxiliary distribution from the distribution parameters
        
        :auxiliaryDistributionParameters: parameters of auxiliaryDistribution (list of floats)

        :distribution: distribution to be updated 
        
        """
        theta_ = np.array(distribution.getParameter())
        theta_[self.activeParameters] = auxiliaryDistributionParameters
        distribution.setParameter(theta_)
        return None

    
    
# ----------------------------------------------------------------
class CrossEntropyAlgorithmStandardSpace(GenericCrossEntropy):
    """
    Class for the  Cross Entropy algorithm in the standard space.


    Attributes
    ----------
    
    :numberOfAuxiliaryInputSample: number of samples of each Cross Entropy iteration (int)
    
    :maximalIterationNumber: maximal number of iterations for the Cross Entropy (int)

    :event: Threshold event defining the reliability problem (:class:`~openturns.ThresholdEvent`)
    
    :rhoQuantile: quantile level defining the intermediate thresholds (float)
    
    :activeParameters: list of two booleans indicating if the mean value (first boolean) and the standard deviation (second boolean) of the auxiliary distribution defined in the standard space have to be optimized
    
    :initialAuxiliaryDistributionParameters: initial values of auxiliary distribution parameters for optimization
        
    """
    

    def __init__(self,*args):
        
        super().__init__(args)
        
        self.inverseIsoprobabilisticTransformation = self.initialDistribution.getInverseIsoProbabilisticTransformation()
        self.initialDistributionPhysicalSpace = copy.deepcopy(self.initialDistribution)
        self.initialDistribution = ot.ComposedDistribution([ot.Normal()]*self.sampleDimension)
        self.auxiliaryDistribution = ot.ComposedDistribution([ot.Normal()]*self.sampleDimension)
        self.initialAuxiliaryDistributionParameters = args[5]
        
    def computeOutputSample(self,
                            inputSample):
        """
        Method to compute the outputSample from the inputSample
        
        :inputSample: list of input :py:class:`openturns.Sample`
        
        
        """        
        outputSample = self.limitStateFunction(self.inverseIsoprobabilisticTransformation(inputSample)) #evaluation on limit state function
        
        return outputSample

    #definition of function that updates the auxiliary distribution based on new theta values
    def updateAuxiliaryDistribution(self,
                                    auxiliaryDistributionParameters):
        """
        Method to update the auxiliary distribution from the distribution parameters
        
        :auxiliaryDistributionParameters: parameters of auxiliaryDistribution (list of floats)
        
        """
        
        if self.activeParameters[0] == True:    
            theta_ = auxiliaryDistributionParameters[0:self.sampleDimension]
            
            aux_param = np.array(self.auxiliaryDistribution.getParameter())
            mask = [True,False]*self.sampleDimension
            aux_param[mask] = theta_
            self.auxiliaryDistribution.setParameter(aux_param)
  
        if self.activeParameters[1] == True :
            theta_ = auxiliaryDistributionParameters[self.sampleDimension:]
            aux_param = np.array(self.auxiliaryDistribution.getParameter())
            mask = [False,True]*self.sampleDimension
            aux_param[mask] = theta_
            self.auxiliaryDistribution.setParameter(aux_param)

        return None 
            

    def optimizeParameters(self,
                           auxiliaryInputSample,
                           auxiliaryOutputSample,
                           eventThresholdLocal,
                           currentAuxiliaryDistributionParameters):
        """
        Method to optimize the auxiliary density parameters
        
        :auxiliaryInputSample: list of input :py:class:`openturns.Sample`
        
        :auxiliaryOutputSample: list of output :py:class:`openturns.Sample`
                
        :eventThresholdLocal: local level of quantile at the corresponding iteration        
        
        :currentAuxiliaryDistributionParameters: parameters of auxiliaryDistribution (list of floats)
        
        
        """ 
        
        if self.activeParameters[1] == True and self.activeParameters[0]==False :
            raise('Error : standard deviations cannot be optimized without means')
        
        y = np.array([self.operator(auxiliaryOutputSample[i,0],eventThresholdLocal[0]) for i in range(auxiliaryOutputSample.getSize())]) #find failure points
        indicesCritic=np.where(y==True)[0].tolist()  # find failure auxiliaryInputSamples indices
        auxiliaryInputSample_critic = auxiliaryInputSample.select(indicesCritic) # select failure auxiliaryInputSamples

        criticSamplesInitialPDFValue = self.initialDistribution.computePDF(auxiliaryInputSample_critic) #evaluate initial PDF on failure auxiliaryInputSamples
        criticSamplesAuxiliaryPDFValue = self.auxiliaryDistribution.computePDF(auxiliaryInputSample_critic)#evaluate auxiliary PDF on failure auxiliaryInputSamples


        if self.activeParameters[0] == True : # optimization of mean values
        
            denom = np.sum(np.array([criticSamplesInitialPDFValue])/np.array([criticSamplesAuxiliaryPDFValue]))        
            mu = np.zeros(self.sampleDimension)
            
            for i in range(self.sampleDimension):
                mu[i] = np.sum(np.array([criticSamplesInitialPDFValue])/np.array([criticSamplesAuxiliaryPDFValue])*np.array(auxiliaryInputSample_critic[:,i]) )/denom
            if self.activeParameters[1] == True : # optimization of std deviations
            
                sigma = np.zeros(self.sampleDimension)
            
                for i in range(self.sampleDimension):
                    diff = (np.array(auxiliaryInputSample_critic[:,i]) - mu[i])**2
                    sigma[i] = np.sqrt(np.sum(np.array([criticSamplesInitialPDFValue])*diff /np.array([criticSamplesAuxiliaryPDFValue]))/denom)
                         
                return np.hstack((mu,sigma))
            else:
                return mu
        else:
            raise('No active parameters have been set')
            
if __name__ == "__main__":
    import openturns as ot

    #Creation of the event
    distribution_R = ot.LogNormalMuSigma(300.0, 30.0, 0.0).getDistribution()
    distribution_F = ot.Normal(75e3, 5e3)
    marginals = [distribution_R, distribution_F]
    distribution = ot.ComposedDistribution(marginals)
    
    # create the model
    model = ot.SymbolicFunction(['R', 'F'], ['R-F/(pi_*100.0)'])
    
    #create the event 
    vect = ot.RandomVector(distribution)
    G = ot.CompositeRandomVector(model, vect)
    event = ot.ThresholdEvent(G, ot.Less(), -50.0)
    
    # Hyperparameters of the algorithm
    n_IS= 1000 # Number of samples at each iteration
    rho_quantile= 0.15 # Quantile determining the percentage of failure samples in the current population 
    n_iter = 10
    # estimation by CMC 
    #Determination of reference probability
    #MonteCarlo experiment
    n_MC = 5e6
    
    # create a Monte Carlo algorithm
    experiment = ot.MonteCarloExperiment()
    algo = ot.ProbabilitySimulationAlgorithm(event, experiment)
    algo.setMaximumOuterSampling(int(n_MC))
    algo.setMaximumCoefficientOfVariation(0.01)
    algo.run()
    # retrieve results
    result = algo.getResult()
    probability = result.getProbabilityEstimate()
         
    #  Cross entropy in physical space   
    ## Definition of auxiliary distribution
    distribution_margin1 = ot.LogNormalMuSigma().getDistribution()
    distribution_margin2 = ot.Normal()
    aux_marginals = [distribution_margin1, distribution_margin2]
    aux_distribution = ot.ComposedDistribution(aux_marginals)
    
    ## Definition of parameters to be optimized
    active_parameters = [True,True,True,True,True] # active parameters from the auxiliary distribution which will be optimized
    ### WARNING : native parameters of distribution have to be considered
    
    bounds = ot.Interval([3,0.09,0.,50e3,2e3], # bounds on the active parameters
                         [7,0.5,0.5,100e3,10e3])
    
    initial_theta= [5.70,0.1,0.,75e3,5e3] # initial value of the active parameters
    
    ## Definition of the algorithm
    CE_physical = CrossEntropyAlgorithmPhysicalSpace(n_IS,
                                                      n_iter,
                                                      event,
                                                      rho_quantile,
                                                      active_parameters,
                                                      bounds,
                                                      aux_distribution,
                                                      initial_theta)
    
    # Run of the algorithm
    CE_physical.run()
     
    CE_physicalresults = CE_physical.getResult()

    # Cross entropy in standard space   
    active_parameters = [True,True] # We optimize both means and standard deviations of auxiliary distribution (in standard space)
        
    initial_theta= [0.,0.,1.,1.] # initial value of the active parameters

         ## Definition of the algorithm
    CE_standard = CrossEntropyAlgorithmStandardSpace(n_IS,
                                                      n_iter,
                                                      event,
                                                      rho_quantile,
                                                      active_parameters,
                                                      initial_theta)
    # Run of the algorithm
    CE_standard.run()
        
    CE_standardresults = CE_standard.getResult()
    
    print('         | P estimate |    P ref   |  diff (%)  | Coeff. of var')
    print('----------------------------------------------------------------')
    print( 'Standard |',"  %5.2E |   %5.2E |     %2.2f  |         %2.2f"%(CE_standardresults.getProbabilityEstimate(),
                                                                 probability,
                                                                 (CE_standardresults.getProbabilityEstimate()-probability)/probability*100,
                                                                 CE_standardresults.getCoefficientOfVariation()))

    print( 'Physical |',"  %5.2E |   %5.2E |     %2.2f  |         %2.2f"%(CE_physicalresults.getProbabilityEstimate(),
                                                                     probability,
                                                                     (CE_physicalresults.getProbabilityEstimate()-probability)/probability*100,
                                                                     CE_physicalresults.getCoefficientOfVariation()))