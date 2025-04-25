import numpy as np
import matplotlib.pyplot as plt
import yaml
from tqdm import tqdm

def make_nosie_data(x,theta,sigma2,random_seed=None):
    if random_seed !=None:
        np.random.seed(random_seed)
    theta0,theta1=theta
    true_y=theta0+theta1*x
    data_y=np.random.normal(true_y,sigma2)
    return data_y

def linear_model(x,theta):
    theta0,theta1=theta
    linear_y=theta0+theta1*x
    return linear_y

def gauss(x,mu,sigma):
    return 1/sigma/np.sqrt(2*np.pi)*np.exp(-(x-mu)**2/2/sigma**2)

class my_MCMC:
    def __init__(self,data_x,data_y,prior_file='./prior.yaml'):
        self.data_x=data_x
        self.data_y=data_y
        self.data_size=len(data_y)
        with open(prior_file, 'r') as f:
            config = yaml.safe_load(f)
        self.my_prior = config["prior"]
        self.theta_size=len(self.my_prior)

    def log_likelihood(self,theta,model=linear_model):
        model_y=model(self.data_x,theta)
        chi2=np.sum((model_y-self.data_y)**2)
        return -0.5*chi2
    
    def log_prior(self,theta):
        total_prior=1
        for idx, (name, settings) in enumerate(self.my_prior.items()):
            if settings['type']=='uni':
                range=settings['range'][1]-settings['range'][0]
                if settings['range'][0]<theta[idx]<settings['range'][1]:
                    total_prior *= 1/range
                else:
                    return -np.inf
            elif settings['type']=='gauss':
                total_prior *= gauss(theta[idx], settings['range'][0] ,settings['range'][1])
        return np.log(total_prior)


    def log_probability(self,theta,model=linear_model):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta,model)
    
    def MCMC(self,start_theta,N_setp,step_size=0.1):
        start_theta=np.array(start_theta)
        pre_theta=start_theta
        N_walker=np.shape(pre_theta)[0]

        if start_theta.ndim==1:
            pre_theta=np.array([start_theta])
            N_walker=1

        size=np.shape(pre_theta)
        
        self.chain=np.zeros((N_setp+1,size[0],size[1]))
        self.chain[0]=pre_theta

        for ii in tqdm(range(N_setp)):
            pre_p=np.array([self.log_probability(jj) for jj in pre_theta])
            post_theta=pre_theta+np.random.uniform(-step_size/2,step_size/2,size=size)
            post_p=np.array([self.log_probability(jj) for jj in post_theta])

            alpha=post_p-pre_p
            jump_pos=alpha>np.log(np.random.uniform(size=N_walker))    
            pre_theta[jump_pos]=post_theta[jump_pos]
            self.chain[ii+1]=pre_theta

        return self.chain
    
    def burn_in(self,discard):
        self.chain=self.chain[discard:,:,:]
        return 
    
    def get_chain(self):
        return self.chain.reshape(-1, self.theta_size)
    