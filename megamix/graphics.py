# -*- coding: utf-8 -*-
"""
Created on Mon May 22 09:38:22 2017

:author: Elina Thibeau-Sutre
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.patches import Ellipse
from sklearn import manifold

    
def ellipses_multidimensional(cov,mean,d1=0,d2=1):
    """
    A method which creates an object of Ellipse class (from matplotlib.patches).
    As it is only a 2D object dimensions may be precised to project it.
    
    @return: ell (Ellipse)
    """
    
    covariance = np.asarray([[cov[d1,d1],cov[d1,d2]],[cov[d1,d2],cov[d2,d2]]])
    lambda_, v = np.linalg.eig(covariance)
    lambda_sqrt = np.sqrt(lambda_)
    
    width = lambda_sqrt[0] * 2
    height = lambda_sqrt[1] * 2
    
    angle = np.rad2deg(np.arccos(v[0,0]))
    
    ell = Ellipse(xy=(mean[d1], mean[d2]),
              width=width, height=height,
              angle=angle)
    ell.set_facecolor('none')
    ell.set_edgecolor('k')
    return ell

def create_graph(self,points,directory,legend):
    """
    This method draws a 2D graph displaying the clusters and their means and saves it as a PNG file.
    If points have more than two coordinates, then it will be a projection including only the first coordinates.
    
    @param points: an array of points (n_points,dim)
    @param log_resp: an array containing the log of soft assignements of every point (n_points,n_components)
    @param directory: the path to the directory where the figure will be saved (str)
    @param legend: a legend for the name of the figure (str)
    """

    n_points,dim = points.shape
    log_resp = self.predict_log_resp(points)
    
    plt.title("iter = " + str(self.iter) + " k = " + str(self.n_components))
    
    if self.covariance_type == "full":
        cov = self.cov
    elif self.covariance_type == "spherical":
        cov = np.asarray([np.eye(dim) * coeff for coeff in self.cov])
    
    x_points = [[] for i in range(self.n_components)]
    y_points = [[] for i in range(self.n_components)]
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    for i in range(self.n_components):
                             
        x_points[i] = [points[j][0] for j in range(n_points) if (np.argmax(log_resp[j])==i)]        
        y_points[i] = [points[j][1] for j in range(n_points) if (np.argmax(log_resp[j])==i)]
        
        if len(x_points[i]) > 0:
            ell = ellipses_multidimensional(cov[i],self.means[i])
            ax.plot(x_points[i],y_points[i],'o',alpha = 0.2)
            ax.plot(self.means[i][0],self.means[i][1],'kx')
            ax.add_artist(ell)
        
    titre = directory + '/figure_' + self.init + "_" + str(legend) + ".png"
    plt.savefig(titre)
    plt.close("all")
    
def create_graph_convergence_criterion(self,directory,legend):
    """
    This method draws a graph displaying the evolution of the convergence criterion :
        - log likelihood for GMM (should never decrease)
        - the lower bound of the log likelihood for VBGMM/DPGMM (should never decrease)
    
    @param directory: the path to the directory where the figure will be saved (str)
    @param legend: a legend for the name of the figure (str)
    """
    
    plt.title("Convergence criterion evolution")
    
    p1, = plt.plot(np.arange(self.iter),self.convergence_criterion_data,marker='x',label='data')
    if self.early_stopping:
        p3, = plt.plot(np.arange(self.iter),self.convergence_criterion_test,marker='o',label='test')
        
    plt.legend(handler_map={p1: HandlerLine2D(numpoints=4)})

    
    titre = directory + '/figure_' + self.init + "_" + str(legend) + "_cc_evolution.png"
    plt.savefig(titre)
    plt.close("all")
    
def create_graph_weights(self,directory,legend):
    """
    This method draws an histogram illustrating the repartition of the weights of the clusters
    
    @param directory: the path to the directory where the figure will be saved (str)
    @param legend: a legend for the name of the figure (str)
    """
    
    plt.title("Weights of the clusters")
    
    plt.hist(np.exp(self.log_weights))
    
    titre = directory + '/figure_' + self.init + "_" + str(legend) + "_weights.png"
    plt.savefig(titre)
    plt.close("all")
    
def create_graph_MDS(self,directory,legend):
    """
    This method displays the means of the clusters on a 2D graph in order to visualize
    the cosine distances between them
    (see scikit-learn doc on manifold.MDS for more information)
    
    @param directory: the path to the directory where the figure will be saved (str)
    @param legend: a legend for the name of the figure (str)
    """
    
    mds = manifold.MDS(2)
    norm = np.linalg.norm(self.means,axis=1)
    means_normed = self.means / norm[:,np.newaxis]
    
    coord = mds.fit_transform(means_normed)
    stress = mds.stress_
    
    plt.plot(coord.T[0],coord.T[1],'o')
    plt.title("MDS of the cosine distances between means, stress = " + str(stress))
    
    titre = directory + '/figure_' + self.init + "_" + str(legend) + "_means.png"
    plt.savefig(titre)
    plt.close("all")
    
def create_graph_entropy(self,directory,legend):
    """
    This method draws an histogram illustrating the repartition of the
    entropy of the covariance matrices
    
    @param directory: the path to the directory where the figure will be saved (str)
    @param legend: a legend for the name of the figure (str)
    """
    if self.covariance_type == 'spherical':
        print('This graph is irrelevant')
    
    else:
        plt.title('Entropy of the covariances')
        
        l = np.linalg.eigvalsh(self.cov)
        norm = np.trace(self.cov.T)
        l = l/norm[:,np.newaxis]
        
        log_l = np.log(l)
        ent = - np.sum(log_l*l, axis=1)
        
        plt.hist(ent)
        
        titre = directory + '/figure_' + self.init + "_" + str(legend) + "_cov_entropy.png"
        plt.savefig(titre)
        plt.close("all")