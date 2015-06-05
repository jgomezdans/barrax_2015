#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.optimize as opt
import json
try:
    import prosail
except ImportError:
    print "We need to have the prosail Python bindings installed"
    print "Available from http://github.com/jgomezdans/prosail"

__author__ = "J Gomez-Dans"
__version__ = "1.0 (05.06.2015)"
__email__ = "j.gomez-dans@ucl.ac.uk"





def plot_config ():
    """Update the MPL configuration"""
    config_json='''{
            "lines.linewidth": 2.0,
            "axes.edgecolor": "#bcbcbc",
            "patch.linewidth": 0.5,
            "legend.fancybox": true,
            "axes.color_cycle": [
                "#FC8D62",
                "#66C2A5",
                "#8DA0CB",
                "#E78AC3",
                "#A6D854",
                "#FFD92F",
                "#E5C494",
                "#B3B3B3"
            ],
            "axes.facecolor": "#eeeeee",
            "axes.labelsize": "large",
            "axes.grid": false,
            "patch.edgecolor": "#eeeeee",
            "axes.titlesize": "x-large",
            "svg.embed_char_paths": "path",
            "xtick.direction" : "out",
            "ytick.direction" : "out",
            "xtick.color": "#262626",
            "ytick.color": "#262626",
            "axes.edgecolor": "#262626",
            "axes.labelcolor": "#262626",
            "axes.labelsize": 14,
            "font.size": 14,
            "legend.fontsize": 14,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14
            
    }
    '''
    s = json.loads ( config_json )
    plt.rcParams.update(s)
    plt.rcParams["axes.formatter.limits"] = [-4,4]
    

def pretty_axes( ax=None ):
    """This function takes an axis object ``ax``, and makes it purrty.
    Namely, it removes top and left axis & puts the ticks at the
    bottom and the left"""
    if ax is None:
        ax = plt.gca()
    ax.spines["top"].set_visible(False)  
    ax.spines["bottom"].set_visible(True)  
    ax.spines["right"].set_visible(False)              
    ax.spines["left"].set_visible(True)  

    ax.get_xaxis().tick_bottom()  
    ax.get_yaxis().tick_left()  
    loc = plt.MaxNLocator( 6 )
    ax.yaxis.set_major_locator( loc )
    # Make text output also look sane...
    np.set_printoptions ( suppress=True, precision=3 )

    ax.tick_params( axis="both", which="both", bottom="on", top="off",  
            labelbottom="on", left="on", right="off", labelleft="on"   )
    

def call_prospect ( n, cab, car, cbrown, cw, cm ):
    """A wrapper for PROSPECT that does away with some numerical 
    instabilities by interpolating over spectrally."""
    rr = prosail.prospect_5b ( n, cab, car, cbrown, cw, cm )
    r = rr[:,0] # Reflectance
    t = rr[:,1] # Transmittance
    x = np.arange ( 400, 2501 )
    rpass = np.isfinite ( r )
    tpass = np.isfinite ( t )
    ri = np.interp ( x, x[rpass], r[rpass])
    ti = np.interp ( x, x[tpass], t[tpass])
    return np.c_[ ri, ti]

def prospect_sensitivity_ssa ( x0=np.array([1.5, 40., 5., 0., 0.0113, 0.0053]), \
                              epsilon=1e-5, do_plots=True ):
    """Local approximation (around ``x0``) of the sensitivity of the PROSPECT
    model to each input parameter. YOu can set up the point around which the
    sensitivity is calculated (``x0``), the value of the finite difference 
    (``epsilon``).
    
    Parameters
    -------------
    x0: array
        A size 6 array, with the centre point around which the partial derivatives
        will be calculated. The different elements are in PROSPECT order, e.g.
        N, Cab, Car, Cbrown, Cw and Cm
    epsilon: float
        The finite difference amount. If you get NaN, make it a bit larger.
    do_plots: bool
        Whether to do some pretty plots
        
    Returns
    ----------
    
    """
    sensitivity = np.zeros((6,2101))
    span = np.array([1.5, 80., 20., 1., 0.0439-0.0043, 0.0152-0.0017 ])
    for i in xrange(6):
        r0 = prosail.prospect_5b ( *x0 ).sum(axis=1)
        xp = x0*1.
        xp[i] = x0[i] + epsilon*span[i]
        r1 = call_prospect ( *xp ).sum(axis=1)
        sensitivity[i,:] = (r0-r1)/epsilon
    
    if do_plots:
        wv = np.arange( 400, 2501 )
        fig, axs = plt.subplots ( figsize=(10,10), nrows=2, ncols=3, 
                                     sharex=True, sharey=True )
        axs = axs.flatten()
        for i,input_parameter in enumerate( ['n', 'cab', 'car', 'cbrown', 'cw', 'cm'] ):
            axs[i].plot ( wv, sensitivity[i,:], '-', lw=2)
            axs[i].set_title( input_parameter )
            axs[i].set_xlim ( 400, 2500 )
            if i in [ 0, 3]:
                axs[i].set_ylabel(r'$\partial f/\partial \mathbf{x}$')
            if i > 2:
                axs[i].set_xlabel ("Wavelength [nm]")
            pretty_axes ( axs[i] )
        plt.figure(figsize=(10,10))
        for i,input_parameter in enumerate( ['n', 'cab', 'car', 'cbrown', 'cw', 'cm'] ):
            plt.plot( wv, sensitivity[i,:], lw=2, label=input_parameter )
            pretty_axes()
        plt.xlim ( 400, 2500)
        plt.ylabel(r'$\partial f/\partial \mathbf{x}$')
        plt.xlabel ("Wavelength [nm]")
        plt.legend(loc='best')
    return wv, sensitivity
    
def prospect_sensitivity_n ( n_samples=20, spaces=10, 
                            minvals = {'n':1, 'cab':10, 'car':0, 'cbrown': 0, 'cw':0., 'cm':0.0 },
                            maxvals = { 'n': 3.5, 'cab': 280, 'car':200, 'cbrown': 1, 'cw':0.4, 'cm': 0.5 },
                            do_plots=True ):
    """Visualise the effect of the PROSPECT parameter $N$, the number of layers. The aim of this is for
    you to view how the response of the leaves changes as a function of this parameter, while the other
    parameters are modified randomly (within the parameter spans provided).
    
    Parameters
    ------------
    n_samples: int
        The number of random model realisations for each bin of N.
    spaces: int
        The number of bins in which to divide the span of N
    minvals: dict
        A dictionary of parameter minimum values in the adequate units.
    maxvals: dict
        A dictionary of parameter minimum values in the adequate units.
    do_plots: bool
        Whether to do plots or not
        
    Returns
    -----------
    xleafn, reflectance, transmittance: The value of N used, as well as the reflectance and transmittance.
    """
    from mpl_toolkits.mplot3d import Axes3D
    refl = {}
    trans = {}
    xbase = np.zeros ( 6 )
    span  = np.zeros ( 6 )
    for i, param in enumerate ( [ "n", "cab", "car", "cbrown", 'cw', 'cm'] ):
        xbase[i] = minvals[param]
        span[i] = (maxvals[param] - minvals[param])*0.5
    spaces = 10
    t = []
    r = []
    xleafn = []
    delta = ( maxvals['n'] - minvals['n'] )/spaces
    for s in np.linspace ( minvals['n'], maxvals['n'], num=spaces ):
        for i in xrange( n_samples ):
            x = xbase + span*np.random.rand(6)
            tx = s + delta*(np.random.rand() -0.5)
            x[0] = tx
            rr = call_prospect ( *x )
            r.append ( rr[:, 0])
            t.append ( rr[:, 1])
            xleafn.append ( tx )
    r = np.array ( r )
    t = np.array ( t )
    xleafn = np.array ( xleafn )
    return xleafn, r, t

def do_index ( mode="refl", sweep_param="cab", band1=650, band2=850, bwidth1=5, bwidth2=5,
        n_samples=1, spaces=20, minvals = {'n':0, 'cab':0, 'car':0, 'cbrown': 0, 'cw':0., 'cm':0.0 },
        maxvals = { 'n': 3.5, 'cab': 80, 'car':200, 'cbrown': 1, 'cw':0.4, 'cm': 0.5 },
        do_plots=True ):
    
    """A function that runs PROSPECT for a particular parameter (named in ``sweep_param``), and 
    calculates a normalised vegetation index with two bands. The function uses a top-hat bandpass
    functin, defined by the band centres (``band1`` and ``band2``), and the bandwiths for each
    band (``bwidth1`` and ``bwidth2``). The function works either for calculations of the reflectance,
    transmittance or single scattering albedo. The chosen parameter is swept between the boundaries
    given by the corresponding entry in the ``minvals`` and ``maxvals`` dictionaries. Additionally, you
    can do some plots. 
    
    Parameters
    --------------
    mode: str
        Whether to use reflectance (``refl``), transmittance (``trans``) or single scattering albedo
        (``ssa``).
    sweep_param: str
        The parameter which you'll be varying, by it's symbol (e.g. "cab").
    band1: int
        The location of the first band in nanometers.
    band2: int
        The location of the second band in nanometers.
    bwidth1: int
        The (half) bandwidth of the first band in nanometers
    bwidth2: int
        The (half) bandwidth of the second band in nanometers
    n_samples: int
        The number of random samples to draw for each value of the sweep parameter
    spaces: int
        The number of chunks in which to divide the span of the sweep parameter (more ``spaces``, 
        more resolution, as it were).
    minvals: dict
        Dictionary with minimum values for each parameter
    maxvals: dict
        Dictionary with maximum values for each parameter
    do_plots: bool
        Whether to do nice scatter plot of the index vs the sweep parameter value
        
    Returns
    --------
    The sweep parameter and the vegetation index as arrays.
        
    
    """
    xbase = np.zeros ( 6 )
    span  = np.zeros ( 6 )
    pars = [ "n", "cab", "car", "cbrown", 'cw', 'cm']
    for i, param in enumerate ( pars ):
        xbase[i] = minvals[param]
        span[i] = ( maxvals[param] - minvals[param] )
        
    r = []
    t = []
    wv = np.arange ( 400, 2501 )
    ss =[]
    delta = ( maxvals[sweep_param] - minvals[sweep_param] )/spaces
    for s in np.linspace ( minvals[sweep_param], maxvals[sweep_param], num=spaces ):
        for i in xrange( n_samples ):
            x = xbase + span*np.random.rand(6)
            tx = s + delta*(np.random.rand() -0.5)
            x[pars.index ( sweep_param )] = tx
            rr = call_prospect ( *x )
            r.append ( rr[:, 0])
            t.append ( rr[:, 1])
            ss.append ( tx )
    r = np.array ( r )
    t = np.array ( t )
    s = np.array ( ss )
    
    isel1 = np.logical_and ( wv >= band1 - bwidth1,  wv <= band1 + bwidth1)
    isel2 = np.logical_and ( wv >= band2 - bwidth2,  wv <= band2 + bwidth2)

    if mode == "refl":
        b1 = r[:, isel1]
        b2 = r[:, isel2]
        
    elif mode == "trans":
        b1 = t[:, isel1]
        b2 = t[:, isel2]
    elif mode == "ssa":
        b1 = r[:, isel1]  + t[:, isel1] 
        b2 = r[:, isel2]  + t[:, isel2]
    else:
        raise ValueError, "`mode` can only 'refl', 'trans' or 'ssa'!"
    
    vi = (b2.mean(axis=1) - b1.mean(axis=1))/(b2.mean(axis=1)+b1.mean(axis=1))
    if do_plots:
        plt.plot ( vi, s, 'o', markerfacecolor="none")
        plt.ylabel ( sweep_param )
        plt.xlabel("VI")
        plt.title ("B1: %g, B2: %G" % ( band1, band2) )
        pretty_axes()
    return s, vi



def red_edge ( mode="refl", sweep_param="cab", band1=670, band2=780,
        n_samples=150, spaces=7, minvals = {'n':1.0, 'cab':15, 'car':10, 'cbrown': 0, 'cw':0., 'cm':0.0 },
        maxvals = { 'n': 2.5, 'cab': 80, 'car':20, 'cbrown': 1, 'cw':0.4, 'cm': 0.5 },
        do_plots=True ):
    
    """A function that runs PROSPECT for a particular parameter (named in ``sweep_param``), and 
    calculates a normalised vegetation index with two bands. The function uses a top-hat bandpass
    functin, defined by the band centres (``band1`` and ``band2``), and the bandwiths for each
    band (``bwidth1`` and ``bwidth2``). The function works either for calculations of the reflectance,
    transmittance or single scattering albedo. The chosen parameter is swept between the boundaries
    given by the corresponding entry in the ``minvals`` and ``maxvals`` dictionaries. Additionally, you
    can do some plots. 
    
    Parameters
    --------------
    mode: str
        Whether to use reflectance (``refl``), transmittance (``trans``) or single scattering albedo
        (``ssa``).
    sweep_param: str
        The parameter which you'll be varying, by it's symbol (e.g. "cab").
    band1: int
        The location of the first band in nanometers.
    band2: int
        The location of the second band in nanometers.
    n_samples: int
        The number of random samples to draw for each value of the sweep parameter
    spaces: int
        The number of chunks in which to divide the span of the sweep parameter (more ``spaces``, 
        more resolution, as it were).
    minvals: dict
        Dictionary with minimum values for each parameter
    maxvals: dict
        Dictionary with maximum values for each parameter
    do_plots: bool
        Whether to do nice scatter plot of the index vs the sweep parameter value
        
    Returns
    --------
    The sweep parameter and the location of the turning point in the red edge.
        
    
    """
    
    xbase = np.zeros ( 6 )
    span  = np.zeros ( 6 )
    pars = [ "n", "cab", "car", "cbrown", 'cw', 'cm']
    for i, param in enumerate ( pars ):
        xbase[i] = minvals[param]
        span[i] = ( maxvals[param] - minvals[param] )
        
    r = []
    t = []
    
    wv = np.arange ( 400, 2501 )
    isel = np.logical_and ( wv >= band1,  wv <= band2 )
    wv = np.arange ( band1, band2+1 )


    ss =[]
    delta = ( maxvals[sweep_param] - minvals[sweep_param] )/spaces
    
    red_edge = []
    rred_edge = []
    unc = []
    fig, axs = plt.subplots( nrows=1, ncols=2, figsize=(12,5))
    for ii, s in enumerate ( np.linspace ( minvals[sweep_param], maxvals[sweep_param], num=spaces ) ): 
        yy = np.zeros( (isel.sum(), n_samples ) )
        for i in xrange( n_samples ):
            x = xbase + span*np.random.rand(6)
            x=np.array([1.5, 40., 5., 0., 0.0113, 0.0053])
            tx = s + delta*(np.random.rand() -0.5)
            x[pars.index ( sweep_param )] = tx
            rr = call_prospect ( *x )
            if mode == "refl":
                y = rr[isel, 0]
            elif mode == "trans":
                y = rr[isel, 1]
            elif mode == "ssa":
                y = rr[isel, 0] + rr[isel, 1]
            yy[:, i] = y
        
        pp, cov= curve_fit(lambda x, a, b, c, d: a + b*x+c*x*x+d*x*x*x, wv, 
                          yy.mean(axis=1))
        ss.append ( s )
        red_edge.append ( -pp[2]/(3*pp[3]))
        if do_plots:
            iloc = np.argmin ( np.abs(wv - -pp[2]/(3*pp[3]) ))
            if np.abs(wv - -pp[2]/(3*pp[3]) )[iloc] <= 5:
                axs[0].plot(wv[iloc], yy.mean(axis=1)[iloc], 'ko')
            axs[0].plot(wv, yy.mean(axis=1), '-')
            
            axs[0].set_xlabel ("Wavelength [nm]")
            axs[0].set_ylabel ( mode )
            pretty_axes ( axs[0])

            
    if do_plots:
        axs[1].plot ( ss, red_edge, 'o-')
        axs[1].set_xlabel(sweep_param)
        axs[1].set_ylabel (r'$\lambda_{red\, edge}\;\left[nm\right]$')
        pretty_axes ( axs[1])
    
    return np.array(ss), np.array(red_edge)


def read_lopex_sample ( sample_no=23, do_plot=True, db_path = "data/LOPEX93/" ):
    """Reads a sample from the LOPEX'93 database, and optionally, plots the
    reflectance and transmittance. By default, there's an oak leaf.
    
    Parameters
    ------------
    sample_no: int
        Integer to the sample number in the database
    do_plot: bool
        Whether to do the plot or not
    db_path: str
        A path for the database
    
    Returns the reflectance and transmittance associated with the sample, typically
    there are 5 replicates, so two arrays of size [5, 2101], over 400 to 2500 nm.
    """
    if sample_no < 1 or sample_no > 116:
        raise ValueError ("There are only 116 samples in this database")
    refl = np.loadtxt("%s/refl.%03d.dat" % ( db_path,  sample_no )).reshape((5,2101))
    trans = np.loadtxt("%s/trans.%03d.dat" % ( db_path,  sample_no ) ).reshape((5,2101))
    if do_plot:
        fig, axs = plt.subplots ( nrows=2, ncols=1, figsize=(10,10), sharex=True )
        for nsample in xrange(5):
            axs[0].plot ( wv, refl[nsample,:])
            axs[1].plot ( wv, trans[nsample,:])
        axs[0].set_title("Reflectance")
        axs[1].set_title("Transmittance")
        axs[1].set_xlabel("Wavelength [nm]")
    return refl, trans




def optimise_random_starts (  cost_function, refl, trans, n_tries = 20, 
                            lobound = np.array ( 
                            [ 1.2, 0, 0, 0, 0.0043, 0.0017]),
         hibound = np.array ( 
                            [ 2.5, 80, 20, 1, 0.0439, 0.0152]), 
         verbose = True, do_plots=True  ):
    """A function for minimising a cost function with reflectance and transmittance 
    values. To account for the local nature of the gradient descent minimiser, the
    optimisation is started from 20 (or ``n_tries``) random points within the parameter
    boundaries. The best (lowest) solution is selected and reported. Additional options 
    let you change these boundaries, as well as the amount of information it reports back,
    and comparison "goodness of fit" plot.
    
    Parameters
    ------------
    cost_function: function
        A function that calculates the cost. Must take three parameters: a 6 element
        vector, a refletance and a transmittance measurement. 
    refl: arr
        A reflectance array measured between 400 and 2500 nm at 1nm intervals
    trans: arr
        A transmittance array measured between 400 and 2500 nm at 1nm intervals
    lobound: arr
        A 6-element array with lower boundaries for the 6 PROSPECT5b parameters
    hibound: arr
        A 6-element array with upper boundaries for the 6 PROSPECT5b parameters
    verbose: bool
        Be verbose
    do_plots: bool
        Do a plot
        
    Returns
    ----------
    The forward modelled reflectance and transmittance, the model parameters at the
    minimum and the value of the cost function there.
    """

    bounds = [ [lobound[i], hibound[i]] for i in xrange(6)]

    store = []
    
    for tries in xrange ( n_tries ):
        x0 = lobound + np.random.rand(6)*(hibound - lobound )
        retval = opt.minimize ( the_cost_function, x0, args=(refl, trans), jac=False, bounds=bounds, \
                       options={"disp":10})
        store.append ( retval )
        if verbose:
            print "Optimisation %d" % ( tries + 1)
            print "\tx_opt:", retval.x
            print "\tcost: ", retval.fun
    i =np.argmin([ res.fun for res in store ])
    fwd = prosail.prospect_5b ( *store[i].x )
    fwd_refl = fwd[:, 0]
    fwd_trans = fwd[:, 1]
    
    if verbose:
        print store[i]  
    
    if do_plots:
        fig, axs = plt.subplots ( nrows=2, ncols=1, figsize=(10,10), sharex=True )
        axs = axs.flatten()
        l1 = axs[0].plot ( wv, refl, '--', 
                    label="Measurements")
        l2 = axs[0].plot (wv, prosail.prospect_5b ( *store[i].x )[:,0], '-', lw=2, 
                     label="Fitted PROSPECT")
        axs[1].plot ( wv, trans, '--')
        axs[1].plot (wv, prosail.prospect_5b ( *store[i].x )[:,1], '-', lw=2)
        axs[0].set_title("Reflectance")
        axs[1].set_title("Transmittance")
        axs[1].set_xlabel("Wavelength [nm]")
        
    return fwd_refl, fwd_trans, store[i].x, store[i].fun


def the_cost_function ( x, refl, trans ):
    """A standard cost function. Returns the cost for an input vector ``x``"""
    retval = prosail.prospect_5b ( *x )
    cost_refl = ( refl - retval[ :, 0])**2
    cost_trans = ( trans - retval[ :, 1])**2
    return np.sum ( cost_refl  + cost_trans )
