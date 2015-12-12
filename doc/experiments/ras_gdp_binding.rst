.. component::
   :module: rasmodel.experiments.ras_gdp_binding

Ras binding to GTP and GDP
==========================

In [PMID2200519]_, the authors prepare nucleotide-free HRAS and incubate with
fluorescent-labeled nucleotides to determine the association kinetics of Ras
with nucleotides. They find that the kinetics indicate a two-step association
mechanism, with the nucleotide first loosely and then tightly bound. A subset
of these experiments are reproduced below.

We use the default model scenario, which includes only Ras, RasGAPs, and
RasGEFs::

    from rasmodel.scenarios.default import model

Preliminaries::

    import numpy as np
    from matplotlib import pyplot as plt
    from pysb.integrate import Solver
    from pysb import *
    from rasmodel import fitting

    # Zero out all initial conditions
    for ic in model.initial_conditions:
        ic[1].value = 0

HRAS-GTP association
--------------------

In Figure 2 of [PMID2200519]_, a titration of mGTP is performed, with the
pseudo-first-order rate constant for Ras-mGTP binding calculated at each
concentration by fitting to a single exponential::

    # In this first experiment, 0.5 uM (500 nM) of HRAS is used:
    model.parameters['HRAS_0'].value = 500.

    # We create an Expression including both HRAS-mGTP and HRAS-mGDP because
    # we will get a fluorescence signal even if GTP is hydrolyzed to GDP:
    Expression('HRAS_mGXP_', model.observables['HRAS_mGTP_closed_'] +
                       model.observables['HRAS_mGDP_closed_'])

    # We use the parameters calculated for experiments with mGTP at 5C
    model.parameters['bind_HRASopen_GTP_kf'].value = 1e-2 # nM^-1 s^-1
    model.parameters['bind_HRASopen_GTP_kr'].value = \
                                    1e-2 / (6.1e4 * 1e-9) # s^-1
    model.parameters['equilibrate_HRASopenGTP_to_HRASclosedGTP_kf'].value = \
                                    4.5 #s^-1

    # Assume that the isomerization reaction is irreversible, as appears
    # to be their assumption for this experiment:
    model.parameters['equilibrate_HRASopenGTP_to_HRASclosedGTP_kr'].value = \
                                    0 #s^-1

The timescales of the experiments are fast and are measured by stopped-flow.
We use a timespan of 10 seconds::

    t = np.linspace(0, 10, 1000)
    sol = Solver(model, t)

We perform the titration by running the association reaction at different mGTP
concentrations (ranging up to about 15 uM) and then measuring the apparent
association rate by fitting to a single exponential equation::

    plt.figure()
    k_list = []

    # Perform titration
    mgtp_concs = np.arange(1, 15) * 1000 # nM (1 - 15 uM)
    for mgtp_conc in mgtp_concs:

        # Titration of labeled GTP:
        model.parameters['mGTP_0'].value = mgtp_conc
        sol.run()

        # Fit to an exponential function to extract the pseudo-first-order rates
        k = fitting.Parameter(1.)
        def expfunc(t):
            # The maximum of the signal will be with all HRAS bound to GTP/GDP
            max_mGXP = model.parameters['HRAS_0'].value
            return max_mGXP * (1 - np.exp(-k()*t))
        res = fitting.fit(expfunc, [k], sol.yexpr['HRAS_mGXP_'], t)

        # Plot data and fits
        plt.plot(t, sol.yexpr['HRAS_mGXP_'], color='b')
        plt.plot(t, expfunc(t), color='r')

        # Keep the fitted rate
        k_list.append(k())

Now we plot the scaling of the rates with mGTP concentration::

    plt.figure()
    plt.plot(mgtp_concs, k_list, marker='o')
    plt.ylim(bottom=0)

HRAS-GDP association measured by a competition assay
----------------------------------------------------

In Figure 3 of [PMID2200519]_, association kinetics for mGDP are measured by
incubating HRAS and a constant amount (2.5 uM) of mGDP with varying
concentrations of unlabeled GDP competitor::

    # A constant amount of labeled GDP
    model.parameters['mGDP_0'].value = 2.5 * 1000 # nM
    model.parameters['mGTP_0'].value = 0

Use the parameters given in the table for this experiment, using a diffusion-limited on rate of :math:`10^7 M^{-1} s^{-1}`::

    model.parameters['bind_HRASopen_GDP_kf'].value = \
                        1e-2 # nM^-1 s^-1
    model.parameters['bind_HRASopen_GDP_kr'].value = \
                        1e-2 / (5.7e4 * 1e-9) # s^-1
    model.parameters['equilibrate_HRASopenGDP_to_HRASclosedGDP_kf'].value = \
                        3.2 #s^-1
    model.parameters['equilibrate_HRASopenGDP_to_HRASclosedGDP_kr'].value = \
                        5e-7 #s^-1

Titrate the unlabeled competitor and calculate the fits by fitting to a
two-parameter single-exponential function::

    k_list = []
    plt.figure()
    gdp_concs = np.arange(0, 22) * 1000 # nM
    for gdp_conc in gdp_concs:
        # Titration of unlabeled GDP
        model.parameters['GDP_0'].value = gdp_conc
        sol.run()

        k = fitting.Parameter(1.)
        A = fitting.Parameter(100.)
        def expfunc(t):
            return A() * (1 - np.exp(-k()*t))
        res = fitting.fit(expfunc, [A, k], sol.yexpr['HRAS_mGXP_'], t)

        plt.plot(t, sol.yexpr['HRAS_mGXP_'], color='b')
        plt.plot(t, expfunc(t), color='r')

        k_list.append(k())

Plot the scaling of the pseudo-first-order rates::

    plt.figure()
    plt.plot(gdp_concs, k_list, marker='o')
    plt.ylim(bottom=0)

