
# SOURCE: experiments/hras_nucleotide_equilibration.rst:16
from rasmodel.scenarios.default import model

# SOURCE: experiments/hras_nucleotide_equilibration.rst:20
import numpy as np
from matplotlib import pyplot as plt
from pysb.integrate import Solver
from pysb import *

# SOURCE: experiments/hras_nucleotide_equilibration.rst:28
ics_to_keep = ['GTP_0', 'GDP_0', 'HRAS_0']
for ic in model.initial_conditions:
    ic_param = ic[1]
    if ic_param.name not in ics_to_keep:
        ic_param.value = 0

# SOURCE: experiments/hras_nucleotide_equilibration.rst:36
t = np.logspace(-3, 3, 1000)

# SOURCE: experiments/hras_nucleotide_equilibration.rst:40
sol = Solver(model, t)
sol.run()

# SOURCE: experiments/hras_nucleotide_equilibration.rst:45
plt.figure()
HRAS_0 = model.parameters['HRAS_0'].value
for obs_name in ['HRAS_GTP_closed_', 'HRAS_GTP_open_',
                 'HRAS_GDP_closed_', 'HRAS_GDP_open_', 'HRAS_nf_']:
    plt.plot(t, sol.yobs[obs_name] / HRAS_0, label=obs_name)
ax = plt.gca()
ax.set_xscale('log')
plt.legend(loc='right')
plt.xlabel('Time (sec)')
plt.ylabel('Fraction of HRAS')
plt.savefig('simulation_1.png')

# SOURCE: experiments/hras_nucleotide_equilibration.rst:68
for ic in model.initial_conditions:
    ic_param = ic[1]
    ic_param.value = 0

# SOURCE: experiments/hras_nucleotide_equilibration.rst:74
HRAS = model.monomers['HRAS']
GTP = model.monomers['GTP']
HRAS_mGTP_0 = Parameter('HRAS_mGTP_0', 4e-6)
model.parameters['GTP_0'].value = 10e-6 # Unlabeled competitor
model.initial(HRAS(gtp=1, s1s2='open', gap=None, gef=None, p_loop=None,
                   CAAX=None, mutant='WT') % GTP(p=1, label='y'),
              HRAS_mGTP_0)

# SOURCE: experiments/hras_nucleotide_equilibration.rst:84
t = np.logspace(1, 6, 1000)
sol = Solver(model, t)
sol.run()

plt.figure()
plt.plot(t, sol.yobs['HRAS_mGTP_'] / HRAS_mGTP_0.value)
plt.ylim(0, 1.05)
ax = plt.gca()
ax.set_xscale('log')

# SOURCE: experiments/hras_nucleotide_equilibration.rst:96
for ic in model.initial_conditions:
    ic_param = ic[1]
    ic_param.value = 0

GDP = model.monomers['GDP']
HRAS_mGDP_0 = Parameter('HRAS_mGDP_0', 4e-6)
model.parameters['GDP_0'].value = 10e-6 # Unlabeled competitor
model.initial(HRAS(gtp=1, s1s2='open', gap=None, gef=None, p_loop=None,
                   CAAX=None, mutant='WT') % GDP(p=1, label='y'),
              HRAS_mGDP_0)

sol = Solver(model, t)
sol.run()

# SOURCE: experiments/hras_nucleotide_equilibration.rst:112
plt.plot(t, sol.yobs['HRAS_mGDP_'] / HRAS_mGDP_0.value)
plt.ylim(0, 1.05)
ax = plt.gca()
ax.set_xscale('log')

plt.savefig('simulation_2.png')

# SOURCE: experiments/kras_mapk.rst:12
Observable('RAS_GTP', RAS(gtp=1) % GTP(p=1))
Observable('RAS_RASGAP', RAS(gap=1) % RASA1(rasgap=1))
Observable('RAS_RAF', RAS(s1s2=1) % RAF(ras=1))
Observable('RAFd', RAF(raf=1) % RAF(raf=1))
Observable('MEKpp', MAP2K1(S218='p', S222='p'))

# SOURCE: experiments/kras_mapk.rst:23
from pysb.integrate import Solver
import numpy

ts = numpy.linspace(0, 1000, 100)
solver = Solver(model, ts)
solver.run()

# SOURCE: experiments/kras_mapk.rst:35
import matplotlib.pyplot as plt
for obs in model.observables:
    plt.plot(ts, solver.yobs[obs.name], label=obs.name)
plt.xlabel('Time (s)')
plt.ylabel('Concentration (nM)')
plt.legend()
plt.show()

# SOURCE: experiments/ras_gdp_binding.rst:16
from rasmodel.scenarios.default import model

# SOURCE: experiments/ras_gdp_binding.rst:20
import numpy as np
from matplotlib import pyplot as plt
from pysb.integrate import Solver
from pysb import *
from rasmodel import fitting

# Zero out all initial conditions
for ic in model.initial_conditions:
    ic[1].value = 0

# SOURCE: experiments/ras_gdp_binding.rst:37
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

# SOURCE: experiments/ras_gdp_binding.rst:60
t = np.linspace(0, 10, 1000)
sol = Solver(model, t)

# SOURCE: experiments/ras_gdp_binding.rst:67
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

# SOURCE: experiments/ras_gdp_binding.rst:95
plt.figure()
plt.plot(mgtp_concs, k_list, marker='o')
plt.ylim(bottom=0)

# SOURCE: experiments/ras_gdp_binding.rst:106
# A constant amount of labeled GDP
model.parameters['mGDP_0'].value = 2.5 * 1000 # nM
model.parameters['mGTP_0'].value = 0

# SOURCE: experiments/ras_gdp_binding.rst:112
model.parameters['bind_HRASopen_GDP_kf'].value = \
                    1e-2 # nM^-1 s^-1
model.parameters['bind_HRASopen_GDP_kr'].value = \
                    1e-2 / (5.7e4 * 1e-9) # s^-1
model.parameters['equilibrate_HRASopenGDP_to_HRASclosedGDP_kf'].value = \
                    3.2 #s^-1
model.parameters['equilibrate_HRASopenGDP_to_HRASclosedGDP_kr'].value = \
                    5e-7 #s^-1

# SOURCE: experiments/ras_gdp_binding.rst:124
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

# SOURCE: experiments/ras_gdp_binding.rst:145
plt.figure()
plt.plot(gdp_concs, k_list, marker='o')
plt.ylim(bottom=0)

# SOURCE: experiments/gxp_exchange.rst:9
from rasmodel.scenarios.default import model

import numpy as np
from matplotlib import pyplot as plt
from pysb.integrate import Solver
from pysb import *

from tbidbaxlipo.util import fitting

# Zero out all initial conditions
for ic in model.initial_conditions:
    ic[1].value = 0

KRAS = model.monomers['KRAS']
GDP = model.monomers['GDP']
GTP = model.monomers['GTP']

Expression('KRAS_mGXP_', model.observables['KRAS_mGTP_closed_'] +
                         model.observables['KRAS_mGDP_closed_'])

# Add an initial condition for HRAS with GDP or GTP pre-bound
# (Concentration units in nM)
Initial(KRAS(gtp=1, gap=None, gef=None, p_loop=None, s1s2='closed', CAAX=None,
             mutant='WT') % GDP(p=1, label='n'),
             Parameter('KRAS_WT_GDP_0', 0.))
Initial(KRAS(gtp=1, gap=None, gef=None, p_loop=None, s1s2='closed', CAAX=None,
             mutant='G13D') % GDP(p=1, label='n'),
             Parameter('KRAS_G13D_GDP_0', 0.))
Initial(KRAS(gtp=1, gap=None, gef=None, p_loop=None, s1s2='closed', CAAX=None,
             mutant='WT') % GTP(p=1, label='n'),
             Parameter('KRAS_WT_GTP_0', 0.))
Initial(KRAS(gtp=1, gap=None, gef=None, p_loop=None, s1s2='closed', CAAX=None,
             mutant='G13D') % GTP(p=1, label='n'),
             Parameter('KRAS_G13D_GTP_0', 0.))

plt.ion()

# SOURCE: experiments/gxp_exchange.rst:48
# WT, GDP:
model.parameters['mGDP_0'].value = 1500.
model.parameters['KRAS_WT_GDP_0'].value = 750.
t = np.linspace(0, 1000, 1000) # 1000 seconds
sol = Solver(model, t)
sol.run()
plt.figure()
plt.plot(t, sol.yexpr['KRAS_mGXP_'], label='WT')

# G13D, GDP:
model.parameters['KRAS_WT_GDP_0'].value = 0
model.parameters['KRAS_G13D_GDP_0'].value = 750.
sol.run()
plt.plot(t, sol.yexpr['KRAS_mGXP_'], label='G13D')
plt.legend(loc='lower right')
plt.title('GDP exchange')
plt.xlabel('Time (s)')
plt.ylabel('[Bound mGDP] (nM)')
plt.show()
plt.savefig('doc/_static/generated/gxp_exchange_1.png', dpi=150)

# SOURCE: experiments/gxp_exchange.rst:74
# WT, GTP
model.parameters['mGDP_0'].value = 0.
model.parameters['mGTP_0'].value = 1500.
model.parameters['KRAS_WT_GDP_0'].value = 0.
model.parameters['KRAS_G13D_GDP_0'].value = 0.
model.parameters['KRAS_WT_GTP_0'].value = 750.
model.parameters['KRAS_G13D_GTP_0'].value = 0.
sol.run()

plt.figure()
plt.plot(t, sol.yexpr['KRAS_mGXP_'], label='WT')

# G13D, GTP
model.parameters['KRAS_WT_GTP_0'].value = 0.
model.parameters['KRAS_G13D_GTP_0'].value = 750.
sol.run()
plt.plot(t, sol.yexpr['KRAS_mGXP_'], label='G13D')
plt.legend(loc='lower right')
plt.title('GTP exchange')
plt.xlabel('Time (s)')
plt.ylabel('[Bound mGTP] (nM)')
plt.show()

plt.savefig('doc/_static/generated/gxp_exchange_2.png', dpi=150)

# SOURCE: experiments/kras_gtp_hydrolysis.rst:9
from rasmodel.scenarios.default import model

import numpy as np
from matplotlib import pyplot as plt
from pysb.integrate import Solver
from pysb import *

from tbidbaxlipo.util import fitting


KRAS = model.monomers['KRAS']
GTP = model.monomers['GTP']

total_pi = 50000

for mutant in KRAS.site_states['mutant']:
    Initial(KRAS(gtp=1, gap=None, gef=None, p_loop=None, s1s2='open',
            CAAX=None, mutant=mutant) % GTP(p=1, label='n'),
            Parameter('KRAS_%s_GTP_0' % mutant, 0))

# SOURCE: experiments/kras_gtp_hydrolysis.rst:31
plt.figure()

t = np.linspace(0, 1000, 1000) # 1000 seconds

for mutant in KRAS.site_states['mutant']:
    # Zero out all initial conditions
    for ic in model.initial_conditions:
        ic[1].value = 0
    model.parameters['KRAS_%s_GTP_0' % mutant].value = total_pi

    sol = Solver(model, t)
    sol.run()
    plt.plot(t, sol.yobs['Pi_'] / total_pi, label=mutant)
    plt.ylabel('GTP hydrolyzed (%)')
    plt.ylim(top=1)
    plt.xlabel('Time (s)')
    plt.title('Intrinsic hydrolysis')

plt.legend(loc='upper left', fontsize=11, frameon=False)
plt.savefig('doc/_static/generated/kras_gtp_hydrolysis_1.png')

# SOURCE: experiments/kras_gtp_hydrolysis.rst:57
plt.figure()
for mutant in KRAS.site_states['mutant']:
    # Zero out all initial conditions
    for ic in model.initial_conditions:
        ic[1].value = 0
    model.parameters['RASA1_0'].value = 50000
    model.parameters['KRAS_%s_GTP_0' % mutant].value = total_pi

    sol = Solver(model, t)
    sol.run()
    plt.plot(t, sol.yobs['Pi_'] / total_pi, label=mutant)
    plt.ylabel('GTP hydrolyzed (%)')
    plt.ylim(top=1)
    plt.xlabel('Time (s)')
    plt.title('GAP-mediated hydrolysis')

plt.legend(loc='upper right', fontsize=11, frameon=False)
plt.savefig('doc/_static/generated/kras_gtp_hydrolysis_2.png')
