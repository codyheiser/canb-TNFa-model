from pysb.core import *
from pysb.bng import *
from pysb.integrate import *
from pysb.util import alias_model_components
from pysb.simulator import ScipyOdeSimulator
#from pysb.simulator.bng import BngSimulator
#from pysb.integrate import odesolve
import numpy as np
import matplotlib.pyplot as plt

Model()
# creates the 'model' object

Monomer('TNFa', ['tnfr1'])
Monomer('TNFR1', ['tnfa'])
Monomer('IKKK', ['state'], {'state': ['n', 'a']})
Monomer('IKK', ['state'], {'state': ['n', 'a', 'i', 'ii']})
Monomer('A20')
Monomer('IkBa', ['nfkb', 'phos', 'loc'], {'phos': ['u', 'p'], 'loc': ['n', 'c']})
Monomer('NFkB', ['ikba', 'dna', 'loc'], {'loc': ['n', 'c']})
Monomer('DNA', ['a20t', 'ikbat'])
Monomer('A20_mRNA')
Monomer('IkBa_mRNA')

Parameter('TNFa_init', 1) # eric set to 1
Parameter('TNFR1_init', 1000)
Parameter('IKKK_init', 10000)
Parameter('IKK_init', 200000)
Parameter('A20_init', 10)
Parameter('IkBa_init', 135000)
Parameter('NFkB_init', 100000)
Parameter('DNA_init', 2)
Parameter('A20_mRNA_init', 1)
Parameter('IkBa_mRNA_init', 1)
Initial(TNFa(tnfr1=None), TNFa_init)
Initial(TNFR1(tnfa=None), TNFR1_init)
Initial(IKKK(state='n'), IKKK_init)
Initial(IKK(state='n'), IKK_init)
Initial(A20, A20_init)
Initial(IkBa(nfkb=None, phos='u', loc='c'), IkBa_init)
Initial(NFkB(ikba=None, dna=None, loc='c'), NFkB_init)
Initial(DNA(a20t=None, ikbat=None), DNA_init)
Initial(A20_mRNA, A20_mRNA_init)
Initial(IkBa_mRNA, IkBa_mRNA_init)


Parameter('ka20', 10000)
Parameter('c5', 0.0005)
Parameter('ka', 0.0001)
Parameter('ki', 0.01)
Parameter('k1', 0.000005)
Parameter('k2', 10000)
Parameter('k3', 0.003)
Parameter('k4', 0.0005)
Parameter('a2', .0000001)
Parameter('tp', 0.01)
Parameter('c5a', 0.0001)
Parameter('a3', 0.0000005)
Parameter('c6a', 0.00002)
Parameter('e2a', 0.05)
Parameter('e1a', 0.005)
Parameter('i1a', 0.002)
Parameter('a1', 0.0000005)
Parameter('c4', 0.5)
Parameter('c3', 0.00075)
Parameter('i1', 0.01)
Parameter('c1', 0.1)
Parameter('kd', 0.0006)
Parameter('kb', 0.000004)
Parameter('q1', 0.00000015)
Parameter('q2', 0.000001)
Parameter('kv', 5)

Observable('Total_TNFa', TNFa(tnfr1=None) + TNFa(tnfr1=1) % TNFR1(tnfa=1))
Observable('Unbound_TNFa', TNFa(tnfr1=None))
Observable('Bound_TNFa', TNFa(tnfr1=1) % TNFR1(tnfa=1))
Observable('Neutral_IKK', IKK(state='n'))
Observable('Active_IKK', IKK(state='a'))
Observable('i_IKK', IKK(state='i'))
Observable('ii_IKK', IKK(state='ii'))
Observable('Neutral_IKKK', IKKK(state='n'))
Observable('Active_IKKK', IKKK(state='a'))
Observable('Total_A20', A20())
Observable('Nuclear_IkBa', IkBa(loc='n'))
Observable('Cytoplasmic_IkBa', IkBa(loc='c'))
Observable('Nuclear_NFkB', NFkB(loc='n'))
Observable('Cytoplasmic_NFkB', NFkB(loc='c'))

Expression('r_activate', kb * Unbound_TNFa)
Expression('IKKKa_Inactivation', ka * ka20 / (ka20 + Total_A20))
Expression('IKKa_Inactivation', k3 * ((k2 + Total_A20) / k2))
Expression('dna_binding', q1 * Nuclear_NFkB)
Expression('dna_dissociation', q2 * Nuclear_IkBa)
Expression('nuclear_a1', a1 * kv)

# TNF rules
Rule('TNFa_binds_TNFR1', TNFa(tnfr1=None) + TNFR1(tnfa=None) | TNFa(tnfr1=1) % TNFR1(tnfa=1), r_activate, kd)
Rule('TNFR1_activates_IKKK',  TNFa(tnfr1=1) % TNFR1(tnfa=1) + IKKK(state='n') >> IKKK(state='a') +  TNFa(tnfr1=1) % TNFR1(tnfa=1), IKKKa_Inactivation)

# IKKK rules
Rule('IKKKa_deactivates', IKKK(state='a') >> IKKK(state='n'), ki)
Rule('IKKKa_activates_IKK', IKKK(state='a') + IKK(state='n') >> IKKK(state='a') + IKK(state='a'), k1)

# IKK rules
Rule('IKKa_deactivates', IKK(state='a') >> IKK(state='n'), IKKa_Inactivation)
Rule('IKKi_to_IKKiiIKKn', IKK(state='i') >> IKK(state='n'), k4)
Rule('IKKa_phos_IkBa', IKK(state='a') + IkBa(nfkb=None, phos='u', loc='c') >> IKK(state='a') + IkBa(nfkb=None, phos='p', loc='c'), a2)

# Compartment changes
Rule('NFkB_nuclear_import', NFkB(ikba=None, dna=None, loc='c') >> NFkB(ikba=None, dna=None, loc='n'), i1)
Rule('IkBa_import_export', IkBa(nfkb=None, phos='u', loc='n') | IkBa(nfkb=None, phos='u', loc='c'), e1a, i1a)

# IkBa_NFKB rules
Rule('IkBaNFkB_formation', NFkB(ikba=None, dna=None, loc='n') + IkBa(nfkb=None, phos='u', loc='n') >>  IkBa(nfkb=2, phos='u', loc='n') % NFkB(ikba=2, dna=None, loc='n'), nuclear_a1)
Rule('IkBaNFkB_nuclear_export', IkBa(nfkb=2, phos='u', loc='n') % NFkB(ikba=2, dna=None, loc='n') >>  IkBa(nfkb=2, phos='u', loc='c') % NFkB(ikba=2, dna=None, loc='c'), e2a)
Rule('IkBaNFkB_degraded', IkBa(nfkb=2, phos='u', loc='c') % NFkB(ikba=2, dna=None, loc='c') + IKK(state='a') >> NFkB(ikba=None, dna=None, loc='c') + IkBa(nfkb=None, phos='p', loc='c') + IKK(state='a'), a3)
Rule('NFkB_Spontaneous', IkBa(nfkb=2, phos='u', loc='c') % NFkB(ikba=2, dna=None, loc='c') >>  NFkB(ikba=None, dna=None, loc='c') + IkBa(nfkb=None, phos='u', loc='c'), c6a)

# DNA binding rules
Rule('NFkB_DNA_complex_ikbat', NFkB(ikba=None, dna=None, loc='n') + DNA(ikbat=None) | NFkB(ikba=None, dna=5, loc='n') % DNA(ikbat=5), dna_binding, dna_dissociation)
Rule('NFkB_DNA_complex_a20t', NFkB(ikba=None, dna=None, loc='n') + DNA(a20t=None) | NFkB(ikba=None, dna=5, loc='n') % DNA(a20t=5), dna_binding, dna_dissociation)

# DNA transcription rules
Rule('NFkB_induces_A20trans', NFkB(ikba=None, dna=5, loc='n') % DNA(a20t=5) >> A20_mRNA(), c1)
Rule('NFkB_induces_IkBatrans', NFkB(ikba=None, dna=5, loc='n') % DNA(ikbat=5) >> IkBa_mRNA(), c1)

# A20 mRNA rules
Rule('A20_translation', A20_mRNA() >> A20(), c4)
Rule('A20_mRNA_degrad', A20_mRNA() >> None, c3)

# IKBa mRNA rules
Rule('IkBa_translation', IkBa_mRNA() >> IkBa(nfkb=None, phos='u', loc='c'), c4)
Rule('IkBa_mRNA_degrad', IkBa_mRNA() >> None, c3)

# Protein spontaneous degredation rules
Rule('IkBa_spont_degrad', IkBa(nfkb=None, phos='u', loc='c') >> None, c5a)
Rule('IkBa_degraded', IkBa(nfkb=None, phos='p', loc='c') >> None, tp)
Rule('A20_spont_degrad', A20() >> None, c5)


# Initial values ranges for various specieis
tnf_init_vals = [0.001, 0.01, 0.1, 1]
ikba_init_vals = [13500, 135000, 270000, 1350000]
a20_init_vals = [0, 1000, 10000, 1000000]
colors = ['r', 'm', 'g', 'b']

# Set up the time duration of the model (60 seconds * # minutes)
tmax = 60 * 90
# Sapmling frequency (in Hertz)
sampling_freq = 0.5

steps = tmax * sampling_freq
tspan = np.linspace(0, tmax, steps)

# Run model
sim = ScipyOdeSimulator(model, tspan=tspan)
sim_result = sim.run(initials= {TNFa(tnfr1=None) : tnf_init_vals})

# Debugging - prints out model species and ODEs
counter=0
species=[]
for val in model.species:
    print('__s' + str(counter), '=', val)
    species.append(val)
    counter+=1

counter=0
for ode in model.odes:
    print('__s' + str(counter), '=', ode)
    counter+=1


df = sim_result.dataframe

def normalize(values):
    """
    Function to generate a normalized version of the observable values in
    an iterable.
    """
    ymin = values.min(0)
    ymax = values.max(0)
    return (values - ymin) / (ymax - ymin)

plt.figure('CANB 8347 Project', figsize=(10,7))
for indx in range(0, 4):
    plt.plot(tspan/60, (df.loc[indx]['Nuclear_NFkB'].iloc[:]),
    c=colors[indx], lw=1.5, label=tnf_init_vals[indx])
    # plt.plot(tspan/60, (df.loc[indx]['Active_IKK'].iloc[:]),
    # c=colors[indx], lw=1.5, label=tnf_init_vals[indx], linestyle=':')

# plt.plot(tspan, normalize(sim_result.observables['Nuclear_NFkB']), lw=2, label='Nuclear NF-kB')
# plt.plot(tspan, normalize(sim_result.observables['Active_IKK']), lw=2, label='Active IKK')

#for obs in model.observables:
#    # plot all observables normalized to their maximum value
#    plt.plot(tspan, normalize(sim_result.observables[obs.name]), lw=2, label=obs.name)

plt.xlabel('Time (minutes)')
plt.ylabel('Normalized Cellular Amount')
plt.legend(loc='best', fontsize='small')
plt.show()
