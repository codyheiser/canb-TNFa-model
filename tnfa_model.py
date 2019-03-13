# -*- coding: utf-8 -*-
"""
tnfa_model.py
@authors: K Bergdorf, C Heiser, VE Kerschberger

PySB model of TNFa-mediated NF-k B signaling from:
	Lipniacki, T., Puszynski, K., Paszek, P., Brasier, A. R. & Kimmel, M. Single TNFα trimers mediating 
	NF-κ B activation: stochastic robustness of NF-κ B signaling. BMC Bioinformatics 8, 376 (2007).

usage: tnfa_model.py [-h] TNFa_init [TNFa_init ...]

PySB model of Lipniacki, et al (2007)

positional arguments:
  TNFa_init   Initial amount(s) of TNFa to model

optional arguments:
  -h, --help  show this help message and exit
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pysb.core import *
from pysb.bng import *
from pysb.integrate import *
from pysb.util import alias_model_components
from pysb.simulator import ScipyOdeSimulator


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

Parameter('TNFa_init', 1) # will be overwritten by list (?)
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

# Define rate constants
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
Parameter('kf', 0.0006)
Parameter('kb', 0.000004)
Parameter('q1', 0.00000015)
Parameter('q2', 0.000001)
Parameter('kv', 5)

# Define observables
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

# Define expressions for dynamic rate constants
#Expression('r_activate', kb * Unbound_TNFa) # remove dependence on free TNFa...
Expression('IKKKa_activation', ka * ka20 / (ka20 + Total_A20))
Expression('IKKa_Inactivation', k3 * ((k2 + Total_A20) / k2))
Expression('dna_binding', q1 * Nuclear_NFkB)
Expression('dna_dissociation', q2 * Nuclear_IkBa)
Expression('nuclear_a1', a1 * kv)

# TNF rules
Rule('TNFa_binds_TNFR1', TNFa(tnfr1=None) + TNFR1(tnfa=None) | TNFa(tnfr1=1) % TNFR1(tnfa=1), kb, kf) # remove dependence on free TNFa...

# IKKK rules
Rule('TNFR1_activates_IKKK',  TNFa(tnfr1=1) % TNFR1(tnfa=1) + IKKK(state='n') >> IKKK(state='a') + TNFa(tnfr1=1) % TNFR1(tnfa=1), IKKKa_activation)
Rule('IKKKa_deactivates', IKKK(state='a') >> IKKK(state='n'), ki)
Rule('IKKKa_activates_IKK', IKKK(state='a') + IKK(state='n') >> IKKK(state='a') + IKK(state='a'), k1)

# IKK rules
Rule('IKKa_deactivates', IKK(state='a') >> IKK(state='i'), IKKa_Inactivation)
# Rule('IKKa_bind_A20', IKK(state='a') + A20() >> IKK(state='a') % A20(), k2)
# Rule('IKKa_to_IKKi', IKK(state='a') % A20() >> IKK(state='i') + A20(), k2)
Rule('IKKi_to_IKKn', IKK(state='i') >> IKK(state='n'), k4)
Rule('IKKa_phos_IkBa', IKK(state='a') + IkBa(nfkb=None, phos='u', loc='c') >> IKK(state='a') + IkBa(nfkb=None, phos='p', loc='c'), a2)

# Compartment changes
Rule('NFkB_nuclear_import', NFkB(ikba=None, dna=None, loc='c') >> NFkB(ikba=None, dna=None, loc='n'), i1)
Rule('IkBa_import_export', IkBa(nfkb=None, phos='u', loc='n') | IkBa(nfkb=None, phos='u', loc='c'), e1a, i1a)

# IkBa_NFKB rules
Rule('IkBaNFkB_formation_c', NFkB(ikba=None, dna=None, loc='c') + IkBa(nfkb=None, phos='u', loc='c') >>  IkBa(nfkb=2, phos='u', loc='c') % NFkB(ikba=2, dna=None, loc='c'), a1) 
Rule('IkBaNFkB_formation_n', NFkB(ikba=None, dna=None, loc='n') + IkBa(nfkb=None, phos='u', loc='n') >>  IkBa(nfkb=2, phos='u', loc='n') % NFkB(ikba=2, dna=None, loc='n'), nuclear_a1)
Rule('IkBaNFkB_nuclear_export', IkBa(nfkb=2, phos='u', loc='n') % NFkB(ikba=2, dna=None, loc='n') >>  IkBa(nfkb=2, phos='u', loc='c') % NFkB(ikba=2, dna=None, loc='c'), e2a)
Rule('IkBaNFkB_phosphorylated', IkBa(nfkb=2, phos='u', loc='c') % NFkB(ikba=2, dna=None, loc='c') + IKK(state='a') >> NFkB(ikba=2, dna=None, loc='c') % IkBa(nfkb=2, phos='p', loc='c') + IKK(state='a'), a3)
Rule('IkBaNFkB_Spontaneous', IkBa(nfkb=2, phos='u', loc='c') % NFkB(ikba=2, dna=None, loc='c') >> NFkB(ikba=None, dna=None, loc='c'), c6a)

# DNA binding rules
Rule('NFkB_DNA_complex_ikbat', NFkB(ikba=None, dna=None, loc='n') + DNA(ikbat=None) | NFkB(ikba=None, dna=5, loc='n') % DNA(ikbat=5), q1, q2)
Rule('NFkB_DNA_complex_a20t', NFkB(ikba=None, dna=None, loc='n') + DNA(a20t=None) | NFkB(ikba=None, dna=5, loc='n') % DNA(a20t=5), q1, q2)

# DNA transcription rules
Rule('NFkB_induces_A20trans', NFkB(ikba=None, dna=5, loc='n') % DNA(a20t=5) >> NFkB(ikba=None, dna=5, loc='n') % DNA(a20t=5) + A20_mRNA(), c1) # conserve DNA
Rule('NFkB_induces_IkBatrans', NFkB(ikba=None, dna=5, loc='n') % DNA(ikbat=5) >> NFkB(ikba=None, dna=5, loc='n') % DNA(ikbat=5) + IkBa_mRNA(), c1) # conserve DNA

# A20 mRNA rules
Rule('A20_translation', A20_mRNA() >> A20_mRNA() + A20(), c4) # conserve mRNA
Rule('A20_mRNA_degrad', A20_mRNA() >> None, c3)

# IKBa mRNA rules
Rule('IkBa_translation', IkBa_mRNA() >> IkBa_mRNA() + IkBa(nfkb=None, phos='u', loc='c'), c4) # conserve mRNA
Rule('IkBa_mRNA_degrad', IkBa_mRNA() >> None, c3)
#Rule('A20_inh_signal', A20() + TNFR1(tnfa=1) % TNFa(tnfr1=1) + IKKK(state='n') >> None, ka20) # this is the one that killed us!!

# Protein spontaneous degredation rules
Rule('IkBa_spont_degrad', IkBa(nfkb=None, phos='u', loc='c') >> None, c5a)
Rule('IkBa_degraded', IkBa(nfkb=None, phos='p', loc='c') >> None, tp)
Rule('IkBaNFkB_degraded', NFkB(ikba=2, dna=None, loc='c') % IkBa(nfkb=2, phos='p', loc='c') >> NFkB(ikba=None, dna=None, loc='c'), tp)
Rule('A20_spont_degrad', A20() >> None, c5)

#print(model.rules)
#print(model.parameters)
#print(model.observables)

# Debugging - prints out model species and ODEs
#counter=0
#species=[]
#for val in model.species:
#	print('__s' + str(counter), '=', val)
#	species.append(val)
#	counter+=1

#counter=0
#for ode in model.odes:
#	print('__s' + str(counter), '=', ode)
#	counter+=1

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='PySB model of Lipniacki, et al (2007)')
	parser.add_argument('TNFa_init', help='Initial amount(s) of TNFa to model', type=float, nargs='+')
	args = parser.parse_args()

	# Run simulation(s)
	tspan = np.linspace(0, 5400, 100) # from 0 to 90 minutes, in seconds
	#ikba_init_vals = [13500, 135000, 270000, 1350000]
	#a20_init_vals = [0, 1000, 10000, 1000000]
	colors = ['r', 'm', 'g', 'b']

	sim = ScipyOdeSimulator(model, tspan=tspan)
	sim_result = sim.run(initials = {TNFa(tnfr1=None): args.TNFa_init})
	df = sim_result.dataframe

	# Plot results
	plt.figure('CANB 8347 Project', figsize=(8,4))

	plt.subplot(121)
	for n in range(0,4):
		plt.plot(tspan/60, df.loc[n]['Active_IKK'].iloc[:], lw=2, label='Active IKK')

	plt.title('Active IKK')
	plt.xlabel('Time (min)')
	plt.ylabel('Activity')
	plt.legend(['{} ng/ml '.format(args.TNFa_init[0]),'{} ng/ml'.format(args.TNFa_init[1]),'{} ng/ml'.format(args.TNFa_init[2]),'{} ng/ml'.format(args.TNFa_init[3])], 
		title = '[TNFa]', loc=0, fontsize = 8)

	plt.subplot(122)
	for n in range(0,4):
		plt.plot(tspan/60, df.loc[n]['Nuclear_NFkB'].iloc[:], lw=2, label='Nuclear NFkB')

	plt.title('Nuclear NFkB')
	plt.xlabel('Time (min)')
	plt.ylabel('Activity')
	plt.legend(['{} ng/ml '.format(args.TNFa_init[0]),'{} ng/ml'.format(args.TNFa_init[1]),'{} ng/ml'.format(args.TNFa_init[2]),'{} ng/ml'.format(args.TNFa_init[3])], 
		title = '[TNFa]', loc=0, fontsize = 8)

	plt.tight_layout()
	plt.show()
