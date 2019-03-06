from pysb import *
from pysb.simulator.bng import BngSimulator
from pysb.integrate import odesolve
import numpy as np
import matplotlib.pyplot as plt

Model() # creates the 'model' object

Monomer('TNFa', ['t'])
Monomer('TNFR1', ['t', 'tn'])
Monomer('IKKK', ['i', 'tn', 'state'], {'state':['n', 'a']})
Monomer('IKK', ['i','a','ik','in','state'], {'state':['n','a','i','ii']})
Monomer('A20', ['tn','i'])
Monomer('IkBa', ['n','i','state'],{'state':['n','c']})
Monomer('NFkB', ['n','d','state'],{'state':['n','c']})
Monomer('DNA', ['a','ikb'])
Monomer('A20_mRNA', ['n'])
Monomer('IkBa_mRNA', ['n'])

Parameter('TNFa_init', 10)
Parameter('TNFR1_init', 100)
Parameter('IKKK_init', 68)
Parameter('IKK_init', 10)
Parameter('A20_init', 100)
Parameter('IkBa_init', 68)
Parameter('NFkB_init', 68)
Parameter('DNA_init', 10)
Parameter('A20_mRNA_init', 0)
Parameter('IkBa_mRNA_init', 0)
Initial(TNFa(t=None),TNFa_init)
Initial(TNFR1(t=None, tn=None),TNFR1_init)
Initial(IKKK(i=None, tn=None, state='n'), IKKK_init)
Initial(IKK(i=None, a=None, ik=None, in=None, state='n'), IKK_init)
Initial(A20(tn=None, i=None), A20_init)
Initial(IkBa(n=1,i=None, state='c'),IkBa_init)
Initial(NFkB(n=1, d=None, state='c'), NFkB_init)
Initial(DNA(a=None, ikb=None), DNA_init)
Initial(A20_mRNA(n=None))
Initial(IkBa_mRNA(n=None))


Parameter('kf', 1)
Parameter('kr', 1000)
Parameter('kif', 15)
Parameter('kir', 100)
Rule('E_binds_S', E(b=None) + S(b=None) | E(b=1) % S(b=1), kf, kr)
Parameter('kcat', 100)
Rule('ES_to_P', E(b=1) % S(b=1) >> E(b=None) + P(), kcat)
Rule('E_binds_I', E(b=None) + I(b=None) | E(b=1) % I(b=1), kif, kir)

Observable('E_free', E(b=None))
Observable('S_free', S(b=None))
Observable('ES_complex', E(b=1)% S(b=1)) # E(b=1) % S(b=1)
Observable('P_total', P())
Observable('E_total', E()) # matches E(b=None) and E(b=1) % S(b=1)
Observable('I_total', I())
Observable('EI_complex', E(b=1) % S(b=1))
Observable('I_free', I())

print(model.rules)
print(model.parameters)
print(model.observables)

tspan = np.linspace(0, 10, 101)
print(tspan)
x = odesolve(model, tspan, verbose=True)

print(model.species)
for rxn in model.reactions:
    print(rxn)

plt.figure('My first PySB model')

for obs in model.observables:
    plt.plot(tspan, x[obs.name], lw=2, label=obs.name)

plt.legend(loc=0)
plt.show()
