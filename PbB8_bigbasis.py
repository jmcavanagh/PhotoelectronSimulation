import sys
import numpy as np
import nwchem_interface as nw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pyscf.data import nist
from pyscf import gto, scf, dft, tddft
from pyscf.geomopt.berny_solver import optimize


def anionic_dft(mol, xc, anion_mult):
    mol.charge = -1
    mol.spin = anion_mult - 1
    mf = dft.UKS(mol)
    mf.xc = xc
    energy = mf.kernel()
    return energy

def neutral_tddft(mol, xc, neutral_mult, n_states):
    mol.charge = 0
    mol.spin = neutral_mult - 1
#    if neutral_mult == 1:
#        mf = dft.RKS(mol)
#    else:
#        mf = dft.UKS(mol)
    mf = dft.UKS(mol)
    mf.xc = xc
    gs = mf.kernel()
    mytd = tddft.TDDFT(mf)
    mytd.singlet = False
    mytd.kernel(nstates = n_states)
    mytd.analyze()
    e_ev = np.asarray(mytd.e) * nist.HARTREE2EV
    
    return gs, e_ev, mytd


def build_mol(output_file, basis_set, xc, anion_mult):
    energy, coord_list = nw.readopt(output_file)
    print(energy)
    coord_string = '; '.join(' '.join([str(i) for i in c]) for c in coord_list)
    print(coord_string)
    mol = gto.Mole()
    mol.build(
        atom = coord_string,
        basis = {"B": gto.basis.parse('''
        B    S
      5.473000E+03           5.550000E-04           0.000000E+00          -1.120000E-04           0.000000E+00
      8.209000E+02           4.291000E-03           0.000000E+00          -8.680000E-04           0.000000E+00
      1.868000E+02           2.194900E-02           0.000000E+00          -4.484000E-03           0.000000E+00
      5.283000E+01           8.444100E-02           0.000000E+00          -1.768300E-02           0.000000E+00
      1.708000E+01           2.385570E-01           0.000000E+00          -5.363900E-02           0.000000E+00
      5.999000E+00           4.350720E-01           0.000000E+00          -1.190050E-01           0.000000E+00
      2.208000E+00           3.419550E-01           0.000000E+00          -1.658240E-01           0.000000E+00
      5.879000E-01           3.685600E-02           1.000000E+00           1.201070E-01           0.000000E+00
      2.415000E-01          -9.545000E-03           0.000000E+00           5.959810E-01           0.000000E+00
      8.610000E-02           2.368000E-03           0.000000E+00           4.110210E-01           1.000000E+00
B    S
      0.0291400              1.0000000
B    P
      1.205000E+01           0.000000E+00           1.311800E-02           0.000000E+00
      2.613000E+00           0.000000E+00           7.989600E-02           0.000000E+00
      7.475000E-01           0.000000E+00           2.772750E-01           0.000000E+00
      2.385000E-01           1.000000E+00           5.042700E-01           0.000000E+00
      7.698000E-02           0.000000E+00           3.536800E-01           1.000000E+00
B    P
      0.0209600              1.0000000
B    D
      6.610000E-01           1.000000E+00           0.000000E+00
      1.990000E-01           0.000000E+00           1.000000E+00
B    D
      0.0604000              1.0000000
B    F
      4.900000E-01           1.0000000
B    F
      0.1630000              1.0000000
        
        '''), "Pb": gto.basis.parse('''
        Pb    S
    544.6750000              0.0003190              0.00000000             0.00000000            -0.0001460              0.00000000
     36.5128000              0.0242140              0.00000000             0.00000000            -0.0083480              0.00000000
     22.7761000             -0.1854660              0.00000000             0.00000000             0.0694480              0.00000000
     14.2262000              0.5460900              0.00000000             0.00000000            -0.2207290              0.00000000
      6.8950000             -0.8705150              0.00000000             0.00000000             0.4003450              0.00000000
      4.3096900             -0.2035780              0.00000000             0.00000000             0.0629640              0.00000000
      1.8008500              0.9089240              0.00000000             0.00000000            -0.5315590              0.00000000
      0.8907680              0.5320000              0.00000000             0.00000000            -0.4479460              0.00000000
      0.3189680              0.0322840              1.0000000              0.00000000             0.3627060              0.00000000
      0.1483520             -0.0052680              0.00000000             1.0000000              0.6966730              0.00000000
      0.0681000              0.00000000             0.00000000             0.00000000             0.00000000             1.0000000
      0.0632880              0.0010520              0.00000000             0.00000000             0.2440990              0.00000000
Pb    S
      0.0251000              1.0000000
Pb    P
     18.6489000             -0.0178000              0.00000000             0.0030930              0.00000000
     11.6679000              0.1793570              0.00000000            -0.0453160              0.00000000
      7.2934900             -0.4167250              0.00000000             0.1173920              0.00000000
      2.0284900              0.5595180              0.00000000            -0.2007620              0.00000000
      1.0409700              0.4925180              0.00000000            -0.1848820              0.00000000
      0.5141900              0.1323700              0.00000000            -0.0054120              0.00000000
      0.3696000              0.00000000             1.0000000              0.00000000             0.00000000
      0.2286510              0.0080590              0.00000000             0.3698780              0.00000000
      0.0958280              0.0007990              0.00000000             0.5468910              0.00000000
      0.0586000              0.00000000             0.00000000             0.00000000             1.0000000
      0.0392290             -0.0000270              0.00000000             0.2294930              0.00000000
Pb    P
      0.0208000              1.0000000
Pb    D
     61.3970000              0.0003250              0.00000000             0.00000000
     12.3735000              0.0132110              0.00000000             0.00000000
      6.9268600             -0.0727810              0.00000000             0.00000000
      2.3308700              0.2697210              0.00000000             0.00000000
      1.2101900              0.4265860              0.00000000             0.00000000
      0.6005160              0.3381390              0.00000000             0.00000000
      0.2807870              0.1358660              1.0000000              0.00000000
      0.1093000              0.0142810              0.00000000             1.0000000
Pb    D
      0.0404000              1.0000000
Pb    F
      0.2900000              1.0000000
Pb    F
      0.1116000              1.0000000
        
        ''')},
        ecp = {"Pb": gto.basis.parse('''
        Pb nelec 60
Pb ul
2       1.0000000              0.0000000
Pb S
2      12.2963030            281.2854990
2       8.6326340             62.5202170
Pb P
2      10.2417900             72.2768970
2       8.9241760            144.5910830
2       6.5813420              4.7586930
2       6.2554030              9.9406210
Pb D
2       7.7543360             35.8485070
2       7.7202810             53.7243420
2       4.9702640             10.1152560
2       4.5637890             14.8337310
Pb F
2       3.8875120             12.2098920
2       3.8119630             16.1902910
Pb G
2       5.6915770             -9.0966650
2       5.7155670            -11.5319960
        ''')},
        symmetry = True,
        charge = -1,
        spin = anion_mult - 1,
        verbose = 4,
    )
    return mol
    

nwoutput = sys.argv[1]
basis_set = sys.argv[2]
#ecp = basis_set
xc = sys.argv[3]
anion_mult = int(sys.argv[4])
neutral_mult = int(sys.argv[5])
n_states = int(sys.argv[6])
mol = build_mol(nwoutput, basis_set, xc, anion_mult)
ags = anionic_dft(mol, xc, anion_mult)
ngs, e_ev, mytd = neutral_tddft(mol, xc, neutral_mult, n_states)

vde1 = (ngs - ags)*nist.HARTREE2EV
e_ev = np.append(e_ev,0)
e_ev += vde1

x = np.arange(1, 6, 0.01)
y = np.zeros_like(x)
#for ex in e_ev:
#    y += np.exp(-100*(x-ex)**2)
#plt.plot(x,y, color= 'b')

y_single = np.exp(-100*(x-e_ev[-1])**2)
xy = mytd.xy


with open(nwoutput[0:-4]+'.txt', 'w+') as log:
    log.write('\n-----------------------------------')
    log.write('\n1st VDE = ' + str(e_ev[-1]) + ' eV')
    log.write('\n-----------------------------------')
    for k in range(0, len(xy)):
        if neutral_mult == 1:
            exc_mat = abs(np.array(xy[k][0], dtype=object))
        else:
            exc_mat = abs(np.array(xy[k][0][0], dtype=object)) 
            #so far this may only work for doublets, I'm unsure about the rest
        max_trans = np.max(exc_mat)
        orbital_transition = np.where(exc_mat == max_trans)
        orbital_transition = [int(h) + 1 for h in orbital_transition]
        log.write('\n-------------------------------\n')
        log.write(str(orbital_transition) + ' ' + str(max_trans))
        log.write('\nEnergy = ' + str(e_ev[k]) + ' eV')
        if orbital_transition[-1] == 1:
            y_single += np.exp(-500*(x-e_ev[k])**2)
        else:
            log.write('\nMULTIELECTRON EXCITATION ' + str(k))
    for k in range(0, len(x)):
        log.write('\n' + str(x[k]) + ' ' + str(y_single[k]))

plt.plot(x, y_single, color='r')
plt.savefig(nwoutput[0:-4]+'.png')
