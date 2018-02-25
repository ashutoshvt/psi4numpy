import psi4

mol = {}

mol["h2_4"] = psi4.geometry("""
H
H 1 R
H 1 D 2 P
H 3 R 1 P 2 T
H 3 D 4 P 1 X
H 5 R 3 P 4 T
H 5 D 6 P 3 X
H 7 R 5 P 6 T

R = 0.75
D = 1.5
P = 90.0
T = 60.0
X = 180.0
symmetry c1
""")

psi4.set_memory(int(5e9), False)
psi4.core.set_output_file('output.dat', False)

psi4.set_options({'basis': 'aug-cc-pvdz',
                  'guess': 'sad',
                  'scf_type': 'pk',
                  'e_convergence': 1e-9,
                  'd_convergence': 1e-9,
                  'r_convergence': 1e-9,
                  'gauge': 'length',
                  'omega': [589, 'nm']})
psi4.core.set_active_molecule(mol["h2_4"])
psi4.properties('ccsd', properties=['rotation'])

