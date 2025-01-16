# Script to convert the Amber ion parameters to Gromacs

# atomic numbers for most of ions we will look at
at_num = {
    'Li' : 3,
    'Na' : 11,
    'K'  : 19,
    'Rb' : 37,
    'Cs' : 55,
    'Tl' : 81,
    'Cu' : 29,
    'Ag' : 47,
    'F'  : 9,
    'Cl' : 17,
    'Br' : 35,
    'I'  : 53,
    'Be' : 4,
    'Ni' : 28,
    'Pt' : 78,
    'Zn' : 30,
    'Co' : 27,
    'Pd' : 46,
    'Cr' : 24,
    'Fe' : 26,
    'Mg' : 12,
    'V'  : 23,
    'Mn' : 25,
    'Hg' : 80,
    'Cd' : 48,
    'Yb' : 70,
    'Ca' : 20,
    'Sn' : 50,
    'Pb' : 82,
    'Eu' : 64,
    'Sr' : 38,
    'Sm' : 62,
    'Ba' : 56,
    'Ra' : 88,
    'Al' : 13,
    'In' : 49,
    'Y'  : 39,
    'La' : 57,
    'Ce' : 58,
    'Pr' : 59,
    'Nd' : 60,
    'Gd' : 64,
    'Tb' : 65,
    'Dy' : 66,
    'Er' : 68,
    'Tm' : 69,
    'Lu' : 71,
    'Hf' : 72,
    'Zr' : 40,
    'U'  : 92,
    'Pu' : 94,
    'Th' : 90
}

# read file
f = open('frcmod.ionslm_hfe_opc3', 'r')
lines = f.readlines()
f.close()

out = open('top.ionslm_hfe_opc3', 'w')

# get sections
mass_idx = lines.index('MASS\n')
nonbon_idx = lines.index('NONBON\n')

# save information in atomtypes dictionary
atomtypes = {}

# read masses
for line in lines[mass_idx+1:nonbon_idx]:
    l = line.split()
    if len(l) > 0:
        at = ''.join(c for c in l[0] if c.isalpha())
        atomtypes[at] = {}
        atomtypes[at]['mass'] = float(l[1])
        atomtypes[at]['charge'] = ''.join(c for c in l[0] if not c.isalpha()).replace('+','')
        if atomtypes[at]['charge'] == '':
            atomtypes[at]['charge'] = '1'
        elif atomtypes[at]['charge'] == '-':
            atomtypes[at]['charge'] = '-1'

# read nonbonded parameters
for line in lines[nonbon_idx+1:]:
    l = line.split()
    if len(l) > 0:
        at = ''.join(c for c in l[0] if c.isalpha())
        atomtypes[at]['Rmin_ov_2'] = float(l[1])
        atomtypes[at]['epsilon'] = float(l[2])

# write in Gromacs format and convert
out.write('Ion parameters of all ions for OPC3 water model (converted from frcmod style) from Li/Merz groups\n')
for at in atomtypes:

    sig = atomtypes[at]['Rmin_ov_2'] * 2 * 2**(-1/6) / 10
    eps = atomtypes[at]['epsilon'] * 4.184
    new_line = f"{at:4s} {at:4s}\t{at_num[at]}\t{atomtypes[at]['mass']:.8f}\t{0:.8f}\tA\t{sig:.8e}\t{eps:.8e}\n"
    out.write(new_line)
    

# write the itp formatting as well
out.write('\n\n')
for at in atomtypes:

    out.write('[ moleculetype ]\n')
    out.write('; molname\tnrexcl\n')
    out.write(f'{at.upper()}\t\t1\n\n')

    out.write('[ atoms ]\n')
    out.write('; id    at type     res nr  residu name at name  cg nr  charge   mass\n')
    out.write(f"1\t{at}\t1\t{at.upper()}\t{at}\t1\t{atomtypes[at]['charge']}\t{atomtypes[at]['mass']}\n\n")