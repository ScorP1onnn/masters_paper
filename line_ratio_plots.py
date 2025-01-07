import numpy as np
from uncertainties import ufloat
import matplotlib.pyplot as plt
from lmfit import Model
from math import *
from scipy import constants as const
from matplotlib.patches import Rectangle
import utils

print(np.log10(10.6e7),utils.log10_of_error(ufloat(10.6e7,2.4e7)))

print(np.log10(2.29e13),utils.log10_of_error(ufloat(2.29e13,0.36e13)))
#exit()
#######################################################################################################################
nll_smg = np.asarray([ufloat(7.5e7, 2e7),
                                       ufloat(7.0e8, 0.7e8) / ufloat(14.7, 0.3),
                                       ufloat(2.14e8, 0.23e8),
                                       ufloat(2.0e8, 0.9e8),
                                       ufloat(1.06e9, 0.15e9) / ufloat(5.7, 0.5),
                                       ufloat(6.5e8, 0.9e8) / 20,
                                       ufloat(6.1e8, 0.5e8),
                                        ])

nll_qso = np.asarray([ufloat(3.97e8, 0.3e8),
                                       ufloat(4.4e8, 0.4e8),

                                       ufloat(8.9e8, 2.4e8) / ufloat(80, 20)])


ir_smg = np.asarray([ufloat(1.1e12, 0.4e12),
    ufloat(1.1e13, 0.2e13),
    ufloat(1.9e13, 0.3e13),
    ufloat(2.8e13, 0),
    ufloat(6.4e13, 1.0e13) / ufloat(5.7, 0.5),
    ufloat(10.2e12, 1.0e12) * (1 - 0.18),
    ufloat(6.7e13, 0)])

ir_qso = np.asarray([ufloat(5e13, 0),
    ufloat(49.6e12, 3.8e12) * (1 - 0.29),
    ufloat(16e12, 0.4e12) * (1 - 0.99)])

ir = np.asarray([
    ufloat(1.1e12, 0.4e12),
    ufloat(1.1e13, 0.2e13),
    ufloat(1.9e13, 0.3e13),
    ufloat(2.8e13, 0),
    ufloat(6.4e13, 1.0e13) / ufloat(5.7, 0.5),
    ufloat(10.2e12, 1.0e12) * (1 - 0.18),
    ufloat(6.7e13, 0),

    ufloat(5e13, 0),
    ufloat(49.6e12, 3.8e12) * (1 - 0.29),
    ufloat(16e12, 0.4e12) * (1 - 0.99)
])
#######################################################################################################################
#order is GN20,HDF850,ID141
#Ok so values in the table are not corrected for AGN fraction





nll_pdbi_smg = np.asarray([ufloat(9.2,2.8),ufloat(1.3,0),ufloat(6.9,1.6)]) * 1e8  #ID141 my value: ufloat(6.9,1.6)]

#ir_pdbi_smg = np.asarray([10**ufloat(13.15,0.04),(ufloat(6.5e12,1e12)/1.7)/0.75,ufloat(9.9e13, 2.3e13) / ufloat(5.8, 0)])
#The reason why HDF850.1 has a L_FIR of 6.5e12 is because I multiplied 3.8e12 (in the Neri et al. 2014 paper) by 1.7 (instead of 1.6)

#Using TIR from Sun et al. 2024 for HDF850.1 (gmf = 2.5)
ir_pdbi_smg = np.asarray([10**ufloat(13.15,0.04),(ufloat(5e12,2e12)),ufloat(9.9e13, 2.3e13) / ufloat(5.8, 0)])
print((ir_pdbi_smg/1e12))   #These values and the one in the paper match up, Great!
print("")
print((nll_pdbi_smg/ir_pdbi_smg)/1e-5) #nll/IR values are also consistent with table
print('')

#NOTE: The inclusion of the IR value from Sun et al. 2024 does not change anything in the graphs since the original value is 5.1e12 and Sun et al. 2024 value is 5e12 :)

nll_pdbi_qso = np.asarray([ufloat(1.3e8, 0.6e8)])
ir_pdbi_qso = np.asarray([ufloat(17.8e12, 1.7e12) * (1 - 0.88)])
print(ir_pdbi_qso/1e12)
print((nll_pdbi_qso/ir_pdbi_qso)/1e-5) #This is also fine (although it is 6.08 or 6.1 where I wronte 6.2 in paper :-) (no clue why)




#order is J2054, J2310

nll_new_qso = np.asarray([2.19e8,3.47e8])
agn_frac = [0.59, 0.70]
ir_new_qso  = np.asarray([ufloat(13.4e12,1.7e12) * (1-agn_frac[0]),ufloat(84.4e12,5.7e12)* (1-agn_frac[1])])
print((nll_new_qso/ir_new_qso) / 1e-5) # This is also fine (1.37 rounded to 1.4)



#############
nll_ir_pdbi_smg = nll_pdbi_smg/ir_pdbi_smg
nll_ir_pdbi_qso = nll_pdbi_qso/ir_pdbi_qso
nll_ir_new_qso = nll_new_qso/ir_new_qso
#############


nll_ir_smg = nll_smg/ir_smg
nll_ir_qso = nll_qso/ir_qso

nll_ir_pdbi_smg_nominal =[]
nll_ir_pdbi_qso_nominal =[]

nll_ir_smg_nominal =[]
nll_ir_smg_err =[]
nll_ir_qso_nominal =[]
nll_ir_qso_err =[]

ir_pdbi_smg_nominal =[]
ir_pdbi_qso_nominal =[]
ir_smg_nominal =[]
ir_qso_nominal =[]

for i in range(len(nll_ir_pdbi_smg)):
    nll_ir_pdbi_smg_nominal.append(nll_ir_pdbi_smg[i].n)
    ir_pdbi_smg_nominal.append(ir_pdbi_smg[i].n)

for i in range(len(nll_ir_pdbi_qso)):
    nll_ir_pdbi_qso_nominal.append(nll_ir_pdbi_qso[i].n)
    ir_pdbi_qso_nominal.append(ir_pdbi_qso[i].n)

for i in range(len(nll_ir_smg)):
    nll_ir_smg_nominal.append(nll_ir_smg[i].n)
    nll_ir_smg_err.append(nll_ir_smg[i].s)
    ir_smg_nominal.append(ir_smg[i].n)

for i in range(len(nll_ir_qso)):
    nll_ir_qso_nominal.append(nll_ir_qso[i].n)
    nll_ir_qso_err.append(nll_ir_qso[i].s)
    ir_qso_nominal.append(ir_qso[i].n)


nll_ir_pdbi_smg_nominal = np.asarray(nll_ir_pdbi_smg_nominal)
nll_ir_pdbi_qso_nominal = np.asarray(nll_ir_pdbi_qso_nominal)
nll_ir_smg_nominal = np.asarray(nll_ir_smg_nominal)
nll_ir_qso_nominal = np.asarray(nll_ir_qso_nominal)
nll_ir_smg_err= np.asarray(nll_ir_smg_err)
nll_ir_qso_err= np.asarray(nll_ir_qso_err)

ir_pdbi_smg_nominal = np.asarray(ir_pdbi_smg_nominal)
ir_pdbi_qso_nominal = np.asarray(ir_pdbi_qso_nominal)
ir_smg_nominal = np.asarray(ir_smg_nominal)
ir_qso_nominal = np.asarray(ir_qso_nominal)

"""***************************including local galaxies**************************************************"""


def luminosity_distance(x):
    z = x
    W_R = 0.  # Omega Radiation
    W_K = 0.  # Omega curvature
    c = 299792.458  # speed of light in km/s
    H_0 = 70  # Hubbles constant
    W_M = 0.3  # Omega matter
    W_V = 1 - W_M  # Omega vacuum
    Tyr = 977.8  # coefficent for converting 1/H into Gyr

    h = H_0 / 100
    W_R = 4.165e-5 / (h * h)
    W_K = 1 - W_M - W_R - W_V
    a_z = 1.0 / (1 + 1.0 * z)
    age = 0
    n = 1000

    for i in range(n):
        a = a_z * (i + 0.5) / n
        a_dot = np.sqrt(W_K + (W_M / a) + (W_R / (a * a)) + (W_V * a * a))
        age = age + 1. / a_dot
    z_age = a_z * age / n
    z_age_Gyr = (Tyr / H_0) * z_age

    DTT = 0.0
    DCMR = 0.0

    # do integral over a=1/(1+z) from az to 1 in n steps, midpoint rule
    for i in range(n):
        a = a_z + (1 - a_z) * (i + 0.5) / n
        adot = np.sqrt(W_K + (W_M / a) + (W_R / (a * a)) + (W_V * a * a))
        DTT = DTT + 1. / adot
        DCMR = DCMR + 1. / (a * adot)

    DTT = (1. - a_z) * DTT / n
    DCMR = (1. - a_z) * DCMR / n
    age = DTT + z_age
    age_Gyr = age * (Tyr / H_0)
    DTT_Gyr = (Tyr / H_0) * DTT
    DCMR_Gyr = (Tyr / H_0) * DCMR
    DCMR_Mpc = (c / H_0) * DCMR

    # tangential comoving distance

    ratio = 1.00
    x = sqrt(abs(W_K)) * DCMR
    if x > 0.1:
        if W_K > 0:
            ratio = 0.5 * (exp(x) - exp(-x)) / x
        else:
            ratio = sin(x) / x
    else:
        y = x * x
        if W_K < 0: y = -y
        ratio = 1. + y / 6. + y * y / 120.
    DCMT = ratio * DCMR
    DA = a_z * DCMT
    DA_Mpc = (c / H_0) * DA
    kpc_DA = DA_Mpc / 206.264806
    DA_Gyr = (Tyr / H_0) * DA
    DL = DA / (a_z * a_z)
    DL_Mpc = (c / H_0) * DL
    return DL_Mpc

data1=np.loadtxt(open(r"local_galaxies/nanyao_lu_local_galaxies.csv"),delimiter=',')
data2=np.loadtxt(open(r"local_galaxies/malhotra_local_galaxies.csv"),delimiter=',')
data3=np.loadtxt(open(r"local_galaxies/daiz_santos_local_galaxies.csv"),delimiter=',')


def lu_local_galaxies_1(data1):

    f_N = []
    z = []
    fwhm = []
    IR_luminosity_log = []

    for i in range(data1.shape[0]):
        f_N.append(ufloat(data1[i][0], data1[i][1]))
        z.append(data1[i][2])
        fwhm.append(data1[i][3])
        IR_luminosity_log.append(data1[i][4])

    f_N = np.asarray(f_N)
    z = np.asarray(z)
    fwhm = np.asarray(fwhm)
    IR_luminosity_log = np.asarray(IR_luminosity_log)

    print(f" median redhsift = {np.median(z)}")

    constant = (4 * np.pi * (1e-17)) / (3.826e26)

    f_N_luminosity = []

    for i in range(data1.shape[0]):
        f_N_luminosity.append(constant * f_N[i] * ((luminosity_distance(z[i]) * 3.086e22) ** 2))
    f_N_luminosity = np.asarray(f_N_luminosity)

    for i in range(data1.shape[0]):
        print(f"{i} : {f_N_luminosity[i]}")

    print(f"mean_lg = {np.mean(f_N_luminosity)}")

    a = []
    b_1=[]
    for i in range(data1.shape[0]):
        a.append(np.log10(f_N_luminosity[i].n) - IR_luminosity_log[i])
        b_1.append(np.log10(f_N_luminosity[i].n))
    a = np.asarray(a)
    print(len(a))

    a_2=[]
    IR_luminosity_log_2=[]
    for i in range(len(a)):
        if a[i]>-4.8:
            a_2.append(a[i])
            IR_luminosity_log_2.append(IR_luminosity_log[i])
        else:continue

    print(len(a_2))
    print(len(IR_luminosity_log_2))

    #plt.scatter(IR_luminosity_log_2, a_2)
    #plt.show()
    return IR_luminosity_log, a, b_1




def malhotra_local_galaxies_2(data):
    print(data)

    fir = []
    f_N = []
    D_L = []

    print(data.shape[0])

    for i in range(data.shape[0]):
        fir.append(data[i][0])
        f_N.append(data[i][1])
        D_L.append(data[i][2])

    print(f_N)



    constant = (4 * np.pi * ((3.086e22) ** 2) * (1e-14)) / (3.846e26)

    print(constant)

    fir_luminosity = []
    for i in range(data.shape[0]):
        fir_luminosity.append(constant * data[i][0] * (data[i][2] ** 2))

    fir_luminosity = np.asarray(fir_luminosity)
    print(fir_luminosity)
    print("")

    ir_luminosity=fir_luminosity/0.75
    #ir_luminosity=fir_luminosity/1

    f_N_luminosity = []

    for i in range(data.shape[0]):
        f_N_luminosity.append(constant * data[i][1] * (data[i][2] ** 2))

    f_N_luminosity = np.asarray(f_N_luminosity)
    print(f_N_luminosity)
    print("")

    f_N_luminosity = f_N_luminosity / 5   # converting to [NII]205

    for i in range(data.shape[0]):
        print(f"{fir_luminosity[i]} : {f_N_luminosity[i]} : {f_N_luminosity[i] / fir_luminosity[i]}")

    f_N_IR_luminoisty = []

    for i in range(data.shape[0]):
        f_N_IR_luminoisty.append(f_N_luminosity[i] / ir_luminosity[i])

    f_N_IR_luminoisty = np.asarray(f_N_IR_luminoisty)
    print(f_N_IR_luminoisty)



    #plt.scatter(np.log10(ir_luminosity), np.log10(f_N_IR_luminoisty))
    #plt.xlabel(r'$L_{IR}$')
    #plt.ylabel(r'$\frac{L_{[NII]}}{L_{IR}}$')
    #plt.show()

    return ir_luminosity,f_N_IR_luminoisty








def daizsantos(data3):
    f_n = []
    z=[]
    ir_log=[]

    for i in range(data3.shape[0]):
        f_n.append(ufloat(data3[i][0],data3[i][1]))
        z.append(data3[i][2])
        ir_log.append(data3[i][4])

    #print(f_n)
    #print(z)
    #print(ir_log)

    f_n = np.asarray(f_n)
    z = np.asarray(z)
    ir_log=np.asarray(ir_log)
    print(f"Hi : {len(f_n)}")



    f_N_luminosity = []

    constant = (4 * np.pi * (1e-17)) / (3.846e26)

    for i in range(data3.shape[0]):
        f_N_luminosity.append(constant * f_n[i] * ((luminosity_distance(z[i]) * 3.086e22) ** 2))
    f_N_luminosity = np.asarray(f_N_luminosity)

    #print(f_N_luminosity)

    f_N_luminosity = f_N_luminosity / 5  # converting to [NII]205



    a = []
    b = []
    for i in range(data3.shape[0]):
        a.append(np.log10(f_N_luminosity[i].n) - ir_log[i])
        b.append(f_N_luminosity[i]/(10**ir_log[i]))
    a = np.asarray(a)
    b = np.asarray(b)
    print(len(a))

    a_2=[]
    b_2 = []
    ir_log_2=[]
    for i in range(len(a)):
        if a[i]>-5:
            a_2.append(a[i])
            b_2.append(b[i])
            ir_log_2.append(ir_log[i])
        else:
            continue

    print(len(a_2))
    print(len(ir_log_2))

    err=0
    err_bars=[]

    for k in range(len(b_2)):
        print(a_2[k],np.log10(b_2[k].n),b_2[k].s/1e-5)
        #print(b_2[k].s/(b_2[k].n * np.log(10)),k)
        err = err + (b_2[k].s/(b_2[k].n * np.log(10)))
        err_bars.append(b_2[k].s/(b_2[k].n * np.log(10)))
    err = err/len(b_2)

    print(err)

    e



    #plt.scatter(ir_log, a)
    #plt.title("Diaz")
    #plt.show()
    return ir_log_2,a_2,err,np.asarray(err_bars)







a,b,b_1=lu_local_galaxies_1(data1)
c,d=malhotra_local_galaxies_2(data2)
e,f,err_diaz,err_bars_diaz=daizsantos(data3)

print(len(a))
print(len(e))
print(len(a)+len(e))
print(len(c))


for i in range(len(nll_ir_pdbi_smg_nominal)):
    print(f"{nll_pdbi_smg[i]*1e-8} : {ir_pdbi_smg_nominal[i]*1e-12} :{nll_ir_pdbi_smg_nominal[i]}")


print(nll_ir_pdbi_smg_nominal)
print(nll_ir_pdbi_qso_nominal[0])
print(nll_ir_new_qso)


#New Lu et al data
constant  = (4*np.pi)/3.846e26

f_lu_table_1 = open(r"local_galaxies/new_lu_et_al/Lu_et_al_table1.dat")
lines_lu_table_1 = f_lu_table_1.readlines()

f_lu_table_nii = open(r"local_galaxies/new_lu_et_al/Lu_et_al_table_nii.dat")
lines_lu_table_nii = f_lu_table_nii.readlines()

name = []
ir = []
dist = []
for i in range(len(lines_lu_table_1)):
    a = lines_lu_table_1[i].split()
    name.append(a[0] + a[1])
    ir.append(float(a[9]))
    dist.append(float(a[11]))

ir = np.asarray(ir)
dist = np.asarray(dist)

f_nii = []

#To get the data
for i in range(len(lines_lu_table_nii)):
    a = lines_lu_table_nii[i].split()
    #f_nii.append(float(a[14]))


    f_nii.append(ufloat(float(a[14]),np.abs(float(a[27]))))


f_nii = np.asarray(f_nii)

print(f_nii)



#To convert negative data to 0
for i in range(len(f_nii)):
    print(f_nii[i].n)
    if f_nii[i].n < 0:
        f_nii[i] = ufloat(0,0)
        #f_nii[i].s = 0




nll_ir_lu = []
new_ir_lu = []
err_new_lu=[]

for i in range(len(f_nii)):
    if f_nii[i]!=0 and name[i] != 'NGC2146NW' and name[i] != 'NGC2146Nuc' and name[i] != 'NGC2146SE':

        l = f_nii[i] * 1e-17 * constant * ((dist[i] * 3.086e22) ** 2)
        #new_ir_lu.append(ir[i])
        #nll_ir_lu.append(np.log10(l) - ir[i])
        print(l,l.n,ir[i])
        a = np.log10(l.n) - ir[i]


        if -5.4< a < -3.7:
            z = 0
            print('hi')
            print(z)
            new_ir_lu.append(ir[i])
            nll_ir_lu.append(np.log10(l.n) - ir[i])
            z=l/(10**ir[i])
            print(z)
            err_new_lu.append(z.s/(z.n*np.log(10)))
            print("")


    else:
        continue


nll_ir_lu = np.asarray(nll_ir_lu)
new_ir_lu = np.asarray(new_ir_lu)
err_new_lu = np.asarray(err_new_lu)

print(err_new_lu,np.mean(err_new_lu),np.median(err_new_lu))







#Pavesi et al 2018, 2019

P_19_N = np.asarray([ufloat(0.04,0),ufloat(0.036,0.012),ufloat(0.032,0.013),ufloat(0.22,0.07)])*1e9
P_19_IR = np.asarray([ufloat(5.2,3.6),ufloat(4.9,(4.4+2.6)/2),ufloat(12,(10+6)/2),ufloat(13,(11+7)/2)]) * 1e11 * 1.5
P_19_c_n = np.asarray([ufloat(24,0),ufloat(41,15),ufloat(61,(40+17)/2),ufloat(17,5.5)])
P_19_N_IR = P_19_N/P_19_IR

print("")
#print(P_19_IR/1e11/1.5)
print(P_19_N_IR[0].s/(P_19_N_IR[0].n * np.log(10)))
print(((P_19_N_IR[0].s/(P_19_N_IR[0].n * np.log(10)))))

err_P_19_18_N_IR = ((P_19_N_IR[0].s/(P_19_N_IR[0].n * np.log(10))) + (P_19_N_IR[1].s/(P_19_N_IR[1].n * np.log(10))) + (P_19_N_IR[2].s/(P_19_N_IR[2].n * np.log(10)))  +(P_19_N_IR[3].s/(P_19_N_IR[3].n * np.log(10)))) /4
print(err_P_19_18_N_IR)





#


"""print("")
print(P_19_N_IR/1e-5)
print(P_19_N_IR[0].s/(P_19_N_IR[0].n * np.log(10)))
print(P_19_N_IR[1].s/(P_19_N_IR[1].n * np.log(10)))
print(P_19_N_IR[2].s/(P_19_N_IR[2].n * np.log(10)))
print(P_19_N_IR[3].s/(P_19_N_IR[3].n * np.log(10)))
exit()"""


P_18_n = np.asarray([ufloat(49e7,10e7)])
P_18_IR = np.asarray([ufloat(3.2e13,0.3e13)])
P_18_c_n = np.asarray([ufloat(930e7,100e7)/P_18_n[0]])
P_18_n_IR = P_18_n/P_18_IR




'''
These values are correct and the same in the paper
print('')
print(nll_ir_pdbi_smg_nominal)
print(nll_ir_pdbi_qso_nominal)
print(nll_ir_new_qso)'''


(nll_ir_smg_err/(nll_ir_smg_nominal * np.log(10)))

plt.errorbar(np.log10(ir_smg_nominal),np.log10(nll_ir_smg_nominal),yerr=(nll_ir_smg_err/(nll_ir_smg_nominal * np.log(10))),fmt='s',color='orange',capsize=3,label='SMG')      #0.1, 0.05, 0.1
plt.errorbar(np.log10(ir_qso_nominal),np.log10(nll_ir_qso_nominal),yerr=(nll_ir_qso_err/(nll_ir_qso_nominal * np.log(10))),fmt='s',color='red',capsize=3,label='QSO')

plt.errorbar(np.log10(ir_pdbi_smg_nominal[0]),np.log10(nll_ir_pdbi_smg_nominal[0]),yerr=(nll_ir_pdbi_smg[0].s/(nll_ir_pdbi_smg_nominal[0]*np.log(10))),fmt='*',color='orange',ecolor='grey',capsize=5,markersize=20) #,markersize=15
plt.errorbar(np.log10(ir_pdbi_smg_nominal[1]),np.log10(nll_ir_pdbi_smg_nominal[1]),yerr=(nll_ir_pdbi_smg[1].s/(nll_ir_pdbi_smg_nominal[1]*np.log(10))) *1.7,uplims=True,fmt='*',color='orange',ecolor='grey',capsize=5,markersize=20)
plt.errorbar(np.log10(ir_pdbi_smg_nominal[2]),np.log10(nll_ir_pdbi_smg_nominal[2]),yerr=(nll_ir_pdbi_smg[2].s/(nll_ir_pdbi_smg_nominal[2]*np.log(10))),fmt='*',color='orange',ecolor='grey',capsize=5,markersize=20)
plt.errorbar(np.log10(ir_pdbi_qso_nominal[0]),np.log10(nll_ir_pdbi_qso_nominal[0]),yerr=(nll_ir_pdbi_qso[0].s/(nll_ir_pdbi_qso_nominal[0]*np.log(10))),fmt='*',color='red',ecolor='grey',capsize=5,markersize=20)

plt.errorbar(np.log10(ir_new_qso[0].n),np.log10(nll_ir_new_qso[0].n),yerr=(nll_ir_new_qso[0].s/(nll_ir_new_qso[0].n*np.log(10))) *1.5,fmt='*',uplims=True,color='red',ecolor='grey',capsize=5,markersize=20)
plt.errorbar(np.log10(ir_new_qso[1].n),np.log10(nll_ir_new_qso[1].n),yerr=(nll_ir_new_qso[1].s/(nll_ir_new_qso[1].n*np.log(10))) *5,fmt='*',uplims=True,color='red',ecolor='grey',capsize=5,markersize=20)


plt.errorbar(new_ir_lu,nll_ir_lu,color='green',fmt='^',label=f'Diaz-Santos+17\nLu+17, Malhotra+01')
plt.errorbar(np.log10(c),np.log10(d),color='green',fmt='^')
plt.errorbar(e,f,color='green',fmt='^')#,yerr=err_bars_diaz


plt.errorbar(np.log10(P_19_IR[0].n),np.log10(P_19_N_IR[0].n),color='darkblue',fmt='^',label=f'Pavesi+18,19')
plt.errorbar(np.log10(P_19_IR[1].n),np.log10(P_19_N_IR[1].n),color='darkblue',fmt='^')
plt.errorbar(np.log10(P_19_IR[2].n),np.log10(P_19_N_IR[2].n),color='darkblue',fmt='^')
plt.errorbar(np.log10(P_19_IR[3].n),np.log10(P_19_N_IR[3].n),color='darkblue',fmt='^')

plt.errorbar(np.log10(P_18_IR[0].n),np.log10(P_18_n_IR[0].n),color='darkblue',fmt='^')


#errorbars of local galaxies and Pavesi
err_P_19_18_N_IR = ((P_19_N_IR[0].s/(P_19_N_IR[0].n * np.log(10))) + (P_19_N_IR[1].s/(P_19_N_IR[1].n * np.log(10))) + (P_19_N_IR[2].s/(P_19_N_IR[2].n * np.log(10))) + (P_18_n_IR[0].s/(P_18_n_IR[0].n * np.log(10))) +(P_19_N_IR[3].s/(P_19_N_IR[3].n * np.log(10)))) /5
plt.errorbar(9.5,-5.25,yerr=(np.mean(err_bars_diaz) + np.mean(err_new_lu))/2,color='green',elinewidth=2,capthick=2,capsize=5,fmt='^',markersize=7)
plt.errorbar(9.7,-5.25,yerr=err_P_19_18_N_IR,color="darkblue",elinewidth=2,capthick=2,capsize=5,fmt='^',markersize=10)
plt.plot([9,10],[-4.9,-4.9],color='black',linewidth=2)
plt.plot([10,10],[-4.9,-6],color='black',linewidth=2)



#Names
plt.text(np.log10(ir_pdbi_smg_nominal[0])+0.025, np.log10(nll_ir_pdbi_smg_nominal[0]) + 0.025, "GN20", fontsize="xx-large")
plt.text(np.log10(ir_pdbi_smg_nominal[1])-0.6, np.log10(nll_ir_pdbi_smg_nominal[1])-0.02, "HDF850.1", fontsize="xx-large")
plt.text(np.log10(ir_pdbi_smg_nominal[2])+0.03, np.log10(nll_ir_pdbi_smg_nominal[2]) + 0.025, "ID141", fontsize="xx-large")
plt.text(np.log10(ir_pdbi_qso_nominal[0])+0.03, np.log10(nll_ir_pdbi_qso_nominal[0]) + 0.025, "PSSJ2322", fontsize="xx-large")
plt.text(np.log10(ir_new_qso[0].n)-0.2, np.log10(nll_ir_new_qso[0].n)+0.05 , "J2054", fontsize="xx-large")
plt.text(np.log10(ir_new_qso[1].n)-0.2, np.log10(nll_ir_new_qso[1].n)+0.05, "J2310", fontsize="xx-large")


plt.xlabel(r'$logL_{IR}$',fontsize = 30)#,fontsize = 16
plt.ylabel(r'log(L$_{[NII]}$/L$_{IR}$)',fontsize = 30)

plt.xticks(fontsize=25) #fontsize=12
plt.yticks(fontsize=25)
plt.legend(loc='upper right',fontsize=18) #fontsize=14
#plt.title(r'$log\frac{L_{[NII]}}{L_{IR}}$ vs $logL_{IR}$')
#plt.axhline(y = -3.5, color = 'grey', linestyle = '--')


#plt.xlim(8.8,14.2)
plt.xlim(9.3,14.2)
plt.ylim(-5.6,-3.4)
#plt.title(r'$log\frac{L_{[NII]}}{L_{IR}}$ vs $logL_{IR}$')
plt.show()




#######################################################################################################################
#[CII]/[NII]


ir_smg_2 = np.asarray([ufloat(1.1e12, 0.4e12),
    ufloat(1.1e13, 0.2e13),
    ufloat(1.9e13, 0.3e13),
    ufloat(2.8e13, 0),
    ufloat(6.4e13, 1.0e13) / ufloat(5.7, 0.5),
    ufloat(6.7e13, 0)])

ir_qso_2 = np.asarray([ufloat(5e13, 0),
    ufloat(49.6e12, 3.8e12) * (1 - 0.29)])

ir_smg_2_nom = []
ir_qso_2_nom = []

for i in range(len(ir_smg_2)):
    ir_smg_2_nom.append(ir_smg_2[i].n)


for i in range(len(ir_qso_2)):
    ir_qso_2_nom.append(ir_qso_2[i].n)

ir_smg_2_nom = np.asarray(ir_smg_2_nom)
ir_qso_2_nom = np.asarray(ir_qso_2_nom)

nll_smg_2 = np.asarray([ufloat(7.5e7, 2e7),
                                       ufloat(7.0e8, 0.7e8) / ufloat(14.7, 0.3),
                                       ufloat(2.14e8, 0.23e8),
                                       ufloat(2.0e8, 0.9e8),
                                       ufloat(1.06e9, 0.15e9) / ufloat(5.7, 0.5),
                                       ufloat(6.1e8, 0.5e8),
                                        ])

nll_qso_2 = np.asarray([ufloat(3.97e8, 0.3e8),ufloat(4.4e8, 0.4e8)])

cll_smg=np.asarray([ufloat(8.4e8,1.0e8), #Hyde
                                   ufloat(0,0),         #9io9
                                   ufloat(7.8e9,1.1e9), #A-1
                                   ufloat(6.69e9,0.23e9),#A-3
                                                             #ID141
                                   ufloat(21e9,4e9)/ufloat(5.7,0.5), #SPT
                                                             #MM
                                   ufloat(10.0e9,1.5e9), #BR1202-N
                                    ])

cll_qso = np.asarray([ufloat(6.5e9,1.0e9),ufloat(16.4e9,2.6e9)])

cll_nll_smg_nominal = []
cll_nll_qso_nominal = []

cll_nll_smg_error = []
cll_nll_qso_error = []

cll_nll_smg = cll_smg/nll_smg_2
cll_nll_qso = cll_qso/nll_qso_2

for i in range(len(cll_nll_smg)):
    cll_nll_smg_nominal.append(cll_nll_smg[i].n)
    cll_nll_smg_error.append(cll_nll_smg[i].s)

for i in range(len(cll_nll_qso)):
    cll_nll_qso_nominal.append(cll_nll_qso[i].n)
    cll_nll_qso_error.append(cll_nll_qso[i].s)

cll_nll_smg_nominal = np.asarray(cll_nll_smg_nominal)
cll_nll_qso_nominal = np.asarray(cll_nll_qso_nominal)
cll_nll_smg_error = np.asarray(cll_nll_smg_error)
cll_nll_qso_error = np.asarray(cll_nll_qso_error)



#Sources in Paper

cll_pdbi_smg = np.asarray([ufloat(0,0),ufloat(1.6e9,0),ufloat(61.6e9,9.8e9)/5.8])
print(cll_pdbi_smg/1e9) #Values are the same in the paper

cll_pdbi_qso = np.asarray([ufloat(1.8e9,1e9)])

cll_new_qso = np.asarray([ufloat(3.36e9,0.12e9),ufloat(8.31e9,0.41e9)])



cll_nll_pdbi_smg = cll_pdbi_smg/nll_pdbi_smg
cll_nll_pdbi_qso = cll_pdbi_qso/nll_pdbi_qso
cll_nll_new_qso = cll_new_qso/nll_new_qso


"""******************************************************************************************************************"""



data_lu = np.loadtxt(r"local_galaxies/lu_daiz_c_n.csv",delimiter=',')
data_malhotra_c_n = np.loadtxt(r"local_galaxies/malhotra_c_n.csv",delimiter=',')

ir_data_lu = []
r_data_lu = []

print(data_lu.shape[0])



for i in range(data_lu.shape[0]):
    ir_data_lu.append(data_lu[i][0])
    r_data_lu.append(data_lu[i][1])


ir_data_malhotra_c_n = []
r_data_malhotra_c_n = []

for i in range(data_malhotra_c_n.shape[0]):
    ir_data_malhotra_c_n.append(data_malhotra_c_n[i][0])
    r_data_malhotra_c_n.append(data_malhotra_c_n[i][1])


"""plt.scatter(ir_data_lu,r_data_lu,marker='^',color='green')
plt.scatter(ir_data_malhotra_c_n,r_data_malhotra_c_n,marker='^',color='green')
plt.show()"""




"""******************************************************************************************************************"""



plt.errorbar(np.log10(ir_smg_2_nom),np.log10(cll_nll_smg_nominal),yerr=(cll_nll_smg_error/(cll_nll_smg_nominal * np.log(10))),fmt='s',color='orange',capsize=3,label='SMG')
plt.errorbar(np.log10(ir_qso_2_nom),np.log10(cll_nll_qso_nominal),yerr=(cll_nll_qso_error/(cll_nll_qso_nominal * np.log(10))),fmt='s',color='red',capsize=3,label='QSO')


plt.errorbar(np.log10(ir_pdbi_smg_nominal[1]),np.log10(cll_nll_pdbi_smg[1].n),yerr=np.nanmean(cll_nll_smg_error/(cll_nll_smg_nominal * np.log(10))),lolims=True,fmt='*',color='orange',ecolor='grey',capsize=5,markersize=20)
plt.errorbar(np.log10(ir_pdbi_smg_nominal[2]),np.log10(cll_nll_pdbi_smg[2].n),yerr=cll_nll_pdbi_smg[2].s/(cll_nll_pdbi_smg[2].n * np.log(10)),fmt='*',color='orange',ecolor='grey',capsize=5,markersize=20)
plt.errorbar(np.log10(ir_pdbi_qso_nominal),np.log10(cll_nll_pdbi_qso[0].n),yerr=cll_nll_pdbi_qso[0].s/(cll_nll_pdbi_qso[0].n * np.log(10)) - 0.17,uplims=True,fmt='*',color='red',ecolor='grey',markersize=20)
plt.errorbar(np.log10(ir_new_qso[0].n),np.log10(cll_nll_new_qso[0].n),yerr=(cll_nll_new_qso[0].s/(cll_nll_new_qso[0].n*np.log(10))) + 0.12,fmt='*',lolims=True,color='red',ecolor='grey',capsize=5,markersize=20)
plt.errorbar(np.log10(ir_new_qso[1].n),np.log10(cll_nll_new_qso[1].n),yerr=(cll_nll_new_qso[1].s/(cll_nll_new_qso[1].n*np.log(10))) + 0.12,fmt='*',lolims=True,color='red',ecolor='grey',capsize=5,markersize=20)#,markersize=15




#local galaxies
plt.errorbar(ir_data_lu,r_data_lu,color='green',fmt='^',label=f'Diaz-Santos+17\nLu+17, Malhotra+01')
plt.errorbar(ir_data_malhotra_c_n,r_data_malhotra_c_n,color='green',fmt='^')



#Pavesi
plt.errorbar(np.log10(P_19_IR[0].n),np.log10(P_19_c_n[0].n),color='darkblue',fmt='^',label=f'Pavesi+18,19')
plt.errorbar(np.log10(P_19_IR[1].n),np.log10(P_19_c_n[1].n),color='darkblue',fmt='^')
plt.errorbar(np.log10(P_19_IR[2].n),np.log10(P_19_c_n[2].n),color='darkblue',fmt='^')
plt.errorbar(np.log10(P_19_IR[3].n),np.log10(P_19_c_n[3].n),color='darkblue',fmt='^')

plt.errorbar(np.log10(P_18_IR[0].n),np.log10(P_18_c_n[0].n),color='darkblue',fmt='^')


err_P_19_18_C_N = ((P_19_c_n[0].s/(P_19_c_n[0].n * np.log(10))) + (P_19_c_n[1].s/(P_19_c_n[1].n * np.log(10))) + (P_19_c_n[2].s/(P_19_c_n[2].n * np.log(10))) + (P_19_c_n[3].s/(P_19_c_n[3].n * np.log(10))) +( P_18_c_n[0].s/(P_18_c_n[0].n * np.log(10))))/5
plt.errorbar(9.7,0.25,yerr=err_P_19_18_C_N,color="darkblue",elinewidth=2,capthick=2,capsize=5,fmt='^',markersize=10)
plt.errorbar(9.5,0.25,yerr=err_P_19_18_C_N/2.5,color="green",elinewidth=2,capthick=2,capsize=5,fmt='^',markersize=7)
plt.plot([9,10],[0.45,0.45],color='black',linewidth=2)
plt.plot([10,10],[0.45,0],color='black',linewidth=2)


plt.text(np.log10(ir_pdbi_smg_nominal[1]) + 0.015,np.log10(cll_nll_pdbi_smg[1].n)-0.095, "HDF850.1", fontsize="xx-large")
plt.text(np.log10(ir_pdbi_smg_nominal[2]) + 0.04,np.log10(cll_nll_pdbi_smg[2].n)+0.02,"ID141",fontsize="xx-large")
plt.text(np.log10(ir_pdbi_qso_nominal) - 0.6,np.log10(cll_nll_pdbi_qso[0].n) + 0.02,"PSSJ2322",fontsize="xx-large")
plt.text(np.log10(ir_new_qso[0].n) + 0.017,np.log10(cll_nll_new_qso[0].n) + 0.029,"J2054",fontsize="xx-large")
plt.text(np.log10(ir_new_qso[1].n) + 0.017,np.log10(cll_nll_new_qso[1].n) + 0.029,"J2310",fontsize="xx-large")




plt.xlabel(r' $logL_{IR}$',fontsize=30)#,fontsize=14
plt.ylabel(r' $log(L_{[CII]}/L_{[NII]})$',fontsize=30)
#plt.xticks([12.25,12.75,13.25,13.75],['12.25','12.75','13.20','13.75'])#fontsize=12
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.legend(loc='upper left',fontsize=18)
plt.xlim(9.3,14.2)
plt.ylim(0.11,2.1)

plt.axhspan(np.log10(2), np.log10(4), color='green', alpha=0.2, lw=0)
plt.axhspan(np.log10(10), 2.2, color='red', alpha=0.2, lw=0)
plt.text(12.5,0.41,"HII region",color='green',fontsize=25) #,fontsize=16
plt.text(12.3,1.95,"PDR/XDR/Shock",color='red',fontsize=25)

#plt.title(r' $log\frac{L_{[CII]}}{L_{[NII]}}$ vs  $logL_{IR}$')
plt.show()
