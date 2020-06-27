#**************************************************
#Python
import numpy as np
import pandas as pd
import os.path
import subprocess
import math
import datetime
import matplotlib
import os.path
from copy import copy, deepcopy
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime, timedelta

#settings
start_date=datetime(2014,12,1)
start_date_scen=start_date+timedelta(days=20)
num_days=20
num_days_scen=7

dtdz=-0.006       # temperature gradient (deg C km-1)
hstat=1709         # elevation of weather station
topo=[0.78,1768,44,0.0172]        # area, elevation, max depth, volume
elev_lake=topo[1]     # elevation of lake

# ---- scenario - INPUT / ADJUSTMENT / CALIBRATION
scenprec_corrfact=1.0              # Factor to correct precipitation in second week of scenario - normally strongly overestimated...

ice_thickness_scen=13              # (cm) Ice thickness (measured) at start of scenario period  (default: -9999)

snow_thickness_scen=0             # (cm w.==!) snow thickness (measured) at start of scenario period  (default: -9999)

snow_dens_scen=600                # (kg m-3) snow density (measured/estimated)  at start of scenario period  (default: -9999)

slushice_thickness_scen=0               # (cm) thickness of refrozen ice ("Sandwich") at start of scenario period  (default: -9999)

# ------------------


format_csv='y'      # raw climate file in csv format

# prepare climate data file
prepare='n'

twater_ini=[4,4,20]       # [Tmin,Tmax,break(m)]

int_slush=[[18,1],[20,1]]       # dates to insert slush

vol_slush=[0.02,0.02]            # volumes of inserted slush in m w.e. (same length as int_slush)

dtdz=-0.006       # temperature gradient (deg C km-1)
hstat=1709         # elevation of weather station

alpha_ice=0.3         # albedo of ice
alpha_water=0.12      # albedo of water
alpha_snow0=0.92      # albedo of fresh snow
alpha_snow1=0.5       # albedo of very old snow

turbulent_fact=1.5  # 5 # factor to increase transfer of water heat to base of ice (to be calibrated)

corrprec=1      # Korrektur des Niederschlags

lakedepth=25     # depth of lake m

first_freezeparam=1.0        # [-] Parameter to accelerate or decelerate the first freezing of the lake

# temperature of river entering the lake and lake geometry
topo=[0.78,1768,44,0.0172]        # area, elevation, max depth, volume
flow=1     #5.8                    # [m3/s] river charactersitic inflow
flow_tcorr=-5    # [deg C] correct temperature of stream flow for elevation / snow melt relative to air temperature
flow_tmin=0      # [deg C] minimum temperature of stream

wwin=[1.2,1,0.8]     # weighting of lake water layering in different depths

elev_lake=topo[1]     # elevation of lake


# number of iterations  for heat conduction (per hour)
iterations_si=150     # snow / ice
iterations_w=150     # water

# number of iterations for snow surface temperature
dt_snoit=120    # s

freq_plots=6     # (hours) frequency of output plots
language_plots='e'      # language of plots (d: deutsch, e: english)

# -----------------------------
# parameters / constants

kice=2.33       # conductivity ice [J s-1 K-1 m-1]
kwater=0.56     # conductivity water
kair=0.001

cice=1890000     # heat capacity of ice
cair=1297
cwater=4217700    # [J m-3 K-1]

emissivity_atm=0.95
emissivity_ice=0.966
emissivity_water=0.957
emissivity_snow=0.97

attenuation_water=1       # m-1
attenuation_ice0=1.0         # m-1
attenuation_snow=10     # m-1   ??? 10
at_rho=250
fact_at=0.1                # increase in ice attenuation per day (ageing)

rho=917   # ice density
rho_water=1000
rho_snow0=70    # density of fresh snow
rho_max=400

compression='y'
rho_compression=450    # kg m-3


L=334000*rho    # [J m-3] latent heat of fusion
Lhe=2260000    # [J kg-1] latent heat of evapoation
Lhs=2594000    # [J kg-1] latent heat of sublimation

bulk_exchange_s=0.3*pow(10,-3)      # bulk exchange coefficients for sensible heat flux (to be calibrated)
bulk_exchange_l=0.8*pow(10,-3)      # bulk exchange coefficients for latent heat flux (to be calibrated)

bulk_exchange_s=1.5*pow(10,-3)      # bulk exchange coefficients for sensible heat flux (to be calibrated)
bulk_exchange_l=1.5*pow(10,-3)      # bulk exchange coefficients for latent heat flux (to be calibrated)


# -----------------------
# settings

dz=0.005      # nodespacing (ice)
dzs=0.0025    # snow
dzw=0.2      # water

nodes=[lakedepth/dzw,200,200]     # water, ice, snow

dt=3600     # [s] timestep

s_b=5.67*pow(10,-8)    # stefan boltzmann

t_wmin=4     # minimum temperature of water (density)

meltlast=0

# *************************************************
# read files
df=pd.read_csv('../data/raw/samedan_2014_2015.dat',header=None,encoding='latin-1',skiprows=14, sep='\s+',names=['STA','Year','Mo','Day','HH','MM','Tair','RH','Wind','Rad','Pressure','Prec'])
df=df.replace(32767, np.NaN)    #Assign missing data as null
df=df.drop('STA',axis=1)
error=df.isnull()  # Grab DataFrame rows where column has errors
right=error[error['Tair']].index.tolist()
a=np.array(right)
b=a-1
b[0]=0
columns=['Tair','RH','Wind','Rad','Pressure','Prec']

df['Year']=df['Year'].astype(int)
df['Mo']=df['Mo'].astype(int)
df['Day']=df['Day'].astype(int)
for i in range(0, a.size):
    df.loc[a[i],columns]=df.loc[b[i],columns] #Use previous data to fill #Error last column not filled

df.loc[0,columns]=df.loc[1,columns]
for i in range(0, df.shape[0]):
    t = datetime(df['Year'][i], df['Mo'][i], df['Day'][i],df['HH'][i], 0)
    df.loc[i,'When']=t
    df.loc[i,'DoY']=t.timetuple().tm_yday
df['DoY']=df['DoY'].astype(int)
df['When'] = pd.to_datetime(df['When'])
df.to_csv('../data/interim/2015_SAM.csv', sep='\t')

# Cloudiness
clmax=np.zeros(shape=(365,24),dtype=float)
for i in range(0,365):
    for j in range(0,24):
        l=np.where(np.logical_and(np.logical_and(df['HH']==j, i-5<=df['DoY']), df['DoY']<=i+5))
        a=np.array([])
        a=np.append(0,l)
        a=np.delete(a,0)
        if a.size:
            clmax[i,j]=df.loc[a,'Rad'].max()/0.98

cl=np.empty(shape=df.shape[0],dtype=float)
for row in df.iterrows():
    a=clmax[df.loc[row[0],'DoY']-1,df.loc[row[0],'HH']]
    if a!=0:
        cl[row[0]]=(1-df.loc[row[0],'Rad']/a)
    # cloudiness at nighttime - use average of day before
    if df.loc[row[0],'Rad']<10:
        l=np.where(np.logical_and(np.logical_and(df['DoY']==df.loc[row[0],'DoY'], 10<=df['HH']), df['HH']<=15))
        b=np.array([])
        b=np.append(0,l)
        b=np.delete(b,0)
        if b.size:
            cl[row[0]]=np.mean(cl[b])
        else:
            cl[row[0]]=cl[row[0]-1]
df['Cl']=cl

#Select Data
mask = (df['When'] >= start_date) & (df['When'] < start_date_scen+ timedelta(days=num_days_scen))
df=df.loc[mask]
df=df.reset_index()

df['Tair']+=(elev_lake-hstat)*dtdz
df['Pressure']*=100
df['Prec']/=1000
df['rho_air']=df['Pressure']/(287.058*(df['Tair']+273.15))
df['ew']=6.1094*np.exp((17.625 * df['Tair'])/(df['Tair']+243.04))*(df['RH']/100)*100
df['huma']=0.622*df['ew']/abs(df['Pressure']-df['ew'])   # specific air humidity
df = df[['When','Rad', 'Tair', 'RH', 'Wind','Pressure','Prec','Cl','ew','rho_air','huma','DoY']]
df=df.round(7)

#output csv
df.to_csv('../data/interim/cloud.csv')

dfs=pd.read_csv('../data/raw/forecast2016.csv',encoding='latin-1',skiprows=1, sep=';')

#Clean file
df['When'] = pd.to_datetime(df['When'])
df['Year']=df['When'].apply(lambda x:x.year)
df['Mo']=df['When'].apply(lambda x:x.month)
df['Day']=df['When'].apply(lambda x:x.day)

mask =(df['When'] < start_date+ timedelta(days=num_days))
df=df.loc[mask]
df=df.reset_index()

#Clean Scenario file
dfs1=dfs[0:14]  #WEEK1
dfs1=dfs1.drop('Unnamed: 11', axis=1)
i=1
dfp=pd.DataFrame({'A' : []})
while i<=dfs1.shape[0]:
    dfp=dfp.append(dfs1.loc[dfs1.index[i]])
    dfs1=dfs1.drop(dfs1.index[i],axis=0)
    i=i+1
dfp=dfp.drop('A',axis=1)
dfp=dfp[['day','2','5','8','11','14','17','20','23']]

dfs2=dfs[15:22]            #WEEK2
dfs2=dfs2.drop({'Unnamed: 11','14','17','20','23'}, axis=1)
dfs2.columns = ['year', 'month','day','Tmax','Tmin','Prec','Prec-percent']
dfs2=dfs2.reset_index()

#Daily Temperature Ranges
dfs1= dfs1.apply(pd.to_numeric, errors='coerce')
for index,row in dfs1.iterrows():
    dfs1.loc[index,'Trange']=dfs1.loc[index,['2','5','8','11','14','17','20','23']].max()-dfs1.loc[index,['2','5','8','11','14','17','20','23']].min()

dfs2= dfs2.apply(pd.to_numeric, errors='coerce')
dfs2['Trange']=dfs2['Tmax']-dfs2['Tmin']

k=0
l=[]
dft=pd.DataFrame({'A' : []})
dff=pd.DataFrame({'A' : []})
c=0
for i in range(0,num_days):
    run=True
    k=0
    while run:
        l.append(df.loc[int(24*i+k),'Tair'])

        k=k+1
        if k==24:
            run=False
    l=np.array(l)
    dft.loc[i,'Tr']=l.max()-l.min()
    dft.loc[i,'When']=df.loc[int(24*i),'When']
    l=[]

#Select Climate variables using daily temperature range matches
j=0
for index,row in dfs1.iterrows():
    b=0
    l=np.array([])
    run=True
    l=abs(dfs1.loc[index,'Trange']-dft['Tr'])
    c=l.min()

    h=np.where(l==c)
    h=h[0]
    dff=dff.append(dft.loc[h[h.size-1]])
    j=j+1
dff=dff.drop('A',axis=1)
dff=dff.reset_index()
dft=dft.reset_index()
dfs1=dfs1.reset_index()

for index,row in dfs2.iterrows():
    b=0
    l=np.array([])
    l=abs(dfs2.loc[index,'Trange']-dft['Tr'])
    c=l.min()
    h=np.where(l==c)
    h=h[0]
    dff=dff.append(dft.loc[h[h.size-1]])
    j=j+1

dff=dff.drop('A',axis=1)
dff=dff.reset_index()
dft=dft.reset_index()
dfs2=dfs2.reset_index()
dft['When'] = pd.to_datetime(dft['When'])
dft['Date']= dft['When'].apply(lambda x:x.date())
dff['When'] = pd.to_datetime(dff['When'])
dff['Day']= dff['When'].apply(lambda x:x.day)

#Create scenario files
dfg=pd.DataFrame({'A' : []})
c=0
for index,row in dff.iterrows():
    l=np.where(df['Day']==dff['Day'][index])
    l=l[0]
    j=0
    while j<24:
        dfg=dfg.append(df.loc[l[0]+j],ignore_index=True)
        dfg.loc[c,'Date']=start_date_scen+timedelta(days=int(index),hours=j)
        j=j+1
        c=c+1

dfg=dfg.drop('A',axis=1)
dfg=dfg.reset_index()
dfp=dfp.reset_index()

#Use forecast2016
j=0
for i in range(0,7):
    l=np.where(dfs1['day']==dfg.loc[j,'Date'].day)
    k=0
    c=0
    if len(l[0])>0:
        while c!=24:
            dfg.loc[j+c,'Tair']=dfs1.iloc[l[0][0],4+k]
            dfg.loc[j+c,'Prec']=dfp.iloc[l[0][0],2+k]
            if dfg.loc[j+c,'Prec']=='-':
                dfg.loc[j+c,'Prec']=0
            c=c+1
            if c%3==0:
                k=k+1

    j=j+24

#WEEK2
temp_dist=[0.2,0,0.1,0.4,0.8,1,0.7,0.4,0.3]
for i in range(0,7):
    dtt=dfs2.loc[i,'Tmax']-dfs2.loc[i,'Tmin']
    dfs2.loc[i,'Tair']=dfs2.loc[i,'Tmax']+dtt
    dfs2.loc[i,'Prec']=dfs2.loc[i,'Prec']*dfs2.loc[i,'Prec-percent']/100/8

c=dfg.shape[0]
for i in range(c,c+7*24):
    dfg.loc[i,'Date']=dfg.loc[i-1,'Date']+ timedelta(hours=1)
    dfg.loc[i,'Tair']=dfs2.loc[math.floor((i-c)/24),'Tmax']+dtt*temp_dist[math.floor((i-c)/24)]
    dfg.loc[i,'Prec']=dfs2.loc[math.floor((i-c)/24),'Prec']

dfg['Prec']=dfg['Prec'].astype(float)
dfg['Prec']=(dfg['Prec'])/3000
dfg=dfg.round(5)

dfg['Prec']=dfg['Prec'].astype(float)
dfg['Prec']=(dfg['Prec'])/3000
dfg=dfg.round(5)
dfg['Date'] = pd.to_datetime(dfg['Date'])

#Add Original Weather Data
df['Time'],df['Date']= df['When'].apply(lambda x:x.time()), df['When'].apply(lambda x:x.date())
mask =(dfg['Date'] < start_date_scen+ timedelta(days=num_days_scen))
dfg=dfg.loc[mask]
dfg['Time'],dfg['Date']= dfg['Date'].apply(lambda x:x.time()), dfg['Date'].apply(lambda x:x.date())
df=pd.concat([pd.DataFrame(df), dfg], ignore_index=True)


df=pd.concat([pd.DataFrame(df), dfg], ignore_index=True)

#output csv
df = df[['When','Date','Time','Rad', 'Tair', 'RH', 'Wind','Pressure','Prec','Cl','huma','rho_air']]
df=df.round(7)
df.to_csv('../data/interim/scenario.csv')

#variables
ice=np.full(nodes[1],np.inf)
snow=np.full((3,nodes[2]),np.inf)
df['Date'] = pd.to_datetime(df['Date'])
nodes=list(map(int, nodes))
fr=0
sn=0
im=0
c=0
i_slush=0
Qbmelt=0
Qswi_i=0.0
ci_icest=0
water=np.zeros(shape=nodes[0])
water+=twater_ini[0]

for i in range(nodes[0]-1,-1,-1):
    d=i*dzw
    if d< twater_ini[2]:
        water[i]=water[i]+((((twater_ini[2]/dzw)-i)*(twater_ini[1]-twater_ini[0]))/(twater_ini[2]/dzw))

slush=np.full((3,nodes[2]),0.0)
icethick=np.full(df.shape[0],0.0)
load=np.full((2,df.shape[0]),0.0)
mtemp=np.full((3,df.shape[0]),0.0)
stype=np.full(df.shape[0],0.0)
snowthick=deepcopy(stype)
icet=deepcopy(stype)
snow_albedo=deepcopy(stype)
slush_thick=deepcopy(icethick)
i_age=np.full(df.shape[0],np.inf)

dfo=pd.DataFrame({'Type' : []})
dfo['Temp']=np.inf
dfo['Date']=df['Date'].dt.date
dfo['Time']=df['Time']
dfo['Icethick']=0
dfo['Snowthick']=0

with PdfPages('../data/processed/figures.pdf') as pdf:
    for index in range(0,df.shape[0]):
        l=np.where(snow[0,]!=np.inf)
        l=l[0]
        ci=l.size
        jj=np.where(ice!=np.inf)
        cj=jj[0].size
        l3=np.where(water!=np.inf)
        ch=l3[0].size
        if ci>0:
            tt=snow[0,ci-1]
            print(index,'Snow')
            print(snow[0,0:ci])
            dfo.loc[index,'Type']='Snow'

        if ci==0 and cj>1:
            tt=ice[0]
            print(index,'Ice')
            print(ice[0:5])
            dfo.loc[index,'Type']='Ice'
        if cj==0:
            tt=water[0]
            print(index,'Water')
            print(water[0:5])
            dfo.loc[index,'Type']='Water'
        dfo.loc[index,'Temp']=tt

        if np.isnan(tt) or tt==np.inf:
            print(snow[0,])
            exit()

        ew=6.1094*np.exp((17.625 * tt)/(tt+243.04))*100
        hum0=0.622*ew/abs(df.loc[index,'Pressure']-ew)
    # *************************************************
    # *************************************************
    # SNOW
        tes=deepcopy(snow[0,])
        tls=deepcopy(tes)

        # accumulation
        t_threshold=1.25
        ptt=df.loc[index,'Prec']
        if df.loc[index,'Tair']>t_threshold+1:
            df.loc[index,'Prec']=0
        if df.loc[index,'Tair']<t_threshold+1 and df.loc[index,'Tair']>t_threshold-1:
            df.loc[index,'Prec']=df.loc[index,'Prec']*(1-(df.loc[index,'Tair']-(t_threshold-1))/2)

        df.loc[index,'p_liq']=ptt-df.loc[index,'Prec']

        if ice[0]==np.inf:
            df.loc[index,'Prec']=0
        sn=sn+df.loc[index,'Prec']

        if sn>=dzs:
            if snow[0,0]==np.inf:
                snow[0,0]=ice[0]
            sn=sn-dzs
            l=np.where(snow[0,]==np.inf)
            l=l[0]
            snow[0,l[0]]=min([df.loc[index,'Tair'],0])
            snow[1,l[0]]=rho_snow0/1000

            #artificial compression of snow
            if compression=='y':
                snow[1,l[0]]=rho_compression/1000

            snow[2,l[0]]=1/24

        #update snow density and albedo
        ts_snow=21.9
        l=np.where(snow[0,]!=np.inf)
        l=l[0]
        if l.size!=0:
            for i in range(0,l.size):
                if snow[1,i]<rho_max/1000 or snow[1,i]==np.inf:
                    snow[1,i]=(rho_max-(rho_max-rho_snow0)*math.exp(-0.03*pow((0.01*rho_max),0.5)*snow[2,i]))/1000
                    snow[1,0]=0.2

                if snow[2,i]!=np.inf:
                    alpha_snow=alpha_snow1+(alpha_snow0-alpha_snow1)*math.exp((-1)*snow[2,i]/ts_snow)
                if snow[2,i]!=np.inf:
                    snow[2,i]=snow[2,i]+1/24

        else:
            alpha_snow=0.85

        # arrays for capacity and conductivity depending on density for snow
        csnow=np.full(nodes[2],0.0)
        ksnow=deepcopy(csnow)
        for i in range(0,nodes[2]):
            if snow[0,i] != np.inf:
                csnow[i]=(1-snow[1,i])*cair+snow[1,i]*cice
            if snow[0,i] != np.inf:
                ksnow[i]=(2.93*pow(snow[1,i],2))+0.01   # according to Mellor (1977)

        # update capacity and conductivity if slush is present
            if slush[0,i]!= 0:
                c=slush[1,i]*cwater+(1-slush[1,i])*cice
                k=slush[1,i]*kwater+(1-slush[1,i])*kice
                if slush[2,i]!=0:
                    ff=(slush[0,i]/slush[2,i])*(1-snow[1,i])
                    csnow[i]=(1-snow[1,i]-ff)*cair+snow[1,i]*cice+ff*c
                    ksnow[i]=(1-snow[1,i]-ff)*kair+snow[1,i]*kice+ff*k

        # -------  energy balance snow
        Qswi=(1-alpha_snow)*df.loc[index,'Rad']
        Qswo=(-1)*alpha_snow*df.loc[index,'Rad']   # outgoing short wave

        Qlwi=(0.68+0.0036*pow(df.loc[index,'huma'],0.5))*(1+0.18*pow(df.loc[index,'Cl'],2))*emissivity_atm*s_b*pow((df.loc[index,'Tair']+273.15),4)
        Qe=df.loc[index,'rho_air']*Lhs*bulk_exchange_l*(df.loc[index,'huma']-hum0)*df.loc[index,'Wind']
        if df.loc[index,'Tair']> 1 :
            Qp=df.loc[index,'p_liq']*(df.loc[index,'Tair']-1)*cwater/dt
        else:
            Qp=0
        gamma=math.exp(-attenuation_snow*dzs/(at_rho/1000.))
        # iterate for snow surface temperature
        if snow[0,0]== np.inf :
            dt_snoit_tt=1
        else:
            dt_snoit_tt=dt_snoit

        snit=dt/dt_snoit_tt
        l=np.where(snow[0,] != np.inf)
        l=l[0]
        ci=l.size
        snit=int(snit)
        for it in range(0,snit):
            if snow[0,0]==np.inf :
                tsno=0
            else:
                if it==0:
                    tsno=snow[0,l[ci-1]]

                else:
                    tsno=tt_ts
                tt_ts=tsno

            Qlwo=(-1)*emissivity_snow*s_b*pow((tsno+273.15),4)
            Qc=df.loc[index,'rho_air']*cair*bulk_exchange_s*(df.loc[index,'Tair']-tsno)*df.loc[index,'Wind']
            Q0snow=(1-gamma)*Qswi+Qlwi+Qlwo+Qc+Qe+Qp

        # calculate temperature of uppermost layer with energy balance
            if ci > 1 :
                tt_ts=tt_ts+Q0snow*dt_snoit_tt/(csnow[l[ci-1]]*(dzs/snow[1,l[ci-1]]))

        # done iteration for snow surface temperature

        l=np.where(snow[0,] != np.inf)
        l=l[0]
        ci=l.size
        if ci > 1 :
            # attribute surface temperature
            tls[l[ci-1]]=tt_ts
            tls[0]=ice[0]

            for i in range(ci-2,0,-1):
                tls[l[ci-2]]=tls[l[ci-1]]+(math.exp(-attenuation_snow*(i+1)*dzs/(at_rho/1000))-math.exp(-attenuation_snow*i*dzs/(at_rho/1000)))*Qswi*dt/(csnow[l[ci-1]]*(dzs/snow[1,l[ci-1]]))

            Qswi_s=Qswi*math.exp(-attenuation_snow*(ci-1)*dzs/(at_rho/1000))

            # ----- snow evaporation
            if Qe < 0 :
                sn=sn+Qe*dt/(Lhs*1000)

            # ----- snow melt
            if Q0snow > 0 and 0<=tls[l[ci-1]] :
                Qmelt=Q0snow+(tls[l[ci-1]]*csnow[l[ci-1]]*(dzs/snow[1,l[ci-1]])/dt)
                melt=Qmelt/L*dt
                if slush[1,l[ci-1]]!=0:
                    m=melt*(dzs/(dzs+slush[0,l[ci-1]]*(slush[1,l[ci-1]])))
                else:
                    m=melt
                sn=sn-m
                tls[l[ci-1]]=0
            else:
                melt=meltlast

            # reduce snow
            if sn <= -dzs :
                sn=sn+dzs
                tls[l[ci-1]]=np.inf

            l=np.where(tls != np.inf)
            l=l[0]
            ci=l.size
            # add liquid precipitation to melt water
            melt=melt+df.loc[index,'p_liq']

            # ----- refreezing of melt water
            if melt > 0 and ci>2:
                for j in range(ci-2,0,-1) :
                    re=(-1)*snow[0,j]*csnow[j]*(dzs/snow[1,j])/L
                    if melt > re :
                        tls[j]=0
                    elif re > 0 :
                        tls[j]=(1-melt/re)*snow[0,j]
                    if melt > re :
                        melt=melt-re
                    elif re > 0:
                        melt=0

            # ----- slush formation (melt water in snow pore space)
            if melt > 0 :
                for j in range(1,ci):
                    sp=dzs/snow[1,j]*(1-snow[1,j])
                    a=sp-slush[0,j]
                    slush[2,j]=sp
                    if melt < a and melt != 0 :
                        slush[0,j]=slush[0,j]+melt
                        melt=0
                        slush[1,j]=1
                    if melt > a :
                        melt=melt-a
                        slush[0,j]=sp
                    if melt < 0 :
                        melt=0
                    if slush[0,j] < 0 :
                        slush[0,j]=0

            jj=np.where(snow[0,] == np.inf)
            jj=jj[0]
            slush[0,jj]=0
            if jj.size>0:
                if jj[0]<= 1 :
                    slush[0,0:2]=dzs

            # ----- manual slush injection (melt water in snow pore space) - injection at 12.00
            if vol_slush[0] != np.inf :
                jj=np.where(df['Date'][index].day == int_slush[0] and df['Date'][index].month == int_slush[1])
                cj=jj[0].size
                if cj > 0 and df.loc[index,'Date'](3,index) == 12 :
                    ij=vol_slush[l[0]]

                    for j in range(1,ci) :
                        sp=dzs/snow[1,j]*[1-snow[1,j]]
                        a=sp-slush[0,j]
                        if ij > 0 :
                            if ij < a :
                                slush[0,j]=slush[0,j]+ij
                                ij=0
                                slush[1,j]=1
                                slush[2,j]=sp
                            else:
                                slush[0,j]=slush[0,j]+a
                                ij=ij-a
                                slush[1,j]=1
                                slush[2,j]=sp

            # ----- refreezing of slush
            for j in range(1,ci) :
                if slush[0,j] != 0 and slush[1,j] > 0 and tls[j] < 0 :
                    re=(-1)*tls[j]*csnow[j]*(dzs/snow[1,j])/L
                    if (slush[0,j]*slush[1,j]) > re :
                        tls[j]=0
                        slush[1,j]=slush[1,j]-(re/(slush[0,j]*slush[1,j]))*slush[1,j]
                    else:
                        if re > 0 :
                            tls[j]=(1+(slush[0,j]*slush[1,j])/re)*tls[j]
                            slush[1,j]=0
            l=np.where(tls != np.inf)
            l=l[0]
            ci=l.size
            tes=deepcopy(tls)

            # -----------------------
            # heat conduction
            if ci>0:
                tll=deepcopy(tls)
                tel=deepcopy(tes)
                for it in range(0,iterations_si) :
                    for j in range(0,ci-1) :
                        if j == 0 :
                            tel[j]=tll[j]-(dt/iterations_si*ksnow[j]/(csnow[j])*(tll[j]-tll[j+1])/pow((dzs/snow[1,j]),2))/2
                        else:
                            tel[j]=tll[j]+((dt/iterations_si*ksnow[j]/(csnow[j])*(tll[j-1]-tll[j])/pow((dzs/snow[1,j]),2))-(dt/iterations_si*ksnow[j]/(csnow[j])*(tll[j]-tll[j+1])/pow((dzs/snow[1,j]),2)))/2
                    tll=deepcopy(tel)

                tes=deepcopy(tel)
                ice[0]=tes[0]   # setting ice surface temperature to value of lowermost snow layer!
                snow[0,]=deepcopy(tes)
                snow[0,0]=tls[0]

                if l.size > 0 :
                    snow[0,l[ci-1]]=tls[l[ci-1]]
            meltlast=melt

        # correct and reset
        if snow[0,1] == np.inf :
            snow.T[0,]=np.inf
        l=np.where(np.logical_and(snow[0,]!=np.inf,snow[0,]<-40))
        l=l[0]

        if l.size > 0 :
            snow[0,l]=-40

        l=np.where(np.logical_and(snow[0,]!=np.inf,snow[0,]>0))
        l=l[0]
        if l.size > 0 :
            snow[0,l]=0

        # ----  forcing layers to measured thicknesses at begin of scenario period
        # if str(df.loc[index,'Date'].date())==str(start_date_scen):

        # *************************************************
        # ICE

        l=np.where(ice != np.inf)
        l=l[0]
        ci=l.size

        # age of ice sheet
        ii=np.where(ice != np.inf)
        ii=ii[0]
        jj=np.where(ice == np.inf)
        jj=jj[0]
        i_age[jj]=np.inf
        for i in range(0,ci):
            if i_age[i] == np.inf:
                i_age[i]=1/24
            else:
                i_age[i]=i_age[i]+1/24

        if ci > 0 :
            attenuation_ice=attenuation_ice0+np.mean(i_age[ii])*fact_at

        else:
            attenuation_ice=attenuation_ice0

        # energy balance ice
        Qswi=(1-alpha_ice)*df.loc[index,'Rad']
        Qswo=(-1)*alpha_ice*df.loc[index,'Rad']   #
        Qlwi=(0.68+0.0036*pow(df.loc[index,'huma'],0.5))*(1+0.18*pow(df.loc[index,'Cl'],2))*emissivity_atm*s_b*pow((df.loc[index,'Tair']+273.15),4)
        Qe=df.loc[index,'rho_air']*Lhs*bulk_exchange_l*(df.loc[index,'huma']-hum0)*df.loc[index,'Wind']                                     #huma==hum0???
        Qp=0
        gamma=math.exp(-attenuation_ice*dz)
        # iterate for ice surface temperature
        if snow[0,1] == np.inf :
            dt_iceit=dt_snoit
        else:
            dt_iceit=dt
        snit=(dt/dt_iceit)

        for it in range(0,int(snit)) :

            if ice[0] == np.inf :
                tice=0
            else:
                tice=ice[0]
            if it > 1 :
                tice=tt_ts
            else:
                tt_ts=tice

            Qlwo=(-1)*emissivity_ice*s_b*pow((tice+273.15),4)     # long-wave outgoing
            Qc=df.loc[index,'rho_air']*cair*bulk_exchange_s*(df.loc[index,'Tair']-tice)*df.loc[index,'Wind']   # sensible heat flux
            Q0ice=(1-gamma)*(Qswi)+Qlwi+Qlwo+Qc+Qe+Qp          # energy balance of top layer
            tt_ts=tt_ts+Q0ice*dt_iceit/(cice*dz)

        l=np.where(ice != np.inf)
        l=l[0]
        ci=l.size
        if ci > 0 :
            te=deepcopy(ice)
            tl=deepcopy(te)
            # calculate temperature of uppermost layer with energy balance
            if snow[0,1] == np.inf :
                tl[0]=tt_ts# apply surface temperature from iteration

                # adapting ice temperature based on direct radiation absorbed within ice sheet

                for i in range(1,ci-1):
                    tl[i]=tl[i]+(math.exp(-attenuation_ice*(i-1)*dz)-math.exp(-attenuation_ice*i*dz))*Qswi*dt/(cice*dz)
                Qswi_i=Qswi*math.exp(-attenuation_ice*(ci-2)*dz)    # radiation below ice sheet
            # below snow coverage - temperature change calculated with heat fluxes from above and below

            if snow[0,1] != np.inf :
                for i in range(0,ci-1):
                    tl[i]=tl[i]+(math.exp(-attenuation_ice*(i-1)*dz)-math.exp(-attenuation_ice*i*dz))*Qswi_s*dt/(cice*dz)
                Qswi_i=Qswi_s*math.exp(-attenuation_ice*(ci-2)*dz)   # radiation below ice sheet

                snow[0,0]=tl[0]     # equalizing ice surface and bottom snow temperature
            # ----- ice evaporation
            if Qe < 0 and snow[0,1] == np.inf :
                if 0<= i_slush :
                    im=im+Qe*dt/(Lhs*917)
                else:
                    i_slush=i_slush+Qe*dt/(Lhw*1000)
                if i_slush < 0 :
                    i_slush=0

            # ----- ice melt
            if Q0ice > 0 and tl[0] >= 0 :
                Qmelt=Q0ice+(tl[0]*cice*dz/dt)
                if Qmelt > 0 :
                    imelt=Qmelt/L*dt
                else:
                    imelt=0
                im=im-imelt
                tl[0]=0
                i_slush=i_slush+imelt
            else:
                melt=0

            # reduce ice cover
            if im <= -dz :
                im=im+dz
                for i in range(0,ci-1):
                    ice[i]=ice[i+1]
                ice[ci-1]=np.inf
                l=np.where(ice != np.inf)
                l=l[0]
                ci=l.size
                tl=deepcopy(ice)

            # ---- freezing of slush at ice surface
            if slush[0,1] > 0 and tl[0] < 0 :
                re=(-1)*tl[0]*cice*dz/L

                if slush[0,1]*slush[1,1] > re :
                    tl[0]=0
                    slush[1,1]=slush[1,1]*(re/(slush[0,1]*slush[1,1]))
                else:
                    if re > 0 :
                        tl[0]=tl[0]+tl[0]*((slush[0,1]*slush[1,1])/re)
                    slush[1,1]=0

            # ---- freezing liquid water on bare ice surface or within ice sheet!
            if i_slush > 0 :
                re=0

                for j in range(0,ci) :
                    if tl[j] < 0 :
                        re=(-1)*tl[j]*cice*dz/L
                    if i_slush > re :
                        tl[j]=0
                        i_slush=i_slush-re
                    else:
                        if re > 0 :
                            tl[j]=tl[j]+tl[j]*(i_slush/re)
                        i_slush=0

            # ------- heat conduction
            l=np.where(ice != np.inf)
            l=l[0]
            ci=l.size
            tll=deepcopy(tl)
            tel=deepcopy(te)
            for it in range(0,iterations_si) :
                for j in range(1,ci-1) :
                    tel[j]=tll[j]+((dt/iterations_si*kice/(cice)*(tll[j-1]-tll[j+1])*pow(dz,-2)))/2
                tll=deepcopy(tel)
            te=deepcopy(tel)

            jj=np.where(tl == np.inf)
            jj=jj[0]
            cj=jj.size
            if cj > 0 :
                te[jj]=np.inf

            # --- check for positive ice temperatures
            jj=np.where(np.logical_and(te>0,te!=np.inf))
            jj=jj[0]
            cj=jj.size
            if cj>0:
                te[jj]=np.inf

            hh=np.where(te<0)
            hh=hh[0]
            ch=hh.size
            if cj > 0 and ch == 0 :
                fr=fr-np.sum(te[jj])*cice*dz/L
                te[jj]=0

            if cj > 0 and ch > 0 :
                p=np.sum(te[jj])/ch
                kk=np.where(np.logical_and(te < -p, te != np.inf) )
                kk=kk[0]
                ck=kk.size
                if ck > 0 :
                    te[kk]=te[kk]+np.sum(te[jj])/ck
                else:
                    fr=fr-np.sum(te[jj])*cice*dz/L
                    te[jj]=0

            # ---  freezing / thawing at base!
            if ci > 1 :
                ice_flux=(-1)*min(te[l])*cice*dz/dt
            else:
                ice_flux=0.
            if ice_flux < 0 :
                ice_flux=0

            if ci > 1 :
                fr=fr+(ice_flux+(turbulent_fact*kwater*(water[0]-water[1])/(dzw))-Qbmelt)*dt/L
            else:
                tl[0]=np.inf

            if fr > dz :
                fr=fr-dz
                te[ci-1]=0
            if fr < (-1)*dz :
                fr=fr+dz
                te[ci-1]=np.inf
                if ci > 1 :
                    te[ci-2]=0

            ice=deepcopy(te)
            ice[0]=tl[0]   # ice loop

        if ice[0] == np.inf :
            ice[1]=np.inf
        if snow[0,0] == np.inf :    #Correction
            snow[0,1]=np.inf

        # Break up of ice cover if it is completely temperate and reasonably thin
        l=np.where(ice != np.inf)
        l=l[0]
        ci=l.size
        if ci > 0 and ci < 4 :
            if min(ice[l]) == 0 :
                ice[l]=np.inf

        ii=np.where(np.logical_and(ice != np.inf, ice<-30))
        ii=ii[0]
        ci=l.size
        if ci>0:
            ice[ii]=-30

        # *************************************************
        # *************************************************
        # WATER

        # iterate for snow surface temperature
        ts_water=water[0]
        # energy balance water
        Qswi=(1-alpha_water)*df.loc[index,'Rad']
        Qswo=(-1)*alpha_water*df.loc[index,'Rad']   #
        Qlwo=(-1)*emissivity_water*s_b*pow((ts_water+273.15),4)
        Qlwi=(0.68+0.0036*pow(df.loc[index,'huma'],0.5))*(1+0.18*pow(df.loc[index,'Cl'],2))*emissivity_atm*s_b*(pow(df.loc[index,'Tair']+273.15,4))
        Qc=df.loc[index,'rho_air']*cair*bulk_exchange_s*(df.loc[index,'Tair']-ts_water)*df.loc[index,'Wind']
        Qe=df.loc[index,'rho_air']*Lhe*bulk_exchange_l*(df.loc[index,'huma']-hum0)*df.loc[index,'Wind']
        Qp=0
        gamma=math.exp(-attenuation_water*dz)
        Q0water=Qswi+Qlwi+Qlwo+Qc+Qe+Qp
        l=np.where(water != np.inf)
        l=l[0]
        ci=l.size
        tew=deepcopy(water)
        tlw=deepcopy(tew)
        # mixing of water during summer - 	include layering of water!
        if ice[0] == np.inf :
            ww=np.full(ci,1.0)
            ww[0:11]=wwin[0]
            ww[10:21]=wwin[1]
            ww[ci-10:ci]=wwin[2]
            ww=ww/np.mean(ww)
            cj=ci
            if water[0] < t_wmin :
                ww=np.full(ci,0.0)
                ww[0]=1
                cj=1
            if Q0water < 0 :
                jj=np.where(water > t_wmin)
                jj=jj[0]
                cj=jj.size
                ww=np.full(ci,0.0)
                if cj > 0 :
                    ww[jj]=1
                else:
                    ww[0]=1
                if cj == 0 :
                    cj=1

            for j in range(0,cj):
                tlw[j]=tlw[j]+Q0water*dt/(cwater*cj*dzw)*ww[j]

            # !!! heat input by rivers and rain !!!
            tr=(df.loc[index,'Tair']+flow_tcorr)
            if tr < flow_tmin :
                tr=flow_tmin
            di_temp=tr-np.mean(water)
            hrain=(df.loc[index,'Prec']*corrprec)*1000000.*topo[0]*di_temp  # cwater gekuerzt!
            tlw=tlw+hrain/(topo[3]*pow(10,9))
            ran_riv=168
            tt=min([ran_riv,index])
            triv=np.mean(df.loc[index-tt:index+1,'Tair'])+flow_tcorr
            if triv < flow_tmin :
                triv=flow_tmin
                di_temp=triv-np.mean(water)
            hriv=flow*dt*di_temp  # cwater gekuerzt!
            tlw=tlw+hriv/(topo[3]*pow(10,9))
        if ice[0] != np.inf :
            tlw[0]=0
            Qbmelt=Qswi_i*(1-gamma)
            for i in range(1,ci):
                tlw[i]=tlw[i]+(math.exp(-attenuation_water*(i-1)*dz)-math.exp(-attenuation_water*i*dz))*Qswi_i*dt/(cwater*dzw)

        tll=deepcopy(tlw)
        tel=deepcopy(tew)

        for j in range(1,ci-1) :
            tel[j]=tll[j]+((dt/iterations_w*kwater/(cwater)*(tll[j-1]-tll[j])/pow(dzw,2))-(dt/iterations_w*kwater/(cwater)*(tll[j]-tll[j+1])/pow(dzw,2)))/2

        tll=deepcopy(tel)
        tew=deepcopy(tel)
        tew[0]=tlw[0]

        # try to eliminate values < t_wmin
        for j in range(0,ci) :
            if tew[j] < t_wmin and max(tew) > t_wmin :
                tp=t_wmin-tew[j]
                tew[j]=t_wmin

                for h in range(j+1,ci) :
                    if tew[h] > t_wmin :
                        if tp > tew[h]-t_wmin :
                            tp=tp-(tew[h]-t_wmin)
                            tew[h]=t_wmin
                        else:
                            tew[h]=tew[h]-tp
                            tp=0
        water=deepcopy(tew)
        water[0]=tlw[0]

        # first ice formation
        tfreeze=(-1)*dz*L/(dzw*cwater)*first_freezeparam
        if np.mean(water[0:2]) < tfreeze and ice[0] == np.inf :
            ice[0:3]=0
            water[0]=0

        # *******************************
        # compile results
        snow_albedo[index]=alpha_water
        l=np.where(ice != np.inf)
        l=l[0]
        ci=l.size
        if ci > 1 :
            icethick[index]=(ci-1)*dz

        if ci > 1 :
            mtemp[1,index]=np.mean(ice[l[0]:l[ci-2]])
        else:
            mtemp[1,index]=np.inf
        if ci > 0 :
            stype[index]=1
        if ci > 0 :
            snow_albedo[index]=alpha_ice

        mtemp[0,index]=np.mean(water)

        l=np.where(snow[0,] != np.inf)
        l=l[0]

        ci=l.size
        for i in range(1,ci):
            snowthick[index]=snowthick[index]+dzs/snow[1,i]
        if ci >= 1:
            snow_albedo[index]=alpha_snow
        if ci > 1 :
            mtemp[2,index]=np.mean(snow[0,l[0]:l[ci-1]])

        else:
            mtemp[2,index]=np.inf
        if ci >= 1 :
            stype[index]=2

        hh=np.where(slush[1,] == 0)
        hh=hh[0]
        ch=hh.size
        if ch > 0 :
            slush_thick[index]=np.sum(slush[0,hh])*1.25
        else:
            slush_thick[index]=0

        load[0,index]=(ci-1)*dzs*1000+np.sum(slush[0,])*1000   # kg m-2
        a=7000000*pow(icethick[index],2)
        b=350000*pow(icethick[index],2)
        if load[0,index]*9.81 > b :
            load[1,index]=1

        # ********************************************
        # ********************************************
        # PLOT
        # one plot every day
        if np.mod(index,freq_plots)==0:
            plt.xlim(-40, 16)
            plt.ylim(-0.6, 0.5)
            plt.ylabel('Depth (m)')
            plt.xlabel('Temperature (deg C)')
            plt.axvline(x=0,color='k',linewidth=0.5).set_dashes([5])
            plt.axhline(y=0,color='k',linewidth=0.5)
            plt.title(str(df.loc[index,'Date'].date())+' Time '+str(df.loc[index,'When'].strftime("%H-%M")))

            if ice[0] != np.inf:
                co=0
            else:
                co=4

            ii=np.where(ice != np.inf)
            ii=ii[0]
            ci=ii.size
            if ci > 0:
                yi=[float(i)*dz*(-1) for i in range(0,ci)]
                xi=ice[0:ci]
                plt.plot(xi,yi,color='r',linewidth=1)
                plt.axhline(y=min(yi),color='g',linewidth=0.5)

            if ci > 0:
                offs=ii[ci-1]*dz*(-1)
            else:
                offs=0
            yw=[float(i)*dz*(-1)+offs for i in range(0,nodes[0])]
            plt.plot(water,yw,'r',linewidth=1)

            ii=np.where(snow[0,] != np.inf)
            sc=1
            ii=ii[0]
            ci=ii.size
            if ci > 1:
                ys=[float(i)*dzs*sc for i in range(0,ci)]
                for j in range(0,ci-1):
                    ys[j]=ys[j]/snow[1,ii[j]]
                ys[0]=dzs/0.5
                tt=snow[0,0:ci]
                plt.axhline(y=max(ys),color='m',linewidth=0.5)
                plt.plot(tt,ys,'r',linewidth=1)

            ii=np.where(snow[0,] == np.inf)
            ii=ii[0]
            ci=ii.size
            if ci > 0:
                slush[0,ii]=0

            ii=np.where(np.logical_and(slush[0,] != 0,slush[1,] <= 0.1))
            ii=ii[0]
            ci=ii.size
            if ci > 0:
                if ii[ci-1] >= len(ys):
                    n=ii[ci-1]-1
                else:
                    n=ii[ci-1]

            ii=np.where(np.logical_and(slush[0,] != 0,slush[1,] > 0.1))
            ii=ii[0]
            ci=ii.size
            if ci > 0:
                if ii[ci-1] >= len(ys):
                    n=ii[ci-1]-1
                else:
                    n=ii[ci-1]

            pdf.savefig()
            plt.close()

# ********************************************
# plot time series
pp = PdfPages('../data/processed/lakeice_series.pdf')

dt=[float(i)/24 for i in range(0,df.shape[0])]
dt=np.array(dt)
plt.xlim(0,index/24)
plt.ylim(-3,50)
plt.axvline(x=0,color='k',linewidth=0.5).set_dashes([5])
plt.ylabel('Ice thickness(cm)')
plt.plot(dt,icethick*100,'r',linewidth=1)
pp.savefig()
plt.clf()

# snow
plt.xlim(0,index/24)
plt.ylim(-3,max(snowthick)*100+3)
plt.ylabel('Snow depth (cm)')
plt.plot(dt,snowthick*100,'r',linewidth=1)
pp.savefig()
plt.clf()

# temp
plt.xlim(0,index/24)
ii=np.where(mtemp!=np.inf)
plt.ylim(min(mtemp[ii]),max(mtemp[0,]))
plt.ylabel('Temperature (deg C)')
ii=np.where(mtemp[2,]!=np.inf)
ii=ii[0]
plt.plot(dt[ii],mtemp[2,ii],'k',linewidth=1)
ii=np.where(mtemp[0,]!=np.inf)
ii=ii[0]
plt.plot(dt[ii],mtemp[0,ii],'r',linewidth=1)
ii=np.where(mtemp[1,]!=np.inf)
ii=ii[0]
plt.plot(dt[ii],mtemp[1,ii],linewidth=1)
pp.savefig()
plt.clf()

# albedo
plt.xlim(0,index/24)
plt.ylim(0,1)
plt.ylabel('Albedo (-)')
plt.plot(dt,snow_albedo,'r',linewidth=1)
pp.savefig()
plt.clf()
pp.close()

dfo['Icethick']=icethick
dfo['Tair']=df['Tair']

#output csv
dfo = dfo[['Date','Time','Type','Temp','Icethick']]
dfo.to_csv('../data/interim/processed_data_4.csv')
