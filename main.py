import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
from parso import parse

LINEWIDTH=5
EDGEWIDTH=0
CAPSTYLE="butt"
COLORMAP="viridis_r"
FIRSTDAY=6 # 0=Mon, 6=Sun
DAYNAMES = ["Maandag", "Dinsdag", "Woensdag", "Donderdag", "Vrijdag", "Zaterdag", "Zondag"]
BASETIME = 15 # 15 minute intervals
SCALE = 7 # 1 week per 360 degrees

C_DAG = 57.22 + 6.66 + 1 + 1.4416 + 0.2042
C_NACHT = 40.66 + 5.03 + 1 + 1.4416 + 0.2042

def main():
    file = "data/verbruik_20221004.csv"
    timeunit = 60 # One hour per block

    df = parse_csv(file)

    #spiral(df,60)
    #line(df)
    #spectrum(df)
    #weekly(df)
    #daily(df)
    cost(df)

    plt.show()

def parse_csv(file):
    # Parse CSV, prepare data
    keepcols = ["Van Datum", "Van Tijdstip", "Register", "Volume"]

    df = pd.read_csv(file,sep=";",decimal=",")[keepcols]

    # Remove injections (Only parse energy consumpion, no solar panels present)
    df = df[df["Register"].isin(["Afname Dag", "Afname Nacht"])]
 
    # Drop NaN
    df = df.dropna()

    # Convert date and time to datetime
    df["Tijdstip"] = pd.to_datetime(df['Van Datum'] + ' ' + df['Van Tijdstip'], format="%d-%m-%Y %H:%M:%S")

    # Faster testing by only using first rows
    #df = df[0:1000]
    df = df.reset_index()

    return df


def spiral(df,timeunit=60):
    samples = timeunit/BASETIME

    # Sample dataframe
    df = pd.DataFrame({"Volume":df["Volume"].groupby(df.index//samples).sum(),"Tijdstip":df["Tijdstip"].groupby(df.index//samples).first()})

    tinit = df["Tijdstip"].min()
    dayinit = tinit.weekday()

    # Start on monday
    tinit -= np.timedelta64(dayinit,"D")
    dayinit = 0

    tend = df["Tijdstip"].max()
    tspan = np.ceil((tend-tinit)/np.timedelta64(1,"D")/SCALE+1)
    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, projection="polar")
    ax.set_rlim(bottom=-2,top=tspan)
    ax.set_facecolor('black')

    round_to = 0.5
    vmax = np.ceil(df["Volume"].max()/round_to)*round_to

    for id, entry in df.iterrows():
        # sample normalized distance from colormap
        ndist = entry["Volume"]/vmax
        color = plt.cm.get_cmap(COLORMAP)(ndist)
        tstart = entry["Tijdstip"]-tinit 
        tstop = tstart + pd.Timedelta(timeunit,'m')
        
        #Convert to days
        offset = 0.005
        tstart = tstart/np.timedelta64(1,"D")-offset
        tstop = tstop/np.timedelta64(1,"D")+offset
        
        nsamples = 10
        t = np.linspace(tstart, tstop, nsamples)/SCALE
        theta = 2 * np.pi * (t)
        #lw=((ax.transData.transform((1,1))-ax.transData.transform((0,0))))[0]*1.75
        lw=8
        arc, = ax.plot(theta, t, lw=lw, zorder=1, color=color, solid_capstyle=CAPSTYLE)
    
    ax.set_rticks(np.arange(0,tspan))
    ax.set_yticklabels((np.arange(0,tspan)+tinit.week).astype(int),fontdict={'verticalalignment': 'center', 'horizontalalignment': 'left'}, alpha=0.6, fontsize=6)
    ax.set_rlabel_position(0)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    ax.set_xticks(np.linspace(0, 2*np.pi, 7, endpoint=False))
    ax.set_xticklabels(np.roll(DAYNAMES,-dayinit))

    # Set the orientation of theta labels
    angles = -1*np.linspace(0,2*np.pi,7, endpoint=False)
    angles = np.rad2deg(angles)

    labels = []
    for label, angle in zip(ax.get_xticklabels(), angles):
        x,y = label.get_position()
        # New label on location of original
        lab = ax.text(x,y, label.get_text(), transform=label.get_transform(),
                    ha=label.get_ha(), va=label.get_va())
        lab.set_rotation(angle)
        labels.append(lab)
    ax.set_xticklabels([]) # Remove original labels

    ax.grid(False,axis='y')

    norm = mpl.colors.Normalize(vmin=0, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=COLORMAP, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ticks=np.linspace(0, vmax, 9), fraction=0.04, aspect=40, pad=0.15, label="Verbruik [kWh]", ax=ax)

def line(df):
    fig = plt.figure()
    ax = plt.subplot(111)

    tinit = df["Tijdstip"].min().to_numpy()

    t = df["Tijdstip"].to_numpy()
    v = df["Volume"].to_numpy()
    cv = np.cumsum(v)

    td = (np.subtract(t,tinit))/np.timedelta64(1, "D")

    # Weighted average per day
    w = (24*60)//BASETIME
    av = np.convolve(v, np.ones(w), 'same')/w

    ax.plot(t,v/0.25,alpha=0.5)
    ax.plot(t,av/0.25,color='C0')
    ax.set_ylabel('Gemiddeld vermogen [kW]')

    ax2 = ax.twinx()
    ax2.plot(t,cv,color='C1')
    ax2.set_ylabel('Verbruik [kWh]')

    months = mdates.MonthLocator(interval=1)
    months_fmt = mdates.DateFormatter('%b')
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(months_fmt)

    days = mdates.DayLocator(interval=7)
    ax.xaxis.set_minor_locator(days)

def spectrum(df):
    dt = 1/24/60*15
    Fs = 1/dt
    fig, ax = plt.subplots()
    ax.magnitude_spectrum(df["Volume"], Fs=Fs)
    ax.set_xscale('log')

def weekly(df):
    samples = np.timedelta64(1,"W")/np.timedelta64(1,"m")
    print(samples)

    df = pd.DataFrame({"Volume":df["Volume"].groupby(df["Tijdstip"].dt.isocalendar().week).sum()})
    fig,ax = plt.subplots()
    ax.bar(df.index,df["Volume"])
    ax.set_xlabel("Week")
    ax.set_ylabel("Verbruik [kWh]")
    ax.grid(axis='y')    

def daily(df):
    df = pd.DataFrame({"Volume":df["Volume"].groupby(df["Tijdstip"].dt.isocalendar().week+df["Tijdstip"].dt.isocalendar().day/7).sum()})
    fig,ax = plt.subplots()
    ax.bar(df.index,df["Volume"],width=1/7)
    ax.set_xlabel("Week")
    ax.set_ylabel("Verbruik [kWh]")
    ax.grid(axis='y')
    print("Gemiddeld dagelijks verbruik: {:.3f} kWh".format(np.mean(df.Volume)))

def cost(df):
    #df_dag = pd.DataFrame({"Volume":df["Volume"].groupby(df["Register"])})
    df = pd.DataFrame({"Dag":df.query("Register == 'Afname Dag'")["Volume"].groupby(df["Tijdstip"].dt.floor("d")).sum()*C_DAG,"Nacht":df.query("Register == 'Afname Nacht'")["Volume"].groupby(df["Tijdstip"].dt.floor("d")).sum()*C_NACHT})
    df["Totaal"] = df.sum(axis=1)
    
    fig,ax = plt.subplots()
    ax.bar(df.index,df["Totaal"]/100)
    ax.set_xlabel("Dag")
    ax.set_ylabel("Kost [€]")
    ax.grid(axis='y')
    print("Gemiddelde dagelijkse kost: €{:.2f}".format(np.mean(df.Totaal/100)))

if __name__ == "__main__":
    main()
