import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator,AutoMinorLocator

LINEWIDTH=5
EDGEWIDTH=0
CAPSTYLE="butt"
COLORMAP="viridis_r"
FIRSTDAY=6 # 0=Mon, 6=Sun
DAYNAMES = ["Maandag", "Dinsdag", "Woensdag", "Donderdag", "Vrijdag", "Zaterdag", "Zondag"]
BASETIME = 15 # 15 minute intervals
SCALE = 7 # 1 week per 360 degrees


def main():
    file = "data/verbruik_20220802.csv"
    timeunit = 60
    samples = timeunit/BASETIME

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
    df_sampled = pd.DataFrame({"Volume":df["Volume"].groupby(df.index//samples).sum(),"Tijdstip":df["Tijdstip"].groupby(df.index//samples).first()})

    df = df_sampled

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

    vmax = df["Volume"].max()

    for id, entry in df.iterrows():
        # sample normalized distance from colormap
        ndist = entry["Volume"]/vmax
        color = plt.cm.get_cmap(COLORMAP)(ndist)
        tstart = entry["Tijdstip"]-tinit 
        tstop = tstart + pd.Timedelta(timeunit,'m') # Per 15 minutes
        
        #Convert to days
        tstart = tstart/np.timedelta64(1,"D")
        tstop = tstop/np.timedelta64(1,"D")
        
        nsamples = 10
        t = np.linspace(tstart, tstop, nsamples)/SCALE
        theta = 2 * np.pi * (t)
        arc, = ax.plot(theta, t, lw=((ax.transData.transform((1,1))-ax.transData.transform((0,0))))[0]*1.5, zorder=1, color=color, solid_capstyle=CAPSTYLE)
    
    #ax.set_rticks([])
    ax.set_rticks(np.arange(0,tspan))
    ax.set_yticklabels((np.arange(0,tspan)+tinit.week).astype(int),fontdict={'verticalalignment': 'center', 'horizontalalignment': 'center'}, alpha=0.2)
    #ax.set_rticks(np.linspace(tinit.week))
    ax.set_rlabel_position(0)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    ax.set_xticks(np.linspace(0, 2*np.pi, 7, endpoint=False))
    ax.set_xticklabels(np.roll(DAYNAMES,-dayinit))

    angles = -1*np.linspace(0,2*np.pi,7, endpoint=False)
    angles = np.rad2deg(angles)

    labels = []
    for label, angle in zip(ax.get_xticklabels(), angles):
        x,y = label.get_position()
        lab = ax.text(x,y, label.get_text(), transform=label.get_transform(),
                    ha=label.get_ha(), va=label.get_va())
        lab.set_rotation(angle)
        labels.append(lab)
    ax.set_xticklabels([])

    ax.grid(False,axis='y')

    norm = mpl.colors.Normalize(vmin=0, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=COLORMAP, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ticks=np.linspace(0, vmax, 8), fraction=0.04, aspect=60, pad=0.15, label="Verbruik [kWh]", ax=ax)

    plt.show()

if __name__ == "__main__":
    main()
