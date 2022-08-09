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
    file = "data/verbruik_20220809.csv"
    timeunit = 60 # One hour per block
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

    plt.show()

if __name__ == "__main__":
    main()
