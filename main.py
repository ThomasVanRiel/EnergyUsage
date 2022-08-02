import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as mpe

LINEWIDTH=5
EDGEWIDTH=0
CAPSTYLE="butt"
COLORMAP="viridis_r"
ALPHA=1
FIRSTDAY=6 # 0=Mon, 6=Sun
DAYNAMES = ["Maandag", "Dinsdag", "Woensdag", "Donderdag", "Vrijdag", "Zaterdag", "Zondag"]

def main():
    file = "data/verbruik_20220802.csv"


    # Parse CSV, prepare data
    keepcols = ["Van Datum", "Van Tijdstip", "Register", "Volume"]

    df = pd.read_csv(file,sep=";",decimal=",")[keepcols]

    # Remove injections (Only parse energy consumpion, no solar panels present)
    df = df[df["Register"].isin(["Afname Dag", "Afname Nacht"])]

    # Drop NaN
    df = df.dropna()

    # Faster testing by only using first rows
    df = df[0:2000]

    SCALE = 7

    # Convert date and time to datetime
    df["Tijdstip"] = pd.to_datetime(df['Van Datum'] + ' ' + df['Van Tijdstip'], format="%d-%m-%Y %H:%M:%S")
    tinit = df["Tijdstip"].min()
    dayinit = tinit.weekday()
    tend = df["Tijdstip"].max()
    tspan = np.ceil((tend-tinit)/np.timedelta64(1,"D")/SCALE+1)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca(projection="polar")
    ax.set_rlim(bottom=-2,top=tspan)

    vmax = df["Volume"].max()

    for id, entry in df.iterrows():
        # sample normalized distance from colormap
        ndist = entry["Volume"]/vmax
        color = plt.cm.get_cmap(COLORMAP)(ndist)
        tstart = entry["Tijdstip"]-tinit
        tstop = tstart + pd.Timedelta(15,'m') # Per 15 minutes
        
        #Convert to days
        tstart = tstart/np.timedelta64(1,"D")
        tstop = tstop/np.timedelta64(1,"D")
        
        nsamples = 10
        t = np.linspace(tstart, tstop, nsamples)/SCALE
        theta = 2 * np.pi * (t)
        arc, = ax.plot(theta, t, lw=((ax.transData.transform((1,1))-ax.transData.transform((0,0))))[0]*2, color=color, solid_capstyle=CAPSTYLE, alpha=ALPHA)
        if EDGEWIDTH > 0:
            arc.set_path_effects([mpe.Stroke(linewidth=LINEWIDTH+EDGEWIDTH, foreground='black'), mpe.Normal()])

    ax.set_rticks([])
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    ax.set_xticks(np.linspace(0, 2*np.pi, 7, endpoint=False))
    ax.set_xticklabels(np.roll(DAYNAMES,-dayinit))
    
    
    plt.show()

if __name__ == "__main__":
    main()
