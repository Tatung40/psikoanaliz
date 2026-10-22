"""Deterministic B13 300 dpi PNGs, server-side matplotlib, no browser rendering."""
from io import BytesIO
import numpy as np
from core.irt_b13_curves import compute_curves

KINDS=("ICC","IIC","TIF","SEM","Theta dağılımı","Wright haritası","Parametre dağılımı","Asimptotlar")

def png_bytes(result,kind,selected=None):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    if kind not in KINDS:raise ValueError("Unknown chart")
    it=result["items"];cv=compute_curves(it)
    names=it["Madde"].astype(str).tolist()
    fig=Figure(figsize=(9,5.5),layout="constrained")
    ax=fig.add_subplot(111)
    if kind in ("ICC","IIC"):
        for nm in (selected or names[:8]):
            if nm not in names:raise ValueError("Unknown item")
            j=names.index(nm)
            ax.plot(cv["theta"],cv[kind.lower()][:,j],label=nm)
        ax.legend(fontsize=7,ncol=2);ax.set_xlabel("Theta")
        ax.set_ylabel("P(X=1|theta)" if kind=="ICC" else "Information")
    elif kind=="TIF":
        ax.plot(cv["theta"],cv["tif"]);ax.set_xlabel("Theta");ax.set_ylabel("Test information")
    elif kind=="SEM":
        ax.plot(cv["theta"],cv["sem"]);ax.set_xlabel("Theta");ax.set_ylabel("Conditional SE")
    elif kind=="Theta dağılımı":
        theta=np.asarray(result["persons"]["Theta"],float)
        ax.hist(theta[np.isfinite(theta)],bins=30)
        ax.set_xlabel("Theta");ax.set_ylabel("Persons")
    elif kind=="Wright haritası":
        theta=np.asarray(result["persons"]["Theta"],float)
        ax.hist(theta[np.isfinite(theta)],bins=30,orientation="horizontal")
        ax.set_xlabel("Persons");ax.set_ylabel("Common theta / b scale")
        for i,(nm,b) in enumerate(zip(names,it["b"])):
            ax.annotate(nm,xy=(0,float(b)),xytext=(-8-(i%3)*10,0),
                textcoords="offset points",fontsize=6,ha="right")
    elif kind=="Parametre dağılımı":
        free={"Rasch":"b","1PL":"b","2PL":"ab","3PL":"abc","4PL":"abcd"}[result["model"]]
        ax.boxplot([np.asarray(it[p],float) for p in free])
        ax.set_xticks(range(1,len(free)+1),list(free))
    elif kind=="Asimptotlar":
        if result["model"] not in ("3PL","4PL"):raise ValueError("Asymptotes only for 3PL/4PL")
        ax.bar(names,it["c"],label="c")
        if result["model"]=="4PL":ax.scatter(names,it["d"],label="d")
        ax.legend()
    ax.set_title(f"{result['model']} — {kind}")
    stream=BytesIO();fig.savefig(stream,format="png",dpi=300)
    return stream.getvalue()
