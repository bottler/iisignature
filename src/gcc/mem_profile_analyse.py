import sqlite3

con = sqlite3.connect("mem_profile.sqlite")

if 0:
    #Check if there are duplicates
    q="select count(*) as AA from A group by method,basis,d,m having AA>1"
    print (con.execute(q).fetchall())

    q="select count(*)  from A"
    print (con.execute(q).fetchall())

    q="select *  from A order by method,basis,d,m"
    all=con.execute(q).fetchall()
    i=0
    for line in all:
        i+=1
        print (line)
    print(i)
    exit(0)


import tabulate, sqlite3, numpy
import numpy as np
#import grapher
#https://www.bastibl.net/publication-quality-plots/
import matplotlib as mpl
save=1
if save:
    mpl.use("pdf")
import matplotlib.pyplot as plt
useLibertine = True
plt.rc('text', usetex=True)
if useLibertine:
    pre="""
\usepackage[T1]{fontenc}
\usepackage[tt=false,type1=true]{libertine}
%\setmonofont{inconsolata}
\usepackage[varqu]{zi4}
\usepackage[libertine]{newtxmath}
"""
    plt.rcParams['text.latex.preamble'] = pre #'\usepackage{libertine},\usepackage[libertine]{newtxmath},\usepackage{sfmath},\usepackage[T1]{fontenc}'
else:
    plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'monospace' : ['Computer Modern']})
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rc('legend', fontsize=8)
plt.rc('axes', labelsize=8)
mylinewidth=0.6
plt.rc('axes',linewidth=mylinewidth)#default 1

width = 3.487
height = width / 1.618
fig, ax = plt.subplots()
fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.97)

a=con.execute("select BYTES, D,M,METHOD,BASIS from A").fetchall()
b=[i for i in a]
bch_extra=0
def get_basic(d,m,method,basis):
    found=False
    for i in b:
        if (d,m,method,basis)==(i[1],i[2],i[3],i[4]):
            if found:
                if usage != i[0]:
                    raise RuntimeError("two inconsistent copies of "+str((d,m,method)))
            else:
                found=True
                usage=i[0]
    if not found:
        raise RuntimeError("no copies of "+str((d,m,method)))
    return usage
#bch_extra=get_basic(0,0,"C","L")
def get(d,m,method,basis):
    usage = get_basic(d,m,method,basis)
    if 0:
        if "C" == method:
            return usage-get_basic(0,0,"C","L")
        return usage
    return usage-get_basic(d,1,method,basis)

#print (tabulate.tabulate(b,headers=["method","d","m", "reps","time"]))

Order = ["C","S"]
d=3
max_m={2:11,3:10,4:6,5:6}[d]
x=list(range(2,1+max_m))
y=[[get(d,m,method,"L") for m in x] for method in Order]
#series = [i for j in [[x,k,"+"] for k in y] for i in j]
#plt.plot(*series)

#d,x,o,8 and all the triangles look bigger , 8 and o look identical
#.,+,* look smaller
#These are the symbols and colours for the timings
#graph which uses the order C/O/S/esig
#so we just need elements 0 and 2
symbols=["v","o","x","d"]
colours=["r","b","g","k"]
symbols=(symbols[0],symbols[2])
colours=(colours[0],colours[2])
for i,nm,symbol,col in zip(y,Order,symbols,colours):
    #plt.plot(x,i,symbol,label=nm)
    nm1=r"\verb|"+nm+"|"
    plt.scatter(x,i,c=col,marker=symbol,label=nm1,edgecolors='none')
#prop=mpl.font_manager.FontProperties("monospace")
legend=plt.legend(loc="upper left")
legend.get_frame().set_linewidth(mylinewidth)
plt.xlabel('level')
plt.yscale('log')
#plt.xlim(1.5,max_m+0.5)
plt.ylabel('usage(bytes) - logarithmic scale')
#grapher.pushplot(plt)
#plt.draw()
#dpi doesn't change the scale in the picture
filename = "/home/jeremyr/Dropbox/phd/graphs/memsweep"+str(d)+"d"+("Lib" if useLibertine else "")

if save:
    fig.set_size_inches(width,height)
    plt.savefig(filename+".pdf")
else:
    plt.show()

#run this to check that the graph looks good in monochrome
if 0:
    #plt.savefig(filename+".png",dpi=300)
    from PIL import Image
    img = Image.open(filename+".png").convert('LA')
    img.save(filename+"bw.png")

    
