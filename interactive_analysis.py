import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor

import functions


# class which acts as state machine to collect click and key information
class clicktracker:
    def __init__(self,fig_ref,RR=None,data_in=None):
        self.RR=RR
        self.fig=fig_ref
        self.data={'P':[],'Q':[],'R':[[0,0]],'S':[],'T':[]}
        self.data_len = {'P': 3, 'Q': 1, 'R': 1, 'S': 1, 'T': 3, 'Idle': 0}
        self.color_dict = {'P': 'r', 'Q': 'g', 'R': 'k', 'S':'b' , 'T': 'm', 'Idle': 0}
        for key in self.data:
            self.data[key] = [[np.nan,np.nan]] * self.data_len[key]
        self.data['R']=[[0,0]]
        self.state='Idle'
        self.resetcounter=0
        if not data_in==None:
            self.data['P'][0][0] = float(data_in[1])
            self.data['P'][0][1] = float(data_in[2])
            self.data['P'][1][0] = float(data_in[3])
            self.data['P'][1][1] = float(data_in[4])
            self.data['P'][2][0] = float(data_in[5])
            self.data['P'][2][1] = float(data_in[6])
            self.data['Q'][0][0] = float(data_in[7])
            self.data['Q'][0][1] = float(data_in[8])
            self.data['R'][0][0] = float(data_in[9])
            self.data['R'][0][1] = float(data_in[10])
            self.data['S'][0][0] = float(data_in[11])
            self.data['S'][0][1] = float(data_in[12])
            self.data['T'][0][0] = float(data_in[13])
            self.data['T'][0][1] = float(data_in[14])
            self.data['T'][1][0] = float(data_in[15])
            self.data['T'][1][1] = float(data_in[16])
            self.data['T'][2][0] = float(data_in[17])
            self.data['T'][2][1] = float(data_in[18])
        #print(self.data)
        #draw all lines and text in default options
        ax=self.fig
        self.plot_ref_dict={}
        self.plot_ref_dictb = {}
        for key in self.data:
            self.plot_ref_dict[key]=[ax.axvline(x=0,linestyle='--',color=self.color_dict[key],alpha=0.5) for j in range(self.data_len[key])]
            self.plot_ref_dictb[key]=[]
            for j in range(self.data_len[key]):
                poin,=ax.plot(0,0,color=self.color_dict[key],marker='x')
                self.plot_ref_dictb[key].append(poin)
        self.key_pressed(['alt+d'])


    def key_pressed(self,value):
        key2state_dict={'alt+p':'P',
                        'alt+t':'T',
                        'alt+q':'Q',
                        'alt+s':'S',
                        'alt+r':'R'
                        }
        if value[0] in key2state_dict:
            self.state=key2state_dict[value[0]]
            self.data[self.state]=[np.nan]*self.data_len[self.state]
            self.resetcounter=0
        elif value[0]=='alt+d':
            print(self.data['P'])
            vaids={}
            vaids['P'] = [self.data['P'][0][0], self.data['P'][2][0]]
            vaids['PR']=[self.data['P'][0][0], self.data['Q'][0][0]]
            vaids['QRS'] = [self.data['Q'][0][0], self.data['S'][0][0]]
            vaids['QT'] = [self.data['Q'][0][0], self.data['T'][2][0]]
            for j,combo in enumerate(vaids):
                try:
                    label=combo+': %d' %(int(vaids[combo][1]-vaids[combo][0]))
                    if combo=='QT':
                        label+=',RR: %d, QTc: %d' % (int(self.RR),int((vaids[combo][1]-vaids[combo][0])/np.sqrt(self.RR/1000)))
                    draw_intervall_marker(self.fig,label,
                                          vaids[combo][0],
                                          vaids[combo][1],
                                          -0.01-0.003*j)
                except:
                    pass

    def onclick(self,value):
        if self.resetcounter<self.data_len[self.state]:
            self.data[self.state][self.resetcounter]=value
            self.plot_ref_dict[self.state][self.resetcounter].set_xdata(value[0])
            self.plot_ref_dictb[self.state][self.resetcounter].set_data(value)
            self.resetcounter+=1


# Plotting
def interactive_ploter_old(container,segments):
    ##### create the base plot:
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(18 / 2.54, 8 / 2.54), dpi=150)
    plt.suptitle('3 clicks: alt+P, 1 click alt+Q,1 click alt+r, 1 click alt+S, 3 clicks alt+T')
    ax.grid(alpha=0.7)
    ax.minorticks_on()
    ax.grid(True, which='minor', linestyle='dotted', alpha=0.7)
    ax.set_ylabel("Signal [A.U.]")
    ax.set_xlabel("Time [s]")

    x = np.array(segments[str(1)].axes[0])
    for j in range(np.shape(container)[0]):
        ax.plot(x, container[j,:], color='b', alpha=0.05)
    # Defining the cursor
    cursor = Cursor(ax, horizOn=True, vertOn=True, useblit=True,
                    color='r', linewidth=1)

    cti=clicktracker(ax)

    def onclick(event):
        #global cti
        cti.onclick([event.xdata, event.ydata])
        fig.canvas.draw()

    def onkey(event):
        #global cti
        cti.key_pressed([event.key, event.xdata, event.ydata])
        fig.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('key_press_event', onkey)

    #print(cti.plot_ref_dict['P'])
    plt.draw()
    plt.pause(20)
    #fig.savefig(dir)
    return cti.data

def draw_intervall_marker(axes_obj,name,x1,x2,y):
    axes_obj.arrow(x1+0.5*(x2-x1),y,0.5*(x2-x1),0,width=0.0005,shape='full',
              length_includes_head=True,color='k',linewidth=0,alpha=0.6)
    axes_obj.arrow(x2+0.5*(x1-x2),y,0.5*(x1-x2),0,width=0.0005,shape='full',
              length_includes_head=True,color='k',linewidth=0,alpha=0.6)
    axes_obj.vlines(x1,min(0,y),max(0,y),linestyle='dashed',color='k',alpha=0.5)
    axes_obj.vlines(x2,min(0,y),max(0,y),linestyle='dashed',color='k',alpha=0.5)
    axes_obj.text(0.5*(x1+x2),y+.002,name)
def interactive_ploter(container,time_axis,add_lines=None,sideplot=None,savename=None, pre_load=None):
    ##### create the base plot:
    fig, (ax,bx) = plt.subplots(1, 2, figsize=(30 / 2.54, 18 / 2.54),
    gridspec_kw={'width_ratios': [3, 1]}, dpi=120)
    suptitle='3 clicks: alt+P, 1 click alt+Q,1 click alt+r, 1 click alt+S, 3 clicks alt+T, alt+D when done'
    
    plt.suptitle(suptitle,y=0.98)
    ax.grid(alpha=0.7)
    ax.minorticks_on()
    ax.grid(True, which='minor', linestyle='dotted', alpha=0.7)
    if sideplot!=None:
        xx, yy, popt_RRs, RR, sigma_RR,hos=sideplot
        bx.grid(alpha=0.7)
        bx.minorticks_on()
        bx.grid(True, which='minor', linestyle='dotted', alpha=0.7)
        bx.plot(xx,yy)
        bx.plot(xx,functions.gaus(xx,popt_RRs[0],popt_RRs[1],popt_RRs[2]))
        secaxb = bx.secondary_xaxis('top', functions=(functions.BPM2RR, functions.RR2BPM))
        secaxb.set_xlabel('BPM')
        secaxb.set_xticks(ticks=[200,150,120,100,80,75,60])
        bx.axvline(x=RR, color='r',linestyle='dashed')
        bx.text(RR*1.05,np.max(yy)*0.9,'RR: %i $\pm$ %i'% (int(RR),int(sigma_RR)),color='r')
        bx.set_xlabel('RR [ms]')
        bx.set_xlim([300,1000])
        bx.text(700,np.max(yy)*0.85,'# Peaks: %i' % (int(np.sum(yy))))
        bx.text(700, np.max(yy) * 0.8, '# HOS: %f' % (float(hos)))
    ax.set_ylabel("Signal [A.U.]")
    ax.set_xlabel("Time [ms]")
    fig.tight_layout()


    x = time_axis
    container=container.T
    for j in range(np.shape(container)[0]):
        ax.plot(x, container[j,:], color='b', alpha=0.05, lw=1)
    if add_lines!=None:
        #try:
        ax.plot(x, add_lines[0], color='r',label='AVGs')
        ax.plot(x, add_lines[1]/np.std(add_lines[1])*np.std(add_lines[0]),
                color='k',label='Bispectral Filters')
        if len(add_lines)==3:
            ax.plot(x,add_lines[2],color='g',label='theory')
        ax.legend(loc=1)
        #except:
         #   print('here')
          #  pass

    # Defining the cursor
    cursor = Cursor(ax, horizOn=True, vertOn=True, useblit=True,
                    color='r', linewidth=1)

    cti=clicktracker(ax,RR=RR,data_in=pre_load)

    def onclick(event):
        #global cti
        cti.onclick([event.xdata, event.ydata])
        fig.canvas.draw()

    def onkey(event):
        #global cti
        cti.key_pressed([event.key, event.xdata, event.ydata])
        fig.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('key_press_event', onkey)
# while True:
    plt.draw()
    plt.pause(200)

    if savename!=None:
        ppath,pic_name=savename
        fig.savefig(ppath+'/'+pic_name+'__timeing_eval.png')
    plt.close()
    return cti.data