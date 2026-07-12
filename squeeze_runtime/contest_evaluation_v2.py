"""Official ICDAR 2026 TROGS contest evaluation, v2 (Nicholas Howe, CC-BY 4.0).

Verbatim third-party code downloaded from the TROGS-26 Test Images release
(Smith ScholarWorks, published 2026-03-27, alongside the 100 test annotation
files). Supersedes contest_evaluation.py for official scoring. Differences
from v1: annotation files gain a per-box confidence section (union of the
two rotations); evaluate() reports three scores -- `cer` against confident
letters only, `all_cer` against all letters (the v1-style score), and
`opt_cer` via masked_cer(), where skipping an obscured letter is free, a
correct guess on one grows the denominator, and a wrong guess is penalised;
and new_convert() fixes the Upsilon/Chi/Psi proxy mapping (old: U/Y/C,
new: Y/C/U), with latin_convert() + best-of-either-converter as amnesty.

The active pipeline uses this module only for official-consistent scoring of
finished prod runs (see squeeze_runtime/official_scoring.py and the
`official_v2` block in prod summary.json). Staging still parses annotations
with contest_evaluation.readBoxFile (v1), which keeps every letter -- note
that THIS module's readBoxFile drops low-confidence boxes by default
(all_conf=False), and its getRowBoxes activates two neighbor-mismatch resets
that v1 left commented out, so switching the staging parser would silently
change what gets transcribed. Two more scoring facts verified empirically:
the v2 reference is '\n'.join(rows) with NO trailing newline (v1's ended
with one), so a transcript file ending in '\n' scores +1 insertion per
squeeze; and under the confident-only `cer` every character emitted for an
obscured box scores as an insertion even when correct.
"""
# -*- coding: utf-8 -*-
"""
Code released in conjunction with the 2026 ICDAR Competition in
Text Recognition on Greek Squeezes (TROGS-26)

Call run_evaluations(pred_dir,gt_dir) to evaluate.
pred_dir should contain your method's transcription files
    Filenames should be {squeeze}_transcript.txt
    Contents are line by line character transcriptions
gt_dir should contain letter box files with ground truth

@author: Nicholas Howe  CC-BY 4.0
"""

import math
import os
from os import listdir
from os.path import isfile, join, split
import numpy as np
from textdistance import levenshtein
import matplotlib.pyplot as plt
from itertools import islice

class Box:
    '''
    # simple class to keep track of coordinates for the four corners of a box
    # note corner order is CCW from NW corner, as stored in letter box file
    '''
    def __init__(self,xnw,ynw,xsw,ysw,xse,yse,xne,yne):
        self.xnw = xnw
        self.ynw = ynw
        self.xsw = xsw
        self.ysw = ysw
        self.xse = xse
        self.yse = yse
        self.xne = xne
        self.yne = yne
        
    def bd(self):
        return [self.xnw,self.ynw,self.xsw,self.ysw,self.xse,self.yse,self.xne,self.yne]

    def center(self):
        return((self.xnw+self.xne+self.xse+self.xsw)/4,(self.ynw+self.yne+self.yse+self.ysw)/4)
    
    def radius(self):
        return (math.sqrt((self.xnw-self.xse)*(self.xnw-self.xse)+(self.ynw-self.yse)*(self.ynw-self.yse))+math.sqrt((self.xne-self.xsw)*(self.xne-self.xsw)+(self.yne-self.ysw)*(self.yne-self.ysw)))/4
    
    def area(self):
        # uses top and left side, assuming orthogonal rectangle
        return math.sqrt(((self.xne-self.xnw)**2+(self.yne-self.ynw)**2)*((self.xsw-self.xnw)**2+(self.ysw-self.ynw)**2))

    def angle(self):
        return math.atan2(self.yne-self.ynw,self.xne-self.xnw)

    def dist(self,box):
        c = self.center()
        c2 = box.center()
        return math.sqrt((c[0]-c2[0])*(c[0]-c2[0])+(c[1]-c2[1])*(c[1]-c2[1]))

    def plot(self,clr='r-'):
        plt.plot([self.xnw,self.xne,self.xse,self.xsw,self.xnw],[self.ynw,self.yne,self.yse,self.ysw,self.ynw],clr)
        
      
# This converter accounts for teams that may have used the wrong conversion on their own
def latin_convert(stringin):
    eqv = {
        89:"C",
        67:"U",
        85:"Y",
        121:"c",
        99:"u",
        117:"y", # above accounts for possible conversion by entrants
        }        
    txt = []
    for char in stringin:
        if ord(char) in eqv:
            txt.append(eqv[ord(char)])
        else:
            txt.append(char)               
    return txt
      

# This converter has some bugs in it, but is included here because some contestants relied on it for their entries
# We took the best score found using either converter
def old_convert(stringin):
    """
    Convert Greek characters to letter proxies.

    Parameters
    ----------
    stringin : String
        Character string possibly containing Greek.

    Returns
    -------
    txt : String
        Converted string with Greek characters replaced by Latin proxy.

    """
    eqv = {
        913:"A",  # ALPHA
        7944:"A", # ALPHA + AIGU
        914:"B",  # BETA
        915:"G",  # GAMMA
        916:"D",  # DELTA
        917:"E",  # EPSILON
        918:"Z",  # ZETA
        919:"H",  # ETA
        920:"Q",  # THETA
        921:"I",  # IOTA
        922:"K",  # KAPPA
        923:"L",  # LAMDA
        924:"M",  # MU
        925:"N",  # NU
        926:"X",  # XI
        927:"O",  # OMICRON
        928:"P",  # PI
        929:"R",  # PI
        931:"S",  # SIGMA
        932:"T",  # TAU
        933:"U",  # UPSILON
        934:"F",  # PHI
        935:"Y",  # PSI
        936:"C",  # CHI
        937:"W",  # OMEGA
        945:"a",  # ALPHA
        8118:"a",  # ALPHA + TILDE
        8119:"a",  # ALPHA + TILDE + CEDILLE
        7942:"a",  # ALPHA + TILDE + TICK
        7943:"a",  # ALPHA + TILDE + BACKTICK
        8070:"a",  # ALPHA + TILDE + TICK + CEDILLE
        8071:"a",  # ALPHA + TILDE + BACKTICK + CEDILLE
        8048:"a",  # ALPHA + GRAVE
        8049:"a",  # ALPHA + AIGU
        7936:"a",  # ALPHA + TICK
        7937:"a",  # ALPHA +BACKTICK
        7940:"a",  # ALPHA + TICK + AIGU
        7941:"a",  # ALPHA + BACKTICK + AIGU
        7938:"a",  # ALPHA + TICK + GRAVE
        7939:"a",  # ALPHA + BACKTICK + GRAVE
        946:"b",  # BETA
        947:"g",  # GAMMA
        948:"d",  # DELTA
        949:"e",  # EPSILON
        7952:"e",  # EPSILON + TICK
        7953:"e",  # EPSILON + BACKTICK
        8050:"e",  # EPSILON + GRAVE
        8051:"e",  # EPSILON + AIGU
        7956:"e",  # EPSILON + TICK + AIGU
        7957:"e",  # EPSILON + BACKTICK + AIGU
        7954:"e",  # EPSILON + TICK + GRAVE
        7955:"e",  # EPSILON + BACKTICK + GRAVE
        950:"z",  # ZETA
        951:"h",  # ETA
        7974:"h",  # ETA + TILDE + TICK + CEDILLE
        7975:"h",  # ETA + TILDE + BACKTICK + CEDILLE
        8086:"h",  # ETA + TILDE + TICK
        8087:"h",  # ETA + TILDE + BACKTICK
        8134:"h",  # ETA + TILDE
        8135:"h",  # ETA + TILDE + CEDILLE
        7970:"h",  # ETA + TICK + GRAVE
        7971:"h",  # ETA + BACKTICK + GRAVE
        8016:"h",  # ETA + TICK?
        8020:"h",  # ETA + TICK + AIGU
        7968:"h",  # ETA + TICK
        7969:"h",  # ETA + BACKTICK
        7972:"h",  # ETA + TICK + AIGU
        7973:"h",  # ETA + BACKTICK + AIGU
        8052:"h",  # ETA + GRAVE
        8053:"h",  # ETA + AIGU ?
        952:"q",  # THETA
        953:"i",  # IOTA
        7990:"i",  # IOTA + TILDE + TICK
        7991:"i",  # IOTA + TILDE + BACKTICK
        8150:"i",  # IOTA + TILDE
        8054:"i",  # IOTA + GRAVE
        8055:"i",  # IOTA + AIGU
        7984:"i",  # IOTA + TICK
        7985:"i",  # IOTA + BACKTICK
        7988:"i",  # IOTA + TICK + AIGU
        7989:"i",  # IOTA + BACKTICK + AIGU
        7986:"i",  # IOTA + TICK + GRAVE
        7987:"i",  # IOTA + BACKTICK + GRAVE
        954:"k",  # KAPPA
        955:"l",  # LAMDA
        956:"m",  # MU
        957:"n",  # NU
        834:"n",  # NU + TILDE
        958:"x",  # XI
        959:"o",  # OMICRON
        8056:"o",  # OMICRON + GRAVE
        8057:"o",  # OMICRON + AIGU
        8000:"o",  # OMICRON + TICK
        8001:"o",  # OMICRON + BACKTICK
        8004:"o",  # OMICRON + TICK + AIGU
        8005:"o",  # OMICRON + BACKTICK + AIGU
        8002:"o",  # OMICRON + TICK + GRAVE
        8003:"o",  # OMICRON + BACKTICK + GRAVE
        960:"p",  # PI
        961:"r",  # RHO
        962:"v",  # SIGMA
        963:"s",  # SIGMA
        964:"t",  # TAU
        965:"u",  # UPSILON
        8022:"u",  # UPSILON + TILDE + TICK
        8023:"u",  # UPSILON + TILDE + BACKTICK
        8166:"u",  # UPSILON + TILDE
        8016:"u",  # UPSILON + TICK
        8017:"u",  # UPSILON + BACKTICK
        8020:"u",  # UPSILON + TICK + AIGU
        8021:"u",  # UPSILON + BACKTICK + AIGU
        8018:"u",  # UPSILON + TICK + GRAVE
        8019:"u",  # UPSILON + BACKTICK + GRAVE
        8058:"u",  # UPSILON + GRAVE
        8059:"u",  # UPSILON + AIGU ?
        966:"f",  # PHI
        967:"y",  # PSI
        968:"c",  # CHI
        969:"w",  # OMEGA
        8102:"w",  # OMEGA + TILDE + TICK + CEDILLE
        8103:"w",  # OMEGA + TILDE + BACKTICK + CEDILLE
        8038:"w",  # OMEGA + TILDE + TICK
        8039:"w",  # OMEGA + TILDE + BACKTICK
        8182:"w",  # OMEGA + TILDE
        8183:"w",  # OMEGA + TILDE + CEDILLE
        8060:"w",  # OMEGA + GRAVE
        8061:"w",  # OMEGA + AIGU
        8032:"w",  # OMEGA + TICK
        8033:"w",  # OMEGA + BACKTICK
        8036:"w",  # OMEGA + TICK + AIGU
        8037:"w",  # OMEGA + BACKTICK + AIGU
        8034:"w",  # OMEGA + TICK + GRAVE
        8035:"w",  # OMEGA + BACKTICK + GRAVE
    
        8217:"'",  # TICK???
        8025:"`",  # BACKTICK???
        803:"_",  # CEDILLE???
        8195:" ",  # WEIRD SPACE???
        903:".",  # WEIRD DOT???
        8228:".",  # WEIRD DOT???
        8311:"7",  # SUPERSCRIPT 7
    
        # Formatting characters
        10:"\n",  # CARRIAGE RETURN
        3:" ",  # SPACE
        32:" "  # SPACE
    }
    for i in range(0, 254):
        eqv[i] = chr(i)
            
    txt = []
    for char in stringin:
        if ord(char) in eqv:
            txt.append(eqv[ord(char)])
        else:
            txt.append(char)               
    txt = "".join(txt)
    return txt.upper()


def new_convert(stringin):
    """
    Convert Greek characters to letter proxies.

    Parameters
    ----------
    stringin : String
        Character string possibly containing Greek.

    Returns
    -------
    txt : String
        Converted string with Greek characters replaced by Latin proxy.

    """
    eqv = {}
    for i in range(0, 254):
        eqv[i] = chr(i)
    eqv.update({
        913:"A",  # ALPHA
        7944:"A", # ALPHA + AIGU
        914:"B",  # BETA
        915:"G",  # GAMMA
        916:"D",  # DELTA
        917:"E",  # EPSILON
        918:"Z",  # ZETA
        919:"H",  # ETA
        920:"Q",  # THETA
        921:"I",  # IOTA
        922:"K",  # KAPPA
        923:"L",  # LAMDA
        924:"M",  # MU
        925:"N",  # NU
        926:"X",  # XI
        927:"O",  # OMICRON
        928:"P",  # PI
        929:"R",  # RHO
        931:"S",  # SIGMA
        932:"T",  # TAU
        933:"Y",  # UPSILON
        934:"F",  # PHI
        935:"C",  # CHI
        936:"U",  # PSI
        937:"W",  # OMEGA
        945:"a",  # ALPHA
        8118:"a",  # ALPHA + TILDE
        8119:"a",  # ALPHA + TILDE + CEDILLE
        7942:"a",  # ALPHA + TILDE + TICK
        7943:"a",  # ALPHA + TILDE + BACKTICK
        8070:"a",  # ALPHA + TILDE + TICK + CEDILLE
        8071:"a",  # ALPHA + TILDE + BACKTICK + CEDILLE
        8048:"a",  # ALPHA + GRAVE
        8049:"a",  # ALPHA + AIGU
        7936:"a",  # ALPHA + TICK
        7937:"a",  # ALPHA +BACKTICK
        7940:"a",  # ALPHA + TICK + AIGU
        7941:"a",  # ALPHA + BACKTICK + AIGU
        7938:"a",  # ALPHA + TICK + GRAVE
        7939:"a",  # ALPHA + BACKTICK + GRAVE
        946:"b",  # BETA
        947:"g",  # GAMMA
        948:"d",  # DELTA
        949:"e",  # EPSILON
        7952:"e",  # EPSILON + TICK
        7953:"e",  # EPSILON + BACKTICK
        8050:"e",  # EPSILON + GRAVE
        8051:"e",  # EPSILON + AIGU
        7956:"e",  # EPSILON + TICK + AIGU
        7957:"e",  # EPSILON + BACKTICK + AIGU
        7954:"e",  # EPSILON + TICK + GRAVE
        7955:"e",  # EPSILON + BACKTICK + GRAVE
        950:"z",  # ZETA
        951:"h",  # ETA
        7974:"h",  # ETA + TILDE + TICK + CEDILLE
        7975:"h",  # ETA + TILDE + BACKTICK + CEDILLE
        8086:"h",  # ETA + TILDE + TICK
        8087:"h",  # ETA + TILDE + BACKTICK
        8134:"h",  # ETA + TILDE
        8135:"h",  # ETA + TILDE + CEDILLE
        7970:"h",  # ETA + TICK + GRAVE
        7971:"h",  # ETA + BACKTICK + GRAVE
        7968:"h",  # ETA + TICK
        7969:"h",  # ETA + BACKTICK
        7972:"h",  # ETA + TICK + AIGU
        7973:"h",  # ETA + BACKTICK + AIGU
        8052:"h",  # ETA + GRAVE
        8053:"h",  # ETA + AIGU ?
        952:"q",  # THETA
        953:"i",  # IOTA
        7990:"i",  # IOTA + TILDE + TICK
        7991:"i",  # IOTA + TILDE + BACKTICK
        8150:"i",  # IOTA + TILDE
        8054:"i",  # IOTA + GRAVE
        8055:"i",  # IOTA + AIGU
        7984:"i",  # IOTA + TICK
        7985:"i",  # IOTA + BACKTICK
        7988:"i",  # IOTA + TICK + AIGU
        7989:"i",  # IOTA + BACKTICK + AIGU
        7986:"i",  # IOTA + TICK + GRAVE
        7987:"i",  # IOTA + BACKTICK + GRAVE
        954:"k",  # KAPPA
        955:"l",  # LAMDA
        956:"m",  # MU
        957:"n",  # NU
        #834:"n",  # NU + TILDE
        958:"x",  # XI
        959:"o",  # OMICRON
        8056:"o",  # OMICRON + GRAVE
        8057:"o",  # OMICRON + AIGU
        8000:"o",  # OMICRON + TICK
        8001:"o",  # OMICRON + BACKTICK
        8004:"o",  # OMICRON + TICK + AIGU
        8005:"o",  # OMICRON + BACKTICK + AIGU
        8002:"o",  # OMICRON + TICK + GRAVE
        8003:"o",  # OMICRON + BACKTICK + GRAVE
        960:"p",  # PI
        961:"r",  # RHO
        962:"v",  # SIGMA
        963:"s",  # SIGMA
        964:"t",  # TAU
        965:"y",  # UPSILON
        8022:"y",  # UPSILON + TILDE + TICK
        8023:"y",  # UPSILON + TILDE + BACKTICK
        8166:"y",  # UPSILON + TILDE
        8016:"y",  # UPSILON + TICK
        8017:"y",  # UPSILON + BACKTICK
        8020:"y",  # UPSILON + TICK + AIGU
        8021:"y",  # UPSILON + BACKTICK + AIGU
        8018:"y",  # UPSILON + TICK + GRAVE
        8019:"y",  # UPSILON + BACKTICK + GRAVE
        8058:"y",  # UPSILON + GRAVE
        8059:"y",  # UPSILON + AIGU ?
        966:"f",  # PHI
        967:"c",  # CHI
        968:"u",  # PSI
        969:"w",  # OMEGA
        8102:"w",  # OMEGA + TILDE + TICK + CEDILLE
        8103:"w",  # OMEGA + TILDE + BACKTICK + CEDILLE
        8038:"w",  # OMEGA + TILDE + TICK
        8039:"w",  # OMEGA + TILDE + BACKTICK
        8182:"w",  # OMEGA + TILDE
        8183:"w",  # OMEGA + TILDE + CEDILLE
        8060:"w",  # OMEGA + GRAVE
        8061:"w",  # OMEGA + AIGU
        8032:"w",  # OMEGA + TICK
        8033:"w",  # OMEGA + BACKTICK
        8036:"w",  # OMEGA + TICK + AIGU
        8037:"w",  # OMEGA + BACKTICK + AIGU
        8034:"w",  # OMEGA + TICK + GRAVE
        8035:"w",  # OMEGA + BACKTICK + GRAVE
    
        8217:"'",  # TICK???
        8025:"`",  # BACKTICK???
        803:"_",  # CEDILLE???
        8195:" ",  # WEIRD SPACE???
        903:".",  # WEIRD DOT???
        8228:".",  # WEIRD DOT???
        8311:"7",  # SUPERSCRIPT 7
    
        # Formatting characters
        10:"\n",  # CARRIAGE RETURN
        3:" ",  # SPACE
        32:" "  # SPACE
    })
            
    txt = []
    for char in stringin:
        if ord(char) in eqv:
            txt.append(eqv[ord(char)])
        else:
            txt.append(char)               
    txt = "".join(txt)
    return txt.upper()


def convert(stringin,use_old=False):
     if use_old:
         return old_convert(stringin)
     else:
         return new_convert(stringin)

def readBoxFile(fname,all_conf=False, get_conf=False):
    """
    Parameters
    ----------
    fname : Name of the file to read

    Returns
    -------
    angle : Primary angle of the text
    boxes : List of boxes
    transcript : Box annotations
    lines : Information about how boxes are arranged in lines
    """
    angle = None
    boxes = []
    transcript = []
    lines = []
    confidence = ''
    try:
        with open(fname, "r") as f:
            phase = 0
            count = 0
            for line in f:
                count = count+1
                line2 = line.replace('*','').replace(' ','')
                if line2=='' or line2=='\n':
                    continue  # skip empty line
                elif line[0]=='#':
                    phase = phase+1
                elif phase==0:  # File should begin with a comment
                    #if line[0]=='# Baseline angle\n':
                    print("Header problem in file {}.".format(fname))
                elif phase==1:
                    #if line=='# Boxes: x1 y1 x2 y2 x3 y3 x4 y4\n':
                    if angle==None:
                        angle = float(line)
                    else:
                        print("Multiple angle data found in {}.".format(fname))
                elif phase==2:
                    #if line=='# Transcript\n':
                    n = line.split()
                    if len(n)==8:
                        box = Box(*[float(c) for c in n])
                        if box.area() < 9:
                            print("Skipping box with very small area (line {}).".format(count))
                        else:
                            boxes.append(box)
                    else:
                        print('Incorrect parameter count for box line ({}) in {}.'.format(len(n),fname))   
                elif phase==3:
                    transcript.append(line.replace('*','').replace('\n','').replace('\r','').replace(' ',''))
                elif phase==4:
                    n = line.split()
                    if len(n)==4:
                        lines.append([float(c) for c in n])
                        if np.any(np.isnan(lines[-1])):
                            print('Bad values found in line specification (file line {}) -- skipping.'.format(count))
                            lines.pop()
                        elif lines[-1][0]==lines[-1][2] and lines[-1][1]==lines[-1][3]:
                            print('Zero-length line found in line specification (file line {}) -- skipping.'.format(count))
                            lines.pop()
                    else:
                        print('Incorrect parameter count for line specifiers ({}) in {}.'.format(len(n),fname))   
                elif phase==5:
                    confidence += line.replace('\n','').replace('\r','').replace(' ','')
    except FileNotFoundError:
        print(f"File '{fname}' not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    if get_conf:
        if len(confidence)>0:
            assert(len(boxes)==len(confidence))
            return [c=='1' for c in confidence]
        else:
            return [True for b in boxes]
    if len(confidence)>0 and not all_conf:
        assert(len(boxes)==len(confidence))
        boxes = [b for (b,c) in zip(boxes,list(confidence)) if c=='1']
        it = iter(confidence)
        transcript = [''.join([ch for (ch,conf) in zip(line,list(islice(it,len(line)))) if conf=='1']) for line in transcript]
    return angle,boxes,transcript,lines


def getConfidence(fname):
    '''
    Returns the confidence vector for a letter box file.
    Reads both the box files for a given squeeze and computes the
    union of the two confidences.
    '''
    def _noneg(a):
        return 1e99 if a<0 else a 
    fdir, fbase = split(fname)
    basepos = min(fbase.find('_Rotation'),_noneg(fbase.find('_Merged')),_noneg(fbase.find('_Copy')))
    files = listdir(fdir)
    clist = []
    for f in files:
        if f.startswith(fbase[:basepos]):
            clist.append(readBoxFile(join(fdir,f), all_conf=True, get_conf=True))
    assert(len(clist)==2)
    conf = clist[0]
    for c in clist[1:]:
        assert(len(conf)==len(c))
        conf = [a or b for a,b in zip(conf,c)]
    return conf


def dist2seg(x,y,seg):
    '''
    # compute the distance from a point to the nearest point on a line segment
    '''
    x0 = seg[0]
    y0 = seg[1]
    x1 = seg[2]
    y1 = seg[3]
    rx = x-x0
    ry = y-y0
    rx1 = x1-x0
    ry1 = y1-y0
    seglen = math.sqrt(rx1**2+ry1**2)
    parcomp = (rx*rx1+ry*ry1)/seglen
    if parcomp<0:
        # closest point is p0 endpoint
        d = math.sqrt(rx**2+ry**2)
    elif parcomp > seglen:
        # closest point is p1 endpoint
        d = math.sqrt((x-x1)**2+(y-y1)**2)
    else:
        # find closest point p along segment
        t = parcomp/seglen
        px = x0+t*rx1
        py = y0+t*ry1
        d = math.sqrt((x-px)**2+(y-py)**2)
    return d


def getRowBoxes(boxes,gangle=None,lines=[]):
    """
    Orders a set of boxes into rows.
    Designed to work with data from a letter box file.
    """
    if len(lines)==0:
        # infer lines by spatial arrangement
        nboxes = len(boxes)
        bclust = np.full((nboxes),-1)
        bseq = np.zeros((nboxes))
        lnbr = np.full((nboxes),-1)
        rnbr = np.full((nboxes),-1)
        
        # speed up inner loop by sorting x coordinates
        xcent = [b.center()[0] for b in boxes]
        xsort = np.sort(xcent)
        xord = np.argsort(xcent)
        xlut = np.argsort(xord)
    
        # find left and right nearest neighbors for each box
        for i in range(nboxes):
            ci = boxes[i].center()
            if (gangle==None):
                # no global angle so compute a local one from the top line of this box
                angle = boxes[i].angle()
            else:
                angle = gangle
            ax = math.cos(angle)
            ay = math.sin(angle)
            rd = np.inf  # current distance to closest right leighbor
            ld = np.inf  # current distance to closest left neighbor
            # we are going to start at the x value of the current box, and check nearby
            # values of x both higher and lower
            klo = xlut[i]
            khi = klo
            assert(ci[0] == xsort[klo])
            while min(ci[0]-xsort[klo],xsort[khi]-ci[0]) < max(rd,ld) and (klo>0 or khi < nboxes-1):
                if (ci[0]-xsort[klo] < xsort[khi]-ci[0] or khi >= nboxes-1) and klo > 0:
                    # reduce klo
                    klo = klo-1
                    j = xord[klo]  # next closest x value below
                elif khi < nboxes-1:
                    # increase khi
                    khi = khi+1
                    j = xord[khi]  # next closest x value above
                else:
                    print('Should not reach this!')
                # check whether box j is closer to i on either the left or the right
                d = boxes[i].dist(boxes[j])
                cj = boxes[j].center()
                rj = boxes[j].radius();
                rjsq = rj*rj
                dcx = cj[0]-ci[0]
                dcy = cj[1]-ci[1]
                dotprod = dcx*ax+dcy*ay
                isRight = dotprod > 0
                if (i!=j) and (d<ld) and not isRight:
                    # found a closer left neighbor, so check position and update if near line
                    c = (ci[0]-d*ax, ci[1]-d*ay)
                    if (cj[0]-c[0])**2+(cj[1]-c[1])**2 < rjsq:
                        ld = d
                        lnbr[i] = j
                if (i!=j) and (d<rd) and isRight:
                    # found a closer right neighbor, so check position and update if near line
                    c = (ci[0]+d*ax, ci[1]+d*ay)
                    if (cj[0]-c[0])**2+(cj[1]-c[1])**2 < rjsq:
                        rd = d
                        rnbr[i] = j
                #print(i,j)
        
        # sanity/correctness check for neighbor detection -- hopefully should never report a mismatch.
        for i in range(len(rnbr)):
            if rnbr[i]>=0 and lnbr[rnbr[i]]!=i:
                if lnbr[rnbr[i]] >= 0 and rnbr[lnbr[rnbr[i]]] == rnbr[i]:
                    # pointing to a neighbor that is mutually satisfied; this link is probably wrong
                    rnbr[i] = -2
            if lnbr[i]>=0 and rnbr[lnbr[i]]!=i:
                if rnbr[lnbr[i]] >= 0 and lnbr[rnbr[lnbr[i]]] == lnbr[i]:
                    # pointing to a neighbor that is mutually satisfied; this link is probably wrong
                    lnbr[i] = -2
        for i in range(len(rnbr)):
            if rnbr[i]>=0 and lnbr[rnbr[i]]!=i:
                if lnbr[rnbr[i]]<0:
                    # link up unassigned box
                    lnbr[rnbr[i]] = i
                else:
                    print('Left of right mismatch at {}: {} to {}.'.format(i,rnbr[i],lnbr[rnbr[i]]))
                    #lnbr[rnbr[i]] = -3
                    rnbr[i]=-3
            if lnbr[i]>=0 and rnbr[lnbr[i]]!=i:
                if rnbr[lnbr[i]]<0:
                    # link up unassigned box
                    rnbr[lnbr[i]] = i
                else:
                    print('Right of left mismatch at {}: {} to {}.'.format(i,lnbr[i],rnbr[lnbr[i]]))
                    #rnbr[lnbr[i]] = -3
                    lnbr[i] = -3
    
        # cluster boxes by row: combine nearest neighbors one by one
        clust = 0
        cid = []
        cseq = []
        for i in range(nboxes):
            if (gangle==None):
                angle = boxes[i].angle()
            else:
                angle = gangle
            if (bclust[i] == -1):
                # we have found a box belonging to a new cluster, so assemble all of its neighbors and give them the same cluster label
                c = [i]
                bclust[i] = clust
                bseq[i] = boxes[i].center()[0]*math.cos(angle)+boxes[i].center()[1]*math.sin(angle)
                j = lnbr[i]
                while (j>=0) and not j in c:
                    bclust[j] = clust
                    c.append(j)
                    bseq[j] = boxes[j].center()[0]*math.cos(angle)+boxes[j].center()[1]*math.sin(angle)
                    j = lnbr[j]
                j = rnbr[i]
                while (j>=0) and not j in c:
                    bclust[j] = clust
                    c.append(j)
                    bseq[j] = boxes[j].center()[0]*math.cos(angle)+boxes[j].center()[1]*math.sin(angle)
                    j = rnbr[j]
                cseq.append(-boxes[i].center()[0]*math.sin(angle)+boxes[i].center()[1]*math.cos(angle))
                cid.append(c)
                clust = clust+1
            # the new cluster is now labeled
            #print(clust,c,cseq,bseq)
        
        rowlist = []
        # at the end of this loop, all the boxes are labeled with a unique cluster id.
        # they may not be sorted left to right or top to bottom yet
        # cseq holds a value that increases from top to bottom, 
        # and bseq holds a value that increases from left to right within a row
        # so the boxes should be reordered and assembled according to these numbers.
        # use ord2lut to convert the numeric values to a list of indices that are in the correct order
        for c in cid:
            # c is a list of the box ids belonging to this cluster
            rowlist.append([boxes[c[i]] for i in np.argsort(bseq[c])])
        return [rowlist[i] for i in np.argsort(cseq)], [cid[i] for i in np.argsort(cseq)]  # reorder rows based on cseq value
    else:
        # attribute boxes to lines based on supplied line information
        # first, identify any polylines:
        label = np.zeros(len(lines),dtype=np.int32)
        for i in range(1,len(lines)):
            if lines[i][0]==lines[i-1][2] and lines[i][1]==lines[i-1][3]:
                # we have a continuation
                label[i] = label[i-1]
            else:
                # new polyline
                label[i] = label[i-1]+1

        # next, find min distance from each box to a line segment
        blabel = np.zeros(len(boxes))
        for i in range(len(boxes)):
            d = dist2seg(*boxes[i].center(),lines[0])
            blabel[i] = 0
            for j in range(1,len(lines)):
                jd = dist2seg(*boxes[i].center(),lines[j])
                if jd<d:
                    d = jd
                    blabel[i] = label[j]
                    
        # now arrange the boxes within the rows
        rowlist = []
        idlist = []
        for i in range(label[-1]+1):
            j = 0
            while j < len(boxes) and blabel[j]!=i:  # keep going until we find the first box with label i
                j = j+1
            if j<len(boxes):  # just make sure we found at least one box assigned to this label
                row = [boxes[j]]
                rowid = [j]
                rank = [0]
                for k in range(j+1,len(boxes)):
                    if blabel[k]==i:
                        row.append(boxes[k])
                        rowid.append(k)
                        rank.append((boxes[k].xnw-boxes[j].xnw)*(boxes[j].xne-boxes[j].xnw)+(boxes[k].ynw-boxes[j].ynw)*(boxes[j].yne-boxes[j].ynw))
                # now we have all the boxes in this row, and rank holds a value that will sort them
                rowlist.append([row[i] for i in np.argsort(rank)])
                idlist.append([int(rowid[i]) for i in np.argsort(rank)])
        return rowlist, idlist
    
    
def orderBoxes(boxes,gangle=None,lines=[]):
    rowlist, idlist = getRowBoxes(boxes,gangle=gangle,lines=lines)
    return [b for row in rowlist for b in row]


def getRowTranscript(fname, all_conf=False):
      """
      Reads a file of annotated letter boxes and arranges them into rows 
      ordered from left to right.
     
      For contest grading purposes, we want to use simple left-to-right,
      top down reading order, explicitly ignoring multicolumn layouts.
      
      To make sure the transcript matches, we need to first generate the 
      multicolumn result and then reorder into full-width lines.
      """   
      gangle, boxes, transcript, lines = readBoxFile(fname, all_conf=True)
      boxes2 = orderBoxes(boxes,gangle,lines)  # reordered to transcript order
      if len(lines)==0:
          # no multicolumn; use transcript rows
          if not all_conf:
              conf = getConfidence(fname)
              it = iter(conf)
              transcript = [''.join([line[i] for i,cf in enumerate(islice(it,len(line))) if cf]) for line in transcript]
              transcript = [line for line in transcript if len(line)>0]  # eliminate empty lines
          return '\n'.join(transcript)
      else:
          # possible multicolumn; use auto-ordering
          rowlist, idlist = getRowBoxes(boxes2,gangle=gangle)  # ignore multicolumn
          if not all_conf:
              conf = getConfidence(fname)
              idlist = [[id for id in idrow if conf[id]] for idrow in idlist]
              idlist = [idr for idr in idlist if len(idr)>0]  # remove empty rows
          lintran = ''.join(transcript)
          return '\n'.join([''.join([lintran[i] for i in rowid]) for rowid in idlist])
      

def masked_cer(
    reference: str,
    confidence: list[bool],
    hypothesis: str,
) -> tuple[float, int, int]:
    """
    Compute a masked Character Error Rate (CER) where:
      - Errors on clear characters are always penalised.
      - Correct guesses on obscured characters are rewarded (grow the denominator).
      - Incorrect guesses on obscured characters are penalised.
      - Skipping an obscured character is free (no penalty, no reward).
      - Spurious insertions are always penalised.

    Parameters
    ----------
    reference   : ground-truth string
    confidence  : per-character boolean flags, same length as reference
                  (True = clearly visible, False = damaged / obscured)
    hypothesis  : reader's transcription (plain string)

    Returns
    -------
    (cer, numerator, denominator)
        cer         : numerator / denominator, or 0.0 when denominator is 0
        numerator   : total edit cost (penalised errors)
        denominator : number of clear characters
                      + number of correctly detected obscured characters

    Cost model
    ----------
    Operation                               Cost
    ────────────────────────────────────────────────
    Match on a clear character               0
    Substitution on a clear character        1
    Deletion of a clear character            1
    Match on an obscured character           0  ← rewarded (grows denominator)
    Substitution on an obscured character    1  ← penalised like any other error
    Deletion of an obscured character        0  ← free to abstain
    Insertion in the hypothesis              1  ← always penalised
    """
    if len(reference) != len(confidence):
        raise ValueError("reference and confidence must have the same length")

    n, m = len(reference), len(hypothesis)

    # ── Forward DP ──────────────────────────────────────────────────────────
    # dp[i][j] = minimum edit cost to align reference[:i] with hypothesis[:j]
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    # Cost of deleting a reference prefix against an empty hypothesis
    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + (1 if confidence[i - 1] else 0)

    # Cost of inserting a hypothesis prefix against an empty reference
    for j in range(1, m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            is_clear = confidence[i - 1]
            match    = reference[i - 1] == hypothesis[j - 1]

            sub_cost = 0 if match    else 1  # same rule for clear and obscured
            del_cost = 1 if is_clear else 0  # skipping an obscured char is free

            dp[i][j] = min(
                dp[i - 1][j - 1] + sub_cost,  # match / substitute
                dp[i - 1][j]     + del_cost,  # delete reference char
                dp[i][j - 1]     + 1,         # insert hypothesis char
            )

    # ── Traceback ────────────────────────────────────────────────────────────
    # Reconstruct one optimal alignment, counting correctly detected obscured
    # characters (these expand the denominator). Diagonal steps (match/sub)
    # are preferred over deletions/insertions when costs are tied, so that
    # correct obscured detections are recognised wherever the DP allows.
    correctly_detected_obscured = 0
    i, j = n, m

    while i > 0 or j > 0:
        if i > 0 and j > 0:
            is_clear = confidence[i - 1]
            match    = reference[i - 1] == hypothesis[j - 1]
            sub_cost = 0 if match    else 1
            del_cost = 1 if is_clear else 0

            if dp[i][j] == dp[i - 1][j - 1] + sub_cost:
                # Diagonal: match or substitution
                if not is_clear and match:
                    correctly_detected_obscured += 1
                i -= 1; j -= 1
            elif dp[i][j] == dp[i - 1][j] + del_cost:
                # Up: delete reference character
                i -= 1
            else:
                # Left: insert hypothesis character
                j -= 1
        elif i > 0:
            i -= 1  # delete remaining reference characters
        else:
            j -= 1  # consume remaining hypothesis characters as insertions

    # ── Final score ──────────────────────────────────────────────────────────
    num_clear   = sum(confidence)
    numerator   = dp[n][m]
    denominator = num_clear + correctly_detected_obscured
    cer         = 0.0 if denominator == 0 else numerator / denominator

    return cer, numerator, denominator

def evaluate(prediction, allgt, conf, old_convert = False):
    """
    Evaluates the given prediction against the 
    corresponding ground truth file.
    Includes line breaks as characters.

    Parameters
    ----------
    prediction : String
        Text transcription to be graded
    allgt : String
        Accepted ground truth transcript (all characters)
    conf : List of booleans
        True indicates mandatory detection

    Returns
    -------
    cer, all_cer, opt_cer, n_err, n_all_err, n_opt_err, n_char, n_all_char, n_opt_char

    """
    if old_convert:
        prediction = ''.join(latin_convert(prediction))
    cpred_rows = convert(prediction,use_old=old_convert).upper().replace(' ','').replace('\t','').split('\n')
    allgt_rows = convert(allgt,use_old=old_convert).upper().replace(' ','').replace('\t','').split('\n')
    it = iter(conf)
    conf_rows = [list(islice(it,len(line))) for line in allgt_rows]
    gt_rows = [''.join([gtc for (gtc,cf) in zip(gtr,confr) if cf]) for gtr,confr in zip(allgt_rows,conf_rows)]
    gt_rows = [gtr for gtr in gt_rows if gtr != '']
    
    n_err = levenshtein('\n'.join(cpred_rows),'\n'.join(gt_rows))  # string with CR, not by rows: sum([levenshtein(predr,gtr) for predr,gtr in zip(cpred_rows,gt_rows)])
    n_char = len('\n'.join(gt_rows)) # was: sum([len(r) for r in gt_rows])
    n_all_err = levenshtein('\n'.join(cpred_rows),'\n'.join(allgt_rows))  # string with CR, not by rows: sum([levenshtein(predr,agtr) for predr,agtr in zip(cpred_rows,allgt_rows)])
    n_all_char = len('\n'.join(allgt_rows)) # was: sum([len(r) for r in gt_rows])
    #conf_wbrk = [cf for cfr in [cfr+[True] for cfr in conf_rows] for cf in cfr]
    #conf_wbrk = conf_wbrk[:-1]    
    conf_wbrk = []
    flip = False
    for i, cfr in enumerate(conf_rows):
        conf_wbrk.extend(cfr)
        if ( i < len(conf_rows) - 1) and not flip:
            flip = any(cfr)
            conf_wbrk.append(flip and any(conf_rows[i + 1]))
        elif i < len(conf_rows) - 1:
            conf_wbrk.append(any(conf_rows[i + 1]))           
    assert(len(conf_wbrk)==len('\n'.join(allgt_rows)))
    opt_cer, n_opt_err, n_opt_char = masked_cer('\n'.join(allgt_rows), conf_wbrk, '\n'.join(cpred_rows)) # string with CR, not by rows: sum([masked_cer(agtr,confr,predr) for predr,agtr,confr in zip(cpred_rows,allgt_rows,conf_rows)])

    #n_err_old = levenshtein(convert(prediction).upper().replace(' ','').replace('\t',''),'\n'.join(gt_rows))
    #assert(n_err_old == n_err)
    #n_all_err_old = levenshtein(convert(prediction).upper().replace(' ','').replace('\t',''),convert(allgt).upper().replace(' ','').replace('\t',''))
    #assert(n_all_err_old == n_all_err)

    cer = n_err/n_char if n_char > 0 else 0.0
    all_cer = n_all_err/n_all_char if n_all_char > 0 else 0.0
    return cer, all_cer, opt_cer, n_err, n_all_err, n_opt_err, n_char, n_all_char, n_opt_char


def clip_squeeze(name):
    i0 = len(name)
    i1 = name.find('_Rotation')
    i2 = name.find('_Merged')
    return name[:min(i1 if i1>0 else i0, i2 if i2>0 else i0)]


def run_evaluations(pred_dir,gt_dir,verbose=False, old_convert=False):
    """
    Runs the evaluation for all files in the ground truth directory.
    For a given ground truth file (named SQUEEZE_letters.txt), 
    it looks for a prediction file named SQUEEZE_pred.txt
    containing the transcribed letters.

    Parameters
    ----------
    pred_dir : String, optional
        Directory containing prediction files. The default is '.'.
    gt_dir : TYPE, optional
        Directory containing ground truth files. The default is '.'.

    Returns
    -------
    Evaluation statistics & list of files

    """
    gt_files = [f for f in listdir(gt_dir) if isfile(join(gt_dir, f))] 
    #gt_files = [f for f in listdir(gt_dir) if isfile(join(gt_dir, f)) and f[-29:]=='_Rotation1_300dpi_letters.txt']  # transcript is same for both rotations
    cer = []
    all_cer = []
    opt_cer = []
    n_err = []
    n_all_err = []
    n_opt_err = []
    n_char = []
    n_all_char = []
    n_opt_char = []
    for gtf in gt_files:
        # first try reading a file for each scanned image:
        predf = os.path.join(pred_dir,gtf[:-12]+'_transcript.txt')
        try:
            with open(predf, "r", encoding='utf8') as f:
                pred = "".join([line for line in f])
        except FileNotFoundError:
            # fall back to single file per squeeze:
            predf2 = os.path.join(pred_dir,clip_squeeze(gtf)+'_transcript.txt')
            try:
                with open(predf2, "r", encoding='utf8') as f:
                    pred = "".join([line for line in f])
            except FileNotFoundError:
                print(f"Unable to find prediction file '{predf}'.")
                pred = ""
        except Exception as e:
            print(f"An error occurred: {str(e)} while reading '{predf}'.")
            pred = ""
        fname = os.path.join(gt_dir,gtf)
        allgt = getRowTranscript(fname,all_conf=True)
        conf = getConfidence(fname)
        sq_cer, sq_all_cer, sq_opt_cer, sq_n_err, sq_n_all_err, sq_n_opt_err, sq_n_char, sq_n_all_char, sq_n_opt_char = evaluate(pred, allgt, conf, old_convert=old_convert)
        cer.append(sq_cer)
        all_cer.append(sq_all_cer)
        opt_cer.append(sq_opt_cer)
        n_err.append(sq_n_err)
        n_all_err.append(sq_n_all_err)
        n_opt_err.append(sq_n_opt_err)
        n_char.append(sq_n_char)
        n_all_char.append(sq_n_all_char)
        n_opt_char.append(sq_n_opt_char)
        if verbose:
            print(f"Result for {gtf}: {sq_n_err} errors in {sq_n_char}, CER = {sq_cer}.")
            print(f"\tAll:\t{sq_n_all_err} errors in {sq_n_all_char}, CER = {sq_all_cer}.")
            print(f"\tOpt:\t{sq_n_opt_err} errors in {sq_n_opt_char}, CER = {sq_opt_cer}.")
    return sum(n_err)/sum(n_char), sum(n_all_err)/sum(n_all_char), sum(n_opt_err)/sum(n_opt_char), n_err, n_all_err, n_opt_err, n_char, n_all_char, n_opt_char, gt_files


if __name__ == "__main__":
    cer, all_cer, opt_cer, n_err, n_all_err, n_opt_err, n_char, n_all_char, n_opt_char, gt_files = run_evaluations(pred_dir='pred',gt_dir='gt')
    print(f'Mean character error rate is {cer}.')
    print(f'Mean character error rate for all characters is {all_cer}.')
    print(f'Mean character error rate for opt-in characters is {opt_cer}.')
