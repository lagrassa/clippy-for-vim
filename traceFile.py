from planGlobals import debug, debugMsg, pause
import windowManager3D as wm
from miscUtil import timeString
import local
import planGlobals as glob
import os

# Tracing interface, one level up from debug...
# targetFile
# level
# - writes to log at every level
# - draws at every level
# - levels 0 and 1 always printed to console
# - determines whether to pause

'''
def trace(genTag, *msg):
    if debug('traceGen'):
        print genTag+':',
        for m in msg: print m,
        print ' '

def tracep(pause, *msg):
    if debug('traceGen'):
        print pause+':',
        for m in msg: print m,
        print ' '
    if pause:
        debugMsg(pause)
'''

# Decides whether to print to console.
# - traceGen is debugged *and*
# - (level is <= 1 or (we are not in the heuristic or debugging in heuristic)) *and*
# - (level is <= minTraceLevel or we are debugging on this tag)

# LPK simplified this

# Lower traceLevel is more urgent

minTraceLevel = 1
def traced(genTag, level):
    return level <= minTraceLevel and (debug(genTag) or genTag == '*')
    # if not debug('traceGen') \
    #    or (level > 1 and (glob.inHeuristic and not debug('debugInHeuristic'))) \
    #    or (level > minTraceLevel and not debug(genTag)):
    #     return False
    # return True

# Always print and log this
def trAlways(*msg, **keys):
    tr('*', 0, *msg, **keys)

# keys is a dictionary
# Possible keywords:  draw, snap, pause
#    draw: draws into a window
#    snap: puts image of listed windows into the log
#    pause: pauses
#    ol: write onto single line, if true

# LPK: make ol work for log as well as terminal

def tr(genTag, level, *msg, **keys):
    if glob.inHeuristic and htmlFileH:
        targetFile = htmlFileH
    elif (not glob.inHeuristic) and htmlFile:
        targetFile = htmlFile
    else:
        targetFile = None
    if ((not glob.inHeuristic) or debug('debugInHeuristic')) and \
        not keys.get('noLog', False):
        if msg and targetFile:
            targetFile.write('<pre>'+level*'  '+' '+'(%d)%s'%(glob.planNum, genTag)+\
                             ': '+str(msg[0])+'</pre>\n')
            for m in msg[1:]:
                targetFile.write('<pre>'+level*'  '+str(m)+'</pre>\n')
        draw = keys.get('draw', [])
        for obj in draw:
            if not obj[0]: continue
            if isinstance(obj, (list, tuple)):
                obj[0].draw(*obj[1:])
            else:
                obj.draw('W')
        if keys.get('snap', []):
            snap(*keys['snap'])
    # Printing to the console
    if not traced(genTag, level): return
    if msg:
        prTag = (genTag+':') if (genTag != '*') else ''
        if keys.get('ol', True):
            # Print on one line
            print level*'  ', prTag,
            for m in msg: print m,
            print '\n',
        else:
            print level*'  ', prTag, msg[0]
            for m in msg[1:]:
                print level*'  ', m
    #debugMsg(genTag)
    #if level >= 0:
    #    debugMsg(genTag+str(level))         # more specific pause tag
    if keys.get('pause', False) or pause(genTag):
        windows = keys.get('snap', [])
        for w in windows:
            wm.getWindow(w).update()
        if glob.PDB:
            raw_input(genTag+' go?')

pngFileId = 0
htmlFile = None
htmlFileH = None
htmlFileId = 0
dirName = None

def traceStart():
    global htmlFile, htmlFileH, htmlFileId, dirName
    dirName = local.genDir + 'log_'+timeString()
    if not os.path.exists(dirName):
        os.makedirs(dirName)
    htmlFile = open(local.htmlGen%(dirName, str(htmlFileId)), 'w')
    htmlFileH = open(local.htmlGenH%(dirName, str(htmlFileId)), 'w')
    htmlFileId += 1
    # Could use this for css styles
    header = '''
<!DOCTYPE html>
<html>
<body>
'''
    htmlFile.write(header)
    htmlFileH.write(header)

def traceEnd():
    global htmlFile, htmlFileH
    footer = '''
</body>'''
    if htmlFile: 
        htmlFile.write(footer)
        htmlFile.close()
        htmlFile = None
    if htmlFileH:
        htmlFileH.write(footer)
        htmlFileH.close()
        htmlFileH = None

def snap(*windows):
    global pngFileId
    for win in windows:
        pngFileName = local.pngGen%(dirName, str(pngFileId))
        if glob.inHeuristic and htmlFileH:
            targetFile = htmlFileH
        elif (not glob.inHeuristic) and htmlFile:
            targetFile = htmlFile
        else:
            return
        if wm.getWindow(win).window.modified:
            wm.getWindow(win).window.saveImage(pngFileName)
            pngFileId += 1
            targetFile.write('<br><img src="%s" border=1 alt="%s"><br>\n'%(pngFileName,win))
        else:
            targetFile.write('<br><b>No change to window %s</b><br>\n'%win)

def trLog(string):
    print string
    if htmlFile:
        htmlFile.write('<pre>%s</pre>\n'%string)
