import fbch
import planGlobals
from planGlobals import debug, debugMsg
import windowManager3D as wm
from miscUtil import timeString, prettyString
import local

# Tracing interface, one level up from debug...

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

minTraceLevel = 1
def traced(genTag, level):
    if not debug('traceGen') \
       or (level > 1 and (fbch.inHeuristic and not debug('inHeuristic'))) \
       or (level > minTraceLevel and not debug(genTag)):
        return False
    return True

def tr(genTag, level, *msg, **keys):
    if msg and htmlFile:
        htmlFile.write('<pre>'+level*'  '+' '+genTag+': '+str(msg[0])+'</pre>\n')
        for m in msg[1:]:
            htmlFile.write('<pre>'+level*'  '+str(m)+'</pre>\n')
    draw = keys.get('draw', [])
    for obj in draw:
        if isinstance(obj, (list, tuple)):
            obj[0].draw(*obj[1:])
        else:
            obj.draw('W')
    if not traced(genTag, level): return
    if msg:
        print level*'  ', genTag+':', msg[0]
        for m in msg[1:]:
            print level*'  ', m
    debugMsg(genTag)
    if level >= 0:
        debugMsg(genTag+str(level))         # more specific pause tag
    if keys.get('pause', False):
        raw_input(genTag+':pause')
    if keys.get('snap', []):
        snap(*keys['snap'])

pngFileId = 0
htmlFile = None
htmlFileId = 0
def traceStart():
    global htmlFile, htmlFileId
    htmlFile = open(local.htmlGen%(str(htmlFileId), timeString()), 'w')
    htmlFileId += 1

def traceEnd():
    global htmlFile
    htmlFile.close()
    htmlFile = None

def snap(*windows):
    global pngFileId
    for win in windows:
        pngFileName = local.pngGen%(str(pngFileId), timeString())
        if wm.getWindow(win).window.modified:
            wm.getWindow(win).window.saveImage(pngFileName)
            pngFileId += 1
            if htmlFile:
                htmlFile.write('<br><img src="%s" border=1 alt="%s"><br>\n'%(pngFileName,win))
        else:
            if htmlFile:
                htmlFile.write('<br><b>No change to window %s</b><br>\n'%win)

def trLog(string):
    if htmlFile:
        htmlFile.write('<br><pre>%s</pre><br>\n'%string)
