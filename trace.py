import planGlobals
from planGlobals import debug, debugMsg
import windowManager3D as wm
from miscUtil import timeString, prettyString
import local
import planGlobals as glob

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
       or (level > 1 and (glob.inHeuristic and not debug('inHeuristic'))) \
       or (level > minTraceLevel and not debug(genTag)):
        return False
    return True

def tr(genTag, level, *msg, **keys):
    if glob.inHeuristic and htmlFileH:
        targetFile = htmlFileH
    elif (not glob.inHeuristic) and htmlFile:
        targetFile = htmlFile
    else:
        targetFile = None
    if msg and targetFile:
        targetFile.write('<pre>'+level*'  '+' '+genTag+': '+str(msg[0])+'</pre>\n')
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

pngFileId = 0
htmlFile = None
htmlFileH = None
htmlFileId = 0
def traceStart():
    global htmlFile, htmlFileH, htmlFileId
    htmlFile = open(local.htmlGen%(str(htmlFileId), timeString()), 'w')
    htmlFileH = open(local.htmlGenH%(str(htmlFileId), timeString()), 'w')
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
        pngFileName = local.pngGen%(str(pngFileId), timeString())
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
